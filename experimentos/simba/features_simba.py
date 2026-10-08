"""Features «tipo SimBA» calculadas con código PROPIO a partir de los landmarks de DeepLabCut.

IMPORTANTE (honestidad de nombre): esto NO es SimBA. Es el mismo estilo de features que SimBA
calcula a partir de una tabla de landmarks (distancias entre puntos, ángulos, áreas/ratios,
velocidades y aceleraciones por frame, y estadísticas móviles/agregadas), implementado desde cero
porque los extractores de SimBA no son aplicables a este esquema (ver README, sección de
compatibilidad). En todos los resultados se lo llama «features tipo SimBA (código propio)».

Entrada por clip: array `(16, 39, 3)` = (x, y, confianza) de los 16 frames que ve el MIL, en píxeles
del cuadro original. Salida: vector de features por clip y sus nombres.

Normalización DENTRO de cada muestra (sin ajuste global, no hay fuga entre folds): todas las
longitudes se dividen por la escala del clip = mediana, sobre sus frames válidos, de la distancia
ojo(s)-hocico. Es una escala por clip y no por frame a propósito: dividir por frame mataría la
dinámica de la propia longitud (p. ej. el acortamiento por giro de cabeza).

Un keypoint con confianza < CONF_MIN (0,50, el `CONF_MINIMA_KEYPOINT` del pipeline) se trata como
AUSENTE (NaN). Un feature por frame es NaN si falta cualquiera de sus puntos. La agregación por
clip ignora NaN; si un clip no tiene ningún frame válido para un feature, queda NaN y se imputa
dentro del fold (mediana del train) en el clasificador. Además hay features de disponibilidad.
"""
from __future__ import annotations

import warnings

import numpy as np

from datos import BP, CONF_MIN

AGREGADOS_ESTATICOS = ("media", "desvio", "min", "max", "rango")
AGREGADOS_DINAMICOS = ("media", "desvio", "max")


def _pt(P, nombre):
    a = P[:, BP[nombre], :]
    xy = a[:, :2].copy()
    xy[a[:, 2] < CONF_MIN] = np.nan
    return xy


def _d(a, b):
    return np.linalg.norm(a - b, axis=1)


def _ang(v):
    return np.degrees(np.arctan2(v[:, 1], v[:, 0]))


def _envolver(grados):
    return (grados + 180.0) % 360.0 - 180.0


def _angulo_entre(u, v):
    """Ángulo no orientado 0..180 entre dos series de vectores."""
    nu, nv = np.linalg.norm(u, axis=1), np.linalg.norm(v, axis=1)
    c = np.einsum("ij,ij->i", u, v) / (nu * nv)
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def _agregar(serie, modos):
    v = serie[~np.isnan(serie)]
    salida = {}
    n = len(v)
    for m in modos:
        if n == 0 or (m == "desvio" and n < 2):
            salida[m] = np.nan
        elif m == "media":
            salida[m] = v.mean()
        elif m == "desvio":
            salida[m] = v.std(ddof=0)
        elif m == "min":
            salida[m] = v.min()
        elif m == "max":
            salida[m] = v.max()
        elif m == "rango":
            salida[m] = v.max() - v.min()
    return salida


def series_por_frame(P, frames, fps):
    """Features por frame (16,) cada una, y las dinámicas (15,) o (14,). Devuelve dos dicts + info."""
    eL, eR = _pt(P, "left_eye"), _pt(P, "right_eye")
    nose, uj, lj = _pt(P, "nose"), _pt(P, "upper_jaw"), _pt(P, "lower_jaw")
    mL, mR = _pt(P, "mouth_end_left"), _pt(P, "mouth_end_right")
    bL, tL = _pt(P, "left_earbase"), _pt(P, "left_earend")
    bR, tR = _pt(P, "right_earbase"), _pt(P, "right_earend")
    nb, ne = _pt(P, "neck_base"), _pt(P, "neck_end")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        E = np.nanmean(np.stack([eL, eR]), axis=0)            # centro de los ojos visibles
        eje = nose - E                                          # eje de la cabeza: ojos -> hocico
        largo_eje = np.linalg.norm(eje, axis=1)
        escala = np.nanmedian(largo_eje) if np.isfinite(largo_eje).any() else np.nan
        s = escala

        est = {}
        # orejas (longitud, ángulo respecto del eje de la cabeza, apertura entre ellas)
        lenL, lenR = _d(bL, tL) / s, _d(bR, tR) / s
        est["oreja_largo_izq"], est["oreja_largo_der"] = lenL, lenR
        est["oreja_largo_media"] = np.nanmean(np.stack([lenL, lenR]), axis=0)
        est["oreja_asimetria_largo"] = np.abs(lenL - lenR)
        angL = _angulo_entre(tL - bL, eje)
        angR = _angulo_entre(tR - bR, eje)
        est["oreja_angulo_izq"], est["oreja_angulo_der"] = angL, angR
        est["oreja_angulo_media"] = np.nanmean(np.stack([angL, angR]), axis=0)
        est["oreja_asimetria_angulo"] = np.abs(angL - angR)
        est["oreja_punta_a_ojo_izq"] = _d(tL, eL) / s
        est["oreja_punta_a_ojo_der"] = _d(tR, eR) / s
        est["oreja_base_a_ojo_media"] = np.nanmean(np.stack([_d(bL, eL), _d(bR, eR)]), axis=0) / s
        est["oreja_asimetria_punta_ojo"] = np.abs(est["oreja_punta_a_ojo_izq"] - est["oreja_punta_a_ojo_der"])
        est["oreja_dist_puntas"] = _d(tL, tR) / s
        est["oreja_dist_bases"] = _d(bL, bR) / s
        est["oreja_separacion_puntas_sobre_bases"] = est["oreja_dist_puntas"] / est["oreja_dist_bases"]
        # ojos / cabeza
        est["ojos_distancia"] = _d(eL, eR) / s
        est["cabeza_largo_ojo_hocico_rel"] = largo_eje / s       # ~1: <1 = acortamiento (giro)
        est["cabeza_inclinacion_sobre_horizontal"] = np.degrees(np.arctan2(np.abs(eje[:, 1]), np.abs(eje[:, 0])))
        # boca / mandíbula
        est["boca_apertura_mandibulas"] = _d(uj, lj) / s
        est["boca_ancho_comisuras"] = _d(mL, mR) / s
        est["boca_hocico_a_mandibula_inf"] = _d(nose, lj) / s
        est["boca_hocico_a_mandibula_sup"] = _d(nose, uj) / s
        est["boca_comisuras_a_ojo_media"] = np.nanmean(np.stack([_d(mL, eL), _d(mR, eR)]), axis=0) / s
        est["boca_asimetria_comisura_ojo"] = np.abs(_d(mL, eL) - _d(mR, eR)) / s
        # cabeza-cuello (aproximada: semántica exacta de neck_* no verificada)
        est["cuello_angulo_cabeza_cuello"] = _angulo_entre(eje, ne - nb)
        est["cuello_dist_ojos_a_neck_end"] = _d(E, ne) / s

        # ---- dinámicas: pares de frames consecutivos, dt físico en segundos ----
        dt = np.diff(frames) / fps                                # (15,)
        dt = np.where(dt > 0, dt, np.nan)
        din = {}
        C = np.nanmean(np.stack([E, nose]), axis=0)
        din["cabeza_velocidad"] = _d(C[1:], C[:-1]) / s / dt
        ang_eje = _ang(eje)
        din["cabeza_velocidad_angular"] = np.abs(_envolver(np.diff(ang_eje))) / dt
        vL, vR = tL - bL, tR - bR                                  # vector de oreja relativo a su base
        vmovL = np.linalg.norm(vL[1:] - vL[:-1], axis=1)
        vmovR = np.linalg.norm(vR[1:] - vR[:-1], axis=1)
        din["oreja_velocidad_relativa"] = np.nanmean(np.stack([vmovL, vmovR]), axis=0) / s / dt
        din["oreja_velocidad_angular"] = np.abs(np.diff(est["oreja_angulo_media"])) / dt
        din["boca_velocidad_apertura"] = np.abs(np.diff(est["boca_apertura_mandibulas"])) / dt
        din["cabeza_aceleracion"] = np.abs(np.diff(din["cabeza_velocidad"])) / ((dt[1:] + dt[:-1]) / 2)
        din["oreja_aceleracion"] = np.abs(np.diff(din["oreja_velocidad_relativa"])) / ((dt[1:] + dt[:-1]) / 2)

        # ---- disponibilidad (fracción de frames con los puntos del grupo) ----
        def frac(*pts):
            return float(np.mean(~np.any(np.isnan(np.concatenate(pts, axis=1)), axis=1)))
        disp = {
            "disp_ojos_ambos": frac(eL, eR), "disp_ojo_alguno": float(np.mean(~np.isnan(E[:, 0]))),
            "disp_oreja_izq": frac(bL, tL), "disp_oreja_der": frac(bR, tR),
            "disp_orejas_ambas": frac(bL, tL, bR, tR),
            "disp_mandibulas": frac(uj, lj), "disp_comisuras": frac(mL, mR),
            "disp_cuello": frac(nb, ne),
            "disp_hocico": frac(nose),
        }
    return est, din, disp, float(escala)


def vector_del_clip(P, frames, fps):
    """`(valores, nombres, grupos)` para un clip."""
    est, din, disp, escala = series_por_frame(P, frames, fps)
    valores, nombres, grupos = [], [], []
    for k, serie in est.items():
        r = _agregar(serie, AGREGADOS_ESTATICOS)
        for m in AGREGADOS_ESTATICOS:
            valores.append(r[m]); nombres.append(f"{k}__{m}"); grupos.append("estatica")
    for k, serie in din.items():
        r = _agregar(serie, AGREGADOS_DINAMICOS)
        for m in AGREGADOS_DINAMICOS:
            valores.append(r[m]); nombres.append(f"{k}__{m}"); grupos.append("dinamica")
    for k, v in disp.items():
        valores.append(v); nombres.append(k); grupos.append("disponibilidad")
    return np.array(valores, dtype=float), nombres, grupos, escala
