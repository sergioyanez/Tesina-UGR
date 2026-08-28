"""Barrido de encuadres sobre los keypoints ya persistidos, sin escribir una sola imagen.

Contesta dos preguntas por cada campo de visión candidato:

    1. ¿Qué proporción de los keypoints de oreja cae DENTRO del recorte?
    2. ¿Qué proporción del recorte se completa con píxeles inventados (relleno)?

No reinfiere pose: consume `pose_completa.csv` y `keypoints_faciales.csv` de la corrida vigente.
Ése es el punto — la etapa 02 separa la inferencia (cara) de la elección (barata), y este script
vive del lado barato.

Uso:
    python openspec/changes/encuadre-que-incluye-las-orejas/medicion_encuadre.py

DOS SESGOS CONOCIDOS, que deben acompañar cualquier número que salga de acá:

  - La inclusión de orejas se mide sólo sobre frames donde la oreja se detectó con confianza
    suficiente (78,5 % de los frames con par ojo-hocico). Una oreja fuera del cuadro tiende a no
    detectarse, así que el denominador excluye justo los peores casos. Favorece a TODOS los
    encuadres por igual: sirve para comparar, no para afirmar un nivel absoluto.

  - El relleno se estima con la caja alineada a ejes del rectángulo rotado, que es una cota
    SUPERIOR: el warp sólo muestrea el rectángulo, no su caja envolvente. Sobrestima en todos los
    encuadres por igual.

La evidencia independiente de ambos sesgos es mirar los recortes. Ver tarea 1.4.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# El script vive dentro del change, tres niveles por debajo de la raíz del repo, y se corre a mano.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from equinos import rutas  # noqa: E402

CONF_MINIMA_KEYPOINT = 0.50
ANCHO_DESTINO = 224          # el alto se deriva del aspecto que pide el campo de visión
OREJAS = ["left_earbase", "left_earend", "right_earbase", "right_earend"]


def _cargar():
    """Une keypoints faciales, keypoints de oreja y dimensiones del video de origen."""
    res = rutas.carpeta("resultados")
    kp = pd.read_csv(res / "keypoints_faciales.csv")
    pose = pd.read_csv(res / "pose_completa.csv")
    inventario = pd.read_csv(res / "inventario_videos.csv")
    clips = pd.read_csv(res / "inventario_clips.csv")

    orejas = pose[pose["bodypart"].isin(OREJAS)]
    ancho = orejas.pivot_table(
        index=["clip", "orden"], columns="bodypart", values=["x", "y", "confianza"]
    )
    ancho.columns = [f"{parte}_{campo}" for campo, parte in ancho.columns]
    ancho = ancho.reset_index()

    dim = clips.merge(
        inventario[["archivo", "ancho", "alto"]],
        left_on="video_origen",
        right_on="archivo",
        how="left",
    )[["clip", "ancho", "alto"]]

    df = kp.merge(ancho, on=["clip", "orden"], how="left").merge(dim, on="clip", how="left")

    # Sólo frames con par ojo-hocico DETECTADO. Los interpolados y manuales se alinean igual, pero
    # medir el encuadre sobre posiciones estimadas mezclaría dos fuentes de error.
    return df[df["x_ojo"].notna() & df["x_hocico"].notna() & df["ancho"].notna()].copy()


def evaluar(df, aire_sobre_ojo, aire_bajo_hocico, semiancho):
    """Mide un campo de visión, expresado en unidades de la distancia ojo-hocico `d`.

    El marco de la cabeza tiene el ojo en 0 y el hocico en 1 sobre el eje axial, y el eje lateral
    perpendicular a él. El recorte abarca [-aire_sobre_ojo, 1 + aire_bajo_hocico] en axial y
    [-semiancho, +semiancho] en lateral.
    """
    p_ojo = df[["x_ojo", "y_ojo"]].to_numpy(float)
    p_hocico = df[["x_hocico", "y_hocico"]].to_numpy(float)
    ancho_frame = df["ancho"].to_numpy(float)
    alto_frame = df["alto"].to_numpy(float)

    eje = p_hocico - p_ojo
    d = np.hypot(eje[:, 0], eje[:, 1])
    u = eje / np.maximum(d, 1e-8)[:, None]          # versor axial: del ojo hacia el hocico
    n = np.stack([-u[:, 1], u[:, 0]], axis=1)       # versor lateral

    axial_min, axial_max = -aire_sobre_ojo, 1.0 + aire_bajo_hocico

    esquinas = np.stack(
        [
            p_ojo + u * (a * d)[:, None] + n * (lat * d)[:, None]
            for a in (axial_min, axial_max)
            for lat in (-semiancho, semiancho)
        ]
    )
    x1, x2 = esquinas[:, :, 0].min(axis=0), esquinas[:, :, 0].max(axis=0)
    y1, y2 = esquinas[:, :, 1].min(axis=0), esquinas[:, :, 1].max(axis=0)

    ancho_dentro = np.clip(np.minimum(x2, ancho_frame) - np.maximum(x1, 0), 0, None)
    alto_dentro = np.clip(np.minimum(y2, alto_frame) - np.maximum(y1, 0), 0, None)
    relleno = 1 - (ancho_dentro * alto_dentro) / np.maximum((x2 - x1) * (y2 - y1), 1e-8)

    medidos = dentro = 0
    for oreja in OREJAS:
        pos = np.stack(
            [df[f"{oreja}_x"].to_numpy(float) - p_ojo[:, 0],
             df[f"{oreja}_y"].to_numpy(float) - p_ojo[:, 1]],
            axis=1,
        )
        axial = (pos * u).sum(axis=1) / np.maximum(d, 1e-8)
        lateral = (pos * n).sum(axis=1) / np.maximum(d, 1e-8)
        fiable = (df[f"{oreja}_confianza"].to_numpy(float) >= CONF_MINIMA_KEYPOINT) & np.isfinite(axial)
        medidos += int(fiable.sum())
        dentro += int(
            (fiable & (axial >= axial_min) & (axial <= axial_max) & (np.abs(lateral) <= semiancho)).sum()
        )

    extension_axial = axial_max - axial_min
    return {
        "orejas_dentro": dentro / max(medidos, 1),
        "orejas_medidas": medidos,
        "relleno_mediano": float(np.median(relleno)),
        "relleno_mayor_25": float((relleno > 0.25).mean()),
        "aspecto": extension_axial / (2 * semiancho),
        "frac_ojo": aire_sobre_ojo / extension_axial,
        "frac_hocico": (aire_sobre_ojo + 1.0) / extension_axial,
        "px_por_d_origen": float(np.median(d)),
    }


CANDIDATOS = {
    # nombre: (aire sobre el ojo, aire bajo el hocico, semiancho lateral), en unidades de d.
    # El vigente se expresa en el campo de visión que produce FRAC_OJO=0,20 / FRAC_HOCICO=0,80.
    "Vigente  224x224 (0,20/0,80)": (0.20 / 0.60, 0.20 / 0.60, 0.50 / 0.60),
    "Rama hector 224x224 (0,25/0,80)": (0.25 / 0.55, 0.20 / 0.55, 0.50 / 0.55),
    "C  orejas p50, lateral ajustado": (0.80, 0.30, 0.75),
    "A  orejas p75, lateral actual": (1.00, 0.40, 0.85),
}


def main():
    rutas.inicializar()
    df = _cargar()
    print(f"Frames con par ojo-hocico detectado: {len(df)}\n")

    cab = (f'{"Encuadre":<34}{"aspecto":>8}{"destino":>10}{"d->px":>7}'
           f'{"orejas":>9}{"relleno p50":>13}{">25%":>8}')
    print(cab)
    print("-" * len(cab))
    for nombre, campo in CANDIDATOS.items():
        r = evaluar(df, *campo)
        # Destino con ancho fijo en ANCHO_DESTINO y alto derivado del aspecto requerido, para que
        # el campo de visión declarado se represente sin deformar.
        alto = int(round(ANCHO_DESTINO * r["aspecto"]))
        # Escala: cuántos píxeles del recorte ocupa la distancia ojo-hocico.
        px_por_d = ANCHO_DESTINO / (2 * campo[2])
        print(
            f"{nombre:<34}{r['aspecto']:>8.2f}{f'{ANCHO_DESTINO}x{alto}':>10}{px_por_d:>7.0f}"
            f"{r['orejas_dentro']:>9.1%}{r['relleno_mediano']:>13.1%}{r['relleno_mayor_25']:>8.1%}"
        )

    print("\nRecordar los dos sesgos documentados en el encabezado de este archivo.")


if __name__ == "__main__":
    main()
