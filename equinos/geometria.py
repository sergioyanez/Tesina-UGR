"""Canal geométrico del clip: la versión vigente (v1) y la corregida (v2).

**Por qué existe la v2.** La auditoría (cap. 11.6 y 17.4) midió sobre los 5 744 frames que las 9
columnas del canal geométrico del MIL (v1) son casi vacías:

  - las cuatro columnas de ojo y hocico, centradas en su punto medio y divididas por su propia
    distancia, valen ±0,5 sobre el versor ojo→hocico: llevan UN solo grado de libertad (la
    inclinación de la cabeza en el plano de la imagen);
  - las tres máscaras de validez se calculaban con `confianza > 0` y no con el umbral del resto del
    pipeline (`CONF_MINIMA_KEYPOINT` = 0,50): valen 1 en el 100 % de los frames y no informan;
  - «oreja» es un único punto (el más confiable entre base y punta de ambas orejas) en ejes de
    imagen, que mezcla posición con orientación.

**Qué es la v2.** Todo se expresa en el marco de la cabeza: eje x = dirección ojo→hocico, eje y
perpendicular, escala = distancia ojo→hocico DEL FRAME (como en v1, sin estadística global, así que
no hay fuga entre folds). El ojo y el hocico definen el marco y NO se vuelven a codificar: lo que
se agrega es lo que el marco deja libre. Por frame (17 valores):

  0-3    orejas: vector base→punta de la oreja izquierda y derecha (largo y ángulo respecto del eje
         de la cabeza; la asimetría se lee entre las dos)
  4-5    boca: vector hocico(`nose`)→`lower_jaw` (apertura de mandíbula y su dirección)
  6      ancho de boca: distancia entre comisuras / escala
  7      distancia interocular / escala
  8      escorzo: escala del frame / mediana de las escalas del clip (giro de la cabeza)
  9-10   inclinación de la cabeza: seno y coseno del ángulo del eje ojo→hocico en la imagen
  11-16  máscaras de validez (oreja izq., oreja der., boca, ancho de boca, interocular, eje)

Un keypoint con confianza < `CONF_MINIMA_KEYPOINT` (0,50) es AUSENTE; una característica que
depende de un punto ausente vale 0 y su máscara 0. El umbral de las máscaras es el del pipeline.

**Limitaciones declaradas.** Es 2D monocular: depende del punto de vista (en un perfil, una oreja
queda oculta y su máscara es 0). «Izquierda/derecha» son las del esquema de SuperAnimal, no las del
observador. Nada de esto se validó con un etólogo.

`geometria="v1"` en `MultimodalEquineDataset` conserva el comportamiento anterior (el del modelo
desplegado). La v2 tiene huella propia (`huella`), así que su caché nunca se mezcla con la de v1.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

CONF_MINIMA_KEYPOINT = 0.50     # el de la etapa 02; `test_geometria_v2` lo compara contra el notebook

VERSIONES = ("v1", "v2")
DIM = {"v1": 9, "v2": 17}

NOMBRES_V2 = (
    "oreja_izq_dx", "oreja_izq_dy", "oreja_der_dx", "oreja_der_dy",
    "mandibula_dx", "mandibula_dy", "ancho_boca", "interocular", "escorzo",
    "cabeza_sen", "cabeza_cos",
    "val_oreja_izq", "val_oreja_der", "val_boca", "val_ancho_boca", "val_interocular", "val_eje")

HOCICO = "nose"
OJOS = ("left_eye", "right_eye")

# Se incrementa si cambia la definición de alguna columna: invalida las cachés viejas.
REVISION_V2 = 1


def huella(version: str = "v2", umbral: float = CONF_MINIMA_KEYPOINT) -> str:
    """Identifica una definición del canal: versión, revisión, umbral y nombres de columnas."""
    if version not in VERSIONES:
        raise ValueError(f"Versión desconocida: {version!r}. Opciones: {VERSIONES}")
    partes = [version, str(REVISION_V2), f"{umbral:.4f}", HOCICO, ",".join(OJOS)]
    if version == "v2":
        partes += list(NOMBRES_V2)
    return hashlib.sha256("|".join(partes).encode()).hexdigest()[:12]


def _puntos(filas, umbral):
    """{bodypart: (x, y)} sólo con los puntos de confianza >= umbral (el mejor si se repite)."""
    mejor: dict[str, tuple] = {}
    for f in filas:
        try:
            c = float(f["confianza"]); x = float(f["x"]); y = float(f["y"])
        except (TypeError, ValueError):
            continue
        if c != c or c < umbral or x != x or y != y:
            continue
        if f["bodypart"] not in mejor or c > mejor[f["bodypart"]][2]:
            mejor[f["bodypart"]] = (x, y, c)
    return {k: (v[0], v[1]) for k, v in mejor.items()}


def _eje(pts):
    """(centro, versor_x, versor_y, escala) del marco de la cabeza, o None si no hay eje."""
    hoc = pts.get(HOCICO)
    ojos = [pts[o] for o in OJOS if o in pts]
    if hoc is None or not ojos:
        return None
    ojo = min(ojos, key=lambda p: math.hypot(p[0] - hoc[0], p[1] - hoc[1]))  # el más cercano al hocico
    dx, dy = hoc[0] - ojo[0], hoc[1] - ojo[1]
    esc = math.hypot(dx, dy)
    if esc < 1e-6:
        return None
    ux, uy = dx / esc, dy / esc
    return ((ojo[0] + hoc[0]) / 2, (ojo[1] + hoc[1]) / 2), (ux, uy), (-uy, ux), esc


def _vec(a, b, ux, vx, esc):
    d = (b[0] - a[0], b[1] - a[1])
    return ((d[0] * ux[0] + d[1] * ux[1]) / esc, (d[0] * vx[0] + d[1] * vx[1]) / esc)


def extraer_v2(filas_por_orden, n: int, umbral: float = CONF_MINIMA_KEYPOINT):
    """Matriz (n, 17) float32 del clip. `filas_por_orden[orden]` son las filas de pose del frame."""
    import numpy as np

    marcos, puntos = [], []
    for orden in range(n):
        pts = _puntos(filas_por_orden.get(orden, []), umbral)
        puntos.append(pts)
        marcos.append(_eje(pts))
    escalas = [m[3] for m in marcos if m]
    mediana = float(np.median(escalas)) if escalas else 1.0

    salida = np.zeros((n, 17), dtype=np.float32)
    for i, (pts, m) in enumerate(zip(puntos, marcos)):
        if m is None:
            continue
        _, u, v, esc = m
        fila = salida[i]
        for k, (b, t) in enumerate((("left_earbase", "left_earend"), ("right_earbase", "right_earend"))):
            if b in pts and t in pts:
                fila[2 * k], fila[2 * k + 1] = _vec(pts[b], pts[t], u, v, esc)
                fila[11 + k] = 1.0
        if HOCICO in pts and "lower_jaw" in pts:
            fila[4], fila[5] = _vec(pts[HOCICO], pts["lower_jaw"], u, v, esc)
            fila[13] = 1.0
        if "mouth_end_left" in pts and "mouth_end_right" in pts:
            a, b = pts["mouth_end_left"], pts["mouth_end_right"]
            fila[6] = math.hypot(a[0] - b[0], a[1] - b[1]) / esc
            fila[14] = 1.0
        if all(o in pts for o in OJOS):
            a, b = pts[OJOS[0]], pts[OJOS[1]]
            fila[7] = math.hypot(a[0] - b[0], a[1] - b[1]) / esc
            fila[15] = 1.0
        fila[8] = esc / mediana
        fila[9], fila[10] = u[1], u[0]       # seno y coseno del ángulo del eje en la imagen
        fila[16] = 1.0
    return salida


def construir_cache(clips, pose_por_clip, version: str, carpeta, umbral: float = CONF_MINIMA_KEYPOINT,
                    n: int = 16) -> Path:
    """Calcula (o reutiliza) el canal geométrico de cada clip y lo guarda en `carpeta`.

    El archivo lleva la huella de la definición en el nombre: cambiar versión, umbral o columnas
    produce otro archivo y nunca se reusa uno de otra definición. `pose_por_clip[clip][orden]` son
    las filas de pose indexadas por POSICIÓN en la secuencia.
    """
    import numpy as np

    if version != "v2":
        raise ValueError("La caché de geometría sólo se define para v2 (v1 se calcula en el dataset).")
    carpeta = Path(carpeta); carpeta.mkdir(parents=True, exist_ok=True)
    destino = carpeta / f"geometria_{version}_{huella(version, umbral)}.npz"
    if destino.exists():
        return destino
    ids = [c["clip"] for c in clips]
    mat = np.stack([extraer_v2(pose_por_clip.get(c, {}), n, umbral) for c in ids])
    tmp = destino.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, clips=np.array(ids), geometria=mat, nombres=np.array(NOMBRES_V2))
    tmp.replace(destino)
    destino.with_suffix(".json").write_text(json.dumps(
        {"huella": huella(version, umbral), "version": version, "umbral": umbral,
         "revision": REVISION_V2, "columnas": list(NOMBRES_V2), "clips": len(ids)}, indent=1))
    return destino
