"""De la pose de un clip a los 16 recortes alineados. **La usan el entrenamiento y la inferencia.**

Este módulo es la respuesta al problema que la auditoría dejó a la vista: había dos preparaciones
distintas para la misma entrada. La etapa 02 muestreaba 16 posiciones fijas, interpolaba huecos,
inyectaba anotaciones manuales y, cuando no había par ojo-hocico, recortaba por caja de YOLO y
metía ese frame igual en la secuencia. `equinos.inferencia` muestreaba las mismas 16 posiciones
fijas pero no interpolaba, no anotaba y no tenía recorte por caja: descartaba el frame.

El resultado era una interfaz **más estricta que el pipeline que fabricó su conjunto de
entrenamiento**, rechazando videos que el dataset habría aceptado, mientras informaba una
confiabilidad medida sobre otra distribución. Nada fallaba.

`preparar_secuencia` es ahora el único lugar donde se decide qué frames entran y cómo se alinean.
Las dos etapas la llaman con la misma pose y los mismos parámetros de la etapa 02, y por
construcción no pueden divergir: lo que cambie acá cambia en las dos a la vez.

**Qué NO hace, y por qué.** No interpola coordenadas y no tiene recorte por caja. Con la selección
de `equinos.secuencia` dejaron de hacer falta: los 16 frames salen de una ventana donde hay
medición real, y anclar cada posición al frame medido más cercano da mejor dato que inventar una
coordenada o que mezclar dos geometrías en la misma secuencia.

Eso deja sin uso a `MAX_HUECO_INTERPOLADO` de la etapa 02, y conviene ser explícito al respecto:
ese parámetro autorizaba a RELLENAR hasta un frame, y acá no se rellena ninguno. Su lugar lo toma
`secuencia.HUECO_MAXIMO_SEG`, que no autoriza nada sino que decide hasta dónde se considera que
la ventana sigue siendo el mismo momento. Son criterios distintos y por eso no comparten número:
uno se validó contra el error de una coordenada estimada, el otro contra la duración de un gesto.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

from equinos import encuadre, pose as sel_pose, secuencia

NOTEBOOK_ETAPA02 = "02_roi_y_normalizacion.ipynb"


class Preparacion(NamedTuple):
    """Los frames elegidos y su transformada, o el motivo por el que no hay secuencia."""

    ordenes: list[int]
    matrices: list
    motivo: str | None
    validos: int
    racha: int
    span_seg: float
    destino: dict | None

    @property
    def hay_secuencia(self) -> bool:
        return self.motivo is None


def parametros(notebook=None) -> dict:
    """Los parámetros declarados en la celda `parameters` de la etapa 02. Se leen, no se copian."""
    from equinos import inferencia

    return inferencia.parametros_etapa02(notebook or NOTEBOOK_ETAPA02)


def geometria(notebook=None) -> dict:
    """Las funciones geométricas de la etapa 02, extraídas del notebook y no reimplementadas."""
    from equinos import interpolacion

    return interpolacion.cargar_geometria(notebook)


def destino_canonico(p: dict) -> dict:
    """El recorte de destino que declara la etapa 02: dónde caen el ojo y el hocico."""
    campo = encuadre.CampoDeVision(
        p["AIRE_SOBRE_OJO"], p["AIRE_BAJO_HOCICO"], p["SEMIANCHO_LATERAL"])
    return encuadre.destino(campo, p["ANCHO_RECORTE"])


def preparar_secuencia(filas_pose, *, total, fps=30.0, p=None, geo=None, destino=None,
                       n=secuencia.FRAMES_POR_CLIP, minimo=secuencia.MINIMO_FRAMES,
                       hueco_maximo_seg=secuencia.HUECO_MAXIMO_SEG):
    """Elige los frames de la secuencia y calcula la transformada de cada uno.

    `filas_pose` son las filas de pose de UN clip, en formato largo (`equinos.pose`), idealmente
    de todos sus frames. `total` es la cantidad de frames del video y `fps` su tasa, que es lo que
    permite respetar el techo temporal de la ventana.

    Devuelve una `Preparacion`. No toca el video ni escribe nada: el que llama aplica las
    matrices sobre los frames que ya tenga. Así la misma decisión sirve para el dataset —que lee
    del disco— y para la interfaz, que tiene los frames en memoria.
    """
    p = p or parametros()
    geo = geo or geometria()
    destino = destino or destino_canonico(p)

    if not filas_pose:
        return Preparacion([], [], "SIN_KEYPOINTS", 0, 0, 0.0, destino)

    pares = sel_pose.seleccionar_par(filas_pose, bodypart_hocico=p["BODYPART_HOCICO"])
    por_orden = {int(x["orden"]): x for x in pares}
    validos = secuencia.frames_validos(pares, p["CONF_MINIMA_KEYPOINT"])
    confianzas = secuencia.confianzas_del_par(pares, p["CONF_MINIMA_KEYPOINT"])

    seleccion = secuencia.elegir_frames(
        validos, total=total, fps=fps, n=n, minimo=minimo,
        hueco_maximo_seg=hueco_maximo_seg, confianzas=confianzas)
    if not seleccion.hay_seleccion:
        return Preparacion([], [], seleccion.motivo, seleccion.validos, seleccion.racha,
                           0.0, destino)

    ordenes, matrices = [], []
    for orden in seleccion.ordenes:
        par = por_orden[orden]
        matriz = geo["similitud_desde_dos_puntos"](
            (float(par["x_ojo"]), float(par["y_ojo"])),
            (float(par["x_hocico"]), float(par["y_hocico"])),
            destino["dest_ojo"], destino["dest_hocico"])
        if matriz is None:
            continue                  # ojo y hocico en el mismo punto: la similitud no existe
        ordenes.append(orden)
        matrices.append(matriz)

    if len(ordenes) < minimo:
        # Pasa si varias transformadas salen degeneradas. Es raro, pero devolver una secuencia
        # más corta de lo declarado sería peor: el modelo la acepta y nadie se entera.
        return Preparacion([], [], "ALINEACION_INSUFICIENTE", seleccion.validos,
                           seleccion.racha, 0.0, destino)

    return Preparacion(ordenes, matrices, None, seleccion.validos, seleccion.racha,
                       seleccion.span_seg, destino)


def recortar(imagen, matriz, destino):
    """Aplica una transformada de similitud y devuelve el recorte canónico."""
    import cv2

    return cv2.warpAffine(imagen, matriz, (destino["ancho"], destino["alto"]),
                          borderMode=cv2.BORDER_REPLICATE)


def pose_por_clip(ruta_csv: str | Path) -> dict[str, list[dict]]:
    """Agrupa un CSV de pose en formato largo por clip, conservando el orden de lectura."""
    import csv

    agrupado: dict[str, list[dict]] = {}
    with Path(ruta_csv).open(encoding="utf-8", errors="replace", newline="") as archivo:
        for fila in csv.DictReader(archivo):
            agrupado.setdefault(fila["clip"], []).append(fila)
    return agrupado
