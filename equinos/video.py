"""Lectura de video compartida por la preparación del dataset y por la inferencia.

La auditoría encontró que la interfaz leía los videos de otra manera que el notebook 01 que
fabricó el conjunto de entrenamiento: el notebook abre con `CAP_PROP_ORIENTATION_AUTO = 1` y
escribe los clips recodificados con `mp4v`; la inferencia abría el video tal cual está
almacenado y se lo daba a DeepLabCut sin recodificar. 69 de 274 videos (25,2 %) declaran rotación
de 90° y, sin la bandera, el frame llega acostado.

Acá vive lo mínimo para que los dos caminos hagan lo mismo:

  - `abrir_video`: `VideoCapture` con la orientación aplicada (el notebook 01 define la misma
    función en su celda 16; si una cambia, la otra tiene que cambiar);
  - `normalizar_video`: recodifica un video con `mp4v` aplicando la orientación, que es lo que el
    paso 10 del notebook 01 le hizo a cada clip. Es la entrada que se le da a DeepLabCut y de la
    que se leen los recortes.

`FOURCC_CLIPS` es el códec con el que el notebook 01 escribió los clips. No se cambia sin
regenerar el dataset.
"""

from __future__ import annotations

from pathlib import Path

FOURCC_CLIPS = "mp4v"


def abrir_video(ruta):
    """`VideoCapture` con la orientación de los metadatos ya aplicada."""
    import cv2

    captura = cv2.VideoCapture(str(ruta))
    captura.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    return captura


def rotacion_declarada(ruta) -> float:
    """Grados de rotación que el contenedor declara (0 si no declara ninguna)."""
    import cv2

    captura = cv2.VideoCapture(str(ruta))
    try:
        return float(captura.get(cv2.CAP_PROP_ORIENTATION_META))
    finally:
        captura.release()


def normalizar_video(origen, destino, fourcc: str = FOURCC_CLIPS) -> dict:
    """Reescribe `origen` en `destino` como lo hizo la segmentación del dataset.

    Lee con la orientación aplicada y escribe con `mp4v` (por omisión) a los mismos fps. El
    escritor se crea con la forma del PRIMER frame real, igual que el notebook 01: `write` descarta
    en silencio los frames de otro tamaño. Devuelve `{frames, fps, ancho, alto, rotacion}` de lo
    escrito; `frames == 0` significa que no se pudo leer nada.
    """
    import cv2

    origen, destino = Path(origen), Path(destino)
    rotacion = rotacion_declarada(origen)
    captura = abrir_video(origen)
    fps = captura.get(cv2.CAP_PROP_FPS) or 30.0
    escritor = None
    escritos, ancho, alto = 0, 0, 0
    while True:
        ok, frame = captura.read()
        if not ok or frame is None:
            break
        if escritor is None:
            alto, ancho = frame.shape[:2]
            escritor = cv2.VideoWriter(
                str(destino), cv2.VideoWriter_fourcc(*fourcc), fps, (ancho, alto))
        escritor.write(frame)
        escritos += 1
    captura.release()
    if escritor is not None:
        escritor.release()
    return {"frames": escritos, "fps": float(fps), "ancho": ancho, "alto": alto,
            "rotacion": rotacion}
