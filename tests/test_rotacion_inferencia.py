"""La inferencia lee el video como la preparación del dataset: orientación aplicada y mp4v.

Sin GPU, sin DeepLabCut y sin datos del proyecto: el video es sintético, con rotación declarada
escrita con ffmpeg (si no hay ffmpeg, esa parte se omite y se avisa).

Uso:
    python3 tests/test_rotacion_inferencia.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

fallos = []


def check(nombre, condicion, detalle=""):
    if condicion:
        print(f"  OK    {nombre}")
    else:
        print(f"  FALLA {nombre}  {detalle}")
        fallos.append(nombre)


import inspect
import shutil
import subprocess
import tempfile

import cv2
import numpy as np

from equinos import inferencia, video

print("\nLA INFERENCIA NORMALIZA POR OMISIÓN")
check("preparar_clip normaliza por omisión",
      inspect.signature(inferencia.preparar_clip).parameters["normalizar"].default is True)
check("clasificar_video normaliza por omisión",
      inspect.signature(inferencia.clasificar_video).parameters["normalizar"].default is True)

print("\nPREPARAR_CLIP PASA POR normalizar_video ANTES DE LA POSE")
llamadas = []
orig_norm, orig_desde = video.normalizar_video, inferencia._preparar_desde
try:
    def falso_norm(origen, destino, fourcc="mp4v"):
        llamadas.append(("normalizar", str(origen)))
        Path(destino).write_bytes(b"x")
        return {"frames": 5, "fps": 30.0, "ancho": 4, "alto": 4, "rotacion": 90.0}

    def falso_desde(ruta, progreso, parametros, info, rotacion):
        llamadas.append(("desde", Path(ruta).name, rotacion))
        return "ok"
    video.normalizar_video, inferencia._preparar_desde = falso_norm, falso_desde
    r = inferencia.preparar_clip("/ruta/clip.mov", parametros={})
    check("se normaliza y luego se prepara sobre la copia",
          [l[0] for l in llamadas] == ["normalizar", "desde"] and llamadas[1][1] == "clip.mp4"
          and llamadas[1][2] == 90.0, llamadas)
    llamadas.clear()
    inferencia.preparar_clip("/ruta/clip.mov", parametros={}, normalizar=False)
    check("normalizar=False reproduce la lectura anterior (no normaliza)",
          [l[0] for l in llamadas] == ["desde"], llamadas)
finally:
    video.normalizar_video, inferencia._preparar_desde = orig_norm, orig_desde

print("\nVIDEO CON ROTACIÓN DECLARADA")
if shutil.which("ffmpeg") is None:
    print("  (sin ffmpeg: se omite)")
else:
    tmp = Path(tempfile.mkdtemp(prefix="test_rot_"))
    try:
        crudo = tmp / "crudo.mp4"
        w = cv2.VideoWriter(str(crudo), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 32))  # 64 ancho x 32 alto
        for i in range(12):
            w.write(np.full((32, 64, 3), 20 * i, np.uint8))
        w.release()
        rotado = tmp / "rotado.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(crudo), "-c", "copy",
                        "-metadata:s:v:0", "rotate=90", str(rotado)], check=True)
        declarada = video.rotacion_declarada(rotado)
        check("el contenedor declara una rotación de 90 o 270 grados", declarada in (90.0, 270.0), declarada)
        sin = cv2.VideoCapture(str(rotado)); _, f0 = sin.read(); sin.release()
        con = video.abrir_video(rotado); _, f1 = con.read(); con.release()
        check("sin la bandera el frame llega acostado (32x64)", f0.shape[:2] == (32, 64), f0.shape)
        check("abrir_video entrega el frame derecho (64x32)", f1.shape[:2] == (64, 32), f1.shape)
        norm = tmp / "norm.mp4"
        meta = video.normalizar_video(rotado, norm)
        check("normalizar_video escribe todos los frames", meta["frames"] == 12, meta)
        c = cv2.VideoCapture(str(norm)); _, f2 = c.read(); c.release()
        check("la copia normalizada es derecha y ya no declara rotación",
              f2.shape[:2] == (64, 32) and video.rotacion_declarada(norm) == 0.0, f2.shape)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

print("\n" + "=" * 52)
if fallos:
    print(f"FALLARON {len(fallos)}: {fallos}")
    sys.exit(1)
print("TODAS LAS PRUEBAS PASARON")

