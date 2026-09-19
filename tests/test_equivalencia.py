"""La preparación del dataset y la de la interfaz tienen que producir la MISMA entrada.

Es la prueba que no existía cuando las dos divergieron. La etapa 02 y `equinos.inferencia` eran
dos implementaciones del mismo preprocesamiento, y se separaron sin que nada fallara: la interfaz
terminó siendo más estricta que el pipeline que fabricó su conjunto de entrenamiento, rechazando
videos que el dataset habría aceptado e informando una confiabilidad medida sobre otra
distribución. Ningún test lo detectó porque no había ninguno que comparara las dos ramas.

Lo que se verifica sobre un clip real del conjunto:

  - que las dos ramas elijan exactamente los mismos fotogramas;
  - que produzcan las mismas transformadas de alineación;
  - que los recortes salgan idénticos píxel a píxel;
  - que la pose que recibe el modelo venga indexada por POSICIÓN en la secuencia y no por índice
    de fotograma del video, que es el bug que hizo que los canales visual y geométrico miraran
    momentos distintos del clip.

Necesita el dataset v2 construido (`python3 -m equinos.dataset`) y los clips en disco. NO necesita
GPU ni correr DeepLabCut: consume la pose ya persistida.

Uso:
    python3 tests/test_equivalencia.py
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from equinos import alineado, dataset, rutas, secuencia  # noqa: E402

fallos = []


def check(nombre, condicion, detalle=""):
    if condicion:
        print(f"  OK    {nombre}")
    else:
        print(f"  FALLA {nombre}  {detalle}")
        fallos.append(nombre)


rutas.inicializar(verbose=False)
resultados = rutas.carpeta("resultados")

ruta_dataset = resultados / dataset.ARCHIVO_DATASET
ruta_pose = resultados / "pose_todos_los_frames.csv"
ruta_pose_sel = resultados / dataset.ARCHIVO_POSE_SELECCIONADA
if not (ruta_dataset.exists() and ruta_pose.exists() and ruta_pose_sel.exists()):
    raise SystemExit(
        f"Faltan artefactos del pipeline v2. Construirlos con:\n"
        f"  python3 -m equinos.reproceso\n  python3 -m equinos.dataset")

inventario = {f["clip"]: f for f in csv.DictReader(
    (resultados / "inventario_clips.csv").open(encoding="utf-8", errors="replace"))}

filas_dataset = defaultdict(list)
for fila in csv.DictReader(ruta_dataset.open(encoding="utf-8")):
    filas_dataset[fila["clip"]].append(fila)

pose_cruda = alineado.pose_por_clip(ruta_pose)
pose_sel = defaultdict(list)
for fila in csv.DictReader(ruta_pose_sel.open(encoding="utf-8")):
    pose_sel[fila["clip"]].append(fila)

# Tres clips con ventanas distintas, para no probar siempre el caso fácil.
elegidos = sorted(filas_dataset)[:2]
por_span = sorted(filas_dataset, key=lambda c: -float(filas_dataset[c][0]["span_seg"]))
elegidos.append(por_span[0])

p = alineado.parametros()
geo = alineado.geometria()
destino = alineado.destino_canonico(p)

print("\nMISMA SELECCIÓN DE FOTOGRAMAS EN LAS DOS RAMAS")
for clip in elegidos:
    meta = inventario[clip]
    total, fps = int(meta["frames"]), float(meta["fps"])

    del_dataset = [int(f["frame"]) for f in
                   sorted(filas_dataset[clip], key=lambda f: int(f["orden"]))]
    # La rama de inferencia: misma función, misma pose, sin pasar por el CSV del dataset.
    prep = alineado.preparar_secuencia(pose_cruda[clip], total=total, fps=fps, p=p, geo=geo,
                                       destino=destino, minimo=secuencia.FRAMES_POR_CLIP)
    check(f"{clip[:34]:<34} elige los mismos fotogramas",
          prep.hay_secuencia and prep.ordenes == del_dataset,
          (prep.ordenes[:4], del_dataset[:4]))

print("\nMISMAS TRANSFORMADAS Y MISMOS RECORTES")
import cv2  # noqa: E402

clip = elegidos[0]
meta = inventario[clip]
prep = alineado.preparar_secuencia(pose_cruda[clip], total=int(meta["frames"]),
                                   fps=float(meta["fps"]), p=p, geo=geo, destino=destino,
                                   minimo=secuencia.FRAMES_POR_CLIP)
base = rutas.dir_trabajo(dataset.CARPETA_FRAMES)
guardados = sorted(filas_dataset[clip], key=lambda f: int(f["orden"]))

captura = cv2.VideoCapture(str(rutas.resolver(meta["ruta_clip"])))
iguales, comparados, maxdif = True, 0, 0
for fila, orden, matriz in zip(guardados, prep.ordenes, prep.matrices):
    captura.set(cv2.CAP_PROP_POS_FRAMES, int(orden))
    ok, imagen = captura.read()
    if not ok:
        continue
    recorte_inferencia = alineado.recortar(imagen, matriz, destino)
    # El del dataset viajó por JPEG, así que no puede ser idéntico bit a bit: se compara contra
    # el mismo viaje, que es lo que el backbone ve en las dos ramas.
    _, buf = cv2.imencode(".jpg", recorte_inferencia,
                          [int(cv2.IMWRITE_JPEG_QUALITY), dataset.CALIDAD_JPEG])
    esperado = cv2.imread(str(base / fila["ruta_frame"]))
    obtenido = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if esperado is None:
        continue
    dif = int(np.abs(esperado.astype(int) - obtenido.astype(int)).max())
    maxdif = max(maxdif, dif)
    iguales = iguales and dif == 0
    comparados += 1
captura.release()

check(f"los {comparados} recortes salen idénticos píxel a píxel", iguales and comparados > 0,
      f"diferencia máxima {maxdif}")

print("\nLA POSE LLEGA INDEXADA POR POSICIÓN, NO POR FOTOGRAMA")
for clip in elegidos:
    ordenes = {int(f["orden"]) for f in pose_sel[clip]}
    check(f"{clip[:34]:<34} orden va de 0 a 15",
          ordenes == set(range(secuencia.FRAMES_POR_CLIP)),
          sorted(ordenes)[:5])

# Y que la guarda del modelo rechace la pose cruda, que es la forma del bug.
from equinos import clasificacion  # noqa: E402

clips_falsos = [{"clip": elegidos[0], "clase": "Sereno", "id_caballo": "0"}]
try:
    clasificacion.MultimodalEquineDataset(
        clips_falsos, [None], {elegidos[0]: 0},
        [dict(f, orden=f["orden"]) for f in pose_cruda[elegidos[0]]])
    check("el modelo rechaza una pose indexada por fotograma", False, "no levantó ValueError")
except ValueError as error:
    check("el modelo rechaza una pose indexada por fotograma",
          "posición en la secuencia" in str(error), str(error)[:80])


print("\n" + "=" * 58)
if fallos:
    print(f"FALLARON {len(fallos)}: {fallos}")
    sys.exit(1)
print("TODAS LAS PRUEBAS PASARON")
