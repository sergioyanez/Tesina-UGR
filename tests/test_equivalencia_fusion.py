"""Las features y la predicción de la fusión por la ruta de INFERENCIA coinciden con las del entrenamiento.

Es el equivalente de `test_equivalencia.py` para la fusión. El entrenamiento calcula las 160
features con `candidato_fusion.features_pose` (pose persistida + `dataset_frames_alineados_v2.csv`);
la inferencia, con `equinos.fusion.features_de_pose` a partir de las filas de pose de
`preparar_clip` (floats, `orden` = posición) y los índices de fotograma del video. Si no dieran lo
mismo, el modelo se mediría sobre una entrada y se serviría sobre otra sin que nada fallara.

Para no probar sólo el caso fácil:
  - los índices de fotograma se desplazan (el video los tiene absolutos; el dataset, relativos al
    clip): las features dinámicas dependen sólo de las diferencias;
  - las filas de pose llegan mezcladas y como floats.

Necesita el dataset v2, el modelo final (`python comparacion/entrenar_modelo_final.py`) y no usa GPU
ni DeepLabCut.

Uso: python tests/test_equivalencia_fusion.py
"""
import csv
import random
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "comparacion"))

import numpy as np  # noqa: E402

import candidato_fusion as cf  # noqa: E402
from equinos import comparacion, fusion, inferencia, rutas  # noqa: E402

fallos = []


def check(nombre, condicion, detalle=""):
    if condicion:
        print(f"  OK    {nombre}")
    else:
        print(f"  FALLA {nombre}  {detalle}")
        fallos.append(nombre)


rutas.inicializar(verbose=False)
datos = comparacion.Datos()
paquete = fusion.cargar_paquete()

tiempos = {}
with (rutas.carpeta("resultados") / "dataset_frames_alineados_v2.csv").open(encoding="utf-8") as f:
    for r in csv.DictReader(f):
        tiempos.setdefault(r["clip"], []).append((int(r["orden"]), float(r["frame"]), float(r["fps"])))

ids = sorted(datos.ids())
azar = random.Random(0)
elegidos = azar.sample(ids, 12)
X_entrenamiento = cf.features_pose(elegidos, datos)

print("\nFEATURES DE POSE: INFERENCIA == ENTRENAMIENTO")
for k, clip in enumerate(elegidos):
    t = sorted(tiempos[clip])
    frames = np.array([x[1] for x in t]) + 1000 + 7 * k  # índices absolutos del video
    filas = [dict(f, x=float(f["x"]), y=float(f["y"]), confianza=float(f["confianza"]),
                  orden=int(f["orden"])) for f in datos.pose([clip])]
    azar.shuffle(filas)
    v = fusion.features_de_pose(filas, frames, t[0][2])
    ref = X_entrenamiento[k]
    check(f"{clip}: 160 features iguales", v.shape == (160,) and np.allclose(v, ref, equal_nan=True),
          f"máx dif {np.nanmax(np.abs(v - ref)) if v.shape == ref.shape else v.shape}")
    check(f"{clip}: NaN en las mismas posiciones", np.array_equal(np.isnan(v), np.isnan(ref)))

print("\nPREDICCIÓN: INFERENCIA == ENTRENAMIENTO")
coinciden_etiqueta = 0
for k, clip in enumerate(elegidos):
    t = sorted(tiempos[clip])
    frames = np.array([x[1] for x in t]) + 500
    filas = [dict(f, x=float(f["x"]), y=float(f["y"]), confianza=float(f["confianza"]),
                  orden=int(f["orden"])) for f in datos.pose([clip])]
    vis = datos.visual(clip)
    pred = inferencia.predecir_con_fusion(vis, filas, frames, t[0][2], paquete=paquete)
    pp = paquete["modelo_pose"].predict_proba(X_entrenamiento[k][None, :])[0]
    pv = paquete["modelo_visual"].predict_proba(vis.mean(axis=0)[None, :])[0]
    ref = paquete["peso"] * pp + (1 - paquete["peso"]) * pv
    obtenida = np.array([pred.probabilidades[c] for c in paquete["clases"]])
    check(f"{clip}: probabilidades iguales", np.allclose(obtenida, ref, atol=1e-12),
          f"{obtenida} vs {ref}")
    check(f"{clip}: misma clase",
          pred.clase == paquete["clases"][int(ref.argmax())])
    coinciden_etiqueta += pred.clase == datos.etiqueta(clip)
print(f"\n  (informativo) {coinciden_etiqueta}/{len(elegidos)} coinciden con la etiqueta; son clips "
      "con que se entrenó el modelo final: es coherencia, no generalización)")

print("\nMETADATOS DEL MODELO FINAL")
m = paquete["metadatos"]
check("la huella del dataset es la vigente", m["huella_dataset"] == datos.huella)
check("entrenado con todos los clips", sorted(m["clips"]) == ids and m["n_clips"] == len(ids))
check("el peso es uno de los admitidos", paquete["peso"] in cf.PESOS)

print("\n" + "=" * 58)
if fallos:
    print(f"FALLARON {len(fallos)}: {fallos}")
    sys.exit(1)
print("TODAS LAS PRUEBAS PASARON")
