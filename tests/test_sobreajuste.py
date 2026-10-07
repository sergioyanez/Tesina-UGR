"""Pruebas del registro del sobreajuste en `entrenar_mil_multimodal`.

Medir el train interno en cada época sólo es inocuo si no toca el azar del entrenamiento: si
consumiera el RNG global o el generador del `DataLoader`, las épocas siguientes verían otro orden
de lotes y otras máscaras de dropout, y el modelo medido ya no sería el que se reporta. Nada
fallaría. Se vigila eso, y que la curva traiga las columnas que el informe lee.

La verificación de punta a punta —mismas 359 predicciones que antes del cambio— se hizo al
re-ejecutar `scripts/evaluar_v2.py` y está registrada en `docs/analisis_sobreajuste.md`.

Datos sintéticos, en CPU. Tarda unos segundos.

Uso:
    python3 tests/test_sobreajuste.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from equinos import clasificacion as cl  # noqa: E402

fallos = []


def check(nombre, condicion, detalle=""):
    if condicion:
        print(f"  OK    {nombre}")
    else:
        print(f"  FALLA {nombre}  {detalle}")
        fallos.append(nombre)


def sintetico(n_caballos=24, clips_por_caballo=2):
    rng = np.random.default_rng(0)
    clips, vectores, pose_filas = [], [], []
    for h in range(n_caballos):
        for j in range(clips_por_caballo):
            nombre = f"{h:03d}_clip_{j:02d}.mp4"
            clase = cl.CLASES[(h + j) % 3]
            clips.append({"clip": nombre, "clase": clase, "id_caballo": f"{h:03d}",
                          "fold": h % 4})
            vectores.append(rng.normal(size=(16, 512)).astype(np.float32)
                            + cl.CLASES.index(clase) * 0.3)
            for orden in range(16):
                for parte, x, yy in (("left_eye", 10, 10), ("nose", 30, 40), ("left_earbase", 5, 0)):
                    pose_filas.append({"clip": nombre, "orden": orden, "bodypart": parte,
                                       "x": x + rng.normal(), "y": yy + rng.normal(),
                                       "confianza": 0.9})
    return clips, vectores, {c["clip"]: i for i, c in enumerate(clips)}, pose_filas


clips, vectores, indice, pose_filas = sintetico()

print("=== una pasada en modo eval no consume el azar del entrenamiento ===")
ds = cl.MultimodalEquineDataset(clips, vectores, indice, pose_filas)
modelo = cl.MILFusionModel(dropout=0.3)
torch.manual_seed(7)
generador = torch.Generator()
generador.manual_seed(7)
antes_global, antes_gen = torch.get_rng_state().clone(), generador.get_state().clone()
modelo.eval()
with torch.no_grad():
    for v, g, m, _, _ in cl.cargador_de_medicion(ds, batch_size=8):
        modelo(v, g, m)
check("el RNG global queda intacto", torch.equal(antes_global, torch.get_rng_state()))

# El control: un DataLoader SIN generador propio sí consume el RNG global aunque no baraje. Si
# esta comprobación dejara de fallar, torch cambió y `cargador_de_medicion` sobra.
for _ in DataLoader(ds, batch_size=8, shuffle=False, collate_fn=cl.collate_fn):
    pass
check("(control) un DataLoader sin generador propio sí lo consume",
      not torch.equal(antes_global, torch.get_rng_state()))
check("el generador del DataLoader de train queda intacto",
      torch.equal(antes_gen, generador.get_state()))

print("=== la curva trae las columnas del sobreajuste ===")
entrenamiento = [c for c in clips if c["fold"] != 0]
prueba = [c for c in clips if c["fold"] == 0]
_, extra = cl.entrenar_mil_multimodal(entrenamiento, prueba, vectores, indice, pose_filas,
                                      dropout=0.3, max_epocas=3, paciencia=10,
                                      estrategia_pesos="raiz")
columnas = {"loss_train", "loss_val", "f1_val", "loss_train_eval", "f1_train", "acc_train",
            "acc_val"}
check("cada época registra train y validación", all(columnas <= set(e) for e in extra["curva"]),
      f"columnas: {sorted(extra['curva'][0])}")
check("se informa el F1 de train de la mejor época",
      0.0 <= extra["f1_train"] <= 1.0 and extra["f1_train"] ==
      extra["curva"][extra["mejor_epoca"] - 1]["f1_train"])

print("=== dos corridas iguales dan la misma curva ===")
_, otra = cl.entrenar_mil_multimodal(entrenamiento, prueba, vectores, indice, pose_filas,
                                     dropout=0.3, max_epocas=3, paciencia=10,
                                     estrategia_pesos="raiz")
check("determinista con la semilla del módulo", otra["curva"] == extra["curva"])

print()
if fallos:
    print(f"FALLARON {len(fallos)}: {fallos}")
    sys.exit(1)
print("Todas las pruebas pasaron.")
