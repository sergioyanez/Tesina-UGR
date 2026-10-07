"""Geometría v2 y máscaras con el umbral del pipeline (>= 0,50), no con `confianza > 0`.

Uso:
    python3 tests/test_geometria_v2.py
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


import numpy as np

from equinos import geometria, inferencia
from equinos.clasificacion import MultimodalEquineDataset

print("\nEL UMBRAL ES EL DEL PIPELINE")
p = inferencia.parametros_etapa02()
check("geometria.CONF_MINIMA_KEYPOINT == parámetro de la etapa 02",
      geometria.CONF_MINIMA_KEYPOINT == p["CONF_MINIMA_KEYPOINT"] == 0.50)


def fila(bp, x, y, c, orden=0, clip="c.mp4"):
    return {"clip": clip, "orden": orden, "bodypart": bp, "x": x, "y": y, "confianza": c}


def pose(conf_oreja, conf_ojo=0.9, conf_oreja_der=0.9):
    f = []
    for o in range(2):
        f += [fila("left_eye", 100, 100, conf_ojo, o), fila("right_eye", 120, 100, 0.9, o),
              fila("nose", 100, 200, 0.9, o), fila("lower_jaw", 105, 215, 0.8, o),
              fila("mouth_end_left", 95, 205, 0.8, o), fila("mouth_end_right", 110, 205, 0.8, o),
              fila("left_earbase", 90, 80, conf_oreja, o), fila("left_earend", 85, 50, conf_oreja, o),
              fila("right_earbase", 130, 80, conf_oreja_der, o), fila("right_earend", 135, 50, conf_oreja_der, o)]
    return f


def ds(filas, **kw):
    clips = [{"clip": "c.mp4", "clase": "Sereno"}]
    return MultimodalEquineDataset(clips, [np.zeros((2, 512), np.float32)], {"c.mp4": 0}, filas, **kw)


print("\nV1: LA MÁSCARA VIEJA ES SIEMPRE 1; CON UMBRAL 0,50 INFORMA")
baja = pose(conf_oreja=0.30, conf_oreja_der=0.30)
_, g_viejo, _ = ds(baja)[0]
_, g_nuevo, _ = ds(baja, umbral_mascara=0.50)[0]
check("v1 histórico: las orejas con confianza 0,30 sigue marcada válida (comportamiento del modelo desplegado)",
      float(g_viejo[0, 8]) == 1.0)
check("v1 con umbral 0,50: las orejas con confianza 0,30 queda inválida", float(g_nuevo[0, 8]) == 0.0)
check("v1 con umbral 0,50: el ojo y el hocico medidos siguen válidos",
      float(g_nuevo[0, 6]) == 1.0 and float(g_nuevo[0, 7]) == 1.0)
_, g_borde, _ = ds(pose(conf_oreja=0.50), umbral_mascara=0.50)[0]
check("el umbral es inclusivo (>= 0,50)", float(g_borde[0, 8]) == 1.0)

print("\nV2")
_, g2, _ = ds(pose(conf_oreja=0.30), geometria="v2")[0]
check("v2 tiene 17 columnas por frame", tuple(g2.shape) == (2, 17) and geometria.DIM["v2"] == 17, g2.shape)
check("v2: oreja izquierda con 0,30 de confianza -> máscara 0 y valores 0",
      float(g2[0, 11]) == 0.0 and float(g2[0, 0]) == 0.0 and float(g2[0, 1]) == 0.0)
check("v2: oreja derecha medida -> máscara 1", float(g2[0, 12]) == 1.0)
_, g2b, _ = ds(pose(conf_oreja=0.9, conf_ojo=0.2), geometria="v2")[0]
check("v2: ojo izquierdo débil, el derecho define el eje (máscara de eje 1)", float(g2b[0, 16]) == 1.0)
sin_ojos = [f for f in pose(0.9) if "eye" not in f["bodypart"]]
_, g2c, _ = ds(sin_ojos, geometria="v2")[0]
check("v2: sin ojos no hay marco de cabeza: todo en 0", float(np.abs(g2c.numpy()).sum()) == 0.0)
check("v2: la escala se normaliza por la cabeza (mandíbula ~ 0,07 de largo ojo-hocico)",
      0.0 < abs(float(g2[0, 4])) + abs(float(g2[0, 5])) < 1.0)
x = np.array([[abs(float(v)) for v in r] for r in g2.numpy()])
check("v2 aporta más de un grado de libertad que ojo+hocico", len({round(v, 4) for v in x[0, :9]}) > 3)

print("\nHUELLAS")
check("v1 y v2 tienen huellas distintas", geometria.huella("v1") != geometria.huella("v2"))
check("el umbral entra en la huella", geometria.huella("v2", 0.5) != geometria.huella("v2", 0.3))
check("la huella es estable", geometria.huella("v2") == geometria.huella("v2"))

print("\n" + "=" * 52)
if fallos:
    print(f"FALLARON {len(fallos)}: {fallos}")
    sys.exit(1)
print("TODAS LAS PRUEBAS PASARON")

