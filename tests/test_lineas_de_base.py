"""Pruebas de las líneas de base y del bootstrap por caballo.

Lo que se vigila son fugas y comparaciones que dan números plausibles y falsos:

  - una línea de base que usa la etiqueta del fold de test sale mejor de lo que es, y hace
    parecer peor al modelo;
  - un bootstrap por clip, o diferencias no pareadas, dan intervalos más estrechos de lo que la
    evidencia permite;
  - el subconjunto sin rasgos delatores tiene que excluir exactamente lo que dice.

Datos sintéticos: no requieren GPU, ni los CSV de una corrida.

Uso:
    python3 tests/test_lineas_de_base.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from equinos import lineas_de_base as lb  # noqa: E402

fallos = []


def check(nombre, condicion, detalle=""):
    if condicion:
        print(f"  OK    {nombre}")
    else:
        print(f"  FALLA {nombre}  {detalle}")
        fallos.append(nombre)


rng = np.random.default_rng(0)
CLASES = np.array(["Disgustado", "Interesado", "Sereno"])
n = 120
y = CLASES[rng.integers(0, 3, n)]
fold = np.repeat(np.arange(4), n // 4)
caballos = np.array([f"c{i // 3:03d}" for i in range(n)])          # 3 clips por caballo
X = rng.normal(size=(n, 5)) + (y == "Disgustado")[:, None] * 1.5

print("=== sin fuga: la etiqueta del fold de test no entra a su propia predicción ===")
base = lb.predecir_lineal(X, y, fold)
y_alterada = y.copy()
y_alterada[fold == 2] = "Sereno"
alterada = lb.predecir_lineal(X, y_alterada, fold)
check("cambiar las etiquetas del fold 2 no cambia sus predicciones",
      (base[fold == 2] == alterada[fold == 2]).all())
check("pero sí las de los folds que entrenan con él",
      not (base[fold != 2] == alterada[fold != 2]).all())

may = lb.predecir_mayoritaria(y_alterada, fold)
check("la mayoritaria del fold 2 sale de los otros folds, no del propio",
      may[fold == 2][0] == lb.predecir_mayoritaria(y, fold)[fold == 2][0])

print("=== bootstrap por caballo ===")
perfecto = y.copy()
malo = np.where(rng.random(n) < 0.5, y, "Sereno")
b = lb.bootstrap_por_caballo(y, {"perfecto": perfecto, "malo": malo, "copia": perfecto.copy()},
                             caballos, referencia="perfecto", replicas=200)
check("cuenta caballos, no clips", b["caballos"] == n // 3, f"dio {b['caballos']}")
check("un modelo perfecto tiene IC [1; 1]", b["modelos"]["perfecto"]["f1_ic95"] == [1.0, 1.0])
dif_copia = b["modelos"]["copia"]["diferencia_referencia_menos_este"]
check("la diferencia contra sí mismo es exactamente cero en todas las réplicas (pareada)",
      dif_copia["ic95"] == [0.0, 0.0] and dif_copia["incluye_cero"])
check("contra un modelo peor, la diferencia es positiva y excluye el cero",
      b["modelos"]["malo"]["diferencia_referencia_menos_este"]["ic95"][0] > 0)
again = lb.bootstrap_por_caballo(y, {"perfecto": perfecto, "malo": malo}, caballos,
                                 referencia="perfecto", replicas=200)
check("determinista con la semilla fija",
      again["modelos"]["malo"]["f1_ic95"] == b["modelos"]["malo"]["f1_ic95"])

print("=== resumen y subconjunto ===")
r = lb.resumen(y, perfecto, fold)
check("resumen perfecto: F1 1 en los 4 folds y desvío 0",
      r["por_fold"] == [1.0] * 4 and r["f1_desvio"] == 0.0)
meta = np.array([[6.0, 30, 1080, 1920, 5.9],      # entra
                 [4.9, 30, 1080, 1920, 4.0],      # corto
                 [6.0, 30, 640, 480, 5.9],        # otra resolución
                 [5.0, 30, 1080, 1920, 4.9]])     # justo en el umbral: entra
mascara = lb.sin_rasgos_delatores(None, meta)
check("excluye cortos y otras resoluciones, admite el umbral",
      mascara.tolist() == [True, False, False, True], f"dio {mascara.tolist()}")

print()
if fallos:
    print(f"FALLARON {len(fallos)}: {fallos}")
    sys.exit(1)
print("Todas las pruebas pasaron.")
