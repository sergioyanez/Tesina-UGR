"""Candidato RF de pose: armado del arreglo, hiperparámetros fijos y contrato (datos sintéticos).

Uso:
    python3 tests/test_candidato_rf_pose.py
"""
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "comparacion"))

import numpy as np  # noqa: E402
import candidato_rf_pose as C  # noqa: E402

fallos = []


def check(nombre, condicion, detalle=""):
    if condicion:
        print(f"  OK    {nombre}")
    else:
        print(f"  FALLA {nombre}  {detalle}")
        fallos.append(nombre)


print("\nARREGLO DE POSE")
a = C.arreglo_de_pose([{"orden": "3", "bodypart": "nose", "x": "10.5", "y": "20", "confianza": "0.9"}])
check("forma (16, 39, 3)", a.shape == (16, 39, 3))
check("el punto queda en su lugar", list(a[3, C.BP["nose"]]) == [10.5, 20.0, 0.9])
check("lo ausente es NaN", np.isnan(a[0]).all() and np.isnan(a[3, C.BP["upper_jaw"]]).all())

print("\nHIPERPARÁMETROS Y CONTRATO")
import evaluar as ev  # noqa: E402
check("hiperparámetros iguales a los del experimento", C.RF_PARAMS == ev.RF_PARAMS)
check("cumple el protocolo Candidato", bool(C.RFPoseSimba.nombre)
      and callable(C.RFPoseSimba.ajustar) and callable(C.RFPoseSimba.predecir))

print("\n" + "=" * 52)
if fallos:
    print(f"FALLARON {len(fallos)}: {fallos}")
    sys.exit(1)
print("TODAS LAS PRUEBAS PASARON")
