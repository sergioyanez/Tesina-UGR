"""El candidato MIL corregido cumple el contrato de `comparacion.Candidato` con datos sintéticos."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "comparacion"))
import candidato_mil  # noqa: E402

CLASES = ("Disgustado", "Interesado", "Sereno")


class _DatosFalsos:
    def __init__(self):
        rng = np.random.default_rng(0)
        self.clips, self._vis, self._pose_filas = [], {}, []
        for h in range(12):
            for j in range(2):
                clase = CLASES[(h + j) % 3]
                cid = f"h{h}_c{j}"
                self.clips.append({"clip": cid, "clase": clase, "id_caballo": f"h{h}"})
                self._vis[cid] = (rng.normal(size=(16, 512)) + CLASES.index(clase)).astype(np.float32)
                for orden in range(16):
                    for bp, (x, y) in {"left_eye": (0, 0), "nose": (10, 0), "left_earbase": (0, -5),
                                       "left_earend": (1, -9), "lower_jaw": (10, 4)}.items():
                        self._pose_filas.append({"clip": cid, "orden": str(orden), "bodypart": bp,
                                                 "x": str(x), "y": str(y), "confianza": "0.9"})

    def etiqueta(self, i): return next(c["clase"] for c in self.clips if c["clip"] == i)
    def caballo(self, i): return next(c["id_caballo"] for c in self.clips if c["clip"] == i)
    def visual(self, i): return self._vis[i]
    def pose(self, ids=None): return self._pose_filas


fallos = []


def check(nombre, cond):
    print(f"  {'OK ' if cond else 'FALLA'} {nombre}")
    if not cond:
        fallos.append(nombre)


datos = _DatosFalsos()
ids = [c["clip"] for c in datos.clips]
cand = candidato_mil.MILMultimodalCorregido(max_epocas=3, paciencia=2)
cand.ajustar(ids[:18], datos)
pred = cand.predecir(ids[18:], datos)
check("una clase válida por id", len(pred) == 6 and set(pred) <= set(CLASES))
check("geometría v2 (17 columnas)", cand.modelo.geom_mlp[0].in_features == 17)
check("época elegida dentro del rango", 1 <= cand.info["mejor_epoca"] <= 3)
check("train interno + validación = train", cand.info["clips_train_interno"] + cand.info["clips_validacion"] == 18)
check("umbral de máscara 0,50", candidato_mil.UMBRAL_MASCARA == 0.50)

print("\n" + "=" * 52)
if fallos:
    print(f"FALLARON {len(fallos)}: {fallos}")
    sys.exit(1)
print("TODAS LAS PRUEBAS PASARON")
