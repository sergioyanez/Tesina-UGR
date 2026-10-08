"""Candidato 2: Random Forest sobre 160 features geométricas de pose «tipo SimBA» (código propio).

Reutiliza `experimentos/simba/features_simba.py` (no se duplica). Hiperparámetros: los del
experimento (`RF_PARAMS` de `experimentos/simba/evaluar.py`: 500 árboles, sqrt, min_samples_leaf=2,
class_weight balanced). Fueron fijados a priori en la auditoría, no elegidos mirando el test, así que
NO se ajusta nada dentro de `ajustar`. Imputación por mediana del train (dentro de cada fold).
Uso: TESINA_EQUINOS_RAIZ=<datos> python comparacion/candidato_rf_pose.py
"""
import csv
import os
import sys
from pathlib import Path

RAIZ_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ_REPO))
sys.path.insert(0, str(RAIZ_REPO / "experimentos" / "simba"))
from equinos import comparacion, rutas  # noqa: E402

# `datos` de experimentos/simba lee la raíz de DATOS_EQUINOS; se alinea con la raíz del pipeline.
os.environ.setdefault("DATOS_EQUINOS", str(rutas.raiz_por_defecto()))
import features_simba as F  # noqa: E402
from datos import BP  # noqa: E402

N_FRAMES = 16
RF_PARAMS = dict(n_estimators=500, max_features="sqrt", min_samples_leaf=2,
                 class_weight="balanced", random_state=0, n_jobs=-1)
ARCHIVO_TIEMPOS = "dataset_frames_alineados_v2.csv"


def _tiempos():
    """`{clip: (frames_de_video (16,), fps)}` del dataset de frames alineados v2 (sólo lectura)."""
    import numpy as np
    por_clip = {}
    with (rutas.carpeta("resultados") / ARCHIVO_TIEMPOS).open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            por_clip.setdefault(r["clip"], []).append((int(r["orden"]), float(r["frame"]), float(r["fps"])))
    return {c: (np.array([x[1] for x in sorted(v)]), sorted(v)[0][2]) for c, v in por_clip.items()}


def arreglo_de_pose(filas):
    """Filas de `Datos.pose` de un clip → array (16, 39, 3) = (x, y, confianza); NaN si falta."""
    import numpy as np
    a = np.full((N_FRAMES, len(BP), 3), np.nan)
    for r in filas:
        a[int(r["orden"]), BP[r["bodypart"]]] = (float(r["x"]), float(r["y"]), float(r["confianza"]))
    return a


class RFPoseSimba:
    nombre = "rf_pose_tipo_simba"

    def _matriz(self, ids, datos):
        import numpy as np
        if not hasattr(self, "_t"):
            self._t = _tiempos()
        por_clip = {}
        for r in datos.pose(ids):
            por_clip.setdefault(r["clip"], []).append(r)
        filas = []
        for i in ids:
            frames, fps = self._t[i]
            filas.append(F.vector_del_clip(arreglo_de_pose(por_clip[i]), frames, fps)[0])
        return np.stack(filas)

    def ajustar(self, ids_train, datos):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import make_pipeline

        self.modelo = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                                    RandomForestClassifier(**RF_PARAMS))
        self.modelo.fit(self._matriz(ids_train, datos), datos.etiquetas(ids_train))

    def predecir(self, ids, datos):
        return self.modelo.predict(self._matriz(ids, datos)).tolist()


if __name__ == "__main__":
    r = comparacion.evaluar(RFPoseSimba, RFPoseSimba.nombre,
                            "RF (500 árboles, min_samples_leaf=2, balanced) sobre 160 features de pose tipo SimBA")
    g = r["global"]
    print(f"F1 macro {g['f1_macro']['media']:.3f} ± {g['f1_macro']['desvio']:.3f} "
          f"| limpio {r['limpio']['f1_macro_oof_por_repeticion']['media']:.3f} "
          f"± {r['limpio']['f1_macro_oof_por_repeticion']['desvio']:.3f} "
          f"| gap {g['gap_train_test_f1']['media']:+.3f}")
