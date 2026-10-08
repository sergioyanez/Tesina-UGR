"""Baseline del protocolo: LogReg sobre la media de los 16 frames (ResNet-18 congelada).

Mismos hiperparámetros fijos que `equinos.lineas_de_base` (StandardScaler + LogReg C=0,1,
class_weight balanced). Es la prueba de humo del evaluador y el candidato de referencia.
Uso: TESINA_EQUINOS_RAIZ=<datos> python comparacion/correr_baseline.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from equinos import comparacion, lineas_de_base  # noqa: E402


class LogRegMediaFrames:
    nombre = "baseline_logreg_media_frames"

    def ajustar(self, ids_train, datos):
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        X = np.stack([datos.visual(i).mean(axis=0) for i in ids_train])
        self.modelo = make_pipeline(StandardScaler(), LogisticRegression(
            C=lineas_de_base.C_ALTA_DIMENSION, max_iter=5000, class_weight="balanced"))
        self.modelo.fit(X, datos.etiquetas(ids_train))

    def predecir(self, ids, datos):
        import numpy as np
        X = np.stack([datos.visual(i).mean(axis=0) for i in ids])
        return self.modelo.predict(X).tolist()


if __name__ == "__main__":
    r = comparacion.evaluar(LogRegMediaFrames, LogRegMediaFrames.nombre,
                            "LogReg (C=0,1, balanced) sobre la media de los 16 frames visuales")
    g = r["global"]
    print(f"F1 macro {g['f1_macro']['media']:.3f} ± {g['f1_macro']['desvio']:.3f} "
          f"| limpio {r['limpio']['f1_macro_oof_por_repeticion']['media']:.3f} "
          f"± {r['limpio']['f1_macro_oof_por_repeticion']['desvio']:.3f} "
          f"| gap {g['gap_train_test_f1']['media']:+.3f}")
