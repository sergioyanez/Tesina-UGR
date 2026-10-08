"""Candidato 3: fusión tardía pose + visual.

- visual: media de los 16 frames (ResNet-18 congelada) -> StandardScaler + LogReg (C=0,1, balanced),
  los mismos hiperparámetros del baseline.
- pose: las 160 features «tipo SimBA» (`experimentos/simba/features_simba.py`, sin duplicar) ->
  imputación por mediana + RandomForest (500 árboles, min_samples_leaf=2, balanced), como en la auditoría.
- Fusión: p = w·p_pose + (1-w)·p_visual, w ∈ {0; 0,25; 0,5; 0,75; 1}. El w se elige DENTRO de
  `ajustar` con probabilidades fuera de muestra de una validación interna de 4 pliegues por caballo
  (StratifiedGroupKFold sobre el train; nunca con el test del fold) maximizando F1 macro; en empate
  gana el w más cercano a 0,5. Después ambos modelos se re-entrenan con todo el train.
Variante fijada antes de mirar resultados de test: tardía (la temprana no se evaluó).

Uso: TESINA_EQUINOS_RAIZ=<datos> python comparacion/candidato_fusion.py
"""
import os
import sys
from pathlib import Path

RAIZ_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ_REPO))
sys.path.insert(0, str(RAIZ_REPO / "experimentos" / "simba"))

PESOS = (0.0, 0.25, 0.5, 0.75, 1.0)
PLIEGUES_INTERNOS = 4
SEMILLA = 0


def _preparar_simba():
    """`experimentos/simba/datos.py` toma la raíz de DATOS_EQUINOS: la alineamos con la de equinos."""
    from equinos import rutas
    os.environ.setdefault("DATOS_EQUINOS", str(rutas.raiz()))


def features_pose(ids, datos):
    """Matriz (n, 160) tipo SimBA para los ids; se calcula una vez por clip y se cachea en `datos`."""
    import numpy as np
    _preparar_simba()
    import features_simba as F
    from datos import BP, BODYPARTS  # experimentos/simba/datos.py

    cache = datos.__dict__.setdefault("_cache_simba", {})
    faltan = [i for i in ids if i not in cache]
    if faltan:
        import csv
        from equinos import rutas
        P = {}
        for f in datos.pose(faltan):
            a = P.setdefault(f["clip"], np.full((16, len(BODYPARTS), 3), np.nan))
            a[int(f["orden"]), BP[f["bodypart"]]] = [float(f["x"]), float(f["y"]), float(f["confianza"])]
        tiempos = {}
        with (rutas.carpeta("resultados") / "dataset_frames_alineados_v2.csv").open(
                encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r["clip"] in P:
                    tiempos.setdefault(r["clip"], []).append((int(r["orden"]), float(r["frame"]),
                                                              float(r["fps"])))
        for i in faltan:
            t = sorted(tiempos[i])
            frames = np.array([x[1] for x in t])
            cache[i] = F.vector_del_clip(P[i], frames, t[0][2])[0]
    return np.stack([cache[i] for i in ids])


def features_visual(ids, datos):
    import numpy as np
    return np.stack([datos.visual(i).mean(axis=0) for i in ids])


def _modelo_pose():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                         RandomForestClassifier(n_estimators=500, max_features="sqrt",
                                                min_samples_leaf=2, class_weight="balanced",
                                                random_state=SEMILLA, n_jobs=4))


def _modelo_visual():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from equinos import lineas_de_base
    return make_pipeline(StandardScaler(), LogisticRegression(
        C=lineas_de_base.C_ALTA_DIMENSION, max_iter=5000, class_weight="balanced"))


def elegir_peso(p_pose, p_vis, y, clases, pesos=PESOS):
    """Peso w con mejor F1 macro sobre probabilidades fuera de muestra; empate -> el más cercano a 0,5."""
    import numpy as np
    from equinos.comparacion import f1_macro
    y = list(y)
    puntaje = {}
    for w in pesos:
        pred = np.asarray(clases)[(w * p_pose + (1 - w) * p_vis).argmax(axis=1)]
        puntaje[w] = f1_macro(y, list(pred))
    mejor = max(puntaje.values())
    w = min((w for w in pesos if puntaje[w] >= mejor - 1e-12), key=lambda w: abs(w - 0.5))
    return w, puntaje


class FusionPoseVisual:
    nombre = "fusion_tardia_pose_visual"

    def ajustar(self, ids_train, datos):
        import numpy as np
        from sklearn.model_selection import StratifiedGroupKFold

        ids = sorted(ids_train)
        y = np.array(datos.etiquetas(ids))
        g = np.array([datos.caballo(i) for i in ids])
        Xp, Xv = features_pose(ids, datos), features_visual(ids, datos)
        clases = np.array(sorted(set(y)))
        p_pose = np.zeros((len(ids), len(clases)))
        p_vis = np.zeros_like(p_pose)
        cv = StratifiedGroupKFold(n_splits=PLIEGUES_INTERNOS, shuffle=True, random_state=SEMILLA)
        for tr, va in cv.split(Xp, y, g):
            p_pose[va] = _modelo_pose().fit(Xp[tr], y[tr]).predict_proba(Xp[va])
            p_vis[va] = _modelo_visual().fit(Xv[tr], y[tr]).predict_proba(Xv[va])
        self.peso, self.puntajes_internos = elegir_peso(p_pose, p_vis, y, clases)
        self.modelo_pose = _modelo_pose().fit(Xp, y)
        self.modelo_visual = _modelo_visual().fit(Xv, y)
        self.clases = self.modelo_pose.classes_

    def predecir(self, ids, datos):
        pp = self.modelo_pose.predict_proba(features_pose(ids, datos))
        pv = self.modelo_visual.predict_proba(features_visual(ids, datos))
        return self.clases[(self.peso * pp + (1 - self.peso) * pv).argmax(axis=1)].tolist()


if __name__ == "__main__":
    import json
    import time
    from equinos import comparacion

    pesos = []

    class _Registro(FusionPoseVisual):
        def ajustar(self, ids_train, datos):
            super().ajustar(ids_train, datos)
            pesos.append(self.peso)

    t0 = time.time()
    r = comparacion.evaluar(_Registro, FusionPoseVisual.nombre,
                            "Fusión tardía: RF sobre 160 features de pose + LogReg sobre visual medio; "
                            "peso w elegido por validación interna por caballo")
    g = r["global"]
    print(f"F1 macro {g['f1_macro']['media']:.3f} ± {g['f1_macro']['desvio']:.3f} "
          f"| limpio {r['limpio']['f1_macro_oof_por_repeticion']['media']:.3f} "
          f"± {r['limpio']['f1_macro_oof_por_repeticion']['desvio']:.3f} "
          f"| gap {g['gap_train_test_f1']['media']:+.3f}")
    from collections import Counter
    print("pesos:", dict(sorted(Counter(pesos).items())), f"| {time.time() - t0:.0f} s")
