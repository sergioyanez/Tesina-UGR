"""SHAP (TreeExplainer) del Random Forest sobre features tipo SimBA, OUT-OF-FOLD.

Por fold se reentrena el RF con el train (mismos parámetros y semilla que `evaluar.py`) y se
explica SU test con `shap.TreeExplainer`. SHAP descompone cada predicción en contribuciones
aditivas de cada feature RESPECTO DE UN VALOR BASE (el valor esperado de la salida del modelo sobre
los datos de entrenamiento: prob. media de la clase); no mide causalidad, y con features
correlacionadas reparte el crédito entre ellas. Lectura etológica: solo asociación, requiere
validación experta.

Salidas: resultados/shap.json, shap_valores_oof.npz y figuras shap_resumen_<clase>.png (200 dpi).
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

import datos
import evaluar as ev

FAMILIAS = {"oreja": "Orejas", "ojos": "Ojos", "cabeza": "Cabeza (orientación/dinámica)",
            "boca": "Boca/mandíbula", "cuello": "Cabeza-cuello", "disp": "Disponibilidad de keypoints"}


def main():
    df = pd.read_csv(datos.SALIDA / "dataset_tabular_simba.csv")
    cols = json.loads((datos.SALIDA / "columnas.json").read_text(encoding="utf-8"))
    nombres = cols["features"]
    X = df[nombres].to_numpy(float); y = df["clase"].to_numpy(object); fold = df["fold"].to_numpy()
    oof = pd.read_csv(datos.SALIDA / "oof_predicciones.csv")["B_simba_propio|RF"].to_numpy(object)
    sv = np.zeros((len(y), len(nombres), 3)); Ximp = np.zeros_like(X); base = {}; imp_rf = []
    pred = np.empty(len(y), object)
    for k in range(4):
        tr, te = fold != k, fold == k
        m = ev.hacer_modelo("RF", X.shape[1]).fit(X[tr], y[tr])
        imp, rf = m[0], m[-1]
        assert list(rf.classes_) == datos.CLASES
        Xtr, Xte = imp.transform(X[tr]), imp.transform(X[te])
        pred[te] = rf.predict(Xte)
        expl = shap.TreeExplainer(rf, feature_perturbation="tree_path_dependent")
        v = expl.shap_values(Xte, check_additivity=True)
        v = np.stack(v, axis=-1) if isinstance(v, list) else v
        sv[te] = v; Ximp[te] = Xte
        base[int(k)] = [float(b) for b in np.atleast_1d(expl.expected_value)]
        imp_rf.append(rf.feature_importances_)
    assert (pred == oof).all(), "el RF de SHAP no reproduce las predicciones OOF"

    glob = np.abs(sv).mean(axis=(0, 2))                       # media |SHAP| sobre clips y clases
    por_clase = {c: np.abs(sv[:, :, i]).mean(0) for i, c in enumerate(datos.CLASES)}
    # firma: correlación (Spearman) entre valor de la feature y su SHAP para esa clase (solo asociación)
    def rank(a): return pd.Series(a).rank().to_numpy()
    def top(imp, n=20, extra=None):
        o = np.argsort(-imp)[:n]
        return [{"rank": i + 1, "feature": nombres[j], "media_abs_shap": float(imp[j]), "grupo": cols["grupos"][nombres[j]],
                 **({k: float(v[j]) for k, v in (extra or {}).items()})} for i, j in enumerate(o)]
    dirs = {}
    for i, c in enumerate(datos.CLASES):
        d = np.array([np.corrcoef(rank(Ximp[:, j]), rank(sv[:, j, i]))[0, 1] if Ximp[:, j].std() > 0 and sv[:, j, i].std() > 0 else 0.0
                      for j in range(len(nombres))])
        dirs[c] = d
    imp_rf = np.mean(imp_rf, axis=0)
    fam = {}
    for j, n in enumerate(nombres):
        f = next((v for k, v in FAMILIAS.items() if n.startswith(k)), "Otros")
        fam[f] = fam.get(f, 0.0) + float(glob[j])
    tot = sum(fam.values())
    salida = {
        "explicacion": "SHAP (TreeExplainer) out-of-fold: cada fold explica su test con el RF entrenado con los otros 3 folds. "
                       "Valores = contribuciones respecto del valor base (valor esperado de la salida del modelo). "
                       "Asociación, no causalidad; requiere validación experta.",
        "valores_base_por_fold_(prob_media_por_clase_Disg_Int_Ser)": base,
        "importancia_global_top20": top(glob),
        "importancia_por_clase_top20": {c: top(por_clase[c], extra={"spearman_valor_vs_shap": dirs[c]}) for c in datos.CLASES},
        "importancia_impureza_RF_top20": top(imp_rf),
        "aporte_por_familia_anatomica_fraccion": {k: v / tot for k, v in sorted(fam.items(), key=lambda kv: -kv[1])},
        "aporte_por_grupo_fraccion": {g: float(sum(glob[j] for j, n in enumerate(nombres) if cols["grupos"][n] == g) / glob.sum())
                                      for g in ("estatica", "dinamica", "disponibilidad")},
    }
    (datos.SALIDA / "shap.json").write_text(json.dumps(salida, ensure_ascii=False, indent=1), encoding="utf-8")
    np.savez_compressed(datos.SALIDA / "shap_valores_oof.npz", shap=sv, X_imputado=Ximp, features=np.array(nombres), clases=np.array(datos.CLASES))
    pd.DataFrame(top(glob, 20)).to_csv(datos.SALIDA / "shap_top20_global.csv", index=False)

    for i, c in enumerate(datos.CLASES):
        plt.figure()
        shap.summary_plot(sv[:, :, i], Ximp, feature_names=nombres, max_display=20, show=False, plot_size=(9, 7.5))
        plt.title(f"SHAP out-of-fold, clase {c} (RF, features tipo SimBA)", fontsize=10)
        plt.tight_layout(); plt.savefig(datos.SALIDA / f"shap_resumen_{c}.png", dpi=200); plt.close("all")
    print(json.dumps([(t["feature"], round(t["media_abs_shap"], 4)) for t in salida["importancia_global_top20"][:10]], ensure_ascii=False))
    print(salida["aporte_por_familia_anatomica_fraccion"], salida["aporte_por_grupo_fraccion"])


if __name__ == "__main__":
    main()
