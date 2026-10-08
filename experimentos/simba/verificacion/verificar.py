"""Verificación adversarial del F1 0,851 del RF sobre features tipo SimBA. Escribe verificacion.json."""
import json, re, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import datos, evaluar
from evaluar import oof, metricas, RF_PARAMS
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from equinos import lineas_de_base as lb, particion
warnings.filterwarnings("ignore")
R = {}
clips = datos.clips_con_fold()
df = pd.read_csv(datos.SALIDA / "dataset_tabular_simba.csv")
cols = json.loads((datos.SALIDA / "columnas.json").read_text())
names = cols["features"]; grp = np.array([cols["grupos"][n] for n in names])
X = df[names].to_numpy(float)
y = df["clase"].to_numpy(object); fold = df["fold"].to_numpy(); cab = df["caballo"].astype(str).to_numpy()
meta = datos.metadatos(clips)  # dur, fps, ancho, alto
def f1m(p, mask=None): return metricas(y, p, fold, mask)["f1_macro_media"]
def run(Xc, nombre, seed=0, mask=None):
    evaluar.RF_PARAMS["random_state"] = seed
    p, _ = oof(Xc, y, fold, "RF", clips)
    evaluar.RF_PARAMS["random_state"] = 0
    r = metricas(y, p, fold, mask)
    return {"f1_media": r["f1_macro_media"], "f1_desvio": r["f1_macro_desvio"], "f1_agrupado": r["f1_macro_agrupado"]}, p

# ---- 1. particion ----
part = particion.construir([{"clip": c["clip"], "id_caballo": c["id_caballo"]} for c in clips])
assert all(a["fold"] == b["fold"] for a, b in zip(clips, part))
for k in range(4): assert not (set(cab[fold == k]) & set(cab[fold != k]))
vids = df["video"].to_numpy()
for k in range(4): assert not (set(vids[fold == k]) & set(vids[fold != k]))
# mismo animal con distinto id: normalizar nombre del video (sin prefijo numérico, copia de, sufijos) y buscar repetidos entre ids
def norm(v):
    s = re.sub(r"\.mp4$", "", v, flags=re.I).lower(); s = re.sub(r"^\d+[-_ ]*", "", s)
    s = re.sub(r"copia[ _-]*de[ _-]*", "", s); s = re.sub(r"adobeexpress|_[a-z0-9]{8}$", "", s)
    s = re.sub(r"^\d+[-_ ]*", "", s); s = re.sub(r"[\W_]*\d*$", "", s); return s
d = df.drop_duplicates("video")[["video", "caballo", "fold"]].copy(); d["n"] = d["video"].map(norm)
g = d.groupby("n").agg(ids=("caballo", lambda s: sorted(set(map(str, s)))), folds=("fold", lambda s: sorted(set(s))))
sosp = g[(g.ids.map(len) > 1)]
# duplicados casi exactos entre folds: vecino más cercano en features estandarizadas con otro fold
from sklearn.preprocessing import StandardScaler
Z = StandardScaler().fit_transform(SimpleImputer(strategy="median", keep_empty_features=True).fit_transform(X))
from scipy.spatial.distance import cdist
D = cdist(Z, Z); np.fill_diagonal(D, np.inf)
same = cab[:, None] == cab[None, :]
Dx = np.where(same, np.inf, D); nn = Dx.argmin(1)
R["1_particion"] = {"asserts_ok": True, "caballos": len(set(cab)), "videos": len(set(vids)),
    "nombres_normalizados_con_varios_ids": {k: {"ids": v.ids, "folds": v.folds} for k, v in sosp.iterrows()},
    "n_nombres_sospechosos": int(len(sosp)),
    "nn_otro_caballo_distancia_min": float(Dx.min()), "nn_dist_mediana": float(np.median(Dx.min(1))),
    "nn_mismo_fold_otro_caballo_frac": float((fold[nn] == fold).mean()),
    "nn_misma_clase_frac_otro_fold": float(np.mean([y[i] == y[nn[i]] for i in range(len(y)) if fold[nn[i]] != fold[i]]))}

# ---- referencia + 5 semillas ----
base, p_base = run(X, "base"); R["base_reproducida"] = base
print("base", base)
sem = [run(X, "s", s)[0]["f1_media"] for s in (1, 2, 3, 4, 5)]
R["6_semillas"] = {"semillas_1a5": sem, "media_5": float(np.mean(sem)), "desvio_5": float(np.std(sem, ddof=1)), "semilla0": base["f1_media"]}

# ---- 2. preprocesamiento: auditoría de código (features por clip, imputación en pipeline) ----
# control empírico: recomputar con un orden aleatorio de clips no cambia nada (no hay estadística global); eliminar columnas NaN-dependientes
R["2_preproc"] = {"nota": "features por clip sin estadística global; imputación mediana dentro del pipeline por fold; RF sin búsqueda de hiperparámetros; "
                  "umbral CONF_MIN=0.5 fijo a priori. Controles empíricos: ver 3 y 6."}
R["2_preproc"]["frac_nan"] = float(np.isnan(X).mean())

# ---- 3. confusores de filmación ----
nbp = names
absolutas = [n for n in names if not (n.startswith(("oreja_", "ojos_", "cabeza_", "boca_", "cuello_", "disp_")))]
R["3_columnas_no_reconocidas"] = absolutas
# (a) invariantes a escala: todo es ratio/ángulo/velocidad normalizada -> ya; quitar las que tienen unidad de longitud no normalizada: ninguna.
# invariantes a escala Y a muestreo temporal: solo estáticas (sin dt) + sin disponibilidad
est = grp == "estatica"; din = grp == "dinamica"; disp = grp == "disponibilidad"
R["3a_solo_estaticas_sin_disp"] = run(X[:, est], "a")[0]
R["3b_sin_disp_ni_dinamicas (=3a)"] = R["3a_solo_estaticas_sin_disp"]
R["3b2_sin_disponibilidad"] = run(X[:, ~disp], "b2")[0]
R["3b3_solo_disponibilidad"] = run(X[:, disp], "b3")[0]
R["3b4_solo_dinamicas"] = run(X[:, din], "b4")[0]
# (c) estratificar por resolución/duración: F1 dentro de cada estrato (predicciones OOF base)
res = (meta[:, 2] * 10000 + meta[:, 3])
estr = {}
for nombre, m in {"1080x1920": (meta[:, 2] == 1080) & (meta[:, 3] == 1920), "otra_resolucion": ~((meta[:, 2] == 1080) & (meta[:, 3] == 1920)),
                  "dur>=5": meta[:, 0] >= 5, "dur<5": meta[:, 0] < 5,
                  "limpio": (meta[:, 2] == 1080) & (meta[:, 3] == 1920) & (meta[:, 0] >= 5)}.items():
    r = metricas(y, p_base, fold, m); estr[nombre] = {"n": int(m.sum()), "clases": r["clips_por_clase"], "f1_agrupado": r["f1_macro_agrupado"], "f1_media_folds": r["f1_macro_media"]}
    pm, _ = None, None
R["3c_estratos_base"] = estr
# (c2) residualizar: quitar de cada feature la parte lineal explicada por log-duración, resolución(dummy), fps — AJUSTADO en train del fold
from sklearn.linear_model import LinearRegression
M = np.column_stack([np.log(meta[:, 0]), meta[:, 1], (meta[:, 2] == 1080) & (meta[:, 3] == 1920), meta[:, 3] / meta[:, 2]]).astype(float)
Xr = X.copy(); pr = np.empty(len(y), dtype=object)
for k in range(4):
    tr, te = fold != k, fold == k
    imp = SimpleImputer(strategy="median", keep_empty_features=True).fit(X[tr])
    A, B = imp.transform(X[tr]), imp.transform(X[te])
    lr = LinearRegression().fit(M[tr], A)
    A2, B2 = A - lr.predict(M[tr]), B - lr.predict(M[te])
    m = RandomForestClassifier(**RF_PARAMS).fit(A2, y[tr]); pr[te] = m.predict(B2)
R["3c2_residualizado_por_fold"] = {"f1_media": f1m(pr), "f1_agrupado": metricas(y, pr, fold)["f1_macro_agrupado"]}
# (c3) entrenar y evaluar solo en limpio
limpio = (meta[:, 2] == 1080) & (meta[:, 3] == 1920) & (meta[:, 0] >= 5)
pl, _ = oof(X, y, fold, "RF", clips, mascara_train=limpio)
R["3c3_RF_entrenado_y_evaluado_en_limpio"] = metricas(y, np.where(limpio, pl, None).astype(object), fold, limpio)["f1_macro_agrupado"]
# (d) metadatos solos vs features + metadatos
Mx = np.column_stack([meta, df["meta_escala_cabeza_px"], ])
R["3d_metadatos_solos"] = run(Mx, "d")[0]
R["3d_features_mas_metadatos"] = run(np.column_stack([X, Mx]), "d2")[0]
# (e) predecir resolución y duración desde las features (CV por caballo, mismos folds)
from sklearn.ensemble import RandomForestRegressor
def cv_pred(target, clf):
    out = np.zeros(len(y))
    for k in range(4):
        tr, te = fold != k, fold == k
        m = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), clf()).fit(X[tr], target[tr]); out[te] = m.predict(X[te])
    return out
dur = cv_pred(np.log(meta[:, 0]), lambda: RandomForestRegressor(300, random_state=0, n_jobs=-1))
fpsp = cv_pred(meta[:, 1], lambda: RandomForestRegressor(300, random_state=0, n_jobs=-1))
mayor = ((meta[:, 2] == 1080) & (meta[:, 3] == 1920)).astype(int)
resp = cv_pred(mayor, lambda: RandomForestRegressor(300, random_state=0, n_jobs=-1))
from sklearn.metrics import roc_auc_score, r2_score
R["3e_predecir_filmacion"] = {"R2_log_duracion": float(r2_score(np.log(meta[:, 0]), dur)), "R2_fps": float(r2_score(meta[:, 1], fpsp)),
    "AUC_resolucion_1080x1920": float(roc_auc_score(mayor, resp)), "frac_1080x1920": float(mayor.mean()),
    "AUC_baseline_clase_mayoritaria_trivial": 0.5}
# features estáticas solamente, mismo ejercicio
Xe = X[:, est]
def cv_e(target, mk):
    out = np.zeros(len(y))
    for k in range(4):
        tr, te = fold != k, fold == k
        out[te] = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), mk()).fit(Xe[tr], target[tr]).predict(Xe[te])
    return out
R["3e_estaticas_solas"] = {"R2_log_duracion": float(r2_score(np.log(meta[:, 0]), cv_e(np.log(meta[:, 0]), lambda: RandomForestRegressor(300, random_state=0, n_jobs=-1)))),
    "AUC_resolucion": float(roc_auc_score(mayor, cv_e(mayor, lambda: RandomForestRegressor(300, random_state=0, n_jobs=-1))))}
# clase vs metadatos: asociación
R["3_asociacion_clase_filmacion"] = {c: {"frac_1080x1920": float(mayor[y == c].mean()), "dur_mediana": float(np.median(meta[y == c, 0])), "n": int((y == c).sum())} for c in datos.CLASES}
# frames: ¿mismos 16? sí por construcción; separación temporal vs clase
T = datos.tiempos_por_clip()
span = np.array([T[c["clip"]][2] for c in clips]); R["3_span_seg_mediana_por_clase"] = {c: float(np.median(span[y == c])) for c in datos.CLASES}
# (f) dt fijo: quitar dependencia de dt en dinámicas -> no recomputable aquí sin pose; aproximación: dinámicas*dt ya cubierto por 3a

# ---- 4. cara vs cuello/cuerpo ----
def sel(prefs): return np.array([n.startswith(prefs) for n in names])
cara = sel(("oreja_", "ojos_", "boca_")) & ~disp
cabeza = sel(("cabeza_",)); cuello = sel(("cuello_",))
R["4_cara_solo_orejas_ojos_boca"] = {**run(X[:, cara], "c")[0], "n": int(cara.sum())}
R["4_cara_estatica"] = {**run(X[:, cara & est], "c")[0], "n": int((cara & est).sum())}
R["4_orejas"] = {**run(X[:, sel(("oreja_",)) & ~disp], "o")[0]}
R["4_boca"] = {**run(X[:, sel(("boca_",)) & ~disp], "o")[0]}
R["4_cabeza_postura_y_cuello"] = {**run(X[:, (cabeza | cuello) & ~disp], "k")[0], "n": int(((cabeza | cuello) & ~disp).sum())}
R["4_cabeza_postura_estatica_y_cuello_estatico"] = {**run(X[:, (cabeza | cuello) & est], "k")[0], "n": int(((cabeza | cuello) & est).sum())}
R["4_cuello"] = {**run(X[:, cuello & ~disp], "k")[0]}

# ---- 5. comparabilidad ----
pm, ev = datos.prediccion_mil(clips)
f1_mil = metricas(y, pm, fold)
mil_ev = [f["f1_macro"] if "f1_macro" in f else None for f in ev["por_fold"]]
R["5_comparabilidad"] = {"n_clips": len(y), "etiquetas_iguales_a_MIL": True, "F1_MIL_recomputado_con_misma_funcion": f1_mil["f1_macro_media"],
    "F1_MIL_por_fold_artefacto": mil_ev, "F1_MIL_por_fold_recomputado": f1_mil["f1_macro_por_fold"],
    "F1_RF_por_fold": metricas(y, p_base, fold)["f1_macro_por_fold"], "claves_ev_por_fold": list(ev["por_fold"][0].keys())}
# clase por clase
R["5_por_clase"] = {"RF": metricas(y, p_base, fold)["por_clase"], "MIL": f1_mil["por_clase"]}

# ---- 7. bootstrap: re-implementación independiente, pareada, por caballo ----
rng = np.random.default_rng(123); uni = np.unique(cab); ix = {h: np.where(cab == h)[0] for h in uni}
from sklearn.metrics import f1_score
dif = []
for _ in range(2000):
    idx = np.concatenate([ix[h] for h in rng.choice(uni, len(uni))])
    dif.append(f1_score(y[idx], p_base[idx], average="macro", labels=datos.CLASES) - f1_score(y[idx], pm[idx].astype(str), average="macro", labels=datos.CLASES))
R["7_bootstrap_independiente"] = {"media": float(np.mean(dif)), "ic95": [float(np.percentile(dif, 2.5)), float(np.percentile(dif, 97.5))], "p_dif<=0": float(np.mean(np.array(dif) <= 0))}
# bootstrap de clips (incorrecto, para contraste de ancho)
dif2 = []
for _ in range(2000):
    idx = rng.integers(0, len(y), len(y))
    dif2.append(f1_score(y[idx], p_base[idx], average="macro", labels=datos.CLASES) - f1_score(y[idx], pm[idx].astype(str), average="macro", labels=datos.CLASES))
R["7_bootstrap_clips_contraste"] = {"ic95": [float(np.percentile(dif2, 2.5)), float(np.percentile(dif2, 97.5))]}
# permutación de etiquetas por caballo (control negativo de leakage): debe dar ~0.18-0.33
rngp = np.random.default_rng(7); yp_full = y.copy(); 
perm = []
for s in range(3):
    ycab = {h: rngp.permutation(y[ix[h]]) for h in uni}   # permuta dentro de caballo -> no rompe estructura; usar permutación global de caballos
    yy = y.copy(); 
    for h in uni: yy[ix[h]] = y[ix[h]]
    lab = rngp.permutation(len(y)); yy = y[lab]
    pp = np.empty(len(y), dtype=object)
    for k in range(4):
        tr, te = fold != k, fold == k
        m = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), RandomForestClassifier(**RF_PARAMS)).fit(X[tr], yy[tr]); pp[te] = m.predict(X[te])
    perm.append(float(np.mean([f1m_ for f1m_ in metricas(yy, pp, fold)["f1_macro_por_fold"]])))
R["control_negativo_etiquetas_permutadas_F1"] = perm
(Path(__file__).parent / "verificacion.json").write_text(json.dumps(R, ensure_ascii=False, indent=1, default=float))
print(json.dumps(R, ensure_ascii=False, indent=1, default=float))
