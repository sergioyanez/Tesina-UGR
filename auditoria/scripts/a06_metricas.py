"""Métricas vigentes, estabilidad entre folds, matrices de confusión, baselines sobre los MISMOS
folds, IC95 bootstrap por caballo (recalculado de forma independiente), cifras históricas e
independencia de las muestras (ICC por caballo, tamaño efectivo)."""
import statistics
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from comun import *
from equinos import clasificacion as CL
from equinos import lineas_de_base as LB
from equinos import pose

clips = particion()
ev = evaluacion_vigente()
pred = pd.DataFrame(ev["predicciones_fuera_de_muestra"])
fold_de = {c["clip"]: c["fold"] for c in clips}
pred["fold"] = pred["clip"].map(fold_de)
assert len(pred) == 359 and pred["clip"].is_unique
etq = list(CLASES)

# ---- métrica: macro explícito, implementación propia vs sklearn ---------------------------------
por_fold = []
for k in range(4):
    d = pred[pred.fold == k]
    mi = CL.f1_macro(d.real.tolist(), d.predicha.tolist())
    sk = f1_score(d.real, d.predicha, average="macro", labels=etq, zero_division=0)
    assert abs(mi - sk) < 1e-12
    pr, rc, f1, sup = precision_recall_fscore_support(d.real, d.predicha, labels=etq, zero_division=0)
    por_fold.append({"fold": k, "n": len(d), "f1_macro": mi, "f1_macro_sklearn": sk,
                     "accuracy": accuracy_score(d.real, d.predicha),
                     "balanced_accuracy": float(np.mean(rc)),
                     "precision": dict(zip(etq, pr)), "recall": dict(zip(etq, rc)), "f1": dict(zip(etq, f1)),
                     "soporte": dict(zip(etq, map(int, sup))),
                     "matriz": confusion_matrix(d.real, d.predicha, labels=etq).tolist()})
f1s = [f["f1_macro"] for f in por_fold]
accs = [f["accuracy"] for f in por_fold]
est = {"f1_macro_media": statistics.fmean(f1s), "f1_macro_desvio_ddof1": statistics.stdev(f1s),
       "f1_macro_desvio_ddof0": float(np.std(f1s)), "min": min(f1s), "max": max(f1s),
       "rango": max(f1s) - min(f1s), "cv_pct": 100 * statistics.stdev(f1s) / statistics.fmean(f1s),
       "f1_por_fold": f1s, "accuracy_media": statistics.fmean(accs),
       "accuracy_desvio": statistics.stdev(accs), "accuracy_por_fold": accs,
       "n_folds": 4, "nota": "con 4 folds el desvío es una estimación muy gruesa (3 grados de libertad)"}
# coincidencia con el JSON vigente
assert abs(est["f1_macro_media"] - ev["resumen"]["f1_macro"]) < 1e-12
assert abs(est["f1_macro_desvio_ddof1"] - ev["resumen"]["f1_macro_desvio"]) < 1e-12

# global
cm = confusion_matrix(pred.real, pred.predicha, labels=etq)
pr, rc, f1, sup = precision_recall_fscore_support(pred.real, pred.predicha, labels=etq, zero_division=0)
glob = {"matriz_filas_real_columnas_predicha": cm.tolist(), "clases": etq,
        "accuracy": accuracy_score(pred.real, pred.predicha),
        "f1_macro_agrupado_sklearn": f1_score(pred.real, pred.predicha, average="macro", labels=etq),
        "f1_micro_agrupado(=accuracy)": f1_score(pred.real, pred.predicha, average="micro", labels=etq),
        "f1_ponderado_agrupado": f1_score(pred.real, pred.predicha, average="weighted", labels=etq),
        "balanced_accuracy": float(np.mean(rc)),
        "por_clase": {k: {"precision": float(pr[i]), "recall": float(rc[i]), "f1": float(f1[i]), "soporte": int(sup[i])}
                      for i, k in enumerate(etq)},
        "errores_totales": int((pred.real != pred.predicha).sum())}
glob["error_dominante"] = {"Sereno→Disgustado": int(cm[2][0]), "Disgustado→Sereno": int(cm[0][2]),
                           "Interesado→Sereno": int(cm[1][2]), "Sereno→Interesado": int(cm[2][1])}

# ---- baselines sobre los mismos folds ----------------------------------------------------------
y = np.array([c["clase"] for c in clips]); fold = np.array([c["fold"] for c in clips])
caballos = np.array([c["id_caballo"] for c in clips])
rng = np.random.default_rng(0)
def f1_por_fold_de(predfun):
    return [CL.f1_macro(y[fold == k].tolist(), predfun(k).tolist()) for k in range(4)]
may = LB.predecir_mayoritaria(y, fold)
f_may = [CL.f1_macro(y[fold == k].tolist(), may[fold == k].tolist()) for k in range(4)]
sims_u, sims_e, acc_u = [], [], []
for _ in range(4000):
    fu, fe = [], []
    for k in range(4):
        n = int((fold == k).sum())
        fu.append(CL.f1_macro(y[fold == k].tolist(), rng.choice(etq, n).tolist()))
        cl_, cn = np.unique(y[fold != k], return_counts=True)
        fe.append(CL.f1_macro(y[fold == k].tolist(), rng.choice(cl_, n, p=cn / cn.sum()).tolist()))
    sims_u.append(statistics.fmean(fu)); sims_e.append(statistics.fmean(fe))
baselines = {
    "mayoritaria_del_train_del_fold": {"f1_macro": statistics.fmean(f_may), "desvio": statistics.stdev(f_may),
                                       "por_fold": f_may, "accuracy": float((may == y).mean()),
                                       "clase_predicha_por_fold": [Counter(y[fold != k].tolist()).most_common(1)[0][0] for k in range(4)]},
    "aleatoria_uniforme_1/3": {"f1_macro_esperado": float(np.mean(sims_u)), "desvio_sim": float(np.std(sims_u)),
                               "p95": float(np.percentile(sims_u, 95)), "simulaciones": 4000},
    "aleatoria_estratificada": {"f1_macro_esperado": float(np.mean(sims_e)), "desvio_sim": float(np.std(sims_e)),
                                "p95": float(np.percentile(sims_e, 95)), "simulaciones": 4000},
    "accuracy_esperada_azar_uniforme": 1 / 3,
}
guard = lineas_de_base()["lineas_de_base"]
baselines["del_repo(objetivo3_lineas_de_base.json)"] = {k: {"f1_macro": v["f1_macro"], "f1_desvio": v["f1_desvio"],
                                                            "accuracy": v.get("accuracy")} for k, v in guard.items()}

# ---- bootstrap por caballo recalculado de forma independiente ----------------------------------
orden, X = caracteristicas_v2()
indice = {c: i for i, c in enumerate(orden)}
pose_filas = pose.leer_pose(RES / "pose_seleccionada_v2.csv")
E = LB.entradas(clips, X, indice, pose_filas)
p_est = LB.predecir_lineal(E["visual_media"], y, fold)
p_1f = LB.predecir_lineal(E["visual_un_frame"], y, fold)
p_meta = LB.predecir_lineal(E["metadatos"], y, fold)
p_geo = LB.predecir_lineal(E["geometria"], y, fold)
mapa = dict(zip(pred["clip"], pred.predicha))
p_mil = np.array([mapa[c["clip"]] for c in clips], dtype=object)
NM = "MIL multimodal (vigente)"
bs = LB.bootstrap_por_caballo(y, {NM: p_mil, "Estática: media 16 frames (LogReg)": p_est,
                                  "Un frame central (LogReg)": p_1f, "Metadatos de filmación (LogReg)": p_meta,
                                  "Geometría sola (LogReg)": p_geo}, caballos, referencia=NM)
rep_json = lineas_de_base()["bootstrap_por_caballo"]["modelos"]["Visual estática: media de 16 frames (LogReg)"]
bs["coincide_con_el_json_del_repo"] = abs(
    bs["modelos"]["Estática: media 16 frames (LogReg)"]["diferencia_referencia_menos_este"]["media"]
    - rep_json["diferencia_referencia_menos_este"]["media"]) < 1e-12
f1_est_fold = [CL.f1_macro(y[fold == k].tolist(), p_est[fold == k].tolist()) for k in range(4)]
bs["por_fold_mil_menos_estatica"] = [a - b for a, b in zip(f1s, f1_est_fold)]
bs["f1_estatica_por_fold"] = f1_est_fold

# ---- independencia: ICC por caballo, DEFF, tamaño efectivo --------------------------------------
ok = (pred.real == pred.predicha).astype(float).values
pred["ok"] = ok; pred["id_caballo"] = pred["clip"].map({c["clip"]: c["id_caballo"] for c in clips})
g = pred.groupby("id_caballo")["ok"]
n_g = g.size().values; mean_g = g.mean().values; N = len(pred); k_g = len(n_g)
grand = ok.mean()
ssb = (n_g * (mean_g - grand) ** 2).sum(); ssw = ((ok - pred.id_caballo.map(g.mean()).values) ** 2).sum()
msb = ssb / (k_g - 1); msw = ssw / (N - k_g)
n0 = (N - (n_g ** 2).sum() / N) / (k_g - 1)
icc = (msb - msw) / (msb + (n0 - 1) * msw)
mbar = N / k_g
deff = 1 + (mbar - 1) * max(icc, 0)
vid = pd.read_csv(SALIDA / "clips_estado.csv", dtype={"id_caballo": str}).set_index("clip")["video_origen"]
pred["video"] = pred["clip"].map(vid)
gv = pred.groupby("video")["ok"]; nv = gv.size().values
indep = {"clips": N, "caballos": k_g, "videos": int(pred.video.nunique()), "clips_por_caballo_medio": mbar,
         "ICC_acierto_por_caballo": float(icc), "efecto_de_diseno": float(deff),
         "n_efectivo_por_caballo_(clips)": float(N / deff),
         "n_efectivo_minimo_(caballos)": k_g, "nota": ("el ICC se estima con ANOVA de una vía sobre el indicador de acierto; "
                                                      "con muchos caballos de 1 clip es ruidoso. El tamaño efectivo real está entre "
                                                      f"{N/deff:.0f} y {k_g}"),
         "clips_por_video_medio": float(N / pred.video.nunique()),
         "pct_clips_en_videos_con_mas_de_un_clip": float(nv[nv > 1].sum() / N * 100),
         "errores_concentrados": None}
err_h = pred.assign(err=1 - pred.ok).groupby("id_caballo").agg(n=("ok", "size"), errores=("err", "sum")).sort_values("errores", ascending=False)
indep["errores_concentrados"] = {"caballos_con_algun_error": int((err_h.errores > 0).sum()),
                                 "top10_acumulan_pct_errores": float(100 * err_h.errores.head(10).sum() / err_h.errores.sum()),
                                 "top10": err_h.head(10).reset_index().to_dict("records"),
                                 "caballos_con_>=2_clips_y_100%_error": int(((err_h.n >= 2) & (err_h.errores == err_h.n)).sum())}
# intervalo de Wilson simple (ingenuo, por clip) vs bootstrap por caballo para la accuracy
from math import sqrt
p0 = ok.mean(); z = 1.96
den = 1 + z * z / N
cen = (p0 + z * z / (2 * N)) / den
half = z * sqrt(p0 * (1 - p0) / N + z * z / (4 * N * N)) / den
indep["accuracy_IC95_ingenuo_por_clip_Wilson"] = [cen - half, cen + half]
indep["accuracy_IC95_bootstrap_por_caballo"] = bs["modelos"][NM]["accuracy_ic95"]
indep["f1_IC95_bootstrap_por_caballo"] = bs["modelos"][NM]["f1_ic95"]
guardar_json({"indep": indep}, "independencia.json")

# ---- cifras históricas -------------------------------------------------------------------------
hist = [
    {"cifra": "0,626 ± 0,065", "modelo": "Un frame + voto (LogReg)", "dataset": "141 clips / 73 caballos", "fecha": "2026-08-28",
     "commit": "6f50a59 (línea de base) / 425c592 (evaluación final)", "protocolo": "GroupKFold por caballo, sin validación interna",
     "artefacto_local": "no (objetivo3_evaluacion.json lo sobrescribió; se cita en project.md)", "vigente": "No"},
    {"cifra": "0,704 ± 0,044", "modelo": "Cabeza «promedio» (MLP sobre media de frames)", "dataset": "153 clips / 83 caballos", "fecha": "2026-09-15",
     "commit": "archivo objetivo3_evaluacion.json (mtime 15/09)", "protocolo": "GroupKFold, 50 épocas fijas",
     "artefacto_local": "sí: objetivo3_evaluacion.json", "vigente": "No"},
    {"cifra": "0,754", "modelo": "MIL multimodal (otra corrida)", "dataset": "257 clips / 113 caballos (según docs)", "fecha": "2026-09-15",
     "commit": "6b66044 (primera aparición, escrito a mano en clasificacion.py); retirado en 5e41c63 (18/09)", "protocolo": "50 épocas fijas",
     "artefacto_local": "NO: sin artefacto que lo respalde", "vigente": "No (no citar)"},
    {"cifra": "0,618", "modelo": "MIL con validación interna", "dataset": "153 clips / 83 caballos", "fecha": "2026-09-18",
     "commit": "docs/informe_mejora_pipeline_mil.md §6", "protocolo": "validación interna por caballo + early stopping",
     "artefacto_local": "no (cifra en el informe)", "vigente": "No"},
    {"cifra": "0,740 ± 0,052", "modelo": "MIL multimodal (MILFusionModel)", "dataset": "359 clips / 133 caballos", "fecha": "2026-09-18 (re-ejecutado 25/09)",
     "commit": "4e51800 (protocolo), cf23ded (selección de frames); archivo objetivo3_evaluacion_v2.json", "protocolo": "4 folds por caballo + validación interna + early stopping",
     "artefacto_local": "sí: objetivo3_evaluacion_v2.json", "vigente": "SÍ (única cifra vigente)"},
    {"cifra": "0,741 ± 0,020", "modelo": "Estática: media de 16 frames + LogReg", "dataset": "359 / 133", "fecha": "2026-09-25",
     "commit": "e75f13a", "protocolo": "mismos folds", "artefacto_local": "sí: objetivo3_lineas_de_base.json", "vigente": "Línea de base vigente"},
    {"cifra": "0,618 ± 0,034", "modelo": "Solo metadatos de filmación (LogReg)", "dataset": "359 / 133", "fecha": "2026-09-25",
     "commit": "e75f13a", "protocolo": "mismos folds", "artefacto_local": "sí: objetivo3_lineas_de_base.json", "vigente": "Línea de base vigente"},
    {"cifra": "0,686", "modelo": "MIL visual puro (sin geometría)", "dataset": "359 / 133", "fecha": "2026-09-18",
     "commit": "docs/informe_mejora_pipeline_mil.md §7", "protocolo": "vigente", "artefacto_local": "no (cifra en el informe)", "vigente": "No (variante)"},
    {"cifra": "0,747 ± 0,057", "modelo": "MIL con fine-tuning del backbone", "dataset": "359 / 133", "fecha": "2026-09-22",
     "commit": "rama feat/finetuning-visual (c6d6f80, 544d323)", "protocolo": "vigente", "artefacto_local": "sí: objetivo3_finetuning_evaluacion.json", "vigente": "No (no adoptado)"},
    {"cifra": "0,698 ± 0,103", "modelo": "MIL con canal geométrico enriquecido", "dataset": "359 / 133", "fecha": "2026-09-22",
     "commit": "rama feat/enriquecer-canal-geometrico", "protocolo": "vigente", "artefacto_local": "sí", "vigente": "No (no adoptado)"},
    {"cifra": "0,755 ± 0,022", "modelo": "MIL reentrenado sobre 100 % del train_val", "dataset": "359 / 133", "fecha": "2026-09-23",
     "commit": "rama feat/validacion-cruzada-anidada-NestedCV (23d4b25)", "protocolo": "reentrenado", "artefacto_local": "sí", "vigente": "No (no adoptado)"},
    {"cifra": "0,784 ± 0,036", "modelo": "MIL con CV anidada (época promedio)", "dataset": "359 / 133", "fecha": "2026-09-23",
     "commit": "rama feat/validacion-cruzada-anidada-NestedCV", "protocolo": "anidado k=4", "artefacto_local": "sí", "vigente": "No (no adoptado; optimista)"},
]
# verificación de las cifras con artefacto
ver = {}
for nombre, valor in (("objetivo3_canal_enriquecido_evaluacion.json", None), ("objetivo3_finetuning_evaluacion.json", None),
                      ("objetivo3_nested_reentrenado_baseline_evaluacion.json", None),
                      ("objetivo3_nested_anidado_baseline_evaluacion.json", None)):
    d = leer_json(RES / nombre)["resumen"]
    ver[nombre] = [d["f1_macro"], d["f1_macro_desvio"]]
v1 = leer_json(RES / "objetivo3_evaluacion.json")["completo"]["f1_macro"]
ver["objetivo3_evaluacion.json"] = v1
guardar_json({"por_fold": por_fold, "estabilidad": est, "global": glob, "baselines": baselines,
              "bootstrap": bs, "historico": hist, "verificacion_historicos": ver}, "metricas.json")
guardar_csv([{k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v) for k, v in h.items()} for h in hist], "cifras_historicas.csv")

# ---- figuras --------------------------------------------------------------------------------------
plt = estilo()
fig, ax = plt.subplots(1, 4, figsize=(14, 3.7), gridspec_kw={"width_ratios": [1.15, 1, 1, 1]})
def heat(a, m, titulo, norm=True):
    m = np.array(m); mm = m / m.sum(axis=1, keepdims=True) if norm else m
    a.imshow(mm, cmap="Blues", vmin=0, vmax=1); a.set_title(titulo, fontsize=9)
    a.set_xticks(range(3)); a.set_yticks(range(3))
    a.set_xticklabels([c[:4] + "." for c in etq], fontsize=8); a.set_yticklabels(etq, fontsize=8)
    a.grid(False)
    for i in range(3):
        for j in range(3):
            a.text(j, i, f"{m[i][j]}\n({100*mm[i][j]:.0f}%)", ha="center", va="center", fontsize=8,
                   color="white" if mm[i][j] > .5 else "black")
    a.set_xlabel("Predicha", fontsize=8); a.set_ylabel("Real", fontsize=8)
heat(ax[0], cm, "Global (359 clips)")
fig.axes[0].set_title("Global (359 clips, fuera de muestra)", fontsize=9)
for k in (0, 1, 2):
    heat(ax[k + 1], por_fold[k]["matriz"], f"Fold {k} (n={por_fold[k]['n']}, F1 {por_fold[k]['f1_macro']:.3f})")
fig.tight_layout(); guardar_fig(fig, "fig_confusion_global_y_folds.png"); plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(11.5, 4))
heat(ax[0], por_fold[3]["matriz"], f"Fold 3 (n={por_fold[3]['n']}, F1 {por_fold[3]['f1_macro']:.3f})")
nombres = ["Mayoritaria", "Azar uniforme", "Azar estrat.", "Solo duración", "Metadatos", "Geometría sola",
           "1 frame central", "Media 16 frames", "MIL vigente"]
vals = [baselines["mayoritaria_del_train_del_fold"]["f1_macro"], baselines["aleatoria_uniforme_1/3"]["f1_macro_esperado"],
        baselines["aleatoria_estratificada"]["f1_macro_esperado"], guard["Solo duración del clip (LogReg)"]["f1_macro"],
        guard["Metadatos de filmación (LogReg)"]["f1_macro"], guard["Geometría sola: media y desvío (LogReg)"]["f1_macro"],
        guard["Visual: 1 frame central (LogReg)"]["f1_macro"], guard["Visual estática: media de 16 frames (LogReg)"]["f1_macro"],
        ev["resumen"]["f1_macro"]]
dev = [baselines["mayoritaria_del_train_del_fold"]["desvio"], baselines["aleatoria_uniforme_1/3"]["desvio_sim"],
       baselines["aleatoria_estratificada"]["desvio_sim"], guard["Solo duración del clip (LogReg)"]["f1_desvio"],
       guard["Metadatos de filmación (LogReg)"]["f1_desvio"], guard["Geometría sola: media y desvío (LogReg)"]["f1_desvio"],
       guard["Visual: 1 frame central (LogReg)"]["f1_desvio"], guard["Visual estática: media de 16 frames (LogReg)"]["f1_desvio"],
       ev["resumen"]["f1_macro_desvio"]]
cols = ["#95a5a6"] * 3 + ["#e0a800"] * 2 + ["#8e6bb0"] + ["#2a7f9e"] * 2 + ["#1b3f5a"]
ax[1].barh(range(len(vals)), vals, xerr=dev, color=cols, capsize=3)
for i, (v_, d_) in enumerate(zip(vals, dev)):
    ax[1].text(v_ + d_ + .01, i, f"{v_:.3f}".replace(".", ","), va="center", fontsize=8)
ax[1].axvline(1 / 3, color="k", ls=":", lw=1); ax[1].text(1 / 3 + .005, -.9, "1/3", fontsize=7)
ax[1].set_yticks(range(len(vals))); ax[1].set_yticklabels(nombres, fontsize=8); ax[1].set_xlim(0, .95)
ax[1].set_xlabel("F1 macro (media ± desvío entre 4 folds)"); ax[1].set_title("Modelo vigente contra líneas de base (mismos folds)")
fig.tight_layout(); guardar_fig(fig, "fig_baselines.png"); plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
x = np.arange(4)
ax[0].bar(x - .2, f1s, .4, color="#1b3f5a", label="MIL vigente")
ax[0].bar(x + .2, f1_est_fold, .4, color="#2a7f9e", label="Media 16 frames (LogReg)")
ax[0].axhline(est["f1_macro_media"], color="#1b3f5a", ls="--", lw=1)
ax[0].set_ylim(.5, .85); ax[0].set_xticks(x); ax[0].set_xticklabels([f"Fold {i}" for i in range(4)])
ax[0].set_ylabel("F1 macro"); ax[0].legend(fontsize=8); ax[0].set_title("Estabilidad entre folds")
for i, v_ in enumerate(f1s):
    ax[0].text(i - .2, v_ + .005, f"{v_:.3f}".replace(".", ","), ha="center", fontsize=8)
for i, v_ in enumerate(f1_est_fold):
    ax[0].text(i + .2, v_ + .005, f"{v_:.3f}".replace(".", ","), ha="center", fontsize=8)
modelos_bs = [k for k in bs["modelos"] if k != NM]
dif = [bs["modelos"][k]["diferencia_referencia_menos_este"] for k in modelos_bs]
ax[1].errorbar([d["media"] for d in dif], range(len(dif)),
               xerr=[[d["media"] - d["ic95"][0] for d in dif], [d["ic95"][1] - d["media"] for d in dif]],
               fmt="o", color="#1b3f5a", capsize=4)
ax[1].axvline(0, color="k", lw=1)
ax[1].set_yticks(range(len(dif))); ax[1].set_yticklabels([k.replace(" (LogReg)", "") for k in modelos_bs], fontsize=8)
ax[1].set_xlabel("Diferencia de F1 macro: MIL − baseline (IC95 bootstrap por caballo)")
ax[1].set_title("¿El MIL supera a cada baseline?")
fig.tight_layout(); guardar_fig(fig, "fig_estabilidad_y_bootstrap.png"); plt.close(fig)

print(json.dumps({"estabilidad": est, "global": glob, "baselines": {k: v for k, v in baselines.items() if k != "del_repo(objetivo3_lineas_de_base.json)"},
                  "bootstrap": bs, "indep": indep, "ver": ver}, ensure_ascii=False, indent=1, default=str)[:9000])
