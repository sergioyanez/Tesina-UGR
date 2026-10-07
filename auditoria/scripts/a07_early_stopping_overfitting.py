"""Early stopping, sobreajuste y prueba empírica de que el test no interviene en el entrenamiento.

Los historiales train/val por época SÍ existen en objetivo3_evaluacion_v2.json (por_fold[*].curva),
así que las curvas salen de ahí; además se re-entrena el fold con el TEST ALTERADO (etiquetas
permutadas) y se verifica que la mejor época, las curvas y los pesos no cambian: el test no
participa de la selección.
"""
import statistics
import tempfile
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from comun import *
from equinos import clasificacion as CL
from equinos import lineas_de_base as LB
from equinos import pose
import inspect

ev = evaluacion_vigente()
clips = particion()
tabla = []; gaps = []
for f in ev["por_fold"]:
    cu = f["curva"]; be = f["mejor_epoca"]
    e_best = next(e for e in cu if e["epoca"] == be)
    ult = cu[-1]
    min_loss = min(cu, key=lambda e: e["loss_val"])
    f1v = [e["f1_val"] for e in cu]
    tabla.append({
        "fold": f["fold"], "mejor_epoca": be, "epocas_ejecutadas": f["epocas_corridas"],
        "epocas_despues_del_mejor": f["epocas_corridas"] - be,
        "f1_validacion": f["f1_validacion"], "f1_train": f["f1_train"], "f1_test": f["f1_macro"],
        "acc_val_mejor": e_best["acc_val"], "acc_train_mejor": e_best["acc_train"],
        "loss_val_mejor": e_best["loss_val"], "loss_val_minima": min_loss["loss_val"],
        "epoca_loss_val_minima": min_loss["epoca"], "loss_val_final": ult["loss_val"],
        "f1_val_final": ult["f1_val"], "caida_f1_val_hasta_el_final": ult["f1_val"] - e_best["f1_val"],
        "f1_val_media_ventana_mejor_pm3": float(np.mean([e["f1_val"] for e in cu if abs(e["epoca"] - be) <= 3])),
        "f1_val_maximo_se_alcanza_una_vez": sum(1 for v in f1v if v == max(f1v)) == 1,
        "gap_train_val_f1": f["f1_train"] - f["f1_validacion"], "gap_train_test_f1": f["f1_train"] - f["f1_macro"],
        "gap_val_test_f1": f["f1_validacion"] - f["f1_macro"],
        "gap_train_val_acc": e_best["acc_train"] - e_best["acc_val"],
        "clips_train_interno": f["clips_train_interno"], "clips_validacion": f["clips_validacion"],
        "caballos_validacion": f["caballos_validacion"], "clips_test": f["n"]})
conf = {"metrica_de_seleccion": "F1 macro de la validación interna (ds_val), implementada en clasificacion.f1_macro",
        "paciencia": 10, "max_epocas": 100, "criterio_de_mejora": "estricto (f1_val > mejor); el empate no reemplaza",
        "checkpoint": "copia del state_dict de la mejor época en memoria (CPU) y se restaura antes de predecir el test",
        "test": "se predice UNA vez, después de cargar el mejor estado",
        "valores_por_defecto_usados_por_evaluar_v2": "max_epocas=100 y paciencia=10 (no se pasan explícitos)",
        "validacion_interna": "25 % de los clips del train_val, por caballo (7 caballos por fold), determinista",
        "optimizador": "Adam lr=1e-3 weight_decay=1e-4, batch 32, CrossEntropy con pesos de clase 'raiz'",
        "scheduler": "ninguno", "ejecucion_total_alcanzo_max_epocas": any(t["epocas_ejecutadas"] >= 100 for t in tabla)}
src = inspect.getsource(CL.entrenar_mil_multimodal)
assert "predecir(dl_pru)" in src and src.index("predecir(dl_pru)") > src.index("load_state_dict(mejor")

res = {"tabla": tabla, "configuracion": conf,
       "medias": {k: statistics.fmean(t[k] for t in tabla) for k in
                  ("mejor_epoca", "epocas_ejecutadas", "f1_validacion", "f1_train", "f1_test", "gap_train_val_f1",
                   "gap_train_test_f1", "gap_val_test_f1", "caida_f1_val_hasta_el_final", "gap_train_val_acc")},
       "desvio_mejor_epoca": statistics.stdev(t["mejor_epoca"] for t in tabla),
       "test_mayor_o_igual_que_val_en_folds": sum(t["f1_test"] >= t["f1_validacion"] for t in tabla)}

# ---- baseline lineal: ¿cuánto memoriza? (gap train-test de la regresión logística estática) ------
orden, X = caracteristicas_v2(); indice = {c: i for i, c in enumerate(orden)}
pose_filas = pose.leer_pose(RES / "pose_seleccionada_v2.csv")
y = np.array([c["clase"] for c in clips]); fold = np.array([c["fold"] for c in clips])
E = LB.entradas(clips, X, indice, pose_filas)
tr_te = []
for k in range(4):
    m = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000, class_weight="balanced"))
    m.fit(E["visual_media"][fold != k], y[fold != k])
    ftr = CL.f1_macro(y[fold != k].tolist(), m.predict(E["visual_media"][fold != k]).tolist())
    fte = CL.f1_macro(y[fold == k].tolist(), m.predict(E["visual_media"][fold == k]).tolist())
    tr_te.append({"fold": k, "f1_train": ftr, "f1_test": fte, "gap": ftr - fte})
res["baseline_estatico_gap"] = {"por_fold": tr_te, "gap_medio": statistics.fmean(t["gap"] for t in tr_te),
                                "f1_train_medio": statistics.fmean(t["f1_train"] for t in tr_te)}

# ---- prueba empírica: el test no interviene en el entrenamiento ---------------------------------
dev_ok = []
tmp = Path(tempfile.mkdtemp(prefix="aud_es_"))
folds = CL.folds_por_ejemplar(clips)
rng = np.random.default_rng(123)
for k, (trval, test) in enumerate(folds):
    alterado = [dict(c) for c in test]
    cl = [c["clase"] for c in alterado]; perm = rng.permutation(len(cl))
    for c, j in zip(alterado, perm):
        c["clase"] = cl[j]                       # etiquetas del test barajadas
    pred, extra = CL.entrenar_mil_multimodal(trval, alterado, X, indice, pose_filas, guardar_en=tmp / f"alt{k}",
                                             dropout=0.3, estrategia_pesos="raiz")
    a = torch.load(tmp / f"alt{k}.pt", map_location="cpu", weights_only=True)
    b = torch.load(DATOS_REPO / "modelos/objetivo3_v2" / f"mil_v2_fold{k}.pt", map_location="cpu", weights_only=True)
    dev_ok.append({"fold": k, "mejor_epoca_igual": extra["mejor_epoca"] == ev["por_fold"][k]["mejor_epoca"],
                   "curva_val_igual": [e["f1_val"] for e in extra["curva"]] == [e["f1_val"] for e in ev["por_fold"][k]["curva"]],
                   "pesos_identicos": all(torch.equal(a[n], b[n]) for n in a)})
res["test_alterado"] = {"descripcion": "re-entrenamiento con las etiquetas del test permutadas al azar", "folds": dev_ok,
                        "el_test_no_interviene": all(d["mejor_epoca_igual"] and d["curva_val_igual"] and d["pesos_identicos"] for d in dev_ok)}
guardar_json(res, "early_stopping_sobreajuste.json")
guardar_csv(tabla, "early_stopping_tabla.csv")

# ---- figuras ---------------------------------------------------------------------------------------
plt = estilo()
fig, axs = plt.subplots(3, 4, figsize=(14, 8.5), sharex="col")
for k, f in enumerate(ev["por_fold"]):
    cu = f["curva"]; ep = [e["epoca"] for e in cu]; be = f["mejor_epoca"]
    a0, a1, a2 = axs[0][k], axs[1][k], axs[2][k]
    a0.plot(ep, [e["loss_train_eval"] for e in cu], color="#2a7f9e", label="Train (sin dropout)")
    a0.plot(ep, [e["loss_val"] for e in cu], color="#c0392b", label="Validación")
    a1.plot(ep, [e["f1_train"] for e in cu], color="#2a7f9e", label="F1 train")
    a1.plot(ep, [e["f1_val"] for e in cu], color="#c0392b", label="F1 validación")
    a1.axhline(f["f1_macro"], color="k", ls="--", lw=1, label="F1 test (mejor época)")
    a2.plot(ep, [e["acc_train"] for e in cu], color="#2a7f9e", label="Acc. train")
    a2.plot(ep, [e["acc_val"] for e in cu], color="#c0392b", label="Acc. validación")
    for a in (a0, a1, a2):
        a.axvline(be, color="#e0a800", lw=1.5)
    a0.set_title(f"Fold {k}: mejor época {be} de {f['epocas_corridas']}", fontsize=10)
    a2.set_xlabel("Época")
    if k == 0:
        a0.set_ylabel("Pérdida (CE ponderada)"); a1.set_ylabel("F1 macro"); a2.set_ylabel("Accuracy")
        a0.legend(fontsize=7); a1.legend(fontsize=7, loc="center right"); a2.legend(fontsize=7, loc="center right")
fig.suptitle("Curvas de entrenamiento por fold (línea ámbar = época elegida con la validación interna)", y=1.0)
fig.tight_layout(); guardar_fig(fig, "fig_curvas_por_fold.png"); plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
x = np.arange(4); w = .26
ax[0].bar(x - w, [t["f1_train"] for t in tabla], w, color="#2a7f9e", label="Train interno")
ax[0].bar(x, [t["f1_validacion"] for t in tabla], w, color="#c0392b", label="Validación interna")
ax[0].bar(x + w, [t["f1_test"] for t in tabla], w, color="#1b3f5a", label="Test")
ax[0].set_xticks(x); ax[0].set_xticklabels([f"Fold {i}" for i in range(4)]); ax[0].set_ylim(.5, 1.02)
ax[0].set_ylabel("F1 macro"); ax[0].legend(fontsize=8); ax[0].set_title("F1 en la época elegida")
for i, t in enumerate(tabla):
    ax[0].text(i - w, t["f1_train"] + .005, f"{t['f1_train']:.2f}".replace(".", ","), ha="center", fontsize=7)
    ax[0].text(i, t["f1_validacion"] + .005, f"{t['f1_validacion']:.2f}".replace(".", ","), ha="center", fontsize=7)
    ax[0].text(i + w, t["f1_test"] + .005, f"{t['f1_test']:.2f}".replace(".", ","), ha="center", fontsize=7)
ax[1].bar(x - .2, [t["gap_train_val_f1"] for t in tabla], .4, color="#8e6bb0", label="Gap train − validación")
ax[1].bar(x + .2, [t["gap_train_test_f1"] for t in tabla], .4, color="#e0a800", label="Gap train − test")
ax[1].set_xticks(x); ax[1].set_xticklabels([f"Fold {i}" for i in range(4)]); ax[1].set_ylabel("Diferencia de F1 macro")
ax[1].legend(fontsize=8); ax[1].set_title("Brechas de generalización")
fig.tight_layout(); guardar_fig(fig, "fig_gaps.png"); plt.close(fig)
print(json.dumps({k: v for k, v in res.items() if k != "tabla"}, ensure_ascii=False, indent=1))
for t in tabla:
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in t.items()})
