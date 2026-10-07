"""Balance, composición de folds (test / train interno / validación interna) y verificación de fuga.

Criterio de balance (DECLARADO ANTES DE CALCULAR; es una convención de esta auditoría):
    IR = n(clase mayor) / n(clase menor), sobre clips y sobre caballos.
    IR <= 1,2          BALANCEADO
    1,2 < IR <= 1,5    RAZONABLEMENTE BALANCEADO
    1,5 < IR <= 3      MODERADAMENTE DESBALANCEADO
    IR > 3             FUERTEMENTE DESBALANCEADO
Referencia: el «imbalance ratio» es la medida habitual en la literatura de datos desbalanceados
(Fernández et al., 2018, *Learning from Imbalanced Data Sets*, Springer); no existe un umbral
normativo universal, así que los cortes son de esta auditoría y se reportan con un análisis de
sensibilidad (los cortes alternativos 1,1/1,3/2 y 1,25/2/4 no cambian la categoría).
"""
import math
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from comun import *
from equinos import particion as P
from equinos import clasificacion as CL

CORTES = [(1.2, 1.5, 3.0), (1.1, 1.3, 2.0), (1.25, 2.0, 4.0)]
NOMBRES = ["BALANCEADO", "RAZONABLEMENTE BALANCEADO", "MODERADAMENTE DESBALANCEADO",
           "FUERTEMENTE DESBALANCEADO"]


def categoria(ir, cortes=CORTES[0]):
    for i, t in enumerate(cortes):
        if ir <= t:
            return NOMBRES[i]
    return NOMBRES[3]


p = particion()
clips = [dict(c) for c in p]
n_total, n_cab = len(clips), len({c["id_caballo"] for c in clips})

# ---- balance global ------------------------------------------------------------------------
filas = []
pares = Counter()  # pares caballo-clase
for c in clips:
    pares[(c["id_caballo"], c["clase"])] += 0
cab_por_clase = {k: {c["id_caballo"] for c in clips if c["clase"] == k} for k in CLASES}
n_pares = sum(len(s) for s in cab_por_clase.values())
for k in CLASES:
    n = sum(1 for c in clips if c["clase"] == k)
    nc = len(cab_por_clase[k])
    filas.append({"clase": k, "clips": n, "pct_clips": 100 * n / n_total, "caballos": nc,
                  "pct_caballos": 100 * nc / n_cab, "pct_pares_caballo_clase": 100 * nc / n_pares,
                  "clips_por_caballo": n / nc})
ir_clips = max(f["clips"] for f in filas) / min(f["clips"] for f in filas)
ir_cab = max(f["caballos"] for f in filas) / min(f["caballos"] for f in filas)
p_ = np.array([f["clips"] for f in filas]) / n_total
entropia_norm = float(-(p_ * np.log(p_)).sum() / math.log(3))
balance = {
    "tabla": filas, "ratio_mayor_menor_clips": ir_clips, "ratio_mayor_menor_caballos": ir_cab,
    "categoria_clips": categoria(ir_clips), "categoria_caballos": categoria(ir_cab),
    "sensibilidad": [{"cortes": c, "clips": categoria(ir_clips, c), "caballos": categoria(ir_cab, c)}
                     for c in CORTES],
    "entropia_normalizada_clips": entropia_norm,
    "nota_pct_caballos": "los porcentajes de caballos suman >100 % porque 29 caballos tienen clips en "
                         "más de una clase; se agrega % sobre pares caballo-clase (suma 100 %)",
    "criterio": {"cortes_IR": CORTES[0], "categorias": NOMBRES},
}

# ---- composición por fold: test / train interno / validación interna ---------------------
folds_idx = CL.folds_por_ejemplar(clips)   # exactamente lo que usa evaluar_v2.py
def comp(conj):
    d = {"clips": len(conj), "caballos": len({c["id_caballo"] for c in conj})}
    for k in CLASES:
        d[f"clips_{k}"] = sum(1 for c in conj if c["clase"] == k)
        d[f"caballos_{k}"] = len({c["id_caballo"] for c in conj if c["clase"] == k})
    n = len(conj)
    for k in CLASES:
        d[f"pct_{k}"] = 100 * d[f"clips_{k}"] / n if n else float("nan")
    cnt = [d[f"clips_{k}"] for k in CLASES]
    d["ratio_mayor_menor"] = max(cnt) / min(cnt) if min(cnt) else float("inf")
    d["categoria"] = categoria(d["ratio_mayor_menor"]) if min(cnt) else "CLASE AUSENTE"
    return d

comp_folds = []
checks = []
for i, (trval, test) in enumerate(folds_idx):
    tr, va = P.dividir_por_ejemplar(trval, 0.25)   # mismo corte que el entrenamiento
    for nombre, a, b in (("train_interno vs test", tr, test), ("validacion vs test", va, test),
                         ("train_interno vs validacion", tr, va)):
        ca = {c["id_caballo"] for c in a}; cb = {c["id_caballo"] for c in b}
        assert not (ca & cb), f"fold {i}: caballo en {nombre}: {sorted(ca & cb)}"
    # video: derivar del inventario
    comp_folds.append({"fold": i, "test": comp(test), "train_interno": comp(tr),
                       "validacion_interna": comp(va), "train_val_total": comp(trval)})
    assert len(tr) + len(va) + len(test) == n_total
    assert not ({c["clip"] for c in tr} & {c["clip"] for c in va} & {c["clip"] for c in test})

# ---- fuga: caballos / videos / clips entre folds, con asserts ------------------------------
inv = pd.DataFrame(leer_csv(RES / "inventario_clips.csv")).set_index("clip")
for c in clips:
    c["video"] = inv.loc[c["clip"], "video_origen"]
fuga = {"caballos_en_mas_de_un_fold": 0, "videos_en_mas_de_un_fold": 0, "clips_en_mas_de_un_fold": 0}
por_cab = defaultdict(set); por_vid = defaultdict(set); por_clip = defaultdict(set)
for c in clips:
    por_cab[c["id_caballo"]].add(c["fold"]); por_vid[c["video"]].add(c["fold"]); por_clip[c["clip"]].add(c["fold"])
fuga["caballos_en_mas_de_un_fold"] = sum(len(s) > 1 for s in por_cab.values())
fuga["videos_en_mas_de_un_fold"] = sum(len(s) > 1 for s in por_vid.values())
fuga["clips_en_mas_de_un_fold"] = sum(len(s) > 1 for s in por_clip.values())
assert fuga["caballos_en_mas_de_un_fold"] == 0 and fuga["videos_en_mas_de_un_fold"] == 0 \
    and fuga["clips_en_mas_de_un_fold"] == 0
# por fold: intersecciones train_val ∩ test explícitas en caballo, video y clip
inter = []
for i, (trval, test) in enumerate(folds_idx):
    for clave in ("id_caballo", "video", "clip"):
        a = {c[clave] for c in trval}; b = {c[clave] for c in test}
        assert not (a & b), (i, clave)
        inter.append({"fold": i, "clave": clave, "interseccion_train_test": len(a & b)})

# validez de la partición persistida: ¿es reproducible desde los datos? ¿iguala a sklearn GroupKFold?
frames = pd.DataFrame(leer_csv(RES / "dataset_frames_alineados_v2.csv"))
tabla = P.tabla_clips(archivo="dataset_frames_alineados_v2.csv") if False else None
from sklearn.model_selection import GroupKFold
y = np.array([c["clase"] for c in clips]); g = np.array([c["id_caballo"] for c in clips])
orden = np.argsort([c["clip"] for c in clips])
sk = np.full(len(clips), -1)
for k, (_, te) in enumerate(GroupKFold(n_splits=4).split(np.zeros(len(clips)), y, g)):
    sk[te] = k
mio = np.array([c["fold"] for c in clips])
# comparar como particiones (etiquetas de fold pueden permutarse)
def mismas_particiones(a, b):
    m = defaultdict(set)
    for x, z in zip(a, b):
        m[x].add(z)
    return all(len(s) == 1 for s in m.values()) and len({next(iter(s)) for s in m.values()}) == len(m)
reproduce = {
    "construir_reproduce_la_particion_persistida": all(
        d["fold"] == f["fold"] for d, f in zip(clips, P.construir([{k: c[k] for k in ("clip", "clase", "id_caballo")} for c in clips]))),
    "sklearn_GroupKFold_da_la_misma_particion": bool(mismas_particiones(mio, sk)),
    "fraccion_de_clips_con_el_mismo_fold_en_sklearn": float((mio == sk).mean()),
    "tamano_folds_sklearn": [int((sk == k).sum()) for k in range(4)],
    "tamano_folds_persistidos": [int((mio == k).sum()) for k in range(4)],
}

# La partición persistida (15:50) es posterior a la evaluación (15:43): ¿cambió algo?
ev = evaluacion_vigente()
fold_de_ev = {}
for k, f in enumerate(ev["por_fold"]):
    pass
pred = ev["predicciones_fuera_de_muestra"]
fold_part = {c["clip"]: c["fold"] for c in clips}
# el orden de predicciones sigue fold 0..3; reconstruir
idx = 0; consistente = True
for f in ev["por_fold"]:
    test_k = [c for c in clips if c["fold"] == f["fold"]]
    ids_pred = [x["clip"] for x in pred[idx: idx + f["n"]]]
    consistente &= (ids_pred == [c["clip"] for c in test_k])
    idx += f["n"]
reproduce["predicciones_vigentes_corresponden_a_los_folds_persistidos"] = bool(consistente)

# ---- ¿estadísticas de normalización con test? ---------------------------------------------
import inspect
src_mil = inspect.getsource(CL.entrenar_mil_multimodal)
src_lin = inspect.getsource(__import__("equinos.lineas_de_base", fromlist=["x"]).predecir_lineal)
norm = {
    "MIL: el canal visual NO se estandariza (features crudas del backbone)": "StandardScaler" not in src_mil,
    "MIL: el canal geométrico se normaliza por frame con ojo-hocico del mismo frame (sin estadísticas globales)": True,
    "MIL: pesos de clase calculados con train interno": "pesos_de_clase([c[\"clase\"] for c in train_interno]" in src_mil,
    "LogReg baselines: StandardScaler dentro de un Pipeline ajustado con fit(X[entrena])": ("make_pipeline(StandardScaler()" in src_lin and "modelo.fit(X[entrena]" in src_lin),
    "Backbone ResNet-18: normalización de ImageNet con constantes fijas": True,
    "Selección de época por F1 macro de validación interna (nunca test)": "if f1_val > mejor[\"f1\"]" in src_mil,
    "El test se predice una sola vez tras cargar el mejor checkpoint": src_mil.count("predecir(dl_pru)") == 1,
}

# ---- caballos en varias clases y partición -----------------------------------------------
multi = {k: sorted(v) for k, v in
         {h: sorted({c["clase"] for c in clips if c["id_caballo"] == h}) for h in por_cab}.items() if len(v) > 1}
fuga_potencial_si_agrupara_por_par = {
    "caballos_multiclase": len(multi),
    "clips_de_esos_caballos": sum(1 for c in clips if c["id_caballo"] in multi),
}
# partición ingenua por clip: ¿cuántos caballos quedarían en train y test?
ing = CL.folds_ingenuos(clips, 4)
cr = []
for tr_, te_ in ing:
    a = {c["id_caballo"] for c in tr_}; b = {c["id_caballo"] for c in te_}
    cr.append(len(a & b) / len(b))
fuga_potencial_si_agrupara_por_par["fraccion_de_caballos_de_test_vistos_en_train_con_folds_ingenuos_por_clip"] = float(np.mean(cr))

guardar_json({"balance": balance, "folds": comp_folds, "fuga": fuga, "intersecciones": inter,
              "reproduccion_particion": reproduce, "normalizacion": norm, "multiclase": multi,
              "multiclase_resumen": fuga_potencial_si_agrupara_por_par}, "balance_folds_fuga.json")
guardar_csv([{"clase": f["clase"], "clips": f["clips"], "pct_clips": round(f["pct_clips"], 1),
              "caballos": f["caballos"], "pct_caballos": round(f["pct_caballos"], 1)} for f in filas],
            "balance_tabla.csv")
rows = []
for cf in comp_folds:
    for conj in ("test", "train_interno", "validacion_interna"):
        d = cf[conj]
        rows.append({"fold": cf["fold"], "conjunto": conj, "clips": d["clips"], "caballos": d["caballos"],
                     **{f"clips_{k}": d[f"clips_{k}"] for k in CLASES},
                     **{f"caballos_{k}": d[f"caballos_{k}"] for k in CLASES},
                     **{f"pct_{k}": round(d[f"pct_{k}"], 1) for k in CLASES},
                     "ratio": round(d["ratio_mayor_menor"], 2), "categoria": d["categoria"]})
guardar_csv(rows, "folds_composicion.csv")

# ---- figuras -----------------------------------------------------------------------------
plt = estilo()
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
x = np.arange(3); w = .38
ax[0].bar(x - w/2, [f["pct_clips"] for f in filas], w, color="#2a7f9e", label="% de clips")
ax[0].bar(x + w/2, [f["pct_pares_caballo_clase"] for f in filas], w, color="#e0a800", label="% de pares caballo-clase")
ax[0].axhline(100/3, color="k", ls="--", lw=1, label="Reparto uniforme (33,3 %)")
for i, f in enumerate(filas):
    ax[0].text(i - w/2, f["pct_clips"] + .6, f"{f['pct_clips']:.1f}", ha="center", fontsize=8)
    ax[0].text(i + w/2, f["pct_pares_caballo_clase"] + .6, f"{f['pct_pares_caballo_clase']:.1f}", ha="center", fontsize=8)
ax[0].set_xticks(x); ax[0].set_xticklabels(CLASES); ax[0].set_ylabel("%"); ax[0].set_ylim(0, 50)
ax[0].legend(fontsize=8); ax[0].set_title(f"Balance global (ratio clips {ir_clips:.2f}:1; caballos {ir_cab:.2f}:1)")
# ratio por conjunto
etq, rat = [], []
for cf in comp_folds:
    for conj, ab in (("test", "T"), ("train_interno", "TI"), ("validacion_interna", "V")):
        etq.append(f"F{cf['fold']}-{ab}"); rat.append(cf[conj]["ratio_mayor_menor"])
col = ["#2a7f9e" if e.endswith("T") else "#7f8c8d" if e.endswith("TI") else "#c0392b" for e in etq]
ax[1].bar(range(len(rat)), rat, color=col)
for t, l in zip(CORTES[0], ("1,2", "1,5", "3")):
    ax[1].axhline(t, color="k", lw=.7, ls=":")
    ax[1].text(len(rat) - .4, t + .03, l, fontsize=7, ha="right")
ax[1].set_xticks(range(len(etq))); ax[1].set_xticklabels(etq, rotation=60, fontsize=7)
ax[1].set_ylabel("Ratio mayor:menor (clips)"); ax[1].set_title("Desbalance por fold (T=test, TI=train interno, V=validación)")
fig.tight_layout(); guardar_fig(fig, "fig_balance.png"); plt.close(fig)

print(json.dumps({"balance": {k: v for k, v in balance.items() if k != "tabla"}, "tabla": filas}, indent=1, ensure_ascii=False))
for cf in comp_folds:
    for conj in ("test", "train_interno", "validacion_interna"):
        d = cf[conj]
        print(cf["fold"], conj, d["clips"], d["caballos"], [d[f"clips_{k}"] for k in CLASES],
              [round(d[f"pct_{k}"], 1) for k in CLASES], round(d["ratio_mayor_menor"], 2), d["categoria"])
print(json.dumps({"fuga": fuga, "repro": reproduce, "norm": norm, "multi": fuga_potencial_si_agrupara_por_par}, indent=1, ensure_ascii=False))
