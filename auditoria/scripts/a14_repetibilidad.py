"""Repetibilidad (devolución de Juan): resumen de a04 (misma partición, 6 semillas) y de a15 (otras particiones)."""
import statistics
import numpy as np
from comun import *

d = leer_json(SALIDA / "reentrenamiento_semillas.json")["semillas"]
s = {k: v for k, v in d.items()}
f1 = [v["f1_macro"] for v in s.values()]; acc = [v["accuracy"] for v in s.values()]
ep = np.array([v["mejor_epoca"] for v in s.values()])
res = {"semillas": list(s), "f1_macro_por_semilla": dict(zip(s, f1)), "accuracy_por_semilla": dict(zip(s, acc)),
       "f1_media_entre_semillas": statistics.fmean(f1), "f1_desvio_entre_semillas": statistics.stdev(f1),
       "f1_min": min(f1), "f1_max": max(f1), "rango": max(f1) - min(f1),
       "desvio_entre_folds_medio": statistics.fmean(v["f1_desvio"] for v in s.values()),
       "mejor_epoca_por_semilla_y_fold": ep.tolist(), "mejor_epoca_min": int(ep.min()), "mejor_epoca_max": int(ep.max()),
       "mejor_epoca_desvio_global": float(ep.std(ddof=1)),
       "f1_train_medio": statistics.fmean(np.mean(v["f1_train"]) for v in s.values()),
       "f1_validacion_medio": statistics.fmean(np.mean(v["f1_validacion"]) for v in s.values()),
       "determinismo_semilla_42": {k: s["42"][k] for k in ("predicciones_iguales_a_las_guardadas", "pesos_identicos_bit_a_bit_por_fold",
                                                         "curvas_validacion_iguales", "curvas_train_iguales")},
       "lectura": ("El resultado con la semilla vigente (42) es reproducible bit a bit en esta máquina; con otras semillas el F1 macro "
                   "varía en el orden de ±0,015 y la mejor época cambia mucho entre semillas. El 0,740 es un punto dentro de una "
                   "distribución de ~0,73–0,76, y la diferencia con el baseline estático (0,741) es menor que esa dispersión.")}
try:
    pa = leer_json(SALIDA / "particiones_alternativas.json")
    res["particiones_alternativas"] = pa
except FileNotFoundError:
    pass
guardar_json(res, "repetibilidad.json")
plt = estilo()
fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
ks = list(s)
ax[0].bar(range(len(ks)), f1, color=["#1b3f5a"] + ["#2a7f9e"] * (len(ks) - 1))
ax[0].axhline(0.7405, color="#c0392b", ls="--", lw=1, label="Media 16 frames + LogReg (0,741)")
ax[0].set_ylim(.6, .82); ax[0].set_xticks(range(len(ks))); ax[0].set_xticklabels([f"semilla {k}" for k in ks], rotation=30, fontsize=8)
for i, v in enumerate(f1):
    ax[0].text(i, v + .004, f"{v:.3f}".replace(".", ","), ha="center", fontsize=8)
ax[0].set_ylabel("F1 macro (media de 4 folds)"); ax[0].legend(fontsize=7); ax[0].set_title("Misma partición, distinta semilla")
for j in range(4):
    ax[1].scatter(range(len(ks)), ep[:, j], label=f"Fold {j}", s=30)
ax[1].set_xticks(range(len(ks))); ax[1].set_xticklabels(ks); ax[1].set_xlabel("Semilla"); ax[1].set_ylabel("Mejor época")
ax[1].legend(fontsize=7); ax[1].set_title("La época elegida varía mucho")
if "particiones_alternativas" in res and res["particiones_alternativas"].get("particiones"):
    pp = res["particiones_alternativas"]["particiones"]
    x = np.arange(len(pp))
    ax[2].bar(x - .2, [q["mil_f1_media"] for q in pp], .4, color="#1b3f5a", label="MIL")
    ax[2].bar(x + .2, [q["estatica_f1_media"] for q in pp], .4, color="#2a7f9e", label="Media 16 frames (LogReg)")
    ax[2].set_xticks(x); ax[2].set_xticklabels([f"partición {q['semilla_de_particion']}" for q in pp], fontsize=8)
    ax[2].set_ylim(.6, .82); ax[2].legend(fontsize=8); ax[2].set_title("Otras asignaciones caballo→fold")
fig.tight_layout(); guardar_fig(fig, "fig_repetibilidad.png"); plt.close(fig)
print(json.dumps({k: v for k, v in res.items() if k != "particiones_alternativas"}, ensure_ascii=False, indent=1))
