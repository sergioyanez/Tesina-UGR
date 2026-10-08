"""Selección de los 16 frames: ventana temporal por clase (posible sesgo), espaciado, origen de la alineación."""
import numpy as np
import pandas as pd
from scipy import stats
from comun import *

fr = pd.DataFrame(leer_csv(RES / "dataset_frames_alineados_v2.csv"))
for c in ("span_seg", "validos", "fps", "frame", "orden"):
    fr[c] = fr[c].astype(float)
cl = fr.groupby("clip").agg(clase=("clase", "first"), span=("span_seg", "first"), validos=("validos", "first"),
                            fps=("fps", "first"), orig=("origen_alineacion", lambda s: ",".join(sorted(set(s)))))
cl["paso_s"] = cl.span / 15
inv = pd.DataFrame(leer_csv(RES / "inventario_clips.csv")).set_index("clip")
cl["dur"] = inv.loc[cl.index, "duracion_seg"].astype(float)
cl["pct_del_clip"] = 100 * cl.span / cl.dur
# gaps reales entre frames consecutivos elegidos (en frames del clip)
fr = fr.sort_values(["clip", "orden"])
fr["d"] = fr.groupby("clip")["frame"].diff()
gap = fr.groupby("clip").d.agg(["min", "max", "median"])
cl["gap_max_frames"] = gap["max"]; cl["gap_min_frames"] = gap["min"]
res = {"origen_alineacion": fr.origen_alineacion.value_counts().to_dict(),
       "clips_solo_deteccion_propia": int((cl.orig == "keypoints").sum()), "clips": len(cl),
       "span_seg_por_clase": {k: {q: float(np.percentile(g.span, x)) for q, x in (("p10", 10), ("p25", 25), ("mediana", 50), ("p75", 75), ("p90", 90))}
                              for k, g in cl.groupby("clase")},
       "paso_entre_frames_s_mediana_por_clase": {k: float(g.paso_s.median()) for k, g in cl.groupby("clase")},
       "pct_del_clip_cubierto_mediana_por_clase": {k: float(g.pct_del_clip.median()) for k, g in cl.groupby("clase")},
       "kruskal_wallis_span_por_clase": dict(zip(("H", "p"), map(float, stats.kruskal(*[g.span.values for _, g in cl.groupby("clase")])))),
       "clips_con_ventana_menor_a_2s": {k: int((g.span < 2).sum()) for k, g in cl.groupby("clase")},
       "validos_ojo_hocico_por_clip_mediana": float(cl.validos.median()),
       "gap_maximo_entre_frames_elegidos_(frames)": {"mediana": float(cl.gap_max_frames.median()), "p95": float(cl.gap_max_frames.quantile(.95)), "max": float(cl.gap_max_frames.max())},
       "metodo": "ventana contigua más larga (≤7 s) con ≥16 mediciones válidas y sin huecos >0,5 s; 16 posiciones equiespaciadas en la ventana ancladas al frame medido más cercano (tolerancia 0,1 s, elige el de mayor confianza); NO es muestreo uniforme sobre el clip ni aleatorio",
       "sesgo": "la ventana depende de la disponibilidad de keypoints y de la duración del clip, y la duración depende de la clase"}
# coincidencia con duración: AUC de span para separar Disgustado del resto
from sklearn.metrics import roc_auc_score
res["AUC_span_Disgustado_vs_resto"] = float(roc_auc_score((cl.clase == "Disgustado").astype(int), -cl.span))
guardar_json(res, "seleccion_frames.json")
plt = estilo()
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
for k in CLASES:
    ax[0].hist(cl[cl.clase == k].span, bins=np.arange(0, 7.6, .5), alpha=.6, color=COLOR[k], label=k)
ax[0].set_xlabel("Duración de la ventana de los 16 frames (s)"); ax[0].set_ylabel("Clips"); ax[0].legend(fontsize=8)
ax[0].set_title("La ventana depende de la clase")
data = [cl[cl.clase == k].paso_s.values for k in CLASES]
bp = ax[1].boxplot(data, labels=CLASES, patch_artist=True, showfliers=False)
for b, k in zip(bp["boxes"], CLASES):
    b.set_facecolor(COLOR[k]); b.set_alpha(.7)
ax[1].set_ylabel("Separación media entre frames (s)"); ax[1].set_title("Lo que el modelo ve: intervalo entre frames")
fig.tight_layout(); guardar_fig(fig, "fig_ventana_por_clase.png"); plt.close(fig)
print(json.dumps(res, ensure_ascii=False, indent=1))
