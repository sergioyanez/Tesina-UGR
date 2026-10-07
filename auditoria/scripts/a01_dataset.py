"""Capítulo Dataset y Balance: videos, clips, caballos, clases, folds, descartes. Solo lectura."""
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from comun import *

v = pd.DataFrame(leer_csv(RES / "inventario_videos.csv"))
c = pd.DataFrame(leer_csv(RES / "inventario_clips.csv"))
p = pd.DataFrame(particion())
rech = leer_json(RES / "dataset_rechazos_v2.json")
fr = pd.DataFrame(leer_csv(RES / "dataset_frames_alineados_v2.csv"))

for col in ("duracion_seg", "fps", "ancho", "alto"):
    v[col] = v[col].astype(float)
c["duracion_seg"] = c["duracion_seg"].astype(float)
v["id_caballo"] = v["archivo"].str.extract(r"^(\d+)")[0]

c["valido"] = c["clip"].isin(set(p["clip"]))
c = c.merge(p[["clip", "fold"]], on="clip", how="left")
motivo = {r["clip"]: r["motivo"] for r in rech["rechazos"]}
c["motivo_descarte"] = c["clip"].map(motivo)
assert (c["valido"] ^ c["motivo_descarte"].notna()).all(), "clip ni válido ni con motivo de descarte"
assert rech["informe"]["clips_aceptados"] == int(c["valido"].sum()) == len(p) == 359

# ---- embudo ------------------------------------------------------------------------------
videos_con_clips = set(c["video_origen"])
videos_validos = set(c.loc[c["valido"], "video_origen"])
embudo = {
    "videos_inventariados": len(v), "caballos_en_videos": int(v["id_caballo"].nunique()),
    "videos_sin_clips": len(v) - len(videos_con_clips),
    "videos_sin_clips_detalle": sorted(set(v["archivo"]) - videos_con_clips),
    "videos_con_clips_candidatos": len(videos_con_clips),
    "clips_candidatos": len(c), "caballos_en_clips_candidatos": int(c["id_caballo"].nunique()),
    "clips_descartados": int((~c["valido"]).sum()),
    "motivos_descarte": Counter(c.loc[~c["valido"], "motivo_descarte"]),
    "clips_validos": int(c["valido"].sum()),
    "videos_con_clips_validos": len(videos_validos),
    "caballos_validos": int(p["id_caballo"].nunique()),
    "pct_clips_validos": 100 * c["valido"].mean(),
    "pct_videos_aprovechados": 100 * len(videos_validos) / len(v),
    "pct_caballos_aprovechados": 100 * p["id_caballo"].nunique() / v["id_caballo"].nunique(),
    "frames_totales": int(len(fr)), "frames_por_clip_unico": sorted(set(fr.groupby("clip").size())),
}
tabla_clase = []
for k in CLASES:
    cc = c[c["clase"] == k]
    pc = p[p["clase"] == k]
    tabla_clase.append({
        "clase": k, "videos_inventariados": int((v["clase"] == k).sum()),
        "clips_candidatos": len(cc), "clips_validos": int(cc["valido"].sum()),
        "clips_descartados": int((~cc["valido"]).sum()),
        "pct_validos": 100 * cc["valido"].mean(),
        "videos_validos": int(cc.loc[cc["valido"], "video_origen"].nunique()),
        "caballos_validos": int(pc["id_caballo"].nunique()),
        "caballos_candidatos": int(cc["id_caballo"].nunique()),
    })
embudo["por_clase"] = tabla_clase

# ---- caballos ----------------------------------------------------------------------------
cab = p.groupby("id_caballo").agg(clips=("clip", "size"), clases=("clase", lambda s: sorted(set(s))),
                                    fold=("fold", lambda s: sorted(set(s))))
assert (cab["fold"].map(len) == 1).all()
cab["n_clases"] = cab["clases"].map(len)
clips_por_cab = cab["clips"]
caballos = {
    "n": int(len(cab)), "clips_por_caballo": {
        "media": float(clips_por_cab.mean()), "mediana": float(clips_por_cab.median()),
        "min": int(clips_por_cab.min()), "max": int(clips_por_cab.max()),
        "p90": float(clips_por_cab.quantile(.9)),
        "con_1_clip": int((clips_por_cab == 1).sum()), "con_5_o_mas": int((clips_por_cab >= 5).sum())},
    "caballos_en_1_clase": int((cab.n_clases == 1).sum()),
    "caballos_en_2_clases": int((cab.n_clases == 2).sum()),
    "caballos_en_3_clases": int((cab.n_clases == 3).sum()),
    "pct_caballos_multiclase": 100 * float((cab.n_clases > 1).mean()),
    "clips_de_caballos_multiclase": int(cab.loc[cab.n_clases > 1, "clips"].sum()),
    "top10_caballos_por_clips": cab["clips"].sort_values(ascending=False).head(10).to_dict(),
    "pct_clips_top10": 100 * float(cab["clips"].sort_values(ascending=False).head(10).sum() / len(p)),
    "caballos_multiclase": cab[cab.n_clases > 1].assign(clases=lambda d: d.clases.map("/".join))
                              [["clips", "clases"]].to_dict("index"),
}
# combinaciones de clases por caballo
caballos["combinaciones"] = Counter("+".join(x) for x in cab["clases"])

# clips por video
cv = p.merge(c[["clip", "video_origen"]], on="clip").groupby("video_origen").size()
videos = {"videos_validos": int(len(cv)), "clips_por_video": {
    "media": float(cv.mean()), "mediana": float(cv.median()), "max": int(cv.max()),
    "con_1_clip": int((cv == 1).sum()), "con_2_o_mas": int((cv >= 2).sum()),
    "clips_en_videos_multiclip": int(cv[cv >= 2].sum())}}

# ---- características técnicas de los clips válidos ---------------------------------------
vv = v.set_index("archivo")
cval = c[c.valido].copy()
cval["ancho"] = cval["video_origen"].map(vv["ancho"]); cval["alto"] = cval["video_origen"].map(vv["alto"])
cval["fps_v"] = cval["fps"].astype(float)
cval["res"] = cval["ancho"].astype(int).astype(str) + "x" + cval["alto"].astype(int).astype(str)
tec = {"duracion_seg_por_clase": {}, "resoluciones_por_clase": {}, "pct_clips_5_a_7s_por_clase": {},
       "pct_clips_menos_4_5s_por_clase": {}}
for k in CLASES:
    d = cval[cval.clase == k]
    tec["duracion_seg_por_clase"][k] = {q: float(np.percentile(d.duracion_seg, x)) for q, x in
                                        (("p10", 10), ("p25", 25), ("mediana", 50), ("p75", 75), ("p90", 90))}
    tec["resoluciones_por_clase"][k] = d["res"].value_counts().to_dict()
    tec["pct_clips_5_a_7s_por_clase"][k] = 100 * float(((d.duracion_seg >= 5) & (d.duracion_seg <= 7.0001)).mean())
    tec["pct_clips_menos_4_5s_por_clase"][k] = 100 * float((d.duracion_seg < 4.5).mean())
tec["clips_fuera_5_7s"] = int(((cval.duracion_seg < 5) | (cval.duracion_seg > 7.0001)).sum())
tec["pct_clips_fuera_5_7s"] = 100 * tec["clips_fuera_5_7s"] / len(cval)
tec["clips_menos_4_5s"] = int((cval.duracion_seg < 4.5).sum())

# ---- consistencia de nombres ------------------------------------------------------------
inc = []
chk = {"clip_prefijo_igual_id": bool((c["clip"].str.extract(r"^(\d+)")[0] == c["id_caballo"]).all()),
       "video_prefijo_igual_id": bool((c["video_origen"].str.extract(r"^(\d+)")[0] == c["id_caballo"]).all()),
       "ids_con_3_digitos": bool(c["id_caballo"].str.fullmatch(r"\d{3}").all()),
       "clases_validas": sorted(set(c["clase"]) | set(p["clase"])),
       "nombres_con_espacio": int(v["archivo"].str.contains(" ").sum()),
       "nombres_con_copia": int(v["archivo"].str.contains("copia|Copia", regex=True).sum()),
       "extensiones": Counter(v["archivo"].str.extract(r"\.([A-Za-z0-9]+)$")[0].str.lower()),
       "carpeta_distinta_de_clase": Counter(v.loc[v.carpeta != v.clase, "carpeta"])}
# mismo video en clases distintas por nombre normalizado
norm = v["archivo"].str.lower().str.replace(r"[^a-z0-9]", "", regex=True)
chk["videos_mismo_nombre_normalizado"] = int(norm.duplicated().sum())
# prefijo numérico con rango 5xx (.avi) vs resto
chk["videos_avi"] = int(v["archivo"].str.lower().str.endswith(".avi").sum())
# ids que aparecen con clases distintas según carpeta de inventario
chk["ids_por_rango"] = {"1xx-9xx": int(c["id_caballo"].nunique())}

# ---- por fold ----------------------------------------------------------------------------
f = []
for k in range(4):
    d = p[p.fold == k]
    f.append({"fold": k, "clips": len(d), "caballos": int(d.id_caballo.nunique()),
              **{f"clips_{x}": int((d.clase == x).sum()) for x in CLASES},
              **{f"caballos_{x}": int(d[d.clase == x].id_caballo.nunique()) for x in CLASES}})
por_fold = f

guardar_json({"embudo": embudo, "caballos": caballos, "videos": videos, "tecnico": tec,
              "consistencia_nombres": chk, "por_fold_global": por_fold}, "dataset.json")
guardar_csv(tabla_clase, "dataset_por_clase.csv")
cab.reset_index().assign(clases=lambda d: d.clases.map("/".join), fold=lambda d: d.fold.map(lambda x: x[0])) \
   .to_csv(SALIDA / "caballos.csv", index=False)
c.to_csv(SALIDA / "clips_estado.csv", index=False)

# ---- figuras -----------------------------------------------------------------------------
plt = estilo()
fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.8))
# embudo
et = ["Videos\ninventariados", "Videos con\nclips", "Clips\ncandidatos", "Clips\nválidos"]
va = [len(v), len(videos_con_clips), len(c), int(c.valido.sum())]
ax[0].bar(range(4), va, color=["#7f8c8d", "#7f8c8d", "#2a7f9e", "#1b5e7a"])
for i, x in enumerate(va):
    ax[0].text(i, x + 4, str(x), ha="center", fontsize=9)
ax[0].set_xticks(range(4)); ax[0].set_xticklabels(et, fontsize=8); ax[0].set_title("Embudo del dataset")
ax[0].set_ylabel("Cantidad")
# por clase: clips y caballos
w = 0.38
x = np.arange(3)
cl = [t["clips_validos"] for t in tabla_clase]; ca = [t["caballos_validos"] for t in tabla_clase]
ax[1].bar(x - w / 2, cl, w, label="Clips", color="#2a7f9e")
ax[1].bar(x + w / 2, ca, w, label="Caballos", color="#e0a800")
for i in range(3):
    ax[1].text(i - w / 2, cl[i] + 2, cl[i], ha="center", fontsize=8)
    ax[1].text(i + w / 2, ca[i] + 2, ca[i], ha="center", fontsize=8)
ax[1].set_xticks(x); ax[1].set_xticklabels(CLASES, fontsize=8); ax[1].legend(fontsize=8)
ax[1].set_title("Clips y caballos por clase (válidos)")
# clips por caballo
ax[2].hist(clips_por_cab, bins=range(1, int(clips_por_cab.max()) + 2), color="#2a7f9e", edgecolor="white")
ax[2].set_xlabel("Clips por caballo"); ax[2].set_ylabel("Caballos"); ax[2].set_title("Clips por caballo")
fig.tight_layout(); guardar_fig(fig, "fig_dataset_resumen.png"); plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
bot = np.zeros(4)
for k in CLASES:
    vals = np.array([r[f"clips_{k}"] for r in por_fold])
    ax[0].bar(range(4), vals, bottom=bot, color=COLOR[k], label=k)
    for i, (a, b) in enumerate(zip(vals, bot)):
        ax[0].text(i, b + a / 2, str(a), ha="center", va="center", color="white", fontsize=8)
    bot += vals
ax[0].set_xticks(range(4)); ax[0].set_xticklabels([f"Fold {i}" for i in range(4)])
ax[0].set_ylabel("Clips (conjunto de test del fold)"); ax[0].legend(fontsize=8); ax[0].set_title("Clips por fold y clase")
for k in CLASES:
    d = cval[cval.clase == k]["duracion_seg"]
    ax[1].hist(d, bins=np.arange(0, 7.6, 0.5), alpha=.6, color=COLOR[k], label=k)
ax[1].axvspan(5, 7, color="grey", alpha=.12, label="Rango del plan (5 a 7 s)")
ax[1].set_xlabel("Duración del clip (s)"); ax[1].set_ylabel("Clips"); ax[1].legend(fontsize=8)
ax[1].set_title("Duración de los clips válidos por clase")
fig.tight_layout(); guardar_fig(fig, "fig_dataset_folds_duracion.png"); plt.close(fig)

print(json.dumps({"embudo": {k: v_ for k, v_ in embudo.items() if k not in ("videos_sin_clips_detalle",)},
                  "caballos": {k: v_ for k, v_ in caballos.items() if k != "caballos_multiclase"},
                  "videos": videos, "tecnico": tec, "chk": chk}, ensure_ascii=False, indent=1, default=str)[:7000])
