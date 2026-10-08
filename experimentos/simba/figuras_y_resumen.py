"""Figuras (PNG 200 dpi) y `resumen.json` (las 12 preguntas del §72) a partir de los resultados ya calculados."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import datos

R = datos.SALIDA
ev = json.loads((R / "evaluacion.json").read_text(encoding="utf-8"))
sh = json.loads((R / "shap.json").read_text(encoding="utf-8"))
par = json.loads((R / "parsimonia_mil.json").read_text(encoding="utf-8"))
comp = json.loads((R / "compatibilidad_dlc_simba.json").read_text(encoding="utf-8"))
M = ev["modelos"]
SB, BL, MI = "B_simba_propio|RF", "C_visual_media|LR", "E_MIL_multimodal|vigente"
C = datos.CLASES
AZUL, NARANJA, GRIS, VERDE = "#2a6f97", "#d1603d", "#8d99ae", "#4f772d"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

# ---- 1. matrices de confusión (suma de folds) ----
fig, ejes = plt.subplots(1, 3, figsize=(12, 4))
for ax, (t, k) in zip(ejes, [("Features tipo SimBA (propias) + RF", SB), ("Baseline estático (ResNet-18 media) + LogReg", BL), ("MIL multimodal vigente", MI)]):
    m = np.array(M[k]["original"]["matriz_confusion_suma"]); mn = m / m.sum(1, keepdims=True)
    ax.imshow(mn, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{m[i, j]}\n{mn[i, j]:.0%}", ha="center", va="center", color="white" if mn[i, j] > .55 else "black")
    ax.set_xticks(range(3), C, rotation=20); ax.set_yticks(range(3), C); ax.set_xlabel("predicha"); ax.set_ylabel("real")
    ax.set_title(f"{t}\nF1 macro {M[k]['original']['f1_macro_media']:.3f} (359 clips fuera de muestra)", fontsize=9)
plt.tight_layout(); plt.savefig(R / "matrices_confusion.png", dpi=200); plt.close()

# ---- 2. importancias ----
fig, ejes = plt.subplots(1, 2, figsize=(13, 6.5))
t = sh["importancia_global_top20"][::-1]
col = {"estatica": AZUL, "dinamica": NARANJA, "disponibilidad": GRIS}
ejes[0].barh([x["feature"] for x in t], [x["media_abs_shap"] for x in t], color=[col[x["grupo"]] for x in t])
ejes[0].set_title("Media |SHAP| (OOF, TreeExplainer), top 20\nazul=estática, naranja=dinámica, gris=disponibilidad", fontsize=9)
ejes[0].tick_params(axis="y", labelsize=7)
t2 = sh["importancia_impureza_RF_top20"][::-1]
ejes[1].barh([x["feature"] for x in t2], [x["media_abs_shap"] for x in t2], color=[col[x["grupo"]] for x in t2])
ejes[1].set_title("Importancia por impureza del RF (media de 4 folds), top 20", fontsize=9); ejes[1].tick_params(axis="y", labelsize=7)
plt.tight_layout(); plt.savefig(R / "importancias.png", dpi=200); plt.close()

# ---- 3. comparación de modelos (ablación) ----
orden = [("T_mayoritaria", "Mayoritaria"), ("K1_control_filmacion|LR", "Control filmación (LR)"), ("A_geom9_actual|LR", "A. 9 geom. actuales (LR)"),
         ("A_geom9_actual|RF", "A. 9 geom. actuales (RF)"), ("B_simba_propio|RF", "B. SimBA-like propio (RF)"), ("B_simba_propio|LR", "B. SimBA-like propio (LR)"),
         ("B2_simba_real_movimiento|LR", "B2. SimBA real, movimiento (LR)"), ("C_visual_media|LR", "C. Baseline estático (LR)"), ("D_visual_mas_geom9|LR", "D. Visual + 9 geom. (LR)"),
         ("F_simba_propio_mas_visual|RF", "F. SimBA-like + visual (RF)"), (MI, "E. MIL multimodal")]
fig, ejes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
for ax, tag, tit in ((ejes[0], "original", "Original: 359 clips, F1 macro por fold (media ± desvío)"), (ejes[1], "limpio_evaluado_en_subconjunto", "Subconjunto limpio: 236 clips (1080x1920, >=5 s)")):
    for i, (k, n) in enumerate(orden):
        d = M[k][tag]; v = d["f1_macro_por_fold"]
        c = NARANJA if k == SB else (VERDE if k == MI else (AZUL if k == BL else GRIS))
        ax.barh(i, d["f1_macro_media"], xerr=d["f1_macro_desvio"], color=c, alpha=.85, capsize=2)
        ax.scatter(v, [i] * len(v), s=10, color="black", zorder=3)
    ax.set_yticks(range(len(orden)), [n for _, n in orden]); ax.invert_yaxis(); ax.set_xlim(0, 1)
    ax.axvline(ev["modelos"]["T_azar_estratificado_esperanza"]["f1_macro_media"], ls=":", color="gray")
    ax.set_title(tit, fontsize=9); ax.set_xlabel("F1 macro (puntos = folds; línea punteada = azar)")
plt.tight_layout(); plt.savefig(R / "comparacion_modelos.png", dpi=200); plt.close()

# ---- 4. gaps ----
fig, ax = plt.subplots(figsize=(7, 4))
g = ev["gaps_sobreajuste"]; mil = M[MI]["sobreajuste_artefacto"]
filas = [("SimBA-like RF", g[SB]["gap_train_validacion_medio"], g[SB]["gap_train_test_medio"]), ("SimBA-like LR", g["B_simba_propio|LR"]["gap_train_validacion_medio"], g["B_simba_propio|LR"]["gap_train_test_medio"]),
         ("Baseline estático LR", g[BL]["gap_train_validacion_medio"], g[BL]["gap_train_test_medio"]), ("MIL", mil["gap_train_validacion_medio"], mil["gap_train_test_medio"])]
x = np.arange(len(filas)); ax.bar(x - .2, [f[1] for f in filas], .4, label="gap train − validación", color=GRIS); ax.bar(x + .2, [f[2] for f in filas], .4, label="gap train − test", color=AZUL)
ax.set_xticks(x, [f[0] for f in filas]); ax.legend(); ax.set_ylabel("diferencia de F1 macro (media de 4 folds)"); ax.set_title("Brecha de sobreajuste", fontsize=10)
plt.tight_layout(); plt.savefig(R / "gaps_sobreajuste.png", dpi=200); plt.close()

# ---- resumen.json ----
b = ev["bootstrap_por_caballo"]
def dif(tag, otro):
    d = b[tag]["modelos"][otro]["diferencia_referencia_menos_este"]
    return {"diferencia_F1_agrupado_SimBA_menos_otro": d["media"], "ic95": d["ic95"], "incluye_cero": d["incluye_cero"]}
f = lambda k, tag="original": M[k][tag]
sb, bl, mi = f(SB), f(BL), f(MI)
Mcl = lambda k: M[k]["limpio_evaluado_en_subconjunto"]
reentr = ev["reentrenado_en_limpio"]
res = {
 "1_se_reutilizaron_los_landmarks": {
   "respuesta": "Sí. No se volvió a correr DeepLabCut: se leyó pose_seleccionada_v2.csv (39 bodyparts x 16 frames x 359 clips, coordenadas del cuadro original, formato largo con likelihood). Para SimBA hubo que CONVERTIR a formato ancho + proyecto SimBA; ver compatibilidad_dlc_simba.json.",
   "importacion_directa_a_SimBA": False},
 "2_que_features": {
   "respuesta": "Principal: «features tipo SimBA calculadas con código propio» (NO es SimBA): orejas (largo, ángulo respecto del eje de la cabeza, asimetrías, separación), ojos/cabeza (distancia interocular, escorzo, inclinación), boca/mandíbula, cabeza-cuello; por frame y agregadas por clip (media, desvío, mín, máx, rango); dinámicas (velocidad, velocidad angular, aceleración de cabeza/oreja/mandíbula: media, desvío, máx); disponibilidad de keypoints. Secundario (B2): features REALES de SimBA (UserDefinedFeatureExtractor), que para 1 animal solo describen movimiento.",
   "extractor_real_de_simba_ejecutado": comp["simba_real"].get("resultado")},
 "3_cuantas_quedaron": {"principal_propio": M[SB]["n_features"], "por_grupo": None, "B2_simba_real": M["B2_simba_real_movimiento|RF"]["n_features"], "nota": "160 = 26 series estáticas x 5 agregados + 7 series dinámicas x 3 + 9 de disponibilidad (conteo exacto en columnas.json)"},
 "4_f1_macro": {"SimBA_propio_RF_original": {"media": sb["f1_macro_media"], "desvio": sb["f1_macro_desvio"], "min": sb["f1_macro_min"], "max": sb["f1_macro_max"], "accuracy": sb["accuracy"], "denominador": "359 clips fuera de muestra, 4 folds por caballo"},
   "baseline_estatico_original": bl["f1_macro_media"], "MIL_original": mi["f1_macro_media"],
   "limpio_pooled_(236 clips)": {"SimBA_RF": Mcl(SB)["f1_macro_agrupado"], "baseline": Mcl(BL)["f1_macro_agrupado"], "MIL": Mcl(MI)["f1_macro_agrupado"]}},
 "5_desvio_entre_folds": {"SimBA_RF": sb["f1_macro_desvio"], "baseline": bl["f1_macro_desvio"], "MIL": mi["f1_macro_desvio"], "nota": "desvío muestral (n=4 folds)"},
 "6_clases_mejor_reconocidas": {"F1_por_clase_SimBA_RF": {c: sb["por_clase"][c]["f1"] for c in C}, "F1_por_clase_baseline": {c: bl["por_clase"][c]["f1"] for c in C}, "F1_por_clase_MIL": {c: mi["por_clase"][c]["f1"] for c in C},
   "respuesta": "En SimBA-RF las tres clases quedan parejas (Sereno > Disgustado > Interesado; ver F1 por clase); la confusión más común es Interesado -> Sereno."},
 "7_top_features_SHAP": [(x["feature"], x["media_abs_shap"]) for x in sh["importancia_global_top20"][:10]],
 "8_supera_al_baseline_estatico": {"original": dif("original", "Baseline"), "limpio": dif("limpio", "Baseline"),
   "respuesta": "En el conjunto original sí (diferencia positiva con IC95 que excluye 0). En el subconjunto limpio la diferencia sigue siendo positiva pero el IC incluye 0: no concluyente."},
 "9_supera_al_MIL": {"original": dif("original", "MIL"), "limpio": dif("limpio", "MIL"),
   "respuesta": "En el conjunto original sí (IC95 excluye 0). En el subconjunto limpio la ventaja baja y el IC incluye 0: no concluyente."},
 "10_menor_overfitting": {"SimBA_RF": {"gap_train_validacion": g[SB]["gap_train_validacion_medio"], "gap_train_test": g[SB]["gap_train_test_medio"], "f1_train": g[SB]["f1_train_completo_medio"]},
   "baseline_LR": {"gap_train_validacion": g[BL]["gap_train_validacion_medio"], "gap_train_test": g[BL]["gap_train_test_medio"]},
   "MIL": {"gap_train_validacion": mil["gap_train_validacion_medio"], "gap_train_test": mil["gap_train_test_medio"]},
   "respuesta": "Parcial. El gap train-test del RF (0,14) es menor que el del MIL (0,24) y el baseline (0,26), pero el RF memoriza el train (F1 train ~0,99) y su gap train-validación interna (0,34, sobre ~7 caballos) NO es menor que el del MIL (0,27). Lo que sostiene la comparación es el test fuera de muestra, no el gap."},
 "11_mas_interpretable": {"respuesta": "Sí en el sentido de que cada feature tiene nombre y significado geométrico y SHAP permite rankearlas; con reservas: hay features muy correlacionadas (SHAP reparte el crédito), la lectura es asociación y no causalidad, 2D depende del punto de vista, y las velocidades de cabeza pueden reflejar movimiento de cámara. Requiere validación experta.",
   "aporte_por_familia_anatomica": sh["aporte_por_familia_anatomica_fraccion"]},
 "12_conviene_incorporarlo": {"respuesta": "Conviene incorporarlo como modelo complementario (liviano, sin GPU en entrenamiento ni inferencia salvo DLC, interpretable) y como línea de base geométrica fuerte; NO conviene sustituir al MIL ni reportar la ventaja como definitiva: en el subconjunto limpio la diferencia no es significativa, la señal puede depender del punto de vista/encuadre, hay un solo observador de etiquetas y 133 caballos. Siguiente paso: validar en un conjunto con punto de vista controlado y revisar con un etólogo.",
   "F_simba_mas_visual_no_mejora_a_SimBA_solo": [f("F_simba_propio_mas_visual|RF")["f1_macro_media"], sb["f1_macro_media"]]},
}
res["3_cuantas_quedaron"]["por_grupo"] = {g_: sum(1 for v in json.loads((R / "columnas.json").read_text())["grupos"].values() if v == g_) for g_ in ("estatica", "dinamica", "disponibilidad")}
(R / "resumen.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(res["8_supera_al_baseline_estatico"]), json.dumps(res["3_cuantas_quedaron"]))
