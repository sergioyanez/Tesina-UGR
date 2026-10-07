"""Ensambla los capítulos: lee auditoria/secciones/_fuente/*.md, expande {{TABLA:x}}, {{FIG:archivo|pie}},
{{CONTEO:matriz}} y {{VAL:archivo.json:ruta.a.valor|decimales}} desde los JSON de auditoria/datos, y escribe
auditoria/secciones/NN_*.md. Los marcadores {{SIMBA_*}} NO se tocan (los integra otro agente después)."""
import re
from collections import Counter

from comun import *

AUDS = AUD / "secciones"
FUENTE = AUDS / "_fuente"


def f(x, d=3):
    if x is None:
        return "—"
    return f"{x:.{d}f}".replace(".", ",")


def pct(x, d=1):
    return f(x, d) + " %"


def md(cols, filas, alin=None):
    alin = alin or ["---"] * len(cols)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join(a if a.startswith(":") or a.endswith(":") else "---" for a in alin) + "|"]
    for r in filas:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def J(n):
    return leer_json(SALIDA / n)


# ---------------------------------------------------------------- tablas
def t_embudo():
    d = J("dataset.json")["embudo"]
    fil = [("Videos inventariados", d["videos_inventariados"], f"{d['caballos_en_videos']} identificadores de caballo"),
           ("Videos que generaron al menos un clip", d["videos_con_clips_candidatos"], f"{d['videos_sin_clips']} sin clips (demasiado cortos)"),
           ("Clips candidatos", d["clips_candidatos"], f"{d['caballos_en_clips_candidatos']} caballos"),
           ("Clips descartados", d["clips_descartados"], ", ".join(f"{k}: {v}" for k, v in d["motivos_descarte"].items())),
           ("**Clips válidos (dataset de modelado)**", f"**{d['clips_validos']}**", f"{d['videos_con_clips_validos']} videos, **{d['caballos_validos']} caballos**, {d['frames_totales']} frames (16 por clip)"),
           ("% clips válidos / % videos aprovechados / % caballos", f"{f(d['pct_clips_validos'],1)} % / {f(d['pct_videos_aprovechados'],1)} % / {f(d['pct_caballos_aprovechados'],1)} %", "sobre candidatos / inventariados / identificados")]
    return md(["Etapa", "Cantidad", "Detalle"], fil)


def t_dataset_clase():
    d = J("dataset.json")["embudo"]["por_clase"]
    fil = [(r["clase"], r["videos_inventariados"], r["clips_candidatos"], r["clips_descartados"], r["clips_validos"], f(r["pct_validos"], 1) + " %",
            r["videos_validos"], r["caballos_validos"]) for r in d]
    fil.append(("**Total**", sum(r["videos_inventariados"] for r in d), sum(r["clips_candidatos"] for r in d), sum(r["clips_descartados"] for r in d),
                sum(r["clips_validos"] for r in d), f(100 * sum(r["clips_validos"] for r in d) / sum(r["clips_candidatos"] for r in d), 1) + " %",
                J("dataset.json")["embudo"]["videos_con_clips_validos"], J("dataset.json")["embudo"]["caballos_validos"]))
    return md(["Clase", "Videos inventariados", "Clips candidatos", "Descartados", "Clips válidos", "% válidos", "Videos con clips válidos", "Caballos"], fil)


def t_caballos():
    c = J("dataset.json")["caballos"]
    cp = c["clips_por_caballo"]
    fil = [("Caballos con clips válidos", c["n"]),
           ("Clips por caballo (media / mediana / mín–máx)", f"{f(cp['media'],2)} / {f(cp['mediana'],0)} / {cp['min']}–{cp['max']}"),
           ("Caballos con un único clip", f"{cp['con_1_clip']} ({f(100*cp['con_1_clip']/c['n'],1)} %)"),
           ("Caballos con 5 o más clips", cp["con_5_o_mas"]),
           ("10 caballos con más clips acumulan", f"{f(c['pct_clips_top10'],1)} % de los clips"),
           ("Caballos en una sola clase", c["caballos_en_1_clase"]),
           ("Caballos en dos clases", c["caballos_en_2_clases"]),
           ("Caballos en las tres clases", c["caballos_en_3_clases"]),
           ("Caballos en más de una clase", f"{c['caballos_en_2_clases']+c['caballos_en_3_clases']} ({f(c['pct_caballos_multiclase'],1)} %), con {c['clips_de_caballos_multiclase']} clips ({f(100*c['clips_de_caballos_multiclase']/359,1)} %)")]
    return md(["Medida", "Valor"], fil)


def t_balance():
    b = J("balance_folds_fuga.json")["balance"]
    fil = [(r["clase"], r["clips"], f(r["pct_clips"], 1) + " %", r["caballos"], f(r["pct_caballos"], 1) + " %") for r in b["tabla"]]
    fil.append(("**Total**", 359, "100 %", "133 (distintos)", "—"))
    return md(["Clase", "Clips", "% clips", "Caballos", "% caballos"], fil) + (
        f"\n\n*Los % de caballos se calculan sobre los 133 caballos distintos y suman más de 100 % porque 29 caballos tienen clips en más de una clase.*")


def t_balance_ratios():
    b = J("balance_folds_fuga.json")["balance"]
    fil = [("Clips", f(b["ratio_mayor_menor_clips"], 2) + " : 1", b["categoria_clips"]),
           ("Caballos", f(b["ratio_mayor_menor_caballos"], 2) + " : 1", b["categoria_caballos"])]
    sens = [(" / ".join(f(x, 2) for x in s["cortes"]), s["clips"], s["caballos"]) for s in b["sensibilidad"]]
    return md(["Unidad", "Ratio mayor : menor", "Clasificación (criterio declarado)"], fil) + "\n\n**Sensibilidad al criterio**\n\n" + md(["Cortes del ratio (balanceado / razonable / moderado)", "Clips", "Caballos"], sens)


def t_folds_global():
    fl = J("dataset.json")["por_fold_global"]
    fil = [(r["fold"], r["caballos"], r["clips"], r["clips_Disgustado"], r["clips_Interesado"], r["clips_Sereno"],
            f"{r['caballos_Disgustado']} / {r['caballos_Interesado']} / {r['caballos_Sereno']}") for r in fl]
    return md(["Fold (test)", "Caballos", "Clips", "Disgustado", "Interesado", "Sereno", "Caballos D / I / S"], fil)


def t_folds_conjuntos():
    rows = []
    import csv as _csv
    for r in leer_csv(SALIDA / "folds_composicion.csv"):
        n = int(r["clips"])
        nom = {"test": "Test", "train_interno": "Train interno", "validacion_interna": "Validación interna"}[r["conjunto"]]
        rows.append((r["fold"], nom, r["caballos"], n,
                     f"{r['clips_Disgustado']} ({r['pct_Disgustado'].replace('.', ',')} %)",
                     f"{r['clips_Interesado']} ({r['pct_Interesado'].replace('.', ',')} %)",
                     f"{r['clips_Sereno']} ({r['pct_Sereno'].replace('.', ',')} %)",
                     r["ratio"].replace(".", ",") + " : 1", r["categoria"].capitalize()))
    return md(["Fold", "Conjunto", "Caballos", "Clips", "Disgustado", "Interesado", "Sereno", "Ratio", "Categoría"], rows)


def t_early_stopping():
    t = J("early_stopping_sobreajuste.json")["tabla"]
    fil = [(r["fold"], r["mejor_epoca"], r["epocas_ejecutadas"], f(r["f1_validacion"]), f(r["f1_test"]), f(r["f1_train"])) for r in t]
    m = J("early_stopping_sobreajuste.json")["medias"]
    fil.append(("**Media**", f(m["mejor_epoca"], 1), f(m["epocas_ejecutadas"], 1), f(m["f1_validacion"]), f(m["f1_test"]), f(m["f1_train"])))
    return md(["Fold", "Mejor época", "Épocas ejecutadas", "Métrica validación (F1 macro)", "Métrica test (F1 macro)", "F1 train interno"], fil)


def t_gaps():
    t = J("early_stopping_sobreajuste.json")["tabla"]
    m = J("early_stopping_sobreajuste.json")["medias"]
    fil = [(r["fold"], f(r["f1_train"]), f(r["f1_validacion"]), f(r["f1_test"]), f(r["gap_train_val_f1"]), f(r["gap_train_test_f1"]),
            f(r["gap_val_test_f1"]), f(r["acc_train_mejor"]), f(r["acc_val_mejor"])) for r in t]
    fil.append(("**Media**", f(m["f1_train"]), f(m["f1_validacion"]), f(m["f1_test"]), f(m["gap_train_val_f1"]), f(m["gap_train_test_f1"]),
                f(m["gap_val_test_f1"]), f(sum(r["acc_train_mejor"] for r in t) / 4), f(sum(r["acc_val_mejor"] for r in t) / 4)))
    return md(["Fold", "F1 train", "F1 val", "F1 test", "Gap train−val", "Gap train−test", "Val−test", "Acc. train", "Acc. val"], fil)


def t_perdida():
    t = J("early_stopping_sobreajuste.json")["tabla"]
    fil = [(r["fold"], r["epoca_loss_val_minima"], r["mejor_epoca"], f(r["loss_val_minima"]), f(r["loss_val_mejor"]), f(r["loss_val_final"]),
            f(r["f1_val_media_ventana_mejor_pm3"]), f(r["f1_val_final"]), f(r["caida_f1_val_hasta_el_final"])) for r in t]
    return md(["Fold", "Época de mínima pérdida val.", "Mejor época (F1)", "Pérdida val. mínima", "Pérdida val. en la mejor época", "Pérdida val. al final", "F1 val medio (mejor ±3 épocas)", "F1 val al final", "Δ F1 val (final − mejor)"], fil)


def t_metricas_fold():
    m = J("metricas.json")
    fil = [(r["fold"], r["n"], f(r["f1_macro"]), f(r["accuracy"]), f(r["balanced_accuracy"]),
            f(r["f1"]["Disgustado"]), f(r["f1"]["Interesado"]), f(r["f1"]["Sereno"])) for r in m["por_fold"]]
    e = m["estabilidad"]
    fil.append(("**Media ± DE**", 359, f"**{f(e['f1_macro_media'])} ± {f(e['f1_macro_desvio_ddof1'])}**", f"{f(e['accuracy_media'])} ± {f(e['accuracy_desvio'])}", "", "", "", ""))
    return md(["Fold", "n", "F1 macro", "Accuracy", "Balanced acc.", "F1 Disgustado", "F1 Interesado", "F1 Sereno"], fil)


def t_estabilidad():
    e = J("metricas.json")["estabilidad"]
    fil = [("F1 macro media entre folds", f(e["f1_macro_media"])), ("Desvío estándar (n−1)", f(e["f1_macro_desvio_ddof1"])),
           ("Desvío estándar poblacional (n)", f(e["f1_macro_desvio_ddof0"])), ("Mínimo (fold 2)", f(e["min"])), ("Máximo (fold 3)", f(e["max"])),
           ("Rango", f(e["rango"])), ("Coeficiente de variación", f(e["cv_pct"], 1) + " %")]
    return md(["Medida", "Valor"], fil)


def t_por_clase():
    g = J("metricas.json")["global"]
    fil = [(k, f(v["precision"]), f(v["recall"]), f(v["f1"]), v["soporte"]) for k, v in g["por_clase"].items()]
    fil.append(("**Macro**", f(sum(v["precision"] for v in g["por_clase"].values()) / 3), f(sum(v["recall"] for v in g["por_clase"].values()) / 3),
                f(g["f1_macro_agrupado_sklearn"]), 359))
    return md(["Clase", "Precision", "Recall", "F1", "Soporte"], fil) + "\n\n*Métricas calculadas sobre las 359 predicciones fuera de muestra agrupadas; la media de F1 por fold (0,740) difiere de la agrupada (0,742) por el promedio.*"


def _cm(m, clases=CLASES):
    fil = [(f"**Real {c}**", *m[i], sum(m[i])) for i, c in enumerate(clases)]
    fil.append(("**Total predicho**", *[sum(m[i][j] for i in range(3)) for j in range(3)], sum(sum(r) for r in m)))
    return md(["", "Pred. Disgustado", "Pred. Interesado", "Pred. Sereno", "Total"], fil)


def t_confusion_global():
    return _cm(J("metricas.json")["global"]["matriz_filas_real_columnas_predicha"])


def t_confusion_folds():
    out = []
    for r in J("metricas.json")["por_fold"]:
        out.append(f"**Fold {r['fold']}** (n = {r['n']}, F1 macro {f(r['f1_macro'])})\n\n" + _cm(r["matriz"]))
    return "\n\n".join(out)


def t_prf_folds():
    fil = []
    for r in J("metricas.json")["por_fold"]:
        for k in CLASES:
            fil.append((r["fold"], k, f(r["precision"][k]), f(r["recall"][k]), f(r["f1"][k]), r["soporte"][k]))
    return md(["Fold", "Clase", "Precision", "Recall", "F1", "Soporte"], fil)


def t_historico():
    fil = [(h["cifra"], h["modelo"], h["dataset"], h["fecha"], h["commit"], h["artefacto_local"], h["vigente"]) for h in J("metricas.json")["historico"]]
    return md(["Cifra", "Modelo", "Dataset", "Fecha", "Commit / origen", "Artefacto", "¿Vigente?"], fil)


def t_baselines():
    b = J("metricas.json")["baselines"]
    g = b["del_repo(objetivo3_lineas_de_base.json)"]
    fil = [("Azar uniforme (1/3 por clase), esperanza", f(b["aleatoria_uniforme_1/3"]["f1_macro_esperado"]), f"P95 sim. {f(b['aleatoria_uniforme_1/3']['p95'])}", "≈ 0,333"),
           ("Azar estratificado (proporciones del train), esperanza", f(b["aleatoria_estratificada"]["f1_macro_esperado"]), f"P95 sim. {f(b['aleatoria_estratificada']['p95'])}", "—"),
           ("Clase mayoritaria (Sereno en los 4 folds)", f(b["mayoritaria_del_train_del_fold"]["f1_macro"]) + " ± " + f(b["mayoritaria_del_train_del_fold"]["desvio"]), "", f(b["mayoritaria_del_train_del_fold"]["accuracy"])),
           ("Solo duración del clip (LogReg)", f"{f(g['Solo duración del clip (LogReg)']['f1_macro'])} ± {f(g['Solo duración del clip (LogReg)']['f1_desvio'])}", "", f(g['Solo duración del clip (LogReg)']['accuracy'])),
           ("Metadatos de filmación (LogReg)", f"{f(g['Metadatos de filmación (LogReg)']['f1_macro'])} ± {f(g['Metadatos de filmación (LogReg)']['f1_desvio'])}", "", f(g['Metadatos de filmación (LogReg)']['accuracy'])),
           ("Geometría sola, media y desvío (LogReg)", f"{f(g['Geometría sola: media y desvío (LogReg)']['f1_macro'])} ± {f(g['Geometría sola: media y desvío (LogReg)']['f1_desvio'])}", "", f(g['Geometría sola: media y desvío (LogReg)']['accuracy'])),
           ("Un frame central (LogReg)", f"{f(g['Visual: 1 frame central (LogReg)']['f1_macro'])} ± {f(g['Visual: 1 frame central (LogReg)']['f1_desvio'])}", "", f(g['Visual: 1 frame central (LogReg)']['accuracy'])),
           ("**Media de los 16 frames (LogReg)**", f"**{f(g['Visual estática: media de 16 frames (LogReg)']['f1_macro'])} ± {f(g['Visual estática: media de 16 frames (LogReg)']['f1_desvio'])}**", "", f(g['Visual estática: media de 16 frames (LogReg)']['accuracy'])),
           ("**MIL multimodal (vigente)**", f"**{f(g['MIL multimodal (vigente)']['f1_macro'])} ± {f(g['MIL multimodal (vigente)']['f1_desvio'])}**", "", f(g['MIL multimodal (vigente)']['accuracy']))]
    return md(["Modelo / línea de base", "F1 macro (media ± DE entre folds)", "Nota", "Accuracy"], fil)


def t_bootstrap():
    b = J("metricas.json")["bootstrap"]["modelos"]
    fil = []
    for k, v in b.items():
        d = v.get("diferencia_referencia_menos_este")
        fil.append((k, f"[{f(v['f1_ic95'][0])}; {f(v['f1_ic95'][1])}]",
                    "—" if not d else f"{d['media']:+.3f}".replace(".", ","), "—" if not d else f"[{d['ic95'][0]:+.3f}; {d['ic95'][1]:+.3f}]".replace(".", ","),
                    "—" if not d else ("sí" if d["incluye_cero"] else "no")))
    return md(["Modelo", "IC95 % del F1 macro", "Diferencia MIL − modelo", "IC95 % de la diferencia", "¿Incluye 0?"], fil)


def t_arq():
    a = J("arquitectura.json")["arquitectura"]
    fil = [("MIL multimodal: parámetros totales", f"{a['MIL multimodal: parametros_totales']:,}".replace(",", " ")),
           ("MIL multimodal: parámetros entrenables", f"{a['MIL multimodal: parametros_entrenables']:,}".replace(",", " ")),
           ("   · geom_mlp (9 → 64 → 128)", f"{a['por_componente']['geom_mlp (9→64→128)']:,}".replace(",", " ")),
           ("   · atención (640 → 128 → 1)", f"{a['por_componente']['attention (640→128→1)']:,}".replace(",", " ") + f" ({f(a['pct_atencion'],1)} %)"),
           ("   · clasificador (640 → 64 → 3)", f"{a['por_componente']['classifier (640→64→3)']:,}".replace(",", " ")),
           ("ResNet-18 sin capa final: parámetros", f"{a['resnet18_sin_fc: parametros_totales']:,}".replace(",", " ") + " (todos congelados)"),
           ("Entrenables tras congelar la ResNet-18", a["resnet18_sin_fc: entrenables_tras_congelar"]),
           ("Pipeline completo: totales / entrenables", f"{a['pipeline_total_parametros']:,}".replace(",", " ") + " / " + f"{a['pipeline_entrenables']:,}".replace(",", " ")),
           ("¿Coincide con la guía (132 355 entrenables)?", "sí")]
    return md(["Elemento", "Valor"], fil)


def t_dimension():
    a = J("arquitectura.json")["arquitectura"]
    r = a["razones"]
    fil = [("Entrada por clip", "16 frames × (512 visuales + 9 geométricas) = 8 336 valores"),
           ("Clips de entrenamiento por fold (train interno / train+val)", "≈ 199 / ≈ 269"),
           ("Caballos de entrenamiento por fold (train interno)", "≈ 93"),
           ("Parámetros entrenables por clip de train interno", f(r["parametros_entrenables_por_clip_train_interno"], 0)),
           ("Parámetros entrenables por caballo de train interno", f(r["parametros_entrenables_por_caballo_train_interno"], 0)),
           ("Valores de entrada por clip / clips de train interno", f(r["dimension_de_entrada_por_clip_sobre_clips_train"], 1))]
    return md(["Medida", "Valor"], fil)


def t_semillas():
    d = J("repetibilidad.json")
    s = J("reentrenamiento_semillas.json")["semillas"]
    fil = [(k, f(v["f1_macro"]), f(v["f1_desvio"]), f(v["accuracy"]), ", ".join(map(str, v["mejor_epoca"]))) for k, v in s.items()]
    fil.append(("**Media ± DE entre semillas**", f"**{f(d['f1_media_entre_semillas'])} ± {f(d['f1_desvio_entre_semillas'])}**", f(d["desvio_entre_folds_medio"]), "", f"rango {d['mejor_epoca_min']}–{d['mejor_epoca_max']}"))
    return md(["Semilla", "F1 macro (media de 4 folds)", "DE entre folds", "Accuracy", "Mejor época por fold (0–3)"], fil)


def t_particiones_alt():
    try:
        pa = J("particiones_alternativas.json")["particiones"]
    except FileNotFoundError:
        return "*(pendiente)*"
    fil = [(p["semilla_de_particion"], f(p["mil_f1_media"]), f(p["mil_f1_desvio"]), f(p["estatica_f1_media"]), f(p["estatica_f1_desvio"]),
            f"{p['diferencia_mil_menos_estatica']:+.3f}".replace(".", ",")) for p in pa]
    return md(["Partición (semilla de asignación)", "MIL F1", "MIL DE", "Media 16 frames F1", "Estática DE", "Dif. MIL − estática"], fil)


def t_calibracion():
    c = J("calibracion.json")["tabla"]
    return md(["Rango de confianza", "Clips", "Confianza media", "Acierto observado"], [(t["rango"], t["n"], f(t["confianza_media"]), f(t["acierto"])) for t in c])


def t_confusores():
    c = J("confusores.json")
    fil = [(k, f"{f(v['f1_macro'])} ± {f(v['f1_desvio'])}", f(v["accuracy"])) for k, v in c["baselines_sin_mirar_la_cara"].items()]
    return md(["Información usada (sin mirar la cara)", "F1 macro (media ± DE)", "Accuracy"], fil)


def t_rotacion():
    c = J("confusores.json")
    fil = [(k, v["videos"], v["rotados"], f(100 * v["rotados"] / v["videos"], 1) + " %") for k, v in c["rotacion_declarada_por_clase"].items()]
    return md(["Clase", "Videos", "Con rotación declarada (90°)", "%"], fil)


def t_ventana():
    s = J("seleccion_frames.json")
    fil = [(k, f(v["p25"], 2), f(v["mediana"], 2), f(v["p75"], 2), f(s["paso_entre_frames_s_mediana_por_clase"][k], 2), s["clips_con_ventana_menor_a_2s"][k]) for k, v in s["span_seg_por_clase"].items()]
    return md(["Clase", "Ventana P25 (s)", "Mediana (s)", "P75 (s)", "Separación mediana entre frames (s)", "Clips con ventana < 2 s"], fil)


def t_tecnico():
    t = J("dataset.json")["tecnico"]
    fil = [(k, f(t["duracion_seg_por_clase"][k]["p25"], 1), f(t["duracion_seg_por_clase"][k]["mediana"], 1), f(t["duracion_seg_por_clase"][k]["p75"], 1),
            f(t["pct_clips_5_a_7s_por_clase"][k], 0) + " %", ", ".join(f"{a} ({b})" for a, b in t["resoluciones_por_clase"][k].items())) for k in CLASES]
    return md(["Clase", "Duración P25 (s)", "Mediana (s)", "P75 (s)", "% en 5–7 s", "Resoluciones (clips)"], fil)


def t_inferencia_prueba():
    d = J("orientacion_inferencia.json")["pruebas_extremo_a_extremo"]
    fil = [(p["video"][:34], p["clase_real"], "90°" if p["rotacion_declarada"] else "0°", p["resultado"] + (f": {p['clase_predicha']}" if p.get("clase_predicha") else f" ({p.get('motivo')})"),
            f(p["segundos"], 0) + " s") for p in d]
    return md(["Video (del dataset)", "Clase real", "Rotación", "Resultado de `clasificar_video`", "Tiempo"], fil)


def t_semilla42():
    d = J("reentrenamiento_semillas.json")["semillas"]["42"]
    fil = [("Predicciones fuera de muestra idénticas a las guardadas", d["predicciones_iguales_a_las_guardadas"]),
           ("Curvas de F1 de validación idénticas (4 folds)", "sí" if d["curvas_validacion_iguales"] else "no"),
           ("Curvas de F1 de train idénticas (4 folds)", "sí" if d["curvas_train_iguales"] else "no"),
           ("Pesos idénticos bit a bit (folds 0–3)", ", ".join("sí" if x else "no" for x in d["pesos_identicos_bit_a_bit_por_fold"])),
           ("Tiempo de re-entrenamiento (4 folds, GPU)", f"{d['segundos']:.0f} s")]
    return md(["Verificación con el código vigente y semilla 42", "Resultado"], fil)


def t_inferencia_resumen():
    r = J("orientacion_inferencia.json")["resumen_e2e"]
    def fila(n, d):
        return (n, d["total"], d["predicciones"], d["negativas"], d["errores"], ", ".join(f"{k}: {v}" for k, v in d["motivos_de_negativa"].items()) or "—", f(d["segundos_medios"], 0) + " s")
    return md(["Grupo", "Videos", "Predicciones", "Negativas", "Errores", "Motivos de negativa", "Tiempo medio"],
              [fila("Con rotación declarada (90°)", r["rotados"]), fila("Sin rotación", r["sin_rotacion"]), fila("**Total**", r)])


# ----------------------------------------------------------------- SimBA
SB = WT / "experimentos" / "simba" / "resultados"


def JS(n):
    return leer_json(SB / n)


def t_simba_tres_enfoques():
    e = JS("evaluacion.json")["modelos"]
    v = leer_json(WT / "experimentos" / "simba" / "verificacion" / "verificacion.json")
    def r(k):
        o = e[k]["original"]
        return f"{f(o['f1_macro_media'])} ± {f(o['f1_macro_desvio'])}"
    fil = [("**MIL multimodal (vigente)**", "Imagen (ResNet-18 congelada, 512/frame) + 9 variables geométricas, 16 frames", "132 355 parámetros entrenables (+ 11,2 M congelados); GPU para extraer características", "Baja: atención por frame, sin significado de rasgo", r("E_MIL_multimodal|vigente")),
           ("Baseline estático (LogReg)", "Media de las 512 características de los 16 frames", "1 539 parámetros; requiere la ResNet-18", "Media: coeficientes sobre características visuales sin nombre", r("C_visual_media|LR")),
           ("**Random Forest sobre 160 features «tipo SimBA» (código propio)**", "Solo landmarks de DLC: orejas, ojos, boca, cabeza-cuello; estáticas (130), dinámicas (21) y disponibilidad (9)", "500 árboles (≈ 26 000 nodos); 3,3 s de ajuste en total, sin GPU (además de DLC)", "Media-alta: cada feature tiene nombre geométrico; SHAP", f"**{f(v['6_semillas']['media_5'])} ± {f(v['6_semillas']['desvio_5'])}** (media de 5 semillas; DE entre semillas). Semilla 0: {r('B_simba_propio|RF')} (DE entre folds)")]
    return md(["Modelo", "Información que usa", "Complejidad", "Interpretabilidad", "F1 macro ± desvío"], fil)


def t_simba_por_fold():
    e = JS("evaluacion.json")["modelos"]
    ks = [("SimBA-RF (semilla 0)", "B_simba_propio|RF"), ("MIL vigente", "E_MIL_multimodal|vigente"), ("Baseline estático", "C_visual_media|LR")]
    fil = []
    for k in range(4):
        fil.append((k, *[f(e[kk]["original"]["f1_macro_por_fold"][k]) for _, kk in ks]))
    fil.append(("**Media ± DE**", *[f"{f(e[kk]['original']['f1_macro_media'])} ± {f(e[kk]['original']['f1_macro_desvio'])}" for _, kk in ks]))
    fil.append(("Accuracy", *[f(e[kk]["original"]["accuracy"]) for _, kk in ks]))
    t1 = md(["Fold", *[n for n, _ in ks]], fil)
    pc = []
    rf = e["B_simba_propio|RF"]["original"]["por_clase"]; mil = J("metricas.json")["global"]["por_clase"]
    for c in CLASES:
        pc.append((c, f(rf[c]["precision"]), f(rf[c]["recall"]), f(rf[c]["f1"]), f(mil[c]["precision"]), f(mil[c]["recall"]), f(mil[c]["f1"]), rf[c]["soporte"]))
    return t1 + "\n\n" + md(["Clase", "RF P", "RF R", "RF F1", "MIL P", "MIL R", "MIL F1", "Soporte"], pc)


NOMBRES_ABL = [("K1_control_filmacion|LR", "K1 Control de filmación (7 variables: duración, fps, resolución, ventana, escala, dt) — post hoc"),
               ("A_geom9_actual|RF", "A. 9 geométricas vigentes (RF)"), ("A_geom9_actual|LR", "A. 9 geométricas vigentes (LogReg)"),
               ("B_simba_propio|RF", "B. 160 features tipo SimBA propias (RF)"), ("B_simba_propio|LR", "B. ídem (LogReg)"),
               ("B_simba_propio_estatico|RF", "B-a. Solo estáticas (130)"), ("B_simba_propio_dinamico|RF", "B-b. Solo dinámicas (21)"),
               ("B_simba_propio_sin_dinamica_de_cabeza|RF", "B-c. Sin dinámica de cabeza (151) — post hoc"), ("B_simba_propio_sin_disponibilidad|RF", "B-d. Sin disponibilidad (151)"),
               ("B2_simba_real_movimiento|LR", "B2. SimBA real, solo movimiento (174; LogReg)"), ("C_visual_media|LR", "C. Baseline estático (LogReg)"),
               ("D_visual_mas_geom9|LR", "D. Visual + 9 geométricas (LogReg)"), ("E_MIL_multimodal|vigente", "E. MIL vigente"), ("F_simba_propio_mas_visual|RF", "F. SimBA propio + visual (RF)")]


def t_simba_ablacion():
    e = JS("evaluacion.json")["modelos"]
    fil = []
    for k, n in NOMBRES_ABL:
        m = e[k]; o = m["original"]; l = m.get("limpio_evaluado_en_subconjunto", {})
        fil.append((n, m.get("n_features") or "—", f"{f(o['f1_macro_media'])} ± {f(o['f1_macro_desvio'])}", f(l.get("f1_macro_media")) if l else "—", f(l.get("f1_macro_agrupado")) if l else "—"))
    return md(["Variante", "Features", "F1 macro original (media ± DE)", "Limpio: media de folds", "Limpio: agrupado"], fil)


def t_simba_orig_limpio():
    e = JS("evaluacion.json"); v = leer_json(WT / "experimentos" / "simba" / "verificacion" / "verificacion.json")
    M = e["modelos"]; R = e["reentrenado_en_limpio"]
    def row(n, k, rk):
        o = M[k]["original"]; l = M[k]["limpio_evaluado_en_subconjunto"]
        return (n, f"{f(o['f1_macro_media'])} ± {f(o['f1_macro_desvio'])}", f(l["f1_macro_media"]), f(l["f1_macro_agrupado"]), f(R[rk]["f1_macro_agrupado"]) if rk else "no reentrenado")
    fil = [row("Baseline estático (LogReg)", "C_visual_media|LR", "C_visual_media|LR"), row("SimBA-RF propio (semilla 0)", "B_simba_propio|RF", "B_simba_propio|RF"), row("MIL vigente", "E_MIL_multimodal|vigente", None)]
    t = md(["Modelo", "Original (359 clips)", "Limpio: media de folds (236 clips)", "Limpio: agrupado", "Entrenado y evaluado solo en limpio (agrupado)"], fil)
    t2 = md(["Control sobre SimBA-RF (verificación adversarial)", "F1 macro"], [
        ("Media de 5 semillas (1–5)", f"{f(v['6_semillas']['media_5'])} ± {f(v['6_semillas']['desvio_5'])}"),
        ("Residualizando las features por duración, fps, resolución y proporción (por fold)", f(v["3c2_residualizado_por_fold"]["f1_media"])),
        ("Clips de duración < 5 s (95 clips; 71 son `Disgustado`)", f(v["3c_estratos_base"]["dur<5"]["f1_agrupado"])),
        ("Clips de duración ≥ 5 s (264 clips)", f(v["3c_estratos_base"]["dur>=5"]["f1_agrupado"])),
        ("Solo metadatos de filmación (verificación)", f(v["3d_metadatos_solos"]["f1_media"])),
        ("Features + metadatos", f(v["3d_features_mas_metadatos"]["f1_media"])),
        ("Solo disponibilidad de keypoints", f(v["3b3_solo_disponibilidad"]["f1_media"])),
        ("Etiquetas permutadas (control negativo, 3 repeticiones)", " / ".join(f(x, 2) for x in v["control_negativo_etiquetas_permutadas_F1"])),
        ("Predecir log-duración desde las 160 features (R²)", f(v["3e_predecir_filmacion"]["R2_log_duracion"], 2)),
        ("Predecir resolución 1080×1920 desde las 160 features (AUC)", f(v["3e_predecir_filmacion"]["AUC_resolucion_1080x1920"], 2))])
    return t + "\n\n" + t2


def t_simba_diferencias():
    b = JS("evaluacion.json")["bootstrap_por_caballo"]
    fil = []
    for conj in ("original", "limpio"):
        for k in ("MIL", "Baseline"):
            d = b[conj]["modelos"][k]["diferencia_referencia_menos_este"]
            fil.append((conj.capitalize(), f"SimBA-RF − {k}", f"{d['media']:+.3f}".replace(".", ","), f"[{d['ic95'][0]:+.3f}; {d['ic95'][1]:+.3f}]".replace(".", ","), "no" if d["incluye_cero"] is False else "sí (no concluyente)"))
    return md(["Conjunto", "Diferencia (F1 agrupado, semilla 0)", "Media", "IC 95 % (bootstrap por caballo)", "¿Incluye 0?"], fil) + "\n\n*Verificación independiente (sklearn, 133 caballos): +0,111 [0,045; 0,186].*"


def t_simba_gaps():
    g = JS("evaluacion.json")["gaps_sobreajuste"]; mil = J("early_stopping_sobreajuste.json")["medias"]
    fil = []
    for k, n in (("B_simba_propio|RF", "SimBA-RF"), ("B_simba_propio|LR", "SimBA LogReg"), ("A_geom9_actual|RF", "9 geométricas (RF)"), ("C_visual_media|LR", "Baseline estático (LogReg)")):
        w = g[k]; fil.append((n, f(w["f1_train_completo_medio"]), f(w["f1_validacion_interna_medio"]), f(w["gap_train_validacion_medio"]), f(w["gap_train_test_medio"])))
    fil.append(("MIL vigente", f(mil["f1_train"]), f(mil["f1_validacion"]), f(mil["gap_train_val_f1"]), f(mil["gap_train_test_f1"])))
    return md(["Modelo", "F1 train", "F1 validación interna", "Gap train−validación", "Gap train−test"], fil)


def t_simba_shap_global():
    s = JS("shap.json")
    fil = [(x["rank"], f"`{x['feature']}`", f(x["media_abs_shap"], 4), x["grupo"]) for x in s["importancia_global_top20"]]
    return md(["#", "Feature (agregado por clip)", "Media del valor absoluto de SHAP (prob.)", "Grupo"], fil)


def t_simba_shap_clase():
    s = JS("shap.json")["importancia_por_clase_top20"]
    fil = []
    for c in CLASES:
        fil.append((c, "; ".join(f"`{x['feature']}` ({f(x['spearman_valor_vs_shap'],2)})" for x in s[c][:5])))
    fam = JS("shap.json")["aporte_por_familia_anatomica_fraccion"]
    return md(["Clase", "Top 5 por media del valor absoluto de SHAP (entre paréntesis, Spearman valor–SHAP: el signo indica la dirección)"], fil) + "\n\n" + md(["Familia anatómica", "Fracción del valor absoluto de SHAP total"], [(k, f(100 * v, 1) + " %") for k, v in fam.items()])


def t_simba_inventario():
    inv = JS("inventario_anatomico.json")["features_candidatas"]
    fil = [(x["feature"], x["clasificacion"], x["motivo"][:150]) for x in inv]
    return md(["Aspecto facial", "Clasificación", "Motivo"], fil)


def t_simba_disponibilidad():
    d = JS("inventario_anatomico.json")["disponibilidad_por_bodypart"]
    ks = ["nose", "upper_jaw", "lower_jaw", "mouth_end_right", "mouth_end_left", "right_eye", "left_eye", "right_earbase", "right_earend", "left_earbase", "left_earend", "neck_base", "neck_end"]
    return md(["Landmark", "% de frames con confianza ≥ 0,50", "% de clips con algún frame ≥ 0,50"], [(k, f(100 * d[k]["frames_conf_ge_0.50"], 1) + " %", f(100 * d[k]["clips_con_algun_frame_ge_0.50"], 1) + " %") for k in ks if k in d])


def t_simba_parsimonia():
    p = JS("evaluacion.json")["parsimonia_medida"]; m = JS("parsimonia_mil.json")
    fil = [("MIL vigente", f"{m['parametros_entrenables_MIL']:,}".replace(",", " ") + " entrenables (+ 11 176 512 congelados)", f"{f(m['entrenamiento_fold0_s'],0)} s (fold 0, GPU, desde características cacheadas, 1 corrida)", f"{f(m['inferencia_ms_por_clip_cpu_desde_caracteristicas_cacheadas'],2)} ms (CPU, sin extraer ResNet)", "Sí (extracción de características)", "Baja"),
           ("Baseline estático (LogReg)", "1 539", f"{f(p['C_visual_media|LR']['fit_s_total_4_folds'],2)} s (4 folds, total)", f"{f(p['C_visual_media|LR']['inferencia_ms_por_clip'],3)} ms", "Sí (ResNet-18)", "Media"),
           ("SimBA-RF (160 features)", f"500 árboles; ≈ {p['B_simba_propio|RF']['nodos_totales_por_fold'][0]:,} nodos".replace(",", " "), f"{f(p['B_simba_propio|RF']['fit_s_total_4_folds'],1)} s (4 folds, total)", f"{f(p['B_simba_propio|RF']['inferencia_ms_por_clip'],2)} ms", "No (solo DeepLabCut)", "Media-alta"),
           ("SimBA LogReg", "483", f"{f(p['B_simba_propio|LR']['fit_s_total_4_folds'],2)} s", f"{f(p['B_simba_propio|LR']['inferencia_ms_por_clip'],3)} ms", "No (solo DeepLabCut)", "Alta")]
    return md(["Modelo", "Tamaño", "Entrenamiento", "Inferencia por clip", "¿Necesita red visual?", "Interpretabilidad"], fil) + "\n\n*No medido: el costo de DeepLabCut (≈ 1 h para todo el dataset según `equinos/pose.py`), común a ambos enfoques, ni el de la ResNet-18.*"


TABLAS = {k[2:]: v for k, v in globals().items() if k.startswith("t_") and callable(v)}


def conteo(texto):
    m = re.search(r"<!--MATRIZ-->(.*?)<!--/MATRIZ-->", texto, re.S)
    filas = [l for l in m.group(1).splitlines() if l.startswith("| R")]
    est = Counter()
    for l in filas:
        c = [x.strip() for x in l.strip().strip("|").split("|")]
        e = re.sub(r"[*`]", "", c[3])
        est[e] += 1
    orden = ["CUMPLIDO", "CUMPLIDO PARCIALMENTE", "NO VERIFICABLE", "NO CUMPLIDO"]
    assert sum(est[o] for o in orden) == len(filas), est
    return len(filas), est, orden


def expandir(texto, ctx):
    def tabla(m):
        n = m.group(1).strip()
        if n not in TABLAS:
            raise KeyError(f"tabla desconocida: {n}")
        return TABLAS[n]()

    def fig(m):
        arch, pie = m.group(1), m.group(2)
        assert (FIGURAS / arch).exists(), arch
        return f"![{pie}](../figuras/{arch})\n\n*{pie}*"

    def val(m):
        arch, ruta, dec = m.group(1), m.group(2), m.group(3)
        o = J(arch)
        for p in ruta.split("."):
            o = o[int(p)] if isinstance(o, list) else o[p]
        return f(o, int(dec or 3))

    def cont(m):
        n, est, orden = ctx["conteo"]
        fil = [(o, est[o]) for o in orden] + [("**Total**", n)]
        return md(["Estado", "Requisitos"], fil)

    texto = re.sub(r"\{\{TABLA:([a-z0-9_]+)\}\}", tabla, texto)
    texto = re.sub(r"\{\{FIG:([^|}]+)\|([^}]+)\}\}", fig, texto)
    texto = re.sub(r"\{\{VAL:([^:}]+):([^|}]+)\|?(\d*)\}\}", val, texto)
    texto = re.sub(r"\{\{CONTEO:matriz\}\}", cont, texto)
    return texto


def main():
    fuentes = sorted(FUENTE.glob("*.md"))
    ctx = {}
    for p in fuentes:
        t = p.read_text(encoding="utf-8")
        if "<!--MATRIZ-->" in t:
            ctx["conteo"] = conteo(t)
    for p in fuentes:
        t = expandir(p.read_text(encoding="utf-8"), ctx)
        (AUDS / p.name).write_text(t, encoding="utf-8")
        print("ensamblado", p.name, len(t))
    if "conteo" in ctx:
        guardar_json({"total": ctx["conteo"][0], "conteo": dict(ctx["conteo"][1])}, "conteo_matriz.json")


if __name__ == "__main__":
    main()
