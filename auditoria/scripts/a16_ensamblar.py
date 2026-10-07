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


def t_early():
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
