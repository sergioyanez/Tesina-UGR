"""Genera `informe/cifras.tex` (macros LaTeX) e `informe/cifras_lista.md` desde los JSON.

Fuentes: `comparacion/resultados/*.json`, `comparacion/*.json`, `comparacion/DECISION.md`.
Verifica (assert) que las cifras coinciden con las que cita DECISION.md. Uso:

    TESINA_EQUINOS_RAIZ=<repo>/datos_equinos /home/sergio/.venvs/tesina-equinos/bin/python informe/generar_cifras.py

(La raíz de datos sólo hace falta para el efecto del par 123/169, que necesita las etiquetas.)
"""
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _comun as c  # noqa: E402
from _comun import (BASE, CLASE_CORTA, CLASES, FUS, MIL, NOMBRE, PAR_PREFIJO, PREFIJO, RF, f1_pooled_54,
                    f1_sesion, intervalo, num, par, pct, sgn)

REG = []   # (macro, valor_tex, valor_legible, fuente, descripcion)


def add(nombre, valor, fuente, desc):
    assert re.fullmatch(r"[A-Za-z]+", nombre), nombre
    assert nombre not in {r[0] for r in REG}, f"macro repetida {nombre}"
    leg = valor.replace(r"\ensuremath{-}", "−").replace(r"\,\%", " %")
    REG.append((nombre, valor, leg, fuente, desc))


def main():
    if not os.environ.get("TESINA_EQUINOS_RAIZ"):
        sys.exit("Falta TESINA_EQUINOS_RAIZ (necesario para las etiquetas del par 123/169).")
    sys.path.insert(0, str(c.RAIZ))
    from equinos import comparacion as cmp_
    datos = cmp_.Datos()
    ids = sorted(x["clip"] for x in cmp_.clips_de_desarrollo(datos.clips, cmp_.cargar_reserva(clips=datos.clips)))
    D = c.cargar()                       # verifica huellas y las cifras principales contra DECISION.md
    dec = c.plano(D["decision"])
    base = c.base
    def en(s):
        assert s in dec, f"DECISION.md no contiene: {s}"
    R = "comparacion/resultados/"

    # ---------- tamaños ----------
    res = D["reserva"]
    s_todo, s_lim = D["sesion"]["intervalos"]["todo"], D["sesion"]["intervalos"]["limpio"]
    n_dev, n_res = s_todo["clips"], res["clips"]
    cab_dev, cab_res = len(D["reserva_caballos"]["desarrollo"]), len(D["reserva_caballos"]["reserva"])
    assert (n_dev, n_res, n_dev + n_res) == (291, 68, 359)
    assert (cab_dev, cab_res, cab_dev + cab_res) == (113, 20, 133) and s_todo["caballos"] == 113 == res["caballos"] + 93
    resumen = json.loads((c.base.CMP / "sesiones_resumen.json").read_text(encoding="utf-8"))
    assert resumen["videos"] == 274
    plan = D["sesion"]["plan"]
    n_ses_eval = len(plan["evaluadas"])
    import csv
    filas = list(csv.DictReader(open(c.base.CMP / "sesiones_de_filmacion.csv", encoding="utf-8")))
    n_ses = len({f["sesion_confirmada"] for f in filas})
    assert (n_ses, n_ses_eval) == (22, 20) and len(filas) == 274
    en("20 sesiones de filmación con clips de desarrollo (de las 22 de la tabla)")
    src_t = "comparacion/resultados/validacion_por_sesion.json; comparacion/reserva_caballos.json"
    add("ClipsTotal", str(n_dev + n_res), src_t, "Clips totales (desarrollo + reserva)")
    add("ClipsDesarrollo", str(n_dev), R + "validacion_por_sesion.json (intervalos.todo.clips)", "Clips de desarrollo")
    add("ClipsReserva", str(n_res), R + "fusion_tardia_pose_visual__reserva.json (clips)", "Clips de la reserva")
    add("CaballosTotal", str(cab_dev + cab_res), "comparacion/reserva_caballos.json", "Caballos totales")
    add("CaballosDesarrollo", str(cab_dev), "comparacion/reserva_caballos.json (desarrollo)", "Caballos de desarrollo")
    add("CaballosReserva", str(cab_res), "comparacion/reserva_caballos.json (reserva)", "Caballos de la reserva")
    add("VideosTotal", str(resumen["videos"]), "comparacion/sesiones_resumen.json", "Videos originales")
    add("SesionesTotal", str(n_ses), "comparacion/sesiones_de_filmacion.csv (sesion_confirmada)", "Sesiones de filmación confirmadas")
    add("SesionesEvaluadas", str(n_ses_eval), R + "validacion_por_sesion.json (plan.evaluadas)", "Sesiones evaluadas (con clips de desarrollo)")
    add("ClipsLimpioDesarrollo", str(s_lim["clips"]), R + "validacion_por_sesion.json (intervalos.limpio)", "Clips del subconjunto limpio en desarrollo")
    add("CaballosLimpioDesarrollo", str(s_lim["caballos"]), R + "validacion_por_sesion.json (intervalos.limpio)", "Caballos del subconjunto limpio en desarrollo")
    m = re.search(r"Subconjunto limpio \((\d+) clips\)", dec)
    assert m
    add("ClipsLimpioReserva", m.group(1), "comparacion/DECISION.md (reserva, subconjunto limpio)", "Clips limpios de la reserva")
    add("CriterioLimpio", "1080\\,$\\times$\\,1920 y duración $\\geq$ 5\\,s", R + "*.json (limpio.criterio)", "Criterio del subconjunto limpio")
    assert D["cand"][FUS]["limpio"]["criterio"] == "ancho 1080, alto 1920, duración >= 5 s"
    for cl in CLASES:
        k = CLASE_CORTA[cl]
        i = CLASES.index(cl)
        fil = D["cand"][FUS]["global"]["matriz_confusion_suma"][i]
        assert sum(fil) % 5 == 0
        add(f"ClipsDev{k}", str(sum(fil) // 5), R + "fusion...json (matriz_confusion_suma / 5)", f"Clips de desarrollo de {cl}")
        add(f"ClipsReserva{k}", str(sum(res["metricas"]["matriz"][i])), R + "fusion...__reserva.json (matriz)", f"Clips de la reserva de {cl}")
        add(f"ClipsLimpio{k}", str(D["cand"][FUS]["limpio"]["clips_por_clase"][cl]), R + "fusion...json (limpio.clips_por_clase)", f"Clips limpios de desarrollo de {cl}")
    add("Repeticiones", str(D["cand"][FUS]["R"]), R + "*.json (R)", "Repeticiones del 5×4")
    add("Folds", str(D["cand"][FUS]["K"]), R + "*.json (K)", "Folds por repetición")
    add("Evaluaciones", str(D["cand"][FUS]["R"] * D["cand"][FUS]["K"]), R + "*.json", "Evaluaciones del 5×4 (R×K)")
    pp = D["pareadas"][0]
    add("ReplicasBootstrap", str(pp["replicas"]), R + "comparaciones_pareadas.json", "Réplicas del bootstrap por caballo")
    add("SemillaBootstrap", str(pp["semilla"]), R + "comparaciones_pareadas.json", "Semilla del bootstrap")
    add("ClipsTrainSesionMin", str(min(r["n_train"] for r in plan["evaluadas"])), R + "validacion_por_sesion.json (plan.evaluadas)", "Mínimo de clips de train por sesión")
    add("ClipsTrainSesionMax", str(max(r["n_train"] for r in plan["evaluadas"])), R + "validacion_por_sesion.json (plan.evaluadas)", "Máximo de clips de train por sesión")
    assert (plan["clips_train_perdidos_total"], plan["clips_train_perdidos_media"]) == (40, 2.0)
    add("ClipsTrainPerdidos", str(plan["clips_train_perdidos_total"]), R + "validacion_por_sesion.json (plan)", "Clips de train quitados por exclusión de caballos (total)")
    add("ClipsTrainPerdidosMedia", num(plan["clips_train_perdidos_media"], 1), R + "validacion_por_sesion.json (plan)", "Ídem, promedio por sesión")
    add("PoseRasgos", "160", R + "fusion...json (descripcion)", "Rasgos de pose")
    assert "160 features de pose" in D["cand"][FUS]["descripcion"]
    add("Fotogramas", "16", "equinos/comparacion.py (Datos: 16 frames por clip)", "Fotogramas seleccionados por clip")

    # ---------- peso w ----------
    en("0,75 en 19 de 20 folds y 1,0 en uno")
    add("PesoPose", "0,75", "comparacion/DECISION.md (peso elegido en validación interna)", "Peso w de la pose en la fusión")
    add("PesoFoldsA", "19", "comparacion/DECISION.md", "Folds en que w = 0,75")
    add("PesoFoldsTotal", "20", "comparacion/DECISION.md", "Folds totales")
    add("PesoPoseUno", "1,0", "comparacion/DECISION.md", "Peso en el fold restante (1 de 20)")

    # ---------- por candidato ----------
    for cand in c.CANDIDATOS:
        P, J = PREFIJO[cand], D["cand"][cand]
        f = f"{R}{cand}.json"
        g, lim = J["global"], J["limpio"]["f1_macro_oof_por_repeticion"]
        add(f"{P}DevF", num(g["f1_macro"]["media"]), f + " (global.f1_macro.media)", f"{NOMBRE[cand]}: F1 macro 5×4 (media de 20 folds)")
        add(f"{P}DevDE", num(g["f1_macro"]["desvio"]), f + " (global.f1_macro.desvio)", "Desvío del F1 macro (20 folds)")
        add(f"{P}DevAcc", pct(g["accuracy"]["media"]), f + " (global.accuracy.media)", "Accuracy 5×4 (porcentaje)")
        add(f"{P}DevAccDE", pct(g["accuracy"]["desvio"]), f + " (global.accuracy.desvio)", "Desvío de la accuracy (porcentaje)")
        add(f"{P}LimpioF", num(lim["media"]), f + " (limpio.f1_macro_oof_por_repeticion.media)", "F1 macro en el subconjunto limpio (media de 5 repeticiones)")
        add(f"{P}LimpioDE", num(lim["desvio"]), f + " (limpio.f1_macro_oof_por_repeticion.desvio)", "Desvío del F1 limpio (5 repeticiones)")
        add(f"{P}Gap", num(g["gap_train_test_f1"]["media"], 2), f + " (global.gap_train_test_f1.media)", "Brecha F1 train − test (2 decimales)")
        add(f"{P}GapDE", num(g["gap_train_test_f1"]["desvio"], 2), f + " (global.gap_train_test_f1.desvio)", "Desvío de la brecha")
        add(f"{P}TrainF", num(g["f1_macro_train"]["media"]), f + " (global.f1_macro_train.media)", "F1 macro en train")
        v, ic_ = f1_pooled_54(D, cand, "todo")
        add(f"{P}AgrupF", num(v), R + "comparaciones_pareadas.json (f1_a/f1_b)", "F1 agrupado fuera de muestra 5×4 (media de 5 repeticiones)")
        add(f"{P}AgrupIC", intervalo(ic_), R + "comparaciones_pareadas.json", "IC95 bootstrap por caballo del F1 agrupado 5×4")
        v, ic_ = f1_pooled_54(D, cand, "limpio")
        add(f"{P}AgrupLimpioF", num(v), R + "comparaciones_pareadas.json", "F1 agrupado 5×4, subconjunto limpio")
        add(f"{P}AgrupLimpioIC", intervalo(ic_), R + "comparaciones_pareadas.json", "IC95 del anterior")
        sv = D["sesion"]["candidatos"][cand]
        v, ic_ = f1_sesion(D, cand, "todo")
        assert abs(v - sv["metricas"]["f1_macro"]) < 1e-12
        add(f"{P}SesionF", num(v), R + "validacion_por_sesion.json (intervalos.todo.f1_agrupado)", "F1 macro agrupado, una sesión afuera")
        add(f"{P}SesionIC", intervalo(ic_), R + "validacion_por_sesion.json", "IC95 bootstrap por caballo")
        add(f"{P}SesionAcc", pct(sv["metricas"]["accuracy"]), R + "validacion_por_sesion.json (candidatos.*.metricas.accuracy)", "Accuracy, una sesión afuera")
        vl, icl = f1_sesion(D, cand, "limpio")
        assert abs(vl - sv["f1_macro_limpio"]) < 1e-12
        add(f"{P}SesionLimpioF", num(vl), R + "validacion_por_sesion.json (intervalos.limpio.f1_agrupado)", "F1 agrupado por sesión, subconjunto limpio")
        add(f"{P}SesionLimpioIC", intervalo(icl), R + "validacion_por_sesion.json", "IC95 del anterior")
        caida = f1_pooled_54(D, cand, "todo")[0] - v
        caida_l = f1_pooled_54(D, cand, "limpio")[0] - vl
        add(f"{P}SesionCaida", sgn(caida), "5×4 agrupado − sesión (calculado)", "Caída del F1 al pasar de 5×4 a sesión (negativo = no cayó)")
        add(f"{P}SesionCaidaLimpio", sgn(caida_l), "ídem, limpio", "Caída en el subconjunto limpio")
        # verificación contra la tabla de sesión de DECISION.md
        en(f"{num(v)} [{num(ic_[0])}; {num(ic_[1])}]")
        en(f"{num(vl)} [{num(icl[0])}; {num(icl[1])}]")
        en(f"{num(f1_pooled_54(D, cand, 'todo')[0])} | {base.sg(caida)}")
        en(f"{num(f1_pooled_54(D, cand, 'limpio')[0])} | {base.sg(caida_l)}")
        for cl in CLASES:
            k = CLASE_CORTA[cl]
            d = g["por_clase"][cl]["f1"]
            add(f"{P}DevF{k}", num(d["media"]), f + f" (global.por_clase.{cl}.f1.media)", f"F1 de {cl} 5×4 (media)")
            add(f"{P}DevF{k}DE", num(d["desvio"]), f + f" (global.por_clase.{cl}.f1.desvio)", f"Desvío del F1 de {cl}")
            add(f"{P}SesionF{k}", num(sv["metricas"]["f1"][cl]), R + f"validacion_por_sesion.json (candidatos.*.metricas.f1.{cl})", f"F1 de {cl}, una sesión afuera")
        en(f"{base.pm(g['f1_macro']['media'], g['f1_macro']['desvio'])} | {base.pm(lim['media'], lim['desvio'])} | +{num(g['gap_train_test_f1']['media'], 2)}")
    fs = D["sesion"]["candidatos"][FUS]["metricas"]["f1"]
    en(f"Disgustado {num(fs['Disgustado'])}, Interesado {num(fs['Interesado'])}, Sereno {num(fs['Sereno'])}")

    # ---------- reserva ----------
    r = res["metricas"]
    P = "FusionReserva"
    f = R + "fusion_tardia_pose_visual__reserva.json"
    # El JSON de la reserva no guarda el IC: se recalcula con el mismo bootstrap (2 000 réplicas, semilla 7)
    sys.path.insert(0, str(c.RAIZ / "comparacion"))
    import validacion_por_sesion as vps
    ids_r = sorted(res["predicciones"])
    ic_raw = vps.bootstrap_por_caballo(
        ids_r, {i: datos.caballo(i) for i in ids_r}, {i: datos.etiqueta(i) for i in ids_r},
        {"f": res["predicciones"]}, set(ids_r))["todo"]["f1_agrupado"]["f"]
    assert abs(ic_raw["valor"] - r["f1_macro"]) < 1e-12
    ic_res = intervalo(ic_raw["ic95"])
    en(f"F1 macro **{num(r['f1_macro'])}**, IC95 bootstrap por caballo {ic_res}")
    add(P + "F", num(r["f1_macro"]), f + " (metricas.f1_macro)", "Fusión en la reserva: F1 macro")
    add(P + "IC", ic_res, f + " (IC95 bootstrap por caballo)", "IC95 del F1 macro de la reserva")
    add(P + "Acc", pct(r["accuracy"]), f + " (metricas.accuracy)", "Accuracy en la reserva (porcentaje)")
    add(P + "AccDec", num(r["accuracy"]), f + " (metricas.accuracy)", "Accuracy en la reserva (decimal)")
    add(P + "Limpio", num(D["reserva"]["limpio_f1_macro"]), f + " (limpio_f1_macro)", "F1 macro de la reserva, subconjunto limpio")
    en(f"F1 macro {num(D['reserva']['limpio_f1_macro'])}")
    for cl in CLASES:
        k = CLASE_CORTA[cl]
        en(f"{cl} {num(r['f1'][cl])}")
        add(f"{P}F{k}", num(r["f1"][cl]), f + f" (metricas.f1.{cl})", f"F1 de {cl} en la reserva")
        add(f"{P}Prec{k}", num(r["precision"][cl]), f + f" (metricas.precision.{cl})", f"Precisión de {cl} en la reserva")
        add(f"{P}Rec{k}", num(r["recall"][cl]), f + f" (metricas.recall.{cl})", f"Recall de {cl} en la reserva")
    en("Matriz de confusión (filas = real; Disgustado, Interesado, Sereno): [16, 1, 0], [3, 12, 1], [1, 4, 30]")
    assert r["matriz"] == [[16, 1, 0], [3, 12, 1], [1, 4, 30]]
    add("ReservaAciertos", str(sum(r["matriz"][i][i] for i in range(3))), f + " (traza de la matriz)", "Clips bien clasificados en la reserva")

    # ---------- diferencias pareadas ----------
    pares = [(FUS, RF), (FUS, MIL), (FUS, BASE), (RF, MIL), (RF, BASE)]
    for a, b in pares:
        N = "Dif" + PAR_PREFIJO[a] + PAR_PREFIJO[b]
        desc = f"{PAR_PREFIJO[a]} − {PAR_PREFIJO[b]}"
        for sub, suf in (("todo", "Dev"), ("limpio", "Limpio")):
            p = par(D, a, b, sub)
            d, ic_ = p["diferencia_a_menos_b"], p["diferencia_ic95"]
            add(N + suf, sgn(d), R + f"comparaciones_pareadas.json ({a} vs {b}, {sub})", f"{desc}, 5×4 {sub}")
            add(N + suf + "IC", intervalo(ic_, signo=True), R + "comparaciones_pareadas.json", f"IC95 de {desc}, 5×4 {sub}")
            en(f"{base.sg(d)} {base.ic(ic_, signo=True)}")
        if a == FUS:
            for sub, suf in (("todo", "Sesion"), ("limpio", "SesionLimpio")):
                x = D["sesion"]["intervalos"][sub]["diferencia_vs_fusion_tardia_pose_visual"][b]
                add(N + suf, sgn(x["diferencia"]), R + f"validacion_por_sesion.json (intervalos.{sub}.diferencia_vs_fusion...)", f"{desc}, una sesión afuera ({sub})")
                add(N + suf + "IC", intervalo(x["ic95"], signo=True), R + "validacion_por_sesion.json", f"IC95 de {desc}, sesión ({sub})")
                en(f"{base.sg(x['diferencia'])} {base.ic(x['ic95'], signo=True)}")

    # ---------- par de identificadores 123/169 ----------
    par_ids = [i for i in ids if datos.caballo(i) in ("123", "169")]
    assert len(par_ids) == 5, par_ids
    resto = [i for i in ids if i not in set(par_ids)]
    add("ParClips", "5", "equinos.comparacion.Datos (clips de los caballos 123 y 169)", "Clips del par 123/169")
    en("Quitar sus 5 clips")
    add("ParRepsSeparado", "4", "comparacion/DECISION.md", "Repeticiones del 5×4 en que el par quedó separado entre train y test (de 5)")
    en("separado entre train y test en 4 de las 5 repeticiones")

    def f1_agr(cand, sel):
        J = D["cand"][cand]
        y = [datos.etiqueta(i) for i in sel]
        return sum(cmp_.f1_macro(y, [J["predicciones_oof"][str(k)][i] for i in sel]) for k in range(5)) / 5
    for cand in c.CANDIDATOS:
        a_, d_ = f1_agr(cand, ids), f1_agr(cand, resto)
        P = "Par" + PAR_PREFIJO[cand]
        add(P + "Antes", num(a_, 4), "calculado de predicciones_oof (291 clips)", f"{NOMBRE[cand]}: F1 agrupado 5×4 con los 291 clips (4 decimales)")
        add(P + "Despues", num(d_, 4), "calculado de predicciones_oof (sin los 5 clips)", "Ídem sin los 5 clips del par")
        add(P + "Dif", sgn(d_ - a_), "calculado", "Cambio al quitar el par")
        en(f"{num(a_, 4)} → {num(d_, 4)} ({base.sg(d_ - a_)})" if False else f"{num(a_, 4)} → {num(d_, 4)}")

    escribir(D)


def escribir(D):
    aqui = Path(__file__).resolve().parent
    L = ["% generado por informe/generar_cifras.py, no editar.",
         "% Cada macro es una cifra del informe, formateada en español (coma decimal). Usar como \\Macro{} o \\Macro\\ ",
         "% (con llaves vacías para que no se coma el espacio siguiente).",
         "% Correspondencia macro = fuente (también en informe/cifras_lista.md):"]
    for n_, v, leg, fu, de in REG:
        L.append(f"%   \\{n_} = {fu}")
    L.append("")
    for n_, v, leg, fu, de in REG:
        L.append(f"\\newcommand{{\\{n_}}}{{{v}}}")
    (aqui / "cifras.tex").write_text("\n".join(L) + "\n", encoding="utf-8")
    M = ["# Cifras del informe (generado por `informe/generar_cifras.py`, no editar)", "",
         "Cada fila es una macro de `cifras.tex`. Usarla como `\\Macro{}`. Las cifras no se escriben a mano en el `.tex`.", "",
         "| Macro | Valor | Qué es | Fuente |", "|---|---|---|---|"]
    for n_, v, leg, fu, de in REG:
        M.append(f"| `\\{n_}` | {leg} | {de} | {fu} |")
    (aqui / "cifras_lista.md").write_text("\n".join(M) + "\n", encoding="utf-8")
    print(f"{len(REG)} macros -> informe/cifras.tex, informe/cifras_lista.md")


if __name__ == "__main__":
    main()
