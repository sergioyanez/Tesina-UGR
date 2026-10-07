"""Evaluación comparativa (ablación) sobre los MISMOS folds por caballo del MIL vigente.

Un solo script: verifica folds, entrena RF/LogReg por fold sobre cada conjunto de features, calcula
métricas, gaps, subconjunto limpio, baselines triviales, bootstrap por caballo y tiempos. Guarda:
  resultados/evaluacion.json, resultados/oof_predicciones.csv
No toca equinos/ ni datos_equinos/. Semillas fijas; sin búsqueda de hiperparámetros.

Uso: python experimentos/simba/evaluar.py
"""
import json
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import datos
from equinos import clasificacion, lineas_de_base as lb, particion

warnings.filterwarnings("ignore")
SEMILLA = 0
RF_PARAMS = dict(n_estimators=500, max_features="sqrt", min_samples_leaf=2,
                 class_weight="balanced", random_state=SEMILLA, n_jobs=-1)
CLASES = datos.CLASES


def hacer_modelo(tipo, n_dims):
    imp = SimpleImputer(strategy="median", keep_empty_features=True)
    if tipo == "RF":
        return make_pipeline(imp, RandomForestClassifier(**RF_PARAMS))
    c = lb.C_ALTA_DIMENSION if n_dims > lb.UMBRAL_ALTA_DIMENSION else lb.C_BAJA_DIMENSION
    return make_pipeline(imp, StandardScaler(), LogisticRegression(C=c, max_iter=5000, class_weight="balanced"))


def f1_clases(y, p, clases=CLASES):
    return clasificacion.f1_macro(list(y), list(p), clases=clases)


def prf_por_clase(y, p):
    out = {}
    for c in CLASES:
        vp = int(((y == c) & (p == c)).sum()); fp = int(((y != c) & (p == c)).sum()); fn = int(((y == c) & (p != c)).sum())
        pr = vp / (vp + fp) if vp + fp else 0.0
        rc = vp / (vp + fn) if vp + fn else 0.0
        out[c] = {"precision": pr, "recall": rc, "f1": 2 * pr * rc / (pr + rc) if pr + rc else 0.0,
                  "soporte": int((y == c).sum())}
    return out


def matriz(y, p):
    return [[int(((y == r) & (p == c)).sum()) for c in CLASES] for r in CLASES]


def metricas(y, pred, fold, mascara=None):
    """F1 macro por fold (media/desvío/min/max), accuracy, P/R/F1 por clase y matrices.

    Con `mascara` (subconjunto) el F1 de cada fold se calcula sobre las clases PRESENTES en ese fold
    del subconjunto (una clase ausente no puede puntuar), y se informa cuáles faltan.
    """
    if mascara is None:
        mascara = np.ones(len(y), bool)
    y, pred, fold = y[mascara], pred[mascara], fold[mascara]
    por_fold, faltan = [], {}
    for k in sorted(set(fold.tolist())):
        m = fold == k
        presentes = [c for c in CLASES if (y[m] == c).any()]
        faltan[int(k)] = [c for c in CLASES if c not in presentes]
        por_fold.append(f1_clases(y[m], pred[m], presentes))
    return {
        "n_clips": int(len(y)), "clips_por_clase": {c: int((y == c).sum()) for c in CLASES},
        "f1_macro_por_fold": por_fold, "f1_macro_media": float(np.mean(por_fold)),
        "f1_macro_desvio": float(np.std(por_fold, ddof=1)), "f1_macro_min": float(np.min(por_fold)),
        "f1_macro_max": float(np.max(por_fold)), "f1_macro_agrupado": f1_clases(y, pred),
        "accuracy": float((y == pred).mean()),
        "accuracy_por_fold": [float((y[fold == k] == pred[fold == k]).mean()) for k in sorted(set(fold.tolist()))],
        "por_clase": prf_por_clase(y, pred), "matriz_confusion_suma": matriz(y, pred),
        "matriz_confusion_por_fold": {int(k): matriz(y[fold == k], pred[fold == k]) for k in sorted(set(fold.tolist()))},
        "clases_ausentes_por_fold": faltan,
    }


def oof(X, y, fold, tipo, clips, mascara_train=None, con_gap=False, solo_pred=False):
    """Predicción fuera de muestra por fold. Devuelve (pred, info)."""
    pred = np.empty(len(y), dtype=object)
    info = {"fit_s": [], "pred_s_por_clip": [], "gap": [], "nodos": [], "parametros": []}
    for k in sorted(set(fold.tolist())):
        tr = (fold != k) if mascara_train is None else ((fold != k) & mascara_train)
        te = fold == k if mascara_train is None else ((fold == k) & mascara_train)
        m = hacer_modelo(tipo, X.shape[1])
        t = time.perf_counter(); m.fit(X[tr], y[tr]); info["fit_s"].append(time.perf_counter() - t)
        t = time.perf_counter(); p = m.predict(X[te]); dt = time.perf_counter() - t
        info["pred_s_por_clip"].append(dt / max(te.sum(), 1))
        pred[te] = p
        est = m[-1]
        if tipo == "RF":
            info["nodos"].append(int(sum(e.tree_.node_count for e in est.estimators_)))
        else:
            info["parametros"].append(int(est.coef_.size + est.intercept_.size))
        if con_gap:
            # gap_train_val: corte interno por ejemplar del train del fold (como el MIL), solo diagnóstico
            idx_tr = np.where(tr)[0]
            ent, val = particion.dividir_por_ejemplar([dict(clips[i], _i=i) for i in idx_tr])
            ie, iv = [c["_i"] for c in ent], [c["_i"] for c in val]
            mi = hacer_modelo(tipo, X.shape[1]).fit(X[ie], y[ie])
            f1_ent_i = f1_clases(y[ie], mi.predict(X[ie])); f1_val_i = f1_clases(y[iv], mi.predict(X[iv]))
            f1_tr = f1_clases(y[tr], m.predict(X[tr])); f1_te = f1_clases(y[te], p)
            info["gap"].append({"fold": int(k), "f1_train_interno": f1_ent_i, "f1_validacion_interna": f1_val_i,
                                "gap_train_validacion": f1_ent_i - f1_val_i, "f1_train_completo": f1_tr,
                                "f1_test": f1_te, "gap_train_test": f1_tr - f1_te})
    return pred, info


def main():
    clips = datos.clips_con_fold()
    df = pd.read_csv(datos.SALIDA / "dataset_tabular_simba.csv")
    cols = json.loads((datos.SALIDA / "columnas.json").read_text(encoding="utf-8"))
    assert df["clip_id"].tolist() == [c["clip"] for c in clips]
    y = df["clase"].to_numpy(dtype=object); fold = df["fold"].to_numpy(); caballos = df["caballo"].astype(str).to_numpy()

    # ---- (6) verificación de folds: misma partición que el MIL, clip a clip ----
    part_nueva = particion.construir([{"clip": c["clip"], "id_caballo": c["id_caballo"]} for c in clips])
    assert all(a["fold"] == b["fold"] for a, b in zip(clips, part_nueva)), "particion_clips.json != particion.construir"
    pred_mil, ev = datos.prediccion_mil(clips)
    tam = [f["n"] for f in ev["por_fold"]]
    assert tam == [int((fold == k).sum()) for k in range(4)], "tamaños de test != MIL"
    for k in range(4):   # ningún caballo en train y test
        assert not (set(caballos[fold == k]) & set(caballos[fold != k]))
    mil_por_clip = {x["clip"]: x for x in ev["predicciones_fuera_de_muestra"]}
    assert all(mil_por_clip[c["clip"]]["real"] == c["clase"] and mil_por_clip[c["clip"]]["id_caballo"] == c["id_caballo"] for c in clips)
    verif = {"clips": len(clips), "caballos": len(set(caballos)), "tamanos_test_por_fold": tam,
             "caballos_test_por_fold": [len(set(caballos[fold == k])) for k in range(4)],
             "caballos_test_MIL_por_fold": [f["caballos_test"] for f in ev["por_fold"]],
             "particion_json_igual_a_construir": True, "tamanos_iguales_a_MIL": True,
             "sin_caballo_en_train_y_test": True, "clips_del_MIL_igual_a_dataset": True}
    assert verif["caballos_test_por_fold"] == verif["caballos_test_MIL_por_fold"]

    # ---- conjuntos de features ----
    meta_pre = datos.metadatos(clips)
    nombres = cols["features"]; grupos = cols["grupos"]
    Xs = df[nombres].to_numpy(dtype=float)
    g = np.array([grupos[n] for n in nombres])
    geom = datos.geometria_actual(clips, None)                    # (n,16,9)
    X_geom9 = np.concatenate([geom.mean(1), geom.std(1)], axis=1)  # igual que lb.entradas['geometria']
    vis = datos.visual(clips); X_vis = vis.mean(1)
    real = pd.read_csv(datos.SALIDA / "dataset_tabular_simba_real.csv")
    assert real["clip_id"].tolist() == df["clip_id"].tolist()
    X_real = real.drop(columns="clip_id").to_numpy(dtype=float)
    T = datos.tiempos_por_clip()
    dt_medio = np.array([np.diff(T[c["clip"]][0]).mean() / T[c["clip"]][1] for c in clips])
    X_filmacion = np.column_stack([meta_pre, df["meta_escala_cabeza_px"].to_numpy(), dt_medio,
                                   np.array([T[c["clip"]][2] for c in clips])])
    conjuntos = {
        "K1_control_filmacion": X_filmacion,
        "A_geom9_actual": X_geom9,
        "B_simba_propio": Xs,
        "B_simba_propio_estatico": Xs[:, g == "estatica"],
        "B_simba_propio_dinamico": Xs[:, g == "dinamica"],
        "B_simba_propio_sin_dinamica_de_cabeza": Xs[:, [not (n.startswith("cabeza_velocidad") or n.startswith("cabeza_aceleracion")) for n in nombres]],
        "B_simba_propio_sin_disponibilidad": Xs[:, g != "disponibilidad"],
        "B2_simba_real_movimiento": X_real,
        "C_visual_media": X_vis,
        "D_visual_mas_geom9": np.concatenate([X_vis, X_geom9], axis=1),
        "F_simba_propio_mas_visual": np.concatenate([X_vis, Xs], axis=1),
    }

    meta = datos.metadatos(clips)
    limpio = (meta[:, 2] == 1080) & (meta[:, 3] == 1920) & (meta[:, 0] >= 5.0)
    assert (limpio == df["meta_subconjunto_limpio"].to_numpy()).all()

    modelos, info_modelos, preds = {}, {}, {}
    for nombre, X in conjuntos.items():
        for tipo in ("RF", "LR"):
            clave = f"{nombre}|{tipo}"
            gap = nombre in ("B_simba_propio", "C_visual_media", "A_geom9_actual", "D_visual_mas_geom9")
            p, info = oof(X, y, fold, tipo, clips, con_gap=gap)
            preds[clave] = p; info_modelos[clave] = info
            modelos[clave] = {"n_features": int(X.shape[1]), "original": metricas(y, p, fold),
                              "limpio_evaluado_en_subconjunto": metricas(y, p, fold, limpio)}
            print(f"{clave:55s} F1 {modelos[clave]['original']['f1_macro_media']:.3f} ± {modelos[clave]['original']['f1_macro_desvio']:.3f}"
                  f" | limpio {modelos[clave]['limpio_evaluado_en_subconjunto']['f1_macro_agrupado']:.3f}", flush=True)

    # reproducción del baseline vigente con el código existente (control de equivalencia)
    ref_lr = lb.predecir_lineal(X_vis, y, fold)
    ref = lb.resumen(y, ref_lr, fold)
    ctrl = {"baseline_codigo_existente_f1": ref["f1_macro"], "baseline_codigo_existente_desvio": ref["f1_desvio"],
            "baseline_mi_pipeline_f1": modelos["C_visual_media|LR"]["original"]["f1_macro_media"],
            "coinciden_predicciones": float((ref_lr == preds["C_visual_media|LR"]).mean())}

    # MIL desde artefactos
    preds["E_MIL_multimodal|vigente"] = pred_mil
    modelos["E_MIL_multimodal|vigente"] = {"n_features": None, "original": metricas(y, pred_mil, fold),
                                           "limpio_evaluado_en_subconjunto": metricas(y, pred_mil, fold, limpio),
                                           "fuente": "objetivo3_evaluacion_v2.json (predicciones fuera de muestra, no reentrenado)",
                                           "sobreajuste_artefacto": ev["resumen"]["sobreajuste"]}

    # triviales
    may = lb.predecir_mayoritaria(y, fold)
    preds["T_mayoritaria"] = may
    modelos["T_mayoritaria"] = {"original": metricas(y, may, fold), "limpio_evaluado_en_subconjunto": metricas(y, may, fold, limpio)}
    az = lb.azar_estratificado(y, fold)
    modelos["T_azar_estratificado_esperanza"] = {"f1_macro_media": az["f1_macro"], "f1_macro_desvio_entre_simulaciones": az["f1_desvio"],
                                                 "percentil_95": az["percentil_95"], "simulaciones": lb.SIMULACIONES_AZAR}

    # ---- subconjunto limpio: reentrenado SOLO con clips limpios (mismos folds), baseline y SimBA-RF ----
    reent = {}
    for clave, X, tipo in (("C_visual_media|LR", X_vis, "LR"), ("B_simba_propio|RF", Xs, "RF"), ("A_geom9_actual|RF", X_geom9, "RF")):
        p, _ = oof(X, y, fold, tipo, clips, mascara_train=limpio)
        reent[clave] = metricas(y, np.where(limpio, p, None).astype(object), fold, limpio)
    # ---- bootstrap por caballo (F1 agrupado de la réplica; diferencia pareada) ----
    SB, BL, MI = "B_simba_propio|RF", "C_visual_media|LR", "E_MIL_multimodal|vigente"
    def boot(mask):
        idx = np.where(mask)[0]
        P = {"SimBA-RF": preds[SB][idx], "Baseline": preds[BL][idx], "MIL": preds[MI][idx]}
        return lb.bootstrap_por_caballo(y[idx], P, caballos[idx], referencia="SimBA-RF")
    bootstrap = {"original": boot(np.ones(len(y), bool)), "limpio": boot(limpio),
                 "nota": "F1 macro AGRUPADO por réplica (clips de los caballos remuestreados), diferencia = SimBA-RF menos el otro, "
                         "pareada; 2000 réplicas, semilla 1; remuestrea caballos, no clips. Las predicciones no se reentrenan."}
    # SimBA-RF vs geom9 RF y MIL vs baseline (contexto)
    ctx = lb.bootstrap_por_caballo(y, {"MIL": preds[MI], "Baseline": preds[BL], "SimBA-RF": preds[SB]}, caballos, referencia="MIL")
    bootstrap["referencia_MIL"] = ctx

    # ---- parsimonia (lo medido aquí) ----
    def resumen_info(clave):
        i = info_modelos[clave]
        return {"fit_s_total_4_folds": float(np.sum(i["fit_s"])), "fit_s_por_fold": [float(v) for v in i["fit_s"]],
                "inferencia_ms_por_clip": float(np.mean(i["pred_s_por_clip"]) * 1000),
                "nodos_totales_por_fold": i["nodos"], "parametros": i["parametros"]}
    parsimonia = {c: resumen_info(c) for c in (SB, "B_simba_propio|LR", BL, "A_geom9_actual|RF", "F_simba_propio_mas_visual|RF")}
    gaps = {c: {"por_fold": info_modelos[c]["gap"],
                "gap_train_validacion_medio": float(np.mean([x["gap_train_validacion"] for x in info_modelos[c]["gap"]])),
                "gap_train_test_medio": float(np.mean([x["gap_train_test"] for x in info_modelos[c]["gap"]])),
                "f1_train_completo_medio": float(np.mean([x["f1_train_completo"] for x in info_modelos[c]["gap"]])),
                "f1_validacion_interna_medio": float(np.mean([x["f1_validacion_interna"] for x in info_modelos[c]["gap"]]))}
            for c in info_modelos if info_modelos[c]["gap"]}

    salida = {"protocolo": {
        "unidad": "clip (359 clips íntegros del MIL vigente, 133 caballos)", "folds": 4, "agrupacion": "por caballo (id_caballo)",
        "RF": RF_PARAMS, "LR": "StandardScaler + LogisticRegression(C=0.1 si >50 dims, 1.0 si no; class_weight=balanced), igual que lineas_de_base",
        "imputacion": "mediana ajustada solo con el train de cada fold (pipeline)", "semilla": SEMILLA,
        "sin_busqueda_de_hiperparametros": True,
        "subconjunto_limpio": "resolución 1080x1920 y duración >= 5 s (236 clips)",
        "limpio_nota": "los modelos se entrenan con TODOS los clips del train y se evalúan solo en los clips limpios del test (igual que la auditoría 25/09); aparte, 'reentrenado_en_limpio' entrena también solo con limpios.",
        "denominador": "todas las métricas, sobre los 359 clips (original) o los 236 clips (limpio) fuera de muestra"},
        "verificacion_folds": verif, "control_equivalencia_baseline": ctrl, "modelos": modelos,
        "reentrenado_en_limpio": reent, "bootstrap_por_caballo": bootstrap, "gaps_sobreajuste": gaps,
        "parsimonia_medida": parsimonia, "versiones": {"numpy": np.__version__, "pandas": pd.__version__}}
    (datos.SALIDA / "evaluacion.json").write_text(json.dumps(salida, ensure_ascii=False, indent=1), encoding="utf-8")
    tabla = pd.DataFrame({"clip_id": df["clip_id"], "caballo": caballos, "clase": y, "fold": fold,
                          "limpio": limpio, **{k: v for k, v in preds.items()}})
    tabla.to_csv(datos.SALIDA / "oof_predicciones.csv", index=False)
    print(json.dumps(ctrl)); print(json.dumps(verif))
    for k in ("original", "limpio"):
        for n in ("SimBA-RF", "Baseline", "MIL"):
            pass
    print(json.dumps({k: v["modelos"] for k, v in bootstrap.items() if k in ("original", "limpio")}, ensure_ascii=False)[:3000])


if __name__ == "__main__":
    main()
