"""Evaluación final del pipeline v2 y entrenamiento del modelo que usa la interfaz.

Corre la validación cruzada por ejemplar —con corte interno y early stopping dentro de cada
fold— persiste el resultado en `objetivo3_evaluacion_v2.json`, que es de donde la app lee TODAS
sus cifras, y entrena el modelo de producción sobre el conjunto completo.

Las épocas de producción no se eligen a mano: salen de la media de la mejor época medida en la
validación cruzada. Es la única estimación honesta disponible de cuánto conviene entrenar con
estos datos, y evita volver al problema de las 50 épocas fijas.

Uso:
    PYTHONPATH="$PWD" python3 scripts/evaluar_v2.py
"""
import json, math, platform, statistics, sys
from equinos import rutas, particion, caracteristicas, clasificacion, pose, secuencia

DROPOUT = 0.3            # elegido por validación interna
PESOS = "raiz"           # ídem: raíz de la frecuencia inversa contra frecuencia inversa
rutas.inicializar(verbose=False)

clips = particion.cargar()
orden, X = caracteristicas.extraer(clips, verbose=False)
indice = {c: i for i, c in enumerate(orden)}
from equinos import dataset as _ds
pose_filas = pose.leer_pose(rutas.carpeta("resultados") / _ds.ARCHIVO_POSE_SELECCIONADA)
folds = clasificacion.folds_por_ejemplar(clips)

r = clasificacion.evaluar(
    clasificacion.entrenar_mil_multimodal, folds,
    "MIL multimodal v2", vectores=X, indice=indice, pose_filas=pose_filas, dropout=DROPOUT, estrategia_pesos=PESOS,
    pesos_en=lambda i: rutas.carpeta("modelos") / "objetivo3_v2" / f"mil_v2_fold{i}")

pred = r["predicciones"]
def por_clase():
    salida = {}
    for c in clasificacion.CLASES:
        dijo = [x for x in pred if x["predicha"] == c]
        real = [x for x in pred if x["real"] == c]
        ac = sum(1 for x in dijo if x["real"] == c)
        pr = ac / len(dijo) if dijo else 0.0
        rc = ac / len(real) if real else 0.0
        salida[c] = {"precision": pr, "recall": rc,
                     "f1": 2 * pr * rc / (pr + rc) if (pr + rc) else 0.0,
                     "support": len(real), "acertadas": ac, "veces_que_lo_dijo": len(dijo)}
    return salida

pc = por_clase()
matriz = {real: {p: sum(1 for x in pred if x["real"] == real and x["predicha"] == p)
                 for p in clasificacion.CLASES} for real in clasificacion.CLASES}
epocas = [f["mejor_epoca"] for f in r["por_fold"]]

# El sobreajuste, medido en la época elegida. No existe un umbral universal: el gap se lee junto
# con las curvas y con que el test acompañe a la validación (docs/analisis_sobreajuste.md).
sobreajuste_por_fold = [{
    "fold": f["fold"], "mejor_epoca": f["mejor_epoca"], "f1_train": f["f1_train"],
    "f1_validacion": f["f1_validacion"], "f1_test": f["f1_macro"],
    "gap_train_validacion": f["f1_train"] - f["f1_validacion"],
    "gap_train_test": f["f1_train"] - f["f1_macro"],
    "epoca_de_minima_loss_val": min(f["curva"], key=lambda e: e["loss_val"])["epoca"],
} for f in r["por_fold"]]

import torch, numpy
resumen = {
    "modelo": "MIL multimodal (MILFusionModel)",
    "clips": len(clips),
    "ejemplares": len({c["id_caballo"] for c in clips}),
    "folds": len(folds),
    "protocolo": ("validación cruzada agrupada por ejemplar; dentro de cada fold, corte interno "
                  "train/validación también por ejemplar, early stopping sobre F1 macro de "
                  "validación; el test se evalúa una sola vez con el mejor checkpoint"),
    "f1_macro": r["f1_macro"][0], "f1_macro_desvio": r["f1_macro"][1],
    "balanced_accuracy": r["balanced_accuracy"][0],
    "accuracy": sum(1 for x in pred if x["real"] == x["predicha"]) / len(pred),
    "precision_macro": statistics.mean(v["precision"] for v in pc.values()),
    "recall_macro": statistics.mean(v["recall"] for v in pc.values()),
    "mejor_epoca_media": statistics.mean(epocas),
    "mejor_epoca_desvio": statistics.stdev(epocas),
    "sobreajuste": {
        "f1_train_media": statistics.mean(x["f1_train"] for x in sobreajuste_por_fold),
        "f1_validacion_media": statistics.mean(x["f1_validacion"] for x in sobreajuste_por_fold),
        "gap_train_validacion_medio": statistics.mean(
            x["gap_train_validacion"] for x in sobreajuste_por_fold),
        "gap_train_test_medio": statistics.mean(x["gap_train_test"] for x in sobreajuste_por_fold),
        "por_fold": sobreajuste_por_fold,
    },
    "dropout": DROPOUT,
    "estrategia_pesos": PESOS,
    "semilla": clasificacion.SEMILLA,
    "seleccion_de_frames": {
        "criterio": "ventana temporal contigua con anclaje al frame mejor medido",
        "frames_por_clip": secuencia.FRAMES_POR_CLIP,
        "ventana_maxima_seg": secuencia.VENTANA_MAXIMA_SEG,
        "hueco_maximo_seg": secuencia.HUECO_MAXIMO_SEG,
        "tolerancia_anclaje_seg": secuencia.TOLERANCIA_ANCLAJE_SEG,
    },
    "entorno": {"python": platform.python_version(), "torch": torch.__version__,
                "numpy": numpy.__version__},
}

salida = {
    "resumen": resumen,
    "por_fold": r["por_fold"],
    "por_clase": pc,
    "matriz_de_confusion": matriz,
    "predicciones_fuera_de_muestra": pred,
}
ruta = rutas.carpeta("resultados") / clasificacion.ARCHIVO_EVALUACION
ruta.write_text(json.dumps(salida, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"evaluación persistida en {ruta.name}")

# --- modelo de producción: todas las clips, épocas = la media medida en CV -------------------
epocas_produccion = math.ceil(statistics.mean(epocas))
modelo, proc = clasificacion.entrenar_produccion(
    clips, X, indice, pose_filas, metrica_validacion=resumen["f1_macro"],
    epocas=epocas_produccion, dropout=DROPOUT, estrategia_pesos=PESOS, verbose=True)
print(f"\népocas de producción: {epocas_produccion} (media medida {statistics.mean(epocas):.1f})")
