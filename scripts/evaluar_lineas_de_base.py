"""Líneas de base del conjunto vigente, con el MIL al lado, y el control del confusor de filmación.

Mide sobre los MISMOS folds por caballo que `scripts/evaluar_v2.py` y persiste
`objetivo3_lineas_de_base.json`. El MIL no se re-entrena: se leen sus predicciones fuera de
muestra de `objetivo3_evaluacion_v2.json`, así que este script mide contra exactamente el modelo
que sirve la interfaz. Hay que correr `evaluar_v2.py` antes.

No escribe pesos ni toca `objetivo3_evaluacion_v2.json` ni `mil_multimodal_produccion.pt`.

Uso:
    PYTHONPATH="$PWD" python3 scripts/evaluar_lineas_de_base.py
"""
import json
import platform

import numpy as np

from equinos import caracteristicas, clasificacion, particion, pose, rutas
from equinos import dataset as _ds
from equinos import lineas_de_base as lb

MIL = "MIL multimodal (vigente)"
ESTATICA = "Visual estática: media de 16 frames (LogReg)"

rutas.inicializar(verbose=False)
resultados = rutas.carpeta("resultados")

clips = particion.cargar()
orden, X = caracteristicas.extraer(clips, verbose=False)
indice = {c: i for i, c in enumerate(orden)}
pose_filas = pose.leer_pose(resultados / _ds.ARCHIVO_POSE_SELECCIONADA)

y = np.array([c["clase"] for c in clips])
fold = np.array([c["fold"] for c in clips])
caballos = np.array([c["id_caballo"] for c in clips])

vigente = json.loads((resultados / clasificacion.ARCHIVO_EVALUACION).read_text(encoding="utf-8"))
pred_mil = {p["clip"]: p["predicha"] for p in vigente["predicciones_fuera_de_muestra"]}
faltan = [c["clip"] for c in clips if c["clip"] not in pred_mil]
assert not faltan, (f"{len(faltan)} clips sin predicción del MIL: la evaluación vigente no "
                    f"corresponde a la partición vigente. Correr scripts/evaluar_v2.py primero.")

E = lb.entradas(clips, X, indice, pose_filas)
predicciones = {
    "Clase mayoritaria (del train del fold)": lb.predecir_mayoritaria(y, fold),
    "Solo duración del clip (LogReg)": lb.predecir_lineal(E["duracion"], y, fold),
    "Metadatos de filmación (LogReg)": lb.predecir_lineal(E["metadatos"], y, fold),
    "Geometría sola: media y desvío (LogReg)": lb.predecir_lineal(E["geometria"], y, fold),
    "Visual: 1 frame central (LogReg)": lb.predecir_lineal(E["visual_un_frame"], y, fold),
    ESTATICA: lb.predecir_lineal(E["visual_media"], y, fold),
    MIL: np.array([pred_mil[c["clip"]] for c in clips], dtype=object),
}

tabla = {nombre: lb.resumen(y, p, fold) for nombre, p in predicciones.items()}
tabla["Azar estratificado (esperanza)"] = lb.azar_estratificado(y, fold)

identidad = clasificacion.evaluar(clasificacion.predecir_por_identidad,
                                  clasificacion.folds_por_ejemplar(clips), "identidad")
tabla["Solo id_caballo (partición por ejemplar)"] = {
    "f1_macro": identidad["f1_macro"][0], "f1_desvio": identidad["f1_macro"][1],
    "por_fold": [f["f1_macro"] for f in identidad["por_fold"]], "accuracy": None}

bootstrap = lb.bootstrap_por_caballo(
    y, {n: predicciones[n] for n in (MIL, ESTATICA, "Metadatos de filmación (LogReg)",
                                     "Visual: 1 frame central (LogReg)")},
    caballos, referencia=MIL)

mascara = lb.sin_rasgos_delatores(clips, E["metadatos"])
confusor = {
    "criterio": (f"resolución {lb.RESOLUCION_MAYORITARIA[0]}x{lb.RESOLUCION_MAYORITARIA[1]} y "
                 f"duración >= {lb.DURACION_MINIMA_SIN_DELATORES} s"),
    "advertencia": ("Se evalúan predicciones fuera de muestra ya existentes; no se re-entrena. "
                    "El subconjunto conserva pocos clips de Disgustado."),
    MIL: lb.en_subconjunto(y, predicciones[MIL], mascara),
    ESTATICA: lb.en_subconjunto(y, predicciones[ESTATICA], mascara),
    "descriptivos_por_clase": {
        c: {col: [float(np.percentile(E["metadatos"][y == c, j], q)) for q in (25, 50, 75)]
            for j, col in enumerate(lb.COLUMNAS_METADATOS) if col in ("duracion_seg", "ventana_seg")}
        for c in clasificacion.CLASES},
}

import sklearn  # noqa: E402

salida = {
    "clips": len(clips), "ejemplares": len(set(caballos.tolist())), "folds": int(fold.max()) + 1,
    "protocolo": ("mismos folds por caballo que objetivo3_evaluacion_v2.json; lineales con "
                  f"C={lb.C_ALTA_DIMENSION} (>{lb.UMBRAL_ALTA_DIMENSION} dims) o "
                  f"C={lb.C_BAJA_DIMENSION}, class_weight=balanced, estandarización ajustada "
                  "sólo con el train de cada fold; hiperparámetros fijados a priori"),
    "lineas_de_base": tabla,
    "bootstrap_por_caballo": bootstrap,
    "confusor_de_filmacion": confusor,
    "entorno": {"python": platform.python_version(), "numpy": np.__version__,
                "sklearn": sklearn.__version__},
}
ruta = resultados / "objetivo3_lineas_de_base.json"
ruta.write_text(json.dumps(salida, ensure_ascii=False, indent=1), encoding="utf-8")

print(f"{'F1 macro':>16}  {'acc':>6}  modelo")
for nombre, r in sorted(tabla.items(), key=lambda kv: kv[1]["f1_macro"]):
    acc = f"{r['accuracy']:.3f}" if r.get("accuracy") is not None else "   —  "
    print(f"  {r['f1_macro']:.3f} ± {r['f1_desvio']:.3f}  {acc}  {nombre}")
print()
for nombre, r in bootstrap["modelos"].items():
    extra = ""
    if "diferencia_referencia_menos_este" in r:
        d = r["diferencia_referencia_menos_este"]
        extra = f"   MIL − éste: {d['media']:+.3f} [{d['ic95'][0]:+.3f}; {d['ic95'][1]:+.3f}]"
    print(f"  IC95 F1 [{r['f1_ic95'][0]:.3f}; {r['f1_ic95'][1]:.3f}]  {nombre}{extra}")
print()
for nombre in (MIL, ESTATICA):
    s = confusor[nombre]
    print(f"  sin rasgos delatores ({s['clips']} clips {s['clips_por_clase']}): "
          f"F1 {s['f1_macro']:.3f}  acc {s['accuracy']:.3f}  {nombre}")
print(f"\npersistido en {ruta.name}")
