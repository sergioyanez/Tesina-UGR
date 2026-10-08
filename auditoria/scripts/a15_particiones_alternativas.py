"""Sensibilidad a la partición: se repite la validación cruzada (mismo código de entrenamiento, semilla 42)
con 4 asignaciones alternativas de caballos a folds (orden aleatorio, mismo reparto voraz por carga) y se
compara el MIL con la regresión logística sobre la media de los 16 frames en CADA partición.
No toca el código ni los datos; la partición vigente no se modifica."""
import statistics
from collections import defaultdict

import numpy as np

from comun import *
from equinos import clasificacion as CL
from equinos import lineas_de_base as LB
from equinos import pose

clips = particion()
orden, X = caracteristicas_v2(); indice = {c: i for i, c in enumerate(orden)}
pose_filas = pose.leer_pose(RES / "pose_seleccionada_v2.csv")
E = LB.entradas(clips, X, indice, pose_filas)
y = np.array([c["clase"] for c in clips]); ids = np.array([c["id_caballo"] for c in clips])

def particion_aleatoria(semilla, k=4):
    rng = np.random.default_rng(semilla)
    por = defaultdict(list)
    for i, h in enumerate(ids):
        por[h].append(i)
    horses = list(por); rng.shuffle(horses)
    horses.sort(key=lambda h: -len(por[h]))           # voraz como GroupKFold, pero con desempate aleatorio
    # desempate aleatorio dentro de igual tamaño
    horses = sorted(horses, key=lambda h: (-len(por[h]), rng.random()))
    carga = [0] * k; fold = np.zeros(len(ids), dtype=int)
    for h in horses:
        d = min(range(k), key=lambda f: (carga[f], rng.random()))
        for i in por[h]:
            fold[i] = d
        carga[d] += len(por[h])
    return fold

salida = {"particiones": []}
for s in (101, 102, 103, 104):
    fold = particion_aleatoria(s)
    cs = [dict(c, fold=int(fold[i])) for i, c in enumerate(clips)]
    for k in range(4):
        a = {c["id_caballo"] for c in cs if c["fold"] == k}; b = {c["id_caballo"] for c in cs if c["fold"] != k}
        assert not (a & b)
    r = CL.evaluar(CL.entrenar_mil_multimodal, CL.folds_por_ejemplar(cs), f"MIL p{s}", vectores=X, indice=indice,
                   pose_filas=pose_filas, dropout=0.3, estrategia_pesos="raiz")
    p_est = LB.predecir_lineal(E["visual_media"], y, fold)
    f_est = [CL.f1_macro(y[fold == k].tolist(), p_est[fold == k].tolist()) for k in range(4)]
    f_mil = [f["f1_macro"] for f in r["por_fold"]]
    reg = {"semilla_de_particion": s, "tamano_folds": [int((fold == k).sum()) for k in range(4)],
           "mil_f1_media": statistics.fmean(f_mil), "mil_f1_desvio": statistics.stdev(f_mil), "mil_f1_por_fold": f_mil,
           "estatica_f1_media": statistics.fmean(f_est), "estatica_f1_desvio": statistics.stdev(f_est), "estatica_f1_por_fold": f_est,
           "diferencia_mil_menos_estatica": statistics.fmean(f_mil) - statistics.fmean(f_est),
           "mejor_epoca": [f["mejor_epoca"] for f in r["por_fold"]],
           "accuracy_mil": float(np.mean([p["real"] == p["predicha"] for p in r["predicciones"]]))}
    salida["particiones"].append(reg)
    guardar_json(salida, "particiones_alternativas.json")
    print(reg, flush=True)
d = [p["diferencia_mil_menos_estatica"] for p in salida["particiones"]]
salida["resumen"] = {"mil_media_de_medias": statistics.fmean(p["mil_f1_media"] for p in salida["particiones"]),
                     "estatica_media_de_medias": statistics.fmean(p["estatica_f1_media"] for p in salida["particiones"]),
                     "dif_media": statistics.fmean(d), "dif_min": min(d), "dif_max": max(d)}
guardar_json(salida, "particiones_alternativas.json")
print(salida["resumen"])
