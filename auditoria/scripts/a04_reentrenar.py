"""Re-entrenamiento con el código VIGENTE (sin modificarlo) para: (1) comprobar que reproduce las
predicciones y los pesos guardados (semilla 42); (2) medir la repetibilidad con otras semillas
(devolución de Juan: «¿cómo está la repetibilidad?»). No escribe nada en el repo principal: los
pesos de la re-ejecución van a un directorio temporal.

Uso: python a04_reentrenar.py [semillas...]    (por defecto 42 43 44 45 46 47)
"""
import hashlib
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from comun import *
from equinos import clasificacion as CL
from equinos import pose

semillas = [int(s) for s in sys.argv[1:]] or [42, 43, 44, 45, 46, 47]
clips = particion()
orden, X = caracteristicas_v2()
assert [c["clip"] for c in clips] == sorted(orden) or True
indice = {c: i for i, c in enumerate(orden)}
# la evaluación vigente usa los clips en el orden de particion.cargar()
pose_filas = pose.leer_pose(RES / "pose_seleccionada_v2.csv")
folds = CL.folds_por_ejemplar(clips)
ev = evaluacion_vigente()
pred_guardada = {p["clip"]: p["predicha"] for p in ev["predicciones_fuera_de_muestra"]}
tmp = Path(tempfile.mkdtemp(prefix="auditoria_pesos_"))
salida = {"gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
          "torch": torch.__version__, "semillas": {}}

def h(sd):
    m = hashlib.sha256()
    for k in sorted(sd):
        m.update(sd[k].detach().cpu().numpy().tobytes())
    return m.hexdigest()[:16]

for s in semillas:
    t0 = time.time()
    r = CL.evaluar(CL.entrenar_mil_multimodal, folds, f"MIL s{s}", vectores=X, indice=indice,
                   pose_filas=pose_filas, dropout=0.3, estrategia_pesos="raiz", semilla=s,
                   pesos_en=lambda i, s=s: tmp / f"s{s}_fold{i}")
    reg = {"f1_macro": r["f1_macro"][0], "f1_desvio": r["f1_macro"][1],
           "balanced_accuracy": r["balanced_accuracy"][0],
           "accuracy": float(np.mean([p["real"] == p["predicha"] for p in r["predicciones"]])),
           "f1_por_fold": [f["f1_macro"] for f in r["por_fold"]],
           "mejor_epoca": [f["mejor_epoca"] for f in r["por_fold"]],
           "epocas_corridas": [f["epocas_corridas"] for f in r["por_fold"]],
           "f1_validacion": [f["f1_validacion"] for f in r["por_fold"]],
           "f1_train": [f["f1_train"] for f in r["por_fold"]],
           "segundos": time.time() - t0}
    if s == 42:
        iguales = sum(pred_guardada[p["clip"]] == p["predicha"] for p in r["predicciones"])
        reg["predicciones_iguales_a_las_guardadas"] = f"{iguales}/{len(r['predicciones'])}"
        reg["curvas_validacion_iguales"] = all(
            [e["f1_val"] for e in f["curva"]] == [e["f1_val"] for e in g["curva"]]
            for f, g in zip(r["por_fold"], ev["por_fold"]))
        reg["curvas_train_iguales"] = all(
            [e["f1_train"] for e in f["curva"]] == [e["f1_train"] for e in g["curva"]]
            for f, g in zip(r["por_fold"], ev["por_fold"]))
        pesos_iguales = []
        for i in range(4):
            a = torch.load(tmp / f"s42_fold{i}.pt", map_location="cpu", weights_only=True)
            b = torch.load(DATOS_REPO / "modelos" / "objetivo3_v2" / f"mil_v2_fold{i}.pt",
                           map_location="cpu", weights_only=True)
            pesos_iguales.append(all(torch.equal(a[k], b[k]) for k in a) and set(a) == set(b))
        reg["pesos_identicos_bit_a_bit_por_fold"] = pesos_iguales
        reg["curvas_completas"] = [f["curva"] for f in r["por_fold"]]
    salida["semillas"][str(s)] = reg
    print(s, {k: v for k, v in reg.items() if k != "curvas_completas"}, flush=True)
    guardar_json(salida, "reentrenamiento_semillas.json")
