"""Parsimonia del MIL vigente: parámetros, inferencia por clip (CPU/GPU) y tiempo de entrenamiento de 1 fold.

Usa el código existente SIN modificarlo (`equinos.clasificacion`). El entrenamiento se corre UNA vez
(fold 0, sin guardar pesos: no se pasa `guardar_en`) solo para medir el tiempo; no se usa para métricas.
Escribe resultados/parsimonia_mil.json.
"""
import json
import time

import numpy as np
import torch

import datos
from equinos import clasificacion, pose

clips = datos.clips_con_fold()
vis = datos.visual(clips)
indice = {c["clip"]: i for i, c in enumerate(clips)}
filas = pose.leer_pose(datos.RES_DATOS / datos.ARCHIVO_POSE)
ds = clasificacion.MultimodalEquineDataset(clips, vis, indice, filas)

m = clasificacion.MILFusionModel(dropout=0.3)
n_param = int(sum(p.numel() for p in m.parameters()))
out = {"parametros_entrenables_MIL": n_param,
       "parametros_backbone_ResNet18_congelado_no_entrenable": 11176512,
       "nota_backbone": "la caché de características usa ResNet-18 ImageNet congelada; extraerla cuesta una pasada GPU por frame y NO está incluida en la inferencia/entrenamiento aquí medidos"}

pesos = datos.RES_DATOS.parent / "modelos" / "objetivo3_v2" / "mil_v2_fold0.pt"
try:
    sd = torch.load(pesos, map_location="cpu")
    sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
    m.load_state_dict(sd)
    out["pesos_fold0_cargados"] = True
except Exception as e:  # la medición de tiempo no depende de los pesos
    out["pesos_fold0_cargados"] = f"No pudo verificarse: {type(e).__name__}"
m.eval()
items = [ds[i] for i in range(60)]
for disp in ("cpu", "cuda"):
    if disp == "cuda" and not torch.cuda.is_available():
        continue
    mm = m.to(disp)
    with torch.no_grad():
        for _ in range(5):
            v, g, _ = items[0]; mm(v[None].to(disp), g[None].to(disp), torch.ones(1, v.shape[0], dtype=torch.bool, device=disp))
        if disp == "cuda": torch.cuda.synchronize()
        t = time.perf_counter()
        for v, g, _ in items:
            mm(v[None].to(disp), g[None].to(disp), torch.ones(1, v.shape[0], dtype=torch.bool, device=disp))
        if disp == "cuda": torch.cuda.synchronize()
    out[f"inferencia_ms_por_clip_{disp}_desde_caracteristicas_cacheadas"] = (time.perf_counter() - t) / len(items) * 1000

train = [c for c in clips if c["fold"] != 0]; test = [c for c in clips if c["fold"] == 0]
t = time.perf_counter()
pred, extra = clasificacion.entrenar_mil_multimodal(train, test, vis, indice, filas, dropout=0.3, estrategia_pesos="raiz",
                                                    max_epocas=100, paciencia=10)
out["entrenamiento_fold0_s"] = time.perf_counter() - t
out["entrenamiento_fold0_epocas"] = extra.get("epocas_corridas")
out["entrenamiento_fold0_dispositivo"] = "cuda" if torch.cuda.is_available() else "cpu"
out["entrenamiento_nota"] = "una sola corrida (no repetida), RTX 3070, desde características cacheadas; sin extracción visual"
(datos.SALIDA / "parsimonia_mil.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(out, indent=1))
