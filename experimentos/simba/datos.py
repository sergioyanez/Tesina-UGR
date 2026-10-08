"""Carga de datos del experimento SimBA-like. SOLO LECTURA sobre `datos_equinos/` del repo principal.

La raíz de datos sale de la variable de entorno `DATOS_EQUINOS` (por defecto, la del repo
principal: el worktree no tiene los datos porque están en .gitignore). Nada de este módulo escribe
fuera de `experimentos/simba/resultados/`.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
RAIZ_WT = AQUI.parents[1]
if str(RAIZ_WT) not in sys.path:
    sys.path.insert(0, str(RAIZ_WT))

REPO_PRINCIPAL = Path(
    "/media/sergio/Expansion/UGR/Licenciatura en Ciencias de datos/cuatrimestre3/materias/"
    "seminario trabajo final/Tesina/expresiones_faciales_equinos")
DATOS = Path(os.environ.get("DATOS_EQUINOS", REPO_PRINCIPAL / "datos_equinos"))
RES_DATOS = DATOS / "resultados"
SALIDA = AQUI / "resultados"
SALIDA.mkdir(exist_ok=True)

CLASES = ["Disgustado", "Interesado", "Sereno"]
CONF_MIN = 0.50                     # equinos.encuadre.CONF_MINIMA_KEYPOINT
ARCHIVO_POSE = "pose_seleccionada_v2.csv"   # los 16 frames por clip que ve el MIL, 39 bodyparts

BODYPARTS = ['nose', 'upper_jaw', 'lower_jaw', 'mouth_end_right', 'mouth_end_left', 'right_eye',
             'right_earbase', 'right_earend', 'right_antler_base', 'right_antler_end', 'left_eye',
             'left_earbase', 'left_earend', 'left_antler_base', 'left_antler_end', 'neck_base',
             'neck_end', 'throat_base', 'throat_end', 'back_base', 'back_end', 'back_middle',
             'tail_base', 'tail_end', 'front_left_thai', 'front_left_knee', 'front_left_paw',
             'front_right_thai', 'front_right_knee', 'front_right_paw', 'back_left_paw',
             'back_left_thai', 'back_right_thai', 'back_left_knee', 'back_right_knee',
             'back_right_paw', 'belly_bottom', 'body_middle_right', 'body_middle_left']
BP = {n: i for i, n in enumerate(BODYPARTS)}


def clips_con_fold():
    """Clips con fold, tal como los usa el MIL vigente (`particion_clips.json`), orden estable."""
    d = json.loads((RES_DATOS / "particion_clips.json").read_text(encoding="utf-8"))
    return d["clips"]


def pose_por_clip(clips):
    """`{clip: array (16, 39, 3)}` con (x, y, confianza) en píxeles del cuadro ORIGINAL."""
    import pandas as pd
    p = pd.read_csv(RES_DATOS / ARCHIVO_POSE)
    idx = p["bodypart"].map(BP).to_numpy()
    salida = {}
    nombres = {c["clip"] for c in clips}
    p = p[p["clip"].isin(nombres)]
    for clip, g in p.groupby("clip", sort=False):
        a = np.full((16, len(BODYPARTS), 3), np.nan)
        a[g["orden"].to_numpy(), g["bodypart"].map(BP).to_numpy()] = g[["x", "y", "confianza"]].to_numpy()
        salida[clip] = a
    faltan = nombres - set(salida)
    assert not faltan, f"clips sin pose: {len(faltan)}"
    return salida


def tiempos_por_clip():
    """`{clip: (frames_de_video (16,), fps, span_seg)}` desde el dataset de frames alineados v2."""
    import pandas as pd
    d = pd.read_csv(RES_DATOS / "dataset_frames_alineados_v2.csv")
    salida = {}
    for clip, g in d.groupby("clip", sort=False):
        g = g.sort_values("orden")
        salida[clip] = (g["frame"].to_numpy(dtype=float), float(g["fps"].iloc[0]),
                        float(g["span_seg"].iloc[0]))
    return salida


def metadatos(clips):
    """Duración, fps, ancho y alto por clip (inventarios), alineado con `clips`."""
    with (RES_DATOS / "inventario_clips.csv").open(encoding="utf-8") as f:
        ic = {r["clip"]: r for r in csv.DictReader(f)}
    with (RES_DATOS / "inventario_videos.csv").open(encoding="utf-8") as f:
        iv = {r["archivo"]: r for r in csv.DictReader(f)}
    filas = []
    for c in clips:
        r = ic[c["clip"]]
        v = iv[r["video_origen"]]
        filas.append([float(r["duracion_seg"]), float(r["fps"]), float(v["ancho"]), float(v["alto"])])
    return np.array(filas)


def video_origen(clips):
    with (RES_DATOS / "inventario_clips.csv").open(encoding="utf-8") as f:
        ic = {r["clip"]: r["video_origen"] for r in csv.DictReader(f)}
    return [ic[c["clip"]] for c in clips]


def visual(clips):
    """`(n, 16, 512)` características ResNet-18 congeladas (caché vigente), en el orden de `clips`."""
    huella = json.loads((RES_DATOS / "caracteristicas_backbone_v2.json").read_text(encoding="utf-8"))
    datos = np.load(RES_DATOS / "caracteristicas_backbone_v2.npz", allow_pickle=True)["caracteristicas"]
    pos = {c: i for i, c in enumerate(huella["clips"])}
    return np.stack([np.asarray(datos[pos[c["clip"]]], dtype=np.float32) for c in clips])


def prediccion_mil(clips):
    e = json.loads((RES_DATOS / "objetivo3_evaluacion_v2.json").read_text(encoding="utf-8"))
    p = {x["clip"]: x["predicha"] for x in e["predicciones_fuera_de_muestra"]}
    assert all(c["clip"] in p for c in clips)
    return np.array([p[c["clip"]] for c in clips], dtype=object), e


def geometria_actual(clips, pose_filas):
    """Las 9 características geométricas vigentes, por frame: `(n, 16, 9)`.

    Usa EXACTAMENTE `equinos.clasificacion.MultimodalEquineDataset` (sin modificarlo): las 9 son
    [x_ojo, y_ojo, x_hocico, y_hocico, x_oreja, y_oreja] normalizadas por el punto medio ojo-hocico y la distancia ojo-hocico, + 3
    máscaras de validez (ojo, hocico, oreja).
    """
    from equinos import clasificacion, pose
    filas = pose.leer_pose(RES_DATOS / ARCHIVO_POSE)
    vis = np.zeros((len(clips), 16, 1), dtype=np.float32)
    indice = {c["clip"]: i for i, c in enumerate(clips)}
    ds = clasificacion.MultimodalEquineDataset(clips, vis, indice, filas)
    return np.stack([ds[i][1].numpy() for i in range(len(clips))])
