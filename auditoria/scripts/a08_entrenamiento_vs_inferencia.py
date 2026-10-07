"""Efecto cuantitativo de una diferencia entre entrenamiento e inferencia que el test de equivalencia
NO cubre: el dataset se construyó desde CLIPS recodificados (cv2.VideoWriter, mp4v), mientras la
interfaz decodifica el VIDEO ORIGINAL. Para una muestra de clips se rehacen los recortes desde el
video fuente (mismos fotogramas, mismas transformadas, mismo viaje por JPEG) y se compara contra las
características guardadas; luego se pasa por el modelo del fold que NO vio al caballo.

No re-ejecuta DeepLabCut (la pose sigue siendo la persistida del clip): el efecto de la pose sobre el
video original no pudo verificarse sin horas de GPU.
"""
import os
import cv2
import numpy as np
import pandas as pd
import torch

from comun import *
from equinos import alineado, caracteristicas as CA, clasificacion as CL, pose, secuencia

os.chdir(WT)   # alineado.parametros() busca el notebook 02 relativo al directorio actual
clips = particion()
clips_inv = pd.DataFrame(leer_csv(RES / "inventario_clips.csv")).set_index("clip")
vid = pd.DataFrame(leer_csv(RES / "inventario_videos.csv")).set_index("archivo")
orden, X = caracteristicas_v2(); indice = {c: i for i, c in enumerate(orden)}
pose_filas = pose.leer_pose(RES / "pose_seleccionada_v2.csv")
pose_cruda = alineado.pose_por_clip(RES / "pose_todos_los_frames.csv")
fr = pd.DataFrame(leer_csv(RES / "dataset_frames_alineados_v2.csv"))
p = alineado.parametros(); geo = alineado.geometria(); destino = alineado.destino_canonico(p)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ext = CA._cargar_backbone(dev)
rng = np.random.default_rng(7)
# muestra estratificada: 14 por clase
muestra = []
for k in CLASES:
    idx = [i for i, c in enumerate(clips) if c["clase"] == k]
    muestra += list(rng.choice(idx, 14, replace=False))
modelos = []
for i in range(4):
    m = CL.MILFusionModel(dropout=0.3).to(dev)
    m.load_state_dict(torch.load(DATOS_REPO / "modelos/objetivo3_v2" / f"mil_v2_fold{i}.pt", map_location=dev, weights_only=True)); m.eval()
    modelos.append(m)

def jpg(im):
    ok, b = cv2.imencode(".jpg", im, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return cv2.imdecode(b, cv2.IMREAD_COLOR)

def feats(recortes):
    lote = np.stack([r[:, :, ::-1] for r in recortes]).astype(np.float32) / 255.0
    t = torch.from_numpy(lote.copy()).permute(0, 3, 1, 2)
    t = (t - torch.tensor(CA.MEDIA_IMAGENET).view(1, 3, 1, 1)) / torch.tensor(CA.DESVIO_IMAGENET).view(1, 3, 1, 1)
    with torch.no_grad():
        return ext(t.to(dev)).flatten(1).cpu().numpy()

filas = []
for i in muestra:
    c = clips[i]; nombre = c["clip"]
    inv = clips_inv.loc[nombre]; v = vid.loc[inv["video_origen"]]
    ruta_src = Path(v["ruta"])
    prep = alineado.preparar_secuencia(pose_cruda[nombre], total=int(inv["frames"]), fps=float(inv["fps"]), p=p, geo=geo,
                                       destino=destino, minimo=secuencia.FRAMES_POR_CLIP)
    cap_clip = cv2.VideoCapture(str(DATOS_REPO / inv["ruta_clip"])); cap_src = cv2.VideoCapture(str(ruta_src))
    rec_src, dif_px = [], []
    ini = int(inv["frame_inicio"])
    for orden_f, M in zip(prep.ordenes, prep.matrices):
        cap_clip.set(cv2.CAP_PROP_POS_FRAMES, int(orden_f)); ok1, im_clip = cap_clip.read()
        cap_src.set(cv2.CAP_PROP_POS_FRAMES, ini + int(orden_f)); ok2, im_src = cap_src.read()
        if not (ok1 and ok2) or im_clip.shape != im_src.shape:
            rec_src = None; break
        dif_px.append(float(np.abs(im_clip.astype(int) - im_src.astype(int)).mean()))
        rec_src.append(jpg(alineado.recortar(im_src, M, destino)))
    cap_clip.release(); cap_src.release()
    if rec_src is None or len(rec_src) != 16:
        filas.append({"clip": nombre, "clase": c["clase"], "ok": False}); continue
    f_src = feats(rec_src); f_tr = np.asarray(X[indice[nombre]])
    rel = float(np.linalg.norm(f_src - f_tr, axis=1).mean() / np.linalg.norm(f_tr, axis=1).mean())
    cos = float(np.mean(np.sum(f_src * f_tr, 1) / (np.linalg.norm(f_src, axis=1) * np.linalg.norm(f_tr, axis=1))))
    ds = CL.MultimodalEquineDataset([c], X, indice, pose_filas)
    vv, vg, _ = ds[0]
    m = modelos[c["fold"]]
    mask = torch.ones(1, 16, dtype=torch.bool, device=dev)
    with torch.no_grad():
        l0 = m(vv.unsqueeze(0).to(dev), vg.unsqueeze(0).to(dev), mask)
        l1 = m(torch.tensor(f_src, dtype=torch.float32).unsqueeze(0).to(dev), vg.unsqueeze(0).to(dev), mask)
    p0, p1 = torch.softmax(l0, 1)[0].cpu().numpy(), torch.softmax(l1, 1)[0].cpu().numpy()
    filas.append({"clip": nombre, "clase": c["clase"], "ok": True, "dif_pixel_media_clip_vs_fuente": float(np.mean(dif_px)),
                  "dif_relativa_features": rel, "coseno_features": cos,
                  "pred_dataset": CL.CLASES[int(p0.argmax())], "pred_fuente": CL.CLASES[int(p1.argmax())],
                  "cambia_prediccion": bool(p0.argmax() != p1.argmax()), "max_dif_prob": float(np.abs(p0 - p1).max())})
d = pd.DataFrame(filas)
ok = d[d.ok]
res = {"muestra": len(d), "comparables": int(len(ok)), "no_comparables": int((~d.ok).sum()),
       "dif_pixel_media_0_255": float(ok.dif_pixel_media_clip_vs_fuente.mean()),
       "dif_relativa_features_media": float(ok.dif_relativa_features.mean()), "dif_relativa_features_max": float(ok.dif_relativa_features.max()),
       "coseno_medio": float(ok.coseno_features.mean()),
       "predicciones_que_cambian": int(ok.cambia_prediccion.sum()), "pct_cambian": float(100 * ok.cambia_prediccion.mean()),
       "max_dif_probabilidad_media": float(ok.max_dif_prob.mean()), "max_dif_probabilidad_max": float(ok.max_dif_prob.max()),
       "nota": ("Misma pose persistida; solo cambia el origen de los píxeles (clip mp4v frente a video original). "
                "Cambiar de predicción no implica error: mide sensibilidad. No cubre diferencias de la pose ni de decodificación "
                "de contenedores .avi/.mkv por búsqueda de fotograma en la interfaz.")}
guardar_json({"resumen": res, "detalle": filas}, "entrenamiento_vs_inferencia.json")
print(json.dumps(res, ensure_ascii=False, indent=1))
