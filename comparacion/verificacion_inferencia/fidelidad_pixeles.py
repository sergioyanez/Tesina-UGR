"""Aísla el efecto de la LECTURA del video (orientación, mp4v) sin DeepLabCut.

Para cada uno de los 16 videos usa la pose y los frames YA elegidos por el entrenamiento (caché)
y reconstruye los recortes desde cuatro fuentes:
  A  el clip del dataset (control: debe dar la caché exacta);
  B  el original normalizado como lo hace la inferencia nueva (ORIENTATION_AUTO + mp4v);
  C  el original con ORIENTATION_AUTO pero sin recodificar;
  D  el original leído como lo hacía la inferencia anterior (sin orientación, sin mp4v).
Compara las características con la caché y la predicción del modelo desplegado.
"""
import csv, json, os, shutil, sys, tempfile
from pathlib import Path
import numpy as np

WT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WT))
import cv2  # noqa: E402
from equinos import alineado, clasificacion, inferencia, pose as sel_pose, video  # noqa: E402

RES = Path(os.environ["TESINA_EQUINOS_RAIZ"]) / "resultados"
RAIZ = Path(os.environ["TESINA_EQUINOS_RAIZ"])
leer = lambda n: list(csv.DictReader((RES / n).open(encoding="utf-8", errors="replace")))  # noqa: E731
inv = {r["archivo"]: r for r in leer("inventario_videos.csv")}
icl = leer("inventario_clips.csv"); ds = leer("dataset_frames_alineados_v2.csv"); pose_sel = leer("pose_seleccionada_v2.csv")
feats = np.load(RES / "caracteristicas_backbone_v2.npz", allow_pickle=True)["caracteristicas"]
nombres = json.load(open(RES / "caracteristicas_backbone_v2.json"))["clips"]
videos = [x["video"] for x in json.load(open(WT / "auditoria/datos/orientacion_inferencia.json"))["pruebas_extremo_a_extremo"]]
p = inferencia.parametros_etapa02(); geo = alineado.geometria(); destino = alineado.destino_canonico(p)
modelo = clasificacion.cargar_produccion()
cos = lambda a, b: float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))  # noqa: E731

def recortes(capturar, idxs, matrices):
    out = []
    for i, m in zip(idxs, matrices):
        capturar.set(cv2.CAP_PROP_POS_FRAMES, int(i)); ok, img = capturar.read()
        if not ok: return None
        out.append(alineado.recortar(img, m, destino))
    return out

res = []
tmp = Path(tempfile.mkdtemp(prefix="fid_"))
try:
    for v in videos:
        clip = next(c for c in icl if c["video_origen"] == v); cn = clip["clip"]
        if cn not in nombres: continue
        dfr = sorted((r for r in ds if r["clip"] == cn), key=lambda r: int(r["orden"]))
        idxs = [int(clip["frame_inicio"]) + int(r["frame"]) for r in dfr]
        ps = [r for r in pose_sel if r["clip"] == cn]
        pares = {int(x["orden"]): x for x in sel_pose.seleccionar_par(ps, bodypart_hocico=p["BODYPART_HOCICO"])}
        mats = [geo["similitud_desde_dos_puntos"]((float(pares[k]["x_ojo"]), float(pares[k]["y_ojo"])),
                (float(pares[k]["x_hocico"]), float(pares[k]["y_hocico"])), destino["dest_ojo"], destino["dest_hocico"])
                for k in range(16)]
        cf = np.asarray(feats[nombres.index(cn)]); pc = inferencia.predecir_desde_caracteristicas(cf, ps, modelo=modelo).clase
        orig = inv[v]["ruta"]; norm = tmp / "n.mp4"; meta = video.normalizar_video(orig, norm)
        fuentes = {"A_clip_dataset": cv2.VideoCapture(str(RAIZ / clip["ruta_clip"])),
                   "B_normalizado_mp4v": cv2.VideoCapture(str(norm)),
                   "C_orientado_sin_recodificar": video.abrir_video(orig),
                   "D_original_crudo": cv2.VideoCapture(str(orig))}
        reg = {"video": v, "clase": inv[v]["clase"], "rotacion": meta["rotacion"], "pred_cache": pc}
        for k, cap in fuentes.items():
            rc = recortes(cap, idxs, mats); cap.release()
            if rc is None: reg[k] = {"error": "lectura incompleta"}; continue
            f = inferencia._caracteristicas_de(rc)
            pr = inferencia.predecir_desde_caracteristicas(f, ps, modelo=modelo).clase
            reg[k] = {"coseno": float(np.mean([cos(a, b) for a, b in zip(f, cf)])),
                      "dif_rel": float(np.mean([np.linalg.norm(a - b) / np.linalg.norm(b) for a, b in zip(f, cf)])),
                      "pred": pr, "igual_cache": pr == pc}
        res.append(reg); print(json.dumps(reg, ensure_ascii=False), flush=True)
finally:
    shutil.rmtree(tmp, ignore_errors=True)
resumen = {k: {"coseno_medio": float(np.mean([r[k]["coseno"] for r in res if "coseno" in r[k]])),
               "dif_rel_media": float(np.mean([r[k]["dif_rel"] for r in res if "coseno" in r[k]])),
               "misma_pred_que_cache": sum(r[k].get("igual_cache", False) for r in res), "n": len(res)}
           for k in ("A_clip_dataset", "B_normalizado_mp4v", "C_orientado_sin_recodificar", "D_original_crudo")}
rot = [r for r in res if r["rotacion"]]; rect = [r for r in res if not r["rotacion"]]
resumen["por_rotacion"] = {g: {k: float(np.mean([r[k]["coseno"] for r in rs if "coseno" in r[k]]))
                               for k in ("B_normalizado_mp4v", "C_orientado_sin_recodificar", "D_original_crudo")}
                           for g, rs in (("con_rotacion", rot), ("sin_rotacion", rect))}
(WT / "comparacion/verificacion_inferencia/fidelidad_pixeles.json").write_text(
    json.dumps({"resumen": resumen, "detalle": res}, ensure_ascii=False, indent=1))
print(json.dumps(resumen, indent=1))
