"""¿La interfaz maneja los videos de celular con metadato de rotación?

El dataset se construyó con `CAP_PROP_ORIENTATION_AUTO = 1` (notebook 01). `equinos/inferencia.py`
abre el video con `cv2.VideoCapture` sin activarlo. Se mide (1) cuántos videos del dataset llevan
rotación declarada y (2) qué hace `inferencia.clasificar_video` sobre videos del propio dataset con
y sin rotación (copias en un directorio temporal: DeepLabCut escribe auxiliares junto a la entrada).
"""
import os
import shutil
import tempfile
import time
from pathlib import Path

import cv2
import pandas as pd

from comun import *

os.chdir(WT)
import sys
N_POR_GRUPO = int(sys.argv[1]) if len(sys.argv) > 1 else 8
v = pd.DataFrame(leer_csv(RES / "inventario_videos.csv"))
p = pd.DataFrame(particion())
c = pd.read_csv(SALIDA / "clips_estado.csv", dtype={"id_caballo": str})
rot = {}
for r in v.itertuples():
    cap = cv2.VideoCapture(r.ruta)
    rot[r.archivo] = float(cap.get(cv2.CAP_PROP_ORIENTATION_META))
    cap.release()
v["rotacion"] = v["archivo"].map(rot)
validos = set(c.loc[c.valido, "video_origen"])
v["en_dataset"] = v["archivo"].isin(validos)
res = {"videos_con_rotacion_declarada": int((v.rotacion != 0).sum()), "videos": len(v),
       "rotaciones": v.rotacion.value_counts().to_dict(),
       "por_clase": {k: int(((v.rotacion != 0) & (v.clase == k)).sum()) for k in CLASES},
       "clips_validos_de_videos_rotados": int(c[c.valido & c.video_origen.isin(set(v[v.rotacion != 0].archivo))].shape[0]),
       "pct_clips_validos_de_videos_rotados": 100 * float(c[c.valido].video_origen.isin(set(v[v.rotacion != 0].archivo)).mean())}
# ¿inferencia.py activa la auto-orientación?
src = (WT / "equinos" / "inferencia.py").read_text(encoding="utf-8")
res["inferencia_menciona_ORIENTATION_AUTO"] = "ORIENTATION_AUTO" in src
res["notebook01_activa_ORIENTATION_AUTO"] = "ORIENTATION_AUTO, 1" in (WT / "01_preparar_dataset_equinos.ipynb").read_text(encoding="utf-8")

# prueba de extremo a extremo con la interfaz (sin GPU extra: usa DeepLabCut igual que la app)
from equinos import inferencia
cand = v[(v.duracion_seg.astype(float) < 12) & (v.duracion_seg.astype(float) > 3) & v.en_dataset]
muestra = pd.concat([cand[cand.rotacion != 0].sample(N_POR_GRUPO, random_state=1), cand[cand.rotacion == 0].sample(N_POR_GRUPO, random_state=1)])
tmp = Path(tempfile.mkdtemp(prefix="aud_inf_"))
pruebas = []
for r in muestra.itertuples():
    destino = tmp / r.archivo.replace(" ", "_")
    shutil.copy(r.ruta, destino)
    cap = cv2.VideoCapture(str(destino)); ok, f = cap.read(); forma_leida = list(f.shape[:2]); cap.release()
    t0 = time.time()
    try:
        out = inferencia.clasificar_video(destino)
        if out.hay_prediccion:
            reg = {"resultado": "PREDICCION", "clase_predicha": out.clase, "probabilidad_max": max(out.probabilidades.values())}
        else:
            reg = {"resultado": "NEGATIVA", "motivo": out.motivo, "detalle": out.detalle[:160], "frames_alineados": out.frames_alineados}
    except Exception as e:  # noqa: BLE001
        reg = {"resultado": "ERROR", "error": repr(e)[:200]}
    pruebas.append({"video": r.archivo, "clase_real": r.clase, "rotacion_declarada": r.rotacion,
                    "forma_del_frame_leido_por_cv2_(alto,ancho)": forma_leida, "segundos": round(time.time() - t0, 1), **reg})
    print(pruebas[-1], flush=True)
res["pruebas_extremo_a_extremo"] = pruebas
guardar_json(res, "orientacion_inferencia.json")
print(json.dumps({k: v for k, v in res.items() if k != "pruebas_extremo_a_extremo"}, ensure_ascii=False, indent=1))
