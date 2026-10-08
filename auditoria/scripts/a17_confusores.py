"""Confusores de filmación más allá de duración/resolución: rotación declarada, contenedor, códec, tasa de bits y
FECHA de la sesión (inferida del nombre de archivo). Se mide cuánto predicen la clase SIN mirar la cara,
con los mismos folds por caballo y regresión logística (C=1, balanced, estandarización por fold)."""
import re
import subprocess
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from comun import *
from equinos import clasificacion as CL

v = pd.DataFrame(leer_csv(RES / "inventario_videos.csv"))
orient = leer_json(SALIDA / "orientacion_inferencia.json") if (SALIDA / "orientacion_inferencia.json").exists() else None
import cv2
def probe(ruta):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=codec_name,bit_rate,pix_fmt,profile:format=format_name,size",
                        "-of", "json", str(ruta)], capture_output=True, text=True)
    try:
        j = json.loads(r.stdout)
        s = j["streams"][0]; f = j["format"]
        return {"codec": s.get("codec_name"), "bit_rate": float(s.get("bit_rate") or f.get("size", 0)) , "pix_fmt": s.get("pix_fmt"),
                "profile": s.get("profile"), "size": float(f.get("size", 0))}
    except Exception:
        return {"codec": None, "bit_rate": np.nan, "pix_fmt": None, "profile": None, "size": np.nan}
meta = {}
for r in v.itertuples():
    p = probe(r.ruta)
    cap = cv2.VideoCapture(r.ruta); p["rotacion"] = float(cap.get(cv2.CAP_PROP_ORIENTATION_META)); cap.release()
    p["ext"] = Path(r.ruta).suffix.lower()
    m = re.search(r"(20\d{2})[-_]?(\d{2})[-_]?(\d{2})", r.archivo)
    p["fecha"] = f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m and 1 <= int(m.group(2)) <= 12 and 1 <= int(m.group(3)) <= 31 else "sin_fecha"
    meta[r.archivo] = p
M = pd.DataFrame(meta).T
M["clase"] = v.set_index("archivo")["clase"]
M["bit_rate"] = M["bit_rate"].astype(float); M["size"] = M["size"].astype(float)

clips = particion()
inv = pd.DataFrame(leer_csv(RES / "inventario_clips.csv")).set_index("clip")
vid = [inv.loc[c["clip"], "video_origen"] for c in clips]
y = np.array([c["clase"] for c in clips]); fold = np.array([c["fold"] for c in clips])
Mc = M.loc[vid].reset_index(drop=True)

def evaluar(X, C=1.0):
    pred = np.empty(len(y), dtype=object)
    for k in range(4):
        tr, te = fold != k, fold == k
        m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000, class_weight="balanced"))
        m.fit(X[tr], y[tr]); pred[te] = m.predict(X[te])
    f = [CL.f1_macro(y[fold == k].tolist(), pred[fold == k].tolist()) for k in range(4)]
    return {"f1_macro": float(np.mean(f)), "f1_desvio": float(np.std(f, ddof=1)), "accuracy": float((pred == y).mean())}

def uno(col):
    return pd.get_dummies(Mc[col]).values.astype(float)
rot = Mc[["rotacion"]].astype(float).values
tec = np.hstack([uno("codec"), uno("ext"), uno("pix_fmt"), np.log1p(Mc[["bit_rate", "size"]].astype(float).fillna(0).values), rot])
fecha = uno("fecha")
mes = pd.get_dummies(Mc["fecha"].str[:7]).values.astype(float)
anio = pd.get_dummies(Mc["fecha"].str[:4]).values.astype(float)
res = {
    "rotacion_declarada_por_clase": {k: {"rotados": int(((M.clase == k) & (M.rotacion != 0)).sum()), "videos": int((M.clase == k).sum())} for k in CLASES},
    "contenedor_por_clase": {k: dict(Counter(M[M.clase == k].ext)) for k in CLASES},
    "codec_por_clase": {k: dict(Counter(M[M.clase == k].codec)) for k in CLASES},
    "clips_por_fecha_distinta": int(Mc.fecha.nunique()), "videos_sin_fecha_en_el_nombre": int((M.fecha == "sin_fecha").sum()),
    "baselines_sin_mirar_la_cara": {
        "solo_rotacion_declarada": evaluar(rot),
        "tecnicos (códec, contenedor, pix_fmt, tasa de bits, tamaño, rotación)": evaluar(tec),
        "solo_año_de_la_sesión (por nombre de archivo)": evaluar(anio),
        "solo_mes_de_la_sesión": evaluar(mes),
        "solo_fecha_exacta_de_la_sesión": evaluar(fecha, C=0.3),
        "técnicos + fecha exacta": evaluar(np.hstack([tec, fecha]), C=0.3)},
    "nota": ("La fecha sale del nombre de archivo (patrones YYYYMMDD / YYYY-MM-DD); es una aproximación a la sesión de filmación, "
             "no un identificador de establecimiento. Un caballo de test nunca visto puede compartir fecha con caballos de train "
             "(misma sesión, mismo lugar): es la vía por la que el contexto puede ayudar sin que haya fuga de individuo.")}
guardar_json(res, "confusores.json")
print(json.dumps(res, ensure_ascii=False, indent=1))
