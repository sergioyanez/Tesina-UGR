"""Duplicados: (1) hash SHA-256 de los videos fuente; (2) firma perceptual por video (dHash de
8 fotogramas equiespaciados) para detectar el mismo material recodificado o con otro nombre;
(3) dHash de los 5744 recortes alineados (los frames que realmente ve el modelo); (4) solapamiento
temporal de clips del mismo video. Todo en SOLO LECTURA."""
import hashlib
from collections import defaultdict
from itertools import combinations

import cv2
import numpy as np
import pandas as pd

from comun import *

v = pd.DataFrame(leer_csv(RES / "inventario_videos.csv"))
c = pd.read_csv(SALIDA / "clips_estado.csv", dtype={"id_caballo": str})
p = pd.DataFrame(particion())
carp_clase = {r.archivo: r.clase for r in v.itertuples()}
fold_de = dict(zip(p["clip"], p["fold"]))
cab_de = dict(zip(p["clip"], p["id_caballo"]))
clase_de = dict(zip(p["clip"], p["clase"]))

# ---- 1) SHA-256 de cada video --------------------------------------------------------------
def sha(ruta):
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()

rutas_vid = {r.archivo: Path(r.ruta) for r in v.itertuples()}
faltan = [a for a, r in rutas_vid.items() if not r.exists()]
hashes = {a: sha(r) for a, r in rutas_vid.items() if r.exists()}
por_hash = defaultdict(list)
for a, h in hashes.items():
    por_hash[h].append(a)
dup_exactos = [x for x in por_hash.values() if len(x) > 1]

# ---- 2) firma perceptual por video -----------------------------------------------------------
def dhash(gris, n=8):
    ch = cv2.resize(gris, (n + 1, n), interpolation=cv2.INTER_AREA)
    return (ch[:, 1:] > ch[:, :-1]).flatten()

def firma_video(ruta, k=8):
    cap = cv2.VideoCapture(str(ruta))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    hs = []
    for i in np.linspace(0, max(n - 2, 0), k).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, im = cap.read()
        if not ok:
            continue
        hs.append(dhash(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)))
    cap.release()
    return np.array(hs)

firmas = {a: firma_video(r) for a, r in rutas_vid.items() if r.exists()}
pares_similares = []
nombres = sorted(firmas)
for a, b in combinations(nombres, 2):
    fa, fb = firmas[a], firmas[b]
    if len(fa) == 0 or len(fb) == 0:
        continue
    m = min(len(fa), len(fb))
    # distancia de Hamming media entre firmas posición a posición (mismo material recodificado
    # conserva la firma); sobre 64 bits.
    d = float((fa[:m] != fb[:m]).sum(axis=1).mean())
    if d <= 6:
        pares_similares.append({"a": a, "b": b, "hamming_medio_64bits": d,
                                "id_a": a.split("_")[0][:3], "id_b": b.split("_")[0][:3],
                                "clase_a": carp_clase[a], "clase_b": carp_clase[b]})
pares_similares.sort(key=lambda x: x["hamming_medio_64bits"])

# ---- 3) dHash de los recortes alineados ------------------------------------------------------
base = DATOS_REPO / "trabajo" / "frames_alineados_v2"
fr = pd.DataFrame(leer_csv(RES / "dataset_frames_alineados_v2.csv"))
hh, sha_f = {}, {}
for r in fr.itertuples():
    ruta = base / r.ruta_frame
    im = cv2.imread(str(ruta), cv2.IMREAD_GRAYSCALE)
    hh[(r.clip, int(r.orden))] = np.packbits(dhash(im, 16)).tobytes()
    sha_f[(r.clip, int(r.orden))] = hashlib.md5(ruta.read_bytes()).hexdigest()
por_h = defaultdict(list)
for k, h in hh.items():
    por_h[h].append(k)
grupos_frames = [g for g in por_h.values() if len({x[0] for x in g}) > 1]
por_md5 = defaultdict(list)
for k, h in sha_f.items():
    por_md5[h].append(k)
grupos_md5 = [g for g in por_md5.values() if len({x[0] for x in g}) > 1]

def describe(g):
    clips = sorted({x[0] for x in g})
    return {"clips": clips, "caballos": sorted({cab_de[x] for x in clips}),
            "clases": sorted({clase_de[x] for x in clips}), "folds": sorted({fold_de[x] for x in clips}),
            "n_frames": len(g)}
desc_f = [describe(g) for g in grupos_frames]
cruce_fold = [d for d in desc_f if len(d["folds"]) > 1]
cruce_clase = [d for d in desc_f if len(d["clases"]) > 1]

# duplicados de frame dentro del mismo clip (selección con anclaje: no debería repetir frame)
dentro = fr.groupby("clip")["frame"].apply(lambda s: s.duplicated().sum())

# ---- 4) clips del mismo video: solapamiento temporal -----------------------------------------
sol = 0
for vid, g in c.groupby("video_origen"):
    g = g.sort_values("frame_inicio")
    ini, fin = g.frame_inicio.values, g.frame_fin.values
    for i in range(len(g) - 1):
        if ini[i + 1] < fin[i]:
            sol += 1

# clips de un mismo video: ¿van a un solo fold? (consecuencia de agrupar por caballo)
vid_folds = p.merge(c[["clip", "video_origen"]], on="clip").groupby("video_origen").fold.nunique()

# pares perceptuales que cruzan caballo/fold/clase
def cruces(pares):
    cab = lambda a: a.split("_")[0].split("-")[0][:3]
    return [x for x in pares if x["id_a"] != x["id_b"]]
pares_otro_caballo = cruces(pares_similares)

res = {
    "videos_hasheados": len(hashes), "videos_no_encontrados": faltan,
    "grupos_de_videos_con_hash_identico": dup_exactos,
    "pares_videos_perceptualmente_similares_hamming<=6": pares_similares,
    "n_pares_similares": len(pares_similares), "n_pares_similares_distinto_id": len(pares_otro_caballo),
    "pares_similares_distinto_id": pares_otro_caballo,
    "frames_analizados": len(hh),
    "grupos_frames_identicos_md5_entre_clips_distintos": len(grupos_md5),
    "grupos_frames_dhash256_identicos_entre_clips_distintos": len(desc_f),
    "de_ellos_cruzan_fold": len(cruce_fold), "de_ellos_cruzan_clase": len(cruce_clase),
    "detalle_grupos_frames": desc_f[:30],
    "frames_repetidos_dentro_del_mismo_clip": int(dentro.sum()),
    "solapamientos_temporales_entre_clips_del_mismo_video": sol,
    "videos_con_clips_en_mas_de_un_fold": int((vid_folds > 1).sum()),
}
guardar_json(res, "duplicados.json")
print(json.dumps({k: (v_ if not isinstance(v_, list) or len(v_) < 6 else f"{len(v_)} elementos")
                  for k, v_ in res.items()}, ensure_ascii=False, indent=1))
for x in pares_similares[:15]:
    print(x)
for d in desc_f[:10]:
    print(d)
