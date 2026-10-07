"""Cribado de posibles caballos con dos identificadores (la identidad sale del prefijo del nombre).
Para cada clip: vector medio de las 16 características del backbone (ImageNet, NO entrenado para
reconocer individuos: es solo un cribado). Se compara la similitud coseno ENTRE identificadores contra
la distribución DENTRO de un mismo identificador. Un par de IDs distintos cuya similitud cae dentro de
lo habitual de un mismo ID es candidato a revisión manual; no es una prueba."""
import numpy as np
import pandas as pd
from comun import *

clips = particion()
orden, X = caracteristicas_v2(); indice = {c: i for i, c in enumerate(orden)}
M = np.stack([np.asarray(X[indice[c["clip"]]]).mean(0) for c in clips])
M = (M - M.mean(0)) / (M.std(0) + 1e-6)             # centrar para que el coseno discrimine
M /= np.linalg.norm(M, axis=1, keepdims=True)
ids = np.array([c["id_caballo"] for c in clips]); fold = np.array([c["fold"] for c in clips])
cls = np.array([c["clase"] for c in clips])
S = M @ M.T
mismo = (ids[:, None] == ids[None, :]) & ~np.eye(len(ids), dtype=bool)
dentro = S[np.triu(mismo)]
# similitud media por par de IDs distintos
uid = sorted(set(ids)); idx = {u: np.where(ids == u)[0] for u in uid}
pares = []
for i, a in enumerate(uid):
    for b in uid[i + 1:]:
        s = S[np.ix_(idx[a], idx[b])].mean()
        pares.append({"id_a": a, "id_b": b, "coseno_medio": float(s), "folds": f"{fold[idx[a][0]]}/{fold[idx[b][0]]}",
                      "clases_a": "/".join(sorted(set(cls[idx[a]]))), "clases_b": "/".join(sorted(set(cls[idx[b]]))),
                      "n_a": len(idx[a]), "n_b": len(idx[b])})
d = pd.DataFrame(pares).sort_values("coseno_medio", ascending=False)
d["ranking"] = range(1, len(d) + 1)
dif_fold = d[d.folds.map(lambda x: x.split("/")[0] != x.split("/")[1])]
r = {"coseno_dentro_del_mismo_id": {"media": float(dentro.mean()), "p5": float(np.percentile(dentro, 5)), "p50": float(np.percentile(dentro, 50)), "n_pares_de_clips": int(len(dentro))},
     "coseno_entre_ids_distintos(media por par de IDs)": {"media": float(d.coseno_medio.mean()), "p95": float(d.coseno_medio.quantile(.95)), "p99": float(d.coseno_medio.quantile(.99))},
     "top15_pares_de_ids_en_folds_distintos": dif_fold.head(15).to_dict("records"),
     "par_123_169": d[(d.id_a == "123") & (d.id_b == "169")].to_dict("records"),
     "aviso": "el backbone es ImageNet, sin entrenamiento para reidentificación; el cribado solo prioriza pares a revisar manualmente"}
guardar_json(r, "identidades.json")
print(json.dumps(r, ensure_ascii=False, indent=1)[:3500])
