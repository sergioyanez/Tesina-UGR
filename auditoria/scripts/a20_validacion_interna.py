"""¿Qué caballos forman la validación interna? `dividir_por_ejemplar` ordena por cantidad de clips (desc.) y
asigna los PRIMEROS caballos a validación hasta cubrir el 25 % de los clips."""
import numpy as np
from collections import Counter
from comun import *
from equinos import clasificacion as CL, particion as P

clips = particion(); res = []
n_por_cab = Counter(c["id_caballo"] for c in clips)
for k, (trval, test) in enumerate(CL.folds_por_ejemplar(clips)):
    tr, va = P.dividir_por_ejemplar(trval, 0.25)
    cv = Counter(c["id_caballo"] for c in va); ct = Counter(c["id_caballo"] for c in tr)
    cl_va = Counter(c["clase"] for c in va)
    res.append({"fold": k, "caballos_validacion": len(cv), "clips_por_caballo_validacion": sorted(cv.values(), reverse=True),
                "media_clips_por_caballo_val": float(np.mean(list(cv.values()))), "media_clips_por_caballo_train_interno": float(np.mean(list(ct.values()))),
                "maximo_clips_por_caballo_train_interno": max(ct.values()), "minimo_clips_por_caballo_val": min(cv.values()),
                "caballos_val_multiclase": sum(1 for h in cv if len({c['clase'] for c in va if c['id_caballo'] == h}) > 1),
                "clases_val": dict(cl_va)})
out = {"por_fold": res, "media_global_clips_por_caballo": float(np.mean(list(n_por_cab.values()))),
       "lectura": ("La validación interna reúne a los 7 caballos con MÁS clips del train_val de cada fold (≈10 clips por caballo frente a ≈2,7 en el conjunto): "
                   "no es una muestra aleatoria de caballos, el train interno pierde a los caballos más representados y la métrica de validación "
                   "pesa muy poco a los caballos con pocos clips (mayoría). Sesgo de selección en la elección de la época.")}
guardar_json(out, "validacion_interna_composicion.json")
print(json.dumps(out, ensure_ascii=False, indent=1))
