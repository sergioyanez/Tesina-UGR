"""Cuánto predice la IDENTIDAD del caballo sin mirar nada: partición por caballo (protocolo) frente a la partición
ingenua por clip (que NO se usa en el trabajo). Mide, sobre el conjunto vigente, el efecto de agrupar."""
import numpy as np
from comun import *
from equinos import clasificacion as CL

clips = particion()
res = {}
r = CL.evaluar(CL.predecir_por_identidad, CL.folds_por_ejemplar(clips), "identidad, por caballo")
res["por_caballo"] = {"f1_macro": r["f1_macro"][0], "f1_desvio": r["f1_macro"][1], "caballos_de_test_nunca_vistos": r["sin_ver"], "clips": r["clips"]}
for s in (42, 1, 2):
    ing = CL.folds_ingenuos(clips, 4, semilla=s)
    r2 = CL.evaluar(CL.predecir_por_identidad, ing, "identidad, ingenua")
    res[f"ingenua_semilla_{s}"] = {"f1_macro": r2["f1_macro"][0], "f1_desvio": r2["f1_macro"][1], "clips_con_caballo_visto": r2["clips"] - r2["sin_ver"], "clips": r2["clips"]}
v = [res[k]["f1_macro"] for k in res if k.startswith("ingenua")]
res["resumen"] = {"f1_solo_identidad_particion_ingenua_media": float(np.mean(v)), "f1_solo_identidad_por_caballo": res["por_caballo"]["f1_macro"],
                  "espurio": float(np.mean(v)) - res["por_caballo"]["f1_macro"]}
guardar_json(res, "identidad_vs_particion.json")
print(json.dumps(res, ensure_ascii=False, indent=1))
