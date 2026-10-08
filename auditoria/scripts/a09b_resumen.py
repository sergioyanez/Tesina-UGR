"""Resumen de la prueba de extremo a extremo de la interfaz (a09). Idempotente: agrega `resumen_e2e` al JSON."""
from collections import Counter
from comun import *

ruta = SALIDA / "orientacion_inferencia.json"
d = leer_json(ruta)
P = d["pruebas_extremo_a_extremo"]
def resumen(sub):
    pred = [p for p in sub if p["resultado"] == "PREDICCION"]
    return {"total": len(sub), "predicciones": len(pred), "negativas": sum(p["resultado"] == "NEGATIVA" for p in sub),
            "errores": sum(p["resultado"] == "ERROR" for p in sub),
            "motivos_de_negativa": dict(Counter(p.get("motivo") for p in sub if p["resultado"] == "NEGATIVA")),
            "coinciden_con_la_etiqueta": sum(p.get("clase_predicha") == p["clase_real"] for p in pred),
            "segundos_medios": sum(p["segundos"] for p in sub) / max(1, len(sub))}
d["resumen_e2e"] = {**resumen(P), "rotados": resumen([p for p in P if p["rotacion_declarada"]]),
                    "sin_rotacion": resumen([p for p in P if not p["rotacion_declarada"]]),
                    "nota": ("Videos del propio dataset: el modelo de producción se entrenó con sus clips, así que la coincidencia con la etiqueta "
                             "NO estima el rendimiento; sólo mide si la cadena funciona de punta a punta. Muestra pequeña: indicativa.")}
guardar_json(d, "orientacion_inferencia.json")
print(json.dumps(d["resumen_e2e"], ensure_ascii=False, indent=1))
