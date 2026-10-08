"""Prueba e2e de la inferencia con la FUSIÓN sobre los 16 videos de la auditoría.

Misma lista y mismo preprocesamiento (`normalizar=True`) que `e2e.py`, que con el MIL dio 16/16.
Son videos de entrenamiento: la coincidencia con la etiqueta es una prueba de COHERENCIA de la
cadena (video -> pose -> features -> modelo), no de generalización.

Además, para el clip del dataset que sale de cada video, compara la predicción de la inferencia con
la que da el modelo sobre la pose y las características persistidas (la «caché» de entrenamiento).
Solo lectura sobre datos_equinos. Uso: python comparacion/verificacion_inferencia/e2e_fusion.py
"""
import csv, json, os, sys, time
from pathlib import Path
import numpy as np

WT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WT))
from equinos import clasificacion, fusion, inferencia  # noqa: E402

RES = Path(os.environ["TESINA_EQUINOS_RAIZ"]) / "resultados"
SALIDA = WT / "comparacion" / "verificacion_inferencia"


def leer(n): return list(csv.DictReader((RES / n).open(encoding="utf-8", errors="replace")))


inv = {r["archivo"]: r for r in leer("inventario_videos.csv")}
icl = leer("inventario_clips.csv")
ds = leer("dataset_frames_alineados_v2.csv")
pose_sel = leer("pose_seleccionada_v2.csv")
feats = np.load(RES / "caracteristicas_backbone_v2.npz", allow_pickle=True)["caracteristicas"]
nombres = json.load(open(RES / "caracteristicas_backbone_v2.json"))["clips"]
videos = [x["video"] for x in json.load(open(WT / "auditoria/datos/orientacion_inferencia.json"))["pruebas_extremo_a_extremo"]]
p = inferencia.parametros_etapa02()
paquete = fusion.cargar_paquete()
precisiones, metricas = fusion.precision_por_clase(), fusion.metricas_globales()
mil = clasificacion.cargar_produccion()

salida = []
for v in videos:
    meta = inv[v]; clase = meta["clase"]
    clip = next((c for c in icl if c["video_origen"] == v), None)
    t0 = time.time(); info = {}
    reg = {"video": v, "clase_real": clase}
    prep = inferencia.preparar_clip(meta["ruta"], parametros=p, normalizar=True, info=info)
    if isinstance(prep, inferencia.Negativa):
        reg.update(resultado="NEGATIVA", motivo=prep.motivo)
        salida.append(reg); print(json.dumps(reg, ensure_ascii=False), flush=True); continue
    recortes, filas = prep
    vec = inferencia._caracteristicas_de(recortes)
    pr = inferencia.predecir_con_fusion(vec, filas, info["ordenes"], info["fps"],
                                        precisiones=precisiones, paquete=paquete, metricas=metricas)
    pm = inferencia.predecir_desde_caracteristicas(vec, filas, modelo=mil)
    reg.update(resultado="PREDICCION", predicha=pr.clase, coincide=pr.clase == clase,
               prob_max=max(pr.probabilidades.values()),
               pose=max(pr.detalle["pose"], key=pr.detalle["pose"].get),
               visual=max(pr.detalle["visual"], key=pr.detalle["visual"].get),
               mil=pm.clase, segundos=round(time.time() - t0, 1),
               ordenes=info["ordenes"], total=info["total"], rotacion=info["rotacion"])
    if clip and clip["clip"] in nombres:
        cn = clip["clip"]
        dfr = sorted((r for r in ds if r["clip"] == cn), key=lambda r: int(r["orden"]))
        ps = [r for r in pose_sel if r["clip"] == cn]
        pc = inferencia.predecir_con_fusion(
            np.asarray(feats[nombres.index(cn)]), ps, [float(r["frame"]) for r in dfr],
            float(dfr[0]["fps"]), precisiones=precisiones, paquete=paquete, metricas=metricas)
        reg["cache"] = {"clip": cn, "pred_cache": pc.clase,
                        "pred_cache_coincide_etiqueta": pc.clase == clase,
                        "pred_inf_igual_cache": pc.clase == pr.clase,
                        "dif_max_prob_inf_vs_cache": max(abs(pr.probabilidades[c] - pc.probabilidades[c])
                                                         for c in pr.probabilidades)}
    salida.append(reg); print(json.dumps(reg, ensure_ascii=False), flush=True)

res = {"modelo": "fusion", "n": len(salida),
       "predicciones": sum(r["resultado"] == "PREDICCION" for r in salida),
       "coinciden_con_etiqueta": sum(bool(r.get("coincide")) for r in salida),
       "mil_coincide_con_etiqueta": sum(r.get("mil") == r["clase_real"] for r in salida),
       "inferencia_igual_cache": sum(bool(r.get("cache", {}).get("pred_inf_igual_cache")) for r in salida),
       "con_cache": sum("cache" in r for r in salida),
       "detalle": salida}
(SALIDA / "e2e_fusion.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
print("COINCIDEN", res["coinciden_con_etiqueta"], "/", res["n"],
      "| igual a la caché", res["inferencia_igual_cache"], "/", res["con_cache"])
