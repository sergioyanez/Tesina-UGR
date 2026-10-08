"""Prueba e2e de `clasificar_video` sobre los 16 videos de la auditoría, antes y después del arreglo.

Modo `legacy` (normalizar=False) = lectura anterior; modo `normalizado` = lectura igualada al
notebook 01 (ORIENTATION_AUTO + mp4v). Compara además, cuadro por cuadro, contra la caché de
entrenamiento (pose_seleccionada_v2, características v2) para el clip del mismo video.
Solo lectura sobre datos_equinos. Uso: python comparacion/verificacion_inferencia/e2e.py MODO
"""
import csv, json, os, sys, time
from pathlib import Path
import numpy as np

WT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WT))
from equinos import inferencia, pose as sel_pose, clasificacion  # noqa: E402

RES = Path(os.environ["TESINA_EQUINOS_RAIZ"]) / "resultados"
modo = sys.argv[1]
SALIDA = WT / "comparacion" / "verificacion_inferencia"
ART = WT / "artefactos_base"; ART.mkdir(exist_ok=True)

def leer(n): return list(csv.DictReader((RES / n).open(encoding="utf-8", errors="replace")))
inv = {r["archivo"]: r for r in leer("inventario_videos.csv")}
icl = leer("inventario_clips.csv")
ds = leer("dataset_frames_alineados_v2.csv")
pose_sel = leer("pose_seleccionada_v2.csv")
feats = np.load(RES / "caracteristicas_backbone_v2.npz", allow_pickle=True)["caracteristicas"]
nombres = json.load(open(RES / "caracteristicas_backbone_v2.json"))["clips"]
videos = [x["video"] for x in json.load(open(WT / "auditoria/datos/orientacion_inferencia.json"))["pruebas_extremo_a_extremo"]]
p = inferencia.parametros_etapa02()
modelo = clasificacion.cargar_produccion()

def cos(a, b): return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

salida = []
for v in videos:
    meta = inv[v]; clase = meta["clase"]
    clip = next((c for c in icl if c["video_origen"] == v), None)
    t0 = time.time(); info = {}
    reg = {"video": v, "clase_real": clase}
    prep = inferencia.preparar_clip(meta["ruta"], parametros=p, normalizar=(modo == "normalizado"), info=info)
    if isinstance(prep, inferencia.Negativa):
        reg.update(resultado="NEGATIVA", motivo=prep.motivo)
        salida.append(reg); print(reg, flush=True); continue
    recortes, filas = prep
    vec = inferencia._caracteristicas_de(recortes)
    pr = inferencia.predecir_desde_caracteristicas(vec, filas, modelo=modelo)
    reg.update(resultado="PREDICCION", predicha=pr.clase, coincide=pr.clase == clase,
               prob_max=max(pr.probabilidades.values()), segundos=round(time.time() - t0, 1),
               ordenes=info["ordenes"], total=info["total"], rotacion=info["rotacion"])
    np.save(ART / f"inf_{modo}_{v}.npy", vec)
    # --- contra la caché de entrenamiento -------------------------------------------------
    if clip and clip["clip"] in nombres:
        cn = clip["clip"]; ini = int(clip["frame_inicio"])
        dfr = sorted((r for r in ds if r["clip"] == cn), key=lambda r: int(r["orden"]))
        cache_frames = [ini + int(r["frame"]) for r in dfr]
        cf = np.asarray(feats[nombres.index(cn)])
        # predicción del modelo desplegado sobre la caché (lo que "vio" en entrenamiento)
        ps = [r for r in pose_sel if r["clip"] == cn]
        pc = inferencia.predecir_desde_caracteristicas(cf, ps, modelo=modelo)
        comunes = sorted(set(cache_frames) & set(info["ordenes"]))
        cos_f, rel_f, dpose = [], [], []
        par_inf = {int(x["orden"]): x for x in sel_pose.seleccionar_par(filas, bodypart_hocico=p["BODYPART_HOCICO"])}
        par_cac = {int(x["orden"]): x for x in sel_pose.seleccionar_par(ps, bodypart_hocico=p["BODYPART_HOCICO"])}
        for fr in comunes:
            a = vec[info["ordenes"].index(fr)]; b = cf[cache_frames.index(fr)]
            cos_f.append(cos(a, b)); rel_f.append(float(np.linalg.norm(a - b) / np.linalg.norm(b)))
            ia, ib = par_inf.get(info["ordenes"].index(fr)), par_cac.get(cache_frames.index(fr))
            if ia and ib:
                dpose.append(max(abs(float(ia[k]) - float(ib[k])) for k in ("x_ojo", "y_ojo", "x_hocico", "y_hocico")))
        reg["cache"] = {
            "clip": cn, "frames_clip": [ini, int(clip["frame_fin"])],
            "frames_comunes": len(comunes), "pred_cache": pc.clase,
            "pred_cache_coincide_etiqueta": pc.clase == clase,
            "pred_inf_igual_cache": pc.clase == pr.clase,
            "coseno_medio_comunes": float(np.mean(cos_f)) if cos_f else None,
            "dif_rel_media_comunes": float(np.mean(rel_f)) if rel_f else None,
            "dif_pose_px_mediana_comunes": float(np.median(dpose)) if dpose else None,
            "ventana_cache": [min(cache_frames), max(cache_frames)],
            "ventana_inferencia": [min(info["ordenes"]), max(info["ordenes"])],
        }
    salida.append(reg); print(json.dumps(reg, ensure_ascii=False), flush=True)

res = {"modo": modo, "n": len(salida),
       "predicciones": sum(r["resultado"] == "PREDICCION" for r in salida),
       "coinciden_con_etiqueta": sum(bool(r.get("coincide")) for r in salida),
       "detalle": salida}
(SALIDA / f"e2e_{modo}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
print("COINCIDEN", res["coinciden_con_etiqueta"], "/", res["n"])
