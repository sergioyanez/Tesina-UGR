"""Inventario anatómico, compatibilidad DLC->SimBA y comparación con las 9 features vigentes.

Todo número sale de los artefactos (pose_seleccionada_v2.csv, dataset_frames_alineados_v2.csv,
inventarios). El denominador de cada tasa está escrito junto a la tasa. Escribe
resultados/inventario_anatomico.json y resultados/compatibilidad_dlc_simba.json.
"""
import json

import numpy as np
import pandas as pd

import datos

clips = datos.clips_con_fold()
P = datos.pose_por_clip(clips)
A = np.stack([P[c["clip"]] for c in clips])            # (359,16,39,3)
n_frames = A.shape[0] * A.shape[1]

disp = {bp: {"frames_conf_ge_0.50": float((A[:, :, i, 2] >= datos.CONF_MIN).mean()),
             "confianza_mediana": float(np.median(A[:, :, i, 2])),
             "clips_con_algun_frame_ge_0.50": float((A[:, :, i, 2] >= datos.CONF_MIN).any(1).mean())}
        for i, bp in enumerate(datos.BODYPARTS)}

FACIALES = ["nose", "upper_jaw", "lower_jaw", "mouth_end_right", "mouth_end_left", "right_eye", "left_eye",
            "right_earbase", "right_earend", "left_earbase", "left_earend", "right_antler_base", "left_antler_base",
            "right_antler_end", "left_antler_end", "throat_base", "throat_end", "neck_base", "neck_end"]

# --- 9 features vigentes: ¿cuánta información independiente llevan? ---
G = datos.geometria_actual(clips, None)               # (359,16,9)
val = G[..., 6:9]
sim = G[..., 0:2] + G[..., 2:4]                       # ojo + hocico
ok = (val[..., 0] == 1) & (val[..., 1] == 1)
norma = np.linalg.norm(G[..., 2:4], axis=-1)
nueve = {
    "columnas": ["x_ojo", "y_ojo", "x_hocico", "y_hocico", "x_oreja", "y_oreja", "valido_ojo", "valido_hocico", "valido_oreja"],
    "definicion_codigo": "equinos/clasificacion.py::MultimodalEquineDataset.__getitem__ (ojo = mejor 'eye' por confianza; hocico = mejor de nose/snout/muzzle/upper_jaw/mouth; oreja = mejor de 'earbase'/'earend' de AMBAS orejas), centradas en el punto medio ojo-hocico y divididas por la distancia ojo-hocico del propio frame; las máscaras son 1 si confianza > 0 (NO >= 0,50).",
    "denominador": f"{int(ok.sum())} de {ok.size} frames (359 clips x 16) con ojo y hocico válidos",
    "ojo_mas_hocico_es_cero_(simetria_exacta)_fraccion": float((np.abs(sim[ok]).max(-1) < 1e-6).mean()),
    "norma_hocico_mediana_(debe_ser_0.5)": float(np.median(norma[ok])),
    "fraccion_mascara_ojo_1": float(val[..., 0].mean()), "fraccion_mascara_hocico_1": float(val[..., 1].mean()),
    "fraccion_mascara_oreja_1": float(val[..., 2].mean()),
    "lectura": "Con el centrado y la escala ojo-hocico, las 4 primeras columnas valen (-0,5, +0,5)*u donde u es el versor ojo->hocico en coordenadas de imagen: llevan UN solo grado de libertad (inclinación de la cabeza en el plano de la imagen). x/y de oreja es la posición de UN punto de oreja (el más confiable, que puede cambiar de base a punta o de lado entre frames) en ejes de imagen, sin rotar al eje de la cabeza, así que mezcla orientación con posición. Las máscaras valen ~1 casi siempre porque se usa confianza > 0.",
}

tabla_features = [
    ("angulo de orejas respecto del eje de la cabeza", "aproximada", "earbase/earend de cada oreja; proyección 2D, depende del punto de vista", ["left_earbase", "left_earend", "right_earbase", "right_earend"], "oreja_angulo_*"),
    ("largo/posición de orejas, asimetría L-R, separación entre puntas", "aproximada", "2D con escorzo; 'izquierda/derecha' es del animal, no de la imagen", ["left_earbase", "left_earend", "right_earbase", "right_earend"], "oreja_*"),
    ("apertura del ojo", "imposible", "un solo keypoint por ojo; no hay párpados superior/inferior", ["left_eye", "right_eye"], None),
    ("parpadeo", "imposible", "no hay apertura de ojo; además se muestrean 16 frames separados 0,2-0,5 s (un parpadeo dura ~0,1-0,4 s)", ["left_eye", "right_eye"], None),
    ("distancia interocular / ojo-hocico", "directamente observable (en 2D)", "ambos ojos y nariz", ["left_eye", "right_eye", "nose"], "ojos_distancia, cabeza_largo_ojo_hocico_rel"),
    ("labios (tensión, comisuras)", "aproximada", "solo comisuras (mouth_end_*) y mandíbulas (upper/lower_jaw), no labios; comisuras visibles en poco frames", ["mouth_end_left", "mouth_end_right", "upper_jaw", "lower_jaw"], "boca_*"),
    ("apertura de mandíbula/boca", "aproximada", "upper_jaw-lower_jaw", ["upper_jaw", "lower_jaw"], "boca_apertura_mandibulas"),
    ("fosas nasales, tensión muscular, surcos faciales", "imposible", "no existen landmarks para esas estructuras", [], None),
    ("orientación/inclinación de cabeza en el plano de la imagen", "directamente observable", "eje ojo(s)->hocico", ["nose", "left_eye", "right_eye"], "cabeza_inclinacion_sobre_horizontal"),
    ("pitch/yaw/roll 3D de la cabeza", "imposible (solo proxy)", "pose 2D monocular; el cociente largo ojo-hocico / mediana del clip es un proxy de escorzo, no de ángulo", ["nose", "left_eye", "right_eye"], "cabeza_largo_ojo_hocico_rel"),
    ("ángulo cabeza-cuello", "aproximada", "neck_base/neck_end con confianza >=0,50 en 25-36 % de los frames; semántica exacta de neck_* no verificada", ["neck_base", "neck_end"], "cuello_*"),
    ("velocidades (cabeza, orejas, mandíbula)", "aproximada", "diferencias entre los 16 frames del clip (separados 0,2-0,5 s); la velocidad de cabeza incluye movimiento de cámara y de cuerpo (no hay marco fijo)", ["nose", "left_eye", "right_eye"], "*_velocidad_*"),
    ("aceleraciones", "aproximada (más ruidosa)", "segunda diferencia sobre muestreo grueso y jitter de keypoints", ["nose", "left_eye", "right_eye"], "*_aceleracion"),
    ("simetría L-R", "aproximada", "depende del punto de vista (vistas de perfil ocultan un lado)", ["left_eye", "right_eye", "left_earbase", "right_earbase"], "*_asimetria_*"),
    ("antler_base/antler_end", "no usadas", "puntos del esquema SuperAnimal-Quadruped pensados para cuernas; su correspondencia anatómica en el caballo No pudo verificarse con los archivos disponibles (antler_end tiene confianza >=0,50 en 0 % de los frames)", ["left_antler_base", "right_antler_base", "left_antler_end", "right_antler_end"], None),
]
inv = {
    "origen_de_pose": "DeepLabCut SuperAnimal-Quadruped (HRNet-W32 + Faster R-CNN ResNet-50-FPN v2), 39 bodyparts; equinos/pose.py",
    "bodyparts": datos.BODYPARTS, "n_bodyparts": len(datos.BODYPARTS),
    "denominador_disponibilidad": f"{n_frames} frames = 359 clips x 16 frames de pose_seleccionada_v2.csv (los 16 frames que ve el MIL; seleccionados por ventana donde ojo y hocico están medidos, por eso nose/eye son altos: es un sesgo de selección, no la tasa general del modelo)",
    "disponibilidad_por_bodypart": disp,
    "puntos_faciales_y_cefalicos": {b: disp[b] for b in FACIALES},
    "features_candidatas": [{"feature": f, "clasificacion": c, "motivo": m, "bodyparts": b, "implementada_como": i} for f, c, m, b, i in tabla_features],
    "comparacion_con_9_vigentes": nueve,
    "redundancia_y_novedad": {
        "redundantes_con_las_9": ["cabeza_inclinacion_sobre_horizontal (equivale a la dirección u de las 4 primeras columnas)",
                                  "disp_* / escala (las máscaras valid_* de las 9 son casi constantes)"],
        "informacion_nueva": ["largo, ángulo y asimetría de orejas relativos al eje de la cabeza (las 9 solo tienen UN punto de oreja)",
                              "boca/mandíbula y comisuras (las 9 no incluyen)", "distancia interocular y escorzo de la cabeza",
                              "dinámica temporal: velocidades y aceleraciones de cabeza/oreja/mandíbula (las 9 se resumen en el MIL por atención, sin derivadas explícitas)",
                              "disponibilidad de keypoints"]},
}
(datos.SALIDA / "inventario_anatomico.json").write_text(json.dumps(inv, ensure_ascii=False, indent=1), encoding="utf-8")

# --- compatibilidad DLC -> SimBA ---
T = datos.tiempos_por_clip()
fps = np.array([T[c["clip"]][1] for c in clips]); span = np.array([T[c["clip"]][2] for c in clips])
dts = np.concatenate([np.diff(T[c["clip"]][0]) / T[c["clip"]][1] for c in clips])
meta = datos.metadatos(clips)
comp = {
    "almacenamiento": {
        "archivo_usado": "datos_equinos/resultados/pose_seleccionada_v2.csv (16 frames por clip) ; pose_todos_los_frames.csv = todos los frames; pose_completa.csv = corrida anterior",
        "formato": "LARGO, una fila por (clip, orden, bodypart): columnas clip, orden, bodypart, x, y, confianza",
        "orden": "posición 0..15 en la secuencia del clip en el CSV seleccionado; en pose_todos_los_frames.csv 'orden' es el índice del frame del video",
        "frame_a_punto": "dataset_frames_alineados_v2.csv enlaza (clip, orden) con el número de frame del video ('frame') y el fps",
        "likelihood": "columna 'confianza' (likelihood de DLC, 0-1); umbral del pipeline 0,50 (CONF_MINIMA_KEYPOINT)",
        "individuos": "DLC devuelve 10 'individuals'; equinos.pose.elegir_individuo se queda con el de mayor confianza media; el CSV ya tiene un solo animal",
        "coordenadas": "píxeles del CUADRO ORIGINAL del video (no del recorte alineado 224x343 ni normalizadas)",
        "coordenadas_max_x_y": [float(np.nanmax(A[..., 0])), float(np.nanmax(A[..., 1]))],
        "resoluciones_de_origen": {f"{int(a)}x{int(b)}": int(((meta[:, 2] == a) & (meta[:, 3] == b)).sum()) for a, b in sorted({(m[2], m[3]) for m in meta})},
        "fps_clips_min_med_max": [float(fps.min()), float(np.median(fps)), float(fps.max())],
        "separacion_temporal_entre_frames_consecutivos_s_(p10,mediana,p90)": [float(np.percentile(dts, q)) for q in (10, 50, 90)],
        "ventana_del_clip_s_(min,mediana,max)": [float(span.min()), float(np.median(span)), float(span.max())],
        "normalizacion_o_alineacion_en_el_CSV": "ninguna: son landmarks crudos de DLC; la alineación afín (recorte 224x343 por ojo-hocico) se aplica a las imágenes, y la normalización ojo-hocico del MIL se hace dentro de MultimodalEquineDataset, no en el archivo",
    },
    "importacion_a_SimBA": {
        "directa": False,
        "motivo": ["el CSV es largo y SimBA espera ancho (<bp>_x, <bp>_y, <bp>_p) o el CSV/H5 multiindex de DLC (scorer/bodyparts/coords)",
                   "SimBA exige un proyecto con project_config.ini, body-part config definido por el usuario (39 nombres), video_info.csv (fps, resolución, px/mm) y los archivos en csv/outlier_corrected_movement_location",
                   "SimBA asume series de frames CONSECUTIVOS a fps constante (ventanas móviles en frames); aquí son 16 frames separados 0,2-0,5 s"],
        "conversion_realizada": "experimentos/simba/simba_real_intento.py: escribe los 359 clips en formato ancho de SimBA y crea el proyecto con ProjectConfigCreator",
        "pixeles_por_mm": "sin calibración: se usó 1,0 (las distancias de SimBA quedan en píxeles del cuadro original, dependientes de la resolución)",
    },
    "roles": "DLC = video -> landmarks (detector/estimador de pose). SimBA = landmarks -> features -> clasificador. Ninguno cumple ambos roles.",
}
sr = json.loads((datos.SALIDA / "simba_real_intento.json").read_text(encoding="utf-8"))
comp["simba_real"] = {k: sr[k] for k in ("resultado", "pasos", "columnas_extraidas_por_clip", "filas_por_clip", "columnas_agregadas") if k in sr}
comp["simba_real"]["instalacion"] = "pip install simba-uw-tf-dev (5.6.1) en ~/.venvs/simba (Python 3.10); requiere tkinter para importar: no está en el sistema y no hay sudo, se usó un stub vacío (experimentos/simba/stub_tkinter), que solo afecta a la GUI"
comp["simba_real"]["alcance_del_extractor"] = "UserDefinedFeatureExtractor con 1 animal solo genera movimiento por bodypart, estadísticas de movimiento global, ventanas móviles de ese movimiento (2/5/6/7,5/15 frames según fps), suma/media de probabilidades y conteo de detecciones de baja probabilidad: NO calcula distancias entre bodyparts, ángulos, áreas ni nada facial. Solo cubre movimiento."
comp["decision"] = ("Las features faciales (ángulos, distancias, ratios) NO las produce el extractor real de SimBA para un solo animal. Conjunto principal: «features tipo SimBA calculadas con código propio» "
                    "(experimentos/simba/features_simba.py). Conjunto secundario B2: features REALES de SimBA (solo movimiento) agregadas por clip.")
(datos.SALIDA / "compatibilidad_dlc_simba.json").write_text(json.dumps(comp, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps({k: v for k, v in nueve.items() if k not in ("lectura", "definicion_codigo", "columnas")}, ensure_ascii=False, indent=1))
print(json.dumps(comp["almacenamiento"]["resoluciones_de_origen"]), comp["almacenamiento"]["coordenadas_max_x_y"],
      comp["almacenamiento"]["separacion_temporal_entre_frames_consecutivos_s_(p10,mediana,p90)"])
