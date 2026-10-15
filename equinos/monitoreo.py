"""Monitoreo continuo: el video con la cara enmarcada y la expresión de cada tramo.

Autor original: Héctor (se integró con ajustes menores). Divide el video en ventanas deslizantes
de 3 s, clasifica cada una con la fusión pose + visual y dibuja sobre el video un recuadro blanco
en la cara con la etiqueta de la ventana que cubre cada fotograma.

Cada etiqueta sale de una ventana de 3 s y no del tramo que usa `inferencia.clasificar_video`,
así que puede diferir de la respuesta global.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np


def generar_ventanas(total_frames: int, fps: float, duracion_seg: float = 3.0, stride_seg: float = 1.0) -> list[tuple[int, int, list[int]]]:
    """Genera ventanas temporales anchas, muestreando 16 frames distribuidos parejos por ventana.

    El modelo fue entrenado sobre clips de varios segundos. Pasarle 16 frames contiguos (0.5s)
    destruye las features dinámicas (velocidad, rango), provocando predicciones erróneas.
    """
    ventanas = []
    ventana_frames = int(duracion_seg * fps)
    stride_frames = int(stride_seg * fps)

    if total_frames <= ventana_frames:
        if total_frames < 16:
            return []
        posiciones = np.linspace(0, total_frames - 1, 16)
        frames = [int(round(p)) for p in posiciones]
        return [(0, total_frames, frames)]

    for inicio in range(0, total_frames - ventana_frames + 1, stride_frames):
        fin = inicio + ventana_frames - 1
        posiciones = np.linspace(inicio, fin, 16)
        frames = [int(round(p)) for p in posiciones]
        ventanas.append((inicio, fin, frames))

    return ventanas


PUNTOS_CABEZA = {
    "left_eye", "right_eye", "nose", "upper_jaw", "lower_jaw",
    "mouth_end_left", "mouth_end_right", "left_earbase", "left_earend",
    "right_earbase", "right_earend"
}

def calcular_bounding_box(filas_frame: list[dict], padding: float = 0.1) -> tuple[int, int, int, int] | None:
    """Calcula el rectángulo envolvente de la cara en un frame, ignorando el resto del cuerpo."""
    puntos_x = []
    puntos_y = []

    for fila in filas_frame:
        if fila.get("bodypart") in PUNTOS_CABEZA and "x" in fila and "y" in fila and float(fila.get("confianza", 0)) > 0.5:
            puntos_x.append(float(fila["x"]))
            puntos_y.append(float(fila["y"]))

    if not puntos_x or not puntos_y:
        return None

    x_min, x_max = min(puntos_x), max(puntos_x)
    y_min, y_max = min(puntos_y), max(puntos_y)

    ancho = x_max - x_min
    alto = y_max - y_min

    # Aplicar padding
    p_x = ancho * padding
    p_y = alto * padding

    return (
        int(max(0, x_min - p_x)),
        int(max(0, y_min - p_y)),
        int(x_max + p_x),
        int(y_max + p_y)
    )


def renderizar_video(ruta_in: str, ruta_out: str, pose_por_frame: dict, predicciones: list[dict]):
    """Toma el video original y genera un nuevo .mp4 con las cajas y etiquetas superpuestas."""
    import cv2
    from equinos.clasificacion import CLASES

    # Colores (BGR) para OpenCV
    color_map = {
        "Sereno": (143, 157, 42),       # #2a9d8f
        "Interesado": (106, 196, 233),  # #e9c46a
        "Disgustado": (81, 111, 231)    # #e76f51
    }

    cap = cv2.VideoCapture(ruta_in)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {ruta_in}")

    ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # El OpenCV de pip no trae el codificador H.264 (`avc1` falla al abrir), así que se escribe
    # `mp4v` y al final se recodifica a H.264 con ffmpeg, que es lo que reproducen los navegadores.
    ruta_cruda = Path(ruta_out).with_name(Path(ruta_out).stem + "_mp4v.mp4")
    out = cv2.VideoWriter(str(ruta_cruda), cv2.VideoWriter_fourcc(*"mp4v"), fps, (ancho, alto))
    if not out.isOpened():
        raise RuntimeError("No se pudo crear el video de salida.")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Bounding Box
        bbox = None
        if frame_idx in pose_por_frame:
            bbox = calcular_bounding_box(pose_por_frame[frame_idx])
            if bbox:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

        # 2. Etiqueta del modelo
        # Buscamos la predicción cuya ventana cubra este frame
        # o la ventana más cercana.
        pred_actual = None
        for p in predicciones:
            if p["frame_inicio"] <= frame_idx <= p["frame_fin"]:
                pred_actual = p
                break

        # Si estamos antes de la primera ventana, usamos la primera.
        # Si estamos después de la última, usamos la última.
        if not pred_actual and predicciones:
            if frame_idx < predicciones[0]["frame_inicio"]:
                pred_actual = predicciones[0]
            elif frame_idx > predicciones[-1]["frame_fin"]:
                pred_actual = predicciones[-1]

        if pred_actual:
            clase = pred_actual["clase"]
            color = color_map.get(clase, (255, 255, 255))
            texto = f"{clase}"

            # Dibujar etiqueta cerca del bbox si existe, sino arriba a la izquierda
            txt_x, txt_y = 50, 50
            if bbox:
                txt_x, txt_y = bbox[0], max(30, bbox[1] - 10)

            # Fondo negro para el texto
            (tw, th), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            cv2.rectangle(frame, (txt_x, txt_y - th - 5), (txt_x + tw, txt_y + 5), (0, 0, 0), -1)
            cv2.putText(frame, texto, (txt_x, txt_y), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    _a_h264(ruta_cruda, ruta_out)


def _a_h264(ruta_in, ruta_out):
    """Recodifica a H.264 yuv420p con `faststart`. Sin ffmpeg deja el mp4v (puede no verse en el navegador)."""
    ruta_in = Path(ruta_in)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        shutil.move(ruta_in, ruta_out)
        return
    try:
        subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-i", str(ruta_in), "-c:v", "libx264",
             "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "23", "-movflags", "+faststart",
             "-an", str(ruta_out)],
            check=True)
    finally:
        ruta_in.unlink(missing_ok=True)


def procesar_video_continuo(ruta_video: str, ruta_out: str, progreso=None, previo: dict | None = None):
    """Orquesta todo el flujo de monitoreo continuo para un video usando la FUSIÓN.

    `previo`: el `info` de `inferencia.clasificar_video(..., info={"conservar": True})`. Si trae la
    copia normalizada y la pose completa, se reutilizan y no se repite DeepLabCut. Quien lo pasó
    se ocupa de borrar `previo["carpeta"]`.
    """
    import tempfile
    import cv2
    from equinos import inferencia, fusion, alineado, pose as sel_pose, secuencia
    from equinos.video import normalizar_video

    avisar = progreso or (lambda e, d="": None)
    p = inferencia.parametros_etapa02()

    previo = previo or {}
    propia = None
    try:
        if previo.get("ruta_normalizada") and previo.get("filas_pose") is not None:
            ruta_norm = str(previo["ruta_normalizada"])
            filas_pose = previo["filas_pose"]
            total, fps = int(previo["total"]), float(previo["fps"])
        else:
            avisar("leer", "normalizando el video (escala, rotación, fps)...")
            propia = Path(tempfile.mkdtemp(prefix="monitoreo_"))
            ruta_norm = str(propia / "normalizado.mp4")
            meta = normalizar_video(ruta_video, ruta_norm)
            total, fps = meta["frames"], meta["fps"]
            if total <= 0:
                raise ValueError("Video vacío o ilegible.")
            avisar("pose", f"estimando keypoints en {total} fotogramas")
            filas_pose = inferencia._pose_completa(ruta_norm, total, p)
        if not filas_pose:
            raise ValueError("No se encontró ningún caballo en el video.")

        pose_por_frame = {}
        for fila in filas_pose:
            pose_por_frame.setdefault(int(fila["orden"]), []).append(fila)

        avisar("alinear", "generando ventanas y alineando caras")
        geo = alineado.geometria()
        destino = alineado.destino_canonico(p)
        pares = sel_pose.seleccionar_par(filas_pose, bodypart_hocico=p["BODYPART_HOCICO"])

        matrices = {}
        for par in pares:
            matriz = geo["similitud_desde_dos_puntos"](
                (float(par["x_ojo"]), float(par["y_ojo"])),
                (float(par["x_hocico"]), float(par["y_hocico"])),
                destino["dest_ojo"], destino["dest_hocico"])
            if matriz is not None:
                matrices[int(par["orden"])] = matriz

        # Qué fotogramas entran en cada ventana. Se eligen con la misma regla que usa la
        # clasificación del clip (`secuencia.elegir_frames`): sólo entre los fotogramas con ojo y
        # hocico medidos con confianza, anclando cada uno a una medición real. Muestrear 16
        # posiciones fijas sin mirar la validez (la versión original) dejaba sin ninguna ventana
        # a casi todos los videos, porque sólo entre el 25 y el 40 % de los fotogramas pasa el umbral.
        validos = secuencia.frames_validos(pares, p["CONF_MINIMA_KEYPOINT"])
        confianzas = secuencia.confianzas_del_par(pares, p["CONF_MINIMA_KEYPOINT"])
        elegidas = []
        for inicio, fin, _ in generar_ventanas(total, fps, duracion_seg=3.0, stride_seg=1.0):
            en_ventana = [v for v in validos if inicio <= v <= fin]
            sel = secuencia.elegir_frames(
                [v - inicio for v in en_ventana], total=fin - inicio + 1, fps=fps,
                minimo=secuencia.MINIMO_FRAMES,
                confianzas={v - inicio: confianzas[v] for v in en_ventana if v in confianzas})
            frames = [inicio + o for o in sel.ordenes]
            # El modelo espera exactamente 16 fotogramas; una ventana con menos se saltea.
            if len(frames) == secuencia.FRAMES_POR_CLIP and all(f in matrices for f in frames):
                elegidas.append((inicio, fin, frames))

        # Recortes alineados y características, sólo de los fotogramas que alguna ventana usa
        necesarios = sorted({f for _, _, frames in elegidas for f in frames})
        recortes_dict = {}
        cap = cv2.VideoCapture(ruta_norm)
        for f in necesarios:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, frame = cap.read()
            if ok and frame is not None:
                recortes_dict[f] = alineado.recortar(frame, matrices[f], destino)
        cap.release()

        avisar("caracteristicas", "extrayendo features visuales con el backbone")
        vectores_dict = {}
        ordenes_validos = sorted(recortes_dict.keys())
        batch_size = 32
        for i in range(0, len(ordenes_validos), batch_size):
            lote_ord = ordenes_validos[i:i+batch_size]
            lote_rec = [recortes_dict[o] for o in lote_ord]
            lote_vec = inferencia._caracteristicas_de(lote_rec)
            for ord_val, vec in zip(lote_ord, lote_vec):
                vectores_dict[ord_val] = vec

        avisar("inferencia", "corriendo la fusión sobre las ventanas")
        try:
            paquete = fusion.cargar_paquete()
        except Exception as e:
            raise RuntimeError(f"Error al cargar el paquete de Fusión: {e}")

        resultados = []

        for inicio, fin, frames in elegidas:
            if not all(f in vectores_dict for f in frames):
                continue
            filas_ventana = []
            for i, f_idx in enumerate(frames):
                for kp in pose_por_frame.get(f_idx, ()):
                    filas_ventana.append(dict(kp, orden=i))
            vectores_ventana = [vectores_dict[f] for f in frames]

            vector_pose = fusion.features_de_pose(filas_ventana, frames, fps)
            vector_visual = np.stack(vectores_ventana)  # (16, 512)

            prob = fusion.probabilidades(paquete, vector_pose, vector_visual)
            probabilidades = prob["fusion"]
            clase_ganadora = max(probabilidades, key=probabilidades.get)

            resultados.append({
                "frame_inicio": inicio,
                "frame_fin": fin,
                "centro": (inicio + fin) // 2,
                "clase": clase_ganadora,
                "probabilidades": probabilidades
            })

        avisar("render", "dibujando cajas y etiquetas en el video")
        # Renderizamos sobre el video normalizado, que es el que concuerda frame a frame
        renderizar_video(ruta_norm, ruta_out, pose_por_frame, resultados)

    finally:
        if propia is not None:
            shutil.rmtree(propia, ignore_errors=True)

    return resultados
