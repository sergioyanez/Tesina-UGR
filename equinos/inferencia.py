"""De un video a una predicción, o a una negativa explicada.

Es lo que la interfaz del objetivo 4 envuelve. La interfaz es intercambiable; esto no.

**Nada de lo que decide algo se escribe acá.** Los umbrales, el encuadre y el keypoint del hocico
se leen de la celda de parámetros del notebook 02; la geometría se extrae de sus celdas; la
selección de keypoints sale de `equinos.pose`; las características, de `equinos.caracteristicas`;
el modelo y la confiabilidad, de `equinos.clasificacion`.

Lo único que vive acá es el pegamento imperativo —abrir el video, recorrer los frames, llamar al
detector— y ese pegamento no toma ninguna decisión. La razón es que si la inferencia difiriera del
preprocesamiento que produjo el conjunto evaluado, aunque sea en un umbral, las entradas serían de
otra distribución y **la confiabilidad medida dejaría de aplicar sin que nada falle**. Es la misma
clase de error silencioso que el proyecto ya encontró con el espejo de videos a medio copiar.

**Negarse es una respuesta de primera clase.** El modelo consume clips íntegros —16 frames
alineados— y sobre el dataset propio sólo 359 de 393 clips (91,3 %) lo fueron. Para un video
cualquiera la negativa sigue siendo una respuesta esperable (hay videos sin cara medida), no un
caso de borde. Predecir con los frames
que haya daría una etiqueta cuya confiabilidad nadie midió.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

from equinos import caracteristicas, clasificacion, rutas

# Qué modelo predice si no se pide otro. Desde la comparación de modelos (`comparacion/DECISION.md`)
# es la fusión pose + visual; el MIL queda como opción explícita (`modelo="mil"`).
MODELOS = ("fusion", "mil")
MODELO_POR_OMISION = "fusion"

NOTEBOOK_ETAPA02 = "02_roi_y_normalizacion.ipynb"
FUNCIONES_GEOMETRIA = ("similitud_desde_dos_puntos", "roi_desde_keypoints")

# Frames que el modelo consume por clip. No es configurable: es la forma de su entrada.
FRAMES_POR_CLIP = 16

# Duración máxima admitida, en segundos. El modelo se midió sobre clips de 5–7 s; los 16 frames
# salen de una ventana de a lo sumo 7 s (`secuencia.VENTANA_MAXIMA_SEG`). Sobre un video de dos
# minutos esa ventana sería un tramo arbitrario del video, que el modelo no vio al entrenarse. La
# entrada sería de otra distribución y la confiabilidad medida no aplicaría.
#
# Se acepta con holgura sobre los 7 s del diseño para no rechazar una toma un poco más larga.
DURACION_MAXIMA_SEG = 15.0


class Negativa(NamedTuple):
    """El sistema no puede predecir, y dice por qué.

    NO es un error de la interfaz ni del video del usuario: es el resultado esperable del
    preprocesamiento en la mayoría de los casos.
    """

    motivo: str
    detalle: str
    frames_alineados: int

    @property
    def hay_prediccion(self) -> bool:
        return False


class Prediccion(NamedTuple):
    """La clase predicha, sus probabilidades y la confiabilidad medida de esa clase."""

    clase: str
    probabilidades: dict
    precision: float
    acertadas: int
    veces_que_lo_dijo: int
    texto_confiabilidad: str
    frames_alineados: int
    atencion: list[float]
    clases_por_frame: list[str]
    # Qué modelo predijo. El MIL trae atención por frame; la fusión no (queda vacía) y en cambio
    # trae en `detalle` las probabilidades de cada rama y el peso de la pose.
    modelo: str = "mil"
    detalle: dict | None = None
    # Qué tramo del video se analizó: `inicio_seg`, `fin_seg` (de la ventana elegida) y
    # `duracion_seg` (del video entero). Lo completa `clasificar_video`; vacío si se predijo sin video.
    tramo: dict | None = None

    @property
    def hay_prediccion(self) -> bool:
        return True


MOTIVOS = {
    "SIN_VIDEO": "No se pudo leer el video.",
    "VIDEO_CORTO": "El video es demasiado corto para extraer 16 fotogramas.",
    "VIDEO_LARGO": "El video es más largo de lo que el modelo vio al entrenarse.",
    "SIN_DETECCION": "No se encontró un equino en el video.",
    "SIN_KEYPOINTS": "Se encontró al animal, pero no sus keypoints faciales con confianza "
                     "suficiente.",
    "ALINEACION_INSUFICIENTE": "No se pudieron alinear los 16 fotogramas que el modelo necesita.",
    "VENTANA_CORTA": "No hay un tramo continuo del video con la cara medida el tiempo suficiente.",
}


# --- lo que se lee de la etapa 02 en vez de duplicarse -----------------------------------------

def _celdas(notebook: str | Path | None = None):
    ruta = Path(notebook) if notebook else Path(__file__).resolve().parent.parent / NOTEBOOK_ETAPA02
    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe {ruta}. La inferencia lee de ahí los umbrales y el encuadre para no "
            "duplicarlos: duplicarlos deja que diverjan y la confiabilidad medida deje de aplicar.")
    return json.loads(ruta.read_text(encoding="utf-8"))["cells"]


def parametros_etapa02(notebook: str | Path | None = None) -> dict:
    """Los parámetros declarados en la celda `parameters` del notebook 02.

    Se leen, no se copian. Copiarlos los deja divergir el día que alguien barre un umbral, y el
    síntoma sería una interfaz que informa una confiabilidad que ya no corresponde a su entrada.
    """
    for celda in _celdas(notebook):
        if "parameters" in celda.get("metadata", {}).get("tags", []):
            espacio: dict = {}
            exec(compile("".join(celda["source"]), "etapa02_parametros", "exec"), espacio)
            return {k: v for k, v in espacio.items() if k.isupper()}
    raise ValueError(f"{NOTEBOOK_ETAPA02} no tiene una celda etiquetada `parameters`.")


def geometria_etapa02(notebook: str | Path | None = None) -> dict:
    """Las funciones geométricas del notebook 02, extraídas y no reimplementadas."""
    for celda in _celdas(notebook):
        if celda["cell_type"] != "code":
            continue
        fuente = "".join(celda["source"])
        if all(f"def {f}" in fuente for f in FUNCIONES_GEOMETRIA):
            espacio: dict = {}
            exec(compile(fuente, "etapa02_geometria", "exec"), espacio)
            return espacio
    raise ValueError(
        f"Ninguna celda de {NOTEBOOK_ETAPA02} define {' y '.join(FUNCIONES_GEOMETRIA)}. "
        "Si se movieron de celda, actualizar FUNCIONES_GEOMETRIA.")


def campo_de_vision(parametros: dict | None = None):
    """El encuadre vigente, armado con los parámetros declarados en la etapa 02."""
    from equinos import encuadre

    p = parametros or parametros_etapa02()
    return encuadre.CampoDeVision(
        p["AIRE_SOBRE_OJO"], p["AIRE_BAJO_HOCICO"], p["SEMIANCHO_LATERAL"]), p["ANCHO_RECORTE"]


# --- la parte que decide, aislada de la GPU y del video ----------------------------------------

def predecir_desde_caracteristicas(vectores, filas_pose, precisiones=None,
                                   modelo=None) -> Prediccion | Negativa:
    """Predice a partir de las características y pose de los frames alineados. Sin video."""
    import torch
    import torch.nn.functional as F

    # Exactamente 16, no "al menos 12". Es el mismo criterio que `particion.tabla_clips` aplica
    # al conjunto de entrenamiento: una secuencia más corta entra igual al modelo —el padding de
    # `collate_fn` la acepta— y devuelve una etiqueta cuya confiabilidad nadie midió.
    if len(vectores) != FRAMES_POR_CLIP:
        return Negativa("ALINEACION_INSUFICIENTE",
                        MOTIVOS["ALINEACION_INSUFICIENTE"]
                        + f" Se alinearon {len(vectores)} de {FRAMES_POR_CLIP}.",
                        len(vectores))

    modelo, clases = modelo or clasificacion.cargar_produccion()
    
    # Reutilizamos MultimodalEquineDataset para el pipeline de características geométricas
    clip_name = filas_pose[0]["clip"] if filas_pose else "inferencia"
    dummy_clips = [{"clip": clip_name, "clase": clases[0]}]
    dummy_indice = {clip_name: 0}
    dummy_vectores = [vectores]
    
    ds = clasificacion.MultimodalEquineDataset(dummy_clips, dummy_vectores, dummy_indice, filas_pose)
    v_vis, v_geom, _ = ds[0]
    
    v_vis = v_vis.unsqueeze(0)
    v_geom = v_geom.unsqueeze(0)
    seq_len = v_vis.shape[1]
    attn_mask = torch.ones((1, seq_len), dtype=torch.bool)
    
    device = next(modelo.parameters()).device
    with torch.no_grad():
        logits, attn_weights = modelo(v_vis.to(device), v_geom.to(device), attn_mask.to(device), return_attn=True)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        atencion = attn_weights.cpu().numpy()[0].tolist()
        # Lectura por instancia, sólo para el gráfico de atención: no es una salida entrenada.
        logits_frame = modelo.logits_por_instancia(v_vis.to(device), v_geom.to(device))
        probs_frame = F.softmax(logits_frame, dim=-1).cpu().numpy()[0]

    clase = clases[probs.argmax()]
    probabilidades = {c: float(p) for c, p in zip(clases, probs)}

    precisiones = precisiones or clasificacion.precision_por_clase()
    p = precisiones[clase]
    return Prediccion(
        clase=clase,
        probabilidades=probabilidades,
        precision=p["precision"],
        acertadas=p["acertadas"],
        veces_que_lo_dijo=p["veces_que_lo_dijo"],
        texto_confiabilidad=clasificacion.texto_de_confiabilidad(clase, precisiones),
        frames_alineados=len(vectores),
        atencion=atencion,
        clases_por_frame=[clases[i] for i in probs_frame.argmax(axis=1)],
    )


def predecir_con_fusion(vectores, filas_pose, frames, fps, precisiones=None, paquete=None,
                        metricas=None, reserva=None) -> Prediccion | Negativa:
    """Predice con la fusión pose + visual a partir de lo que ya calculó el pipeline. Sin video.

    `vectores`: los 16 vectores visuales (16, 512). `filas_pose`: pose de los 16 frames con `orden`
    = posición en la secuencia. `frames`: índice de fotograma del video de cada uno de los 16 (los
    `ordenes` de `preparar_clip`), y `fps` del video: las features dinámicas dividen por el tiempo
    real entre frames. Las 160 features salen de `vector_del_clip`, igual que en el entrenamiento.
    """
    from equinos import fusion

    if len(vectores) != FRAMES_POR_CLIP:
        return Negativa("ALINEACION_INSUFICIENTE",
                        MOTIVOS["ALINEACION_INSUFICIENTE"]
                        + f" Se alinearon {len(vectores)} de {FRAMES_POR_CLIP}.",
                        len(vectores))

    paquete = paquete or fusion.cargar_paquete()
    vector_pose = fusion.features_de_pose(filas_pose, frames, fps)
    prob = fusion.probabilidades(paquete, vector_pose, vectores)
    probabilidades = prob["fusion"]
    clase = max(probabilidades, key=probabilidades.get)

    precisiones = precisiones or fusion.precision_por_clase()
    p = precisiones[clase]
    return Prediccion(
        clase=clase,
        probabilidades=probabilidades,
        precision=p["precision"],
        acertadas=p["acertadas"],
        veces_que_lo_dijo=p["veces_que_lo_dijo"],
        texto_confiabilidad=fusion.texto_de_confiabilidad(clase, precisiones, metricas, reserva),
        frames_alineados=len(vectores),
        atencion=[],
        clases_por_frame=[],
        modelo="fusion",
        detalle={"pose": prob["pose"], "visual": prob["visual"], "peso_pose": prob["peso_pose"],
                 "precision_reserva": (reserva or fusion.precision_reserva())[clase]},
    )


# --- la cadena completa ------------------------------------------------------------------------

def _sin_progreso(etapa: str, detalle: str = "") -> None:
    pass


def preparar_clip(ruta_video, progreso=None, parametros=None, normalizar=True, info=None):
    """Video → 16 recortes alineados, o una `Negativa` con el motivo.

    `normalizar=True` (por omisión) hace con el video lo mismo que la segmentación del notebook 01
    le hizo a cada clip: lo lee con `CAP_PROP_ORIENTATION_AUTO = 1` y lo recodifica con `mp4v`
    (`equinos.video.normalizar_video`); DeepLabCut y los recortes salen de esa copia, no del
    original. Sin eso un video de celular con rotación declarada llegaba acostado al modelo de pose
    (69 de 274 videos). `normalizar=False` reproduce la lectura anterior y existe sólo para medir
    la diferencia. `info`, si es un dict, se completa con lo que se eligió (`ordenes`, `total`,
    `fps`, `rotacion`), para poder comparar contra la caché de entrenamiento.

    **No decide nada por su cuenta.** Qué frames entran y cómo se alinean lo resuelve
    `equinos.alineado.preparar_secuencia`, que es exactamente la función que usa la generación del
    dataset. Acá sólo queda el pegamento imperativo: abrir el video, correr la pose, leer los
    frames elegidos y aplicar las transformadas.

    Antes esto era una segunda implementación del preprocesamiento, y había divergido: muestreaba
    16 posiciones fijas sin mirar la validez, gateaba la pose detrás de YOLO, y no tenía ni el
    rescate por interpolación ni el recorte por caja que la etapa 02 sí aplicaba. El efecto era
    una interfaz más estricta que el pipeline que fabricó su conjunto de entrenamiento: rechazaba
    videos que el dataset habría aceptado, informando una confiabilidad medida sobre otra
    distribución. Nada de eso fallaba.

    Devuelve `(recortes, filas_pose)` donde `filas_pose` ya viene con `orden` renumerado a la
    posición en la secuencia, que es lo que `MultimodalEquineDataset` espera.

    Si `info` trae `conservar=True`, la copia normalizada NO se borra y la pose completa queda en
    `info["filas_pose"]` (con la copia en `info["ruta_normalizada"]` y su carpeta en
    `info["carpeta"]`, que borra quien la pidió). Es para que `monitoreo` dibuje sobre el mismo
    video sin repetir la pasada de DeepLabCut, que es lo que más tarda.
    """
    import shutil
    import tempfile

    from equinos import video

    if not normalizar:
        return _preparar_desde(ruta_video, progreso, parametros, info, rotacion=None)

    avisar = progreso or _sin_progreso
    avisar("leer", "normalizando el video como se hizo con los clips")
    # En el temporal del sistema y no bajo la raíz de datos: DeepLabCut deja auxiliares junto a
    # la entrada y la raíz puede ser de sólo lectura o estar en Drive.
    carpeta = Path(tempfile.mkdtemp(prefix="equinos_inf_"))
    conservar = bool(info is not None and info.get("conservar"))
    try:
        copia = carpeta / (Path(ruta_video).stem + ".mp4")
        meta = video.normalizar_video(ruta_video, copia)
        if conservar:
            info.update(ruta_normalizada=copia, carpeta=carpeta)
        if meta["frames"] <= 0:
            return Negativa("SIN_VIDEO", MOTIVOS["SIN_VIDEO"], 0)
        return _preparar_desde(copia, progreso, parametros, info, rotacion=meta["rotacion"])
    finally:
        if not conservar:
            shutil.rmtree(carpeta, ignore_errors=True)


def _preparar_desde(ruta_video, progreso, parametros, info, rotacion):
    """El cuerpo de `preparar_clip`, sobre un video que ya está en la forma que se quiere leer."""
    import cv2

    from equinos import alineado, secuencia

    avisar = progreso or _sin_progreso
    p = parametros or parametros_etapa02()

    avisar("leer", "abriendo el video")
    captura = cv2.VideoCapture(str(ruta_video))
    total = int(captura.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = captura.get(cv2.CAP_PROP_FPS) or 30.0
    captura.release()
    if info is not None:
        info.update(total=total, fps=float(fps), rotacion=rotacion)

    if total <= 0:
        return Negativa("SIN_VIDEO", MOTIVOS["SIN_VIDEO"], 0)
    if total < secuencia.MINIMO_FRAMES:
        return Negativa("VIDEO_CORTO",
                        MOTIVOS["VIDEO_CORTO"] + f" Tiene {total} fotogramas (mínimo "
                        f"{secuencia.MINIMO_FRAMES}).", 0)

    duracion = total / fps if fps > 0 else 0.0
    if duracion > DURACION_MAXIMA_SEG:
        return Negativa(
            "VIDEO_LARGO",
            MOTIVOS["VIDEO_LARGO"]
            + f" Dura {duracion:.0f} s y el máximo admitido son {DURACION_MAXIMA_SEG:.0f}. "
            "Recortá el momento de interés a una toma de 5 a 7 segundos.",
            0)

    # La pose corre sobre el video ENTERO y se conserva entera. Antes se la gateaba detrás de una
    # detección de YOLO sobre los 16 frames muestreados: si el detector fallaba en esos 16, el
    # video se rechazaba sin haberle dado a DeepLabCut —que trae su propio detector— la
    # oportunidad de encontrar la cara. La etapa 02 nunca hizo ese gateo.
    avisar("pose", f"estimando keypoints en {total} fotogramas")
    filas_pose = _pose_completa(ruta_video, total, p)
    if info is not None:
        info["filas_pose"] = filas_pose
    if not filas_pose:
        return Negativa("SIN_DETECCION", MOTIVOS["SIN_DETECCION"], 0)

    avisar("alinear", "eligiendo la ventana y alineando")
    prep = alineado.preparar_secuencia(filas_pose, total=total, fps=fps, p=p,
                                       minimo=secuencia.FRAMES_POR_CLIP)
    if not prep.hay_secuencia:
        return Negativa(
            prep.motivo,
            MOTIVOS.get(prep.motivo, MOTIVOS["ALINEACION_INSUFICIENTE"])
            + f" Se midió el par ojo-hocico en {prep.validos} de {total} fotogramas, pero no hay "
            f"ninguna ventana con {secuencia.FRAMES_POR_CLIP} mediciones seguidas dentro de "
            f"{secuencia.VENTANA_MAXIMA_SEG:.0f} s.",
            prep.validos)

    if info is not None:
        info.update(ordenes=[int(o) for o in prep.ordenes], validos=prep.validos,
                    span_seg=float(prep.span_seg))

    por_orden: dict[int, list] = {}
    for fila in filas_pose:
        por_orden.setdefault(int(fila["orden"]), []).append(fila)

    captura = cv2.VideoCapture(str(ruta_video))
    recortes, pose_secuencia = [], []
    for posicion, (orden, matriz) in enumerate(zip(prep.ordenes, prep.matrices)):
        captura.set(cv2.CAP_PROP_POS_FRAMES, int(orden))
        ok, imagen = captura.read()
        if not ok or imagen is None:
            continue
        recortes.append(alineado.recortar(imagen, matriz, prep.destino))
        # `orden` renumerado a la posición en la secuencia: es lo que el modelo espera, y pasarle
        # el índice del frame del video hace que el canal geométrico lea otros fotogramas.
        pose_secuencia.extend(dict(f, orden=posicion) for f in por_orden.get(orden, ()))
    captura.release()

    if len(recortes) != secuencia.FRAMES_POR_CLIP:
        return Negativa(
            "ALINEACION_INSUFICIENTE",
            MOTIVOS["ALINEACION_INSUFICIENTE"]
            + f" Se pudieron leer {len(recortes)} de {secuencia.FRAMES_POR_CLIP} fotogramas.",
            len(recortes))
    return recortes, pose_secuencia


def _pose_completa(ruta_video, total, parametros):
    """La pose de TODOS los frames del video, en formato largo. Una sola pasada de DeepLabCut."""
    import contextlib
    import io

    import deeplabcut

    from equinos import pose as sel_pose

    ruta_video = str(ruta_video)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        resultado = deeplabcut.video_inference_superanimal(
            [ruta_video],
            superanimal_name="superanimal_quadruped",
            model_name="hrnet_w32",
            detector_name="fasterrcnn_resnet50_fpn_v2",
        )
    _limpiar_auxiliares(ruta_video)

    if not isinstance(resultado, dict) or not resultado:
        return []
    tabla = next(iter(resultado.values()))
    return sel_pose.a_formato_largo(
        tabla, list(enumerate(range(total))), Path(ruta_video).name)


def _limpiar_auxiliares(ruta_video):
    """DeepLabCut deja un .h5, un .pickle y un video etiquetado junto al archivo de entrada.

    El video etiquetado pesa casi como el original. Sin limpiarlos, cada inferencia de la interfaz
    duplicaría el espacio ocupado por el video que subió el usuario.
    """
    ruta = Path(ruta_video)
    for archivo in ruta.parent.glob(f"{ruta.stem}_*"):
        if archivo.name == ruta.name:
            continue
        if archivo.suffix in (".h5", ".pickle") or archivo.name.endswith(".mp4"):
            archivo.unlink(missing_ok=True)


def tramo_analizado(ordenes, total, fps) -> dict:
    """El tramo del video que cubren los 16 fotogramas elegidos, en segundos.

    `ordenes`: índices de fotograma (los de `preparar_clip`); `total` y `fps`: los del video ya
    normalizado, que es el que se leyó. El fin es el final del último fotograma elegido.
    """
    fps = fps if fps and fps > 0 else 30.0
    duracion = total / fps
    inicio = min(ordenes) / fps
    fin = min((max(ordenes) + 1) / fps, duracion)
    return {"inicio_seg": float(inicio), "fin_seg": float(fin), "duracion_seg": float(duracion)}


def clasificar_video(ruta_video, progreso=None, normalizar=True, modelo=MODELO_POR_OMISION,
                     info=None):
    """Punto de entrada de la interfaz: un video, una `Prediccion` o una `Negativa`.

    `modelo`: "fusion" (por omisión, el ganador de la comparación) o "mil" (el MIL multimodal).
    `info`: dict opcional que se completa con lo elegido; ver `preparar_clip` (`conservar`).
    """
    if modelo not in MODELOS:
        raise ValueError(f"Modelo desconocido: {modelo!r}. Opciones: {', '.join(MODELOS)}")
    avisar = progreso or _sin_progreso
    info = {} if info is None else info
    preparado = preparar_clip(ruta_video, progreso=progreso, normalizar=normalizar, info=info)
    if isinstance(preparado, Negativa):
        return preparado
    
    recortes, filas_pose = preparado

    avisar("caracteristicas", "extrayendo características con el backbone congelado")
    vectores = _caracteristicas_de(recortes)
    avisar("predecir", "clasificando")
    if modelo == "mil":
        resultado = predecir_desde_caracteristicas(vectores, filas_pose)
    else:
        resultado = predecir_con_fusion(vectores, filas_pose, info["ordenes"], info["fps"])
    if resultado.hay_prediccion:
        resultado = resultado._replace(
            tramo=tramo_analizado(info["ordenes"], info["total"], info["fps"]))
    return resultado


def _caracteristicas_de(recortes):
    """Backbone congelado sobre los recortes ya alineados, con la misma normalización.

    **El viaje por JPEG no es un descuido: es obligatorio.** Los recortes del conjunto evaluado se
    escribieron a disco como `.jpg` y se volvieron a leer para extraer sus características, así que
    el modelo se midió sobre imágenes que pasaron por esa compresión con pérdida. Extraer del
    tensor en memoria, que es lo que parece más limpio, da características un 9,4 % distintas en
    promedio y le entrega al modelo una distribución sobre la que nadie lo evaluó.

    Verificado el 28/08/2026 sobre un clip del propio dataset: con el viaje por JPEG la diferencia
    contra la caché es EXACTAMENTE cero; sin él, 0,85 de diferencia máxima sobre vectores de norma
    26. Quien quiera «optimizar» esta línea tiene que rehacer esa verificación primero.
    """
    import cv2
    import numpy as np
    import torch

    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = caracteristicas._cargar_backbone(dispositivo)

    comprimidos = []
    for recorte in recortes:
        ok, buffer = cv2.imencode(".jpg", recorte)
        if not ok:
            raise RuntimeError("No se pudo comprimir el recorte a JPEG.")
        comprimidos.append(cv2.imdecode(buffer, cv2.IMREAD_COLOR))
    recortes = comprimidos

    lote = np.stack([r[:, :, ::-1] for r in recortes]).astype(np.float32) / 255.0  # BGR -> RGB
    tensor = torch.from_numpy(lote.copy()).permute(0, 3, 1, 2)
    media = torch.tensor(caracteristicas.MEDIA_IMAGENET).view(1, 3, 1, 1)
    desvio = torch.tensor(caracteristicas.DESVIO_IMAGENET).view(1, 3, 1, 1)
    with torch.no_grad():
        salida = extractor(((tensor - media) / desvio).to(dispositivo)).flatten(1)
    return salida.cpu().numpy()
