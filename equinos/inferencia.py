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
alineados— y sobre el dataset propio sólo 141 de 368 clips (38 %) lo fueron. Para un video
cualquiera, la negativa es el resultado MÁS PROBABLE, no un caso de borde. Predecir con los frames
que haya daría una etiqueta cuya confiabilidad nadie midió.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

from equinos import caracteristicas, clasificacion, rutas

NOTEBOOK_ETAPA02 = "02_roi_y_normalizacion.ipynb"
FUNCIONES_GEOMETRIA = ("similitud_desde_dos_puntos", "roi_desde_keypoints")

# Frames que el modelo consume por clip. No es configurable: es la forma de su entrada.
FRAMES_POR_CLIP = 16

# Duración máxima admitida, en segundos. El modelo se midió sobre clips de 5–7 s, y los 16 frames
# se muestrean repartidos por todo el video: sobre un video de dos minutos eso da 16 fotogramas
# separados por ocho segundos, que no es una secuencia de expresión sino una colección de momentos
# distintos. La entrada sería de otra distribución y la confiabilidad medida no aplicaría.
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

    if not (12 <= len(vectores) <= FRAMES_POR_CLIP):
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
    )


# --- la cadena completa ------------------------------------------------------------------------

def _sin_progreso(etapa: str, detalle: str = "") -> None:
    pass


def preparar_clip(ruta_video, progreso=None, parametros=None):
    """Video → 16 recortes alineados, o una `Negativa` con el motivo.

    Reutiliza el detector, el modelo de pose, la selección de keypoints y el encuadre tal como
    están declarados en la etapa 02. Cuesta ~35 s por clip en GPU: 15 del detector y 20 de la
    pose. `progreso(etapa, detalle)` se llama al entrar en cada una.
    """
    import cv2
    import numpy as np

    avisar = progreso or _sin_progreso
    p = parametros or parametros_etapa02()
    geo_funcs = geometria_etapa02()
    campo, ancho = campo_de_vision(p)

    from equinos import encuadre

    destino = encuadre.destino(campo, ancho)

    avisar("leer", "abriendo el video")
    captura = cv2.VideoCapture(str(ruta_video))
    total = int(captura.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        captura.release()
        return Negativa("SIN_VIDEO", MOTIVOS["SIN_VIDEO"], 0)
    if total < 12:
        captura.release()
        return Negativa("VIDEO_CORTO",
                        MOTIVOS["VIDEO_CORTO"] + f" Tiene {total} fotogramas (mínimo 12).", 0)

    fps = captura.get(cv2.CAP_PROP_FPS) or 30.0
    duracion = total / fps if fps > 0 else 0.0
    if duracion > DURACION_MAXIMA_SEG:
        captura.release()
        return Negativa(
            "VIDEO_LARGO",
            MOTIVOS["VIDEO_LARGO"]
            + f" Dura {duracion:.0f} s y el máximo admitido son {DURACION_MAXIMA_SEG:.0f}. "
            "Recortá el momento de interés a una toma de 5 a 7 segundos.",
            0)

    num_a_extraer = min(total, FRAMES_POR_CLIP)
    indices = np.linspace(0, total - 1, num_a_extraer).astype(int)
    frames = []
    for i in indices:
        captura.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, imagen = captura.read()
        frames.append(imagen if ok else None)
    captura.release()
    if all(f is None for f in frames):
        return Negativa("SIN_VIDEO", MOTIVOS["SIN_VIDEO"], 0)

    avisar("detectar", f"buscando al equino en {FRAMES_POR_CLIP} fotogramas (~15 s)")
    cajas = _detectar(frames, p)
    if not any(c is not None for c in cajas):
        return Negativa("SIN_DETECCION", MOTIVOS["SIN_DETECCION"], 0)

    avisar("pose", "estimando keypoints faciales (~20 s)")
    pares, filas_pose = _keypoints(ruta_video, frames, indices, p)
    validos = sum(1 for par in pares if par is not None)
    if validos == 0:
        return Negativa("SIN_KEYPOINTS", MOTIVOS["SIN_KEYPOINTS"], 0)

    avisar("alinear", "recortando y alineando")
    recortes = []
    for imagen, par in zip(frames, pares):
        if imagen is None or par is None:
            continue
        matriz = geo_funcs["similitud_desde_dos_puntos"](
            par[0], par[1], destino["dest_ojo"], destino["dest_hocico"])
        if matriz is None:
            continue
        recortes.append(cv2.warpAffine(imagen, matriz, (destino["ancho"], destino["alto"]),
                                       borderMode=cv2.BORDER_REPLICATE))

    if not (12 <= len(recortes) <= FRAMES_POR_CLIP):
        return Negativa(
            "ALINEACION_INSUFICIENTE",
            MOTIVOS["ALINEACION_INSUFICIENTE"] + f" Se alinearon {len(recortes)} de "
            f"{FRAMES_POR_CLIP}. Es el resultado más frecuente: sobre el conjunto propio sólo "
            "el 38 % de los clips llegó a los 16.",
            len(recortes))
    return recortes, filas_pose


def _detectar(frames, parametros):
    """Detector de equinos, con el mismo umbral que declara la etapa 02."""
    import numpy as np
    from ultralytics import YOLO

    detector = YOLO("yolo11x.pt")
    clase_equino = next((i for i, n in detector.names.items()
                         if n.lower() in ("horse", "caballo")), None)
    if clase_equino is None:
        raise RuntimeError("El detector no expone una clase de equino.")

    cajas = []
    for imagen in frames:
        if imagen is None:
            cajas.append(None)
            continue
        resultado = detector.predict(imagen, classes=[clase_equino],
                                     conf=parametros["CONF_MINIMA_DETECCION"], verbose=False)[0]
        detecciones = resultado.boxes
        if detecciones is None or len(detecciones) == 0:
            cajas.append(None)
            continue
        confianzas = detecciones.conf.cpu().numpy()
        cajas.append(tuple(detecciones.xyxy.cpu().numpy()[int(np.argmax(confianzas))].tolist()))
    return cajas


def _keypoints(ruta_video, frames, indices, parametros):
    """Par ojo-hocico por frame, con el modelo y los criterios de la etapa 02.

    Devuelve `None` en los frames donde el par no supera el umbral de confianza. NO rellena ni
    interpola: el relleno temporal tiene un tope validado y relajarlo acá para que entren más
    videos sería inventar posiciones que nadie midió, justo cuando el resultado se le muestra a
    una persona.
    """
    import contextlib
    import io

    import deeplabcut

    from equinos import pose as sel_pose

    ruta_video = str(ruta_video)
    with contextlib.redirect_stdout(io.StringIO()):
        resultado = deeplabcut.video_inference_superanimal(
            [ruta_video],
            superanimal_name="superanimal_quadruped",
            model_name="hrnet_w32",
            detector_name="fasterrcnn_resnet50_fpn_v2",
        )
    _limpiar_auxiliares(ruta_video)

    if not isinstance(resultado, dict) or not resultado:
        return [None] * len(indices), []
    tabla = next(iter(resultado.values()))

    # A formato largo con `equinos.pose`, que es donde vive la decisión de qué individuo seguir.
    filas = sel_pose.a_formato_largo(
        tabla, list(enumerate(indices)), Path(ruta_video).name)

    if not filas:
        return [None] * len(indices), []

    pares = sel_pose.seleccionar_par(filas, bodypart_hocico=parametros["BODYPART_HOCICO"])
    umbral = parametros["CONF_MINIMA_KEYPOINT"]
    por_orden = {int(p["orden"]): p for p in pares}

    salida = []
    for orden in range(len(indices)):
        p = por_orden.get(orden)
        if (p is None or p.get("x_ojo") in (None, "") or p.get("x_hocico") in (None, "")
                or float(p.get("conf_ojo") or 0) < umbral
                or float(p.get("conf_hocico") or 0) < umbral):
            salida.append(None)
            continue
        salida.append(((float(p["x_ojo"]), float(p["y_ojo"])),
                       (float(p["x_hocico"]), float(p["y_hocico"]))))
    return salida, filas


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


def clasificar_video(ruta_video, progreso=None):
    """Punto de entrada de la interfaz: un video, una `Prediccion` o una `Negativa`."""
    avisar = progreso or _sin_progreso
    preparado = preparar_clip(ruta_video, progreso=progreso)
    if isinstance(preparado, Negativa):
        return preparado
    
    recortes, filas_pose = preparado

    avisar("caracteristicas", "extrayendo características con el backbone congelado")
    vectores = _caracteristicas_de(recortes)
    avisar("predecir", "clasificando")
    return predecir_desde_caracteristicas(vectores, filas_pose)


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
