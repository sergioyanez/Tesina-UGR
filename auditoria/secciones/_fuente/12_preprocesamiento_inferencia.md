# 12. Preprocesamiento, selección de frames y coherencia entrenamiento–inferencia

## 12.1 Cadena de preprocesamiento

| Etapa | Implementación | Evidencia |
|---|---|---|
| Inventario | OpenCV (fps, frames, resolución) | `inventario_videos.csv` (274) |
| Segmentación en clips de 5–7 s | `cv2.VideoWriter` **`mp4v`** (recodificación con pérdida), con `CAP_PROP_ORIENTATION_AUTO = 1`; videos más cortos que 5 s entran enteros con un piso de 1 s | notebook 01 (celdas del Paso 10) |
| Pose | DeepLabCut 3.0.1 `video_inference_superanimal`, **`superanimal_quadruped`**, detector `fasterrcnn_resnet50_fpn_v2`, red `hrnet_w32`, sobre **todos** los frames de cada clip (39 *bodyparts*) | `equinos/reproceso.py` (l. 195), `equinos/inferencia.py` (l. 311), `pose_todos_los_frames.csv` |
| Frame válido | Par ojo–hocico con confianza ≥ 0,50 (`CONF_MINIMA_KEYPOINT`) | `02_roi_y_normalizacion.ipynb` (parámetros), `secuencia.frames_validos` |
| Selección de 16 frames | Ver §12.2 | `equinos/secuencia.py` |
| Alineado | Transformada de similitud a una pose canónica (ojo y hocico fijos) con encuadre 224×343 que incluye las orejas; `borderMode = BORDER_REPLICATE` | `equinos/alineado.py`, `equinos/encuadre.py` |
| Recorte → JPEG (calidad 95) → lectura | El backbone ve la imagen **después** del viaje por JPEG | `equinos/dataset.py` (`CALIDAD_JPEG`) |
| Características | ResNet-18 congelada, normalización de ImageNet, 512 por frame | `equinos/caracteristicas.py` |
| Geometría | 3 puntos normalizados + 3 indicadores de validez por frame | `clasificacion.MultimodalEquineDataset` |

**Sobre el uso de DeepLabCut (devolución docente, §1.5 punto 4).** *No* se usa un modelo específico de caballos ni se ajustó (*fine-tuning*) con datos propios: se aplica tal cual el modelo preentrenado generalista **SuperAnimal-Quadruped**. El repositorio no contiene ninguna referencia a un subconjunto o *benchmark* de caballos (búsqueda de «Horse-10» / «Horse-30» sobre código, notebooks y documentos: sin resultados). **No pudo verificarse con los archivos disponibles** si entre los datos con que se entrenó SuperAnimal-Quadruped hay material de caballos (corresponde citar a Ye et al., 2024). La calidad de los keypoints propios no se midió contra anotación humana en el conjunto vigente: sólo hay 14 puntos anotados a mano (`keypoints_manuales.csv`), que el conjunto vigente ya no usa (los 5 744 frames provienen de detección propia).

## 12.2 Selección de los 16 frames

Método (`secuencia.elegir_frames`, documentado en el docstring del módulo): **no es muestreo uniforme sobre el clip, ni consecutivo, ni aleatorio**. Se busca la **ventana temporal contigua más larga** (≤ 7 s) que tenga ≥ 16 frames con par ojo–hocico válido y ningún tramo sin medición mayor a 0,5 s; dentro de ella se reparten 16 posiciones equiespaciadas y cada una se **ancla al frame medido más cercano** (tolerancia 0,1 s, eligiendo el de mayor confianza). Los 16 frames salen siempre de mediciones reales (no se interpola).

{{TABLA:ventana}}

{{FIG:fig_ventana_por_clase.png|Figura 12.1. Duración de la ventana de los 16 frames por clase (izquierda) y separación entre frames consecutivos (derecha).}}

**Posible sesgo (evidencia):** la ventana abarca el 96–98 % del clip, así que su duración depende de la duración del clip, y esta depende de la clase. El intervalo entre frames es **la mitad en `Disgustado` (mediana 0,19 s) que en `Interesado` y `Sereno` (0,39 s)**; un criterio de dos a uno en la forma de muestrear el tiempo está asociado a la clase (Kruskal–Wallis sobre la ventana, p ≈ 2·10⁻²⁶; el AUC de la ventana para separar `Disgustado` del resto es 0,87). Como el modelo es invariante al orden, esto no actúa como «velocidad» sino como **diferencia en la diversidad de poses** que aparecen entre los 16 frames. Es el mecanismo por el cual la duración del clip (F1 0,544 sola) puede entrar al modelo aunque no se le dé como variable. El aprovechamiento por clase (§3.1) también es desigual: 31 de los 35 clips con ventana < 2 s son `Disgustado`.

## 12.3 ¿Entrenamiento e inferencia ven la misma entrada?

La función que decide qué frames entran y cómo se alinean es **la misma** (`alineado.preparar_secuencia`), con los mismos parámetros leídos del notebook 02; también el extractor (`caracteristicas._cargar_backbone`), el viaje por JPEG (calidad 95) y la geometría (`MultimodalEquineDataset`). `tests/test_equivalencia.py` (15/15 pruebas del repositorio pasan) verifica que dataset e inferencia eligen los mismos frames y producen los mismos recortes. Diferencias, ordenadas por relevancia:

| Elemento | Entrenamiento | Inferencia (`app_inferencia.py`) | Diferencia / efecto |
|---|---|---|---|
| **Origen de los píxeles** | **Clips recodificados `mp4v`** (segmentación del notebook 01) | **Video original** del usuario | **Diferencia no cubierta por `test_equivalencia`** (que compara clip contra clip). Medido sobre 28 clips comparables: diferencia de píxel media 4,1/255; diferencia relativa de características 12 % (máx. 35 %); coseno 0,990; **la predicción cambia en 2 de 28 clips (7,1 %)**, con cambio máximo de probabilidad de 0,66 en un caso (`datos/entrenamiento_vs_inferencia.json`) |
| **Orientación del video** | Se aplica `CAP_PROP_ORIENTATION_AUTO = 1` al crear los clips (los clips quedan en orientación de visualización) | `inferencia.py` **no** la activa (0 apariciones); `cv2.VideoCapture` entrega los frames como están almacenados | 69 de 274 videos (25,2 %) declaran rotación de 90°; originan el 35,4 % de los clips válidos. Ver §12.5 |
| Unidad de entrada | Clip ≤ 7 s ya cortado | Video de hasta 15 s completo (`DURACION_MAXIMA_SEG`); se elige la mejor ventana ≤ 7 s | La ventana elegida puede no ser el tramo etiquetado; el modelo se midió sólo con clips ≤ 7 s |
| Lectura de frames | Frames de un archivo `.mp4` recodificado | `captura.set(POS_FRAMES)` + `read()` sobre el original (`.mp4`, `.mkv`, `.avi`) | La búsqueda por fotograma es inexacta en algunos contenedores; ver la negativa por «15 de 16 fotogramas» en §12.5 |
| Pose | DeepLabCut sobre el clip recodificado | DeepLabCut sobre el video original | Mismo modelo y parámetros; **el efecto de la recodificación sobre la pose no pudo verificarse** (requiere re-ejecutar DeepLabCut, ≈ horas de GPU) |
| Checkpoint | 4 modelos de fold (CV) | `mil_multimodal_produccion.pt` (359 clips, 29 épocas, sin validación) | Esperable; el rendimiento del modelo desplegado se **infiere** de la CV, no se midió directamente |
| Arquitectura, 16 frames, normalización, extracción | Iguales | Iguales (`MILFusionModel`, 16 exactos, constantes ImageNet, JPEG 95) | Sin diferencia |
| Detector | `fasterrcnn_resnet50_fpn_v2` de DeepLabCut | Ídem (YOLO ya no participa) | Sin diferencia |

**Evaluación.** Las diferencias de origen de píxeles y de orientación son **debilidades reales** del eslabón entrenamiento–inferencia que el proyecto no medía. Su efecto agregado no se puede acotar con estos datos más que con las pruebas de esta auditoría; la confiabilidad que muestra la interfaz (74 % de acierto, precisión por clase) corresponde a clips de entrenamiento y **no necesariamente aplica a videos nuevos subidos por un usuario**.

## 12.4 Documentación desactualizada que contradice al código (menor)

- `equinos/inferencia.py` (docstring del módulo): afirma que «sobre el dataset propio sólo 141 de 368 clips (38 %) lo fueron» íntegros; hoy son 359 de 393 (91,3 %). Y el comentario de `DURACION_MAXIMA_SEG` dice que los 16 frames «se muestrean repartidos por todo el video», lo cual ya no es cierto (ventana ≤ 7 s).
- `app_inferencia.py` (docstring): «cuando el modelo dice `Disgustado`, acierta una de cada tres veces»; el valor vigente es 71 de 96 (74 %).
- `openspec/project.md` conserva una tabla «Estado actual» con `249 videos → 368 clips → 5888 frames` y secciones de riesgos con 234 videos / 342 clips; las cifras vigentes son 274 / 393 / 359.

## 12.5 Prueba de la interfaz sobre videos del propio dataset

Se ejecutó `inferencia.clasificar_video` (la misma cadena que usa la aplicación) sobre videos del dataset, con y sin rotación declarada (copias en un directorio temporal; DeepLabCut en GPU):

{{TABLA:inferencia_prueba}}

{{TABLA:inferencia_resumen}}

**Cómo leerlo.** La cadena funciona de punta a punta (16 de 16 videos dieron predicción, 0 negativas, 0 errores; ≈ 35 s por video). **Pero la predicción coincide con la etiqueta en sólo 8 de 16 videos (3 de 8 con rotación, 5 de 8 sin rotación)**, y el modelo de producción **se entrenó con los clips de estos mismos videos**: debería acertar muy por encima del 74 % de la validación cruzada. Una coincidencia del 50 % es una **señal de alarma sobre el eslabón entrenamiento–inferencia** (recodificación `mp4v`, orientación, ventana elegida sobre el video completo en lugar del clip etiquetado, lectura por búsqueda de fotograma). Es una muestra de 16, no una estimación; no se aisló qué diferencia pesa más. Se la reporta como evidencia contradictoria con la confiabilidad que muestra la interfaz. La interfaz arranca sin excepciones (`datos/app_streamlit.json`) y cita sus métricas del JSON vigente.
