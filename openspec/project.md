# Contexto del Proyecto

## Identificación

- **Tema:** Detección Automática de Expresiones Faciales en Equinos mediante Aprendizaje Profundo en Secuencias de Video
- **Institución:** Universidad del Gran Rosario (UGR) — Licenciatura en Ciencias de Datos
- **Asignatura:** Seminario de Trabajo Final (Tesina)
- **Integrantes:** Héctor R. Liceaga, Sergio M. Yañez
- **Docentes:** Alejandro Hernández, Brian Gauna, Jorge Fernández
- **Fecha del plan de trabajo:** Mayo de 2026 — **entrega final proyectada: septiembre de 2026**
- **Temática de interés:** Aprendizaje Automático y Visión por Computadora

## Problema e hipótesis

**Problemática.** La evaluación del estado mental y emocional de los equinos depende hoy casi
exclusivamente de la observación humana: es subjetiva, no estandarizada y demanda mucho tiempo.
Los enfoques basados en imágenes estáticas pierden información crucial, porque las expresiones
faciales y los comportamientos asociados a emociones en equinos son **eventos dinámicos**: la
evolución temporal en pocos segundos es la que permite una clasificación robusta.

**Hipótesis.** Es factible desarrollar un modelo computacional basado en arquitecturas de
aprendizaje profundo capaz de identificar y clasificar automáticamente expresiones faciales
asociadas a distintos estados emocionales en equinos a partir de secuencias de video cortas,
alcanzando un nivel de precisión útil para la asistencia de expertos humanos.

## Objetivos

**General.** Construir un sistema informático para la detección automatizada de expresiones
faciales en caballos mediante el procesamiento de secuencias espacio-temporales.

**Específicos.**

1. Conformar una base de datos de videos cortos (5 a 7 segundos) de rostros de equinos.
2. Desarrollar un *pipeline* de preprocesamiento: extracción de fotogramas clave y detección de la
   región de interés (ROI) facial.
3. Aplicar técnicas de clasificación de Aprendizaje Automático orientadas a video para identificar
   estados emocionales.
4. Desarrollar una aplicación o interfaz para el uso del modelo resultante.

## Estado actual

| Objetivo específico | Estado |
|---|---|
| 1. Base de datos de videos cortos (5–7 s) | 🟢 Cumplido — 34 videos → 92 clips segmentados |
| 2. Pipeline de preprocesamiento | 🟡 Implementado, **pendiente de validar en Colab** |
| 3. Clasificación orientada a video | 🔴 No iniciado |
| 4. Aplicación / interfaz | 🔴 No iniciado |

**Artefactos existentes:**

- `01_preparar_dataset_equinos.ipynb` (37 celdas). Bloque 1 (descarga e inventario), Bloque 2
  (segmentación en clips de 5–7 s) y Bloque 3 (extracción de frames): implementados y **ejecutados
  con éxito**.
- `02_roi_y_normalizacion.ipynb` (43 celdas). Detección del equino, keypoints faciales, ROI y
  normalización de pose. **Escrito y con la geometría verificada, pero todavía no ejecutado**: los
  modelos preentrenados requieren GPU y descarga de pesos, disponibles sólo en Colab.
- `tests/test_geometria.py`. 21 pruebas de las funciones geométricas del notebook 02, extraídas del
  propio notebook para que no puedan divergir. Corren sin GPU ni internet.

**Qué falta para cerrar el objetivo 2:** ejecutar el notebook 02 en Colab, adaptar la extracción de
coordenadas del Paso 10 a la estructura real que devuelva el modelo de pose, y evaluar el reporte
de cobertura del Paso 14. Ver `openspec/changes/roi-normalizacion-pose/`.

## Dataset

**Origen.** Carpeta de Google Drive (`folder_url` embebida en el notebook, descarga vía `gdown`),
con **una subcarpeta por clase** — la subcarpeta *es* la etiqueta. Existe además una copia local
en `videos/<clase>/` (1.8 GB, idéntica a Drive), usada para verificar el pipeline fuera de Colab.
**No versionar esa carpeta.**

**Composición actual (34 videos) — el dataset se ampliará en los próximos días:**

| Clase | Videos | Clips (5–7 s) | **Caballos distintos** |
|---|---:|---:|---:|
| Sereno | 23 | 59 | 15 |
| Interesado | 8 | 23 | 8 |
| Disgustado | 3 | 10 | **1** (`037`) |
| **Total** | **34** | **92** | **22** |

**La columna que manda es la de caballos, no la de videos ni la de clips.** Dos consecuencias:

- **Disgustado no es evaluable hoy.** Con `GroupKFold` por caballo, `037` cae entero en train o
  entero en test; en cualquiera de los dos casos una de las particiones queda sin la clase. El
  problema de 3 clases no es entrenable de forma honesta con estos datos.
- **La identidad del caballo predice la etiqueta.** Sólo 2 de 22 ejemplares (`054`, `055`)
  aparecen en más de una clase. El modelo puede acertar reconociendo al caballo.

**Plan B acordado (a decidir más adelante):** si no se consiguen videos de Disgustado en varios
ejemplares, reducir a 2 clases — Sereno vs Interesado, 15 y 8 caballos.

**Características técnicas.** Resolución uniforme 1080×1920 (**vertical**, filmado con celular);
fps ~30 (algunos casos a 29.97 / 30.03 / 59.94); duración entre **8 y 37 segundos**
(mediana ≈ 17 s). Convención de nombre de archivo: `<id_ejemplar>_<descriptor>.mp4`
(p. ej. `037_box-7-1_28oDPfEr.mp4`, `155_20260707_092149.mp4`) — el prefijo numérico identifica al
**caballo**, y varios videos comparten ejemplar.

## Pipeline implementado en el notebook

```
Drive (carpetas por clase)
  → gdown.download_folder            → /content/dataset_equinos_videos/<clase>/*.mp4
  → inventario técnico (OpenCV)      → resultados_equinos/inventario_videos.csv
  [Bloque 2]
  → segmentación en clips de 5-7 s   → /content/clips_equinos/<clase>/<video>_clip_NN.mp4
  → inventario de clips              → resultados_equinos/inventario_clips.csv
  [Bloque 3]
  → extracción de 16 frames uniformes por CLIP (np.linspace sobre CAP_PROP_FRAME_COUNT)
                                     → /content/frames_equinos/<clase>/<clip>/frame_NNN.jpg
  → registro de frames               → resultados_equinos/dataset_frames.csv

  [notebook 02 — requiere GPU, todavía no ejecutado]
  → detección del equino             · YOLO preentrenado en COCO, clase "horse"
  → suavizado temporal por clip      · mediana móvil + relleno de huecos
  → keypoints faciales               → resultados_equinos/keypoints_faciales.csv
  → ROI cuadrada + alineación        · transformada de similitud (ojo, hocico)
                                     → /content/frames_alineados/<clase>/<clip>/frame_NNN.jpg
                                     → resultados_equinos/dataset_frames_alineados.csv
```

**Segmentación (Bloque 2).** Ventanas de `DURACION_CLIP = 6.0 s` (configurable), sin solape por
defecto; el resto final se absorbe en la última ventana si no supera los 7 s, y se descarta si no
llega a 5 s. Recorte frame-a-frame con `cv2.VideoWriter` para obtener límites exactos y evitar
problemas con los nombres de archivo que contienen espacios.

**Ejecutado y verificado** sobre los 34 videos reales (13/08/2026, copia local en `videos/`):
**92 clips**, ninguno fuera del rango 5–7 s, **92.8 % del metraje conservado** (575 s de 620 s),
**22 caballos distintos**, 2.73 GB, 1472 frames. Los 92 clips se releyeron uno por uno: 0 con
error de decodificación, 0 con desfasaje de frames, 0 con resolución o fps incorrectos.

**Caso borde detectado y manejado:** en 6 clips (siempre el último de su video)
`CAP_PROP_FRAME_COUNT` sobreestima la cantidad real de frames y la lectura termina antes. El guard
`if not ret: break` los deja en 5.27–5.77 s, dentro de rango, y `duracion_seg` guarda la duración
real, no la teórica. No hace falta corregirlo, pero **`frames` del inventario de videos no es
confiable como conteo exacto**.

| Clase | Videos | Clips | Factor |
|---|---:|---:|---:|
| Sereno | 23 | 59 | 2.6× |
| Interesado | 8 | 23 | 2.9× |
| Disgustado | 3 | 10 | 3.3× |
| **Total** | **34** | **92** | **2.7×** |

**CSV generados:**

- `inventario_videos.csv` — `archivo`, `carpeta`, `ruta`, `fps`, `frames`, `duracion_seg`,
  `ancho`, `alto`.
- `inventario_clips.csv` — `clase`, `id_caballo`, `video_origen`, `clip`, `ruta_clip`,
  `frame_inicio`, `frame_fin`, `frames`, `fps`, `duracion_seg`.
- `dataset_frames.csv` — `clase`, `id_caballo`, `video_origen`, `clip`, `frame`, `ruta_frame`,
  `ruta_clip`.

`id_caballo` (prefijo numérico del nombre original, vía `obtener_id_caballo`) se arrastra hasta el
nivel de frame: es la clave de agrupamiento para particionar sin fuga entre conjuntos.

Todas las rutas son **efímeras** (`/content/...`): el notebook arranca borrándolas (Paso 0) y
vuelve a descargar desde Drive en cada sesión.

## Stack técnico

- **Entorno:** Google Colab (GPU) + Python 3
- **Datos / video:** `gdown`, `opencv-python`, `pandas`, `numpy`, `tqdm`
- **Visualización:** `matplotlib`
- **ML:** `scikit-learn` (métricas, validación cruzada), **PyTorch** (previsto para el modelo)
- **Previsto en el plan:** TensorFlow y **DeepLabCut** como alternativas / pose estimation
- **Almacenamiento de datos crudos:** Google Drive (no versionado en el repo)

## Metodología (según plan de trabajo)

1. **Investigación tecnológica** — estado del arte en modelos capaces de procesar información
   espacio-temporal.
2. **Preprocesamiento** — limpieza, extracción y procesamiento de fotogramas, asegurando que los
   rostros estén alineados a lo largo de los clips.
3. **Entrenamiento y selección** — pruebas de concepto con varios modelos candidatos; selección de
   la arquitectura definitiva por métricas de rendimiento.
4. **Validación final** — validación cruzada y análisis de resultados para evitar sobreajuste.

## Riesgos y limitaciones conocidas

Estas son las tensiones reales entre el estado del dataset y lo que el plan exige. Conviene
resolverlas antes de entrenar:

1. **Volumen muy bajo.** 34 videos → 92 clips. La segmentación mejora la situación pero sigue
   siendo insuficiente para entrenar desde cero cualquier arquitectura espacio-temporal → obliga a
   *transfer learning* y aumentación agresiva.
2. **Desbalance severo.** El recorte no lo corrige: 59 / 23 / 10 clips (≈64% / 25% / 11%), casi la
   misma proporción que en origen. Un clasificador trivial que siempre prediga "Sereno" alcanza
   64% de accuracy: **la accuracy global no es métrica válida**; usar F1 macro, balanced accuracy y
   matriz de confusión. La clase Disgustado proviene de **un solo caballo** (`037`), por lo que el
   modelo puede aprender el ejemplar en lugar de la expresión.
3. ~~Duración fuera de especificación~~ — **resuelto** en el Bloque 2 (ventanas de 5–7 s).
4. **Normalización de pose no implementada.** Inspección visual de los frames (13/08/2026): el
   encuadre es cerrado y la cabeza ocupa buena parte del cuadro, así que **el recorte de ROI no es
   crítico por resolución**. El problema real es la **pose**: hay desde frontal completo (los dos
   de Disgustado) hasta perfil total (`030`, `026`, `906`), donde se pierde un ojo y cambia la
   geometría del hocico. El objetivo específico 2 pide rostros "alineados a lo largo de los clips":
   eso apunta a landmarks / DeepLabCut para **normalizar pose**, más que para recortar.
4b. **El fondo está correlacionado con la clase.** Los 3 videos de Disgustado son interior de box
   (pared de madera, sombra); Interesado y Sereno son casi todos exterior (pasto, árboles,
   alambrados). Un modelo puede separar Disgustado sin mirar la cara. Se apila sobre el problema
   del caballo único: son dos confundidos sobre la misma clase. El recorte a ROI facial sí ayuda
   acá, eliminando el fondo del campo receptivo.
5. **Riesgo de fuga de datos (*data leakage*).** Los 16 frames de un clip son casi idénticos, los
   clips de un video son contiguos, y varios videos comparten ejemplar. Un split aleatorio a nivel
   *frame* o *clip* infla las métricas de forma masiva → **particionar por caballo**
   (`GroupKFold` sobre `id_caballo`, ya presente en los CSV). Sólo hay **22 caballos**, y eso
   —no los 1472 frames— es el tamaño muestral efectivo.
6. **Extracción uniforme ≠ fotogramas clave.** `np.linspace` sobre un clip de 6 s muestrea a
   ~1 frame cada 0.4 s. Es mejor que sobre el video completo, pero para modelado temporal conviene
   evaluar ventanas contiguas a fps alto.
7. **Descartado tras verificación:** varios videos tienen marca de agua de `clideo.com` por haber
   pasado por un editor online. Se revisaron los metadatos de encoder: el reprocesado
   (`Lavf58.76.100`) está en 67 % de Disgustado, 62 % de Interesado y 65 % de Sereno — parejo entre
   clases, **no es un confundido**.
8. **Etiquetado sin protocolo documentado.** No hay registro de quién etiquetó, con qué criterio ni
   acuerdo inter-observador. EquiFACS es el marco de referencia natural para justificarlo. La
   etiqueta se hereda del video completo hacia todos sus clips, lo que asume que la expresión se
   mantiene constante durante los 8–37 s originales — supuesto fuerte y no verificado.

## Convenciones del proyecto

- **Idioma:** todo el código, comentarios, nombres de variables, documentación y entregables en
  **español** (el notebook ya lo hace: `df_videos`, `extraer_frames_uniformes`, `resultados_dir`).
- **Nombres de clases:** capitalizados, tal como aparecen las carpetas en Drive — `Sereno`,
  `Interesado`, `Disgustado`.
- **Notebooks:** numerados y con prefijo de orden (`01_preparar_dataset_equinos.ipynb`), divididos
  en "Bloques" con "Pasos" numerados en celdas markdown.
- **Rutas Colab:** `/content/dataset_equinos_videos`, `/content/clips_equinos`,
  `/content/frames_equinos`, `/content/resultados_equinos`, `/content/modelos_equinos`.
- **Unidad de análisis:** el **clip** de 5–7 s, no el video completo ni el frame suelto.
- **Nombre de clip:** `<stem_normalizado>_clip_NN.mp4` (el stem original se normaliza con
  `normalizar_nombre` porque varios archivos contienen espacios).
- **Reproducibilidad:** `random_state=42` donde aplique.
- **Datos crudos:** nunca se versionan en el repositorio; viven en Google Drive.

## Cronograma

Es el del plan de trabajo original. **El equipo indicó (13/08/2026) que puede no cumplirse**, así
que no debe usarse para recortar alcance: se prioriza hacer bien la normalización de pose y la
ampliación del dataset por sobre llegar a la fecha.

| Actividad | May | Jun | Jul | Ago | Sep |
|---|:-:|:-:|:-:|:-:|:-:|
| Revisión bibliográfica y antecedentes | X | | | | |
| Preparación de algoritmos y preprocesamiento | X | X | | | |
| Revisión de bases de datos | | X | X | | |
| Elaboración de modelos de aprendizaje | | | X | X | |
| Análisis, conclusiones y propuestas | | | | X | |
| Elaboración de informe final y presentación | | | | | X |

## Índice propuesto del informe final

1. Introducción
2. Descripción y Etología del Equino
3. Técnicas de Ciencia de Datos y Visión por Computadora
4. Desarrollo Experimental (Materiales, Métodos y Algoritmos)
5. Resultados y Discusión
6. Conclusiones
7. Bibliografía
8. Anexos

## Antecedentes y bibliografía

- **Wathan, J. et al. (2015).** *Equine Facial Action Coding System (EquiFACS)* — estandarización
  de unidades de acción facial en caballos; marco de referencia para el criterio de etiquetado.
- **Mathis, A. et al. (2018).** *DeepLabCut: markerless pose estimation of user-defined body parts
  with deep learning* — herramienta candidata para detección de ROI / landmarks faciales.
- Bibliografía adicional a definir durante el transcurso de la investigación.

**Enfoque diferencial:** el estado del arte combina CNN con análisis temporal para clasificar
emociones animales y detectar dolor a partir de patrones faciales; este trabajo se apoya en esas
metodologías pero se enfoca deliberadamente en la **riqueza secuencial del video** por sobre la
imagen estática.

## Impacto esperado

Mejora objetiva del bienestar animal: proveer a la medicina veterinaria y a la etología una
herramienta no invasiva de asistencia en la detección de emociones en equinos. En lo tecnológico,
el pipeline aporta metodologías innovadoras en la región, extrapolables al estudio de otras
especies.
