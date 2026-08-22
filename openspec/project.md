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
| 2. Pipeline de preprocesamiento | 🟡 Ejecutado 92/92 clips, pero **alineación 59,5 %, bajo el umbral de 70 %** |
| 3. Clasificación orientada a video | 🔴 No iniciado |
| 4. Aplicación / interfaz | 🔴 No iniciado |

**Artefactos existentes:**

- `00_pipeline.ipynb` (11 celdas). Orquestador: ejecuta las etapas en orden con `papermill`,
  saltea las que ya están completas, corta la cadena ante un fallo y guarda una copia ejecutada de
  cada notebook. Es el único notebook que hace falta abrir para correr el pipeline.
- `equinos/` — paquete de infraestructura de ejecución (`rutas`, `checkpoint`, `cache`, `etapas`,
  `reporte`).
  No contiene lógica científica: resuelve dónde viven los artefactos, cómo se reanuda una etapa
  interrumpida y en qué estado está el pipeline.
- `01_preparar_dataset_equinos.ipynb` (41 celdas). Bloque 1 (inventario), Bloque 2
  (segmentación en clips de 5–7 s) y Bloque 3 (extracción de frames): implementados y **ejecutados
  con éxito**. La segmentación es reanudable, con checkpoint por video.
- `02_roi_y_normalizacion.ipynb` (48 celdas). Detección del equino, keypoints faciales, ROI y
  normalización de pose. Geometría verificada con 21 pruebas y **ejecutado completo sobre los 92
  clips** (17/08/2026). Ver más abajo la cobertura medida.

  **Estructura que devuelve `deeplabcut.video_inference_superanimal`** (confirmada, no supuesta):
  un `dict` indexado por ruta de video, cuyo valor es un `DataFrame` con columnas MultiIndex
  `scorer / individuals / bodyparts / coords`, filas = índice de frame, y `coords` ∈
  {`x`, `y`, `likelihood`}. Entre los bodyparts hay `nose` y `upper_jaw`.

  Rendimiento medido: ~15 s de detector + ~20 s de pose + ~10 s de video etiquetado por clip en
  GPU, es decir **alrededor de una hora para los 92 clips**. DeepLabCut deja un `.h5`, un
  `.pickle` y un video etiquetado junto a cada clip; el notebook los borra tras extraer los
  keypoints, porque el video etiquetado pesa casi como el clip original y ensucia
  `clips_equinos/`, que otros pasos recorren.
- `tests/test_geometria.py`. 21 pruebas de las funciones geométricas del notebook 02, extraídas del
  propio notebook para que no puedan divergir. Corren sin GPU ni internet.
- `tests/test_persistencia.py`. 46 pruebas del paquete `equinos`: resolución de rutas,
  reanudación, invalidación del checkpoint ante cambio de configuración, integridad de la caché y
  estado de las etapas. También sin GPU ni internet.

**Qué falta para cerrar el objetivo 2:** decidir qué hacer con la cobertura de 59,5 %, que quedó
bajo el umbral de 70 % — barrer `CONF_MINIMA_KEYPOINT` (no requiere reinferir) o pasar a anotación
manual de las ROI de los ejemplares peores. Ver `openspec/changes/roi-normalizacion-pose/`.

## Dataset

**Origen.** La carpeta compartida `Videos de prueba` de Google Drive, con **una carpeta por
clase**. Vive en "Compartidos conmigo", así que Colab sólo la ve si tiene un ACCESO DIRECTO en Mi
unidad (Drive → clic derecho → Organizar → Agregar acceso directo); eso no copia ni mueve nada.
`catalogo.raiz_videos()` la localiza y `00_pipeline.ipynb` se la pasa a la etapa 01 como parámetro.
Existe además una copia local en `videos/<clase>/` (1.8 GB), usada para verificar el pipeline
fuera de Colab. **No versionar esa carpeta.**

**La carpeta de una clase NO es toda la clase.** Parte del material etiquetado vive en subcarpetas
de trabajo, y **cuáles suman no es igual en todas las clases**: `Disgustado` suma
`Seleccionar cabeza?` y `Se tapa una parte de la cara`; `Interesado` y `Sereno`, sólo
`Seleccionar cabeza?`. La declaración está en `equinos/catalogo.py` (`SUBCARPETAS_POR_CLASE`) y es
el único lugar donde se decide qué entra al dataset. Lo no declarado queda afuera pero se informa
con nombre y conteo — `Separar cabeza?`, bajo `Sereno`, es el caso vigente. Sobre el espejo local
esa declaración lleva el catálogo de 122 a 133 videos, y `Disgustado` de 31 a 41.

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

## Almacenamiento de los artefactos

Todo cuelga de una **raíz de proyecto persistente**, que en Colab vive en Google Drive y fuera de
Colab en un directorio local. La resuelve `equinos.rutas`, que detecta el entorno solo: el mismo
código de notebook corre en la máquina de desarrollo (sin GPU, para verificar geometría) y en
Colab.

```
MyDrive/tesina_equinos/            <- RAIZ (equinos.rutas.NOMBRE_PROYECTO)
├── videos/<clase>/*.mp4           datos crudos, se leen montados
├── clips/<clase>/*.mp4            92 clips, ~2.7 GB
├── resultados/*.csv               inventarios, datasets y reporte de cobertura
├── cache/<etapa>.zip              frames empaquetados + manifiesto .json
├── checkpoints/<etapa>.json       estado de reanudación + CSV parcial
├── ejecuciones/<etapa>_<ts>.ipynb copias ejecutadas por papermill
├── modelos/                       pesos entrenados (objetivo 3)
└── codigo/equinos/                copia del paquete, para el bootstrap en Colab
```

**Qué va a Drive y qué al disco local de la sesión.** El criterio es el tamaño *unitario*, no el
total: Drive montado tiene una latencia por archivo que domina con archivos chicos, al punto de que
escribir los ~1472 JPEG de frames tarda más que escribir los 2.7 GB de clips. Los CSV y los clips
van directo a Drive; los frames se generan en `/content` y se persisten en un único `.zip`.

**Las rutas de los CSV son relativas a la raíz.** Guardarlas absolutas (`/content/clips_equinos/…`)
fue la causa directa de que el notebook 02 no encontrara los clips del 01: en una sesión nueva no
apuntaban a nada. Relativas, el mismo CSV sirve en Colab, en local y en otra cuenta de Drive.

**Esa decisión es lo que hace portable el pipeline entre entornos.** Copiar `tesina_equinos/` de
Drive a la raíz local alcanza para seguir una corrida donde iba, sin reinferir nada: cada CSV
resuelve sus rutas contra la raíz vigente, sea cual sea. En local la raíz se fija con
`TESINA_EQUINOS_RAIZ` y los videos crudos con `TESINA_EQUINOS_VIDEOS`; conviene que la raíz esté
en disco interno, porque los frames alineados son miles de archivos chicos y el disco externo es
exFAT por USB.

**El notebook nunca declara en qué entorno está.** Lo detecta `equinos.rutas`, y
`equinos.entorno.verificar()` comprueba en el arranque de cada etapa que las dependencias sean las
que el pipeline necesita —fuera de Colab no instala nada, porque ahí el intérprete no es
descartable—. El orquestador elige además el kernel con el que papermill ejecuta las etapas
(`python3` en Colab, `tesina-equinos` en local) y falla de entrada si no está registrado: ejecutar
con el intérprete equivocado es un fallo silencioso, no un error.

## Pipeline implementado

Se ejecuta desde `00_pipeline.ipynb`, que encadena las etapas y saltea las ya completas.

```
[00_pipeline] papermill, un kernel por etapa, salteo de etapas completas

  [01 — etapa preparar_dataset]
  → videos leídos de Drive montado   · sin descarga, sin gdown
  → inventario técnico (OpenCV)      → resultados/inventario_videos.csv
  → segmentación en clips de 5-7 s   → clips/<clase>/<video>_clip_NN.mp4
     · reanudable, checkpoint por video
  → inventario de clips              → resultados/inventario_clips.csv
  → 16 frames uniformes por CLIP (np.linspace sobre CAP_PROP_FRAME_COUNT)
                                     → /content + cache/frames_equinos.zip
  → registro de frames               → resultados/dataset_frames.csv

  [02 — etapa roi_normalizacion, requiere GPU]
  → detección del equino             · YOLO preentrenado en COCO, clase "horse"
  → suavizado temporal por clip      · mediana móvil + relleno de huecos
  → keypoints faciales               → resultados/keypoints_faciales.csv
     · reanudable, checkpoint por clip; el clip se copia al disco local antes de inferir
       porque DeepLabCut escribe sus auxiliares junto al archivo de entrada
  → ROI cuadrada + alineación        · transformada de similitud (ojo, hocico)
                                     → /content + cache/frames_alineados.zip
                                     → resultados/dataset_frames_alineados.csv
  → reporte de cobertura             → resultados/reporte_cobertura.csv
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

- `inventario_videos.csv` — `archivo`, `clase`, `carpeta`, `ruta`, `fps`, `frames`,
  `duracion_seg`, `ancho`, `alto`. **`clase` es la etiqueta** (sale del catálogo); `carpeta` es la
  carpeta real de la que salió el archivo, que puede ser una subcarpeta y entonces no coincide con
  la clase. Los CSV anteriores a esta convención no tienen `clase`.
- `inventario_clips.csv` — `clase`, `id_caballo`, `video_origen`, `clip`, `ruta_clip`,
  `frame_inicio`, `frame_fin`, `frames`, `fps`, `duracion_seg`.
- `dataset_frames.csv` — `clase`, `id_caballo`, `video_origen`, `clip`, `frame`, `ruta_frame`,
  `ruta_clip`.

`id_caballo` (prefijo numérico del nombre original, vía `obtener_id_caballo`) se arrastra hasta el
nivel de frame: es la clave de agrupamiento para particionar sin fuga entre conjuntos.

`ruta`, `ruta_clip` y `ruta_frame` se guardan **relativas** a la raíz de proyecto y se resuelven con
`equinos.rutas.resolver()`. `ruta_frame` es relativa al directorio de frames, que es donde se
expande la caché.

## Reanudación

Colab corta la sesión por límite de tiempo o por inactividad, y la etapa de keypoints tarda
alrededor de una hora. Las etapas largas persisten su progreso por unidad de trabajo —el video en
la segmentación, el clip en los keypoints— y al relanzarse retoman en la primera pendiente.

El checkpoint registra los parámetros con los que se generó y **se invalida solo** si cambian. Esto
no es comodidad: reanudar con otra configuración produciría un artefacto que mezcla dos corridas,
que no falla ni avisa y arrastra el error hasta las métricas finales. Cada etapa declara qué claves
la afectan; la extracción de keypoints declara la lista vacía a propósito, porque DeepLabCut
devuelve coordenadas y likelihood sin depender de ningún umbral nuestro —el filtrado ocurre
después—, así que barrer umbrales no obliga a reinferir la hora de cómputo.

## Cobertura medida del preprocesamiento facial

Primera corrida completa: **17/08/2026**, los 92 clips en Colab con GPU.

Los 1472 frames se generaron (100 %), pero eso no es la métrica que importa: el notebook siempre
produce un recorte, y cuando el par ojo/hocico no supera el umbral cae al camino degradado, una ROI
cuadrada desde la caja del detector **sin alinear**. Lo que decide es cuántos se alinearon de
verdad por keypoints.

| | Frames | Alineados por keypoints |
|---|---:|---:|
| **Global** | 1472 | **876 (59,5 %)** |
| Sereno | 944 | 54,8 % |
| Interesado | 368 | 76,9 % |
| Disgustado | 160 | 47,5 % |

**59,5 % está por debajo del umbral de 70 %** fijado en la spec, con
`CONF_MINIMA_KEYPOINT = 0.50`.

Cobertura por ejemplar, los ocho peores:

| Caballo | Cobertura |
|---|---:|
| 920 | 8,8 % |
| 053 | 14,6 % |
| 031 | 29,2 % |
| 057 | 32,8 % |
| 155 | 43,8 % |
| 054 | 44,8 % |
| 055 | 46,9 % |
| 037 | 47,5 % |

**El fallo es sistemático, no aleatorio.** No se reparte parejo entre frames: se concentra en
ejemplares concretos, que quedan casi enteros sin alinear mientras otros funcionan bien. Las
diferencias por clase son en realidad diferencias por caballo — Disgustado baja porque `037` baja,
y `037` es el único ejemplar de esa clase.

Dos consecuencias que condicionan el objetivo 3:

1. **El dataset actual mezcla dos tipos de imagen**: ~60 % recortes normalizados por pose y ~40 %
   recortes de caja sin alinear. Cuál le toca a cada frame está correlacionado con el ejemplar, así
   que el modelo podría separar clases por el aspecto del recorte en vez de por la expresión. Es el
   mismo problema del fondo que este preprocesamiento venía a resolver, reintroducido por la puerta
   de atrás.
2. **`CONF_MINIMA_KEYPOINT` se puede barrer sin reinferir.** `keypoints_faciales.csv` guarda las
   coordenadas y su likelihood para los 1472 frames, y el filtrado ocurre después. `equinos.reporte.confianzas()`
   calcula qué cobertura daría cada umbral leyendo ese CSV: segundos, no la hora de inferencia.
   Es la primera vía a explorar antes de decidir anotación manual.

## Punto de partida de la próxima sesión

Estado al 21/08/2026, rama `ejecucion-local-con-gpu`.

**El cómputo se mudó a la máquina local.** El acceso a GPU de Colab quedó cortado por límites de
la suscripción, y la GPU local —que se daba por inservible— funciona: lo que fallaba era el `torch`
instalado (`+cu130` contra un driver CUDA 12.2), no la placa. Colab se conserva como ruta válida.
El procedimiento de instalación está en el README; el entorno, en `requirements-local.txt`.

### La primera acción, concreta

**Traer los artefactos de Drive y re-derivar los recortes con el encuadre nuevo**, ahora en local.
El código está implementado y verificado; falta correrlo. No requiere reinferir.

1. Copiar `MyDrive/tesina_equinos/` (carpetas `clips`, `resultados`, `cache`, `checkpoints`) a la
   raíz local que apunte `TESINA_EQUINOS_RAIZ`.
2. `etapas.diagnostico()` **antes que nada**: una copia a medias deja el checkpoint sin sus cachés,
   y una etapa incompleta puede darse por completa sin avisar.
3. Abrir `02_roi_y_normalizacion.ipynb` con el kernel *Tesina equinos (GPU local)* y **confirmar
   que `LIMPIAR_ETAPA = False`**: en `True` borra el checkpoint y obliga a reinferir (~4 h acá,
   porque DeepLabCut infiere el clip entero y no sólo los 16 frames muestreados).
4. Ejecutar todas sus celdas. El bucle debe informar `Clips pendientes: 0 de 210` — ésa es la señal
   de que la pose persistida está evitando la reinferencia. Unos 30-45 min.
5. Medir: `reporte.calidad_recortes()`, `reporte.cobertura()`. **Los números no deben moverse por
   la mudanza**: si cambian, cambió algo más que el entorno.

Por el orquestador no se puede: vería la etapa completa y la saltearía, y la única forma de que no
la saltee es forzarla, lo que borra el checkpoint.

**Pendiente de la sesión anterior, que sigue vigente:** el change `catalogo-de-videos-por-clase`
está implementado y sin correr. Con el catálogo declarado el dataset pasa de 146 videos a **133**:
los 13 que se van son la carpeta de descarga `Sereno-20260819T000125Z-1-001/Sereno`, que entraba
como `Sereno` sin aviso. Verificado sobre el espejo local el 21/08/2026.

### Dónde quedó el dataset

| | inicio | ahora |
|---|---|---|
| Videos | 34 | **146** |
| Clips | 92 | **210** |
| Frames esperados | 1472 | **3360** |
| Frames generados | 100 % | **100 %** |
| Alineados / esperados | 59,5 % | **68,0 %** |

Frames por clase: Sereno 1680 (50 %), Interesado 1312 (39 %), Disgustado 368 (11 %). Entre los
2286 alineados el reparto es Sereno ~1038, Interesado ~1035, Disgustado ~212.

**Disgustado ya no proviene de un solo caballo:** aparecen `037`, `153`, `022`, `054` y `909`. Eso
levanta la restricción que motivaba el plan B de 2 clases y habilita evaluar las tres bajo
partición por ejemplar.

### Tres causas encontradas y resueltas, en orden de tamaño

1. **Orientación de video** (720 frames en juego). 42 de los 146 videos guardan la rotación como
   metadato y OpenCV entregaba el frame acostado; el detector no reconocía un caballo girado 90
   grados. Perdían el 55,6 % de sus frames contra 0,0 % de los derechos. Corregido con
   `CAP_PROP_ORIENTATION_AUTO`; verificado: 0,0 % de pérdida y paridad entre ambos grupos.
2. **Encuadre canónico imposible** (2228 recortes afectados). El recorte abarcaba 2,5 veces la
   distancia ojo-hocico, que es el 26 % de la diagonal: pedía 1454 px de lado en cuadros de 1080
   de ancho. Sólo el 2,5 % salía sin relleno. Corregido a separación 0,60 más `BORDER_REPLICATE`;
   **falta verificar**.
3. **Bodypart del hocico** (24 frames). Refutado: `upper_jaw` da 54,4 % contra 62,4 % de `nose`.
   Queda `nose` fijo. Resultado negativo, documentado y cerrado.

### Lo que hay que recordar sobre el método

Las tres causas se manifestaban tarde y apuntando al lugar equivocado. Lo que destapó la más
grande fue **medir la pérdida contra los frames esperados en vez de contra los generados**:
`cobertura()` informaba 68,1 % cuando la cifra honesta era 51,9 %. Y la segunda apareció al
preguntarse si un recorte geométricamente exacto era además utilizable — el Paso 16 da error
1e-13 aunque el hocico esté sobre una oreja, porque mide la transformada contra los mismos puntos
con que se construyó.

Conviene desconfiar de toda métrica cuyo denominador no se haya elegido a propósito.

### Cuando se cierre el objetivo 2

El umbral del 70 % se fijó con 34 videos y 92 clips sobre la mesa, como proxy de "¿es utilizable
el pipeline automático?". Con 146 videos, 2286 frames alineados y Disgustado multi-caballo, vale
preguntarse si esa pregunta no está contestada por otra vía aunque el número quede en 68,0 %.
**Es una decisión de Sergio, no una conclusión.**

Los 2286 frames alineados forman por sí solos un dataset sin el confounder de mezclar recortes
alineados con recortes de caja — que era el problema a resolver antes de entrenar. Son 2,6 veces
los 876 de partida, y el balance entre Sereno e Interesado queda 45/45.

### Deudas conocidas, sin change abierto

- **El pipeline no detecta que el dataset de entrada cambió.** `etapas.estado()` mira artefactos y
  cachés, no los datos. Incorporar el conteo de videos al `hash_config` alcanzaría.
- **No hay modo de re-derivar sin recomputar por el orquestador.** Hay que correr el notebook 02
  suelto.
- **`00_pipeline.ipynb` no se sincroniza** y hay que subirlo a mano, lo que además apaga el
  permiso del secret `GITHUB_TOKEN` para ese notebook.
- **Etiquetado:** `501_8v1_LY1ucwR9.avi` estaba como archivo idéntico en Interesado y Disgustado;
  se resolvió como Disgustado. La validación de homónimos ahora cubre también el mismo archivo
  suelto en la clase y dentro de una de sus subcarpetas, pero copias con nombres **distintos** en
  dos clases siguen pasando sin aviso. Conviene revisar el criterio con que se armaron las
  carpetas.

## Stack técnico

- **Entornos de ejecución (dos, equivalentes):** Google Colab con GPU, y máquina local
  con GPU (Ubuntu 22.04, driver NVIDIA 535, RTX 3070 Laptop 8 GB). Python 3.10 en ambos.
  Las versiones locales están pinneadas en `requirements-local.txt`; en Colab se instalan
  por sesión desde la celda de arranque, tomando ese archivo como referencia.
- **Orquestación:** `papermill` (un kernel por etapa, copia ejecutada de cada notebook)
- **Infraestructura propia:** paquete `equinos` (rutas persistentes, checkpointing, caché, estado)
- **Datos / video:** `opencv-python`, `pandas`, `numpy`, `tqdm`
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
  `Interesado`, `Disgustado`. La etiqueta de un video NO se deduce de su carpeta contenedora: sale
  del catálogo declarado en `equinos/catalogo.py`. En los CSV la etiqueta es la columna `clase`.
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
