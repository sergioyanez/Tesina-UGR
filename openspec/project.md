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
| 1. Base de datos de videos cortos | 🟢 Cumplido y ampliado — **249 videos → 368 clips → 5888 frames** (28/08/2026, con `PISO_VIDEO_CORTO = 1.0`) |
| 2. Pipeline de preprocesamiento | 🟢 Ejecutado 368/368 clips; **el dataset admite partición por caballo** desde el 28/08/2026. Cobertura 73,5 % con estimadas, 68,7 % sólo detectadas — pero el global dejó de ser el criterio (ver "El número que hay que citar"). **Encuadre corregido el 28/08/2026 para incluir las orejas** (ver abajo) |
| 3. Clasificación orientada a video | 🟡 Modelo entrenado y evaluado; **la hipótesis no queda sostenida** y el criterio declarado no se cumple (0,626 contra 0,691). Falta el notebook de la etapa. Ver abajo |
| 4. Aplicación / interfaz | 🟡 Planificado — change `interfaz-de-inferencia`, 4 artefactos y 26 tareas, ninguna hecha |

### Objetivo 3 — resultado, y por qué es un resultado negativo bien medido (28/08/2026)

**Modelo final: clasificador de un frame con voto, F1 macro 0,626 ± 0,065.** Cabeza lineal sobre
las características de una ResNet-18 congelada, entrenada a nivel de frame y agregada por voto
entre los 16 frames del clip. **NO cumple el criterio declarado antes de entrenar (0,691).**

**La hipótesis del trabajo no queda sostenida por esta evidencia.** El modelo elegido es
*invariante al orden*: barajar los frames del clip no cambia una predicción. Se probaron 18
configuraciones —tres cabezas temporales, dos regímenes de aumentación, dos de capacidad, con las
muestras igualadas— y en ninguna el orden aportó: la atención da predicciones idénticas con los
frames barajados, y la GRU rinde peor con el orden real.

Verificado que el tensor está en el orden que declara el CSV, porque con los frames barajados de
fábrica esa conclusión sería trivial y no diría nada.

**Lo que sí quedó demostrado, y es el resultado positivo del trabajo:**

| Línea de base | F1 macro |
|---|---:|
| Clase mayoritaria | 0,188 ± 0,029 |
| Sólo `id_caballo`, partición por ejemplar | 0,188 ± 0,029 |
| Sólo `id_caballo`, **partición ingenua por clip** | 0,526 ± 0,116 |
| **Modelo final** | **0,626 ± 0,065** |

- **Las imágenes tienen señal**: de 0,188 a 0,626 con partición honesta por ejemplar. El
  preprocesamiento facial del objetivo 2 produce una entrada de la que un clasificador lineal
  extrae bastante más que las clases a priori.
- **El confounder de individuo está cuantificado**: sin agrupar por caballo se llega a 0,53 *sin
  mirar un píxel*, porque 63 de los 73 ejemplares aparecen en una sola clase. Esos +0,338 son la
  fuga que el protocolo elimina, y la justificación medida de agrupar por ejemplar.

**Confusión dominante, contraria a lo previsto.** El eje que el modelo mezcla es
`Sereno`/`Disgustado` (12 y 9 errores), no `Disgustado`/`Interesado` (5), que era lo que anticipaba
el techo del caso `178`.

**Ablación por `origen_alineacion`:** con los 82 clips de detección propia el F1 macro cae a 0,518
y el recall de `Disgustado` a 0,000 —ese subconjunto tiene 7 clips de la clase—. No invalida el
resultado completo: mide cuánto se apoya en posiciones estimadas, y resulta ser un argumento **a
favor** del relleno temporal y la anotación manual.

**Los errores se concentran:** 22 de 73 ejemplares fallan alguna vez y los 10 peores acumulan el
68 %. Cinco de esos diez están entre los que aparecen en más de una clase. `13`, `44` y `231`
fallan en el 100 % de sus clips: candidatos a revisión de etiqueta, no a mejora de arquitectura.

**Qué haría falta, con los números al lado: más ejemplares.** La aumentación temporalmente
coherente se probó con 20 vistas por clip y no movió nada (−0,005 a +0,000) — veinte vistas del
mismo clip siguen siendo el mismo caballo. Y ninguna de las tres cabezas, en dos regímenes de
capacidad, superó a una regresión logística sobre características congeladas. El cuello de botella
son los 73 ejemplares, y sobre todo los 15 de `Disgustado`.

La evidencia completa está en `openspec/changes/clasificacion-orientada-a-video/`:
`lineas_de_base.md`, `cabezas_temporales.md` y `resultado_final.md`.

### El encuadre del recorte, y por qué se corrigió (28/08/2026)

El recorte facial es un rectángulo de **224×343** en el marco de la cabeza, con el eje ojo-hocico
sobre la vertical. Se declara como **campo de visión en unidades de la distancia ojo-hocico `d`**:
1,00 `d` de aire sobre el ojo, 0,30 bajo el hocico, ±0,75 lateral. El alto no se elige, se deriva
del aspecto que ese campo exige.

**Por qué dejó de ser cuadrado.** La observación es de Héctor R. Liceaga: en los equinos la
posición de las orejas —hacia atrás o aplanadas— es el indicador principal de disgusto y estrés, y
el recorte podía estar dejándolas fuera. Se pudo verificar sin reinferir, porque `pose_completa.csv`
guarda los 39 bodyparts y entre ellos están `left_earbase`/`left_earend` y sus simétricos. Medido:
con el encuadre anterior el **vector base→punta de la oreja era legible en el 4,3 % de los frames**.

El déficit era asimétrico —la punta de oreja está a 0,81 `d` sobre el ojo en la mediana y el
recorte mostraba 0,33 `d`, mientras que lateralmente sobraba campo—, y un recorte cuadrado no puede
corregir eso: ganar aire vertical obliga a ampliar el lateral. Por eso la corrección que proponía la
rama `hector` (`FRAC_OJO` de 0,20 a 0,25) sólo llega al 13,7 %: dirección correcta, palanca
equivocada.

El encuadre adoptado lleva el vector de oreja a **77,2 %** empatando exacto el relleno mediano del
encuadre anterior (14,1 %), porque gasta todo el presupuesto en vertical y recorta campo lateral.
De paso la cara queda 11 % más grande (`d`: 134 → 149 px) y con menos fondo.

**La métrica que decide es el vector de oreja, no el conteo suelto de keypoints.** La orientación
de la oreja se lee de base→punta; una base sin su punta no dice hacia dónde apunta. El conteo
suelto sobrestima: el candidato descartado informaba 64,4 % contando keypoints y 50,6 % contando
vectores.

**Lo que NO cambió:** el par de anclaje sigue siendo ojo-hocico, así que las 14 anotaciones
manuales siguen válidas; no se reinfirió pose; y la composición del dataset es idéntica —141 clips
íntegros, 73 ejemplares, 21/56/64—, porque el encuadre no decide qué frames se alinean. Sólo
cambiaron los píxeles.

Cada frame registra ahora `relleno_fraccion`: la fracción del recorte completada con píxeles
inventados. Depende de la distancia de filmación, que depende del ejemplar, que predice la clase, y
por eso se registra por frame — para poder ablacionarlo, igual que `origen_alineacion`.

La evidencia completa está en `openspec/changes/encuadre-que-incluye-las-orejas/medicion.md`, y el
barrido se rehace en segundos con `equinos.encuadre.barrido()`, sin escribir una sola imagen.

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

**Qué falta para cerrar el objetivo 2:** reemplazar el criterio de aceptación —hoy la cobertura
global, que se mueve con la composición del dataset— por cobertura por clase, y atacar la
alineación de `Disgustado` (62,1 %). La vía barata sigue siendo barrer `CONF_MINIMA_KEYPOINT` con
`reporte.confianzas()`, que no requiere reinferir. Ver "Punto de partida de la próxima sesión".

## Dataset

**Origen.** La carpeta compartida `Videos de prueba` de Google Drive, con **una carpeta por
clase**. Vive en "Compartidos conmigo", así que Colab sólo la ve si tiene un ACCESO DIRECTO en Mi
unidad (Drive → clic derecho → Organizar → Agregar acceso directo); eso no copia ni mueve nada.
`catalogo.raiz_videos()` la localiza y `00_pipeline.ipynb` se la pasa a la etapa 01 como parámetro.
Fuera de Colab la fuente es `~/videos_de_prueba`, la copia local de esa carpeta compartida, y se
declara con `TESINA_EQUINOS_VIDEOS`. **Es la única fuente admitida.** El espejo que existía bajo la
raíz de artefactos se retiró el 25/08/2026: estaba a medio copiar, el pipeline lo eligió sin avisar
y produjo una corrida entera de números de otro dataset. **No versionar los videos.**

**La carpeta de una clase NO es toda la clase.** Parte del material etiquetado vive en subcarpetas
de trabajo, y **cuáles suman no es igual en todas las clases**: `Disgustado` suma
`Seleccionar cabeza?` y `Se tapa una parte de la cara`; `Interesado` y `Sereno`, sólo
`Seleccionar cabeza?`. La declaración está en `equinos/catalogo.py` (`SUBCARPETAS_POR_CLASE`) y es
el único lugar donde se decide qué entra al dataset. Lo no declarado queda afuera pero se informa
con nombre y conteo — `Separar cabeza?`, bajo `Sereno`, es el caso vigente. Sobre el espejo local
esa declaración lleva el catálogo de 122 a 133 videos, y `Disgustado` de 31 a 41.

**Composición actual (25/08/2026):**

| Clase | Videos | Clips | Frames | **Caballos en el dataset** | Caballos en el catálogo |
|---|---:|---:|---:|---:|---:|
| Sereno | 93 | 160 | 2560 | 54 | 54 |
| Interesado | 71 | 120 | 1920 | 60 | 60 |
| Disgustado | 70 | 62 | 992 | **32** | 42 |
| **Total** | **234** | **342** | **5472** | **113** | 117 |

Las columnas de caballos no suman: 11 ejemplares aparecen en más de una clase, así que el total son
**caballos distintos** (113 en el dataset, 117 en el catálogo) y no la suma de la columna.

`Disgustado` rinde menos clips que videos porque su video mediano dura 2,9 s: son tomas cortas que
entran enteras como un único clip. Los 10 ejemplares de la clase que aún no llegan al dataset están
todos por debajo del piso de 2 s — su video más largo mide 1,93 s.

**La columna que manda es la de caballos, no la de videos ni la de clips.** Dos consecuencias:

- **`Disgustado` pasó a ser evaluable.** Con 32 ejemplares, `GroupKFold` por caballo ya no deja
  una partición sin la clase. Al 17/08/2026 la clase venía de **un solo caballo** (`037`) y eso
  motivaba un plan B de 2 clases; **ese plan B queda sin motivo**. Lo que mantenía la restricción
  no era el material sino el piso de duración de los clips, que dejaba entrar 11 de 42 ejemplares.
- **La identidad del caballo todavía puede predecir la etiqueta.** Sólo 11 de los 146 ejemplares
  aparecen en más de una clase, así que la partición por `id_caballo` sigue siendo obligatoria.

**Características técnicas.** Resolución mayormente 1080×1920 (**vertical**, filmado con celular);
fps ~30 (algunos casos a 29.97 / 30.03 / 59.94). Las duraciones son muy dispares entre clases:
`Interesado` y `Sereno` promedian 11 s, `Disgustado` **4,5 s** con mediana de 2,9 s. Convención de nombre de archivo: `<id_ejemplar>_<descriptor>.mp4`
(p. ej. `037_box-7-1_28oDPfEr.mp4`, `155_20260707_092149.mp4`) — el prefijo numérico identifica al
**caballo**, y varios videos comparten ejemplar.

## Almacenamiento de los artefactos

Todo cuelga de una **raíz de proyecto persistente**, que en Colab vive en Google Drive y fuera de
Colab en un directorio local. La resuelve `equinos.rutas`, que detecta el entorno solo: el mismo
código de notebook corre en la máquina de desarrollo (sin GPU, para verificar geometría) y en
Colab.

```
MyDrive/tesina_equinos/            <- RAIZ en Colab (equinos.rutas.NOMBRE_PROYECTO)
<repo>/datos_equinos/              <- RAIZ en local, oficial desde el 25/08/2026
├── videos/                        NO existe y no debe volver a existir: una copia del material
│                                 bajo la raíz se convierte en fuente candidata sin avisar
├── clips/<clase>/*.mp4            342 clips, ~6 GB
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
`TESINA_EQUINOS_RAIZ` y los videos crudos con `TESINA_EQUINOS_VIDEOS`, y las dos variables viven
además en el `kernel.json` del kernel `tesina-equinos`, para que un Jupyter abierto por fuera del
lanzador no arranque sin ellas. La raíz oficial está en el disco externo, que es exFAT: la objeción
de velocidad —los frames alineados son miles de archivos chicos— quedó anotada pero no se
manifestó, y exFAT tampoco admite enlaces simbólicos, lo que descartó apuntar la carpeta de videos
a la fuente.

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
  → ROI 224x343 + alineación        · transformada de similitud (ojo, hocico), campo de
                                       visión que incluye las orejas
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

> **Los números de esta sección son los de la primera corrida completa (17/08/2026, 92 clips en
> Colab) y se conservan porque explican el diagnóstico.** Para el estado vigente —342 clips,
> 5472 frames, cobertura por clase— ver "El número que hay que citar", más abajo.

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

Estado al cierre del **25/08/2026**. Rama `ejecucion-local-gpu`, **sin pushear** por decisión de
Sergio: vive sólo en la máquina local.

### Qué pasó, en una frase

Entró al dataset todo el material catalogado —234 videos, contra 146— y se sacó el piso de duración
que dejaba afuera a tres de cada cuatro ejemplares de `Disgustado`.

### Dónde está cada cosa ahora

| | ruta |
|---|---|
| Artefactos (clips, resultados, cachés, checkpoints) | `<repo>/datos_equinos` — **raíz oficial desde el 25/08** |
| Videos crudos, fuente única | `~/videos_de_prueba` (234 videos) |
| Entorno de Python con GPU | `~/.venvs/tesina-equinos` |
| Artefactos de la corrida de 146 videos, como respaldo | `~/tesina_equinos` |

`datos_equinos/videos` **ya no existe**: era un espejo a medio copiar y el pipeline lo eligió sin
avisar. Sin él, una corrida sin `TESINA_EQUINOS_VIDEOS` corta con el mensaje de
`catalogo.raiz_videos` en vez de medir sobre material equivocado.

### Cómo se opera, sin recordar nada de memoria

Desde la carpeta del repositorio:

| comando | qué hace |
|---|---|
| `./correr_local.sh estado` | en qué estado está cada etapa, sin ejecutar nada |
| `./correr_local.sh reportes` | cobertura y calidad de recortes de la última corrida |
| `./correr_local.sh entorno` | versiones instaladas y GPU visible |
| `./correr_local.sh etapa01` | re-ejecuta el notebook 01 sin borrar el checkpoint |
| `./correr_local.sh etapa02` | re-ejecuta el notebook 02 sin reinferir lo ya hecho |
| `./correr_local.sh` | abre Jupyter Lab — elegir el kernel *Tesina equinos (GPU local)* |

El kernel `tesina-equinos` lleva `TESINA_EQUINOS_RAIZ` y `TESINA_EQUINOS_VIDEOS` en su
`kernel.json`, así que un notebook abierto desde cualquier Jupyter arranca con las rutas correctas.
**Un kernel que ya estaba abierto sigue con las variables viejas: hay que reiniciarlo.**

### El número que hay que citar

La cobertura global **no** es comparable entre corridas: se mueve con la composición del dataset.
Sobre 342 clips y 5472 frames da **69,9 %**, y ese número bajó respecto del 70,2 % anterior sin que
ninguna clase empeorara.

| subconjunto | frames | alineados |
|---|---:|---:|
| `Disgustado` | 992 | **62,1 %** (antes 57,6 %) |
| `Interesado` | 1920 | 80,3 % (antes 80,2 %) |
| `Sereno` | 2560 | 65,1 % (antes 65,4 %) |
| **global** | **5472** | **69,9 %** |

El global cae porque la mezcla se corrió hacia la clase que peor alinea: `Disgustado` pasó del
9,6 % al 18,1 % de los frames. Comparar el global antes y después compara dos datasets distintos.

Para el confounder, la métrica que se movió de verdad es **cuántos ejemplares quedan con material
alineado**: `Disgustado` pasó de 11 a **32** de sus 42 caballos.

**Y falta una métrica más, que es la que decide si se puede entrenar: clips con los 16 frames
alineados.** El modelo consume clips, no frames sueltos, y un clip con 11 frames alineados y 5
recortes de caja mete el confondir de recorte adentro de una sola muestra. Se pide con
`reporte.clips_integros()`.

Estado al **28/08/2026**, con el relleno temporal (`MAX_HUECO_INTERPOLADO = 1`), el piso de
video corto en 1,0 s y 14 frames anotados a mano:

| | clips íntegros | `Disgustado` clips/caballos |
|---|---:|---|
| sólo posiciones detectadas | **82** de 368 | **7 / 5** |
| con estimadas y anotadas (vigente) | **141** de 368 | **21 / 15** |

**Con 15 ejemplares en `Disgustado`, `GroupKFold` por caballo es viable.** Verificado con 3, 4 y
5 particiones: ninguna clase queda vacía en ningún fold.

Cómo se llegó, porque el orden importa y ninguna palanca sola alcanzaba:

| | clips íntegros | `Disgustado` ejemplares |
|---|---:|---:|
| punto de partida | 81 | 5 |
| + 15 videos nuevos filmados | 81 | 5 |
| + relleno temporal de keypoints | 130 | 8 |
| + `PISO_VIDEO_CORTO` a 1,0 s | 133 | 10 |
| **+ 14 frames anotados a mano** | **141** | **15** |

**Advertencia para el diseño experimental:** con 5 folds, dos de ellos tienen sólo 2 clips de
`Disgustado` en test. La métrica de esa clase va a tener una varianza enorme. Conviene **3 o 4
folds**, donde el mínimo por fold sube a 5 y 3 clips respectivamente, y reportar el desvío entre
folds y no sólo la media.

**La proyección con la que se planificó el change no se sostuvo, y conviene registrarlo.** El
diseño esperaba 180 clips íntegros y 19 ejemplares de `Disgustado` rellenando huecos interiores.
Medido: 130 y 8 (sobre 354 clips). Para llegar a 17 ejemplares haría falta un tope de 3, cuyo error mediano ya
supera la línea de base. La interpolación movió el dataset de *imposible* a *insuficiente* — es un
avance real, pero **no sustituye a filmar más caballos de la clase minoritaria**.

### Lo que se cerró el 25/08

1. **`fuente-unica-de-videos`.** La corrida del 24/08 midió sobre 223 de los 234 videos y publicó
   72,0 % de cobertura, por encima del umbral. Era falso: faltaban 12 videos de `Sereno` que
   alinean al 55,8 %, y sobre el material completo el valor es 70,2 %. Quedó una sola fuente
   declarada, el espejo retirado, las rutas en el kernel y el `501_8v1_LY1ucwR9.avi` —etiqueta
   contradictoria— fuera de todos los artefactos.
2. **`piso-de-video-corto`.** `PISO_VIDEO_CORTO = 2.0`: un video más corto que un clip entra
   entero. Los videos que llegan al piso de 5 s conservan exactamente sus clips —verificado clip
   por clip—, así que el diseño de 6 s del plan sigue rigiendo donde puede cumplirse.

### La primera acción de la próxima sesión

**`interpolacion-temporal-de-keypoints` está implementado y medido (28/08/2026): las 17 tareas
cerradas.** Lo que sigue es conseguir ejemplares nuevos de `Disgustado`; ninguna palanca de
software mueve ya la métrica que bloquea.

Lo que dejó, además del relleno:

- `equinos/interpolacion.py` y `tests/test_interpolacion.py` (49 pruebas), con la validación de
  dejar-uno-afuera completa y su línea de base.
- `origen_alineacion` por frame en `dataset_frames_alineados.csv`, y los reportes desglosados: la
  cobertura se puede recalcular contando sólo posiciones detectadas, que es la comparable con las
  corridas anteriores al relleno.
- **Una lección de método que conviene no repetir:** el umbral de aceptación se declaró antes de
  medir —como corresponde— pero sin anclarlo en ninguna referencia física, y resultó imposible de
  cumplir por construcción: exigía cinco veces más precisión que el propio movimiento del caballo
  entre frames. Hacen falta las dos cosas, declararlo antes **y** anclarlo en una línea de base
  medible. Está contado en `validacion.md` del change.

El plan que resta:

1. ~~El experimento de dejar uno afuera.~~ **Hecho.** `MAX_HUECO_INTERPOLADO = 1`,
   `MAX_EXTREMO_SOSTENIDO = 0`; sólo el largo 1 queda por debajo de la línea de base.
2. ~~Implementar el relleno y re-derivar.~~ **Hecho.** 81 → 130 clips íntegros; `Disgustado` de 5
   a 8 ejemplares, no a los 19 esperados.
3. ~~Anotar a mano los frames que completan un clip.~~ **Hecho (28/08/2026).** De los 23
   candidatos se anotaron 14 y se saltearon 9 —los que tenían confianza 0 en ambos keypoints:
   la cara está fuera de cuadro o tapada, y no había nada que marcar—. `Disgustado` pasó de 10
   a **15 ejemplares** y de 13 a **21 clips íntegros**. Se anota con `./correr_local.sh anotar`,
   en el navegador, y quedan trazados con `origen_alineacion = "manual"`.
4. ~~Bajar `PISO_VIDEO_CORTO` a 1,0 s.~~ **Hecho (28/08/2026).** Los 11 caballos de
   `Disgustado` que se perdían enteros por durar menos de 2 s: entraron 7, y sus clips alinean
   al **68,0 %**, mejor que el 63,1 % del resto de la clase — 16 frames sobre 1,5 s se mueven
   menos entre sí. La contrapartida es que esos clips cubren una ventana temporal más corta que
   los de 6 s, y eso hay que declararlo en el informe. Corrido con `-p PISO_VIDEO_CORTO 1.0`;
   **el default del notebook sigue en 2.0 y hay que cambiarlo si se adopta.**
5. **Barrer `CONF_MINIMA_KEYPOINT`** con `reporte.confianzas()`, que también lee el CSV y no
   reinfiere: puede rescatar frames antes de tener que estimarlos.
4. **Reemplazar el criterio de aceptación**: `COBERTURA_MINIMA_ACEPTABLE` sobre el global premia
   dejar afuera a la clase difícil. El criterio útil es cobertura por clase más clips íntegros por
   ejemplar.
5. **Más caballos de `Disgustado`**, en paralelo, porque depende de terceros y tiene el plazo más
   largo. Son caballos nuevos lo que hace falta, no más videos de los 32 que ya están.
6. **Recién después, el objetivo 3.** La aumentación de datos pertenece a esa etapa: es un
   regularizador de entrenamiento, no una mejora del dataset, y no crea ejemplares nuevos, que es
   lo único que rompe el confounder.

### Deudas que quedan abiertas

- **La rama no está pusheada.** Mientras siga así, Colab no puede correr este código: su
  sincronización clona `main` desde GitHub.
- **La purga de una unidad retirada del catálogo es manual.** El checkpoint nunca poda las unidades
  que salieron del universo: el `501` sobrevivió a una re-corrida entera y hubo que sacarlo a mano
  de los dos CSV parciales. Candidato a change propio.
- **`etapas.estado()` no compara los artefactos contra el catálogo.** Informa «completa» mirando
  sólo que existan y tengan filas, así que con material nuevo el orquestador saltea la etapa en
  silencio. Por eso existen los subcomandos `etapa01` y `etapa02`.
- **El change `ejecucion-local-con-gpu` tiene la tarea 2.4 sin cerrar**: verificar que las celdas de
  arranque siguen funcionando en Colab. Sólo se puede comprobar ejecutando allá.

### Dónde quedó el dataset

| | inicio | 22/08 | ahora |
|---|---|---|---|
| Videos | 34 | 146 | **234** |
| Clips | 92 | 210 | **342** |
| Frames | 1472 | 3360 | **5472** |
| Alineados | 59,5 % | 68,0 % | **69,9 %** |
| Frames de `Disgustado` | 160 (11 %) | 368 (11 %) | **992 (18,1 %)** |
| Ejemplares de `Disgustado` | **1** | 5 | **32** de 42 |

**`Disgustado` dejó de ser la clase que no se puede evaluar.** Con 32 ejemplares admite partición
por caballo sin que la clase desaparezca de una de las particiones, y el motivo que sostenía el
plan B de 2 clases queda enterrado con datos: no era que el material viniera de un solo caballo,
era que el piso de duración dejaba entrar uno solo.

### Tres causas encontradas y resueltas, en orden de tamaño

1. **Orientación de video** (720 frames en juego). 42 de los 146 videos guardan la rotación como
   metadato y OpenCV entregaba el frame acostado; el detector no reconocía un caballo girado 90
   grados. Perdían el 55,6 % de sus frames contra 0,0 % de los derechos. Corregido con
   `CAP_PROP_ORIENTATION_AUTO`; verificado.
2. **Encuadre canónico imposible** (2228 recortes afectados). El recorte abarcaba 2,5 veces la
   distancia ojo-hocico, que es el 26 % de la diagonal: pedía 1454 px de lado en cuadros de 1080
   de ancho. Sólo el 2,5 % salía sin relleno. Corregido a separación 0,60 más `BORDER_REPLICATE`;
   verificado el 22/08 — los recortes mayormente vacíos pasaron de 97,5 % a 0,2 %.
3. **Bodypart del hocico** (24 frames). Refutado: `upper_jaw` da 54,4 % contra 62,4 % de `nose`.
   Queda `nose` fijo. Resultado negativo, documentado y cerrado.

A esas tres se suma una cuarta, de otra naturaleza y más grande que las dos últimas: **el piso de
duración de los clips**, que no descartaba tomas malas sino ejemplares enteros de la clase
minoritaria.

### Lo que hay que recordar sobre el método

Las causas se manifestaban tarde y apuntando al lugar equivocado. Lo que destapó la más grande fue
**medir la pérdida contra los frames esperados en vez de contra los generados**: `cobertura()`
informaba 68,1 % cuando la cifra honesta era 51,9 %. La segunda apareció al preguntarse si un
recorte geométricamente exacto era además utilizable — el Paso 16 da error 1e-13 aunque el hocico
esté sobre una oreja, porque mide la transformada contra los mismos puntos con que se construyó.

El 25/08 el mismo error apareció por tercera vez y con otra cara: una cobertura de 72,0 % que
superaba el umbral **porque le faltaban los videos que peor alinean**, y una de 69,9 % que baja
mientras todas las clases mejoran o quedan igual. En los dos casos el problema no es el numerador
sino el universo sobre el que se calcula.

Conviene desconfiar de toda métrica cuyo denominador no se haya elegido a propósito, y de toda
comparación entre dos números calculados sobre datasets distintos.

### Deudas conocidas, sin change abierto

- **La cobertura global sigue siendo el criterio de aceptación** (`COBERTURA_MINIMA_ACEPTABLE`),
  aunque se demostró que se mueve con la composición del dataset. Reemplazarlo por cobertura por
  clase es la primera acción propuesta.
- **`00_pipeline.ipynb` no se sincroniza** y hay que subirlo a mano, lo que además apaga el
  permiso del secret `GITHUB_TOKEN` para ese notebook.
- **Etiquetado:** copias del mismo video con nombres **distintos** en dos clases siguen pasando sin
  aviso; `catalogo.verificar()` sólo detecta homónimos. Conviene revisar el criterio con que se
  armaron las carpetas.
- **Clips de largo variable (2 a 7 s).** Los 16 frames de un clip abarcan distinto tiempo según el
  clip. Queda registrado en `duracion_seg` de `inventario_clips.csv` y admite una ablación, pero
  hay que informarlo.

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

1. **Volumen bajo.** 234 videos → 342 clips → 5472 frames (25/08/2026). Es 3,7 veces el dataset
   de partida, pero sigue siendo insuficiente para entrenar desde cero cualquier arquitectura
   espacio-temporal → obliga a *transfer learning* y aumentación agresiva.
2. **Desbalance, atenuado y todavía presente.** 160 / 120 / 62 clips (47 % / 35 % / 18 %) contra
   el 64 / 25 / 11 % de julio. Un clasificador trivial que siempre prediga "Sereno" alcanza
   47 % de accuracy: **la accuracy global sigue sin ser métrica válida**; usar F1 macro, balanced
   accuracy y matriz de confusión. `Disgustado` ya no proviene de un solo caballo —son 32
   ejemplares—, pero la identidad del caballo sigue pudiendo predecir la clase, así que la
   partición por `id_caballo` es obligatoria.
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
