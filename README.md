# Detección Automática de Expresiones Faciales en Equinos

Tesina de la **Licenciatura en Ciencias de Datos** — Universidad del Gran Rosario (UGR).
Asignatura: Seminario de Trabajo Final.

Detección y clasificación de expresiones faciales en equinos a partir de secuencias de video
cortas (5–7 s) mediante aprendizaje profundo.

**Integrantes:** Héctor R. Liceaga · Sergio M. Yañez
**Docentes:** Alejandro Hernández · Brian Gauna · Jorge Fernández

---

## Contexto del proyecto

El documento de referencia es **[`openspec/project.md`](openspec/project.md)**: contiene los
objetivos, el estado de cada uno, la caracterización del dataset, el pipeline, las convenciones y
—sobre todo— las **limitaciones conocidas** que condicionan cualquier resultado. Leerlo antes de
tocar el código.

El plan de trabajo original está en [`plan_de_trabajo.pdf`](plan_de_trabajo.pdf).

## Estructura

```
├── 00_pipeline.ipynb                   Orquestador: ejecuta las etapas en orden (Colab)
├── 01_preparar_dataset_equinos.ipynb   Inventario, clips de 5-7 s y frames (Colab)
├── 02_roi_y_normalizacion.ipynb        ROI facial y normalización de pose (Colab)
├── equinos/                            Infraestructura de ejecución
│   ├── rutas.py                        Raíz persistente en Drive y resolución de rutas
│   ├── catalogo.py                     Qué videos de Drive pertenecen a cada clase
│   ├── checkpoint.py                   Reanudación de etapas interrumpidas
│   ├── cache.py                        Empaquetado de artefactos numerosos
│   └── etapas.py                       Registro del pipeline y su estado
├── tests/
│   ├── test_catalogo.py                Pruebas del catálogo de videos por clase
│   ├── test_geometria.py               Pruebas de la geometría del notebook 02
│   └── test_persistencia.py            Pruebas del paquete equinos
├── openspec/
│   ├── project.md                      Contexto completo del proyecto
│   ├── config.yaml                     Reglas para generación de artefactos
│   └── changes/                        Cambios planificados (propuesta, spec, diseño, tareas)
├── plan_de_trabajo.pdf                 Plan de trabajo presentado
└── videos/                             Dataset — NO versionado (ver abajo)
```

## Los datos no están en el repositorio

La carpeta `videos/` pesa ~1.9 GB y tres archivos superan los 100 MB, el límite duro de GitHub.
El dataset vive en **Google Drive**, y los notebooks lo leen desde ahí montando la unidad: no se
descarga nada en cada sesión.

Para trabajar en local, copiar la carpeta de Drive a `videos/` respetando su estructura, y
apuntar el pipeline ahí con `export TESINA_EQUINOS_VIDEOS=/ruta/a/videos`.

**La carpeta de una clase no es toda la clase.** Parte del material etiquetado vive en subcarpetas
de trabajo, y cuáles suman **no es igual en todas**:

```
Videos de prueba/
├── Disgustado/                         videos sueltos
│   ├── Seleccionar cabeza?/            + suma a Disgustado
│   └── Se tapa una parte de la cara/   + suma a Disgustado
├── Interesado/
│   └── Seleccionar cabeza?/            + suma a Interesado
└── Sereno/
    ├── Seleccionar cabeza?/            + suma a Sereno
    └── Separar cabeza?/                NO suma: sin etiqueta decidida
```

Esa declaración vive en `equinos/catalogo.py` (`SUBCARPETAS_POR_CLASE`) y es el único lugar donde
se decide qué entra al dataset. Lo que no está declarado queda afuera, pero el pipeline lo informa
con nombre y conteo al arrancar: descartar y perder no son lo mismo. Para incorporar una
subcarpeta alcanza con agregarla ahí.

## Cómo se ejecuta

Todo sale de **`00_pipeline.ipynb`**. Es el único notebook que hace falta abrir: ejecuta las etapas
en orden, saltea las que ya están completas y guarda una copia ejecutada de cada una.

**Preparación, una sola vez:**

1. En Google Drive, agregar un **acceso directo** a la carpeta compartida `Videos de prueba` en
   Mi unidad (Compartidos conmigo → clic derecho → Organizar → Agregar acceso directo). No copia
   ni mueve los videos, y sin eso Colab no puede leerla como ruta. Es la carpeta que contiene
   `Sereno`, `Interesado` y `Disgustado`.
2. En Colab, cargar un secret `GITHUB_TOKEN` (panel izquierdo, icono de llave) con permiso de
   lectura del repo. Es necesario porque el repositorio es privado. Alternativa sin token: copiar
   a mano la carpeta `equinos/` y los notebooks `NN_*.ipynb` a `MyDrive/tesina_equinos/codigo/`.
3. Verificar que haya cuota libre en Drive: los clips ocupan ~2.7 GB.

**Después, cada vez:** abrir `00_pipeline.ipynb` y ejecutarlo de arriba a abajo. El Paso 2 muestra
en qué estado está cada etapa sin ejecutar nada.

### En una máquina local con GPU

El pipeline corre igual fuera de Colab, y los notebooks no necesitan cambios: detectan el entorno
solos. Verificado el 21/08/2026 sobre Ubuntu 22.04, driver NVIDIA 535 y una RTX 3070 Laptop de
8 GB, que le alcanza de sobra —la inferencia usa 1,64 GB de VRAM.

**Preparación, una sola vez:**

```bash
uv venv --python 3.10 ~/.venvs/tesina-equinos
VIRTUAL_ENV=~/.venvs/tesina-equinos uv pip install -r requirements-local.txt \
    --extra-index-url https://download.pytorch.org/whl/cu121
~/.venvs/tesina-equinos/bin/python -m ipykernel install --user \
    --name tesina-equinos --display-name "Tesina equinos (GPU local)"
```

El `+cu121` no es cosmético: una rueda de torch compilada para una CUDA más nueva que la del
driver deja la GPU invisible y el pipeline cae a CPU **sin fallar**. `nvidia-smi` informa la CUDA
del driver; tiene que ser igual o mayor que la de la rueda.

**Variables de entorno** (en el shell desde el que se lanza Jupyter):

| variable | para qué | valor sugerido |
|---|---|---|
| `TESINA_EQUINOS_RAIZ` | dónde se escriben los artefactos | una carpeta en disco interno |
| `TESINA_EQUINOS_VIDEOS` | dónde están los videos crudos | una copia de `Videos de prueba` |

La raíz conviene tenerla en disco interno y no en el externo: los frames alineados son miles de
archivos chicos, que es el peor patrón de acceso para un exFAT por USB. Sin esas variables el
pipeline igual arranca, pero escribe en `<repo>/datos_equinos`.

**`<repo>/videos/` NO sirve como fuente de videos.** Es un espejo viejo y divergente: comparado
archivo por archivo contra `inventario_videos.csv` (22/08/2026), comparten 135 nombres, pero a él
le faltan 11 que están en Drive y tiene 12 que el inventario nunca vio. Sirvió para verificar
geometría sin GPU y para eso alcanza; regenerar clips desde ahí construiría un dataset distinto
del que produjo la pose, y nada fallaría al hacerlo. La fuente es la carpeta compartida `Videos de
prueba` (5,3 GiB), que en local hay que copiar aparte:

```bash
rclone copy "tesina:Videos de prueba" ~/videos_de_prueba --transfers 8 -P
```

**Artefactos ya calculados.** Los CSV guardan rutas **relativas a la raíz**, así que copiar
`MyDrive/tesina_equinos/` dentro de la raíz local basta para seguir donde iba, sin reinferir. Hacen
falta `clips/` (el paso de alineación lee los píxeles del clip, no de la caché de frames),
`resultados/` y `checkpoints/`; `cache/frames_equinos.zip` recién para entrenar. **`videos/` no**:
es material viejo, los videos crudos salen de la carpeta compartida `Videos de prueba`. Después de copiar, correr
`etapas.diagnostico()` antes que cualquier otra cosa: una copia a medias deja el checkpoint sin
sus cachés y una etapa incompleta puede darse por completa.

**Después, cada vez:** usar `./correr_local.sh`, que fija las variables antes de arrancar nada.

| comando | qué hace |
|---|---|
| `./correr_local.sh` | abre Jupyter Lab (elegir el kernel *Tesina equinos (GPU local)*) |
| `./correr_local.sh estado` | en qué estado está cada etapa, sin ejecutar nada |
| `./correr_local.sh reportes` | cobertura y calidad de recortes de la última corrida |
| `./correr_local.sh entorno` | versiones instaladas y GPU visible |
| `./correr_local.sh etapa02` | re-ejecuta el notebook 02 con papermill, sin limpiar la etapa |

Existe porque olvidarse las variables **no falla**: el pipeline cae a una raíz vacía y al espejo
viejo de videos, corre entero y produce números de otro dataset.

### Por qué hay un orquestador

Cada notebook de Colab arranca con un `/content` vacío, así que el `02` no encontraba los clips ni
los CSV que generaba el `01`. Ahora todos los artefactos viven en `MyDrive/tesina_equinos/` y el
paso de datos entre etapas ocurre por ahí.

Además, Colab corta la sesión por límite de tiempo o inactividad, y la etapa de keypoints tarda
alrededor de una hora. Las etapas largas guardan su progreso por clip: si se corta, volver a
ejecutar el pipeline retoma donde había quedado en vez de empezar de nuevo.

Los parámetros que atraviesan varias etapas se fijan en el Paso 1 del orquestador y se inyectan con
`papermill`. Cambiar un umbral ahí invalida automáticamente el checkpoint de la etapa afectada, para
que no se mezclen resultados de dos configuraciones distintas —un error que no falla ni avisa, sólo
arrastra números equivocados hasta el final.

## Los notebooks de etapa

Pensados para Google Colab. Se pueden ejecutar sueltos, de arriba a abajo, y en ese caso toman los
valores por defecto de su celda de parámetros.

### `01_preparar_dataset_equinos.ipynb`

| Bloque | Qué hace | Salida |
|---|---|---|
| 1 | Lectura de los videos en Drive e inventario técnico | `resultados/inventario_videos.csv` |
| 2 | Segmentación en clips de 5–7 s (reanudable) | `resultados/inventario_clips.csv` + `clips/` |
| 3 | Extracción de 16 frames uniformes por clip | `resultados/dataset_frames.csv` + `cache/frames_equinos.zip` |

El Paso 0 monta Drive y resuelve la raíz de proyecto. **No borra nada**: el borrado es opt-in
(`LIMPIAR_ETAPA = True`) y sólo afecta a esta etapa. La versión anterior borraba las carpetas en
cada ejecución, lo que volvía imposible reanudar: reabrir el notebook destruía el trabajo hecho.

### `02_roi_y_normalizacion.ipynb` — requiere GPU

| Bloque | Qué hace | Salida |
|---|---|---|
| 1 | Configuración y funciones geométricas | — |
| 2 | Detección del equino y suavizado temporal | — |
| 3 | Keypoints faciales (reanudable) | `resultados/keypoints_faciales.csv` |
| 4 | ROI, alineación y recorte | `resultados/dataset_frames_alineados.csv` + `cache/frames_alineados.zip` |
| 5 | Control de calidad y decisión | `resultados/reporte_cobertura.csv` |

Recorta la cara del equino y la alinea, de modo que el ojo y el hocico caigan siempre en la misma
posición. Sin esto, un modelo puede separar las clases mirando el fondo —los 3 videos de Disgustado
son interior de box y el resto exterior— en lugar de la expresión facial.

El Paso 10 imprime el esquema de keypoints que expone el modelo y deja explícito cuáles eligió como
ojo y hocico. La extracción ya está adaptada a la estructura real de DeepLabCut, verificada en
Colab. **La corrida completa lleva alrededor de una hora** para los 92 clips.

El **Paso 14 decide si el pipeline sirve**: si la cobertura de keypoints no supera el umbral, hay
que ajustar umbrales o pasar a anotar manualmente las 92 ROI. La alternativa manual está prevista
en el diseño, no es un plan de emergencia.

### Pendiente

Los objetivos específicos 3 y 4 —modelo de clasificación, entrenamiento, evaluación e interfaz de
uso— todavía no están implementados.

## Pruebas

```
python3 tests/test_geometria.py
python3 tests/test_persistencia.py
python3 tests/test_catalogo.py
python3 tests/test_pose.py
python3 tests/test_entorno.py
```

`test_geometria.py` verifica las funciones geométricas del notebook 02 —suavizado, relleno,
transformada de similitud y cálculo de ROI— con 21 casos que incluyen los degenerados. **Las extrae
del propio notebook**, así que no hay dos copias que puedan divergir.

`test_persistencia.py` cubre el paquete `equinos` con 46 casos: resolución de rutas entre sesiones,
reanudación tras un corte, invalidación del checkpoint ante cambio de configuración, integridad de
la caché y estado de las etapas.

`test_entorno.py` verifica el chequeo de dependencias, sobre todo que las versiones se comparen
como números y no como texto: `torch` reporta `2.5.1+cu121` y, comparado como cadena, `"1.9"` es
mayor que `"1.26"` —un entorno roto pasaría por bueno. Corre en cualquier intérprete, incluso en
uno que no cumpla los requisitos.

Ninguna de las dos requiere GPU, internet ni pesos preentrenados. Es deliberado: son justamente los
puntos donde el error no rompe nada y se manifiesta como un número creíble y equivocado.

## Dos reglas que no se negocian

Están justificadas en detalle en `openspec/project.md`, pero se repiten acá porque invalidan
cualquier resultado si se ignoran:

1. **Particionar por caballo, nunca por frame ni por clip.** Los 16 frames de un clip son casi
   idénticos y varios videos comparten ejemplar. Usar `GroupKFold` sobre la columna `id_caballo`.
   El tamaño muestral efectivo son los **22 caballos**, no los 92 clips ni los 1472 frames.

2. **La accuracy global no es una métrica válida.** Las clases están desbalanceadas
   (64 % / 25 % / 11 %): predecir siempre "Sereno" ya da 64 %. Reportar **F1 macro**,
   **balanced accuracy** y matriz de confusión.

## Estado actual del dataset

| Clase | Videos | Clips (5–7 s) | Caballos distintos |
|---|---:|---:|---:|
| Sereno | 23 | 59 | 15 |
| Interesado | 8 | 23 | 8 |
| Disgustado | 3 | 10 | **1** |
| **Total** | **34** | **92** | **22** |

En ampliación. La prioridad de recolección es **Disgustado en otros ejemplares**: hoy la clase
proviene de un solo caballo, lo que la vuelve no evaluable bajo partición por caballo.

## Stack

Python 3.10 · OpenCV · pandas · numpy · matplotlib · scikit-learn · PyTorch · papermill ·
DeepLabCut · ultralytics
Previsto: TensorFlow.

Dos entornos de ejecución equivalentes: **Google Colab** (instala por sesión desde la celda de
arranque) y **local con GPU** (entorno pinneado en `requirements-local.txt`). Las versiones son un
conjunto que se verifica junto, porque DeepLabCut fija `numpy<2` y arrastra a todo lo compilado
contra numpy.
