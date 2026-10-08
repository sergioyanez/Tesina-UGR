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

El plan de trabajo vigente está en [`docs/plan_de_trabajo_corregido.pdf`](docs/plan_de_trabajo_corregido.pdf).
Qué número sale de qué código y rama: [`docs/registro_de_experimentos.md`](docs/registro_de_experimentos.md).

## Estructura

```
├── 00_pipeline.ipynb                   Orquestador: ejecuta las etapas en orden (Colab)
├── 01_preparar_dataset_equinos.ipynb   Inventario, clips de hasta 7 s y frames
├── 02_roi_y_normalizacion.ipynb        Pose, ROI facial y normalización
├── 03_clasificacion.ipynb              Objetivo 3: líneas de base, MIL y evaluación (reproduce los scripts)
├── 04_figuras_tesis.ipynb              Figuras de la tesis a partir de los resultados vigentes
├── app_inferencia.py                   Objetivo 4: interfaz Streamlit
├── equinos/                            Paquete del pipeline
│   ├── catalogo.py · rutas.py · …      Catálogo de videos, rutas, checkpoints, caché
│   ├── pose.py · secuencia.py          Keypoints y selección de los 16 frames por ventana
│   ├── alineado.py · encuadre.py       Recorte alineado, compartido por entrenamiento e inferencia
│   ├── particion.py                    Clips íntegros y partición por caballo
│   ├── caracteristicas.py              ResNet-18 congelada y su caché
│   ├── clasificacion.py                Métricas, MIL multimodal, validación cruzada, producción
│   ├── fusion.py                       Modelo de fusión pose + visual para inferir y su confiabilidad
│   ├── lineas_de_base.py               Líneas de base, confusor de filmación, bootstrap por caballo
│   └── inferencia.py                   De un video a una predicción, o a una negativa explicada
├── scripts/
│   ├── evaluar_v2.py                   Evaluación vigente y modelo de producción
│   └── evaluar_lineas_de_base.py       Líneas de base sobre los mismos folds
├── tests/                              Guiones de prueba (python3 tests/test_X.py)
├── docs/                               Plan de trabajo, informes, registro de experimentos
└── openspec/                           Contexto (project.md), specs y changes
```

El dataset **no vive en el repositorio ni bajo la raíz de artefactos** (ver abajo).

## Los datos no están en el repositorio

El dataset pesa 5,3 GiB y varios archivos superan los 100 MB, el límite duro de GitHub. Vive en
**Google Drive**, y en Colab los notebooks lo leen desde ahí montando la unidad: no se descarga
nada en cada sesión.

Para trabajar en local, copiar la carpeta compartida a un directorio **fuera del repositorio** y
declararlo con `export TESINA_EQUINOS_VIDEOS=/ruta/a/videos_de_prueba`. **Fuera del repositorio, y
en particular fuera de la raíz de artefactos:** una copia del material bajo la raíz se convierte en
fuente candidata y el pipeline puede elegirla sin avisar. Pasó el 24/08/2026 y costó una corrida
entera de números de otro dataset.

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
| `TESINA_EQUINOS_RAIZ` | dónde se escriben los artefactos | `<repo>/datos_equinos` |
| `TESINA_EQUINOS_VIDEOS` | dónde están los videos crudos | `~/videos_de_prueba` |

Las dos están además en el `kernel.json` del kernel `tesina-equinos`, para que un Jupyter abierto
por fuera de `correr_local.sh` no arranque sin ellas. Si falta `TESINA_EQUINOS_VIDEOS` el pipeline
**corta**, informando qué rutas revisó: no hay ninguna copia del material bajo la raíz que pueda
sustituir a la fuente, y no debe volver a haberla.

La fuente es la carpeta compartida `Videos de prueba` (5,3 GiB, 234 videos), que en local hay que
copiar aparte:

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
| `./correr_local.sh interfaz` | levanta la interfaz de inferencia (Streamlit) |

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

El recorte es un rectángulo de **224×343**, no un cuadrado. Se declara como campo de visión en
unidades de la distancia ojo-hocico `d` —`AIRE_SOBRE_OJO = 1.00`, `AIRE_BAJO_HOCICO = 0.30`,
`SEMIANCHO_LATERAL = 0.75`— y el alto se deriva del aspecto que ese campo exige. La forma
rectangular **no deforma la cara**: lo que lo impide es la isotropía de la transformada de
similitud, no que el destino sea cuadrado.

Existe porque el encuadre cuadrado dejaba las orejas fuera, y en los equinos su posición es el
indicador principal de disgusto: el vector base→punta de la oreja era legible en el 4,3 % de los
frames, y con el encuadre nuevo lo es en el 77,2 %. Los candidatos se comparan sin escribir una sola
imagen con `equinos.encuadre.barrido()`, sobre la pose ya persistida. Ver el change
`encuadre-que-incluye-las-orejas`.

`dataset_frames_alineados.csv` registra por frame `origen_alineacion` —detectado, interpolado,
manual o recorte de caja— y `relleno_fraccion`, la parte del recorte completada con píxeles
inventados. Los dos existen para poder entrenar con y sin, no para tranquilizar: dependen del
ejemplar, y el ejemplar predice la clase.

El Paso 10 imprime el esquema de keypoints que expone el modelo y deja explícito cuáles eligió como
ojo y hocico. La extracción ya está adaptada a la estructura real de DeepLabCut, verificada en
Colab. **La corrida completa lleva alrededor de una hora** para los 92 clips.

El **Paso 14 decide si el pipeline sirve**: si la cobertura de keypoints no supera el umbral, hay
que ajustar umbrales o pasar a anotar manualmente las 92 ROI. La alternativa manual está prevista
en el diseño, no es un plan de emergencia.

## La interfaz de inferencia

```
./correr_local.sh interfaz
```

Recibe una toma corta de la cara de un caballo y devuelve la expresión predicha —`Sereno`,
`Interesado` o `Disgustado`— **junto con la confiabilidad medida de esa predicción**. La lógica
vive en `equinos/inferencia.py`; `app_inferencia.py` es sólo la presentación.

**Modelo (desde el 08/10/2026):** la **fusión tardía pose + visual**, ganadora de la comparación de
modelos (`comparacion/DECISION.md`), reemplaza al MIL por omisión. Se entrena con todos los clips
(`python comparacion/entrenar_modelo_final.py`, peso de la pose w = 0,75) y se guarda en
`<raíz>/modelos/fusion_pose_visual/fusion_pose_visual.joblib` (fuera del repo, junto con sus
metadatos: versión de scikit-learn, huella del dataset, fecha, lista de clips). El MIL sigue
disponible con `inferencia.clasificar_video(ruta, modelo="mil")`.

**Qué necesita:** la máquina con GPU, el entorno de `requirements-local.txt`, y el modelo
anterior (`modelos/objetivo3_v2/mil_multimodal_produccion.pt`) sólo si se pide el MIL. Corre DeepLabCut sobre todos los
fotogramas del video, así que tarda **alrededor de 80 s para una toma de 7 s** (medido el 18/09). La
preparación es la misma función que fabricó el conjunto de entrenamiento
(`alineado.preparar_secuencia`); YOLO ya no participa.

**Qué NO hace, y conviene decirlo antes de que alguien lo suponga:**

- **No es una herramienta de diagnóstico veterinario.** Es la demostración de un pipeline.
- **No siempre responde.** Necesita 16 fotogramas donde se vean el ojo y el hocico; si no los
  consigue, se niega y explica por qué. Negarse es un resultado normal, no un error:
  sobre el propio conjunto de datos el 91 % de los clips tiene una ventana utilizable.
- **No acepta videos largos.** Máximo 15 s, idealmente una toma de 5 a 7 s. Los 16 fotogramas se
  eligen dentro de la mejor ventana de hasta 7 s, que en un video largo puede no ser el tramo que
  interesa.
- **No reentrena, no procesa lotes y no guarda nada** del video que se le sube.

**El límite del entregable, medido y no supuesto.** La fusión tiene F1 macro **0,868 ± 0,046** en
desarrollo (113 caballos, 291 clips, 5×4 por caballo) y **0,834** en la reserva (20 caballos, 68
clips, un solo uso; IC95 [0,698; 0,925]); en tomas filmadas en condiciones parejas baja a 0,740 y
0,767. Los números que muestra la app salen de `comparacion/resultados/`. Cuando dice una clase,
acierta (desarrollo / reserva):

| la fusión dice | desarrollo (clips por repetición) | reserva (68 clips) |
|---|---:|---:|
| `Disgustado` | 72 de 89 — 81 % | 16 de 20 — 80 % |
| `Interesado` | 90 de 102 — 88 % | 12 de 17 — 71 % |
| `Sereno` | 90 de 100 — 89 % | 30 de 31 — 97 % |

Con la reserva, cada precisión se apoya en pocas decenas de clips. La ventaja sobre el RF de pose
es mínima y la pose domina la decisión (puede arrastrar rasgos de filmación). El modelo en uso se
ajustó con desarrollo + reserva: no tiene una medida propia. Para comparar, el MIL anterior (F1 0,740 ± 0,052, `objetivo3_evaluacion_v2.json`):

| el modelo dice | acierta |
|---|---:|
| `Disgustado` | 71 de 96 — **74 %** |
| `Interesado` | 94 de 123 — 76 % |
| `Sereno` | 102 de 140 — 73 % |

Y hay que decir con la misma claridad lo que el número del MIL **no** demostraba: un promedio lineal de las
características de los 16 frames rinde lo mismo (0,741 ± 0,020), y la duración y la resolución del
clip, sin mirar la cara, ya dan 0,618 (`objetivo3_lineas_de_base.json`). Ver la sección siguiente.

Por eso la interfaz muestra ese número al lado de la etiqueta y con el mismo tamaño, en vez de
dejarlo en un descargo al pie. Una pantalla que devuelve «Disgustado, 87 %» es indistinguible, para
quien la mira, de una herramienta validada.

## El resultado del objetivo 3, con sus líneas de base

Sobre los mismos 4 folds por caballo (359 clips, 133 caballos):

| Modelo | F1 macro |
|---|---:|
| Clase mayoritaria / sólo `id_caballo` | 0,183 ± 0,033 |
| Azar estratificado | 0,325 ± 0,025 |
| Metadatos de filmación (duración, fps, resolución), sin mirar la cara | 0,618 ± 0,034 |
| Visual, 1 frame central | 0,679 ± 0,024 |
| **MIL multimodal (vigente)** | **0,740 ± 0,052** |
| Visual, media de los 16 frames | 0,741 ± 0,020 |

- El modelo **lee la cara**: supera a los metadatos (+0,121, IC95 por caballo [+0,031; +0,209]) y
  conserva 0,705 sobre los clips 1080×1920 de al menos 5 s, donde la filmación no delata la clase.
- **No está demostrado que la arquitectura aporte**: contra el promedio estático la diferencia es
  −0,004, IC95 [−0,066; +0,058]. Ni la atención, ni el orden, ni el canal geométrico tienen un aporte
  demostrado. La hipótesis temporal no queda sostenida.
- **Sobreajusta el train** (F1 0,98 contra 0,71 de validación); el *early stopping* mantiene el test
  en línea con la validación. Ver [`docs/analisis_sobreajuste.md`](docs/analisis_sobreajuste.md).
- **Las etiquetas tienen un solo anotador**: hay criterio escrito con EquiFACS, pero el acuerdo
  sólo se midió en un piloto sobre imágenes. Ver
  [`docs/protocolo_de_etiquetado.md`](docs/protocolo_de_etiquetado.md).

Para reproducirlo: `PYTHONPATH="$PWD" python3 scripts/evaluar_v2.py` y después
`PYTHONPATH="$PWD" python3 scripts/evaluar_lineas_de_base.py`.

## Pruebas

Son guiones, no pytest: cada uno se corre con `python3 tests/test_X.py` e informa OK/FALLA por caso.

```
for t in tests/test_*.py; do PYTHONPATH="$PWD" python3 "$t" || echo "FALLÓ $t"; done
```

`test_equivalencia.py` tiene una comprobación que falla («elige los mismos fotogramas», clip 013)
aunque los recortes salen idénticos píxel a píxel; está registrada como deuda.

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
   El tamaño muestral efectivo son los **133 caballos**, no los 359 clips ni los 5744 frames.

2. **La accuracy global no alcanza.** Reportar **F1 macro**, **balanced accuracy**, métricas por
   clase y matriz de confusión, **siempre al lado de las líneas de base** —incluida la de metadatos
   de filmación, que es el piso honesto—.

## Estado actual del dataset

Conjunto vigente (`particion_clips.json`, selección por ventana del 18/09/2026):

| Clase | Clips | % | Caballos distintos |
|---|---:|---:|---:|
| Sereno | 138 | 38,4 % | 52 |
| Interesado | 118 | 32,9 % | 60 |
| Disgustado | 103 | 28,7 % | 57 |
| **Total** | **359** | | **133** |

De 274 videos y 393 clips candidatos entran 359 (91,3 %). Los caballos no suman 133 porque 29
aparecen en más de una clase. Desbalance mayor:menor 1,34 en clips y 1,15 en caballos.
**Las clases no se filmaron igual**: `Disgustado` tiene clips de 3,2 s de mediana contra 6,0 s de
las otras dos, y los 26 clips 640×480 son todos `Interesado`.

## Stack

Python 3.10 · OpenCV · pandas · numpy · matplotlib · scikit-learn · PyTorch · papermill ·
DeepLabCut · ultralytics
Previsto: TensorFlow.

Dos entornos de ejecución equivalentes: **Google Colab** (instala por sesión desde la celda de
arranque) y **local con GPU** (entorno pinneado en `requirements-local.txt`). Las versiones son un
conjunto que se verifica junto, porque DeepLabCut fija `numpy<2` y arrastra a todo lo compilado
contra numpy.
