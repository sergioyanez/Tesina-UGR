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
├── 01_preparar_dataset_equinos.ipynb   Descarga, clips de 5-7 s y frames (Colab)
├── 02_roi_y_normalizacion.ipynb        ROI facial y normalización de pose (Colab)
├── tests/
│   └── test_geometria.py               Pruebas de la geometría del notebook 02
├── openspec/
│   ├── project.md                      Contexto completo del proyecto
│   ├── config.yaml                     Reglas para generación de artefactos
│   └── changes/                        Cambios planificados (propuesta, spec, diseño, tareas)
├── plan_de_trabajo.pdf                 Plan de trabajo presentado
└── videos/                             Dataset — NO versionado (ver abajo)
```

## Los datos no están en el repositorio

La carpeta `videos/` pesa ~1.9 GB y tres archivos superan los 100 MB, el límite duro de GitHub.
El dataset vive en **Google Drive** y el notebook lo descarga solo con `gdown`.

Para trabajar en local, copiar la carpeta de Drive a `videos/` respetando la estructura
`videos/<clase>/<archivo>.mp4`. La subcarpeta **es** la etiqueta.

## Los notebooks

Ambos están pensados para Google Colab y se ejecutan de arriba a abajo, en orden.

### `01_preparar_dataset_equinos.ipynb`

| Bloque | Qué hace | Salida |
|---|---|---|
| 1 | Descarga desde Drive e inventario técnico | `inventario_videos.csv` |
| 2 | Segmentación en clips de 5–7 s | `inventario_clips.csv` + `clips_equinos/` |
| 3 | Extracción de 16 frames uniformes por clip | `dataset_frames.csv` + `frames_equinos/` |

El **Paso 0 borra las carpetas temporales** de Colab antes de empezar: es intencional, cada sesión
parte de cero y vuelve a descargar desde Drive.

### `02_roi_y_normalizacion.ipynb` — requiere GPU

| Bloque | Qué hace | Salida |
|---|---|---|
| 1 | Configuración y funciones geométricas | — |
| 2 | Detección del equino y suavizado temporal | — |
| 3 | Keypoints faciales | `keypoints_faciales.csv` |
| 4 | ROI, alineación y recorte | `dataset_frames_alineados.csv` + `frames_alineados/` |
| 5 | Control de calidad y decisión | reporte de cobertura |

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
```

Verifica las funciones geométricas del notebook 02 —suavizado, relleno, transformada de similitud y
cálculo de ROI— con 21 casos que incluyen los degenerados. **Las extrae del propio notebook**, así
que no hay dos copias que puedan divergir. No requiere GPU, internet ni pesos preentrenados.

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

Google Colab · Python 3 · OpenCV · pandas · numpy · gdown · matplotlib · scikit-learn · PyTorch
Previstos: TensorFlow, DeepLabCut.
