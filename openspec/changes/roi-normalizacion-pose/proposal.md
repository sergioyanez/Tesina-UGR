# Propuesta: Detección de ROI facial y normalización de pose

**Objetivo específico del plan de trabajo:** el **2** — *"Desarrollar un pipeline que permita el
preprocesamiento de los videos, incluyendo la extracción de fotogramas clave y detección de la
región de interés (ROI)"*. Cubre además el requisito metodológico de *"asegurar que los rostros de
los equinos estén alineados a lo largo de los clips"*.

## Por qué

Hoy los frames que produce el Bloque 3 son la imagen completa 1080×1920, y eso arrastra dos
problemas que invalidan cualquier resultado de clasificación:

1. **El fondo predice la clase.** Los 3 videos de Disgustado son interior de box (pared de madera,
   sombra); Interesado y Sereno son casi todos exterior (pasto, árboles, alambrados). Un modelo
   puede separar Disgustado sin mirar la cara ni una vez.
2. **La pose de la cabeza varía sin control.** Hay desde frontal completo hasta perfil total, donde
   se pierde un ojo y cambia por completo la geometría del hocico. Comparar microexpresiones entre
   clips con poses tan distintas no tiene sentido geométrico.

Sin resolver esto, el modelo del objetivo 3 aprendería fondo e identidad del ejemplar en lugar de
expresión facial, y el trabajo no podría defender sus conclusiones.

## Qué cambia

- **Nuevo notebook `02_roi_y_normalizacion.ipynb`**, que consume `inventario_clips.csv` (salida del
  Bloque 2 de `01_preparar_dataset_equinos.ipynb`) y produce frames faciales recortados y
  alineados.
- **Detección del equino** en cada frame con un detector preentrenado en COCO, que ya incluye la
  clase `horse` y por lo tanto **no requiere anotación manual**.
- **Estimación de keypoints faciales** (ojo y hocico como mínimo) con un modelo de pose
  preentrenado para cuadrúpedos, también sin anotación.
- **Normalización de pose** mediante transformada de similitud: todos los recortes salen con el ojo
  y el hocico en la misma posición canónica, sin importar la orientación original de la cabeza.
- **Suavizado temporal** de las detecciones a lo largo del clip, para que la ROI no salte entre
  frames consecutivos.
- **Nuevo CSV `dataset_frames_alineados.csv`**, con las mismas columnas de trazabilidad que
  `dataset_frames.csv` (incluido `id_caballo`) más las métricas de calidad de la detección.
- **Reporte de cobertura**: en qué porcentaje de frames se detectó el caballo y los keypoints, por
  clase y por caballo. Es la métrica que decide si el pipeline es utilizable.

**No** se elimina el pipeline actual: los frames sin recortar se conservan para poder comparar el
rendimiento del modelo con y sin normalización. Esa comparación es en sí misma un resultado
presentable en el capítulo de Resultados.

## Capacidades

**Nuevas:**

- `preprocesamiento-facial` — detección de ROI facial, estimación de keypoints, normalización de
  pose y control de calidad de la detección.

**Modificadas:** ninguna. El Bloque 2 (segmentación en clips) y el Bloque 3 (extracción de frames)
quedan intactos; este cambio agrega una etapa posterior.

## Impacto

- **Nuevas dependencias:** `ultralytics` (detección) y un modelo de pose para cuadrúpedos. Ambas se
  instalan en Colab; los pesos preentrenados se descargan en tiempo de ejecución.
- **Requiere GPU** para tiempos razonables sobre 1472 frames. Colab la provee; el entorno local no.
- **No modifica** `01_preparar_dataset_equinos.ipynb` ni ninguno de sus CSV.
- **Habilita** el objetivo 3: el modelo se entrenará sobre los frames alineados.

## Riesgo principal

Los modelos preentrenados son de propósito general y **no se pudieron validar fuera de Colab** por
falta de acceso a los pesos en el entorno local. Es posible que los keypoints faciales no sean
suficientemente precisos en tomas de perfil cerrado o con poca luz.

Por eso el paso de control de cobertura no es opcional: **si la detección de keypoints no supera un
umbral razonable de frames, hay que caer al plan alternativo** de anotación manual de la ROI. Con
92 clips, anotar una caja por clip es viable en pocas horas de trabajo humano y da una referencia
exacta contra la cual medir cualquier detector automático.
