# Plan de Implementación: Pipeline Equinos

Este documento detalla los cambios técnicos que realizaré en la rama `hector` para cumplir con las tareas planteadas en el informe.

## User Review Required

> [!IMPORTANT]
> Se requieren decisiones sobre hiperparámetros. En este plan propongo valores iniciales (ej. un umbral de distancia de Hamming de 2 para descartar frames muy parecidos), pero esto podría ajustarse luego viendo los resultados empíricos.

## Proposed Changes

A continuación se detallan las modificaciones propuestas agrupadas por archivo.

### Notebook 02: Detección de ROI y Normalización

La limpieza temporal de frames (imagededup) se introducirá justo antes de correr YOLO, para evitar que la red neuronal procese frames repetidos (ahorrando memoria y tiempo).

#### [MODIFY] [`02_roi_y_normalizacion.ipynb`](file:///C:/Users/hrlic/Desktop/UGR/Lic%20Ciencia%20de%20Datos/Materias/Seminario/Tesina-UGR/02_roi_y_normalizacion.ipynb)
- **Paso 1 (Dependencias):** Agregar `imagededup` al comando de instalación `!pip install`.
- **Paso 0a (Parámetros):** 
  - Cambiar `FRAC_OJO = 0.20` a `FRAC_OJO = 0.25` para dar más espacio en la parte superior y asegurar que las orejas queden dentro del recorte.
  - Agregar `UMBRAL_PHASH = 2` como parámetro para el hashing.
- **Paso 6 (Detección en cada frame):** 
  - Reemplazar la extracción estrictamente uniforme `np.linspace(0, total, 16)` por un sobre-muestreo inicial (ej. 48 frames).
  - Instanciar `PHash()` de `imagededup`.
  - Recorrer el sobre-muestreo, generar el hash de cada frame con `encode_image(image_array=frame)`, y calcular la distancia de Hamming respecto al último frame aceptado.
  - Conservar el frame solo si la distancia es `>= UMBRAL_PHASH`, hasta llegar a un máximo de `NUM_FRAMES` (16). Si se acaba el clip antes de llegar a 16, se quedan los frames únicos que haya.

### Notebook 03: Entrenamiento y Balanceo (Nuevo)

Crearemos el inicio del notebook 3, dejando estructuradas las estrategias de Data Augmentation y balanceo.

#### [NEW] `03_entrenamiento.ipynb`
- **Paso de Data Augmentation:** Configurar transformaciones probabilísticas (espejado horizontal `RandomHorizontalFlip`, ligeras rotaciones `RandomRotation` de ±15 grados, y ajustes de brillo/contraste) enfocadas especialmente en las clases minoritarias (Disgustado, Interesado).
- **Esqueleto del modelo:** Carga de los recortes normalizados desde Drive, definición de DataLoader con muestreo ponderado (WeightedRandomSampler) para equilibrar los lotes, y definición de la arquitectura base (ej. ResNet o red 3D).

## Verification Plan

### Manual Verification
- Validar visualmente la calidad de los recortes tras el ajuste de `FRAC_OJO`.
- Correr el notebook 02 sobre un video "Sereno" de prueba y constatar que descarta correctamente los frames idénticos (reduciendo artificialmente la carga de trabajo de la red).
