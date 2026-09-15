# Implementación de la Nueva Arquitectura y Pipeline

Se proponen 4 cambios principales a la arquitectura del proyecto para cumplir con los requerimientos solicitados.

## 1. Muestreo Uniforme Estricto en el Preprocesamiento

Actualmente, el pipeline usa `np.linspace(0, total - 1, 16).astype(int)`. Si un video tiene menos de 16 frames, esto genera frames duplicados. 
**Cambios:**
- Modificaremos la lógica en `01_preparar_dataset_equinos.ipynb` (y `equinos/inferencia.py` o donde se extraigan) para garantizar un muestreo estrictamente equidistante sin duplicados.
- Si un clip tiene menos de 16 frames: extraeremos el máximo posible de frames únicos equidistantes (por ejemplo, 12, 13 o 14).
- Si el clip es demasiado corto (ej. menos de 12 frames), se descartará directamente.
- Al permitir que los clips tengan menos de 16 frames (ej. 12-16), se habilita el uso de secuencias de longitud variable que el modelo deberá rellenar (padding).

## 2. Clase `Dataset` en PyTorch

Actualmente no hay una clase `Dataset` formal de PyTorch; los datos se manejan en memoria.
**Cambios:**
- Crear `MultimodalEquineDataset` (en `equinos/clasificacion.py` u otro módulo).
- **Entradas:** 
  1. Características visuales cacheadas de ResNet-18.
  2. Coordenadas de pose (orejas, ojos, hocico) leídas de `pose_completa.csv`.
- **Lógica de Padding:** Dado que la longitud puede variar entre 12 y 16, usaremos `pad_sequence`.
- **Attention Masks:** Se generará una máscara booleana (`True` para frame real, `False` para padding).
- **Gestión de NaNs:** Se detectarán los keypoints faltantes (común en las orejas), reemplazando los NaNs por ceros (o la media). Se retornará una `validity_mask` adicional para indicar al MLP geométrico qué keypoints son sintéticos.

## 3. Arquitectura Multimodal

Crearemos un nuevo modelo en PyTorch (`nn.Module`).
**Cambios:**
- **Visual:** Usa directamente el embedding de 512 dimensiones de ResNet-18 (que seguirá congelada y pre-extraída).
- **Geométrico:** Un MLP pequeño que recibe los keypoints (aplanados) y la `validity_mask` (concatenada o como pesos). El MLP producirá un embedding geométrico (ej. 128 o 256 dimensiones).
- **Fusión:** Concatenación del vector visual y geométrico.

## 4. Multiple Instance Learning (MIL) basado en Atención

**Cambios:**
- Implementar el mecanismo de Atención MIL (Ilse et al., 2018) dentro de la arquitectura.
- La red calculará un peso de atención (un escalar) para cada frame en la secuencia fusionada.
- Aplicar la *Attention Mask* temporal calculada en el Dataset para asignar un peso de `-inf` (antes del softmax) a los frames que son relleno, asegurando que su peso sea exactamente 0 y no contribuyan al Pooling.
- La predicción final a nivel de clip (bag) se obtendrá de la suma ponderada de las características de cada frame, seguida de una capa final de clasificación a 3 clases.
- **Pérdida (Loss):** Se calculará a nivel de clip usando `CrossEntropyLoss`.

## User Review Required

> [!WARNING]
> La extracción de características cacheadas (`caracteristicas_backbone.npz`) asume que cada clip tiene **exactamente 16 frames** de tamaño fijo (`n_clips, 16, 512`). Si introducimos secuencias de tamaño variable en la Etapa 1, la caché de características también tendrá tensores de distinto tamaño.
> Para no romper esto, sugiero rellenar con ceros las imágenes y los CSV en disco o hacerlo puramente en tiempo de ejecución (Dataset). ¿Preferís que el "padding" se haga en disco (generando frames negros) o extraemos solo los frames válidos (ej. 12 frames), guardamos tensores de largo variable, y la clase `Dataset` en PyTorch hace el zero-padding de los arrays numpy antes del modelo?

## Open Questions

> [!IMPORTANT]
> 1. Para los keypoints del MLP, ¿querés que usemos las coordenadas absolutas relativas al frame o coordenadas relativas normalizadas por algún punto fijo (ej. distancia ojo-hocico)?
> 2. Respecto a los keypoints faltantes de la oreja: si reemplazamos con 0, ¿te parece bien que además le pasemos al MLP la "validity_mask" concatenada a los keypoints? Ejemplo: Si hay 6 coordenadas (x,y de 3 puntos), pasamos 6 valores + 3 flags booleanos = 9 entradas.
