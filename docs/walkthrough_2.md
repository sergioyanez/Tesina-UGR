# Cambios Arquitectónicos Completados

Se implementaron todos los requerimientos solicitados referidos a la arquitectura Multimodal MIL y al pipeline de extracción de secuencias de longitud variable. A continuación, el detalle de las modificaciones:

## 1. Muestreo Uniforme Estricto

Se actualizaron los notebooks y scripts base para prevenir repeticiones de frames causadas por `np.linspace` cuando los clips son cortos. 
- En [`01_preparar_dataset_equinos.ipynb`](file:///c:/Users/hrlic/Desktop/UGR/Lic%20Ciencia%20de%20Datos/Materias/Seminario/Tesina-UGR/01_preparar_dataset_equinos.ipynb) y [`equinos/inferencia.py`](file:///c:/Users/hrlic/Desktop/UGR/Lic%20Ciencia%20de%20Datos/Materias/Seminario/Tesina-UGR/equinos/inferencia.py), se estableció un filtro inicial que descarta directamente clips menores a 12 frames (el nuevo mínimo admisible).
- Se modificó el generador de índices para asegurar unicidad: `np.linspace(0, total - 1, min(total, 16))`. De este modo la densidad muestral es consistente y no hay duplicados.
- Se ajustaron [`equinos/particion.py`](file:///c:/Users/hrlic/Desktop/UGR/Lic%20Ciencia%20de%20Datos/Materias/Seminario/Tesina-UGR/equinos/particion.py) y [`equinos/reporte.py`](file:///c:/Users/hrlic/Desktop/UGR/Lic%20Ciencia%20de%20Datos/Materias/Seminario/Tesina-UGR/equinos/reporte.py) para que acepten como "íntegros" los clips que contengan entre 12 y 16 frames.

## 2. Caché de Características de Longitud Variable

Para habilitar el padding dinámico en PyTorch y no desperdiciar espacio en disco, los tensores extraídos ahora se guardan en `.npz` usando _Numpy Object Arrays_ en lugar del rígido stack matricial `(n, 16, 512)`.
- **Modificado**: [`equinos/caracteristicas.py`](file:///c:/Users/hrlic/Desktop/UGR/Lic%20Ciencia%20de%20Datos/Materias/Seminario/Tesina-UGR/equinos/caracteristicas.py).
- Se implementó `np.empty(..., dtype=object)` como contenedor variable, lo que permite apilar arreglos asimétricos y procesarlos luego con `pad_sequence`.

## 3. DataLoader y Normalización Geométrica 

Se agregó al core de clasificación la clase `MultimodalEquineDataset` de PyTorch.
- **Normalización**: Calcula la distancia `ojo-hocico` como unidad escalar y su punto medio como el origen `(0,0)`. Esto cumple con no pasar coordenadas absolutas en píxeles.
- **Imputación de faltantes**: Los NaNs (generalmente orejas ocultas) caen a las nuevas coordenadas `(0.0, 0.0)` del centro facial tras el normalizado, mitigando ruidos por defecto posicional.
- **Validity Mask**: Se anexa un booleano concatenado indicando con `1.0` un punto medido empíricamente o `0.0` si es un `NaN` imputado, dando al MLP visibilidad exacta de esta síntesis.
- **Padding dinámico**: Mediante un `collate_fn`, aplica dinámicamente `torch.nn.utils.rnn.pad_sequence` a todo batch variable.

## 4. Arquitectura de Atención MIL Multimodal

Se construyó el modelo `MILFusionModel` nativo de PyTorch en [`equinos/clasificacion.py`](file:///c:/Users/hrlic/Desktop/UGR/Lic%20Ciencia%20de%20Datos/Materias/Seminario/Tesina-UGR/equinos/clasificacion.py).
- **Procesamiento independiente por frame**: Antes del pooling, la `Validity Mask` y `Geometría` cruzan por un `MLP` geométrico a nivel de frame para evitar aprender la densidad temporal como confounder. Luego se concatenan al vector visual de ResNet.
- **Masked MIL**: La _Attention Mask_ calculada en base a las longitudes reales del batch se usa para silenciar el peso de atención de los pads con `-1e9`. Así, los frames falsos reciben peso $0$ estricto tras la activación de Softmax y no contaminan el agrupe _bag_ (Pooling).
- **Entrenamiento end-to-end**: Se configuró un script auto-contenido `entrenar_mil_multimodal` que instancia el DataLoader y aplica `CrossEntropyLoss` directamente sobre el tensor global del clip, equilibrando la representación de cada clase por frecuencia poblacional. Se vinculó esto exitosamente al [`03_clasificacion.ipynb`](file:///c:/Users/hrlic/Desktop/UGR/Lic%20Ciencia%20de%20Datos/Materias/Seminario/Tesina-UGR/03_clasificacion.ipynb) que ahora lee la pose y lanza el nuevo entramado.
