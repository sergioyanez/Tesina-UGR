# coherencia-entrenamiento-inferencia Specification

## Purpose
TBD - created by archiving change base-comun-comparacion-de-modelos. Update Purpose after archive.
## Requirements
### Requirement: La inferencia aplica la orientación del video

La inferencia DEBE (MUST) leer el video con `CAP_PROP_ORIENTATION_AUTO = 1`, como la segmentación del notebook 01, y NO DEBE (MUST NOT) entregar a la pose ni al modelo frames sin rotar de un video con rotación declarada.

#### Scenario: video de celular con rotación de 90 grados
- **WHEN** se clasifica un video cuyo contenedor declara rotación 90
- **THEN** el video que recibe DeepLabCut es derecho (alto mayor que ancho en un video vertical) y ya no declara rotación

### Requirement: La inferencia recodifica como los clips del dataset

La inferencia DEBE (MUST) recodificar el video con `mp4v` antes de estimar la pose y de recortar, y DEBE (MUST) leer los frames de esa copia. Una opción `normalizar=False` DEBE (MUST) conservar la lectura anterior sólo para medición.

#### Scenario: comparación antes y después
- **WHEN** se llama `preparar_clip(..., normalizar=False)`
- **THEN** no se recodifica y se reproduce el comportamiento previo

### Requirement: Diferencias residuales declaradas

Toda diferencia residual entre la entrada de entrenamiento y la de inferencia DEBE (MUST) documentarse con su causa; la ventana elegida sobre un video completo y la pose sobre el video entero en lugar de sobre el clip NO DEBEN (MUST NOT) presentarse como eliminadas.

#### Scenario: reporte de la prueba extremo a extremo
- **WHEN** se reporta la coincidencia sobre los 16 videos del dataset
- **THEN** se informa el antes, el después y la causa de cada discrepancia restante

### Requirement: Las features de pose de la inferencia coinciden con las del entrenamiento

La inferencia DEBE (MUST) calcular las 160 features con `vector_del_clip` desde la pose de los 16 frames, sus índices de fotograma y el `fps`, con la misma convención que el entrenamiento (`P` de forma `(16, 39, 3)`, orden = posición en la secuencia). Para un mismo clip, el vector calculado por la ruta de inferencia DEBE (MUST) ser igual al del entrenamiento y las probabilidades fusionadas DEBEN (MUST) coincidir.

#### Scenario: clip del dataset
- **WHEN** se alimenta la ruta de inferencia con la pose y los fotogramas persistidos de un clip del dataset
- **THEN** el vector de 160 features es igual (`allclose`, NaN en las mismas posiciones) al del entrenamiento y la clase predicha es la misma

#### Scenario: pose con keypoints ausentes
- **WHEN** faltan keypoints de un frame
- **THEN** los valores quedan en NaN como en el entrenamiento y el modelo los imputa por la mediana

