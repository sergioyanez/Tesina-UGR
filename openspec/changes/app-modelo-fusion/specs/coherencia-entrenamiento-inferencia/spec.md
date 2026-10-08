## ADDED Requirements

### Requirement: Las features de pose de la inferencia coinciden con las del entrenamiento

La inferencia DEBE (MUST) calcular las 160 features con `vector_del_clip` desde la pose de los 16 frames, sus índices de fotograma y el `fps`, con la misma convención que el entrenamiento (`P` de forma `(16, 39, 3)`, orden = posición en la secuencia). Para un mismo clip, el vector calculado por la ruta de inferencia DEBE (MUST) ser igual al del entrenamiento y las probabilidades fusionadas DEBEN (MUST) coincidir.

#### Scenario: clip del dataset
- **WHEN** se alimenta la ruta de inferencia con la pose y los fotogramas persistidos de un clip del dataset
- **THEN** el vector de 160 features es igual (`allclose`, NaN en las mismas posiciones) al del entrenamiento y la clase predicha es la misma

#### Scenario: pose con keypoints ausentes
- **WHEN** faltan keypoints de un frame
- **THEN** los valores quedan en NaN como en el entrenamiento y el modelo los imputa por la mediana
