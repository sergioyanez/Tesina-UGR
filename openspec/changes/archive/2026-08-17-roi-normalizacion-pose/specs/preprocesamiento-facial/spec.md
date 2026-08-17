# Especificación: preprocesamiento-facial

> Términos normativos según RFC 2119: **DEBE** ≡ MUST, **NO DEBE** ≡ MUST NOT.

## ADDED Requirements

### Requirement: Detección del equino en cada frame

El sistema DEBE (MUST) localizar al equino en cada frame de cada clip usando un detector preentrenado,
sin requerir anotación manual. Cuando haya más de un equino detectado, el sistema DEBE quedarse con
la detección de mayor confianza.

#### Scenario: Frame con un equino claramente visible

- **WHEN** se procesa un frame donde el detector encuentra un equino con confianza mayor al umbral
- **THEN** el sistema registra la caja delimitadora en coordenadas de píxel de la imagen original
- **AND** registra la confianza de la detección

#### Scenario: Frame con varios equinos

- **WHEN** el detector devuelve más de una caja de clase equino sobre el umbral
- **THEN** el sistema conserva únicamente la de mayor confianza
- **AND** registra la cantidad total de detecciones para poder auditar el caso después

#### Scenario: Frame sin detección

- **WHEN** el detector no encuentra ningún equino sobre el umbral de confianza
- **THEN** el sistema marca el frame como sin detección en lugar de inventar una caja
- **AND** el frame queda contabilizado en el reporte de cobertura

### Requirement: Suavizado temporal de las detecciones

El sistema DEBE suavizar las cajas detectadas a lo largo de cada clip mediante una mediana móvil,
de modo que una detección aislada mal ubicada no produzca un salto en la ROI. El suavizado NO DEBE (MUST NOT)
aplanar el movimiento sostenido del animal o de la cámara.

#### Scenario: Detección aislada errónea

- **WHEN** un clip tiene una secuencia de cajas estables con una sola caja muy desplazada en el medio
- **THEN** la caja suavizada de esa posición coincide con la de sus vecinas

#### Scenario: Movimiento sostenido del animal

- **WHEN** las cajas del clip se desplazan de forma progresiva y consistente entre frames
- **THEN** el suavizado conserva ese desplazamiento sin aplanarlo

#### Scenario: Huecos sin detección dentro del clip

- **WHEN** algunos frames del clip no tienen detección pero otros sí
- **THEN** el sistema rellena los huecos con la detección válida más cercana
- **AND** registra cuántos frames debieron rellenarse

### Requirement: Estimación de keypoints faciales

El sistema DEBE estimar keypoints faciales del equino —como mínimo un punto ocular y un punto en el
hocico— mediante un modelo preentrenado para cuadrúpedos, sin anotación manual. El sistema NO DEBE (MUST NOT)
asumir nombres fijos de keypoints: DEBE descubrir el esquema que expone el modelo en tiempo de
ejecución y seleccionar los faciales a partir de él.

#### Scenario: Keypoints faciales disponibles

- **WHEN** el modelo devuelve el punto ocular y el del hocico con confianza suficiente
- **THEN** el sistema los registra en coordenadas de la imagen original
- **AND** los usa como referencia para la normalización de pose

#### Scenario: Esquema de keypoints desconocido

- **WHEN** se carga el modelo de pose por primera vez
- **THEN** el sistema imprime el listado completo de keypoints disponibles
- **AND** deja explícito cuáles seleccionó como ocular y como hocico

#### Scenario: Keypoints faciales ausentes o de baja confianza

- **WHEN** no se obtiene el par ocular/hocico con confianza suficiente en un frame
- **THEN** el sistema no normaliza la pose de ese frame
- **AND** recurre a la ROI derivada de la caja de detección, dejándolo marcado en el CSV

### Requirement: Normalización de pose por transformada de similitud

El sistema DEBE (MUST) alinear cada recorte facial mediante una transformada de similitud que lleve el
punto ocular y el del hocico a posiciones canónicas fijas. La transformación DEBE preservar la
relación de aspecto de la cara, es decir, escalar de forma uniforme en ambos ejes.

#### Scenario: Cabezas en poses distintas

- **WHEN** se alinean dos frames cuya cabeza aparece con rotación, escala y posición muy distintas
- **THEN** en ambos recortes normalizados el ojo y el hocico caen en la misma posición

#### Scenario: Escalado uniforme

- **WHEN** se aplica la transformada de similitud a un recorte
- **THEN** la escala aplicada es idéntica en el eje horizontal y en el vertical, sin deformar la cara

#### Scenario: Puntos de referencia degenerados

- **WHEN** el punto ocular y el del hocico coinciden, de modo que no definen un eje
- **THEN** el sistema no produce una transformación inválida y trata el frame como no normalizable

### Requirement: Región de interés cuadrada

El sistema DEBE (MUST) producir recortes cuadrados, para que el reescalado posterior a la entrada de la red
no deforme la cara. La ROI DEBE cubrir todos los keypoints faciales disponibles más un margen
configurable, y DEBE recortarse a los bordes de la imagen cuando se salga de ellos.

#### Scenario: ROI que excede el borde de la imagen

- **WHEN** la ROI calculada con su margen se sale del área de la imagen original
- **THEN** el sistema la recorta a los límites de la imagen sin producir coordenadas negativas ni
  mayores a la dimensión correspondiente

#### Scenario: Recorte final

- **WHEN** se guarda un frame procesado
- **THEN** la imagen resultante es cuadrada y de dimensiones uniformes en todo el dataset

### Requirement: Trazabilidad hasta el nivel de caballo

El sistema DEBE (MUST) arrastrar `id_caballo`, `video_origen` y `clip` en cada registro de frame alineado,
de manera que la partición por ejemplar siga siendo posible aguas abajo.

#### Scenario: Generación del CSV de frames alineados

- **WHEN** se completa el procesamiento de todos los clips
- **THEN** el CSV resultante contiene, por cada frame, su clase, `id_caballo`, `video_origen`,
  `clip`, la ruta del frame alineado y los indicadores de calidad de la detección

### Requirement: Reporte de cobertura de detección

El sistema DEBE (MUST) reportar qué proporción de frames obtuvo detección del equino y keypoints faciales
válidos, desglosado por clase y por caballo. Este reporte es el criterio que decide si el pipeline
automático es utilizable o si hay que recurrir a anotación manual.

#### Scenario: Cobertura suficiente

- **WHEN** finaliza el procesamiento
- **THEN** el sistema informa el porcentaje de frames con detección y con keypoints, global y
  desglosado por clase y por caballo

#### Scenario: Cobertura insuficiente en una clase o caballo

- **WHEN** alguna clase o algún caballo queda por debajo del umbral de cobertura aceptable
- **THEN** el sistema lo señala explícitamente como advertencia
- **AND** no se continúa al entrenamiento sin decidir antes qué hacer con esos casos
