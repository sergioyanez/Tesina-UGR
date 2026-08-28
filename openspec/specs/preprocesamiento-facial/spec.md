# preprocesamiento-facial Specification

## Purpose
TBD - created by archiving change roi-normalizacion-pose. Update Purpose after archive.
## Requirements
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

### Requirement: Relleno temporal de los keypoints faltantes

Cuando en un clip falte la detección de los keypoints de un frame y existan frames vecinos con
detección válida, el sistema DEBE (MUST) estimar la posición faltante interpolando temporalmente
entre esos vecinos, y construir la transformada de similitud con los puntos así obtenidos. La
estimación NO DEBE (MUST NOT) construirse por un camino distinto del que usa un frame con
detección propia: se interpolan los puntos, no la transformada.

El relleno DEBE estar acotado por un tope declarado de frames consecutivos sin detección. Superado
ese tope, el frame DEBE seguir cayendo al recorte de caja sin alinear. El tope DEBE ser un
parámetro auditable de la etapa y su valor DEBE surgir de la validación, no del criterio de quien
implementa.

Este requisito existe porque la unidad que consume el modelo es el clip y no el frame. Con
cobertura del 69,9 % por frame, sólo 80 de 342 clips tenían sus 16 frames alineados, y la clase
minoritaria quedaba en 5 clips de 4 ejemplares: un dataset que no admite partición por caballo. El
80 % de los huecos son de uno o dos frames rodeados de detecciones válidas, y la cabeza de un
caballo no se desplaza de forma arbitraria en ese lapso.

Los frames al principio o al final del clip, que no tienen vecinos detectados de ambos lados, DEBEN
tratarse por un mecanismo distinto y con su propio tope, y NO DEBEN contarse como interpolados.

#### Scenario: Hueco interior dentro del tope

- **WHEN** un frame no tiene detección de keypoints
- **AND** existen frames con detección antes y después, separados por no más que el tope declarado
- **THEN** el sistema estima el ojo y el hocico interpolando entre esos vecinos
- **AND** construye el recorte con la misma transformada de similitud que usaría con una detección
  propia

#### Scenario: Hueco interior más largo que el tope

- **WHEN** la cantidad de frames consecutivos sin detección supera el tope declarado
- **THEN** el sistema no interpola esos frames
- **AND** esos frames caen al recorte de caja sin alinear, como antes

#### Scenario: Clip sin ninguna detección

- **WHEN** ningún frame del clip tiene detección válida de keypoints
- **THEN** el sistema no estima ninguna posición
- **AND** el clip entero queda con recortes de caja sin alinear

### Requirement: Trazabilidad del origen de la alineación

Cada frame DEBE (MUST) registrar en los artefactos **cómo** quedó alineado: por detección propia,
por interpolación entre vecinos, por sostenimiento en un extremo del clip, o por recorte de caja
sin alinear. Una marca binaria de "alineado" NO DEBE ser el único dato disponible.

Los reportes de cobertura DEBEN poder desglosarse por ese origen, y NO DEBEN presentar como
equivalentes un frame detectado y uno estimado.

Sin esa distinción, los frames estimados se vuelven invisibles en cuanto se calcula la primera
métrica, no se puede contestar cuántas posiciones fueron observadas y cuántas inferidas, y no se
puede entrenar con y sin ellos para medir su efecto.

#### Scenario: Reporte de cobertura sobre un dataset con frames rescatados

- **WHEN** se informa la cobertura de alineación de una corrida que usó relleno temporal
- **THEN** el reporte distingue los frames alineados por detección de los estimados
- **AND** permite recalcular la cobertura considerando únicamente los detectados

#### Scenario: Ablación de los frames estimados

- **WHEN** se quiere entrenar excluyendo los frames cuya posición fue estimada
- **THEN** el dataset permite filtrarlos por el origen registrado, sin recomputar nada

### Requirement: Validación del relleno contra detecciones conocidas

El relleno temporal DEBE (MUST) validarse ocultando frames que **sí** tienen detección válida,
estimando su posición con el mismo procedimiento y midiendo el error contra la posición realmente
detectada. La validación NO DEBE (MUST NOT) medir la estimación contra los mismos datos con que se
construyó.

El error DEBE informarse en píxeles dentro del recorte final, que es el desplazamiento que ve el
modelo, y DEBE acompañarse de su distribución —al menos mediana, percentil 90 y máximo—, no
solamente de un promedio.

El umbral de aceptación DEBE declararse antes de medir, y el resultado DEBE informarse aunque no lo
cumpla.

El umbral DEBE anclarse en una **línea de base medida**: el desplazamiento del recorte entre dos
detecciones propias consecutivas. Ese valor es el piso físico del problema —lo que se mueve la
cabeza del caballo entre dos frames muestreados— y ninguna estimación puede ser sistemáticamente
más precisa que él sin estar sobreajustando. Un umbral fijado sin esa referencia no distingue "el
relleno no sirve" de "el criterio era imposible de cumplir".

**Umbral vigente.** Un largo de hueco se acepta si, medido en píxeles dentro del recorte de 224, su
**mediana** y su **percentil 90** no superan los de la línea de base. Se informa además el máximo,
que no forma parte del criterio pero sí del informe, porque lo que arruina una muestra es la cola.

> **Historia de este umbral, que hay que poder contar.** El 28/08/2026 se declaró, antes de medir,
> un umbral de mediana ≤ 3,0 px y p90 ≤ 8,0 px. Ningún largo de hueco lo cumplió: el mejor caso
> dio 13,99 px. Al medir después la línea de base —16,28 px de mediana, 78,22 de p90— quedó claro
> que el umbral exigía que la estimación fuera cinco veces más exacta que el propio movimiento del
> caballo entre frames, y que la interpolación de un hueco de 1 es de hecho **más precisa que usar
> la detección real del frame vecino**. El umbral se re-declaró el 28/08/2026 anclándolo en la
> línea de base. El resultado contra el umbral original queda registrado en `validacion.md`, y se
> informa junto con el nuevo: cambiar un criterio después de ver los números sólo es legítimo si
> queda escrito que se cambió, por qué, y qué daba antes.

La validación DEBE estratificarse por la cantidad de frames consecutivos que se está estimando, de
modo que el tope de relleno surja de la curva de error medida.

Este requisito existe porque una verificación que no puede fallar ya produjo una conclusión
equivocada en este proyecto: el error de alineación de 1e-13 px que se informaba comparaba la
transformada contra los mismos puntos con que se la había construido, y daba cero aunque el recorte
fuera inutilizable.

#### Scenario: Validación por omisión de una detección conocida

- **WHEN** se valida el relleno sobre un frame con detección válida
- **THEN** el sistema oculta esa detección, estima la posición desde los vecinos y la compara
  contra la real
- **AND** informa el error en píxeles del recorte final, con su distribución

#### Scenario: El error supera el umbral declarado

- **WHEN** la distribución de error medida no cumple el umbral declarado para un largo de hueco
- **THEN** ese largo de hueco queda por encima del tope de relleno
- **AND** los frames correspondientes siguen cayendo al recorte de caja

#### Scenario: Sesgo conocido de la validación

- **WHEN** se informa el resultado de la validación
- **THEN** se hace constar que los frames usados para validar son los que sí se detectaron, que no
  son una muestra aleatoria de los que fallaron

