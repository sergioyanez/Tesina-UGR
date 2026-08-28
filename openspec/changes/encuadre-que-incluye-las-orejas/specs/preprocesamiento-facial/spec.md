# preprocesamiento-facial — delta

> Equivalencia RFC 2119: **DEBE (MUST)**, **NO DEBE (MUST NOT)**, **DEBERÍA (SHOULD)**.

## ADDED Requirements

### Requirement: Región de interés que incluye las orejas

El recorte facial canónico DEBE (MUST) contener las orejas del equino, no sólo el eje ojo-hocico.
La posición de las orejas es un rasgo informativo de la expresión —en la etología equina, el
indicador principal de disgusto y estrés—, y un recorte que las deja fuera impide que el
clasificador del objetivo 3 pueda usarlas, sin producir ningún error observable.

El encuadre DEBE (MUST) declararse como un **campo de visión en unidades de la distancia
ojo-hocico `d`**, con tres parámetros independientes: el aire por encima del ojo, el aire por
debajo del hocico y el semiancho lateral. NO DEBE (MUST NOT) declararse como dos fracciones sobre
un lado único, porque esa parametrización obliga a que el campo vertical y el lateral crezcan
juntos y oculta que el déficit del encuadre es asimétrico.

El recorte resultante NO DEBE (MUST NOT) exigirse cuadrado. Su relación de aspecto DEBE ser la que
resulte del campo de visión declarado, y las dimensiones en píxeles del destino DEBEN respetarla.

El par de anclaje DEBE seguir siendo el ojo y el hocico. Los keypoints de oreja NO DEBEN (MUST NOT)
usarse como puntos de alineación: su disponibilidad medida es de ~60 % por oreja contra la del par
facial, y anclarse en ellos cambiaría la cobertura de la etapa a cambio de nada — las orejas entran
por campo de visión, no por anclaje.

#### Scenario: Encuadre declarado por campo de visión

- **WHEN** se configura la etapa de normalización
- **THEN** el encuadre queda expresado como aire sobre el ojo, aire bajo el hocico y semiancho
  lateral, todos en unidades de `d`
- **AND** las dimensiones del recorte de destino se derivan de esa relación de aspecto

#### Scenario: Recorte no cuadrado

- **WHEN** el campo de visión declarado es más alto que ancho
- **THEN** el sistema produce recortes rectangulares de dimensiones uniformes en todo el dataset
- **AND** no reescala los ejes de forma independiente para forzarlos a un cuadrado

#### Scenario: El par de anclaje no cambia

- **WHEN** se alinea un frame bajo el nuevo encuadre
- **THEN** los puntos que definen la transformada siguen siendo el ojo y el hocico
- **AND** las posiciones anotadas manualmente bajo el encuadre anterior siguen siendo válidas sin
  volver a anotarse

### Requirement: Inclusión de orejas como criterio de aceptación del encuadre

La etapa DEBE (MUST) medir e informar qué proporción de los keypoints de oreja cae dentro del
recorte, y esa proporción DEBE ser un criterio de aceptación del encuadre, al mismo nivel que la
cobertura de alineación. Una cobertura de alineación alta con las orejas fuera del cuadro NO DEBE
(MUST NOT) considerarse un preprocesamiento aceptable.

La medición DEBE hacerse sobre los keypoints de oreja ya persistidos en la pose completa,
proyectándolos por la misma transformada de similitud que produce el recorte. NO DEBE requerir
volver a inferir la pose: eso es lo que permite comparar varios encuadres antes de escribir una
sola imagen.

**Criterio declarado, antes de medir.** Un encuadre candidato es aceptable si cumple las dos
condiciones:

1. **Contiene las orejas de al menos la mitad de los frames.** Anclado en la distribución medida:
   la punta de oreja está a 0,81 `d` por encima del ojo en la mediana, de modo que un campo que
   contenga la mediana contiene la mitad de los casos. Operativamente, inclusión de keypoints de
   oreja ≥ 50 %.
2. **No empeora el relleno respecto del encuadre vigente.** La línea de base **no es cero**: el
   encuadre actual ya produce 22,2 % de relleno mediano. El candidato no debe superarlo.

Ambas condiciones se anclan en cantidades medidas sobre la corrida vigente y no en números
redondos. El resultado DEBE informarse aunque ningún candidato las cumpla, y si el criterio se
re-declara, DEBE quedar escrito que se cambió, por qué, y qué daba antes.

Al informar la inclusión de orejas DEBE hacerse constar su **sesgo de denominador**: sólo se mide
sobre frames donde la oreja fue detectada con confianza suficiente, y una oreja fuera del cuadro
tiende a no detectarse, así que la métrica favorece a todos los encuadres por igual y sobrestima a
todos.

#### Scenario: Barrido de encuadres sin reprocesar imágenes

- **WHEN** se comparan varios campos de visión candidatos
- **THEN** el sistema informa para cada uno la inclusión de keypoints de oreja y la distribución
  del relleno
- **AND** no vuelve a inferir la pose ni escribe imágenes para producir esa comparación

#### Scenario: Encuadre que mejora las orejas empeorando el relleno

- **WHEN** un candidato supera el 50 % de inclusión de orejas pero su relleno mediano excede el de
  la línea de base
- **THEN** el sistema informa que no cumple el criterio declarado
- **AND** informa igualmente ambas métricas, para que la compensación quede a la vista

#### Scenario: Informe de la métrica de orejas

- **WHEN** se informa la inclusión de orejas de una corrida
- **THEN** se informa junto al denominador sobre el que se calculó
- **AND** se hace constar que los frames sin oreja detectada no son una muestra aleatoria

### Requirement: Trazabilidad del relleno del recorte

Cada frame DEBE (MUST) registrar qué fracción de su recorte se completó con píxeles inventados por
replicación de borde. Una marca binaria de "tuvo relleno" NO DEBE ser el único dato disponible.

Este requisito existe porque la cantidad de relleno depende de a qué distancia se filmó al animal,
que correlaciona con el ejemplar, que predice la clase. Es la misma clase de atajo que el encuadre
canónico existe para evitar, y ampliar el campo de visión lo aumenta. Sin la fracción registrada
por frame no se puede contestar cuánto del resultado del clasificador se apoya en ese artefacto.

Los reportes de cobertura DEBEN poder desglosarse por esa fracción, y el dataset DEBE permitir
filtrar por ella sin recomputar nada, del mismo modo que se filtra por `origen_alineacion`.

#### Scenario: Registro de la fracción de relleno

- **WHEN** se escribe un frame alineado cuyo rectángulo canónico se sale del cuadro de origen
- **THEN** el registro del frame incluye la fracción de su área completada por replicación

#### Scenario: Ablación por relleno

- **WHEN** se quiere entrenar excluyendo los frames con relleno por encima de un valor
- **THEN** el dataset permite filtrarlos por la fracción registrada, sin volver a procesar imágenes

## MODIFIED Requirements

### Requirement: Normalización de pose por transformada de similitud

El sistema DEBE (MUST) alinear cada recorte facial mediante una transformada de similitud que lleve el
punto ocular y el del hocico a posiciones canónicas fijas. La transformación DEBE preservar la
relación de aspecto de la cara, es decir, escalar de forma uniforme en ambos ejes.

La preservación de la relación de aspecto es una propiedad de la **transformada**, no de la forma
del recorte de destino. Un destino rectangular NO deforma la cara mientras la escala siga siendo
isótropa: la relación de aspecto del destino determina el **campo de visión**, no el estiramiento.
Confundir ambas cosas es lo que ató históricamente el encuadre a un cuadrado.

#### Scenario: Cabezas en poses distintas

- **WHEN** se alinean dos frames cuya cabeza aparece con rotación, escala y posición muy distintas
- **THEN** en ambos recortes normalizados el ojo y el hocico caen en la misma posición

#### Scenario: Escalado uniforme

- **WHEN** se aplica la transformada de similitud a un recorte
- **THEN** la escala aplicada es idéntica en el eje horizontal y en el vertical, sin deformar la cara

#### Scenario: Destino rectangular sin deformación

- **WHEN** el recorte de destino es más alto que ancho
- **THEN** la escala aplicada sigue siendo idéntica en ambos ejes
- **AND** la cara no queda estirada ni comprimida respecto del frame de origen

#### Scenario: Puntos de referencia degenerados

- **WHEN** el punto ocular y el del hocico coinciden, de modo que no definen un eje
- **THEN** el sistema no produce una transformación inválida y trata el frame como no normalizable

### Requirement: Reporte de cobertura de detección

El sistema DEBE (MUST) reportar qué proporción de frames obtuvo detección del equino y keypoints faciales
válidos, desglosado por clase y por caballo. Este reporte es el criterio que decide si el pipeline
automático es utilizable o si hay que recurrir a anotación manual.

El reporte DEBE incluir además la **inclusión de keypoints de oreja** y la **distribución del
relleno del recorte**. La cobertura de alineación sola NO DEBE (MUST NOT) presentarse como criterio
suficiente de que el preprocesamiento es utilizable: mide cuántos frames se pudieron alinear, no
qué parte de la cara quedó dentro del recorte, y un encuadre puede tener cobertura alta dejando
fuera el rasgo más informativo de la clase de interés.

#### Scenario: Cobertura suficiente

- **WHEN** finaliza el procesamiento
- **THEN** el sistema informa el porcentaje de frames con detección y con keypoints, global y
  desglosado por clase y por caballo
- **AND** informa la inclusión de keypoints de oreja y la distribución del relleno

#### Scenario: Cobertura insuficiente en una clase o caballo

- **WHEN** alguna clase o algún caballo queda por debajo del umbral de cobertura aceptable
- **THEN** el sistema lo señala explícitamente como advertencia
- **AND** no se continúa al entrenamiento sin decidir antes qué hacer con esos casos

#### Scenario: Cobertura alta con las orejas fuera del recorte

- **WHEN** la cobertura de alineación supera su umbral pero la inclusión de orejas queda por debajo
  del criterio declarado
- **THEN** el sistema señala que el preprocesamiento no cumple el criterio de encuadre
- **AND** no se considera cerrada la etapa por la sola cobertura de alineación

## REMOVED Requirements

### Requirement: Región de interés cuadrada

**Reason**: El recorte cuadrado es incompatible con incluir las orejas. Medido sobre los 5.586
frames con par ojo-hocico detectado de la corrida del 28/08/2026, el encuadre cuadrado vigente deja
fuera el 88 % de los keypoints de oreja y el 97 % de las puntas. La causa es que el déficit del
encuadre es vertical —la punta de oreja está a 0,81 `d` por encima del ojo en la mediana, y el
recorte muestra 0,33 `d`— mientras que lateralmente sobra campo. Con un recorte cuadrado, ganar
aire vertical obliga a ampliar también el lateral, lo que lleva el cuadrado canónico a 1.393 px de
lado sobre cuadros de 1.080 de ancho: exactamente el encuadre que se descartó el 19/08/2026 por
producir relleno masivo.

La justificación original del requisito —evitar que el reescalado deforme la cara— se preserva por
otra vía: la deformación la impide la **isotropía de la transformada de similitud**, no la forma
del destino. Ver el requisito «Normalización de pose por transformada de similitud», modificado
para dejarlo explícito.

**Migration**: El encuadre pasa a declararse por campo de visión según el requisito «Región de
interés que incluye las orejas». Los artefactos de imagen de la corrida vigente se regeneran; los
CSV de pose, los keypoints faciales y las anotaciones manuales no se ven afectados, porque el par
de anclaje no cambia. Los consumidores aguas abajo que asuman entradas de 224×224 —en particular el
change `clasificacion-orientada-a-video`— deben actualizarse a la forma resultante del barrido.
