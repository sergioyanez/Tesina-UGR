# inferencia-y-interfaz — delta

> Equivalencia RFC 2119: **DEBE (MUST)**, **NO DEBE (MUST NOT)**, **DEBERÍA (SHOULD)**.

## ADDED Requirements

### Requirement: Modelo de producción entrenado sobre el conjunto completo

El sistema DEBE (MUST) desplegar un modelo entrenado sobre **todos** los clips íntegros, y NO DEBE
(MUST NOT) desplegar uno de los modelos de validación cruzada: cada uno vio sólo una fracción del
material y elegir cuál sería arbitrario.

La métrica que se informe del modelo de producción DEBE (MUST) ser la obtenida por validación
cruzada. NO DEBE (MUST NOT) medirse sobre los datos con los que se entrenó, ni presentarse como
mejor por haber entrenado con más material: entrenar con todo mejora el modelo pero no hay forma
honesta de medir cuánto sin dejar material afuera.

El modelo persistido DEBE registrar su procedencia: sobre qué conjunto se entrenó, con qué
características y de qué corrida del preprocesamiento provienen.

#### Scenario: Despliegue del modelo

- **WHEN** se prepara el modelo que va a usar la interfaz
- **THEN** se entrena sobre todos los clips íntegros disponibles
- **AND** se persiste junto con la procedencia del conjunto que lo entrenó

#### Scenario: Métrica informada

- **WHEN** la interfaz informa qué tan confiable es el modelo
- **THEN** informa la métrica de validación cruzada
- **AND** no informa ninguna métrica calculada sobre el conjunto de entrenamiento

### Requirement: Negarse cuando el preprocesamiento no alcanza

El sistema DEBE (MUST) producir una **negativa explicada** cuando el preprocesamiento no logre un
clip íntegro —16 frames alineados— a partir del video recibido. NO DEBE (MUST NOT) predecir sobre
un clip parcialmente alineado.

Un clip a medio alinear mezcla recortes por keypoints con recortes de caja dentro de la misma
muestra. A qué frame le toca cuál correlaciona con el ejemplar, que predice la clase: es la muestra
contaminada que la capacidad de clasificación prohíbe, y una etiqueta emitida ahí tiene una
confiabilidad que nadie midió.

La negativa DEBE indicar el motivo —sin detección del animal, sin keypoints faciales suficientes,
video demasiado corto— y DEBE tratarse como una salida normal del sistema y no como un error.

Sobre el dataset propio sólo **141 de 368 clips (38 %)** resultaron íntegros, de modo que la
negativa es el resultado **más probable** para un video cualquiera. El sistema DEBE estar diseñado
para ese caso, no para la excepción.

Los topes del relleno temporal NO DEBEN (MUST NOT) relajarse para que entren más videos: subirlos
sería inventar posiciones que nadie midió, justo en el momento en que el resultado se le muestra a
una persona.

#### Scenario: Video del que no se obtiene un clip íntegro

- **WHEN** el preprocesamiento produce menos de 16 frames alineados
- **THEN** el sistema informa que no puede predecir y por qué
- **AND** no emite ninguna clase ni ninguna probabilidad

#### Scenario: Video sin detección del animal

- **WHEN** el detector no encuentra un equino en el video
- **THEN** la negativa indica esa causa concreta, y no un mensaje genérico

#### Scenario: La negativa no es un error

- **WHEN** el sistema se niega a predecir
- **THEN** lo comunica como un resultado esperable del preprocesamiento
- **AND** no lo presenta como una falla de la interfaz ni del video del usuario

### Requirement: La confiabilidad medida acompaña a toda predicción

Cada predicción DEBE (MUST) mostrarse junto a la **precisión medida de la clase predicha** —de
cuántas veces que el modelo dijo esa clase, cuántas acertó—, obtenida de las predicciones fuera de
muestra de la validación cruzada.

La precisión, y no el recall, es lo que DEBE mostrarse al usuario: contesta la pregunta que se hace
quien recibe la respuesta. Son números distintos y confundirlos engaña.

La interfaz DEBE (MUST) informar que el modelo **no alcanzó el criterio declarado** para
considerarse útil, y DEBE mostrar las probabilidades de las tres clases y no sólo la ganadora.

Esta información NO DEBE (MUST NOT) ubicarse en un descargo al pie ni en una ayuda secundaria: DEBE
estar en la salida principal, con el mismo peso que la etiqueta. Una etiqueta sola es la lectura
más engañosa posible de este modelo.

El sistema NO DEBE (MUST NOT) presentarse como herramienta de diagnóstico veterinario, y DEBE
decirlo de forma explícita.

#### Scenario: Predicción de la clase de menor precisión

- **WHEN** el modelo predice la clase cuya precisión medida es la más baja
- **THEN** la interfaz muestra esa precisión junto a la etiqueta
- **AND** la muestra con la misma prominencia que la clase predicha

#### Scenario: Probabilidades de las tres clases

- **WHEN** se emite una predicción
- **THEN** se muestran las probabilidades de las tres clases
- **AND** no se muestra únicamente la de la clase ganadora

#### Scenario: Alcance declarado

- **WHEN** una persona usa la interfaz por primera vez
- **THEN** encuentra dicho que el resultado no es un diagnóstico veterinario
- **AND** encuentra dicho que el modelo no alcanzó su criterio de aceptación

### Requirement: La inferencia usa el mismo código que produjo el conjunto evaluado

El sistema DEBE (MUST) reutilizar las etapas de preprocesamiento y extracción de características
tal como se ejecutaron para producir el conjunto sobre el que se midió el modelo. NO DEBE (MUST
NOT) reimplementarlas.

Una reimplementación equivalente que difiera en el encuadre, en un umbral de confianza o en el
orden de los frames produciría entradas de otra distribución, y la confiabilidad medida dejaría de
aplicar **sin que nada falle**. Es la misma clase de error silencioso que el proyecto ya encontró
con el espejo de videos a medio copiar.

El sistema DEBE informar el costo de la inferencia antes de empezar y DEBE dar señales de progreso
por etapa: son ~35 s por clip en GPU, y una interfaz que parece colgada es peor que una lenta que
avisa.

#### Scenario: Coherencia con el conjunto evaluado

- **WHEN** se procesa un video nuevo
- **THEN** se usan las mismas funciones de detección, pose, alineación y extracción que produjeron
  el dataset evaluado
- **AND** el encuadre y los umbrales son los mismos que están declarados en la etapa 02

#### Scenario: Inferencia larga

- **WHEN** el usuario envía un video
- **THEN** la interfaz informa de antemano cuánto puede tardar
- **AND** muestra en qué etapa del preprocesamiento está mientras procesa
