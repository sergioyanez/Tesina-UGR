# clasificacion-de-expresiones (delta)

> Convención de lenguaje normativo: el proyecto se escribe en español, pero conserva los términos
> RFC 2119 entre paréntesis en la primera cláusula normativa de cada requisito. `DEBE (MUST)`
> equivale a MUST y `NO DEBE (MUST NOT)` a MUST NOT.

## ADDED Requirements

### Requirement: La unidad de clasificación es el clip íntegro

El sistema DEBE (MUST) clasificar clips completos —los 16 frames alineados— y NO DEBE (MUST NOT)
clasificar frames sueltos como muestras independientes.

Un clip cuyos frames no estén todos alineados NO DEBE entrar al conjunto de entrenamiento ni al
de evaluación. Mezclar frames alineados por keypoints con recortes de caja dentro de una misma
muestra mete el confounder de recorte adentro del ejemplo: a qué frame le toca cuál está
correlacionado con el ejemplar, que a su vez predice la clase.

Este requisito existe porque la hipótesis del trabajo afirma que la expresión es un evento
dinámico y que su evolución en pocos segundos es lo que permite clasificarla. Clasificar frames
sueltos contestaría otra pregunta.

#### Scenario: Clip con todos sus frames alineados

- **WHEN** se arma el conjunto de datos para entrenar
- **THEN** el clip entra como una única muestra de 16 frames
- **AND** su etiqueta es la clase de la carpeta de origen del video

#### Scenario: Clip con frames sin alinear

- **WHEN** al menos uno de los 16 frames del clip no quedó alineado
- **THEN** el clip queda fuera del conjunto, tanto de entrenamiento como de evaluación
- **AND** la cantidad de clips excluidos se informa junto con los resultados

### Requirement: Partición por ejemplar

Toda partición de datos DEBE (MUST) agrupar por `id_caballo`, de modo que ningún ejemplar
aparezca a la vez en entrenamiento y en evaluación. NO DEBE (MUST NOT) particionarse por clip ni
por frame.

La cantidad de particiones DEBE ser 3 o 4. Con 5, algún fold queda con 2 clips de la clase
minoritaria en test, y una métrica calculada sobre dos muestras no es informativa.

Los resultados DEBEN reportarse como media **y desvío entre folds**. Una media sola oculta que
con 15 ejemplares en la clase minoritaria el resultado depende de qué caballos tocaron en cada
partición.

Este requisito existe porque el ejemplar predice la clase en este dataset: 26 caballos aparecen
en más de una clase pero muchos aparecen en una sola, y varios aportan más de un clip. Un modelo
que reconozca al individuo obtendría una métrica alta sin haber aprendido nada sobre expresiones.

#### Scenario: Un ejemplar con varios clips

- **WHEN** un caballo aporta más de un clip íntegro
- **THEN** todos sus clips caen del mismo lado de la partición

#### Scenario: Un ejemplar presente en más de una clase

- **WHEN** un caballo aparece etiquetado en dos clases distintas
- **THEN** la partición lo trata como un único grupo, no como un grupo por clase
- **AND** todos sus clips, de cualquier clase, caen del mismo lado

### Requirement: Líneas de base declaradas antes de entrenar

El sistema DEBE (MUST) evaluar, sobre exactamente las mismas particiones que el modelo, al menos
tres líneas de base:

1. **clase mayoritaria** — el piso absoluto;
2. **un solo frame del clip**, sin información temporal — mide si la dimensión temporal aporta;
3. **sólo la identidad del ejemplar**, sin mirar la imagen — mide cuánta señal hay en el
   confounder.

Las tres DEBEN estar implementadas y medidas **antes** de reportar cualquier resultado del
modelo. El resultado del modelo NO DEBE presentarse sin ellas al lado.

La tercera es la que decide si el problema es abordable con este material: si predice bien, la
clase está determinada por el individuo y ningún modelo que mire la imagen puede distinguir
expresión de ejemplar.

La segunda pone a prueba la hipótesis del trabajo. Si el modelo secuencial no le gana, esa
hipótesis no queda sostenida por la evidencia disponible, y DEBE informarse así.

#### Scenario: El modelo no supera a la línea de base de un frame

- **WHEN** el F1 macro del modelo secuencial no supera al del clasificador de un solo frame
- **THEN** se informa que la evidencia no sostiene el aporte de la dimensión temporal
- **AND** el resultado se publica igual, sin buscar una configuración que revierta la conclusión

#### Scenario: La línea de base de identidad predice bien

- **WHEN** el clasificador que sólo ve `id_caballo` alcanza un F1 macro comparable al del modelo
- **THEN** se informa que la clase está confundida con el ejemplar en este dataset
- **AND** se hace constar que el resultado del modelo no es atribuible a la expresión facial

### Requirement: Métricas que el desbalance no vuelva engañosas

Los resultados DEBEN (MUST) informarse con **F1 macro** y **balanced accuracy**. La accuracy
global NO DEBE (MUST NOT) usarse como métrica de aceptación.

DEBE acompañarse de la **matriz de confusión** sobre las predicciones fuera de muestra, y del
detalle por clase.

Con la distribución vigente en clips íntegros —15 % `Disgustado`, 40 % `Interesado`, 45 %
`Sereno`— un clasificador que nunca prediga `Disgustado` alcanza 85 % de accuracy. `Disgustado`
es la clase de interés clínico y la que costó todo el trabajo del objetivo 2.

#### Scenario: Un modelo que nunca predice la clase minoritaria

- **WHEN** el modelo no predice `Disgustado` en ninguna muestra fuera de entrenamiento
- **THEN** la matriz de confusión lo hace visible
- **AND** el F1 macro refleja la falla aunque la accuracy global sea alta

### Requirement: Ablación por origen de la alineación

El sistema DEBE (MUST) entrenar y evaluar dos veces: sobre todos los clips íntegros, y sobre el
subconjunto cuyos 16 frames se alinearon por **detección propia** (`origen_alineacion` =
`keypoints`). Ambos resultados DEBEN informarse.

NO DEBE presentarse un único número sin aclarar cuál de los dos conjuntos lo produjo.

Este requisito existe porque 280 de los frames alineados tienen posiciones que nadie observó
—interpoladas o anotadas a mano— y en la defensa hay que poder contestar cuánto del resultado se
apoya en ellas. El subconjunto sólo-detectado tiene menos datos y probablemente rinda peor; eso
no lo invalida, lo cuantifica.

#### Scenario: Reporte de resultados

- **WHEN** se informa el desempeño del modelo
- **THEN** se informan las dos corridas, con la cantidad de clips y ejemplares de cada una
- **AND** se hace constar que la corrida completa incluye posiciones estimadas

### Requirement: Criterio de aceptación anclado y declarado

El criterio para considerar útil al modelo DEBE (MUST) declararse **antes** de entrenar y DEBE
anclarse en las líneas de base medidas, no en un valor absoluto elegido a criterio.

El resultado DEBE informarse aunque no cumpla el criterio. Si el criterio se cambia después de
ver los números, DEBE quedar escrito que se cambió, por qué, y qué daba antes.

Este requisito existe porque el proyecto ya cometió el error inverso: en
`interpolacion-temporal-de-keypoints` el umbral se declaró antes de medir —correcto— pero sin
anclarlo en ninguna referencia física, resultó imposible de cumplir por construcción, y hubo que
re-declararlo. Declararlo antes protege contra elegir el número que conviene; anclarlo protege
contra elegir uno que nada podría cumplir.

#### Scenario: El modelo no alcanza el criterio declarado

- **WHEN** el F1 macro medio no supera a las líneas de base por el margen declarado
- **THEN** el resultado se informa igual, con su distribución entre folds
- **AND** no se buscan configuraciones adicionales con el objetivo de revertir la conclusión

#### Scenario: Se decide cambiar el criterio después de medir

- **WHEN** la evidencia muestra que el criterio declarado estaba mal fundado
- **THEN** se re-declara de forma explícita, con el motivo
- **AND** se informa el resultado contra el criterio viejo y contra el nuevo
