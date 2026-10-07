> Convención: `DEBE (MUST)` y `NO DEBE (MUST NOT)` tienen el sentido de RFC 2119.

## MODIFIED Requirements

### Requirement: Líneas de base declaradas antes de entrenar

El sistema DEBE (MUST) evaluar, sobre exactamente las mismas particiones que el modelo y sobre el
conjunto vigente, al menos estas líneas de base:

1. **clase mayoritaria** — el piso absoluto;
2. **un solo frame del clip**, sin información temporal — mide si la dimensión temporal aporta;
3. **sólo la identidad del ejemplar**, sin mirar la imagen — mide cuánta señal hay en el
   confounder;
4. **visual estática**: un clasificador lineal sobre el promedio de las características de los
   16 frames — mide si la agregación aprendida (atención, orden, canal geométrico) aporta sobre
   un promedio uniforme;
5. **metadatos de filmación**: un clasificador que sólo ve duración, fps, resolución y ventana
   del clip — mide cuánto de la clase está determinado por cómo se filmó.

Todas DEBEN estar persistidas junto al modelo en un archivo de resultados, con el F1 macro medio
y su desvío entre folds. El resultado del modelo NO DEBE presentarse sin ellas al lado.

La tercera es la que decide si el problema es abordable con este material: si predice bien, la
clase está determinada por el individuo y ningún modelo que mire la imagen puede distinguir
expresión de ejemplar.

La segunda y la cuarta ponen a prueba la hipótesis del trabajo. Si el modelo secuencial no les
gana, esa hipótesis no queda sostenida por la evidencia disponible, y DEBE informarse así.

La quinta fija el piso honesto contra el que se afirma que el modelo mira la cara: superar a la
clase mayoritaria no alcanza si la filmación ya separa las clases.

#### Scenario: El modelo no supera a la línea de base de un frame

- **WHEN** el F1 macro del modelo secuencial no supera al del clasificador de un solo frame
- **THEN** se informa que la evidencia no sostiene el aporte de la dimensión temporal
- **AND** el resultado se publica igual, sin buscar una configuración que revierta la conclusión

#### Scenario: La línea de base de identidad predice bien

- **WHEN** el clasificador que sólo ve `id_caballo` alcanza un F1 macro comparable al del modelo
- **THEN** se informa que la clase está confundida con el ejemplar en este dataset
- **AND** se hace constar que el resultado del modelo no es atribuible a la expresión facial

#### Scenario: El modelo empata con la línea de base visual estática

- **WHEN** el intervalo de confianza de la diferencia entre el modelo y la línea de base visual
  estática incluye el cero
- **THEN** se informa que no está demostrado que la agregación aprendida aporte sobre un promedio
- **AND** el resultado se publica igual, sin buscar una configuración que revierta la conclusión

#### Scenario: Los metadatos de filmación predicen la clase

- **WHEN** la línea de base de metadatos supera claramente al azar estratificado
- **THEN** se informa como confusor del dataset
- **AND** se usa esa línea de base, y no la clase mayoritaria, como piso para afirmar que el
  modelo aprovecha la imagen

## ADDED Requirements

### Requirement: Intervalos de confianza por ejemplar

El F1 macro del modelo DEBE (MUST) informarse con un intervalo de confianza calculado
remuestreando **caballos**, no clips, sobre las predicciones fuera de muestra. Las diferencias
contra las líneas de base DEBEN calcularse pareadas sobre las mismas réplicas.

Los clips de un mismo caballo no son independientes: un intervalo por clip sería más estrecho de
lo que la evidencia permite.

#### Scenario: Reporte del resultado principal

- **WHEN** se informa el F1 macro del modelo vigente
- **THEN** se informa también su intervalo del 95 % por bootstrap de ejemplares
- **AND** el intervalo de la diferencia contra cada línea de base relevante

### Requirement: Control del confusor de filmación

El sistema DEBE (MUST) informar el rendimiento del modelo restringido al subconjunto de clips sin
los rasgos de filmación que delatan la clase —resolución distinta de la mayoritaria o duración
menor a 5 s—, con la cantidad de clips por clase de ese subconjunto.

#### Scenario: Evaluación sin los rasgos delatores

- **WHEN** se evalúan las predicciones fuera de muestra sólo sobre clips 1080×1920 de al menos 5 s
- **THEN** se informa el F1 macro y la accuracy del modelo y de la línea de base estática en ese
  subconjunto
- **AND** se hace constar cuántos clips de cada clase quedan

### Requirement: Registro del sobreajuste

El entrenamiento DEBE (MUST) registrar por época, además de la pérdida y el F1 de validación, la
pérdida, el F1 macro y la accuracy del train interno evaluado sin dropout. La evaluación del
train NO DEBE alterar el estado aleatorio del entrenamiento ni ninguna predicción.

La evaluación DEBE informar por fold el F1 del train interno en la mejor época, y el gap contra
validación y contra test.

#### Scenario: Curvas persistidas

- **WHEN** termina la validación cruzada del modelo
- **THEN** cada fold persiste la curva con columnas de train y de validación
- **AND** el resumen informa el gap medio train−validación y train−test

#### Scenario: Registrar el train no cambia el modelo

- **WHEN** se entrena con el registro del train activado
- **THEN** las predicciones de test son idénticas a las de una corrida sin ese registro

### Requirement: Trazabilidad del modelo desplegado

El archivo de procedencia del modelo de producción DEBE (MUST) registrar la huella de las
características con las que efectivamente se entrenó, tomada del mismo archivo que usa la caché
vigente.

#### Scenario: La caché cambia de versión

- **WHEN** el conjunto de características vigente pasa a otro archivo de huella
- **THEN** la procedencia del modelo registra la huella del archivo vigente, no la de uno anterior
