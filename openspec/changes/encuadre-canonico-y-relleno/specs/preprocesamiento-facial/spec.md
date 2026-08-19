# preprocesamiento-facial (delta)

> Convención de lenguaje normativo: `DEBE (MUST)` equivale a MUST y `NO DEBE (MUST NOT)` a
> MUST NOT, según RFC 2119.

## ADDED Requirements

### Requirement: Encuadre canónico satisfacible

El encuadre canónico DEBE (MUST) ser satisfacible con la geometría real del dataset: la porción
del frame que el recorte abarca NO DEBE (MUST NOT) exceder sistemáticamente el tamaño del cuadro
de origen.

La relación es determinista y conviene tenerla presente al elegir los parámetros: el recorte
abarca `1 / (separación canónica)` veces la distancia entre los dos puntos de anclaje. Elegir esos
puntos sin contrastarlos contra la distancia real que tienen en el dataset produce un encuadre
imposible, y el síntoma no es un error sino un recorte lleno de relleno.

#### Scenario: Elección de las posiciones canónicas

- **WHEN** se fijan las posiciones canónicas de los puntos de anclaje
- **THEN** la porción de frame que implican se contrasta contra la distancia medida entre esos
  puntos en el dataset
- **AND** contra las dimensiones de los videos de origen

#### Scenario: Recorte que excede el frame

- **WHEN** el cuadrado canónico de un frame concreto excede los límites de la imagen
- **THEN** el sistema completa la parte faltante replicando el borde
- **AND** NO DEBE (MUST NOT) completarla con un valor constante de alto contraste

### Requirement: Auditoría de la calidad geométrica del recorte

El sistema DEBE (MUST) poder informar, para los recortes alineados y sin abrir las imágenes, qué
proporción de cada uno proviene de relleno y qué escala tenía la cara en el origen, desglosado por
ejemplar.

Es una propiedad que la verificación de la alineación no cubre: ésa mide el par detectado contra
su posición canónica y da error nulo aunque los puntos estén sobre la anatomía equivocada, porque
se contrasta contra los mismos puntos con que se construyó la transformada.

El desglose por ejemplar es lo que importa. Un relleno parejo sería una pérdida de calidad; un
relleno concentrado en algunos ejemplares es un atajo que el clasificador puede aprender, porque
el ejemplar predice la clase.

#### Scenario: Auditoría del dataset alineado

- **WHEN** se solicita el reporte de calidad de los recortes
- **THEN** el sistema informa la distribución de relleno y la de escala
- **AND** las desglosa por ejemplar
- **AND** señala los recortes cuya geometría los vuelve sospechosos

## MODIFIED Requirements

### Requirement: Región de interés cuadrada

El sistema DEBE (MUST) producir recortes cuadrados de lado fijo, aptos como entrada de un backbone
preentrenado, con la cara del equino contenida en ellos.

El lado en píxeles NO DEBE (MUST NOT) confundirse con el encuadre: el primero es resolución de
salida y lo fija la arquitectura del modelo; el segundo es qué porción del frame entra, y lo fija
la separación entre las posiciones canónicas de los puntos de anclaje.

#### Scenario: Frame con par de referencia válido

- **WHEN** el frame tiene el par ocular/hocico por encima del umbral de confianza
- **THEN** el sistema produce el recorte por transformada de similitud al encuadre canónico

#### Scenario: Frame sin par de referencia válido

- **WHEN** el frame no tiene par de referencia utilizable
- **THEN** el sistema recorta una región cuadrada derivada de la caja de detección
- **AND** lo marca como no alineado, para que las etapas siguientes puedan distinguirlo
