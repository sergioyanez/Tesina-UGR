# persistencia-artefactos (delta)

> Convención de lenguaje normativo: el proyecto se escribe en español, pero conserva los términos
> RFC 2119 entre paréntesis en la primera cláusula normativa de cada requisito. `DEBE (MUST)`
> equivale a MUST y `NO DEBE (MUST NOT)` a MUST NOT.

## ADDED Requirements

### Requirement: Composición declarada del dataset por categoría

La pertenencia de un video a una clase DEBE (MUST) surgir de una declaración explícita y NO DEBE
(MUST NOT) deducirse de la posición del archivo en el árbol de carpetas. Para cada categoría el
sistema DEBE declarar qué subcarpetas aportan videos además de los que están sueltos en la raíz
de esa categoría, y esa lista DEBE poder ser **distinta en cada categoría**.

La declaración DEBE vivir en un único lugar del código, fuera de las celdas de los notebooks: la
composición del dataset es una decisión de etiquetado que hay que poder auditar y citar en el
informe final, no un detalle de implementación de una etapa.

Deducir la clase del nombre de la carpeta contenedora decide dos cosas con un solo dato —qué
archivos entran y con qué etiqueta— y por eso no admite que un video esté en una subcarpeta de
trabajo y a la vez tenga etiqueta. El costo no es un fallo sino material invisible, concentrado
en la clase de la que menos hay.

#### Scenario: Video suelto en la carpeta de su categoría

- **WHEN** un video está directamente en la carpeta de una categoría
- **THEN** el sistema lo cataloga con esa categoría como etiqueta

#### Scenario: Video en una subcarpeta declarada para esa categoría

- **WHEN** un video está dentro de una subcarpeta que la categoría declara como aportante
- **THEN** el sistema lo cataloga con la etiqueta de la categoría, no con el nombre de la
  subcarpeta
- **AND** registra de qué carpeta salió, para poder rastrear después la procedencia de cada
  muestra

#### Scenario: Subcarpeta declarada en una categoría y no en otra

- **WHEN** dos categorías tienen una subcarpeta con el mismo nombre y sólo una de ellas la declara
- **THEN** el sistema suma los videos únicamente en la categoría que la declaró

#### Scenario: Video en una subcarpeta no declarada

- **WHEN** un video está bajo una categoría pero dentro de una subcarpeta que no fue declarada
- **THEN** el sistema NO lo incorpora al catálogo
- **AND** informa cuántos son y en qué carpeta están, porque descartar y perder no son lo mismo

#### Scenario: Subcarpeta declarada que no existe en el disco

- **WHEN** una categoría declara una subcarpeta que no está presente
- **THEN** el sistema lo informa y continúa, en lugar de fallar o de omitirlo en silencio

#### Scenario: Carpeta anidada que se llama igual que una categoría

- **WHEN** existe una carpeta anidada cuyo nombre coincide con el de una categoría, sin ser la
  carpeta de esa categoría —una descarga parcial de Drive, por ejemplo
- **THEN** el sistema NO incorpora su contenido al catálogo

#### Scenario: Nombres de carpeta con distinta grafía entre entornos

- **WHEN** la misma carpeta aparece con signos, acentos o mayúsculas distintas según el entorno
  —`Seleccionar cabeza?` en Drive y `Seleccionar cabeza_` en una copia bajada como ZIP
- **THEN** el sistema la reconoce como la misma carpeta declarada
- **AND** informa con qué nombre real la encontró

## MODIFIED Requirements

### Requirement: Lectura de los videos sin descarga repetida

El sistema DEBE (MUST) leer los videos crudos desde la carpeta persistente en Drive, sin volver a
descargarlos en cada sesión.

El sistema DEBE resolver **dónde está** esa carpeta en vez de suponerla, y DEBE exigir que la
candidata contenga efectivamente carpetas de categoría antes de aceptarla. Una carpeta que existe
pero no tiene el dataset adentro es el peor resultado posible: no falla, produce un inventario en
cero y las etapas siguientes procesan nada.

Cuando la carpeta la localice un componente y la consuma otro, el que la localiza DEBE pasársela
al que la consume. Dos búsquedas independientes son dos respuestas posibles, y la divergencia no
se manifiesta como error sino como un inventario que no corresponde a la carpeta que se informó.

#### Scenario: Videos ya presentes en Drive

- **WHEN** comienza la etapa de inventario y la carpeta de videos contiene las carpetas de
  categoría esperadas
- **THEN** el sistema los usa directamente, sin ejecutar ninguna descarga
- **AND** el tiempo de arranque de la etapa no depende del tamaño del dataset

#### Scenario: Carpeta de videos vacía o incompleta

- **WHEN** la carpeta de videos no existe o no contiene ningún video
- **THEN** el sistema falla con un mensaje que indica qué ruta esperaba y qué encontró
- **AND** NO DEBE producir un inventario vacío que las etapas siguientes interpreten como válido

#### Scenario: El dataset vive en una carpeta compartida

- **WHEN** las carpetas de categoría no están bajo la raíz de proyecto sino en una carpeta
  compartida a la que hay que acceder por un acceso directo
- **THEN** el sistema la localiza y la usa
- **AND** si no la encuentra, el mensaje indica qué rutas revisó y cómo crear el acceso directo

#### Scenario: El orquestador ejecuta la etapa de inventario

- **WHEN** el orquestador lanza la etapa que construye el inventario
- **THEN** le pasa la carpeta de videos que él mismo localizó
- **AND** la etapa inventaria esa carpeta y no otra que hubiera resuelto por su cuenta

#### Scenario: Nombres de archivo repetidos entre la categoría y su subcarpeta

- **WHEN** el mismo nombre de archivo aparece más de una vez en el catálogo, sea entre dos
  categorías o entre la raíz de una categoría y una de sus subcarpetas declaradas
- **THEN** el sistema falla e informa todas las rutas en conflicto
- **AND** NO DEBE continuar, porque el nombre del archivo es la clave que une las etapas y un
  choque mezclaría muestras de distinta etiqueta bajo la misma clave
