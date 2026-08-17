# persistencia-artefactos Specification

## Purpose
TBD - created by archiving change persistencia-y-orquestacion-colab. Update Purpose after archive.
## Requirements
### Requirement: Raíz de proyecto persistente

El sistema DEBE (MUST) resolver todas las rutas de artefactos contra una única raíz de proyecto que
sobreviva al cierre de la sesión de Colab. Ningún notebook DEBE (MUST NOT) escribir artefactos
destinados a otra etapa en rutas efímeras del entorno de ejecución.

La raíz DEBE ser configurable en un solo lugar, de modo que el pipeline pueda ejecutarse también
fuera de Colab —en la máquina local o en otro entorno— cambiando únicamente ese valor.

#### Scenario: Ejecución en Colab con Drive disponible

- **WHEN** un notebook arranca en Colab y el usuario autoriza el montaje de Google Drive
- **THEN** el sistema monta Drive y fija la raíz del proyecto dentro de la unidad montada
- **AND** crea las subcarpetas de artefactos que falten
- **AND** informa la ruta absoluta resuelta, para que quede visible en la salida del notebook

#### Scenario: Ejecución fuera de Colab

- **WHEN** un notebook arranca en un entorno donde no existe el módulo de integración de Colab
- **THEN** el sistema resuelve la raíz contra el directorio local configurado, sin intentar montar
  Drive
- **AND** el resto del pipeline funciona sin cambios en el código

#### Scenario: Raíz sin permiso de escritura

- **WHEN** la raíz de proyecto resuelta no admite escritura
- **THEN** el sistema falla de inmediato con un mensaje que identifica la ruta y el problema
- **AND** NO DEBE continuar la ejecución escribiendo en una ubicación alternativa silenciosamente

### Requirement: Lectura de los videos sin descarga repetida

El sistema DEBE (MUST) leer los videos crudos desde la carpeta persistente en Drive, sin volver a
descargarlos en cada sesión.

#### Scenario: Videos ya presentes en Drive

- **WHEN** comienza la etapa de inventario y la carpeta de videos de la raíz de proyecto contiene
  los archivos esperados
- **THEN** el sistema los usa directamente, sin ejecutar ninguna descarga
- **AND** el tiempo de arranque de la etapa no depende del tamaño del dataset

#### Scenario: Carpeta de videos vacía o incompleta

- **WHEN** la carpeta de videos no existe o no contiene ningún video
- **THEN** el sistema falla con un mensaje que indica qué ruta esperaba y qué encontró
- **AND** NO DEBE producir un inventario vacío que las etapas siguientes interpreten como válido

### Requirement: Limpieza selectiva y no destructiva

El sistema NO DEBE (MUST NOT) borrar artefactos persistentes como efecto secundario del arranque de
un notebook. Cualquier borrado de artefactos DEBE ser una acción explícita, dirigida a una etapa
concreta, y separada del flujo normal de ejecución.

#### Scenario: Re-ejecución de una etapa temprana

- **WHEN** se vuelve a ejecutar el notebook de preparación del dataset con artefactos posteriores ya
  generados
- **THEN** los artefactos de las etapas posteriores permanecen intactos
- **AND** el usuario puede decidir cuáles invalidar

#### Scenario: Invalidación deliberada de una etapa

- **WHEN** el usuario invoca explícitamente la limpieza de una etapa determinada
- **THEN** el sistema borra los artefactos de esa etapa y su checkpoint
- **AND** informa qué borró

### Requirement: Checkpointing reanudable de etapas largas

Toda etapa que procese los clips de a uno y supere varios minutos de ejecución DEBE (MUST) persistir su
progreso incrementalmente, de modo que una interrupción no obligue a recomenzar. Al relanzarse, la
etapa DEBE retomar en la primera unidad de trabajo pendiente.

El progreso DEBE persistirse con una frecuencia tal que una interrupción no descarte más que una
fracción menor del trabajo ya hecho.

#### Scenario: Sesión interrumpida a mitad de una etapa

- **WHEN** una etapa procesó 60 de 92 clips y la sesión de Colab se desconecta
- **THEN** el progreso de esos 60 clips queda persistido en la raíz de proyecto
- **AND** al volver a ejecutar la etapa, ésta procesa solamente los 32 clips restantes
- **AND** el resultado final es equivalente al de una corrida ininterrumpida

#### Scenario: Etapa ya completa

- **WHEN** se ejecuta una etapa cuyo checkpoint cubre la totalidad de las unidades de trabajo
- **THEN** la etapa no vuelve a procesar ningún clip
- **AND** deja disponibles sus artefactos de salida para las etapas siguientes

### Requirement: Invalidación del checkpoint ante cambio de configuración

El checkpoint de una etapa DEBE (MUST) registrar los parámetros de configuración con los que fue generado.
Si la configuración vigente difiere de la registrada, el sistema NO DEBE reutilizar el checkpoint.

Este requisito existe porque mezclar resultados producidos con dos configuraciones distintas genera
un artefacto silenciosamente inválido: no falla, produce números plausibles y erróneos.

#### Scenario: Cambio de un umbral del preprocesamiento

- **WHEN** se modifica un parámetro que afecta el resultado de la etapa y existe un checkpoint
  previo generado con el valor anterior
- **THEN** el sistema descarta el checkpoint y reprocesa la etapa completa
- **AND** informa explícitamente que lo descartó y qué parámetros cambiaron

#### Scenario: Cambio que no afecta el resultado

- **WHEN** se modifica un parámetro que no interviene en el cálculo de la etapa
- **THEN** el checkpoint sigue siendo válido y la reanudación procede normalmente

### Requirement: Caché comprimida de artefactos numerosos

Los artefactos que consistan en miles de archivos pequeños DEBEN (MUST) persistirse en la raíz de proyecto
como un único archivo comprimido por etapa, y expandirse al disco local de la sesión para su uso.

#### Scenario: Persistencia tras generar los frames

- **WHEN** una etapa termina de generar sus frames en el disco local de la sesión
- **THEN** el sistema los empaqueta en un único archivo comprimido dentro de la raíz de proyecto
- **AND** registra cuántos archivos contiene, para poder verificar la integridad al expandirlo

#### Scenario: Etapa posterior que consume esos frames

- **WHEN** una etapa posterior necesita los frames y la caché comprimida existe
- **THEN** el sistema la expande al disco local de la sesión sin regenerar los frames
- **AND** verifica que la cantidad de archivos expandidos coincida con la registrada
- **AND** si no coincide, trata la caché como inválida en lugar de usarla

### Requirement: Trazabilidad de rutas en los CSV

Los CSV que registran rutas de archivos DEBEN (MUST) permitir localizar esos archivos desde una sesión
distinta de aquella en que se generaron.

Este requisito existe porque hoy `inventario_clips.csv` guarda rutas absolutas de `/content`, que en
una sesión nueva no apuntan a nada.

#### Scenario: CSV generado en una sesión y leído en otra

- **WHEN** una etapa lee un CSV producido por una sesión anterior y resuelve las rutas que contiene
- **THEN** las rutas resueltas apuntan a los archivos reales en la raíz de proyecto vigente
- **AND** la verificación de existencia de archivos no reporta faltantes espurios

