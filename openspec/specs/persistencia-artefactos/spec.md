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

### Requirement: Fuente única declarada del material de video

El material de video DEBE (MUST) resolverse contra una única fuente declarada explícitamente para
la ejecución en curso. El sistema NO DEBE (MUST NOT) sustituirla en silencio por otra copia del
material que encuentre disponible, aunque esa copia exista y tenga la estructura esperada.

Si la fuente declarada no está disponible, el sistema DEBE interrumpir la ejecución informando qué
rutas revisó y cómo declarar la correcta. Continuar con otra copia NO DEBE ser un resultado
posible: una copia parcial no produce un fallo sino una métrica plausible y equivocada, que es
peor, porque se publica sin que nadie la cuestione.

La declaración DEBE viajar con el entorno donde el pipeline se ejecuta, y no depender de que quien
lo arranca recuerde fijarla en cada sesión.

#### Scenario: Fuente declarada disponible

- **WHEN** el entorno declara la carpeta de videos y esa carpeta contiene carpetas de clase
- **THEN** el sistema cataloga el material desde ahí
- **AND** informa la ruta absoluta que resolvió, para que quede en la salida del notebook
- **AND** registra en el inventario la ruta de cada video, con la fuente incluida

#### Scenario: Fuente declarada ausente

- **WHEN** el entorno no declara ninguna carpeta de videos
- **AND** ninguna de las rutas de la cascada contiene carpetas de clase
- **THEN** el sistema interrumpe la ejecución
- **AND** enumera las rutas que revisó y cómo declarar la carpeta correcta

#### Scenario: Copia parcial del material bajo la raíz de artefactos

- **WHEN** existe bajo la raíz de artefactos una copia del material con la estructura de clases
- **AND** esa copia no es la fuente declarada
- **THEN** el sistema no la usa como fuente
- **AND** la corrida no puede completarse midiendo sobre ella

#### Scenario: Arranque por fuera del lanzador

- **WHEN** un notebook del pipeline se ejecuta desde un Jupyter arrancado sin el lanzador del
  proyecto
- **THEN** el entorno de ejecución aporta igualmente la raíz de artefactos y la fuente de videos
- **AND** la corrida resuelve las mismas rutas que resolvería a través del lanzador

### Requirement: Admisión de los videos más cortos que un clip

Un video cuya duración no alcance para formar un clip del largo objetivo NO DEBE (MUST NOT) quedar
excluido del dataset por ese solo motivo. El sistema DEBE (MUST) admitirlo entero como una única
muestra siempre que supere un piso declarado explícitamente, y ese piso DEBE ser un parámetro
auditable de la etapa, no una constante implícita en el código de segmentación.

La regla DEBE aplicarse únicamente cuando la segmentación no produjo ninguna ventana. Un video que
alcanza el largo objetivo DEBE segmentarse igual que antes, sin que la admisión de videos cortos
altere sus clips.

Este requisito existe porque el piso de duración no filtra tomas de mala calidad: filtra
ejemplares. Medido sobre el catálogo de 234 videos, la clase minoritaria tenía 42 caballos y su
video mediano duraba 2,9 segundos, así que un piso de 5 segundos dejaba entrar 11 y descartaba 31
ejemplares completos. Como el ejemplar está correlacionado con la clase, ese descarte no protege la
calidad del dato: alimenta el confounder que el trabajo tiene que resolver antes de entrenar.

El parámetro que fija el piso DEBE formar parte de la configuración que invalida el checkpoint de la
etapa, para que cambiarlo no pueda mezclar muestras generadas con dos criterios distintos.

Las muestras resultantes tienen largo variable, así que cada muestra DEBE registrar su duración en
el inventario, de modo que la heterogeneidad quede auditable y se pueda informar o ablacionar.

#### Scenario: Video más corto que el largo objetivo pero por encima del piso

- **WHEN** un video no alcanza la duración mínima para segmentarse
- **AND** su duración supera el piso declarado para videos cortos
- **THEN** el sistema lo admite entero como una única muestra
- **AND** registra la duración real de esa muestra en el inventario

#### Scenario: Video por debajo del piso

- **WHEN** la duración de un video no alcanza el piso declarado
- **THEN** el sistema no genera ninguna muestra a partir de ese video
- **AND** lo informa, en vez de descartarlo en silencio

#### Scenario: Video que alcanza el largo objetivo

- **WHEN** un video admite al menos una ventana de segmentación
- **THEN** sus muestras son exactamente las que produciría la segmentación sin la regla de videos
  cortos
- **AND** las muestras generadas en corridas anteriores siguen siendo válidas

#### Scenario: Cambio del piso declarado

- **WHEN** se ejecuta la etapa con un piso distinto del registrado en el checkpoint
- **THEN** el sistema descarta el progreso previo y vuelve a segmentar todo el material
- **AND** no mezcla muestras generadas con los dos criterios

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

