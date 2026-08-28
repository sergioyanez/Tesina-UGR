# orquestacion-notebooks Specification

## Purpose
TBD - created by archiving change persistencia-y-orquestacion-colab. Update Purpose after archive.
## Requirements
### Requirement: Ejecución encadenada del pipeline

El sistema DEBE ofrecer un único punto de entrada que ejecute los notebooks del pipeline en el orden
de sus dependencias, sin intervención manual entre etapas.

Cada notebook DEBE ejecutarse en un proceso de kernel propio. Las etapas NO DEBEN (MUST NOT)
comunicarse por estado en memoria compartida: el único canal entre etapas son los artefactos
persistidos.

#### Scenario: Corrida completa desde cero

- **WHEN** se ejecuta el orquestador sin ningún artefacto previo
- **THEN** las etapas corren en orden de dependencia
- **AND** cada una encuentra las entradas producidas por la anterior
- **AND** al terminar existen todos los artefactos finales del pipeline

#### Scenario: Fallo de una etapa

- **WHEN** una etapa termina con error
- **THEN** el orquestador detiene el pipeline sin ejecutar las etapas siguientes
- **AND** informa qué etapa falló y dónde quedó la copia ejecutada con el error
- **AND** NO DEBE continuar con etapas que dependen de la que falló

### Requirement: Salteo de etapas ya completas

El orquestador DEBE (MUST) consultar el estado de cada etapa antes de ejecutarla y saltear aquellas cuyos
artefactos de salida ya existen y están completos.

Este requisito es lo que hace utilizable el pipeline en Colab, donde una corrida completa excede el
tiempo de sesión disponible: permite avanzar en tramos a lo largo de varias sesiones.

#### Scenario: Reanudación tras un corte de sesión

- **WHEN** se ejecuta el orquestador con las dos primeras etapas ya completas
- **THEN** esas dos etapas se saltean e informan que ya estaban completas
- **AND** la ejecución arranca en la primera etapa pendiente

#### Scenario: Reejecución forzada de una etapa

- **WHEN** el usuario solicita explícitamente reejecutar una etapa determinada
- **THEN** el orquestador la ejecuta aunque sus artefactos existan
- **AND** las etapas posteriores que dependen de ella quedan marcadas como pendientes

### Requirement: Parámetros centralizados

El orquestador DEBE (MUST) permitir fijar en un solo lugar los parámetros que atraviesan varias etapas, e
inyectarlos en cada notebook al ejecutarlo. Un notebook ejecutado por el orquestador NO DEBE
requerir edición manual de sus celdas para tomar esos parámetros.

#### Scenario: Barrido de un umbral del preprocesamiento

- **WHEN** se cambia un umbral en el orquestador y se relanza el pipeline
- **THEN** la etapa afectada recibe el valor nuevo
- **AND** su checkpoint previo se invalida por el cambio de configuración
- **AND** las etapas no afectadas conservan sus artefactos

#### Scenario: Ejecución de un notebook por separado

- **WHEN** un notebook se abre y ejecuta directamente, sin pasar por el orquestador
- **THEN** usa sus valores por defecto declarados
- **AND** produce el mismo resultado que si el orquestador le hubiera inyectado esos mismos valores

### Requirement: Registro de la ejecución

El orquestador DEBE (MUST) guardar, por cada etapa ejecutada, una copia del notebook con sus salidas, en la
raíz de proyecto persistente.

Este requisito no es operativo sino metodológico: esa copia es la evidencia reproducible de qué se
ejecutó, con qué parámetros y con qué resultado, que respalda los números presentados en el informe
final.

#### Scenario: Etapa ejecutada con éxito

- **WHEN** una etapa termina correctamente
- **THEN** queda persistida una copia del notebook con todas sus salidas y su marca de tiempo
- **AND** los parámetros con los que se la ejecutó son legibles dentro de esa copia

#### Scenario: Etapa fallida

- **WHEN** una etapa falla a mitad de ejecución
- **THEN** la copia persistida conserva las salidas de las celdas ejecutadas y el error
- **AND** la copia de la última ejecución exitosa NO DEBE ser sobrescrita por la fallida

### Requirement: Diagnóstico del estado del pipeline

El sistema DEBE (MUST) poder informar, sin ejecutar ninguna etapa, en qué estado está cada una: pendiente,
parcial o completa.

#### Scenario: Consulta de estado al abrir una sesión nueva

- **WHEN** el usuario ejecuta el diagnóstico al comienzo de una sesión
- **THEN** obtiene por cada etapa su estado y los artefactos que produjo
- **AND** para las etapas parciales, cuántas unidades de trabajo quedan pendientes
- **AND** no se modifica ningún artefacto

### Requirement: Ejecución en más de un entorno de cómputo

El pipeline DEBE (MUST) poder ejecutarse tanto en Colab como en una máquina local con GPU, sin que
ninguno de los dos entornos sea un caso especial que haya que reparar antes de correrlo. El
notebook NO DEBE (MUST NOT) declarar en cuál de los dos está: el sistema lo detecta.

Ningún resultado del pipeline puede depender del entorno donde se ejecutó. Los artefactos
producidos en uno DEBEN poder leerse en el otro sin conversión, lo que exige que las rutas
persistidas sigan siendo relativas a la raíz de proyecto.

Depender de un solo proveedor de cómputo ya bloqueó el trabajo una vez, al agotarse el acceso a
GPU de la suscripción. Y en una tesina que se defiende, "el resultado se obtuvo en el entorno que
había ese día" no es una respuesta: la equivalencia entre entornos es parte de la trazabilidad.

#### Scenario: Ejecución en Colab

- **WHEN** el pipeline arranca dentro de Colab
- **THEN** monta Drive, resuelve la raíz de proyecto en la carpeta de Drive y usa el disco local
  de la sesión como área de trabajo efímera

#### Scenario: Ejecución en una máquina local

- **WHEN** el pipeline arranca fuera de Colab
- **THEN** no intenta montar Drive ni sincronizar código desde el repositorio remoto
- **AND** resuelve la raíz de proyecto y la carpeta de videos a partir de las variables de entorno
  del proyecto, o de las ubicaciones por defecto si no están definidas

#### Scenario: Artefactos producidos en un entorno y leídos en el otro

- **WHEN** una etapa lee un artefacto generado por una corrida del otro entorno
- **THEN** lo resuelve contra la raíz vigente y lo encuentra, sin reprocesar ni reinferir nada

#### Scenario: Detección de GPU al arrancar una etapa

- **WHEN** una etapa que usa modelos de visión arranca
- **THEN** informa si hay GPU disponible y cuál es, antes de procesar el primer clip

### Requirement: Entorno de dependencias declarado y pinneado

El conjunto de dependencias del pipeline DEBE (MUST) estar declarado en un archivo versionado del
repositorio, con versiones fijas, y NO DEBE (MUST NOT) existir únicamente como una línea de
instalación dentro de una celda de notebook.

La declaración DEBE incluir las versiones que se verificaron compatibles entre sí, no sólo los
nombres de los paquetes: DeepLabCut fija `numpy<2` y el resto del stack científico se compila
contra una versión concreta de numpy, así que el conjunto es la unidad que funciona o no funciona.

Un entorno cuyo contenido exacto no está escrito no es reproducible por el tribunal, ni por el
otro integrante del equipo, ni por el propio autor dentro de seis meses.

#### Scenario: Reconstrucción del entorno desde cero

- **WHEN** alguien instala el entorno a partir del archivo de dependencias declarado
- **THEN** obtiene un conjunto de versiones en el que DeepLabCut, el detector, el stack científico
  y la biblioteca de tensores conviven sin conflicto

#### Scenario: Instalación de dependencias dentro de un notebook

- **WHEN** un notebook llega a su paso de instalación de dependencias fuera de Colab
- **THEN** NO instala nada, porque el entorno local ya está declarado y construido aparte
- **AND** verifica que las versiones presentes son las que el pipeline necesita, fallando de
  inmediato si no lo son

#### Scenario: Entorno local incompleto

- **WHEN** el entorno local no tiene alguna dependencia obligatoria del pipeline
- **THEN** la etapa falla en su arranque, con el nombre del paquete faltante y cómo instalarlo
- **AND** NO comienza a procesar clips

### Requirement: Kernel de ejecución verificado

El orquestador DEBE (MUST) ejecutar cada notebook de etapa con el intérprete del entorno declarado,
y DEBE verificar antes de lanzar la primera etapa que ese kernel existe y es el correcto.

Papermill ejecuta por nombre de kernel. Un nombre que resuelve a otro intérprete produce el peor
fallo posible: la corrida arranca, procesa, y falla —o peor, no falla— con un stack de versiones
distinto del verificado. Es la misma familia de fallo silencioso que el salteo de etapas ya
completas, y se detecta igual: verificando antes en vez de confiar.

#### Scenario: Kernel del entorno declarado ausente

- **WHEN** el orquestador va a lanzar una etapa y el kernel del entorno no está registrado
- **THEN** falla antes de ejecutar nada, indicando cómo registrarlo

