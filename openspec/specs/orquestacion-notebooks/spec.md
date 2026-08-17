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

