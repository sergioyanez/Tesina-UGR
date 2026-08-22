# orquestacion-notebooks (delta)

> Convención de lenguaje normativo: el proyecto se escribe en español, pero conserva los términos
> RFC 2119 entre paréntesis en la primera cláusula normativa de cada requisito. `DEBE (MUST)`
> equivale a MUST y `NO DEBE (MUST NOT)` a MUST NOT.

## ADDED Requirements

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
