# persistencia-artefactos (delta)

> Convención de lenguaje normativo: el proyecto se escribe en español, pero conserva los términos
> RFC 2119 entre paréntesis en la primera cláusula normativa de cada requisito. `DEBE (MUST)`
> equivale a MUST y `NO DEBE (MUST NOT)` a MUST NOT.

## ADDED Requirements

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
