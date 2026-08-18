# preprocesamiento-facial (delta)

> Convención de lenguaje normativo: el proyecto se escribe en español, pero conserva los términos
> RFC 2119 entre paréntesis en la primera cláusula normativa de cada requisito. `DEBE (MUST)`
> equivale a MUST, `NO DEBE (MUST NOT)` a MUST NOT y `DEBERÍA (SHOULD)` a SHOULD.

## ADDED Requirements

### Requirement: Persistencia de la pose completa

El sistema DEBE (MUST) persistir la salida cruda del modelo de pose —todos los bodyparts que el
modelo expone, para todos los frames procesados, con sus coordenadas y su confianza— como artefacto
de la etapa, y NO DEBE (MUST NOT) descartarla junto con los intermedios del modelo.

La razón es que la elección de qué keypoints usar es una decisión analítica que se revisa varias
veces, mientras que la inferencia es la operación más cara del pipeline. Sin este artefacto, cada
revisión de esa decisión obliga a reinferir, y el costo desalienta justamente el análisis que la
haría correcta.

#### Scenario: Fin de la inferencia de pose

- **WHEN** termina la estimación de pose sobre el conjunto de clips
- **THEN** el sistema escribe una tabla con una fila por frame y por bodypart, o equivalente
- **AND** cada fila incluye el clip, el frame, el nombre del bodypart, sus coordenadas y su confianza
- **AND** la tabla queda en la raíz de proyecto, alcanzable sin volver a ejecutar el modelo

#### Scenario: Análisis posterior de keypoints

- **WHEN** se quiere evaluar un keypoint distinto del que se venía usando
- **THEN** el sistema responde a partir de la tabla persistida
- **AND** no requiere volver a ejecutar el modelo de pose

#### Scenario: Trazabilidad hacia el CSV de keypoints

- **WHEN** existen tanto la tabla de pose completa como el CSV de keypoints faciales
- **THEN** cada fila del CSV de keypoints es reconstruible desde la tabla completa aplicando el
  criterio de selección vigente
- **AND** esa correspondencia es verificable de forma automática

### Requirement: Barrido de candidatos a keypoint facial

El sistema DEBE (MUST) ofrecer, sobre la tabla de pose persistida, un barrido que informe la
cobertura de alineación alcanzable con cada candidato a keypoint facial y con la estrategia de
máxima confianza por frame, desglosada por clase y por caballo.

El barrido NO DEBE (MUST NOT) requerir ejecutar el modelo de pose.

#### Scenario: Comparación entre candidatos

- **WHEN** se solicita el barrido sobre los candidatos a hocico
- **THEN** el sistema informa, para cada candidato, la proporción de frames que superarían el umbral
  de confianza junto con el keypoint ocular
- **AND** informa el mismo dato para la estrategia de máxima confianza por frame

#### Scenario: Candidato ausente en el modelo

- **WHEN** un candidato del listado no figura entre los bodyparts que el modelo expone
- **THEN** el sistema lo informa como ausente
- **AND** continúa el barrido con los candidatos restantes en lugar de fallar

#### Scenario: Efecto geométrico de alternar keypoints

- **WHEN** el barrido evalúa la estrategia de máxima confianza por frame
- **THEN** el sistema informa además cuánto varía la distancia ojo-hocico entre los candidatos
  involucrados
- **AND** deja explícito que esa variación se traslada a la escala del recorte normalizado

## MODIFIED Requirements

### Requirement: Estimación de keypoints faciales

El sistema DEBE estimar keypoints faciales del equino —como mínimo un punto ocular y un punto en el
hocico— mediante un modelo preentrenado para cuadrúpedos, sin anotación manual. El sistema NO DEBE (MUST NOT)
asumir nombres fijos de keypoints: DEBE descubrir el esquema que expone el modelo en tiempo de
ejecución y seleccionar los faciales a partir de él.

Cuando el esquema del modelo ofrezca varios candidatos para un mismo punto facial, el sistema NO
DEBE (MUST NOT) resolver la elección por el orden de una lista fijada a priori. La selección DEBE
(MUST) sustentarse en la confianza medida sobre el propio dataset, y el keypoint elegido DEBE (MUST)
quedar registrado como parámetro explícito de la etapa, de modo que la corrida sea reproducible y la
decisión, auditable.

#### Scenario: Keypoints faciales disponibles

- **WHEN** el modelo devuelve el punto ocular y el del hocico con confianza suficiente
- **THEN** el sistema los registra en coordenadas de la imagen original
- **AND** los usa como referencia para la normalización de pose

#### Scenario: Esquema de keypoints desconocido

- **WHEN** se carga el modelo de pose por primera vez
- **THEN** el sistema imprime el listado completo de keypoints disponibles
- **AND** deja explícito cuáles seleccionó como ocular y como hocico

#### Scenario: Varios candidatos para el mismo punto facial

- **WHEN** el esquema del modelo expone más de un bodypart que podría cumplir el rol de hocico
- **THEN** el sistema usa el que se haya fijado como parámetro de la etapa
- **AND** deja registrado cuál usó junto a los resultados de la corrida

#### Scenario: Keypoints faciales ausentes o de baja confianza

- **WHEN** no se obtiene el par ocular/hocico con confianza suficiente en un frame
- **THEN** el sistema no normaliza la pose de ese frame
- **AND** recurre a la ROI derivada de la caja de detección, dejándolo marcado en el CSV
