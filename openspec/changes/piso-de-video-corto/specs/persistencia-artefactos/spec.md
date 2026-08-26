# persistencia-artefactos (delta)

> Convención de lenguaje normativo: el proyecto se escribe en español, pero conserva los términos
> RFC 2119 entre paréntesis en la primera cláusula normativa de cada requisito. `DEBE (MUST)`
> equivale a MUST y `NO DEBE (MUST NOT)` a MUST NOT.

## ADDED Requirements

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
