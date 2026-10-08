# comparacion-de-modelos Specification

## Purpose
TBD - created by archiving change base-comun-comparacion-de-modelos. Update Purpose after archive.
## Requirements
### Requirement: Reserva de caballos persistida y de un solo uso

El sistema DEBE (MUST) mantener una reserva del 15 % de los caballos, estratificada por clase dominante del caballo, con semilla fija, persistida con la huella del dataset en `comparacion/reserva_caballos.json`. La reserva NO DEBE (MUST NOT) usarse en ninguna evaluación de desarrollo. `evaluar_en_reserva` DEBE (MUST) exigir un flag explícito y un motivo, registrar cada uso en un log y negarse a un segundo uso.

#### Scenario: acceso sin autorización
- **WHEN** se llama `evaluar_en_reserva` sin `uso_autorizado_por_sergio=True`
- **THEN** levanta `PermissionError` y no deja rastro en el log

#### Scenario: segundo uso
- **WHEN** el log ya registra un uso
- **THEN** la función se niega

### Requirement: Particiones repetidas por caballo

Las evaluaciones de desarrollo DEBEN (MUST) usar R = 5 particiones de 4 folds por caballo (`StratifiedGroupKFold`, semillas fijas) persistidas en `comparacion/particiones.json`, sin ningún caballo en train y test del mismo fold y sin ningún caballo de la reserva.

#### Scenario: cruce de caballo
- **WHEN** un caballo aparece en train y test de una evaluación
- **THEN** `verificar_particiones` falla por aserción

### Requirement: Interfaz de candidato por ids de clip

Cada modelo candidato DEBE (MUST) implementar `ajustar(ids_train, datos)` y `predecir(ids, datos)` y DEBE (MUST) elegir hiperparámetros sólo dentro del train de cada evaluación. NO DEBE (MUST NOT) modificar el evaluador.

#### Scenario: candidato nuevo
- **WHEN** un worktree implementa un modelo
- **THEN** llama a `comparacion.evaluar` y obtiene `comparacion/resultados/<candidato>.json` con el esquema fijo

### Requirement: Métricas del evaluador

El evaluador DEBE (MUST) informar F1 macro (`average='macro'`), accuracy, precisión/recall/F1 por clase, matrices de confusión, media ± desvío sobre las 20 evaluaciones y por repetición, la métrica en el subconjunto limpio (1080×1920 y duración ≥ 5 s) y el gap train−test.

#### Scenario: reporte
- **WHEN** termina una evaluación
- **THEN** el JSON valida contra el esquema `comparacion-v1` y declara el denominador de cada tasa

### Requirement: Incertidumbre pareada por caballo

Toda diferencia entre candidatos DEBE (MUST) acompañarse de un IC95 por bootstrap pareado remuestreando caballos, y una diferencia cuyo IC incluya 0 NO DEBE (MUST NOT) presentarse como superioridad.

#### Scenario: diferencia con IC que incluye 0
- **WHEN** `comparar` devuelve `incluye_cero = true`
- **THEN** el informe la trata como empate

### Requirement: Criterio declarado antes de comparar

`comparacion/PROTOCOLO.md` DEBE (MUST) declarar candidatos, criterio de elección, condición de no inferioridad en el subconjunto limpio y regla de empate antes de que exista un resultado comparativo, y DEBE (MUST) registrar la decisión de Sergio del 07/10/2026 sobre la reserva (15 %, un solo uso, con el ganador).

#### Scenario: enmienda posterior
- **WHEN** el criterio cambia después de ver resultados
- **THEN** se agrega una enmienda fechada y los resultados previos se rotulan como anteriores

### Requirement: Geometría v2 sin romper la v1

El canal geométrico v2 DEBE (MUST) calcular las máscaras con confianza ≥ 0,50 (`CONF_MINIMA_KEYPOINT`) y tener huella de caché propia; la v1 y el modelo desplegado NO DEBEN (MUST NOT) cambiar por omisión.

#### Scenario: oreja con confianza 0,30
- **WHEN** una oreja tiene confianza 0,30
- **THEN** su máscara v2 vale 0 y su valor 0

### Requirement: Validación interna estratificada por caballo

El sistema DEBE (MUST) ofrecer una validación interna por caballo estratificada por clase con semilla fija, sin caballos en ambos lados, conservando el modo histórico para reproducibilidad.

#### Scenario: modo histórico
- **WHEN** se pide `modo="mas_clips"`
- **THEN** el corte es idéntico a `dividir_por_ejemplar`

### Requirement: Tabla de sesiones de filmación

El sistema DEBE (MUST) producir `comparacion/sesiones_de_filmacion.csv` con una fila por video, la fecha deducida con su fuente, una sesión propuesta, y las columnas `lugar` y `sesion_confirmada` VACÍAS. NO DEBE (MUST NOT) inventar fechas para los videos sin fecha deducible.

#### Scenario: video sin fecha
- **WHEN** ni el nombre ni los metadatos traen fecha
- **THEN** `fuente_fecha = ninguna` y `sesion_propuesta = sin_fecha`

