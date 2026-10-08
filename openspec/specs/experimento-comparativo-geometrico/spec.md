# experimento-comparativo-geometrico Specification

## Purpose
TBD - created by archiving change experimento-comparativo-simba. Update Purpose after archive.
## Requirements
### Requirement: Honestidad de nombre del extractor

Toda salida del experimento que use features calculadas con código propio DEBE (MUST) rotularlas «features tipo SimBA calculadas con código propio» y NO DEBE (MUST NOT) llamarlas SimBA.

#### Scenario: features propias
- **WHEN** se reporta un resultado con `features_simba.py`
- **THEN** el rótulo aclara que es código propio y no SimBA

### Requirement: Mismos folds que el modelo vigente

El experimento DEBE (MUST) usar la partición por caballo de `particion_clips.json` y DEBE (MUST) verificar con asserts que los conjuntos de test coinciden clip a clip con los del MIL y que ningún caballo está en train y test.

#### Scenario: fold distinto
- **WHEN** la partición difiere de la del MIL
- **THEN** la evaluación se detiene con un error

### Requirement: Sin fuga en normalización e imputación

Toda imputación o estandarización ajustada DEBE (MUST) ajustarse solo con el train de cada fold. Los hiperparámetros NO DEBEN (MUST NOT) elegirse con el test.

#### Scenario: valores ausentes
- **WHEN** un keypoint tiene confianza menor que 0,50
- **THEN** se trata como ausente y se imputa con estadísticos del train del fold

### Requirement: Denominador y controles de confusores

Toda tasa DEBE (MUST) indicar el conjunto sobre el que se calculó, y el experimento DEBE (MUST) reportar el subconjunto limpio (1080x1920 y duración >= 5 s) junto con un control de filmación.

#### Scenario: reporte
- **WHEN** se publica un F1 macro
- **THEN** indica 359 clips (original) o 236 clips (limpio)

### Requirement: Incertidumbre por caballo

Las diferencias entre modelos DEBEN (MUST) acompañarse de un IC95 por bootstrap remuestreando caballos, y una diferencia cuyo IC incluya 0 NO DEBE (MUST NOT) presentarse como superioridad.

#### Scenario: limpio
- **WHEN** el IC95 de la diferencia incluye 0
- **THEN** se informa como no concluyente

### Requirement: Aislamiento

El experimento NO DEBE (MUST NOT) modificar `equinos/` ni escribir en `datos_equinos/` del repo principal.

#### Scenario: ejecución
- **WHEN** se corre `correr_todo.sh`
- **THEN** solo se escriben archivos bajo `experimentos/simba/resultados/`

