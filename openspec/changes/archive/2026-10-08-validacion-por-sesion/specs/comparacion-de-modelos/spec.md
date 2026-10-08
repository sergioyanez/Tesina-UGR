> Convención: `DEBE (MUST)` y `NO DEBE (MUST NOT)` tienen el sentido de RFC 2119.

## ADDED Requirements

### Requirement: Validación complementaria por sesión de filmación

El sistema DEBE (MUST) ofrecer un análisis complementario leave-session-out que, para cada sesión con clips de desarrollo, use como test los clips de esa sesión y como train los clips de las otras sesiones excluyendo los de todo caballo presente en la sesión de test. El análisis NO DEBE (MUST NOT) usar clips de la reserva ni llamar a `evaluar_en_reserva`, y NO DEBE (MUST NOT) modificar `equinos/comparacion.py`, `particiones.json` ni `reserva_caballos.json`.

#### Scenario: caballo en dos sesiones
- **WHEN** un caballo aparece en la sesión de test y en otra sesión
- **THEN** sus clips de la otra sesión quedan fuera del train y se cuentan como clips perdidos

#### Scenario: reserva
- **WHEN** se arma la partición
- **THEN** ningún clip de un caballo de la reserva aparece en train ni en test

### Requirement: Mapeo completo clip a sesión

Todo clip de desarrollo DEBE (MUST) tener una sesión confirmada, obtenida del vínculo clip → video → `sesion_confirmada`; la ausencia de sesión DEBE (MUST) detener el análisis por aserción.

#### Scenario: clip sin sesión
- **WHEN** un clip de desarrollo no tiene sesión
- **THEN** `plan_de_particion` falla por aserción

### Requirement: Métrica agrupada con incertidumbre por caballo

El análisis DEBE (MUST) informar como métrica principal el F1 macro agrupado de las predicciones fuera de muestra, con una predicción por clip, y DEBE (MUST) acompañarlo de IC95 bootstrap por caballo (2 000 réplicas, semilla 7) para cada candidato y para la diferencia pareada fusión − otro. Una sesión cuyo train quede sin alguna clase DEBE (MUST) saltearse y registrarse.

#### Scenario: sesión con una sola clase
- **WHEN** la sesión de test contiene una sola clase
- **THEN** sus predicciones se incluyen en el F1 agrupado y no se calcula un F1 propio de la sesión

### Requirement: El análisis no cambia la elección

El análisis NO DEBE (MUST NOT) cambiar el ganador de `comparacion/DECISION.md`; si su resultado contradice la elección, DEBE (MUST) registrarse como limitación.

#### Scenario: resultado desfavorable
- **WHEN** la fusión deja de ser primera en leave-session-out
- **THEN** se informa como limitación explícita y el ganador se mantiene
