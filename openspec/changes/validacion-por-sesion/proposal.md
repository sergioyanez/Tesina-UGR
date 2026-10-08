## Why

La principal salvedad del ganador (fusión tardía pose + visual) es el confusor de filmación: las
features de pose predicen la duración del clip y los candidatos caen entre 10 y 13 puntos en el
subconjunto limpio. El protocolo agrupa por caballo, no por sesión, y dejó el leave-session-out
«habilitado, sin ejecutar». Hace falta saber si el ranking y el nivel de acierto se sostienen cuando
el modelo se prueba con sesiones de filmación que no vio. Contribuye al objetivo específico 4
(evaluación rigurosa y análisis de confusores).

## What Changes

- Módulo nuevo `comparacion/validacion_por_sesion.py`: mapea clip → video → `sesion_confirmada`, arma una
  partición leave-session-out con exclusión de los caballos presentes en la sesión de test, corre los
  cuatro candidatos con exactamente la misma partición y calcula F1 macro agrupado con IC95 bootstrap
  por caballo (y la diferencia pareada fusión − otro).
- Resultados en `comparacion/resultados/validacion_por_sesion.json`.
- Test autónomo `tests/test_validacion_por_sesion.py`.
- Enmienda fechada (08/10/2026) a `comparacion/PROTOCOLO.md`, escrita ANTES de correr, y sección de
  interpretación en `comparacion/DECISION.md` y `docs/caminos_explorados.md`.
- Es un análisis complementario: NO cambia el ganador y NO toca la reserva ni `equinos/comparacion.py`.

## Capabilities

### New Capabilities
(ninguna)

### Modified Capabilities
- `comparacion-de-modelos`: se agrega el requisito de un análisis complementario leave-session-out que respete la reserva y la exclusión por caballo.

## Impact

Archivos nuevos en `comparacion/` y `tests/`; cambios de documentación. No se modifican `equinos/`,
`particiones.json` ni `reserva_caballos.json`. No se usa `evaluar_en_reserva`.
