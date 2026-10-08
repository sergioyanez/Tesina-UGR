## Why

La auditoría del pipeline encontró cuatro problemas que impiden comparar modelos con honestidad: la inferencia lee los videos distinto que la preparación del dataset (orientación, recodificación `mp4v`); el canal geométrico del MIL es casi vacío (máscaras con `confianza > 0` y ojo+hocico con un solo grado de libertad); la validación interna usa los siete caballos con más clips; y no existe un protocolo común, ni reserva, ni particiones repetidas para comparar candidatos distintos. Antes de medir modelos en worktrees separados hace falta una base común.

Contribuye al **objetivo específico 3** (clasificación y evaluación) y al **objetivo 4** (interfaz de inferencia coherente con lo evaluado).

## What Changes

- `equinos/video.py` (nuevo) y `equinos/inferencia.py`: la inferencia normaliza el video como el notebook 01 (`CAP_PROP_ORIENTATION_AUTO = 1` y recodificación `mp4v`) antes de la pose y de los recortes. `normalizar=False` conserva la lectura anterior para medir.
- `equinos/geometria.py` (nuevo): canal geométrico **v2** (17 columnas, marco de la cabeza, orejas/mandíbula/boca/interocular/escorzo, máscaras con umbral 0,50) con huella propia. `MultimodalEquineDataset(geometria="v1"|"v2", umbral_mascara=...)`. La v1 y el modelo desplegado NO cambian por omisión.
- `equinos/particion.py`: `dividir_estratificada` (StratifiedGroupKFold por caballo, semilla fija) y `dividir_validacion(modo=...)`; `entrenar_mil_multimodal(validacion_interna=..., geometria=...)`. El modo histórico `mas_clips` sigue siendo el valor por omisión.
- `equinos/comparacion.py` (nuevo) y `comparacion/`: reserva del 15 % de los caballos, particiones 5×4, interfaz de candidato, evaluador con esquema fijo, comparación bootstrap pareada por caballo, `PROTOCOLO.md`, baseline de humo y tabla de sesiones de filmación.
- Tests nuevos: rotación en inferencia, geometría v2 y máscaras, comparación (reserva, particiones, validación interna).

## Capabilities

### New Capabilities
- `comparacion-de-modelos`: reserva, particiones repetidas, interfaz de candidato, evaluador y comparación pareada.
- `coherencia-entrenamiento-inferencia`: la inferencia lee el video como la preparación del dataset.

### Modified Capabilities
<!-- Los requisitos de `clasificacion-de-expresiones` no cambian: la v1 y la validación histórica siguen siendo el valor por omisión. -->

## Impact

- Código: `equinos/{video,geometria,comparacion}.py`, `equinos/{inferencia,clasificacion,particion}.py`, `comparacion/`, `tests/`.
- El modelo desplegado `mil_multimodal_produccion.pt` NO se re-entrena ni se toca; sí cambia el preprocesamiento de la inferencia (ahora coherente con el entrenamiento), así que sus predicciones sobre un video pueden diferir de las anteriores.
- Sin dependencias nuevas. Los datos se leen de `datos_equinos/` sin escribirlos.
