## Context

Datos en `datos_equinos/` (gitignored), leídos en solo lectura. 359 clips, 133 caballos, 4 folds por caballo.

## Goals / Non-Goals

- Goals: cuantificar la señal geométrica; comparar contra baseline estático y MIL en los mismos folds.
- Non-Goals: reemplazar el MIL; re-entrenar el MIL; afinar hiperparámetros; claims causales.

## Decisions

- Features propias en vez de SimBA: `UserDefinedFeatureExtractor` (1 animal) solo da movimiento por bodypart; se corrió igual (B2) y se reporta aparte. Todo rótulo dice «tipo SimBA, código propio».
- Escala por clip (mediana ojo-hocico), no por frame, para conservar la dinámica del escorzo.
- NaN: confianza < 0,50 = ausente; imputación por mediana dentro del fold + features de disponibilidad.
- RF con parámetros fijos; LogReg igual al baseline vigente. Sin búsqueda.
- MIL tomado de `objetivo3_evaluacion_v2.json` (predicciones fuera de muestra), no re-entrenado.
- Bootstrap por caballo sobre F1 agrupado, diferencia pareada.

## Risks / Trade-offs

- Punto de vista y duración pueden confundir; se mitiga con el subconjunto limpio y controles (post hoc, declarados).
- Un solo observador de etiquetas.
