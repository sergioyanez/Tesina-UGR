## Why

La comparación de modelos (`comparacion/DECISION.md`, 08/10/2026) eligió, con un protocolo declarado antes de medir, la **fusión tardía pose + visual** como ganadora: F1 macro 0,868 en desarrollo y 0,834 en la reserva de 20 caballos (único uso), contra 0,722 del MIL multimodal que hoy sirve la app. Sergio decidió que la fusión reemplace al MIL en la app. Hoy la app y `equinos.inferencia` sólo saben cargar el MIL (`mil_multimodal_produccion.pt`) y mostrar su confiabilidad (`objetivo3_evaluacion_v2.json`), que ya no describe al modelo recomendado.

Contribuye al **objetivo específico 4** (aplicación o interfaz para el uso del modelo resultante), apoyado en el resultado del objetivo 3.

## What Changes

- Script `comparacion/entrenar_modelo_final.py`: ajusta `FusionPoseVisual` con **todos** los clips (desarrollo + reserva) y persiste un paquete `joblib` con los dos modelos sklearn, el peso `w` y metadatos (versión de sklearn, huella del dataset, fecha, lista de clips). No usa `evaluar_en_reserva`.
- Módulo nuevo `equinos/fusion.py`: carga del paquete, matriz de pose `(16, 39, 3)` desde las filas de pose del pipeline, las 160 features con `vector_del_clip`, probabilidades fusionadas y confiabilidad leída de los resultados de la comparación.
- `equinos/inferencia.py`: ruta de predicción con la fusión a partir de lo que ya calcula el pipeline; la fusión es el modelo **por omisión** y el MIL queda como opción explícita (`modelo="mil"`). La salida conserva la interfaz (`clase`, `probabilidades`, `precision`, ...).
- `app_inferencia.py`: textos y métricas del modelo vigente; el gráfico de atención del MIL sólo aparece si el modelo es el MIL; con la fusión se muestran las probabilidades de cada rama (pose y visual).
- Pruebas: equivalencia entrenamiento ↔ inferencia de las features y de la predicción, y contrato de la fusión.
- **BREAKING**: `clasificar_video` ya no usa el MIL salvo pedido explícito. La confiabilidad mostrada deja de ser la del MIL.

## Capabilities

### New Capabilities
- `modelo-fusion-en-la-app`: el modelo final de fusión, su entrenamiento con todos los clips, su persistencia con metadatos, su uso por omisión en la inferencia y la confiabilidad que acompaña a cada predicción.

### Modified Capabilities
- `coherencia-entrenamiento-inferencia`: se agrega el requisito de que las features de pose de la inferencia coincidan con las del entrenamiento.

## Impact

- Código: `equinos/fusion.py` (nuevo), `equinos/inferencia.py`, `app_inferencia.py`, `comparacion/entrenar_modelo_final.py` (nuevo), `tests/`.
- Dependencias de ejecución nuevas en la app: `scikit-learn` y `joblib` (ya están en el venv), y `experimentos/simba/features_simba.py` como fuente única de las features.
- Artefacto fuera del repo: `<raíz>/modelos/fusion_pose_visual/fusion_pose_visual.joblib` (la raíz de datos está ignorada por git).
- Documentación: `README.md` y `openspec/project.md`.
- Sin cambios en el preprocesamiento ni en el protocolo de comparación; la reserva no se vuelve a usar.
