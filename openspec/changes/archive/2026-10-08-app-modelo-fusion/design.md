## Context

La app corre `preparar_clip` (video → 16 recortes alineados + pose de los 16 frames con `orden` renumerado), extrae 16 vectores ResNet-18 y los pasa al MIL. La fusión ganadora (`comparacion/candidato_fusion.py`) necesita, por clip: (a) la media de los 16 vectores visuales; (b) las 160 features de `vector_del_clip(P, frames, fps)` con `P` de forma `(16, 39, 3)`, los índices de fotograma de los 16 frames y el `fps`. El pipeline ya produce todo eso: `info["ordenes"]` y `info["fps"]` de `preparar_clip`, y la pose por posición.

## Goals / Non-Goals

**Goals:** servir la fusión por omisión con la misma interfaz de salida; que las features de la inferencia sean idénticas a las del entrenamiento; trazabilidad del artefacto; confiabilidad que corresponda al modelo servido.

**Non-Goals:** reentrenar o reevaluar la comparación; tocar la reserva; borrar el MIL; mejorar el preprocesamiento.

## Decisions

1. **Paquete joblib con objetos sklearn, no la clase `FusionPoseVisual`.** Se persiste `{modelo_pose, modelo_visual, peso, clases, metadatos}`. Pickle de la clase obligaría a que `candidato_fusion` sea importable con el mismo nombre de módulo en la app. Con objetos sklearn sólo se exige la versión de sklearn, que queda registrada y se verifica al cargar (advertencia si difiere).
2. **Entrenamiento con todos los clips (359).** El protocolo permite el modelo final con todo el material una vez usada la reserva. `w` se vuelve a elegir por la validación interna por caballo de `FusionPoseVisual.ajustar` (no se fija el 0,75 a mano). El script no llama a `evaluar_en_reserva`; la métrica del modelo final NO se calcula sobre los datos con que se entrenó.
3. **Ubicación: `rutas.carpeta("modelos")/fusion_pose_visual/`.** Donde ya viven los pesos del MIL (`modelos/objetivo3_v2`). Queda fuera del repo (`datos_equinos/*` está ignorado); la ruta se documenta en el README.
4. **Features por una única función.** `equinos/fusion.py` arma `P` a partir de las filas de pose con la misma convención que `candidato_fusion.features_pose` y llama a `features_simba.vector_del_clip`; la prueba de equivalencia compara ambas rutas sobre clips del dataset.
5. **Confiabilidad leída, no escrita.** Las precisiones por clase salen de las predicciones fuera de muestra de desarrollo (`comparacion/resultados/fusion_tardia_pose_visual.json`, suma de 5 repeticiones, conteos divididos por R) y se informa además el F1 de la reserva (`..._reserva.json`). Ningún número a mano.
6. **Selección del modelo: parámetro `modelo` ("fusion" por omisión, "mil").** `clasificar_video(..., modelo="fusion")`. El MIL conserva su ruta intacta (`predecir_desde_caracteristicas`).
7. **`Prediccion` conserva sus campos** y suma `modelo` y `detalle` (con valores por omisión). Con la fusión, `atencion` y `clases_por_frame` van vacíos y `detalle` lleva las probabilidades por rama y el peso. La app dibuja la atención sólo si hay.

## Alternativas descartadas

- **Mantener el MIL.** Quedó tercero (0,722 contra 0,868; IC95 de la diferencia [+0,100; +0,193]) y las tres correcciones no lo mejoraron. Se conserva como opción explícita sólo para reproducir lo anterior.
- **Servir sólo el RF de pose.** Es más simple (sin ResNet en inferencia), pero la fusión le gana en desarrollo (+0,026, IC95 [+0,002; +0,051]) y el criterio declarado no permitió aplicar la regla de simplicidad. La rama visual ya se calcula de todos modos.
- **Persistir la instancia de `FusionPoseVisual` con pickle.** Acopla el artefacto al nombre de módulo y a `Datos`; el paquete de objetos sklearn es más portable.
- **Reutilizar `w = 0,75` fijo.** Es lo que casi siempre eligió la validación interna, pero fijarlo a mano esconde la regla; se aplica la misma regla que se evaluó.
- **Copiar `vector_del_clip` a `equinos/`.** Duplicaría la fuente de las features y permitiría que divergieran en silencio.

## Risks / Trade-offs

- **Confusor de filmación**: las features de pose predicen la duración del clip (auditoría, R² 0,53); en el subconjunto limpio la fusión baja a 0,740 (desarrollo) y 0,767 (reserva). La app lo declara.
- **El peso de la pose domina** (0,75 en 19 de 20 folds): la rama visual aporta poco; se informa.
- **Dependencia de `experimentos/simba`** desde `equinos/`: se aísla en una función con importación perezosa y se documenta; `datos.py` de ese paquete crea `experimentos/simba/resultados/` al importarse.
- **Confiabilidad de desarrollo con el modelo final entrenado con más datos**: se declara que las cifras son las de la validación, no las del modelo final.
- **Versión de sklearn**: si difiere de la del entrenamiento se avisa; el paquete se regenera con el script.
