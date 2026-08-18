## Why

Contribuye al **objetivo específico 2** (ROI facial y normalización de pose), que hoy está bloqueado:
la cobertura de alineación quedó en **59,5 %** (876 de 1472 frames), por debajo del umbral de 70 %
fijado como criterio de aceptación.

El diagnóstico de `reporte.cuello_de_botella()` (18/08/2026) ya identificó la causa y no deja lugar
a dudas: **el hocico frena 359 frames y el ojo sólo 10**, una relación de 36 a 1. La confianza
mediana del ojo es 0,891 y la del hocico 0,709. Si el hocico se resolviera por completo, la
cobertura llegaría a **83,9 %**; y para cruzar el 70 % alcanza con recuperar 155 de esos 359 frames,
un 43 % de los que el hocico bloquea. Es la primera vía identificada que llega al umbral sin
anotación manual.

La asimetría tiene un origen concreto en el código del notebook 02. El ojo se resuelve eligiendo
**el keypoint ocular más confiable en cada frame** entre todos los que expone el modelo. El hocico,
en cambio, se fija recorriendo una lista de candidatos por orden —`["nose", "snout", "muzzle",
"upper_jaw", "mouth"]`— y quedándose con el primero que exista, que resultó ser `nose`. La elección
es por posición en una lista escrita a mano, no por evidencia.

**Y hoy no se puede corregir barato**, porque el `.h5` que DeepLabCut produce —con todos los
bodyparts de todos los frames— se borra como intermedio; lo único que sobrevive es
`keypoints_faciales.csv`, ya reducido al par ojo/hocico elegido. Cada experimento de bodypart
cuesta, por lo tanto, una reinferencia completa de ~1 h. Probar `upper_jaw` a ciegas y descubrir que
tampoco alcanza serían dos reinferencias tiradas.

Esta propuesta invierte ese orden: **persistir una vez la pose completa** y convertir la elección
del bodypart en un barrido offline que cuesta segundos y se decide con datos.

## What Changes

- El notebook 02 **persiste la tabla de pose completa** —todos los bodyparts, todos los frames, con
  coordenadas y confianza— como artefacto de la etapa, en lugar de descartarla junto con el resto
  de los intermedios de DeepLabCut.
- Se agrega al paquete `equinos` un **barrido offline de candidatos a hocico** que, leyendo esa
  tabla, informa la cobertura alcanzable con cada bodypart y con la estrategia de máxima confianza
  por frame, sin reinferir.
- La selección del keypoint del hocico pasa a **decidirse por evidencia** y a quedar registrada como
  parámetro explícito de la etapa, en lugar de emerger del orden de una lista de candidatos.
- Los videos nuevos de la clase Disgustado entran en la **misma reinferencia**, aprovechando que hay
  que rehacerla de todos modos.

**No incluido, y es deliberado:** el pipeline no detecta que el dataset de entrada cambió
—`etapas.estado()` mira artefactos y cachés, no los datos—, por lo que agregar videos a Drive no
re-dispara nada. Es un defecto real y merece su propio change; acá se lo sortea a mano con
`ETAPAS_A_FORZAR`, que ya existe para eso.

## Capabilities

### New Capabilities

Ninguna. La propuesta refuerza una capacidad existente en lugar de introducir una nueva.

### Modified Capabilities

- `preprocesamiento-facial`: el requisito «Estimación de keypoints faciales» hoy exige descubrir el
  esquema del modelo en tiempo de ejecución, pero no dice nada sobre **cómo elegir** entre varios
  candidatos ni obliga a conservar la pose completa. Se agrega que la salida cruda de pose debe
  persistirse y que la elección del keypoint del hocico debe sustentarse en su confianza medida
  sobre el dataset, no en un orden de precedencia fijado a priori.

## Impact

- **Notebooks:** `02_roi_y_normalizacion.ipynb` (persistencia de la pose; el bodypart del hocico
  pasa a parámetro inyectable por papermill). `00_pipeline.ipynb` a mano, para forzar las etapas.
- **Paquete `equinos`:** módulo nuevo de barrido de bodyparts; `etapas.py` registra el artefacto
  nuevo entre los de `roi_normalizacion`.
- **Artefactos:** se agrega `resultados/pose_completa.csv`. Su tamaño crece con el número de
  bodyparts del modelo y hay que verificar que siga siendo un CSV razonable antes de darlo por
  bueno.
- **Costo de cómputo:** una reinferencia de DeepLabCut sobre todos los clips, incluidos los nuevos
  de Disgustado. Es el precio de entrada, y a cambio los experimentos posteriores de bodypart
  quedan gratis.
- **Riesgo geométrico a resolver en el diseño:** `FRAC_HOCICO = 0.74` supone un punto anatómico
  fijo. `nose` (la punta) y `upper_jaw` no están en el mismo lugar del eje de la cabeza, así que
  alternar entre ellos por frame alteraría la distancia ojo-hocico y con ella la escala del recorte,
  de forma sistemática. La estrategia de máxima confianza por frame **no puede adoptarse sin
  evaluar este efecto**, aunque sea la que use el ojo.
- **Sin impacto** sobre los objetivos 3 y 4, que siguen sin iniciarse.
