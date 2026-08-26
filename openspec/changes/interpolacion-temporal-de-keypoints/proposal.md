## Why

Contribuye al **objetivo específico 2** y desbloquea el 3: hoy el dataset no se puede entrenar de
forma honesta, y la causa no es el material sino el hueco de un frame acá y otro allá.

La cobertura de alineación se venía midiendo por frame — 69,9 % sobre 5472 frames. **Esa no es la
unidad que consume el modelo.** Un clasificador orientado a video consume clips de 16 frames, y un
clip donde 11 frames están alineados por keypoints y 5 son recortes de caja sin alinear mete
adentro de una sola muestra exactamente el confounder que este preprocesamiento venía a eliminar.

Contado por clip, el dataset disponible es mucho más chico de lo que parecía:

| | clips | caballos | `Disgustado` |
|---|---:|---:|---|
| clips con los 16 frames alineados | **80** de 342 | 49 | **5 clips, 4 caballos** |

Con 4 ejemplares en `Disgustado`, un `GroupKFold` por caballo deja folds sin la clase: **entrenar
sobre el subconjunto limpio no es posible hoy**.

Pero los frames que fallan **no están dispersos al azar**. De los 440 huecos interiores, 350
—el 80 %— son de uno o dos frames consecutivos, rodeados de frames que sí se alinearon. La cabeza
del caballo no se teletransporta entre dos frames de un clip de 2 a 7 segundos: la posición del ojo
y del hocico en un frame fallido está acotada por la de sus vecinos.

Si se rellenan esos huecos, el dataset entrenable cambia de escala **sin filmar nada y sin volver a
inferir**:

| | clips completos | caballos | `Disgustado` |
|---|---:|---:|---|
| hoy | 80 | 49 | 5 clips / 4 caballos |
| rellenando sólo huecos interiores | **180** | 84 | **24 clips / 19 caballos** |
| rellenando además los extremos | 330 de 342 | 113 | 61 clips / 32 caballos |

El pipeline **ya hace esto para las cajas del detector**: `suavizar_cajas()` y
`rellenar_faltantes()` existen desde el principio y nadie las discute. Lo que falta es el análogo
para los dos keypoints de los que sale la transformada.

## What Changes

- Los keypoints faltantes de un clip se **rellenan por interpolación temporal** entre los frames
  vecinos que sí se detectaron, y la transformada de similitud se construye después, como siempre,
  con `similitud_desde_dos_puntos()`. **Se interpolan los puntos, no la matriz.**
- La interpolación tiene un **tope de hueco declarado** (`MAX_HUECO_INTERPOLADO`): más allá de esa
  cantidad de frames consecutivos sin detección, el frame sigue cayendo al recorte de caja.
- Los extremos del clip son un caso aparte y más débil —no hay dos vecinos que acoten, sólo uno—,
  con su propio tope (`MAX_EXTREMO_SOSTENIDO`) y su propia validación.
- Los artefactos ganan una columna **`origen_alineacion`** ∈ {`keypoints`, `interpolada`,
  `extremo_sostenido`, `caja`}. `alineado` sigue existiendo, pero deja de ser suficiente: hay que
  poder separar en cualquier métrica, y ablacionar, los frames rescatados.
- Se agrega una **validación que puede fallar**: se enmascaran frames que **sí** se alinearon, se
  interpola su posición desde los vecinos y se mide el error contra la posición real, en píxeles.
  Con un umbral de aceptación declarado **antes** de medir.

## Capabilities

### New Capabilities

Ninguna.

### Modified Capabilities

- `preprocesamiento-facial`: el requisito "Suavizado temporal de las detecciones" cubre hoy las
  cajas del detector y no los keypoints, y "Normalización de pose por transformada de similitud"
  supone que los dos puntos vienen de la detección de ese frame. Se agrega el relleno temporal de
  los keypoints con su tope, su validación y la trazabilidad de cómo se alineó cada frame.

## Impact

- **Notebook 02** (`02_roi_y_normalizacion.ipynb`): función de relleno nueva junto a
  `rellenar_faltantes()`, dos parámetros nuevos, y la columna `origen_alineacion` en
  `dataset_frames_alineados.csv`.
- **`equinos/reporte.py`**: la cobertura tiene que poder informarse separando frames alineados por
  detección de frames rescatados.
- **No requiere reinferir DeepLabCut.** Los keypoints ya están en `keypoints_faciales.csv` y
  `pose_completa.csv`; se re-derivan los recortes, que es el camino de ~14 minutos.
- **Cambia el significado de la cobertura publicada.** Todo número anterior queda medido con otro
  criterio de "alineado" y no es comparable sin desglosar por `origen_alineacion`.
- **Riesgo asumido y acotado:** un frame mal interpolado produce un recorte descentrado que nada
  detecta. Por eso la validación es parte del change y no un paso posterior.
