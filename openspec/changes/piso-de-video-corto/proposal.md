## Why

Contribuye al **objetivo específico 1** (construcción del dataset) y ataca de frente el confounder
que había que resolver antes de entrenar.

El plan de trabajo fija clips de 5 a 7 segundos, y `DURACION_MINIMA = 5.0` implementa ese piso: un
video más corto no produce ningún clip. Medido sobre el catálogo completo de 234 videos, ese piso
**no estaba filtrando tomas malas: estaba filtrando ejemplares**.

| | Disgustado | Interesado | Sereno |
|---|---:|---:|---:|
| caballos en el catálogo | **42** | 60 | 54 |
| caballos que llegan al dataset con piso de 5 s | **11** | 49 | 47 |

`Disgustado` es la clase que limita todo, su video mediano dura **2,9 s**, y 55 de sus 70 videos no
alcanzaban el piso: 31 de sus 42 ejemplares quedaban afuera enteros. Como el ejemplar está
correlacionado con la clase, sostener el piso no cuidaba la calidad del dato — alimentaba
exactamente el confounder que hace que un clasificador pueda aprender a identificar caballos en vez
de expresiones. El motivo original del plan B de 2 clases ("`Disgustado` viene de un solo caballo")
estaba sostenido por el piso, no por el material.

El costo tampoco era el que se suponía. DeepLabCut infiere el clip entero, así que la etapa 02
escala con **segundos de material**, no con cantidad de clips: pasar de 280 a 342 clips mueve la
corrida de ~2,5 h a ~2,7 h.

## What Changes

- Parámetro nuevo en el notebook 01: **`PISO_VIDEO_CORTO = 2.0`**. Un video más corto que un clip
  entra **entero** como un único clip si dura al menos ese piso.
- La regla se aplica **sólo cuando no salió ninguna ventana**. Para los videos que llegan al piso de
  5 s no cambia absolutamente nada: el diseño de 6 s del plan de trabajo sigue rigiendo para todo lo
  que puede cumplirlo, y los clips ya generados no se mueven.
- `PISO_VIDEO_CORTO` entra en `CLAVES_SEGMENTACION`, así que el checkpoint de la etapa 01 se
  invalida solo al cambiarlo.
- El dataset pasa a tener **clips de largo variable (2 a 7 s)**: 16 frames que abarcan distinto
  tiempo según el clip. Queda auditable en la columna `duracion_seg` de `inventario_clips.csv`.

Resultado medido:

| | antes | después |
|---|---:|---:|
| clips / frames | 280 / 4480 | **342 / 5472** |
| frames de `Disgustado` | 432 (9,6 %) | **992 (18,1 %)** |
| ejemplares de `Disgustado` | 11 de 42 | **32 de 42** |
| ejemplares de `Disgustado` con ≥16 frames alineados | 4 | **12** |
| cobertura de `Disgustado` | 57,6 % | **62,1 %** |
| cobertura global | 70,2 % | 69,9 % |

## Capabilities

### New Capabilities

Ninguna.

### Modified Capabilities

- `persistencia-artefactos`: es donde `catalogo-de-videos-por-clase` puso la composición del
  dataset ("Composición declarada del dataset por categoría"). Se agrega ahí el requisito de que la
  duración de un video no lo excluya del dataset por sí sola, que es la otra mitad de la misma
  decisión: qué material entra.

## Impact

- **Notebook 01** (`01_preparar_dataset_equinos.ipynb`): celda de parámetros, `calcular_ventanas` y
  `CLAVES_SEGMENTACION`.
- **Requiere re-ejecutar las dos etapas.** La clave nueva invalida el checkpoint de la 01, que
  regenera los 342 clips. La 02 sólo infiere los 62 nuevos, y eso está **verificado, no supuesto**:
  los 280 clips previos se regeneraron con `frame_inicio` y `frame_fin` idénticos.
- **La cobertura global deja de ser un criterio útil.** Bajó de 70,2 % a 69,9 % mientras ninguna
  clase empeoraba de verdad (`Disgustado` +4,5 pp, `Interesado` +0,1, `Sereno` −0,3): el global cae
  sólo porque la mezcla se corrió hacia la clase que peor alinea. Comparar el global antes y después
  compara dos datasets distintos. Reportar cobertura **por clase**, y para el confounder, cuántos
  ejemplares quedan con material alineado suficiente.
- **Los clips cortos no son material de descarte:** alinean al 68,5 % contra 70,2 % de los largos, y
  en `Disgustado` alinean mejor (65,5 %) que los propios clips largos de la clase.
