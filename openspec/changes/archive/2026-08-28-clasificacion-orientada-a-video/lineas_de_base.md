# Líneas de base — resultados

Corrida el **28/08/2026** sobre la partición persistida en `resultados/particion_clips.json`.
Reproducible con `lineas_de_base.py`. 141 clips íntegros, 73 ejemplares, reparto
`Disgustado` 15 % / `Interesado` 40 % / `Sereno` 45 %.

Las tres líneas de base del diseño, corridas sobre los mismos folds. Reproducible: dos corridas
seguidas dan los mismos números hasta el tercer decimal.

## La partición

`GroupKFold` por `id_caballo`, 4 folds, sin fuga verificada. Ninguna clase queda vacía en ningún
fold; el fold más chico tiene 4 clips de `Disgustado` en test.

| fold | clips | ejemplares | Disgustado | Interesado | Sereno |
|---:|---:|---:|---:|---:|---:|
| 0 | 36 | 18 | 4 | 18 | 14 |
| 1 | 35 | 18 | 5 | 14 | 16 |
| 2 | 35 | 18 | 7 | 10 | 18 |
| 3 | 35 | 19 | 5 | 14 | 16 |

`GroupKFold` no lleva `random_state` y no lo necesita: es determinista. La reproducibilidad
depende de que la entrada llegue siempre en el mismo orden, y por eso el reparto ordena los
ejemplares y desempata por nombre. Está probado en `tests/test_particion.py`.

## Resultados, protocolo real

| Línea de base | F1 macro | Balanced accuracy | Recall `Disgustado` |
|---|---:|---:|---:|
| 1. Clase mayoritaria | 0,188 ± 0,029 | 0,333 ± 0,000 | 0,000 |
| 2. Sólo `id_caballo` | 0,188 ± 0,029 | 0,333 ± 0,000 | 0,000 |
| **3. Un frame + voto, sin orden** | **0,626 ± 0,065** | 0,638 ± 0,069 | 0,346 ± 0,131 |

La tercera usa el **mismo backbone congelado** que va a usar el modelo —ResNet-18 de ImageNet,
512 dimensiones por frame— con una cabeza lineal encima y pesos por clase balanceados. Usa los 16
frames pero es **invariante a su orden**: barajar el clip no cambia una sola predicción. Por eso
aísla exactamente lo que el modelo secuencial tiene que aportar, y no cualquier otra ventaja del
backbone o de disponer de más frames.

**Las imágenes tienen señal, y bastante.** De 0,188 a 0,626 sin usar información temporal, con
partición por ejemplar. Es el resultado más importante de esta etapa y era necesario conocerlo
antes de elegir arquitectura.

### Las dos primeras

**Dan exactamente lo mismo, y era inevitable.** Con partición por ejemplar, un caballo de
test nunca se vio en entrenamiento: el modelo de identidad **cayó al voto mayoritario en los 141
clips, el 100 %**. No tenía con qué predecir.

Eso no es un defecto de la medición. **Es la demostración de que la partición hace lo que dice
hacer:** neutraliza por completo el confounder de individuo, al menos frente a un modelo que sólo
mira la identidad.

## El número que sí mide el confounder

La pregunta que la línea de base de identidad quería contestar —*¿la clase está determinada por el
individuo?*— no se puede contestar con un protocolo que impide que la identidad transfiera. Así que
se midió también con una **partición ingenua por clip**, la que haría alguien que no agrupa:

| | F1 macro | Balanced accuracy |
|---|---:|---:|
| Sólo `id_caballo`, partición **por ejemplar** | 0,188 ± 0,029 | 0,333 |
| Sólo `id_caballo`, partición **ingenua por clip** | **0,526 ± 0,116** | 0,585 |

**Sin agrupar por caballo se llega a F1 macro 0,53 sin mirar un solo píxel.**

La causa es una propiedad del dataset, no del método: **63 de los 73 ejemplares aparecen en una
sola clase**. Para esos, saber qué caballo es equivale a saber la clase. Sólo 10 aparecen en más de
una (`37, 54, 55, 178, 218, 230, 237, 906, 907, 910`).

Esa diferencia de **+0,338** es la fuga que la partición por ejemplar elimina. Es también la
respuesta cuantificada a por qué ningún número de este trabajo se mide sobre cortes por clip: un
resultado de 0,50 bajo partición ingenua sería **peor** que ignorar las imágenes por completo.

> Este número se informa como diagnóstico del dataset. **No se usa para reportar ningún resultado
> del trabajo**, y el script lo dice en su encabezado para que no se cite fuera de contexto.

## Criterio de aceptación, declarado antes de entrenar (tarea 2.4)

Sale de la Decisión 7 del diseño, que lo definió antes de tener estos números: *el modelo se
considera útil si su F1 macro medio supera al de las tres líneas de base en al menos un desvío
estándar entre folds*. Anclado en las líneas de base y no en un número redondo, porque un 0,60
puede ser excelente o irrelevante según contra qué se lea.

La línea de base más alta es la de un frame: **0,626 ± 0,065**. Aplicando la regla:

> **El modelo secuencial se considera útil si su F1 macro medio supera 0,691.**

Es una vara exigente, y lo es a propósito: la hipótesis del trabajo dice que la evolución temporal
es lo que permite clasificar. Si el modelo secuencial no le gana a un clasificador que ignora el
orden de los frames, **la hipótesis no queda sostenida por la evidencia de este dataset**, y eso
hay que escribirlo tal cual. El resultado se informa aunque no se cumpla.

**Qué NO forma parte del criterio, y por qué se informa igual.** El recall de `Disgustado` de la
línea de base de un frame es **0,346**. Un modelo puede subir el F1 macro y a la vez empeorar la
clase de interés clínico. Ese número acompaña todo resultado que se reporte, pero no se convierte
en condición: el diseño no lo declaró como tal, y agregarlo ahora que se conocen los números sería
exactamente lo que este proyecto evita.

## Qué más queda declarado

- **La accuracy global no se usa.** Con este reparto llega a 0,45 prediciendo siempre `Sereno`.
- **La partición está persistida y es obligatoria.** Todas las corridas —las variantes de cabeza
  temporal, la ablación por `origen_alineacion`— tienen que leer `particion_clips.json`. Comparar
  números medidos sobre cortes distintos no dice nada.
- **Las características del backbone congelado están cacheadas** (`caracteristicas_backbone.npz`,
  141 × 16 × 512). Comparar cabezas temporales cuesta segundos. **Pero con la caché no se puede
  aumentar**: la aumentación cambia los píxeles y obligaría a repasar el backbone en cada época.
  Si el entrenamiento final usa aumentación, tiene que re-extraer, y no se puede comparar un
  modelo aumentado contra cabezas medidas sin aumentar.
