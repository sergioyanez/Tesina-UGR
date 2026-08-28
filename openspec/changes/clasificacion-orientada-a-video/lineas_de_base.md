# Líneas de base — resultados

Corrida el **28/08/2026** sobre la partición persistida en `resultados/particion_clips.json`.
Reproducible con `lineas_de_base.py`. 141 clips íntegros, 73 ejemplares, reparto
`Disgustado` 15 % / `Interesado` 40 % / `Sereno` 45 %.

Estas dos líneas de base **no miran la imagen**. La tercera —un solo frame con el backbone
congelado, tarea 2.3— necesita el modelo y va aparte.

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

**Las dos dan exactamente lo mismo, y era inevitable.** Con partición por ejemplar, un caballo de
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

## Qué queda declarado a partir de acá

- **El piso es F1 macro 0,188 ± 0,029.** Cualquier modelo se lee contra ese número, no contra cero
  y no contra la accuracy global —que con este reparto llega a 0,45 prediciendo siempre `Sereno`—.
- **El recall de `Disgustado` de las dos líneas de base es 0,000.** Es el número a vigilar: un
  modelo puede subir el F1 macro sin predecir nunca la clase de interés clínico.
- **La partición está persistida y es obligatoria.** Todas las corridas —las variantes de cabeza
  temporal, la ablación por `origen_alineacion`— tienen que leer `particion_clips.json`. Comparar
  números medidos sobre cortes distintos no dice nada.

Falta la tercera línea de base (tarea 2.3, un solo frame sin información temporal) antes de
declarar el criterio de aceptación en la tarea 2.4. Es la que pone a prueba la hipótesis del
trabajo, y con toda probabilidad la que va a fijar el piso real.
