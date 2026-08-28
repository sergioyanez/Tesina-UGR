# Resultado de la validación — 28/08/2026

Ejecutado con `equinos.interpolacion.informe_validacion()` sobre la corrida de 354 clips y 3900
frames detectados (249 videos). No reinfiere DeepLabCut: lee `keypoints_faciales.csv`.

## Lo que se declaró antes de medir

Umbral de aceptación, escrito en la spec **antes** de correr nada: por cada largo de hueco, error
en píxeles dentro del recorte de 224 con **mediana ≤ 3,0 px** y **p90 ≤ 8,0 px**.

## Lo que dio

### Huecos interiores

| largo | n | mediana | p90 | máx | px original | frac. cara | ¿cumple? |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 1 | 2568 | 13,99 | 60,45 | 1673,53 | 51,3 | 0,104 | no |
| 2 | 4300 | 16,33 | 65,17 | 1560,03 | 57,7 | 0,121 | no |
| 3 | 5442 | 17,34 | 69,06 | 1780,52 | 61,4 | 0,128 | no |
| 4 | 6160 | 18,25 | 71,79 | 1156,93 | 64,6 | 0,135 | no |
| 5 | 6515 | 19,12 | 74,18 | 1863,65 | 68,7 | 0,142 | no |
| 6 | 6582 | 20,35 | 75,05 | 753,31 | 73,9 | 0,150 | no |
| 7 | 6433 | 21,23 | 77,49 | 870,22 | 75,8 | 0,158 | no |
| 8 | 6096 | 21,92 | 80,76 | 976,64 | 77,4 | 0,162 | no |

### Extremos (sostener el valor del vecino)

| largo | n | mediana | p90 | máx | ¿cumple? |
|---:|---:|---:|---:|---:|:--|
| 1 | 1740 | 16,87 | 78,97 | 3988,67 | no |
| 2 | 1204 | 22,91 | 88,20 | 4157,36 | no |
| 3 | 750 | 29,03 | 92,74 | 4228,30 | no |
| 4 | 352 | 32,78 | 109,94 | 4137,97 | no |

**Contra el umbral declarado: `MAX_HUECO_INTERPOLADO = 0` y `MAX_EXTREMO_SOSTENIDO = 0`.**
Ningún estrato lo cumple. Se informa aunque no cumpla, como exige la spec.

## Por qué ese resultado no significa lo que parece

El umbral estaba mal fundado, y la evidencia es una medición que faltaba: **cuánto se mueve el
recorte entre dos detecciones propias consecutivas.**

| | mediana | p90 |
|---|---:|---:|
| línea de base: vecino detectado (n=3107) | **16,28 px** | 78,22 px |
| interpolación de un hueco de 1 (n=2568) | **13,99 px** | 60,45 px |

La interpolación de un hueco de 1 es **más precisa que usar la detección real del frame vecino**.
El umbral de 3,0 px exigía que la estimación fuera cinco veces más exacta que el propio movimiento
del caballo entre dos frames muestreados — a 16 frames por clip de ~6 s, hay ~0,4 s entre frames, y
en ese lapso la cabeza se desplaza unos 60 px en la imagen original. Ninguna estimación puede
cumplirlo, y una que lo cumpliera estaría sobreajustando.

Leído contra la línea de base, **sólo el largo 1 queda por debajo**: los huecos de 2 o más ya
superan la variación natural del dataset.

## Sesgo conocido, medido

Los frames usados para validar son los que **sí** se detectaron, que no son una muestra aleatoria
de los que fallaron. El sesgo se puede acotar mirando el error contra la velocidad aparente entre
los vecinos:

| cuartil de movimiento | n | mediana px recorte | p90 |
|---|---:|---:|---:|
| quieto (q1) | 11021 | 8,80 | 56,28 |
| lento (q2) | 11026 | 13,30 | 65,27 |
| rápido (q3) | 11024 | 24,61 | 70,51 |
| muy rápido (q4) | 11025 | 39,91 | 93,57 |

Correlación velocidad / error en el recorte: **+0,286**.

El error **sí** crece con el movimiento, y la detección tiende a fallar justamente cuando el
caballo se mueve. Por lo tanto el error real sobre los huecos verdaderos será **mayor** que el
medido acá, y hay que decirlo en el informe. La magnitud del sesgo está acotada por la tabla: entre
el cuartil quieto y el muy rápido el error mediano se multiplica por 4,5.

## Distribución real de los huecos (354 clips)

| largo | huecos | frames | acumulado |
|---:|---:|---:|---:|
| 1 | 261 | 261 | 58 % |
| 2 | 97 | 194 | 79 % |
| 3 | 41 | 123 | 88 % |
| 4 | 22 | 88 | 93 % |
| ≥5 | 31 | 223 | 100 % |

452 huecos interiores, 889 frames dentro de ellos. La forma coincide con la que estimaba el
diseño sobre 342 clips (58 % / 80 % / 89 %).

## Lo que rescataría cada tope

Contado sobre los datos reales, no proyectado:

| tope | clips íntegros | `Disgustado` clips/caballos | `Interesado` | `Sereno` |
|---:|---:|---:|---:|---:|
| 0 (hoy) | 81 | **6 / 5** | 37 / 27 | 38 / 24 |
| 1 | 130 | **10 / 8** | 56 / 35 | 64 / 36 |
| 2 | 153 | **15 / 12** | 65 / 37 | 73 / 38 |
| 3 | 166 | **20 / 17** | 70 / 39 | 76 / 41 |

## Conclusión: la interpolación sola no alcanza

Esto corrige la expectativa con la que se planificó el change. El diseño proyectaba **180 clips
íntegros y 19 ejemplares de `Disgustado`** rellenando huecos interiores. Medido:

- con el tope que la evidencia respalda (**1**), son **130 clips y 8 ejemplares**;
- para llegar a 17 ejemplares hace falta el tope **3**, cuyo error mediano (17,34 px) supera la
  línea de base (16,28 px): ya no está dentro del ruido propio del dataset.

**Con 8 ejemplares, `Disgustado` sigue sin admitir `GroupKFold` por caballo.** La interpolación
mueve el dataset de "imposible" a "insuficiente", que es un avance real pero no el que se esperaba.
No sustituye a filmar más caballos de la clase minoritaria: los dos hacen falta.

## Decisión tomada — 28/08/2026

**El umbral se re-declaró anclándolo en la línea de base.** Un largo de hueco se acepta si su
mediana y su p90 no superan los de dos detecciones propias consecutivas (16,28 px y 78,22 px).

Resultado: **`MAX_HUECO_INTERPOLADO = 1`** y **`MAX_EXTREMO_SOSTENIDO = 0`**.

Lo que hay que poder contar en la defensa, y por eso queda escrito acá y en la spec:

1. El umbral original (3,0 / 8,0 px) se declaró **antes** de medir, como corresponde.
2. **No lo cumplió ningún largo de hueco.** Se informa: contra ese umbral, ambos topes son 0.
3. Al medir después la línea de base quedó claro que ese umbral era imposible de cumplir por
   construcción: exigía cinco veces más precisión que el propio movimiento del caballo entre dos
   frames muestreados, y la interpolación de un hueco de 1 resultó **más precisa que usar la
   detección real del frame vecino** (13,99 contra 16,28 px).
4. El criterio se cambió de forma explícita, con su motivo, y el resultado se informa contra
   ambos umbrales. `informe_validacion()` imprime las dos columnas (`vigente/histórico`) en cada
   corrida, así que la comparación no depende de que alguien recuerde este documento.

La lección de método, que vale más que el número: **declarar un umbral antes de medir protege
contra elegir el que conviene, pero no contra elegir uno sin referencia física.** Hacen falta las
dos cosas — declararlo antes, y anclarlo en una línea de base medible.
