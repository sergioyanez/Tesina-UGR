# Diseño — interpolación temporal de keypoints

## Contexto

La restricción dejó de ser cuántos frames se alinean y pasó a ser **cuántos clips quedan íntegros**.
Un clip mezclado no es una muestra a medias: es una muestra contaminada, porque la mitad de sus
frames tiene el aspecto de un recorte alineado y la otra mitad el de un recorte de caja, y a qué
frame le toca cuál está correlacionado con el ejemplar, que a su vez predice la clase.

Eso fija la prioridad del diseño — **rescatar frames sólo mientras se pueda demostrar que el
rescate es correcto**. Un frame mal interpolado no falla: produce un recorte descentrado que entra
al dataset como si fuera bueno. Es peor que dejarlo afuera.

## Decisión 1 — Interpolar los keypoints, no la matriz

Se interpolan las cuatro coordenadas —ojo y hocico— y la transformada se construye después con
`similitud_desde_dos_puntos()`, igual que para cualquier otro frame.

**Alternativa descartada — interpolar la matriz de similitud.** Es viable (el conjunto de matrices
de similitud es cerrado bajo combinación lineal en la parametrización `a, b, tx, ty`), pero tiene
dos desventajas: el error queda expresado en unidades sin interpretación física, y habría dos
caminos distintos para construir un recorte. Interpolando los puntos hay **un solo camino** y el
error se mide en píxeles sobre la imagen original, que es lo que se puede validar y contar en el
informe.

## Decisión 2 — Tope de hueco declarado, no relleno indiscriminado

`MAX_HUECO_INTERPOLADO` acota cuántos frames consecutivos sin detección se pueden rellenar. La
distribución medida sobre la corrida actual:

| largo del hueco interior | huecos | acumulado |
|---:|---:|---:|
| 1 frame | 255 | 58 % |
| 2 frames | 95 | **80 %** |
| 3 frames | 39 | 89 % |
| 4 frames | 20 | 93 % |
| ≥5 frames | 31 | 100 % |

Los 16 frames se muestrean uniformemente a lo largo del clip, así que en un clip de 6 s dos frames
consecutivos están a ~0,4 s. Un hueco de 2 son ~0,8 s de movimiento no observado; uno de 10, más de
4 segundos, y ahí interpolar es inventar. El valor inicial propuesto es **2**, que cubre el 80 % de
los huecos, y la validación es la que decide si se puede subir.

**Alternativa descartada — rellenar cualquier hueco.** Rescata 100 clips más y destruye la
posibilidad de defender el dataset: no hay forma de sostener que se sabe dónde estaba el hocico
cuatro segundos después de la última vez que se lo vio.

## Decisión 3 — Los extremos son un mecanismo aparte

Un hueco al principio o al final del clip no está acotado por dos lados: sólo se puede sostener el
valor del vecino más cercano, que es extrapolar. Es más débil por construcción, así que va con su
propio parámetro (`MAX_EXTREMO_SOSTENIDO`, propuesto en 1), su propia marca en
`origen_alineacion` y su propia validación. Que el número final —330 clips de 342— dependa de este
mecanismo no lo vuelve más sólido: si la validación de extremos no pasa, quedan los 180 clips del
relleno interior, que ya alcanzan para entrenar.

## Decisión 4 — La validación deja frames afuera a propósito

El error que no se puede cometer otra vez es verificar algo contra sí mismo, como hacía el Paso 16
al medir la transformada contra los mismos puntos con que se construyó.

El protocolo es **dejar uno afuera** sobre los frames que sí se detectaron: hay 3824 frames
alineados y 318 clips con tres o más, así que la base es amplia. Para cada uno de esos frames se
oculta su detección, se interpola su posición desde los vecinos y se compara contra la real. El
error se informa en tres unidades, porque cada una responde algo distinto:

1. **píxeles en la imagen original** — la magnitud cruda;
2. **fracción de la distancia ojo-hocico** — la magnitud relativa al tamaño de la cara, que es lo
   comparable entre videos de distinta escala;
3. **píxeles dentro del recorte de 224** — el desplazamiento que efectivamente vería el modelo.

El umbral de aceptación se declara **antes** de medir, en la spec, y se declara sobre la tercera
unidad, que es la que importa: si el desplazamiento mediano dentro del recorte supera unos pocos
píxeles, el rescate no sirve. Se informa la distribución completa —mediana, p90 y máximo—, no sólo
el promedio, porque lo que arruina una muestra es la cola.

La validación se estratifica por largo de hueco: un hueco de 1 y uno de 4 no tienen por qué tener
el mismo error, y es exactamente esa curva la que fija `MAX_HUECO_INTERPOLADO` con evidencia en vez
de con criterio.

## Decisión 5 — `origen_alineacion` en los artefactos

Cada frame registra cómo se alineó: `keypoints`, `interpolada`, `extremo_sostenido` o `caja`.

Sin esa columna, el rescate se vuelve invisible en cuanto se calcula la primera métrica, y no habría
manera de contestar en la defensa "¿cuántos de esos frames los detectó el modelo y cuántos los
inventaron ustedes?". Con la columna, además, la ablación —entrenar con y sin frames rescatados— es
un filtro de una línea.

## Riesgos

- **Un frame interpolado es un frame que nadie vio.** Aunque el error mediano sea chico, entran al
  dataset posiciones estimadas. La mitigación es el tope, la validación estratificada y la columna
  que permite excluirlos.
- **La validación mide sobre frames que sí se detectaron, que no son una muestra aleatoria** de los
  que fallaron: si la detección falla justamente cuando el caballo se mueve rápido, el error real
  de los huecos será mayor que el medido. Hay que decirlo en el informe, y se puede acotar
  comparando el error según la velocidad aparente entre los vecinos.
- **La cobertura deja de ser comparable con la histórica** salvo desglosada por `origen_alineacion`.
- **Quedan 12 clips sin ningún frame alineado** (1 `Disgustado`, 11 `Sereno`). Este change no los
  toca: sin una sola detección no hay de dónde interpolar.
