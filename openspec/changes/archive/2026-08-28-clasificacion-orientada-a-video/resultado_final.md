# Objetivo 3 — resultado final

Corrida el **28/08/2026**. Reproducible con `evaluacion_final.py`. Predicciones fuera de muestra,
pesos por fold y métricas en `resultados/objetivo3_evaluacion.json` y `modelos/objetivo3/`.

## El modelo elegido no es el que se esperaba elegir

**Clasificador de un frame con voto**: cabeza lineal sobre las características del backbone
congelado, entrenada a nivel de frame y agregada por voto mayoritario entre los 16 frames del clip.

Se lo diseñó como **línea de base** para poner a prueba la hipótesis del trabajo, y terminó siendo
el mejor de las **18 configuraciones** probadas — también el más simple y el más estable. Elegir
otro porque «es el que corresponde a un trabajo sobre video» sería elegir contra la evidencia.

**Es invariante al orden.** Barajar los frames del clip no cambia una predicción.

## Contra el criterio declarado antes de entrenar

| | |
|---|---:|
| Criterio declarado (tarea 2.4) | 0,691 |
| **F1 macro obtenido** | **0,626 ± 0,065** |
| | **NO CUMPLE** (−0,065) |

| Métrica | Valor |
|---|---:|
| F1 macro | 0,626 ± 0,065 |
| Balanced accuracy | 0,638 ± 0,069 |
| Recall `Disgustado` | 0,346 ± 0,131 |

Por fold: 0,685 · 0,554 · 0,678 · 0,588.

El criterio se declaró antes de entrenar y **se informa aunque no se cumpla**, que era la condición
con la que se declaró.

## Matriz de confusión — fuera de muestra, agregada sobre los 4 folds

| real \ predicha | Disgustado | Interesado | Sereno | total |
|---|---:|---:|---:|---:|
| **Disgustado** | **7** | 5 | 9 | 21 |
| **Interesado** | 2 | **49** | 5 | 56 |
| **Sereno** | 12 | 8 | **44** | 64 |
| total | 21 | 62 | 58 | 141 |

Tres lecturas:

- **`Interesado` se clasifica bien** (49 de 56, recall 0,88). Es la clase con mejor cobertura de
  orejas dentro del recorte (70,5 %) y la de mejor disponibilidad sostenida por clip.
- **`Disgustado` no colapsa, pero apenas se sostiene**: 7 de 21. No es el caso de «nunca predice la
  clase minoritaria» —el modelo la predice 21 veces— pero acierta un tercio.
- **La confusión dominante contradice lo que el diseño anticipaba.** Se esperaba que `Disgustado`
  se confundiera con `Interesado`, por el caso `178` donde dos tomas contiguas del mismo ejemplar
  tienen clases opuestas. Lo medido es al revés: `Disgustado` → `Sereno` (9) supera a
  `Disgustado` → `Interesado` (5), y el error más voluminoso de todos es `Sereno` → `Disgustado`
  (12). El eje que el modelo confunde es **Sereno/Disgustado**, no el que se había previsto.

## Ablación por `origen_alineacion` (tarea 4.4)

Sólo los clips cuyos 16 frames se alinearon por **detección propia** del modelo de pose, sin
interpolación ni anotación manual. Mismos folds, para que sea comparable.

| | Clips | Ejemplares | F1 macro | Recall `Disgustado` |
|---|---:|---:|---:|---:|
| Completo | 141 | 73 | **0,626 ± 0,065** | 0,346 |
| Sólo detección propia | 82 | 50 | 0,518 ± 0,060 | **0,000** |

**−0,108 de F1 macro, y `Disgustado` colapsa por completo.**

Esto **no invalida el resultado completo** — es exactamente lo que el diseño anticipó al declarar
la ablación. El subconjunto de detección propia tiene **7 clips de `Disgustado`** repartidos en 4
folds: uno o dos por fold. Con eso ninguna métrica de esa clase es sostenible, y el recall 0,000 lo
refleja.

Lo que la ablación dice es **cuánto del resultado se apoya en posiciones estimadas**: sin ellas el
dataset no alcanza para la clase de interés clínico. Es un argumento a favor del relleno temporal y
de la anotación manual del objetivo 2, no en contra.

## Errores por ejemplar (tarea 4.5)

- **22 de 73 ejemplares** tienen al menos un error. Los otros 51 no fallan nunca.
- Los 10 peores concentran **28 de los 41 errores (68 %)**.
- Peores: `906` 5/10 · `13` 4/4 · `237` 3/6 · `56` 3/4 · `910` 3/4 · `178` 2/3 · `230` 2/3 ·
  `231` 2/2 · `37` 2/3 · `44` 2/2.

**Los errores se concentran, y eso es información sobre el dataset y no sobre el modelo.** Cuatro
de los diez peores —`237`, `910`, `178`, `230`, `37`— están entre los 10 ejemplares que aparecen
en **más de una clase**. Que el modelo falle justo donde el mismo caballo tiene clips etiquetados
distinto es coherente con el techo declarado del caso `178`: son eventos continuos cortados en
tomas, y la frontera es ruido irreducible.

`13` y `44` fallan en el 100 % de sus clips, y `231` también. Son candidatos a revisión de
etiqueta o de calidad, no a mejora de arquitectura.

## Conclusión del objetivo 3

**La hipótesis del trabajo no queda sostenida por esta evidencia.** Sobre 18 configuraciones, en
dos regímenes de capacidad y con muestras igualadas, el orden temporal no aportó en ninguna: la
atención da predicciones idénticas con los frames barajados y la GRU rinde peor con el orden real.
El mejor modelo es invariante al orden.

**Lo que sí quedó demostrado, y es un resultado positivo del trabajo:**

- **Las imágenes tienen señal.** De 0,188 (clase mayoritaria) a 0,626 con partición honesta por
  ejemplar. El preprocesamiento facial del objetivo 2 produce una entrada de la que un clasificador
  lineal extrae bastante más que las clases a priori.
- **El protocolo es sólido y está cuantificado.** Sin agrupar por caballo se llega a F1 macro 0,53
  *sin mirar un píxel*; agrupando, esa vía queda en 0,188. Los +0,338 de diferencia son la fuga que
  el protocolo elimina.
- **El resultado es honesto por construcción.** Criterio declarado antes de entrenar, sin early
  stopping sobre el fold de test, 18 configuraciones informadas y no sólo la mejor.

## Qué haría falta para mejorarlo (tarea 5.4)

La respuesta es **más ejemplares**, y está medida, no supuesta:

1. **Más caballos, no más clips del mismo caballo.** El cuello de botella son los 73 ejemplares, y
   sobre todo los **15 de `Disgustado`**. La aumentación se probó y no movió nada (−0,005 a
   +0,000): 20 vistas del mismo clip siguen siendo el mismo caballo, la misma toma y la misma
   expresión.
2. **Más material de `Disgustado` con detección propia.** La ablación lo muestra: con 7 clips la
   clase colapsa. Es la clase de interés clínico y la que menos material tiene.
3. **Revisar las etiquetas de los ejemplares que fallan siempre** (`13`, `44`, `231`) y las
   fronteras entre tomas contiguas de un mismo caballo con clases opuestas.

Lo que **no** haría falta: más arquitectura. Se probaron tres cabezas en dos regímenes de capacidad
y ninguna superó a una regresión logística sobre características congeladas.
