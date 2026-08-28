# Medición del encuadre — evidencia

Reproducible con `medicion_encuadre.py`, en este mismo directorio. Corrida el **28/08/2026** sobre
los artefactos de la etapa 02 vigente: 5.888 frames, de los cuales **5.586 tienen par ojo-hocico
detectado**. No se reinfirió pose.

## Por qué se pudo medir sin reinferir

`superanimal_quadruped` expone 39 bodyparts, entre ellos `left_earbase`, `left_earend`,
`right_earbase` y `right_earend`, y la etapa 02 los persiste **todos** en `pose_completa.csv` —21 MB,
formato largo— aunque después sólo use el par facial. Esa decisión, tomada en el change
`pose-completa-y-bodypart-hocico`, es la que permite contestar hoy una pregunta que no se había
formulado entonces, sin pagar la hora de inferencia.

## 1. Dónde están las orejas respecto del eje ojo-hocico

Coordenadas en el marco de la cabeza, en unidades de la distancia ojo-hocico `d`. El eje axial va
del ojo hacia el hocico, así que un valor negativo está **por encima** del ojo.

| Keypoint | n | axial p50 | axial p90 | lateral p50 | lateral p90 |
|---|---:|---:|---:|---:|---:|
| `left_earbase` | 3.485 | −0,47 | −0,69 | 0,22 | 0,66 |
| `left_earend` | 3.232 | −0,78 | −1,15 | 0,39 | 0,93 |
| `right_earbase` | 3.380 | −0,47 | −0,74 | 0,23 | 0,68 |
| `right_earend` | 3.255 | −0,79 | −1,18 | 0,40 | 1,04 |

Punto más alto de las orejas, por frame (n = 4.387): **p50 = 0,81 `d`**, p75 = 0,99, p90 = 1,23,
p95 = 1,49 por encima del ojo. Semiancho lateral máximo: p50 = 0,57 `d`, p90 = 1,11.

**El encuadre vigente muestra 0,33 `d` por encima del ojo y 0,83 `d` a cada lado.** Falta casi el
cuádruple de aire vertical; lateralmente sobra. Ésa es toda la causa del problema, y es la que
explica por qué mover una fracción sobre un lado único no lo resuelve.

## 2. Cuántos keypoints de oreja entran en el recorte vigente

| Keypoint | Dentro de 224×224 |
|---|---:|
| `left_earbase` / `right_earbase` | 19,8 % / 20,0 % |
| `left_earend` / `right_earend` | 2,8 % / 3,3 % |
| **Todos** | **11,9 %** |

El 88 % de los keypoints de oreja queda fuera, y las **puntas** —que son las que definen la
orientación de la oreja, el rasgo etológicamente informativo— quedan fuera en el 97 % de los casos.

## 3. Barrido de candidatos

Ancho de destino fijo en 224; el alto se deriva del aspecto que pide cada campo de visión.

| Encuadre | Aspecto | Destino | `d` → px | Orejas dentro | Relleno p50 | >25 % relleno |
|---|---:|---:|---:|---:|---:|---:|
| Vigente — 224×224 (`0,20/0,80`) | 1,00 | 224×224 | 134 | 11,9 % | 22,2 % | 42,5 % |
| Rama `hector` (`0,25/0,80`) | 1,00 | 224×224 | 123 | 27,3 % | 26,0 % | 52,9 % |
| **C** — 0,80 / 0,30 / ±0,75 | 1,40 | 224×314 | 149 | **64,6 %** | **21,2 %** | **39,9 %** |
| **A** — 1,00 / 0,40 / ±0,85 | 1,41 | 224×316 | 132 | **82,0 %** | 28,3 % | 57,0 % |

Contra el criterio declarado en la spec —inclusión de orejas ≥ 50 % **y** relleno mediano no peor
que la línea de base de 22,2 %—:

- **C lo cumple.** Quintuplica la inclusión de orejas y además mejora levemente el relleno, porque
  recorta campo lateral, que es donde sobraba. La cara queda un 11 % más grande que hoy.
- **A no lo cumple**: gana orejas pero paga 6 puntos de relleno.
- **La corrección propuesta en la rama `hector` no lo cumple** por ninguno de los dos lados: la
  dirección es correcta y la palanca equivocada, porque con recorte cuadrado el 45 % del campo que
  agrega se va a los costados.

## 4. Sesgos de esta medición

1. **Denominador de la inclusión de orejas.** Se mide sólo donde la oreja se detectó con confianza
   ≥ 0,50: 4.387 de los 5.586 frames con par facial (78,5 %). Una oreja fuera del cuadro tiende a
   no detectarse, así que el denominador excluye justo los peores casos. **Todos los encuadres
   quedan favorecidos por igual**: la comparación entre filas es válida, los niveles absolutos son
   optimistas.
2. **Estimación del relleno.** Se calcula con la caja alineada a ejes del rectángulo rotado, no con
   el rectángulo. Es una **cota superior** en todas las filas por igual: el warp sólo muestrea el
   rectángulo. Los niveles absolutos son pesimistas.

Los dos sesgos apuntan en direcciones opuestas y ninguno se corrige con más aritmética sobre los
mismos keypoints. La evidencia independiente es mirar los recortes.

## 5. Inspección visual

> Pendiente — tarea 1.4. Muestra de ~20 recortes de `Disgustado` bajo C y bajo A.

## 6. Decisión

> Pendiente — tarea 1.5.

## Crédito

La observación de que el recorte podía estar dejando las orejas fuera es de **Héctor R. Liceaga**,
en `analisis_y_tareas.md` de la rama `hector`: *"en los caballos, la posición de las orejas (hacia
atrás o planas) es el indicador número uno de disgusto o estrés"*. La medición de este documento la
confirma y cuantifica, y muestra además que la corrección concreta que proponía —`FRAC_OJO = 0,25`—
es insuficiente por un orden de magnitud.
