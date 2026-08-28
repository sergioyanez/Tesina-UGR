# Medición del encuadre — evidencia y decisión

Reproducible con `medicion_encuadre.py`, en este mismo directorio, que corre el barrido y genera
la hoja de contacto. La lógica vive en `equinos/encuadre.py`, con 43 pruebas en
`tests/test_encuadre.py`.

Corrida el **28/08/2026** sobre los artefactos de la etapa 02 vigente: 5.888 frames, de los cuales
**5.586 tienen par ojo-hocico detectado**. No se reinfirió pose.

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

## 2. La métrica que decide: el vector de oreja

El primer instinto es contar keypoints de oreja dentro del recorte. **Sobrestima, y no por poco.**
Lo que informa el estado del animal es la **orientación** de la oreja —echada hacia atrás,
aplanada—, y eso se lee del vector base→punta. Una base sin su punta no dice hacia dónde apunta.

La diferencia es material: el candidato C mete el 89 % de las bases y sólo el 38 % de las puntas,
así que informa 64,4 % contando keypoints sueltos y **50,6 %** contando vectores legibles. Por eso
el criterio de la spec se ancló en el vector. Desglose por keypoint:

| Keypoint | Vigente | C | D | A |
|---|---:|---:|---:|---:|
| `left_earbase` | 19,8 % | 89,3 % | — | 94,2 % |
| `left_earend` | **2,8 %** | **40,6 %** | — | 69,3 % |
| `right_earbase` | 20,0 % | 88,4 % | — | 94,3 % |
| `right_earend` | **3,3 %** | **36,3 %** | — | 68,4 % |

Con el encuadre vigente, el vector de oreja es legible en el **4,3 %** de los frames.

## 3. Barrido de candidatos

Ancho de destino fijo en 224; el alto se deriva del aspecto que pide cada campo de visión. Los tres
números del encuadre son aire sobre el ojo / aire bajo el hocico / semiancho lateral, en `d`.

| Encuadre | Destino | `d`→px | **Vector oreja** | kp sueltos | Relleno p50 | p90 | >25 % |
|---|---:|---:|---:|---:|---:|---:|---:|
| Vigente 0,33/0,33/0,83 | 224×224 | 134 | 4,3 % | 11,7 % | 14,1 % | 32,8 % | 21,8 % |
| `hector` 0,45/0,36/0,91 | 224×224 | 123 | 13,7 % | 27,1 % | 18,8 % | 37,5 % | 30,2 % |
| C 0,80/0,30/0,75 | 224×314 | 150 | 50,6 % | 64,4 % | **12,5 %** | 31,2 % | 16,9 % |
| **D 1,00/0,30/0,75** | **224×343** | **149** | **77,2 %** | 78,2 % | **14,1 %** | 35,9 % | 22,5 % |
| E 1,20/0,30/0,75 | 224×373 | 149 | 85,7 % | 83,3 % | 17,2 % | 39,1 % | 29,8 % |
| F 1,20/0,25/0,70 | 224×392 | 160 | 85,2 % | 80,7 % | 15,6 % | 37,5 % | 25,7 % |
| G 1,40/0,25/0,70 | 224×424 | 160 | 88,7 % | 82,4 % | 18,8 % | 42,2 % | 35,4 % |
| A 1,00/0,40/0,85 | 224×316 | 132 | 78,3 % | 81,9 % | 18,8 % | 40,6 % | 32,8 % |

**La corrección propuesta en la rama `hector`** (`FRAC_OJO = 0,25`) lleva el vector de oreja de
4,3 % a 13,7 % y empeora el relleno. La dirección es correcta y la palanca equivocada: con recorte
cuadrado, casi la mitad del campo que agrega se va a los costados, donde ya sobraba.

**D y A rinden casi igual en orejas** (77,2 % contra 78,3 %) pero D paga mucho menos relleno
(14,1 % contra 18,8 %). La diferencia es enteramente el campo lateral: D conserva el ±0,75 `d`
angosto de C y gasta todo su presupuesto en vertical, que es donde estaba el déficit. Es la
asimetría de la sección 1 rindiendo en la práctica.

## 4. Sesgos de esta medición

1. **Denominador de la inclusión de orejas.** Se mide sólo donde la oreja se detectó con confianza
   ≥ 0,50: 3.345 frames tienen alguna oreja completa detectada, de 5.586 con par facial. Una oreja
   fuera del cuadro tiende a no detectarse, así que el denominador excluye justo los peores casos.
   **Todos los encuadres quedan favorecidos por igual**: la comparación entre filas es válida, los
   niveles absolutos son optimistas. No se corrige con más aritmética sobre los mismos keypoints;
   la evidencia independiente es la sección 5.
2. **Estimación del relleno — corregida durante este barrido.** La primera versión aproximaba el
   recorte por la caja envolvente del rectángulo rotado, que lo sobrestima de forma sistemática
   (daba 22,2 % de línea de base en vez de 14,1 %). Se reemplazó por el muestreo por rejilla de 8×8
   que ya usaba `reporte.calidad_recortes`, que mide el rectángulo y no su envolvente. **Este sesgo
   quedó eliminado**; sólo queda la discretización de la rejilla, que afecta a todas las filas por
   igual.

## 5. Inspección visual

Seis ejemplares distintos de `Disgustado`, un frame del medio de un clip íntegro con par detectado,
bajo los cuatro encuadres. Generada por `medicion_encuadre.py <salida.png>`.

- **En el encuadre vigente, en los seis, el recorte está dominado por el hocico y la frente, y las
  orejas están cortadas.** Confirma el 4,3 % sin ambigüedad.
- **C mete las orejas en unos y las corta en otros** —en `205` se ve claramente que la punta de una
  oreja queda afuera—, consistente con su 50,6 %.
- **D las mete en todos los casos legibles**, y con margen visible en `205`, `208` y `209`.
- **A también, pero se le nota el campo lateral extra**: en `205` y `209` entra bastante más cielo,
  poste y techo. Ese fondo adicional es exactamente el confounder que el recorte a ROI existe para
  mitigar, y es un argumento contra A que la tabla sola no muestra.
- **Hallazgo lateral, ajeno a este change:** el clip de `054` sale inutilizable en los cuatro
  encuadres —desenfocado y en primerísimo plano, con keypoints probablemente mal puestos—. No es un
  problema de encuadre; queda anotado para revisar aparte.

## 6. Decisión

**Se adopta D: aire sobre el ojo 1,00 `d`, aire bajo el hocico 0,30 `d`, semiancho lateral
0,75 `d`. Destino 224×343.**

Contra el criterio declarado en la spec:

| Condición declarada | Línea de base | D | ¿Cumple? |
|---|---:|---:|:--:|
| Vector de oreja legible ≥ 50 % | 4,3 % | **77,2 %** | ✅ |
| Relleno mediano no peor que la línea de base | 14,0625 % | **14,0625 %** | ✅ (empate exacto) |

Y tres propiedades que no estaban en el criterio pero pesan a favor:

- **La cara queda un 11 % más grande** que hoy (`d`: 134 → 149 px), porque D recorta campo lateral.
- **Menos fondo** que el encuadre vigente (±0,75 `d` contra ±0,83), o sea menos confounder de
  escena, no más.
- **Ninguna reinferencia ni re-anotación**, y la composición del dataset no cambia.

**Lo que hay que decir del otro lado, y no está bien esconderlo:** el empate en el relleno es de la
**mediana**, que es lo que el criterio nombra. En la cola D es algo peor que la línea de base —p90
de 35,9 % contra 32,8 %, y 22,5 % de recortes con más de un cuarto de relleno contra 21,8 %—. Es un
costo real, chico, y es la razón por la que el change agrega `relleno_fraccion` por frame: para
poder medir después si ese relleno explica algo del resultado del clasificador en vez de suponerlo.

**Por qué no A**, que da 78,3 % de vector de oreja: sube el relleno mediano a 18,8 % —incumple la
segunda condición— y agrega campo lateral que en la hoja de contacto se ve como fondo. Paga 1,1
puntos de oreja con 4,7 puntos de relleno y más confounder de escena.

**Por qué no E, F ni G**, que llegan a 85–89 %: todos superan la línea de base de relleno. Quedan
anotados: si al entrenar el objetivo 3 la ablación por `relleno_fraccion` muestra que el relleno no
aporta señal espuria, **F es el candidato natural para una segunda vuelta** —85,2 % de vector de
oreja con 15,6 % de relleno, el mejor de los que incumplen y por poco.

## 7. La corrida entregada (28/08/2026)

Re-ejecutada sólo la parte de recorte de la etapa 02, sin reinferir: el checkpoint declara
`CLAVES_KEYPOINTS = []`, así que cambiar el encuadre no lo invalida y los 368 clips siguieron
marcados como completos.

**La composición del dataset es idéntica**, que es lo que la tarea 4.3 exigía verificar:

| | antes | después |
|---|---:|---:|
| filas / clips / caballos | 5888 / 368 / 124 | 5888 / 368 / 124 |
| frames alineados | 4326 (73,5 %) | 4326 (73,5 %) |
| origen (kp/caja/interp/manual) | 4046 / 1562 / 266 / 14 | 4046 / 1562 / 266 / 14 |
| clips íntegros | 141 (21/56/64) | 141 (21/56/64) |
| ejemplares con clip íntegro | 73 | 73 |

Sólo cambiaron los píxeles: recorte 224×343, `d` = 149 px.

**El barrido predijo la corrida real.** Relleno mediano predicho 14,06 %, medido 14,1 %. Vector de
oreja predicho 77,2 %, medido 77,2 %. Es la validación de que comparar encuadres sobre los
keypoints persistidos —sin escribir una sola imagen— dice lo que va a pasar.

Inspección de los JPEG efectivamente entregados, cuatro frames repartidos de cada clip: en `205` y
`208` se ven las dos orejas en los cuatro frames, y la cara se mantiene en la misma posición
canónica a lo largo del clip. También se ve el costo: esos dos clips tienen relleno 0,39 y 0,41, y
el cielo replicado se nota en las esquinas superiores.

## 8. Hallazgo de la revalidación de la interpolación (tarea 5.3)

`interpolacion.informe_validacion()` sobre la geometría nueva recomienda ahora
**`MAX_HUECO_INTERPOLADO = 2`**, contra el 1 vigente:

| | línea de base | largo 1 | largo 2 |
|---|---:|---:|---:|
| archivado (224×224, 3.900 frames detectados) | 16,28 px | 13,99 ✅ | 16,33 ❌ *(por 0,05 px)* |
| ahora (224×343, 4.046 frames detectados) | 18,66 px | 15,75 ✅ | 18,36 ✅ *(por 0,30 px)* |

**No se cambia el tope en este change**, por tres razones:

1. **La comparación mezcla dos cambios.** La corrida archivada midió sobre 3.900 frames detectados
   y ésta sobre 4.046: entre medio el dataset creció. No se puede atribuir el vuelco al encuadre.
2. **El margen es de 0,30 px sobre una mediana**, después de haber fallado por 0,05. Un criterio
   que se decide en el tercer decimal necesita más evidencia que una corrida.
3. **Cambiar el tope cambia la composición del dataset** —más frames rescatados, más clips
   íntegros—, y eso está explícitamente fuera del alcance de este change.

Queda como change aparte, con la ventaja de que ahora la validación se corre en segundos.

## Crédito

La observación de que el recorte podía estar dejando las orejas fuera es de **Héctor R. Liceaga**,
en `analisis_y_tareas.md` de la rama `hector`: *"en los caballos, la posición de las orejas (hacia
atrás o planas) es el indicador número uno de disgusto o estrés"*. La medición de este documento la
confirma y cuantifica, y muestra además que la corrección concreta que proponía —`FRAC_OJO = 0,25`—
recupera 9 de los 73 puntos porcentuales que estaban en juego.
