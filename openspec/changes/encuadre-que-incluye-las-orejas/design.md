# Diseño — encuadre que incluye las orejas

## Context

La etapa 02 alinea cada frame por una transformada de similitud que lleva el ojo y el hocico a
posiciones canónicas dentro de un recorte cuadrado de 224×224. El encuadre vigente
(`FRAC_OJO = 0,20`, `FRAC_HOCICO = 0,80`) se fijó el 19/08/2026 para resolver un problema real: los
valores anteriores pedían un cuadrado de 2,5 veces la distancia ojo-hocico, imposible de satisfacer
en cuadros de 1.080 px de ancho, y el 97,5 % de los recortes salía con relleno.

Ese arreglo optimizó lo que se estaba midiendo —cuánto del recorte era cara y no borde inventado— y
nadie midió lo que no se estaba mirando: **qué parte de la cara entra**. Medido ahora sobre los
5.586 frames con par ojo-hocico detectado, entra el 11,7 % de los keypoints de oreja y el 3 % de
las puntas — y el vector base→punta, que es lo que permite leer la orientación de la oreja, es
legible en el **4,3 %**.

La restricción del dataset ordena la lectura de todo lo que sigue: **141 clips íntegros de 73
ejemplares distintos**, con `Disgustado` en 21 clips de 15 caballos. Con 73 ejemplares, cualquier
artefacto del recorte que correlacione con el individuo —y el relleno correlaciona con la distancia
de filmación— es señal que el clasificador puede aprender en lugar de la expresión. Y con 15
ejemplares en la clase minoritaria, no hay margen para un encuadre que reduzca la cantidad de clips
utilizables.

Un dato que ordena el alcance: **el encuadre no afecta qué frames quedan alineados.** Un frame se
alinea si tiene el par ojo-hocico —propio, interpolado o manual—, y eso no depende del campo de
visión. Los 141 clips íntegros, los 73 ejemplares y el reparto 21/56/64 son **idénticos antes y
después de este change**. Lo único que cambia son los píxeles.

## Goals / Non-Goals

**Goals:**

- Que el recorte canónico contenga las orejas en una fracción mayoritaria de los frames.
- Elegir el encuadre con evidencia medida, no con un valor propuesto de antemano.
- No aumentar el relleno respecto del encuadre vigente, o dejar la compensación explícita y medible
  si se decide aceptarla.
- No tocar nada que obligue a reinferir pose ni a re-anotar a mano.
- Dejar registrado por frame el artefacto que este cambio agrava, para poder ablacionarlo después.

**Non-Goals:**

- Cambiar el par de anclaje o el modelo de pose.
- Mejorar la cobertura de alineación. Es ortogonal y no se toca.
- Deduplicar frames por similitud perceptual (`imagededup`, propuesto en la rama `hector`): es un
  change aparte, y adoptarlo acá mezclaría dos causas en una misma corrida.
- El objetivo 3. Este change lo desbloquea, no lo empieza.

## Decisions

### Decisión 1 — El encuadre se declara como campo de visión, no como fracciones de un lado

Tres parámetros en unidades de la distancia ojo-hocico `d`: aire sobre el ojo, aire bajo el hocico,
semiancho lateral. `FRAC_OJO` y `FRAC_HOCICO` se derivan de ellos, no al revés.

El motivo no es cosmético. Con `FRAC_OJO`/`FRAC_HOCICO` sobre un lado único, el campo vertical y el
lateral están atados: no se puede pedir más aire arriba sin pedir más a los costados. Esa atadura
es la que hace que el problema parezca insoluble, porque el déficit medido es asimétrico:

```
Necesario arriba del ojo (punta de oreja):  p50 0,81 d   p75 0,99 d   p90 1,23 d
Disponible arriba del ojo:                       0,33 d
Necesario a los lados (semiancho oreja):    p50 0,57 d   p90 1,11 d
Disponible a los lados:                          0,83 d
```

Falta casi el cuádruple de aire vertical; lateralmente sobra. La parametrización correcta es la que
deja expresar eso.

**Alternativa descartada — mover `FRAC_OJO` de 0,20 a 0,25**, que es la corrección propuesta en la
rama `hector`. Medida: lleva el vector de oreja legible de 4,3 % a 13,7 % y sube el relleno mediano
de 14,1 % a 18,8 %. Es la dirección correcta con la palanca equivocada: paga relleno para ganar poco,
porque el 45 % del campo que agrega se va a los costados, donde ya sobraba.

### Decisión 2 — El anclaje sigue siendo ojo-hocico; las orejas entran por campo de visión

Las orejas no se usan como puntos de alineación. Entran porque el recorte las abarca.

**Alternativa descartada — anclar en el punto medio de las bases de oreja y el hocico.** Garantizaría
por construcción que las orejas caigan en una posición canónica, que es más elegante. Se descarta
por disponibilidad: `left_earbase` tiene 62,4 % de confianza suficiente y `right_earbase` 60,5 %,
contra el par facial que sostiene la cobertura actual. Cambiar el anclaje haría caer la cantidad de
frames alineables, y con 15 ejemplares en `Disgustado` no hay margen para perder clips íntegros a
cambio de elegancia geométrica.

**Alternativa descartada — transformada afín de tres puntos (ojo, hocico, oreja) o Procrustes sobre
el conjunto facial.** Absorbería la rotación de la oreja dentro de la normalización, que es
justamente la señal que se quiere que el modelo vea. Normalizar el rasgo que se quiere clasificar
lo borra.

**Alternativa descartada — no tocar el recorte y pasar los keypoints de oreja como características
auxiliares al clasificador.** Es barato y no re-procesa nada. Se descarta porque hereda el 60 % de
disponibilidad como una entrada faltante en el 40 % de los frames, y porque convierte un problema
de píxeles en un problema de imputación, sobre 141 muestras. Queda anotada como extensión posible
si el encuadre nuevo no alcanza.

### Decisión 3 — El recorte deja de ser cuadrado

El requisito de recorte cuadrado se retira. La justificación que lo sostenía —que el reescalado no
deforme la cara— la cubre la **isotropía de la transformada de similitud**, que se conserva. La
forma del destino determina el campo de visión, no el estiramiento; atarlas fue una confusión.

El eje ojo-hocico está a 24° de la vertical del cuadro en la mediana (93,4 % de los frames por
debajo de 45°), porque el material es video de celular en 1080×1920. Un rectángulo alto se estira
sobre el eje largo del cuadro, que es donde hay píxeles disponibles.

**Alternativa descartada — cuadrado más grande.** Para llegar al aire vertical necesario hace falta
`FRAC_OJO ≈ 0,40`, o sea 2,5 `d` de lado: 1.393 px sobre cuadros de 1.080. Es el encuadre que ya se
descartó el 19/08/2026. Reintroducirlo sería deshacer un arreglo con evidencia.

**Alternativa descartada — dos recortes por frame**, uno facial y uno de orejas, alimentando dos
ramas. Duplica el cómputo y los parámetros entrenables de la cabeza, contra 141 muestras. La
restricción del dataset la descarta sola.

### Decisión 4 — El encuadre lo eligió el barrido, y la métrica que lo decidió no era la esperada

Se declararon los candidatos y el criterio, y el valor se fijó con la medición. **Encuadre
adoptado: D — 1,00 `d` de aire sobre el ojo, 0,30 bajo el hocico, ±0,75 lateral, destino 224×343.**

La evidencia completa está en `medicion.md`; lo esencial:

| Encuadre | Destino | `d`→px | **Vector oreja** | kp sueltos | Relleno p50 | >25 % |
|---|---:|---:|---:|---:|---:|---:|
| Vigente 0,33/0,33/0,83 | 224×224 | 134 | 4,3 % | 11,7 % | 14,1 % | 21,8 % |
| `hector` 0,45/0,36/0,91 | 224×224 | 123 | 13,7 % | 27,1 % | 18,8 % | 30,2 % |
| C 0,80/0,30/0,75 | 224×314 | 150 | 50,6 % | 64,4 % | 12,5 % | 16,9 % |
| **D 1,00/0,30/0,75** | **224×343** | **149** | **77,2 %** | 78,2 % | **14,1 %** | 22,5 % |
| A 1,00/0,40/0,85 | 224×316 | 132 | 78,3 % | 81,9 % | 18,8 % | 32,8 % |

**Lo que el barrido cambió respecto de lo previsto.** El diseño entraba apostando a C, que era el
único candidato que mejoraba las orejas sin empeorar el relleno. El barrido mostró que C mete el
89 % de las **bases** de oreja y sólo el 38 % de las **puntas**, y que sin punta no se lee la
orientación —que es el rasgo informativo—. Medido en vectores de oreja legibles, C rinde 50,6 % y
no 64,4 %. Eso obligó a re-anclar el criterio en el vector y a buscar un candidato mejor.

D es ese candidato, y la razón por la que existe es la asimetría de la Decisión 1: **conserva el
campo lateral angosto de C y gasta todo el presupuesto en vertical**. Iguala a A en orejas (77,2 %
contra 78,3 %) pagando el relleno de la línea de base en vez de 4,7 puntos más, y encima con menos
fondo que el encuadre vigente.

**Alternativa descartada — A (±0,85 lateral).** Da 1,1 puntos más de oreja y cuesta 4,7 puntos de
relleno mediano, incumpliendo el criterio declarado. En la hoja de contacto ese campo lateral extra
se ve directamente como cielo, poste y techo: más confounder de escena, que es lo que el recorte a
ROI existe para mitigar.

**Alternativas descartadas — E, F y G (1,20–1,40 `d` de aire arriba).** Llegan al 85–89 % de vector
de oreja y las tres superan la línea de base de relleno. F queda anotado como candidato de una
segunda vuelta: es el que menos la supera (15,6 %) y si la ablación por `relleno_fraccion` muestra
que el relleno no aporta señal espuria, es el próximo escalón.

**Alternativa descartada — fijar un encuadre sin barrer.** Habría fijado C, que era la apuesta
razonable a priori y rinde la mitad que D en la métrica que importa.

### Decisión 5 — El relleno se registra por frame en lugar de intentar eliminarlo

Se agrega `relleno_fraccion` al CSV de frames alineados, con el mismo rol que `origen_alineacion`:
no evita el artefacto, lo vuelve auditable y ablacionable.

Con 73 ejemplares, el relleno es un confounder plausible —depende de la distancia de filmación, que
depende del ejemplar, que predice la clase—. La respuesta del proyecto a esa clase de problema ya
está establecida: registrar el origen y poder entrenar con y sin. Se repite el patrón en vez de
inventar uno nuevo.

Se conserva `BORDER_REPLICATE`. **Alternativa descartada — relleno por reflexión**: produce una
cara espejada parcial en el borde, que es un patrón con estructura y por lo tanto más aprendible
que el borde estirado. **Alternativa descartada — descartar los frames con relleno alto**: costaría
clips íntegros, y son 141.

### Decisión 6 — No se reinfiere pose ni se re-anota

`pose_completa.csv` (21 MB, 39 bodyparts × 5.888 frames) y `keypoints_faciales.csv` están en disco.
El notebook 02 ya declara que todo lo que sigue a la inferencia consume únicamente
`keypoints_faciales.csv`. El barrido y el recorte nuevo consumen esos artefactos.

`keypoints_manuales.csv` es `(clip, orden, x_ojo, y_ojo, x_hocico, y_hocico)` en coordenadas del
frame original. Como el par de anclaje no cambia, **las 14 anotaciones siguen siendo válidas sin
tocarlas**. Es la consecuencia práctica más importante de la Decisión 2.

Los topes del relleno temporal están medidos en píxeles dentro del recorte, y la escala del recorte
**sí** se mueve: `d` ocupa 134 px hoy y 149 px en el encuadre adoptado. Pero el criterio vigente no es un
número de píxeles sino una **comparación contra una línea de base medida del mismo modo** —el
desplazamiento entre dos detecciones propias consecutivas—, y ambos lados escalan igual. El
criterio es por lo tanto invariante al cambio de escala, y la revalidación de la tarea 5.3 se corre
para confirmarlo, no esperando que dé distinto. Si diera distinto, se re-declara con su historia,
como ya se hizo el 28/08/2026.

## Risks / Trade-offs

- **Las dos métricas del barrido comparten sesgos conocidos.** → La inclusión de orejas se mide sólo
  donde la oreja se detectó (78,5 % de los frames), y una oreja fuera del cuadro tiende a no
  detectarse: sobrestima a todos los encuadres por igual. El relleno se estima con la caja alineada
  a ejes del rectángulo rotado, que lo sobrestima en todos por igual. *Mitigación:* ambos sesgos se
  informan junto a los números; la comparación entre encuadres es válida aunque los niveles
  absolutos no lo sean; y la inspección visual de la muestra de `Disgustado` es evidencia
  independiente de ambos.
- **Más campo de visión es más relleno, y el relleno correlaciona con el ejemplar.** → *Mitigación:*
  `relleno_fraccion` por frame + criterio de aceptación que no admite empeorar la línea de base +
  ablación posterior.
- **Más campo de visión puede ser menos resolución sobre la cara.** → *Mitigación:* medido, no
  supuesto. El destino crece en píxeles a la par del campo, y como el encuadre adoptado además
  **recorta** campo lateral —±0,75 `d` contra ±0,83 `d` del vigente—, la cara queda un 11 % más
  grande que hoy (`d`: 134 → 149 px). No se pierde resolución facial; lo que se sacrifica es campo
  a los costados, donde la medición dice que sobraba.
- **El change `clasificacion-orientada-a-video` asume entradas de 224×224.** → *Mitigación:* está en
  planificación y sin implementar; se actualiza su design antes de aplicarlo. Los backbones
  preentrenados aceptan entrada no cuadrada por *pooling* adaptativo, así que el impacto es de
  documentación y de coste por frame, no de arquitectura.
- **Riesgo metodológico: parecer que se eligió el encuadre que convenía.** → *Mitigación:* el
  criterio se declara en la spec antes de correr el barrido, anclado en la línea de base medida, y
  el resultado se informa aunque no se cumpla.

## Migration Plan

1. Barrido sobre keypoints persistidos, sin escribir imágenes. Elección entre C y A con la tabla
   más la inspección visual.
2. Parametrización nueva en el notebook 02 y `destino_canonico()` con campo de visión.
3. Actualización de `tests/test_geometria.py`: las pruebas que fijan el recorte cuadrado pasan a
   fijar la isotropía de la escala y la relación de aspecto declarada.
4. Re-ejecución del warp y escritura de los 5.888 JPEG, con `relleno_fraccion`.
5. Recálculo de cobertura, inclusión de orejas y composición. Revalidación de los topes de
   interpolación.
6. Actualización de `openspec/project.md`, `README.md` y el design de
   `clasificacion-orientada-a-video`.

**Rollback:** el encuadre queda en la celda de parámetros; volver a los valores vigentes y re-correr
el paso 4 reproduce el dataset anterior. Los artefactos de pose y las anotaciones manuales no se
tocan en ningún paso, así que el rollback nunca cuesta reinferencia.

## Open Questions

Las tres que abrió este diseño quedaron contestadas por el barrido del 28/08/2026. Se dejan con su
respuesta porque el recorrido es parte de lo que hay que poder contar.

- **¿C o A?** → **Ninguno de los dos: D.** El barrido incorporó candidatos intermedios al ver que
  la métrica correcta era el vector de oreja, y D iguala a A en orejas pagando el relleno de la
  línea de base. Ver Decisión 4.
- **¿Las bases de oreja alcanzan, o hacen falta las puntas?** → **Hacen falta las puntas, y es
  la pregunta que cambió la decisión.** C mete el 89 % de las bases y el 38 % de las puntas; medido
  en vectores legibles rinde 50,6 % en vez de 64,4 %. El criterio de la spec se re-ancló en el
  vector base→punta.
- **¿Conviene informar la inclusión de orejas desglosada por clase?** → Se resuelve al implementar
  el reporte (tarea 5.1). El desglose por keypoint ya resultó más informativo que el desglose por
  clase para tomar esta decisión, y sobre un denominador ya sesgado el corte por clase puede
  confundir más de lo que aclara.

Queda una pregunta nueva, para después de entrenar:

- **¿El relleno aporta señal espuria?** Si la ablación por `relleno_fraccion` dice que no, **F**
  (1,20/0,25/±0,70: 85,2 % de vector de oreja con 15,6 % de relleno) es el próximo escalón de
  encuadre, y este change deja el barrido listo para volver a correrlo.
