# Diseño — encuadre que incluye las orejas

## Context

La etapa 02 alinea cada frame por una transformada de similitud que lleva el ojo y el hocico a
posiciones canónicas dentro de un recorte cuadrado de 224×224. El encuadre vigente
(`FRAC_OJO = 0,20`, `FRAC_HOCICO = 0,80`) se fijó el 19/08/2026 para resolver un problema real: los
valores anteriores pedían un cuadrado de 2,5 veces la distancia ojo-hocico, imposible de satisfacer
en cuadros de 1.080 px de ancho, y el 97,5 % de los recortes salía con relleno.

Ese arreglo optimizó lo que se estaba midiendo —cuánto del recorte era cara y no borde inventado— y
nadie midió lo que no se estaba mirando: **qué parte de la cara entra**. Medido ahora sobre los
5.586 frames con par ojo-hocico detectado, entra el 11,9 % de los keypoints de oreja y el 3 % de
las puntas.

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
rama `hector`. Medida: lleva la inclusión de orejas de 11,9 % a 27,3 % y sube el relleno mediano de
22,2 % a 26,0 %. Es la dirección correcta con la palanca equivocada: paga relleno para ganar poco,
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

### Decisión 4 — El encuadre concreto lo elige un barrido, no esta decisión

Se declaran los candidatos y el criterio; el valor se fija con la medición. Medido sobre la corrida
vigente, sin escribir una sola imagen:

| Encuadre | Destino | `d` → px | Orejas dentro | Relleno mediano | >25 % relleno |
|---|---:|---:|---:|---:|---:|
| **Vigente** — 224×224, `0,20/0,80` | 224×224 | 134 | 11,9 % | 22,2 % | 42,5 % |
| Rama `hector` — 224×224, `0,25/0,80` | 224×224 | 123 | 27,3 % | 26,0 % | 52,9 % |
| **C** — 0,80 `d` arriba, 0,30 abajo, ±0,75 lateral | 224×314 | 149 | **64,6 %** | **21,2 %** | **39,9 %** |
| **A** — 1,00 `d` arriba, 0,40 abajo, ±0,85 lateral | 224×316 | 132 | **82,0 %** | 28,3 % | 57,0 % |

Las dimensiones del destino no se eligen: se derivan del aspecto que pide cada campo de visión, con
el ancho fijo en 224. Reproducible con `medicion_encuadre.py`, que acompaña a este change.

**C cumple las dos condiciones del criterio declarado** —supera el 50 % de inclusión de orejas y no
empeora el relleno; de hecho lo mejora—. **A** llega al 82 % pagando seis puntos de relleno, y por
eso no se adopta de entrada: la decisión de aceptar esa compensación necesita ver los recortes, no
sólo la tabla.

El barrido se completa con **inspección visual de una muestra de recortes de `Disgustado`**, que es
la clase donde el rasgo importa y la que tiene menos ejemplares. Es también el único control contra
el sesgo de denominador de la métrica.

**Alternativa descartada — fijar C ahora y ahorrarse el barrido.** Tentador porque C ya cumple. Se
descarta porque los cuatro números de la tabla comparten dos sesgos conocidos (ver Riesgos) y
ninguno se corrige con más aritmética sobre los mismos keypoints. Una muestra mirada a ojo cuesta
minutos y es la única evidencia independiente disponible.

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
**sí** se mueve: `d` ocupa 134 px hoy, 149 px en C y 132 px en A. Pero el criterio vigente no es un
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
  supuesto. El destino crece en píxeles a la par del campo, y como C además **recorta** campo
  lateral —±0,75 `d` contra ±0,83 `d` del vigente—, la cara queda un 11 % más grande que hoy (`d`:
  134 → 149 px). A queda prácticamente igual (132 px). Ninguno de los dos pierde resolución facial;
  lo que C sacrifica es campo a los costados, donde la medición dice que sobraba.
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

- **¿C o A?** Lo resuelve el paso 1. C cumple el criterio; A lo cumple en orejas y no en relleno.
- **¿Las bases de oreja alcanzan, o hacen falta las puntas?** La orientación de la oreja se lee del
  vector base→punta. Si con C entran las bases pero no las puntas en una fracción alta, puede ser
  suficiente para el modelo o puede no serlo, y esto no se contesta con la geometría: se contesta
  mirando los recortes, y en última instancia con el clasificador.
- **¿Conviene informar la inclusión de orejas desglosada por clase?** Probablemente sí —importa más
  en `Disgustado`—, pero el desglose por clase sobre un denominador ya sesgado puede confundir más
  de lo que aclara. Se decide al implementar el reporte.
