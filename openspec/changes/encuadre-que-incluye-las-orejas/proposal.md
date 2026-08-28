# Propuesta — el recorte canónico debe incluir las orejas

## A qué objetivo contribuye

**Objetivo específico 2** del plan de trabajo: *desarrollar un pipeline de preprocesamiento —
extracción de fotogramas clave y detección de la región de interés (ROI) facial*. Este change
corrige el **encuadre** de esa ROI, que el objetivo 2 dio por cerrado el 28/08/2026 sobre un
criterio —la cobertura de alineación— que no mira qué parte de la cara queda dentro del recorte.

Bloquea el objetivo 3: entrenar con el encuadre actual y corregirlo después obliga a repetir el
entrenamiento entero.

## El problema, medido

La observación es de Héctor R. Liceaga (`analisis_y_tareas.md`, rama `hector`): en los equinos **la
posición de las orejas —hacia atrás o aplanadas— es el indicador principal de disgusto y estrés**,
y el recorte de 224×224 podría estar dejándolas fuera.

No hizo falta reinferir nada para verificarlo. `superanimal_quadruped` expone `left_earbase`,
`left_earend`, `right_earbase` y `right_earend`, y esos keypoints ya están en `pose_completa.csv`
para los 5.888 frames de la corrida vigente. Proyectándolos por la misma transformada de similitud
que usa la etapa 02, sobre los 5.586 frames con par ojo-hocico detectado:

| Keypoint | Dentro del recorte 224×224 |
|---|---:|
| `left_earbase` / `right_earbase` | 19,8 % / 20,0 % |
| `left_earend` / `right_earend` | **2,8 % / 3,3 %** |
| **Todos los keypoints de oreja** | **11,7 %** |
| **Vector base→punta legible** (la métrica que decide) | **4,3 %** |

**El 88 % de los keypoints de oreja cae fuera del recorte, y las puntas quedan fuera en el 97 % de
los casos.** Como la orientación de la oreja se lee del vector base→punta, el rasgo queda legible
en apenas el 4,3 % de los frames: el modelo del objetivo 3 nunca vería lo que la etología equina
señala como el indicador más informativo para la clase de interés clínico.

La causa es geométrica y está cuantificada. En unidades de la distancia ojo-hocico `d`:

```
Punta de oreja, por encima del ojo:   p50 = 0,81 d    p75 = 0,99 d    p90 = 1,23 d
El recorte actual muestra:            0,33 d por encima del ojo
Semiancho lateral de las orejas:      p50 = 0,57 d    (el recorte muestra 0,83 d)
```

**El déficit es vertical, no lateral.** El recorte es cuadrado, pero lo que falta es aire por
encima del ojo y lo que sobra es campo a los costados. Esa asimetría es la que hace que el
problema no se pueda resolver moviendo `FRAC_OJO`.

## Por qué ahora, y por qué es barato

Ahora, porque **es el último momento en que sale barato**: el objetivo 3 todavía no entrenó. Después
de entrenar, corregir el encuadre invalida la corrida completa.

Barato, porque el diseño de la etapa 02 ya separó la parte cara de la analítica —*"la parte cara
(inferir) se paga una vez, y la parte analítica (elegir) se rehace en segundos"*—. Concretamente:

- **No se reinfiere DeepLabCut** (~1 h). `pose_completa.csv` y `keypoints_faciales.csv` están en disco.
- **No se re-anota a mano.** El par de anclaje sigue siendo ojo-hocico, así que las 14 anotaciones
  de `keypoints_manuales.csv` siguen siendo válidas tal cual. Las orejas entran por **campo de
  visión**, no por anclaje: no se agrega dependencia de un keypoint nuevo, que además sólo tiene
  ~60 % de disponibilidad por oreja.
- **La escala se mueve a favor** (`d` mapea a 134 px hoy y a 149 px en el encuadre adoptado, que
  recorta campo lateral). Los topes de la interpolación están medidos en píxeles dentro del
  recorte, pero su criterio es una comparación contra una línea de base medida del mismo modo, así
  que es invariante a la escala: se revalida para confirmarlo.

Lo que se re-ejecuta es el warp y la escritura de los 5.888 JPEG, más el recálculo de cobertura y
composición. Minutos.

## Qué cambia

- **La ROI deja de ser cuadrada.** Pasa a ser un rectángulo vertical en el marco de la cabeza, con
  el eje ojo-hocico sobre la vertical. Es **BREAKING** respecto de la spec vigente, que exige
  recortes cuadrados, y respecto de los artefactos de la corrida actual.
- **El encuadre se declara como campo de visión en unidades de `d`** —aire sobre el ojo, aire bajo
  el hocico, semiancho lateral— en vez de como dos fracciones sobre un lado fijo. Es la
  parametrización que hace visible la asimetría del problema.
- **El vector de oreja legible pasa a ser criterio de aceptación de la etapa**, medible sobre los
  keypoints ya persistidos, junto a la cobertura de alineación. La unidad es el vector base→punta y
  no el conteo suelto de keypoints, porque lo que informa es la orientación de la oreja.
- **El encuadre concreto se elige con un barrido**, no de antemano. El criterio se declara antes de
  medir y se ancla en el encuadre actual como línea de base. **Adoptado: 1,00 `d` de aire sobre el
  ojo, 0,30 bajo el hocico, ±0,75 lateral; destino 224×343.**
- **Cada frame registra la fracción de su recorte que salió de relleno** (`relleno_fraccion`), del
  mismo modo que ya registra `origen_alineacion`, para poder ablacionar su efecto.

## Lo que NO cambia

El par de anclaje (ojo-hocico), la transformada de similitud, la escala isótropa, el camino de
degradación por caja, la etapa 01, la inferencia de pose y las anotaciones manuales.

## Alcance

**Entra:** la parametrización del encuadre, el barrido y la elección con evidencia, la métrica de
inclusión de orejas, la columna `relleno_fraccion`, la re-ejecución del warp, el recálculo de
cobertura y composición, la revalidación de los topes de interpolación y la actualización de
`project.md` y `README.md`.

**No entra:** cambiar el par de anclaje; usar keypoints de oreja como puntos de alineación; el
`imagededup` de la rama `hector`, que es un change aparte; y el objetivo 3, que arranca después.

## Capacidades

### Nuevas

Ninguna. El cambio es de requisitos dentro de una capacidad existente.

### Modificadas

- `preprocesamiento-facial`: el requisito **«Región de interés cuadrada»** se reemplaza por uno de
  encuadre rectangular que incluye las orejas; el requisito de **reporte de cobertura** suma la
  inclusión de orejas como criterio de aceptación; y el de **trazabilidad** suma la fracción de
  relleno por frame.

## Riesgo principal

**El encuadre que incluye las orejas cuesta relleno.** Ampliar el campo de visión hace que el
rectángulo canónico se salga más seguido del cuadro de origen, y `BORDER_REPLICATE` inventa
píxeles. Ese relleno correlaciona con la distancia de filmación, que correlaciona con el ejemplar,
que predice la clase: es exactamente el atajo que el encuadre canónico existe para evitar.

Es una compensación real y hay que medirla, no suponerla. El encuadre actual ya parte de un 14,1 %
de relleno mediano, así que **la línea de base contra la que anclar el criterio no es cero**, y hay
al menos un encuadre candidato que mejora las orejas sin empeorar el relleno. Cuál se adopta lo
decide el barrido.

Dos sesgos conocidos de la medición que abre este change, que deben acompañar cualquier número que
se informe:

1. **La inclusión de orejas se mide sólo donde la oreja se detectó** (78,5 % de los frames con par
   ojo-hocico). Si la oreja está fuera del cuadro tiende a no detectarse, así que el denominador
   excluye justo los peores casos y **todos los encuadres salen favorecidos por igual**.
2. **El relleno se estima muestreando el recorte con una rejilla**, midiendo el rectángulo y no su
   caja envolvente. Sólo queda la discretización de la rejilla, que afecta a todos los encuadres
   por igual.

## Impacto

- `02_roi_y_normalizacion.ipynb`: celda de parámetros, `destino_canonico()`, el paso de recorte y
  alineación, y los pasos de cobertura.
- `tests/test_geometria.py`: las pruebas que fijan el recorte cuadrado.
- `equinos/reporte.py`: lectura de la cobertura, que suma la inclusión de orejas.
- `dataset_frames_alineados.csv`: nueva columna `relleno_fraccion`; los 5.888 JPEG se regeneran.
- `openspec/specs/preprocesamiento-facial/spec.md`, `openspec/project.md`, `README.md`.
- **Aguas abajo:** el change `clasificacion-orientada-a-video` asume entradas de 224×224 y debe
  actualizarse a la nueva forma antes de implementarse.
