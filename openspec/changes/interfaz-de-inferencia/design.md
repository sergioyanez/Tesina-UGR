# Diseño — interfaz de inferencia

## Context

El objetivo 3 dejó un clasificador evaluado con honestidad y **un resultado negativo**: F1 macro
0,626 ± 0,065 contra un criterio declarado de 0,691, con la hipótesis del trabajo sin sostén. El
objetivo 4 pide poner ese modelo en manos de alguien.

Cuatro hechos medidos ordenan todo lo que sigue. Ninguno es una suposición:

1. **La interfaz va a negarse más veces de las que va a predecir.** El modelo consume clips
   íntegros —16 frames alineados—, y sobre el dataset propio sólo **141 de 368 clips (38 %)** lo
   fueron. No hay motivo para esperar que un video traído por un tercero rinda mejor.
2. **Cuando dice `Disgustado`, acierta una de cada tres veces.** De la matriz de confusión fuera de
   muestra: precisión 0,33 para `Disgustado`, 0,79 para `Interesado`, 0,76 para `Sereno`.
3. **Cada clip cuesta ~35 s en GPU** (15 s de detector + 20 s de pose), antes del modelo, que tarda
   milisegundos. Sin GPU es varias veces más.
4. **No existe un modelo de producción.** Los cuatro pesos guardados son de validación cruzada.

## Goals / Non-Goals

**Goals:**

- Que alguien que no es del equipo pueda dar un video y obtener una respuesta.
- Que esa respuesta venga con su confiabilidad medida, en el mismo lugar y con el mismo peso.
- Que la interfaz se niegue cuando el preprocesamiento no alcanza, en vez de predecir sobre un
  clip contaminado.
- Que la lógica sea la misma que la evaluada, no una reimplementación parecida.

**Non-Goals:**

- Diagnóstico veterinario. La interfaz no lo es y tiene que decirlo.
- Reentrenar desde la interfaz, cuentas de usuario, lote, despliegue público.
- Mejorar el modelo. Lo que haría falta son ejemplares, y eso no es trabajo de este change.

## Decisión 1 — La confiabilidad es parte de la salida principal, no una nota al pie

**Es la decisión central del change.** Una pantalla que devuelve «Disgustado, 87 %» es
indistinguible, para quien la mira, de una herramienta validada. Esta no lo es.

Lo que se muestra junto a cada predicción, siempre:

- **La precisión medida de esa clase**, que es el número que le importa al usuario: *cuando el
  modelo dijo esto, ¿cuántas veces acertó?* `Disgustado` 0,33 · `Interesado` 0,79 · `Sereno` 0,76.
- **Que el modelo no alcanzó el criterio declarado** para considerarse útil.
- Las probabilidades de las tres clases, no sólo la ganadora.

**Alternativa descartada — un descargo legal al pie.** No cambia lo que la persona lee ni lo que
recuerda. Si el número de confiabilidad no está al lado del resultado, no está.

**Alternativa descartada — no mostrar probabilidades**, para no dar falsa precisión. Se descarta al
revés: ocultarlas deja la etiqueta sola, que es la lectura más engañosa de todas. El problema no es
mostrar el 87 %, es mostrarlo sin decir que el modelo acierta un tercio cuando dice esa clase.

**Por qué precisión y no recall.** El recall contesta «de los `Disgustado` reales, ¿cuántos
encontró?», que le sirve a quien evalúa el modelo. Al usuario le importa la pregunta inversa, y son
números muy distintos: `Disgustado` tiene recall 0,35 y precisión 0,33, pero `Interesado` tiene
recall 0,88 y precisión 0,79.

## Decisión 2 — Negarse es una respuesta de primera clase

Si el preprocesamiento no produce 16 frames alineados, la interfaz **no predice**. Devuelve el
motivo: sin detección del animal, sin keypoints faciales suficientes, o clip demasiado corto.

Esto no es un caso de borde. Con el 38 % de clips íntegros medido sobre el dataset propio, **la
negativa es el resultado más probable**, y la interfaz tiene que tratarla como una salida normal y
explicada, no como un error.

**Alternativa descartada — predecir con los frames que haya.** Es lo que haría un sistema que
quiere parecer útil. Un clip a medio alinear mezcla recortes por keypoints con recortes de caja, y
a qué frame le toca cuál correlaciona con el ejemplar: es exactamente la muestra contaminada que la
spec de clasificación prohíbe. Predecir ahí da una etiqueta cuya confiabilidad no está medida.

**Alternativa descartada — rellenar hasta 16 con interpolación agresiva.** El relleno temporal
tiene un tope validado (`MAX_HUECO_INTERPOLADO = 1`) y subirlo para que entren más videos sería
inventar posiciones que nadie midió, justo en el momento en que el resultado se le muestra a una
persona.

## Decisión 3 — El modelo de producción se entrena sobre los 141, y su métrica sigue siendo la de validación cruzada

Se entrena una vez, sobre todos los clips íntegros, y se persiste con su procedencia. Los cuatro
pesos de validación cruzada se conservan: son los que respaldan la métrica.

**La métrica del modelo de producción es 0,626 ± 0,065**, la que salió de la validación cruzada. No
se vuelve a medir sobre los datos con los que se entrenó, y ese número no mejora por entrenar con
más datos: mejoraría el modelo, pero no hay forma honesta de medirlo sin dejar material afuera.

**Alternativa descartada — desplegar uno de los cuatro modelos de fold.** Cada uno vio el 75 % del
material y elegir cuál sería arbitrario. **Alternativa descartada — promediar los cuatro.** Es un
ensamble, no el modelo evaluado; su métrica no es la que se reportó.

## Decisión 4 — Una interfaz, no un notebook más

Los notebooks 00–03 son **etapas del pipeline**: procesan el dataset entero por lote, bajo
`papermill`, con checkpoints. La inferencia sobre un video suelto es otra cosa, y meterla en esa
cadena la ensuciaría con un caso que no comparte ni entrada ni modo de ejecución.

La interfaz es una **aplicación local** sobre `equinos/inferencia.py`, que es donde vive la lógica.
Corre en la máquina que tiene la GPU, que es la única donde los 35 s por clip son tolerables.

**Alternativa descartada — un notebook `04_inferencia.ipynb`.** Sería consistente con 00–03 pero no
es «una aplicación o interfaz para el uso del modelo» en ningún sentido útil: seguiría requiriendo
que quien la use lea y ejecute celdas.

**Alternativa descartada — desplegarla en la web.** Arrastra YOLO, DeepLabCut y una GPU. Fuera de
alcance para septiembre, y sin valor para la defensa.

**Qué framework concreto** —Gradio, Streamlit u otro— se decide al implementar, con una condición:
la lógica va en `equinos/inferencia.py` y la interfaz es una envoltura delgada. Cambiar de
framework no puede obligar a reimplementar nada.

## Decisión 5 — La inferencia reutiliza las etapas 01 y 02, no las reimplementa

`equinos/inferencia.py` orquesta el mismo código que produjo el dataset evaluado: segmentación,
detección, pose, alineación y extracción de características.

Es la condición para que el número reportado signifique algo. Una reimplementación «equivalente»
que difiera en el encuadre, en el umbral de confianza o en el orden de los frames produciría
entradas de otra distribución, y la confiabilidad medida dejaría de aplicar sin que nada falle.

La consecuencia a aceptar es que la interfaz hereda **toda** la cadena de dependencias, incluida la
restricción de `numpy<2` de DeepLabCut.

## Riesgos

- **Que la interfaz lave el resultado negativo.** → *Mitigación:* Decisión 1, y una revisión
  explícita antes de cerrar: mirar la pantalla y preguntarse si alguien que no leyó la tesina
  entendería que el modelo acierta un tercio de los `Disgustado`.
- **Que la tasa de negativa la vuelva inútil en la práctica.** Con 38 % de clips íntegros, dos de
  cada tres videos no obtienen respuesta. → *Mitigación:* medirla sobre videos nuevos e informarla
  como resultado del objetivo 4, no esconderla. Si es muy alta, **es un hallazgo sobre el
  preprocesamiento**, no un defecto de la interfaz.
- **35 s por clip.** → *Mitigación:* señales de progreso por etapa y la expectativa dicha de
  antemano. Una interfaz que parece colgada es peor que una lenta que avisa.
- **Que se use como diagnóstico.** → *Mitigación:* está en los no-objetivos y va escrito en la
  interfaz misma.

## Open Questions

- **¿Qué framework?** Se decide al implementar; la Decisión 4 fija que sea intercambiable.
- **¿La interfaz acepta un video largo y lo segmenta, o exige un clip de 5–7 s?** Segmentar es más
  amable y reutiliza la etapa 01; exigir el clip es más simple y más predecible. Se decide con el
  primer video de prueba real.
- **¿Se informa la tasa de negativa acumulada de la sesión?** Sería honesto y no cuesta nada, pero
  puede confundir a quien sube un solo video.
