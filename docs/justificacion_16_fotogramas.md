# Justificación de la selección de 16 fotogramas por clip

Documento de trabajo (28/10/2026). No modifica notebooks, modelos ni resultados. Las cifras
marcadas «medido» se recalcularon sobre `datos_equinos/resultados/dataset_frames_alineados_v2.csv`
(359 clips del conjunto final).

---

## 0. Dos correcciones previas a la premisa

1. **El modelo final no es el MIL.** Según el criterio de elección aplicado en el Capítulo 5
   (`informe/capitulos/05_resultados.tex`, §«Aplicación del criterio de elección»), el ganador es la
   **fusión tardía de pose y apariencia** (bosque aleatorio de pose + regresión logística sobre la
   media visual). El MIL multimodal es el Candidato 4 y el modelo con que se trabajó hasta el 07/10.
   La justificación tiene que valer para los cuatro candidatos, que comparten la misma entrada de 16
   fotogramas, y no sólo para el MIL.
2. **El valor de 400 ms no está documentado en el repositorio.** No aparece en ningún notebook,
   commit, `openspec/`, `docs/` ni en el plan de trabajo. Lo más cercano es una observación posterior
   en `openspec/project.md` (desde el commit inicial): *«`np.linspace` sobre un clip de 6 s muestrea a
   ~1 frame cada 0.4 s»*, que describe la consecuencia del muestreo y no lo justifica. Por eso el
   texto académico presenta los 400 ms como **supuesto de diseño inicial declarado por los autores**,
   no como dato documentado ni como umbral bibliográfico.

---

## A. Reconstrucción histórica

| Fecha | Etapa | Qué pasó con los 16 fotogramas | Respaldo |
|---|---|---|---|
| — | Plan de trabajo | Clips de **5 a 7 s**; se habla de «microexpresiones dinámicas» y de «capturar la evolución temporal». No fija una cantidad de fotogramas ni una duración mínima de expresión. | `docs/plan_de_trabajo.pdf`, `docs/plan_de_trabajo_corregido.pdf` |
| 13/08 | Primera versión | Aparecen los 16 fotogramas: segmentación de los videos en clips de 5–7 s y **16 fotogramas uniformes por clip con `np.linspace` sobre todo el clip**, extraídos **antes** de estimar la pose. No hay justificación escrita del número. | commit `113b001` («Bloque 3: extracción de 16 frames uniformes por clip»); `02_roi_y_normalizacion.ipynb` (`NUM_FRAMES = 16`, `indices_uniformes`); `openspec/project.md` |
| 13/08–28/08 | Recuperación de fotogramas | Como las posiciones se fijaban sin mirar la calidad, un fotograma con pose fallida se perdía; se intentó recuperarlo por interpolación temporal de keypoints y anotación manual. Sólo 80 de 342 clips tenían sus 16 fotogramas alineados. | `openspec/changes/archive/2026-08-28-interpolacion-temporal-de-keypoints/` |
| 28/08 | Piso de video corto | Se admiten videos enteros más cortos que 5 s (piso de 1–2 s). Los clips pasan a ser de **largo variable (2 a 7 s)**: los mismos 16 fotogramas abarcan distinto tiempo según el clip. | `openspec/changes/archive/2026-08-28-piso-de-video-corto/design.md` |
| 28/08 | Prueba del orden | 141 clips, sólo apariencia: el orden real no mejoró frente al barajado en ninguna de 18 configuraciones. | `openspec/changes/archive/2026-08-28-clasificacion-orientada-a-video/cabezas_temporales.md`; cap. 5 §«La Hipótesis Temporal» |
| 15/09 | MIL multimodal | La bolsa del MIL es de 16 fotogramas. | commit `d872458` |
| 18/09 | **Selección por ventana** | Se invierte el orden: primero se estima la pose en **todos** los fotogramas, después se eligen los 16. Se busca la ventana contigua más larga (≤ 7 s) con ≥ 16 fotogramas válidos (confianza de ojo y hocico ≥ 0,50) y sin huecos > 0,5 s; se reparten 16 posiciones equiespaciadas **dentro de esa ventana** y cada una se ancla al fotograma válido de mayor confianza dentro de una tolerancia de min(0,10 s; paso/3). Sin interpolación ni anotación manual. Clips utilizables: 153 → **359 de 393**. | commit `a6b7bf9`; `equinos/secuencia.py` (docstring y `elegir_frames`); cap. 4 §«Selección de los 16 Fotogramas» |
| 07/10–17/10 | Comparación final | Los cuatro candidatos consumen los mismos 16 fotogramas. Nunca se probó otra cantidad. | `equinos/comparacion.py`; `informe/cifras.tex` (`\Fotogramas = 16`) |

### Respuestas puntuales

- **¿Muestreo uniforme de todo el video al principio?** Sí, sobre todo el *clip* (no el video
  original): `np.linspace(0, total-1, 16)`.
- **¿La versión final sigue siendo uniforme?** Sólo aproximadamente y **dentro de la ventana válida**,
  no del clip: las posiciones ideales son equiespaciadas, pero el anclaje las corre hacia fotogramas
  bien medidos. `auditoria/datos/seleccion_frames.json` lo dice explícitamente: «NO es muestreo
  uniforme sobre el clip ni aleatorio».
- **¿Antes o después de validar la pose?** En la versión final, **después**: la selección sólo elige
  entre fotogramas ya validados. En la primera versión era al revés.
- **¿16 constante en todos los modelos?** En el conjunto final, sí: medido, los 359 clips tienen
  exactamente 16 fotogramas alineados. Matiz: el código admite relajar hasta un mínimo de 12
  (`MINIMO_FRAMES = 12` en `equinos/secuencia.py`), pero ningún clip del conjunto final quedó por
  debajo de 16.
- **La ventana depende de la clase** (medianas: Disgustado 2,87 s; Interesado 5,91 s; Sereno 5,93 s;
  Kruskal-Wallis p ≈ 2·10⁻²⁶; `auditoria/datos/seleccion_frames.json`). Por lo tanto el paso temporal
  entre fotogramas también: mediana 0,19 s en Disgustado y ≈ 0,39 s en las otras dos clases.

---

## B. Justificación matemática

**Notación.** T = duración de la ventana, N = fotogramas, Δt = separación entre observaciones
consecutivas. Con extremos incluidos: Δt = T / (N − 1).

**1. Caso de diseño (T = 7 s, N = 16).** Δt = 7/15 ≈ **0,467 s = 467 ms** > 400 ms. Una expresión de
400 ms puede caer entera entre dos observaciones. **No hay garantía de captura.**

**2. Cuántas muestras harían falta.** Un evento visible durante un intervalo cerrado de duración d
queda muestreado con seguridad si Δt ≤ d. Para d = 0,4 s y T = 7 s:

  N − 1 ≥ T/d = 17,5  ⇒  **N ≥ 19** (Δt = 7/18 ≈ 0,389 s).

| N | Δt con T = 7 s | ¿Δt ≤ 0,4 s? |
|---|---|---|
| 8 | 1,000 s | no |
| 16 | 0,467 s | no |
| 19 | 0,389 s | sí (mínimo ideal) |
| 20 | 0,368 s | sí |
| 32 | 0,226 s | sí |

**3. Lo que sí aporta el muestreo de 16 (cobertura probabilística).** Bajo muestreo ideal y si la
expresión puede empezar en cualquier instante con igual probabilidad, la probabilidad de que al
menos una observación caiga dentro de un evento de duración d < Δt es d/Δt. Para d = 0,4 s:
0,4 / 0,467 ≈ **0,86**. Es una cifra ilustrativa: depende de un supuesto (inicio uniforme) que no
fue medido.

**4. El procedimiento real se aleja del caso ideal.** El anclaje permite correr cada posición hasta
min(0,10 s; Δt/3), y si no hay fotograma válido dentro de esa tolerancia se toma el más cercano;
el único tope duro es el hueco de 0,5 s *sin mediciones válidas*. Medido sobre los 359 clips:

- separación entre fotogramas consecutivos: mediana 0,334 s, p95 0,533 s, **máximo 0,734 s**;
- **267 de 359 clips** (74 %) tienen al menos un intervalo > 0,4 s entre fotogramas elegidos;
- en cambio, en Disgustado (ventanas cortas) el paso mediano es 0,19 s.

Es decir: incluso N = 19 no garantizaría la captura con este procedimiento; habría que acotar
también el corrimiento del anclaje.

**5. Lo que la bibliografía dice de la duración.** Tomberg et al. (2023) codificaron con EquiFACS, a
25 fotogramas/s, expresiones faciales de caballos y reportaron una duración media de **202 ± 115 ms**,
con 66,5 % de las expresiones por debajo de 500 ms; también describen modos «macro» de varios
segundos (p. ej. AU17 ≈ 2,2 s). El umbral de 500 ms que usan para «microexpresión» proviene de la
literatura humana (Yan et al., 2013), no de una definición propia de EquiFACS. Consecuencias:

- **No hay base bibliográfica para 400 ms como duración mínima** de una expresión facial equina;
  la evidencia disponible sugiere que muchos movimientos faciales del caballo son **más breves**.
- El valor sólo puede defenderse como **supuesto de diseño**: una escala aproximada para dimensionar
  la separación entre observaciones.
- El argumento coherente para nuestras etiquetas es otro: la etiqueta se asigna a un **tramo**
  que la etiquetadora recortó porque en él se manifestaba la expresión, y se hereda a los clips
  (cap. 2, §«Procedimiento de Etiquetado»), no a un evento puntual. 16 observaciones repartidas
  muestrean ese tramo en varios momentos. (Que la expresión se mantenga durante todo el tramo es un
  supuesto del etiquetado, no algo medido.)

**6. Cobertura ≠ detección ≠ clasificación.**

- *Cobertura temporal*: que haya una observación dentro del intervalo en que la expresión está
  presente. Es una propiedad del muestreo.
- *Detección*: que ese fotograma muestre la expresión de forma visible (cara de perfil, sin
  desenfoque, orejas en el encuadre, keypoints correctos). La cobertura es necesaria pero no
  suficiente: el umbral de confianza 0,50 es un piso, no una garantía de nitidez.
- *Clasificación*: que el modelo asigne la clase correcta a partir de las representaciones
  agregadas. Depende de las características, del modelo y de la calidad de las etiquetas, y en
  nuestro caso también del confusor de filmación.

---

## C. Texto académico (para `04_desarrollo.tex`)

### Dónde integrarlo

1. **`informe/capitulos/04_desarrollo.tex:124`** — reemplazar el párrafo agregado en `b015a53`
   («El número de \Fotogramas{} fotogramas se heredó…») por la nueva subsección (abajo), colocada
   inmediatamente después del párrafo del algoritmo (línea 122). El párrafo actual tiene una
   imprecisión: dice «separadas por unos 0,4 s», cuando el paso ideal es 0,47 s y el real llega a
   0,73 s.
2. **`informe/capitulos/05_resultados.tex`, §«Limitaciones» (línea ~216)** — agregar un ítem breve
   (propuesto abajo).
3. **`informe/capitulos/06_conclusiones.tex`, §«Trabajo Futuro» (línea ~41)** — opcional: un ítem
   sobre evaluar la cantidad de fotogramas.
4. **`informe/referencias.bib`** — agregar `tomberg2023` y `yan2013` (BibTeX abajo).

No se toca el Capítulo 3: §«Modelos Sensibles e Invariantes al Orden Temporal» ya establece la
distinción que el texto nuevo invoca.

### Texto LaTeX

```latex
\subsubsection{Justificación de la Selección de \Fotogramas{} Fotogramas por Clip}
Un clip de varios segundos contiene cientos de fotogramas, mientras que la configuración facial que
justifica su etiqueta puede hacerse visible durante una fracción de ese tiempo y en un instante que
no se conoce de antemano. Representar el clip exige entonces elegir una cantidad limitada de
fotogramas que cubra distintos momentos sin procesar todos los originales.

En el diseño inicial se tomó como escala de referencia una expresión de unos 400~ms. Ese valor fue
un supuesto de diseño de los autores y no un umbral establecido para la especie: no se encontró
bibliografía que lo fije como duración mínima de una expresión facial equina. Por el contrario, al
codificar con EquiFACS videos de caballos a 25 fotogramas por segundo, \textcite{tomberg2023}
reportaron una duración media de los movimientos faciales de $202 \pm 115$~ms, con dos tercios de
ellos por debajo de 500~ms, el límite que la literatura humana usa para definir una
microexpresión \parencite{yan2013}. Este trabajo no analiza microexpresiones en ese sentido técnico
ni unidades de acción aisladas: la etiqueta se asigna a un tramo de video que la etiquetadora
recortó porque en él se manifestaba la expresión, y se hereda a sus clips (Capítulo~2). Los
\Fotogramas{} fotogramas buscan observar ese tramo en varios momentos, no registrar un movimiento
breve.

Los clips considerados en el diseño duraban hasta unos 7~s. Se decidió distribuir \Fotogramas{}
observaciones de forma equiespaciada a lo largo del clip para no concentrar la muestra al
principio, en el medio o al final, ya que se desconocía en qué instante se manifestaba la expresión
relevante. Este reparto amplía la cobertura temporal, pero no garantiza capturar un evento de
400~ms. Con $N$ observaciones equiespaciadas en una ventana de duración $T$, incluidos los
extremos, la separación entre observaciones consecutivas es
\[
\Delta t = \frac{T}{N-1},
\]
y para $T = 7$~s y $N = \Fotogramas{}$ resulta $\Delta t = 7/15 \approx 0{,}47$~s, mayor que 400~ms:
una expresión de esa duración podría ocurrir íntegramente entre dos observaciones. Para que la
separación no superara los 400~ms bajo un muestreo ideal harían falta al menos 19 fotogramas
($\Delta t = 7/18 \approx 0{,}39$~s). En el procedimiento implementado la separación es además
irregular, porque cada posición se ancla al fotograma válido más cercano: en los \ClipsTotal{}
clips del conjunto final la separación entre fotogramas consecutivos tuvo una mediana de 0,33~s y un
máximo de 0,73~s, y en 267 clips al menos un intervalo superó los 400~ms. Por otra parte, una
separación suficientemente pequeña es sólo una condición de cobertura: no asegura que el fotograma
capturado tenga la nitidez, la orientación y la calidad de puntos clave necesarias para que la
expresión sea visible, ni que el modelo la clasifique correctamente.

Fijar un número de fotogramas tuvo además un motivo práctico. Da a cada clip una representación de
tamaño fijo, de \Fotogramas{}$\times$512 características visuales y \Fotogramas{} vectores de pose,
independiente de su duración; acota el costo de la extracción de características, y evita tomar
fotogramas contiguos, casi idénticos entre sí, que agregarían redundancia sin información nueva.
El número también condiciona qué clips son utilizables, porque la ventana debe reunir esa cantidad
de fotogramas válidos: exigir más fotogramas sólo puede mantener o reducir la cantidad de clips, y
con \Fotogramas{} se conservaron \ClipsTotal{} de los 393 candidatos.

La cantidad se fijó en la primera versión del \textit{pipeline}, cuando los fotogramas se elegían
por muestreo uniforme sobre todo el clip y antes de estimar la pose, y se mantuvo en las versiones
posteriores para que las comparaciones entre representaciones, modelos y protocolos de validación
se hicieran sobre la misma entrada. Lo que cambió fue el procedimiento: en la versión final las
posiciones se reparten de forma equiespaciada dentro de la ventana válida, no dentro del clip
completo, y sólo entre fotogramas cuya pose ya fue validada. El valor \Fotogramas{} no se optimizó:
no se evaluaron otras cantidades, y la única comparación medida es contra un solo fotograma
(Capítulo~5). Es una decisión de diseño razonada, no una cantidad que se haya demostrado óptima.

Finalmente, representar distintos instantes no equivale a modelar el orden temporal. Ninguno de
los cuatro candidatos usa la secuencia de los fotogramas: la línea de base promedia sus
representaciones; el bosque aleatorio, y con él la fusión, resume la pose de la ventana en rasgos
agregados, y el MIL \parencite{ilse2018} trata los \Fotogramas{} fotogramas como una bolsa y
aprende a ponderar los más informativos mediante una atención sin codificación de posición, por lo
que es invariante a su orden. El reparto equiespaciado favorece que la bolsa incluya momentos
distintos del clip, pero los \Fotogramas{} fotogramas no permiten afirmar que el sistema aprenda la
evolución temporal de la expresión, hipótesis que con los datos disponibles no quedó respaldada
(Capítulo~5).
```

### Ítem para Limitaciones (`05_resultados.tex`)

```latex
  \item \textit{Cantidad de fotogramas no evaluada.} Los \Fotogramas{} fotogramas por clip se
  heredaron del diseño inicial y no se compararon con otras cantidades. Con ventanas de hasta 7~s,
  la separación entre fotogramas puede superar los 400~ms, de modo que expresiones breves pueden
  quedar sin observar.
```

### BibTeX (verificado)

```bibtex
@article{tomberg2023,
  author  = {Tomberg, C. and Petagna, M. and {de Selliers de Moranville}, L.-A.},
  title   = {Horses (\textit{Equus caballus}) facial micro-expressions: insight into discreet social information},
  journal = {Scientific Reports},
  volume  = {13},
  pages   = {8625},
  year    = {2023},
  doi     = {10.1038/s41598-023-35807-z}
}

@article{yan2013,
  author  = {Yan, Wen-Jing and Wu, Qi and Liang, Jing and Chen, Yu-Hsin and Fu, Xiaolan},
  title   = {How fast are the leaked facial expressions: The duration of micro-expressions},
  journal = {Journal of Nonverbal Behavior},
  volume  = {37},
  number  = {4},
  pages   = {217--230},
  year    = {2013},
  doi     = {10.1007/s10919-013-0159-8}
}
```

Referencias de contexto (verificadas, opcionales; no son fundamento del número 16):

- Wang et al. (2016), *Temporal Segment Networks*, ECCV, pp. 20–36: divide el video en un número
  fijo de segmentos y toma uno por segmento; argumenta que fotogramas consecutivos son muy
  redundantes y que el costo queda independiente de la duración. Es el antecedente más cercano a
  «N fijo, repartido en todo el video».
- Tran et al. (2015), *C3D*, ICCV, pp. 4489–4497: usa clips de 16 fotogramas **contiguos**. Es otra
  cosa (≈ 0,5 s a 30 fps); **no** sirve para justificar nuestro 16 y conviene no citarlo como tal.
- Sultani et al. (2018), CVPR: MIL sobre video con 32 segmentos por video como instancias de la
  bolsa. Ejemplo de MIL con N fijo elegido por diseño.

---

## D. Respuesta para la defensa (≈ 1 minuto)

> **«¿Por qué exactamente 16 y no 8, 20 o 32?»**
>
> Dieciséis es una decisión de diseño, no una cantidad que hayamos demostrado óptima; no
> comparamos otros valores y lo decimos en el informe. La elegimos al principio con una idea
> concreta: los clips duraban hasta 7 segundos y no sabíamos en qué momento aparecía la
> expresión, así que repartimos 16 observaciones a lo largo del clip para cubrir distintos
> momentos. Eso da una separación de unos 0,47 segundos. Tomamos como referencia expresiones
> de unos 400 milisegundos, y por eso sabemos que 16 no garantiza capturarlas; harían falta
> al menos 19, y aun así la cobertura no asegura que el fotograma sea nítido ni que el modelo
> acierte. Con 8 la separación sería de un segundo; con 32 duplicaríamos el costo, tomaríamos
> fotogramas muy redundantes y, como exigimos que todos tengan pose válida, perderíamos clips.
> Después mantuvimos 16 para que todas las comparaciones de modelos usaran la misma entrada.
> Y una aclaración: usar 16 fotogramas no significa modelar el tiempo; nuestros modelos los
> tratan como un conjunto, y la prueba del orden no mostró mejora.

*Si preguntan por los 400 ms:* «Fue un supuesto nuestro. La bibliografía específica (Tomberg y
colaboradores, 2023) muestra que muchos movimientos faciales del caballo duran menos, alrededor de
200 ms. Por eso no clasificamos movimientos breves sino el estado sostenido del clip.»

*Si preguntan por qué 32 perdería clips:* es cierto por construcción (la ventana debe tener al menos
N fotogramas válidos), pero **cuántos** se perderían no se midió; no dar una cifra.

---

## Fuentes consultadas

- Tomberg et al. (2023): https://pmc.ncbi.nlm.nih.gov/articles/PMC10224940/
- Yan et al. (2013): https://www.doi.org/10.1007/s10919-013-0159-8
- Wang et al. (2016): https://arxiv.org/abs/1608.00859
- Tran et al. (2015): https://arxiv.org/abs/1412.0767
- Sultani et al. (2018): https://arxiv.org/abs/1801.04264
- Wathan et al. (2015), EquiFACS: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0131738
