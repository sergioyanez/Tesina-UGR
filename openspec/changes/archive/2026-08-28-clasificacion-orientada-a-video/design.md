# Diseño — clasificación orientada a video

## Contexto

141 clips íntegros de 73 ejemplares distintos, tres clases desbalanceadas 21/56/64. Ese tamaño es la
restricción que ordena todas las decisiones de abajo; ninguna es una elección de arquitectura
"por mérito" sino por lo que 141 muestras permiten sostener.

Un dato que conviene tener presente al leer los resultados: hay un techo de exactitud que no
depende del modelo. En `178_20260701-081313` la toma etiquetada `Interesado` termina donde
empieza la etiquetada `Disgustado` — son contiguas, y los frames de esa frontera son casi
idénticos con clases opuestas. Es ruido irreducible en el borde de un evento continuo, medido y
documentado en el change `orientacion-de-video-en-origen`.

## Decisión 1 — Backbone congelado, cabeza temporal chica

Un extractor preentrenado en ImageNet procesa cada uno de los 16 frames sin actualizarse, y una
cabeza entrenable modela la secuencia. Con el backbone congelado los parámetros entrenables
quedan en el orden de 10⁴–10⁵ contra las 141 muestras; con el backbone descongelado serían 10⁷ y
el sobreajuste es inmediato.

**Alternativa descartada — entrenar una red de video de punta a punta (I3D, SlowFast, VideoMAE).**
Es lo que usaría la literatura del área, y con 141 muestras memoriza el conjunto de
entrenamiento en pocas épocas. Su ventaja —aprender características espacio-temporales propias
del dominio— sólo se materializa con órdenes de magnitud más de datos.

**Alternativa descartada — fine-tuning parcial de las últimas capas.** Es el punto medio
razonable y queda como extensión si el backbone congelado da resultados prometedores. No se
adopta de entrada porque agrega una decisión más —cuántas capas descongelar— que con este
tamaño de conjunto de validación no se puede tomar con evidencia.

## Decisión 2 — Modelar la secuencia, y poder demostrar que hace falta

La hipótesis del trabajo dice que la expresión es un **evento dinámico** y que la evolución
temporal es lo que permite clasificarla. Eso no se asume: se pone a prueba.

Se entrena también un clasificador que ve **un solo frame** del clip y vota, sin ninguna
información temporal. Si el modelo secuencial no le gana, la hipótesis no queda sostenida por la
evidencia de este dataset, y eso hay que escribirlo tal cual.

La cabeza temporal concreta —promedio de características, GRU, atención sobre los 16 pasos— se
elige dentro del change, comparando sobre los mismos folds. Es una decisión menor comparada con
la anterior y no merece fijarse de antemano.

## Decisión 3 — La partición es por ejemplar, y es lo primero que se construye

`GroupKFold` agrupando por `id_caballo`, con **4 folds**. Verificado sobre los datos reales:
ninguna clase queda vacía en ningún fold, y el fold más chico tiene 3 clips de `Disgustado` en
test contra los 2 que deja una partición de 5.

Nunca por clip ni por frame: **10 de los 73 ejemplares tienen clips íntegros en más de una
clase** y varios aportan hasta 3 clips, así que particionar por clip pondría al mismo caballo en
entrenamiento y en test. Eso no da error, da una métrica optimista y falsa. Por el mismo motivo
el grupo es el ejemplar y no el par ejemplar-clase.

**Alternativa descartada — hold-out simple.** Con 15 ejemplares en la clase minoritaria, un
único corte deja la métrica a merced de qué tres caballos cayeron en test. La validación cruzada
no mejora el modelo, mejora lo que se puede afirmar sobre él.

## Decisión 4 — Tres líneas de base, declaradas antes de entrenar

Sin ellas un F1 macro de 0,55 no significa nada: puede ser un modelo que aprendió algo o uno que
descubrió la clase mayoritaria.

1. **Clase mayoritaria.** El piso absoluto. Sobre esta distribución da F1 macro ≈ 0,21.
2. **Un solo frame, sin información temporal.** Responde si la dimensión temporal aporta.
3. **Sólo la identidad del ejemplar**, sin mirar la imagen: un clasificador que recibe
   únicamente `id_caballo`. **Es la línea de base más importante y la menos habitual.** Mide
   cuánta señal hay en el confounder que se viene combatiendo desde el principio. Si predice
   bien, es que la clase está determinada por el individuo en este dataset, y cualquier modelo
   que mire la imagen puede estar aprendiendo eso mismo por otra vía.

## Decisión 5 — La ablación por `origen_alineacion` es parte del resultado, no un extra

Se entrena y evalúa dos veces: con los 141 clips íntegros, y con los 82 que lo son por detección
propia. Se reportan los dos.

El segundo tiene menos datos y `Disgustado` cae a 5 ejemplares, así que probablemente rinda
peor. **Eso no invalida el primero**: lo que muestra es cuánto del resultado se apoya en
posiciones estimadas, que es exactamente lo que hay que poder contestar. La columna se agregó en
`interpolacion-temporal-de-keypoints` para esto.

## Decisión 6 — Métricas: F1 macro y balanced accuracy, con desvío entre folds

La accuracy global no se usa. Con 15 % / 40 % / 45 %, un clasificador que nunca prediga
`Disgustado` llega a 85 % de accuracy sin haber aprendido nada, y `Disgustado` es justamente la
clase de interés clínico.

Se reporta la media **y el desvío entre folds**. Con 3 clips de la clase minoritaria en el fold
más chico, la media sola oculta que el resultado puede variar mucho según qué caballos tocaron.

Se acompaña siempre de la **matriz de confusión agregada** sobre las predicciones fuera de
muestra: es lo que muestra si el modelo confunde `Disgustado` con `Interesado` —que sería
esperable, y consistente con el caso `178`— o si directamente nunca lo predice.

## Decisión 7 — El criterio de aceptación se declara antes y se ancla en las líneas de base

Es la lección del change anterior, donde el umbral se declaró antes de medir pero sin referencia
física, resultó imposible de cumplir, y hubo que re-declararlo dejando constancia.

**Criterio declarado:** el modelo se considera útil si su F1 macro medio supera al de las tres
líneas de base **en al menos un desvío estándar entre folds**. Anclado en las líneas de base y no
en un número redondo, porque un 0,60 puede ser excelente o irrelevante según contra qué se lea.

El resultado se informa aunque no se cumpla.

## Riesgos

- **El modelo no supera las líneas de base.** Es un resultado posible con 141 muestras. Se
  publica igual, y se acompaña de qué haría falta para cambiarlo — que es material, no
  arquitectura.
- **La línea de base de identidad predice bien.** Sería la peor noticia: significaría que la
  clase está fuertemente determinada por el individuo en este dataset, y que ningún modelo puede
  distinguir expresión de ejemplar con este material. Hay que medirlo antes de invertir en
  arquitectura.
- **Sobreajuste al conjunto de validación por probar muchas configuraciones.** Con 141 muestras,
  probar veinte variantes y quedarse con la mejor es elegir ruido. Se acota de entrada la
  cantidad de configuraciones a comparar y se deja registrado cuántas se probaron.
- **El techo por frames de frontera.** Documentado; no se puede eliminar, sí declarar.
- **La oreja no está disponible de forma sostenida, y menos que nada en `Disgustado`.** Medido
  antes de entrenar con `disponibilidad_orejas.py`, sobre los 2.256 frames de los 141 clips
  íntegros y con el encuadre vigente:

  | | oreja dentro | fuera del recorte | no detectada |
  |---|---:|---:|---:|
  | `Disgustado` | 48,5 % | 9,5 % | 40,5 % |
  | `Interesado` | 70,5 % | 2,3 % | 26,6 % |
  | `Sereno` | 42,8 % | 10,8 % | **46,3 %** |
  | **total** | 54,7 % | 7,3 % | 37,6 % |

  El change `encuadre-que-incluye-las-orejas` resolvió lo que le correspondía —las orejas fuera
  del cuadro pasaron del 59,8 % al 7,3 %—. Lo que queda es un techo de **37,6 % de frames sin
  oreja detectada**, invariante al encuadre porque la detección ocurre sobre el frame original,
  antes de recortar: oclusión, desenfoque, o la oreja apuntando fuera de cámara.

  A nivel de **clip**, que es la unidad que consume el modelo, `Disgustado` es la peor: de sus 21
  clips íntegros sólo 8 tienen la oreja en la mitad de sus frames y **ninguno la tiene en los
  16**, contra 42 de 56 en `Interesado`. Es además la clase con menos ejemplares, así que la
  limitación pega donde más duele.

  **Cómo se usa este número, y cómo no.** Es una explicación candidata declarada de antemano por
  si `Disgustado` no se separa — no una excusa preparada, y tampoco un filtro: los 141 clips
  entran al entrenamiento igual, porque el clasificador consume píxeles y no keypoints.

  Y por eso mismo **"no detectada" no es "no visible"**: una oreja puede estar perfectamente en el
  recorte sin que el modelo de pose la haya encontrado. El 54,7 % es una **cota inferior** de
  cuántas veces el clasificador puede ver la oreja, no una medición de lo que ve. Si se cita en la
  memoria, hay que citarlo así.
