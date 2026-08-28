# Diseño — piso de video corto

## Contexto

La restricción real del trabajo no es el tamaño del dataset sino **de cuántos caballos distintos
sale cada clase**. Con 11 ejemplares en `Disgustado` contra 49 y 47, el ejemplar predice la clase, y
un clasificador entrenado así puede estar identificando caballos. Ese es el confounder anotado como
bloqueante antes de entrenar, y el piso de duración era una de sus causas: no filtraba tomas malas,
filtraba ejemplares, y filtraba justamente los de la clase de la que menos hay.

Eso fija la prioridad del diseño — **maximizar ejemplares recuperados por unidad de desviación
respecto del plan de trabajo**.

## Decisión 1 — Admitir el video corto entero, en vez de bajar el piso general

`PISO_VIDEO_CORTO` decide **qué material entra**; `DURACION_CLIP` decide **cuántas muestras salen**
de ese material. Son dos decisiones distintas y conviene tocar sólo la primera.

La regla nueva se aplica sólo cuando la segmentación no produjo ninguna ventana, así que los videos
que llegan al piso de 5 s conservan exactamente sus clips de antes.

**Alternativa descartada — bajar `DURACION_MINIMA` a 2 s.** No requiere código, y recupera los
mismos 32 ejemplares. Pero baja el piso para todo: los videos largos empiezan a producir colas
cortas que no aportan ejemplares nuevos, y diluyen a la clase que se quería recuperar.

| | clips | `Disgustado` % de frames | caballos D/I/S | clips < 5 s |
|---|---:|---:|---|---:|
| piso 5 s (antes) | 280 | 9,6 % | 11/49/47 | 0 % |
| bajar `DURACION_MINIMA` a 2 | 404 | 16,1 % | 32/60/54 | 30,7 % |
| **admitir el video corto entero (≥2 s)** | **342** | **18,1 %** | 32/60/54 | 18,1 % |

**Alternativa descartada — bajar `DURACION_CLIP` a 3 s.** Da 610 clips, pero la mayoría son cortes
del mismo material largo: muestras más parecidas entre sí. Infla el tamaño aparente del dataset sin
agregar información, y en una tesina que se defiende eso es peor que no inflarlo. El propio notebook
ya advertía que sólo es válido si la partición agrupa por caballo.

**Alternativa descartada — solapamiento.** Multiplica las muestras de los videos largos, no toca a
`Disgustado`, y es la única opción que sí encarece la etapa 02 de verdad (4,2 h contra 2,7 h),
porque DeepLabCut vuelve a inferir el material solapado.

**Alternativa descartada — sostener el rango 5-7 s del plan.** Es la opción sin desviación
metodológica, y deja a `Disgustado` con 11 de sus 42 ejemplares. La desviación que evita es menor
que el problema que conserva.

## Decisión 2 — El piso en 2 s

Con 2 s entran 32 de los 42 ejemplares; con 3 s, 25; con 1,5 s, 34. El salto grande está entre 3 y
2 s, y bajar a 1,5 s agrega dos ejemplares a cambio de clips que ya son notoriamente cortos. A 2 s
ningún clip queda con menos de 16 frames, que es lo que la extracción necesita: el más corto tiene
60.

## Decisión 3 — El parámetro entra en la clave del checkpoint

`CLAVES_SEGMENTACION` define qué invalida el trabajo previo. Sin agregar `PISO_VIDEO_CORTO`, cambiar
el piso habría dejado el checkpoint válido y la etapa habría informado "completa" con la
configuración vieja adentro: el modo de falla que el módulo de checkpoint dice explícitamente que
existe para evitar.

## Decisión 4 — Reusar el checkpoint de la etapa 02, pero verificándolo

Como los videos largos conservan sus ventanas, los 280 clips previos deberían regenerarse idénticos,
y la etapa 02 puede reusar su checkpoint e inferir sólo los 62 nuevos: 40 minutos en vez de 2,7 h.

Eso se **verificó** en vez de suponerse: se guardó el `inventario_clips.csv` anterior y se comparó
`frame_inicio`, `frame_fin` y `clase` de los 280 clips comunes. Cero diferencias, cero clips
desaparecidos. Sin esa comprobación, un clip regenerado distinto habría quedado con la pose del
clip viejo — sin error y sin aviso.

## Riesgos

- **Clips de largo variable (2 a 7 s).** Los 16 frames de un clip abarcan distinto tiempo según el
  clip, y eso es una heterogeneidad real del dataset. Queda registrada en `duracion_seg` y admite
  una ablación; no se puede ignorar al informar.
- **`Disgustado` sigue siendo la clase que peor alinea** (62,1 %) y 20 de sus 32 ejemplares aportan
  menos de un clip completo alineado. El material ya está adentro; el problema que queda es de
  calidad de alineación, no de material descartado.
- **La cobertura global dejó de ser comparable en el tiempo.** Se mueve con la composición del
  dataset, así que premia dejar afuera a la clase difícil. Hay que reemplazarla por cobertura por
  clase en el informe y en `reporte.cobertura()`.
