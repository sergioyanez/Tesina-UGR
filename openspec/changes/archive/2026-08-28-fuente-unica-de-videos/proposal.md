## Why

Contribuye al **objetivo específico 1** (construcción del dataset) y, sobre todo, a que sus
métricas sean citables. El 24 y el 25/08/2026 el pipeline corrió entero, sin fallar, sobre un
dataset que no era el declarado, y produjo un número que estuvo a punto de darse por bueno.

La corrida informó **72,0 % de cobertura de alineación**, por encima del umbral de 70 % que el
proyecto se fijó y que no se alcanzaba desde el principio. El número era falso. El pipeline había
resuelto los videos contra `datos_equinos/videos`, un espejo del dataset que se estaba copiando
*durante* esa misma sesión y que tenía 223 de los 234 videos catalogados: faltaban 12 de `Sereno`.
Medido sobre el material completo, la cobertura es **70,2 %**. De los 4 pp de diferencia contra la
corrida anterior, la mitad venía de que faltaban justamente los videos que peor alinean (55,8 %).

Nada avisó, y no por descuido: la cascada de `catalogo.raiz_videos` termina en `<raíz>/videos`,
que es la disposición correcta en Colab. En local, con la raíz cayendo a la del repositorio, esa
carpeta existía y tenía carpetas de clase adentro — candidata válida, silencio total. El lanzador
`correr_local.sh` fija las variables que evitan eso, pero **un Jupyter abierto por fuera del
lanzador no las tiene**, y así fue como se corrió.

El mismo espejo reintrodujo `501_8v1_LY1ucwR9.avi`, el video que estaba byte a byte idéntico en
`Disgustado` y en `Interesado` y que se había decidido borrar el 22/08. La corrida del 24/08 abortó
por ese duplicado —`catalogo.verificar()` hizo lo que debe—, se borró sólo la copia de `Interesado`
y el video volvió al dataset como `Disgustado`.

Había además **dos raíces de artefactos divergentes**: `~/tesina_equinos` con la corrida de 146
videos y `<repo>/datos_equinos` con la de 223, y ningún criterio escrito sobre cuál era la buena.

## What Changes

- **Una sola fuente declarada de videos**: `~/videos_de_prueba`, la copia de la carpeta compartida,
  con los 234 videos y sin el `501`.
- **Se retira el espejo bajo la raíz de artefactos.** Sin candidata falsa en la cascada, una corrida
  sin la fuente declarada ya no cae en una copia parcial: muere con el mensaje de
  `catalogo.raiz_videos`, que nombra lo que revisó.
- **El kernel de Jupyter lleva las rutas.** `TESINA_EQUINOS_RAIZ` y `TESINA_EQUINOS_VIDEOS` pasan a
  estar en el `kernel.json` del kernel `tesina-equinos`, así que un notebook abierto desde cualquier
  Jupyter arranca con las rutas correctas, se haya usado el lanzador o no.
- **Raíz de artefactos oficial: `<repo>/datos_equinos`**, que es donde están los artefactos buenos.
  `correr_local.sh` cambia su valor por defecto en consecuencia.
- **El `501` se retira de los artefactos derivados**: de los dos checkpoints, de sus dos CSV
  parciales, y sus clips y frames se apartan del árbol de trabajo.
- **`correr_local.sh etapa01`**, simétrico al `etapa02` que ya existía. Hace falta porque
  `etapas.estado()` informa «completa» mirando sólo que los artefactos existan: con material nuevo
  el orquestador saltea la etapa, y el único modo de incorporarlo sin borrar horas de cómputo es
  correr el notebook directo con el checkpoint puesto.

## Capabilities

### New Capabilities

Ninguna.

### Modified Capabilities

- `persistencia-artefactos`: el requisito "Lectura de los videos sin descarga repetida" y la
  cascada de resolución que le agregó `catalogo-de-videos-por-clase` admiten hoy varias carpetas
  candidatas y eligen la primera que tenga clases adentro. Se agrega el requisito de que el
  material de video tenga **una sola fuente declarada**, y de que la ausencia de esa fuente sea un
  fallo y no una elección silenciosa.

## Impact

- **Lanzador:** `correr_local.sh` — raíz por defecto, subcomando `etapa01`, y el comentario de
  cabecera que explica por qué existe.
- **Entorno local, fuera del repositorio:** `kernel.json` del kernel `tesina-equinos` (respaldo en
  `kernel.json.bak-20260825`); espejo retirado a `datos_equinos/videos_espejo_retirado_20260825`;
  artefactos del `501` en `datos_equinos/_retirado_501_20260825`; respaldo de los checkpoints en
  `datos_equinos/checkpoints_backup_20260825`.
- **No cambia el paquete `equinos` ni los notebooks.** La cascada de `raiz_videos` ya fallaba con un
  mensaje correcto; lo que faltaba era que fuera alcanzable.
- **Cambian todos los denominadores.** Los artefactos pasan de 223 a 234 videos, de 251 a 280 clips
  y de 4016 a 4480 frames. La cobertura publicada de esa corrida —72,0 %— no debe citarse: el valor
  sobre el material completo es 70,2 %.
- **No se pudo usar un enlace simbólico.** El disco externo es exFAT y `ln -s` devuelve «Operación
  no permitida», así que la fuente no se puede apuntar desde la raíz de artefactos.
