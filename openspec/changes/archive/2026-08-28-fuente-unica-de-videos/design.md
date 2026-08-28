# Diseño — fuente única de videos

## Contexto

El pipeline tiene tres lugares donde puede haber videos —la carpeta compartida de Drive, su copia
local, y el espejo bajo la raíz de artefactos— y una cascada que elige el primero que sirva. Esa
cascada se escribió para Colab, donde la raíz vive en Drive y `<raíz>/videos` es material legítimo.
En local la misma regla convierte cualquier copia de trabajo en una fuente candidata.

El costo no fue un fallo: fue una métrica plausible y equivocada, publicada por encima de un umbral
que el proyecto venía sin alcanzar. Eso fija la prioridad del diseño — **que no haya dos copias que
puedan ganar, y que la ausencia de la única sea ruidosa**.

## Decisión 1 — Retirar el espejo en vez de sincronizarlo

Se retira `datos_equinos/videos`. Sin candidata bajo la raíz, la cascada se queda sin dónde caer y
`raiz_videos` termina en su `SystemExit`, que ya nombra lo que revisó y cómo declarar la carpeta.

**Alternativa descartada — sincronizar el espejo con la fuente.** Copiar los 12 videos que faltaban
y borrar el `501` dejaba todo funcionando hoy, con dos copias de 5 GB que pueden volver a divergir
mañana sin que nada lo indique. Es exactamente la situación que produjo el error: el espejo no
estaba corrupto, estaba desactualizado.

**Alternativa descartada — un enlace simbólico a la fuente.** Era la opción elegida: mata la clase
entera de bug, no duplica nada y no puede divergir. **No es posible**: el disco externo es exFAT y
`ln -s` devuelve «Operación no permitida».

## Decisión 2 — Las rutas viajan en el kernel, no en el lanzador solamente

`kernel.json` admite un diccionario `env` que jupyter\_client aplica al proceso del kernel. Ahí van
`TESINA_EQUINOS_RAIZ` y `TESINA_EQUINOS_VIDEOS`.

El lanzador ya las exportaba, pero sólo protege a quien lo usa; la corrida que falló se hizo con un
Jupyter abierto por fuera. El kernel es el lugar donde el pipeline realmente se ejecuta, y es
común a las dos formas de arrancarlo. Es configuración de máquina, no del repositorio: vive fuera
del control de versiones a propósito, porque las rutas son de esta máquina.

**Alternativa descartada — hardcodear la ruta en `catalogo.py` como candidata local.** Mete una
ruta de una máquina en el código de la tesina, y traslada el problema a la siguiente máquina.

## Decisión 3 — La raíz oficial es la del repositorio, en el disco externo

Es donde están los artefactos buenos (16 GB) y donde hay 2,6 TB libres, contra 178 GB en el NVMe.
Coincide además con la raíz por defecto cuando no hay variables, así que el olvido cae en el lugar
correcto en vez de en uno vacío.

La objeción de velocidad —miles de JPEG chicos sobre exFAT por USB— sigue en pie y está anotada,
pero no se manifestó: la etapa 02 corrió a 29 s/clip, más rápido que lo que predice el ritmo medido
de DeepLabCut. `~/tesina_equinos`, con la corrida de 146 videos, queda como respaldo y deja de
usarse.

## Decisión 4 — La purga del `501` se hace sobre los checkpoints, no sobre los CSV finales

Todos los artefactos finales de las dos etapas se reconstruyen desde el CSV parcial del checkpoint.
Sacar el video de ahí y de la lista de unidades completas alcanza para que desaparezca de todo lo
derivado en la corrida siguiente; editar los CSV finales habría durado hasta la próxima ejecución.

Esto expone un hueco que este change **no** resuelve: el checkpoint nunca poda las unidades que
salieron del universo. Un video que se borra del catálogo sigue en los artefactos hasta que alguien
lo saca a mano, como se hizo acá.

## Riesgos

- **La purga de unidades retiradas es manual.** Si mañana se saca otro video del catálogo, sus
  filas sobreviven en el CSV parcial y nadie avisa. Candidato a un change propio.
- **El kernel hay que reiniciarlo** para que tome las variables nuevas; una sesión abierta de antes
  sigue con las viejas.
- **Las variables siguen pudiendo faltar** si alguien corre el paquete desde un Python suelto. En
  ese caso la fuente no resuelve y el pipeline corta, que es el comportamiento buscado, pero la raíz
  sí cae a `datos_equinos` sin decir nada — y esa es la correcta hoy, lo que hace el modo de falla
  menos visible de lo que sería deseable.
- **exFAT no admite enlaces simbólicos**, así que no hay forma de que la raíz de artefactos apunte a
  la fuente. Si el proyecto se mudara a un disco con soporte, el symlink sería mejor que el retiro.
