# Propuesta: Persistencia de artefactos y orquestación de notebooks en Colab

**Objetivo específico del plan de trabajo:** habilita el **2** (pipeline de preprocesamiento) y es
condición previa del **3** (modelo) y del **4** (interfaz). No agrega capacidad científica nueva:
elimina el obstáculo de infraestructura que hoy impide **terminar de ejecutar** el objetivo 2.

## Por qué

El pipeline está partido en notebooks numerados que se ejecutan en Colab, y **ninguna de sus salidas
sobrevive al cierre de la sesión**. Tres consecuencias concretas, todas observadas:

1. **El notebook 02 no encuentra las entradas que produce el 01.** El 01 escribe los clips en
   `/content/clips_equinos` y los CSV en `/content/resultados_equinos`; el 02 los busca en esas
   mismas rutas (`02_roi_y_normalizacion.ipynb`, Paso 2). Al abrir el 02 en una sesión nueva,
   `/content` está vacío. El notebook detecta el faltante y avisa, pero no puede continuar.
2. **Volver a ejecutar el 01 destruye lo ya hecho.** Su Paso 0 hace `shutil.rmtree` sobre
   `clips_equinos`, `frames_equinos` y `resultados_equinos`. No hay forma de reanudar: sólo de
   empezar de cero.
3. **No hay reanudación ante un corte.** El 02 tarda alrededor de una hora (~45 s × 92 clips) y no
   escribe ningún punto de guardado intermedio. Si la sesión se desconecta —cosa que Colab hace por
   límite de tiempo o por inactividad— se pierde el trabajo completo. Esto ya ocurrió: la corrida
   del 02 se interrumpió sin dejar resultados aprovechables.

A esto se suma un costo recurrente evitable: el 01 **re-descarga los ~1.9 GB de videos con `gdown`
en cada sesión**, aunque esos videos ya viven en Google Drive y pueden leerse montando la unidad.

El problema deja de ser tolerable justo ahora. Con dos notebooks se podía sostener a mano; los
objetivos 3 y 4 agregan partición, entrenamiento, evaluación e interfaz, y el entrenamiento es
precisamente la etapa más larga y más expuesta a cortes de sesión.

## Qué cambia

- **Google Drive como raíz única del proyecto.** Una celda de arranque idéntica en todos los
  notebooks monta Drive y resuelve todas las rutas contra una carpeta de proyecto persistente. Los
  videos se leen montados, sin `gdown` y sin volver a descargar.
- **Nuevo paquete `equinos/` en el repo**, con la resolución de rutas, el checkpointing y la caché
  comprimida. La lógica de rutas pasa a vivir en un solo lugar en vez de duplicada y divergente en
  cada notebook.
- **Checkpointing por unidad de trabajo.** Las etapas largas persisten su progreso por clip a
  medida que avanzan, y al relanzarse retoman en el primer clip pendiente.
- **Caché comprimida para los artefactos numerosos.** Los frames son miles de JPEG sueltos y Drive
  montado es muy lento con archivos chicos: se trabaja en el disco local de la sesión y se persiste
  un único `.zip` por etapa.
- **Nuevo notebook orquestador `00_pipeline.ipynb`**, que ejecuta los notebooks en orden con
  `papermill`, saltea las etapas cuyo resultado ya está en Drive y guarda una copia ejecutada de
  cada uno.
- **`01` y `02` adaptados** para consumir las rutas del paquete en lugar de literales `/content`.

**No** cambia ninguna decisión metodológica: ni la segmentación en clips, ni la extracción de
frames, ni la geometría de la ROI, ni los umbrales del preprocesamiento facial. Los CSV mantienen
sus columnas. Este cambio es de infraestructura de ejecución.

## Capacidades

**Nuevas:**

- `persistencia-artefactos` — raíz de proyecto en Drive, resolución de rutas, checkpointing
  reanudable y caché comprimida de artefactos numerosos.
- `orquestacion-notebooks` — ejecución encadenada de los notebooks del pipeline, con salteo de
  etapas ya completas y registro de la ejecución.

**Modificadas:** ninguna capacidad especificada. `preprocesamiento-facial` conserva todos sus
requisitos; sólo cambia de dónde lee y a dónde escribe, que esa spec no fija.

## Alternativas descartadas

- **Un único notebook monolítico.** Resolvería el pasaje de datos sin infraestructura, pero vuelve
  inevitable re-ejecutar el preprocesamiento entero en cada iteración de entrenamiento, e impide
  presentar el pipeline por etapas en el informe. Se descarta.
- **`gdown` en cada notebook, descargando las salidas intermedias desde Drive.** Mantiene la
  descarga repetida, requiere subir a mano cada artefacto intermedio y no da reanudación. Es el
  costo actual sin el beneficio.
- **`%run` de los notebooks dentro del orquestador.** `%run` no ejecuta `.ipynb` directamente, y
  compartir un mismo kernel entre etapas reintroduce el acoplamiento por estado en memoria que
  queremos evitar. `papermill` ejecuta cada notebook en un kernel propio y deja copia ejecutada.
- **Convertir los notebooks a módulos `.py` y orquestar con un script.** Es más limpio como
  software, pero sacrifica el notebook como documento narrativo con salidas visibles, que en una
  tesina es parte del entregable y de la evidencia ante el tribunal. Se descarta por eso.
- **Guardar los frames sueltos directamente en Drive.** Es lo más simple, pero Drive montado
  penaliza fuertemente la escritura de miles de archivos pequeños, al punto de dominar el tiempo
  total de la etapa. De ahí la caché comprimida.

## Impacto

- **Nueva dependencia:** `papermill` (sólo en el notebook orquestador).
- **Requiere** que el usuario autorice el montaje de Drive al inicio de cada sesión de Colab. La
  carpeta de videos está en su propia cuenta, así que hay permiso de lectura y escritura.
- **Modifica** `01_preparar_dataset_equinos.ipynb` y `02_roi_y_normalizacion.ipynb`, únicamente en
  sus celdas de configuración de rutas y de limpieza inicial.
- **Consume espacio en Drive**: los clips (~2.7 GB) y las cachés comprimidas de frames pasan a
  residir de forma permanente. Hay que verificar cuota disponible antes de la primera corrida.
- **Desbloquea** las tres tareas abiertas de `roi-normalizacion-pose`, que dependen de una corrida
  completa del notebook 02.

## Riesgo principal

La reanudación es correcta sólo si el checkpoint es **consistente con los parámetros con los que se
generó**. Si se cambia un umbral del preprocesamiento y la etapa retoma un checkpoint viejo, el
resultado mezcla dos configuraciones distintas y es científicamente inválido —el peor tipo de error,
porque no falla: produce un número creíble y equivocado.

Por eso el checkpoint DEBE registrar la configuración que lo generó e invalidarse solo cuando ésta
cambie, en lugar de confiar en que quien ejecuta se acuerde de borrarlo a mano.
