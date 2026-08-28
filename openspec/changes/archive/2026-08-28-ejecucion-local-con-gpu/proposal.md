## Why

El pipeline se ejecuta hoy exclusivamente en Colab, y el acceso a GPU quedó cortado por los
límites de la suscripción. Eso bloquea la primera acción pendiente —re-derivar los recortes con el
encuadre canónico nuevo— y con ella la verificación del objetivo específico 2.

La máquina de desarrollo tiene una GPU que hasta ahora se daba por inservible. Medido el
21/08/2026, no lo es:

| | Esta máquina | Colab free (T4) |
|---|---|---|
| GPU | RTX 3070 Laptop, 8 GB VRAM | T4, 16 GB VRAM |
| CPU | i7-11800H, 8 núcleos / 16 hilos | 2 vCPU |
| RAM | 16 GB | ~12,7 GB |
| Límite de sesión | ninguno | 12 h, con cortes |

La creencia registrada de que "no hay GPU utilizable" venía de un diagnóstico correcto con una
causa mal atribuida: `torch.cuda.is_available()` daba `False`, pero no por la GPU sino porque el
`torch` instalado era `2.12.0+cu130` —compilado para CUDA 13— contra un driver 535, que es CUDA
12.2. El propio error lo decía. Con una rueda `cu121` en un entorno aislado, la misma GPU responde.

Prueba de humo sobre un clip real del dataset (`901_20260702_091138`, 152 frames, 1920x1080):

| Etapa | Medición local |
|---|---|
| YOLO11x, 16 frames | 2,5 s (153 ms/frame), **16/16 con caja** |
| DeepLabCut `superanimal_quadruped`, clip completo | 1,0 min (394 ms/frame), 39 bodyparts |
| VRAM pico | 1,64 GB de 8,4 |
| Orientación | el frame sale 1080x1920, o sea `CAP_PROP_ORIENTATION_AUTO` corrige también acá |

Los 8 GB de VRAM no son la restricción que se temía: el pipeline usa una quinta parte.

Correr en local además **elimina cinco de las siete trampas de operación** que ya costaron
sesiones enteras: la sincronización del código desde GitHub, el push previo obligatorio, el secret
`GITHUB_TOKEN` que se apaga al re-subir el orquestador, el `site-packages` compartido que un
notebook muta por debajo del otro, y la latencia por archivo de Drive montado.

## What Changes

- El proyecto declara un **entorno de ejecución local reproducible y pinneado**
  (`requirements-local.txt`), con el conjunto de versiones que se verificó que conviven: numpy
  1.26.4, scipy 1.13.1, torch 2.5.1+cu121, DeepLabCut 3.0.1, ultralytics, papermill. Hoy ese
  conjunto sólo existe como una línea de `pip install` dentro de una celda, y sólo para Colab.
- Las celdas `!pip install` de los tres notebooks pasan a **condicionarse al entorno**. Hoy corren
  siempre: en local instalarían DeepLabCut sobre el Python del sistema y le bajarían numpy a 1.x,
  rompiendo el stack científico de la máquina entera.
- La resolución de rutas y de la carpeta de videos en local se **documenta y se fija** vía
  `TESINA_EQUINOS_RAIZ` y `TESINA_EQUINOS_VIDEOS`. El mecanismo ya existe en el paquete; lo que
  falta es que esté escrito y que la raíz local no caiga por defecto en el disco externo exFAT.
- El orquestador deja de exigir la maquinaria de Drive cuando no está en Colab, y **verifica el
  kernel** con el que papermill ejecuta las etapas.
- La documentación (`README.md`, `openspec/project.md`) gana el procedimiento de ejecución local,
  incluida la migración de los artefactos de Drive a la raíz local.

**Colab sigue siendo una ruta de primera clase.** No se elimina nada de lo que hoy funciona ahí:
Héctor conserva su acceso y el proyecto conserva el fallback si la laptop falla.

## Impact

- Specs afectadas: `orquestacion-notebooks`.
- Código: `00_pipeline.ipynb`, `01_preparar_dataset_equinos.ipynb`, `02_roi_y_normalizacion.ipynb`,
  `requirements-local.txt` (nuevo), `README.md`, `openspec/project.md`.
- Sin cambios en la lógica de etapas, checkpoints, cachés ni geometría: la migración no debe poder
  alterar ningún número del informe. Que los CSV guarden rutas **relativas** a la raíz es lo que
  vuelve esto posible, y es una decisión que ya estaba tomada.

## Alternativas descartadas

- **Actualizar el driver NVIDIA a la serie 580** para que el `torch` cu130 ya instalado funcione.
  Habría servido, pero toca el sistema entero de un portátil con gráficos híbridos Intel+NVIDIA,
  donde un driver a medio instalar deja la máquina sin entorno gráfico. Instalar una rueda cu121
  en un entorno aislado consigue lo mismo, es reversible y no puede romper nada fuera del venv.
- **Instalar el stack sobre el Python del sistema** (`~/.local`). Es exactamente lo que hay que
  evitar: DeepLabCut fija `numpy<2` y ahí vive numpy 2.2.6 con scipy y pandas compilados contra
  esa versión. La misma incompatibilidad que en Colab, pero permanente y sobre la máquina de
  trabajo.
- **Migrar a local y abandonar Colab.** Simplifica el código —se irían el sync, el token y el
  montaje de Drive— pero deja al otro integrante sin ruta de ejecución y al proyecto sin
  alternativa ante una falla de hardware, a cambio de borrar código que hoy funciona.
- **Re-correr todo desde cero en local en vez de traer los artefactos de Drive.** Son ~4 h de GPU
  (210 clips x ~180 frames x 0,394 s) porque DeepLabCut infiere el clip entero y no sólo los 16
  frames muestreados. Queda como validación posterior de la migración, no como camino de entrada.
