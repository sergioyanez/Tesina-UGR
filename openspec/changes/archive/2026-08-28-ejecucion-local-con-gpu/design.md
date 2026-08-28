# Diseño técnico — ejecución local con GPU

## Contexto

`equinos/rutas.py` ya fue escrito bicéfalo: `en_colab()` detecta el entorno, `raiz_por_defecto()`
respeta `TESINA_EQUINOS_RAIZ`, y `catalogo.raiz_videos()` respeta `TESINA_EQUINOS_VIDEOS`. Los CSV
guardan rutas **relativas** a la raíz. Nada de eso hay que inventarlo: la migración consiste en
completar lo que falta alrededor, que es el entorno de dependencias y las celdas de arranque.

## Decisión 1 — Entorno aislado con una rueda de torch para CUDA 12.1

El driver instalado es 535.274.02, que expone CUDA 12.2. El `torch 2.12.0+cu130` del sistema pide
CUDA 13 y falla con *"The NVIDIA driver on your system is too old"*. Las ruedas `cu12x` de PyTorch
funcionan con cualquier driver de la serie 12 por compatibilidad de versión menor.

Se elige `torch 2.5.1+cu121` / `torchvision 0.20.1+cu121`, verificado el 21/08/2026:

```
cuda disponible: True | NVIDIA GeForce RTX 3070 Laptop GPU | 8.4 GB
matmul 4096^2 x20: 6.1 TFLOPS fp32
```

El entorno vive en `/home/sergio/.venvs/tesina-equinos`, **fuera del repositorio y fuera del disco
externo**. Dos razones: exFAT no admite los enlaces simbólicos que usa un venv, y el repositorio no
debe contener 5 GB de dependencias.

Alternativa descartada: subir el driver a la serie 580. Ver `proposal.md`.

## Decisión 2 — Conjunto de versiones verificado, no un rango

DeepLabCut 3.0.1 usa el backend PyTorch y **no necesita TensorFlow** (`tensorflow` sólo aparece
bajo los extras `tf*`, que no se instalan). Sus restricciones que atan al resto: `numpy<2`,
`matplotlib<3.9`, `pandas>=2.2`, `albumentations<=1.4.3`.

Conjunto verificado conviviendo en un mismo intérprete:

| paquete | versión |
|---|---|
| python | 3.10.12 |
| numpy | 1.26.4 |
| scipy | 1.13.1 |
| pandas | 2.3.3 |
| opencv-python | 4.11.0 |
| torch / torchvision | 2.5.1+cu121 / 0.20.1+cu121 |
| deeplabcut | 3.0.1 |
| ultralytics | 8.4.100 |
| papermill | 2.7.0 |

`scipy<1.14` sigue pinneado por el mismo motivo que en Colab: es la última serie que convive con
numpy 1.x. Que acá el entorno nazca limpio evita el modo de fallo de Colab —numpy reemplazado en
disco mientras ya está importado en memoria— pero el pin se conserva porque la restricción es real
en ambos entornos.

## Decisión 3 — Las celdas `!pip install` se condicionan, no se borran

Hoy corren incondicionalmente en los tres notebooks. En local eso instalaría DeepLabCut sobre el
Python del sistema (numpy 2.2.6, scipy 1.15.3, pandas 2.3.3 en `~/.local`) y rompería el stack
científico de la máquina entera, no sólo el del pipeline.

Se condicionan a `EN_COLAB`, que las celdas de arranque ya calculan. Fuera de Colab la celda no
instala: **verifica**. Un entorno incompleto tiene que fallar en el arranque, no a los cuarenta
minutos y con un traceback de otra cosa.

No se borran porque Colab sigue siendo una ruta de primera clase: ahí la instalación por sesión es
obligatoria, la máquina es efímera.

## Decisión 4 — Raíz local en NVMe, videos en el disco externo

Por defecto la raíz local es `<repo>/datos_equinos`, o sea el disco externo exFAT por USB. Los
frames alineados son miles de JPEG chicos y ese es el peor patrón de acceso posible ahí.

- `TESINA_EQUINOS_RAIZ=/home/sergio/tesina_equinos` — artefactos, en NVMe (192 GB libres).
- `TESINA_EQUINOS_VIDEOS=<repo>/videos` — los 147 videos crudos, que ya están en el externo y se
  leen secuencialmente una sola vez por corrida.

Esto no toca `rutas.py`: es configuración, y el mecanismo ya existe. Lo que faltaba era que
estuviera escrito en alguna parte.

## Decisión 5 — Kernel propio registrado para papermill

El orquestador ejecuta con `kernel_name="python3"`. En local ese nombre resuelve al Python del
sistema, no al del venv: papermill correría las etapas con numpy 2 y sin CUDA, que es exactamente
el fallo silencioso que hay que impedir.

Se registra un kernel `tesina-equinos` desde el venv y el orquestador lo elige según el entorno,
verificando antes de lanzar la primera etapa que existe.

## Riesgos

- **Deriva entre los dos entornos.** Colab instala desde una línea de `pip install` y local desde
  `requirements-local.txt`: pueden separarse con el tiempo. Mitigación: la celda de Colab pasa a
  citar el archivo como fuente de verdad de las versiones, y el arranque de cada etapa imprime las
  versiones efectivas, que quedan en el `.ipynb` ejecutado de `ejecuciones/`.
- **Throttling térmico.** Es un portátil de 80 W: una corrida de horas no sostiene el reloj del
  primer minuto. Afecta a las estimaciones de tiempo, no a los resultados.
- **Migración incompleta de artefactos.** Bajar a medias `tesina_equinos/` de Drive dejaría el
  checkpoint sin sus cachés y `etapas.estado()` podría dar una etapa por completa. Mitigación:
  correr `etapas.diagnostico()` como primer paso después de copiar, antes de ejecutar nada.
