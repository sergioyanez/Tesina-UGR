#!/usr/bin/env bash
# Lanza el pipeline en la máquina local con las rutas correctas ya fijadas.
#
# Existe por un motivo concreto: si Jupyter arranca SIN estas variables, el pipeline no falla.
# `catalogo` caía al espejo de videos de la raíz de proyecto, que el 24/08/2026 era una copia a
# medio hacer —223 de 234 videos— y produjo una corrida entera de números de otro dataset.
#
# Desde el 25/08/2026 hay dos defensas más, y ésta ya no es la única:
#   - el espejo `datos_equinos/videos` se retiró, así que una corrida sin TESINA_EQUINOS_VIDEOS
#     ya no encuentra clases y muere con el mensaje de `catalogo.raiz_videos`, en vez de seguir;
#   - el kernel «Tesina equinos (GPU local)» lleva las dos variables en su `kernel.json`, así que
#     un notebook abierto desde cualquier Jupyter arranca con las rutas correctas igual.
# El disco externo es exFAT y no admite enlaces simbólicos: por eso el espejo no se pudo
# reemplazar por un symlink a la fuente, que sería lo más simple.
#
# Uso:
#   ./correr_local.sh              abre Jupyter Lab (elegir el kernel "Tesina equinos (GPU local)")
#   ./correr_local.sh estado       en qué estado está cada etapa, sin ejecutar nada
#   ./correr_local.sh reportes     cobertura y calidad de los recortes de la última corrida
#   ./correr_local.sh anotar       genera la página para marcar a mano el ojo y el hocico
#   ./correr_local.sh anotar-que   qué falta anotar, sin generar nada
#   ./correr_local.sh anotar-cargar <csv>   incorpora el CSV que bajó el navegador
#   ./correr_local.sh entorno      versiones instaladas y GPU visible
#   ./correr_local.sh etapa01      re-ejecuta el notebook 01 con papermill (LIMPIAR_ETAPA = False,
#                                  PISO_VIDEO_CORTO = 1.0 s; sobrescribible por entorno)
#   ./correr_local.sh etapa02      re-ejecuta el notebook 02 con papermill (LIMPIAR_ETAPA = False)
#   ./correr_local.sh interfaz     levanta la interfaz de inferencia (Streamlit)
#
# Las rutas y el piso se pueden sobrescribir desde afuera:
#   TESINA_EQUINOS_RAIZ=/otra/raiz ./correr_local.sh estado
#   PISO_VIDEO_CORTO=2.0 ./correr_local.sh etapa01     (vuelve al piso anterior)

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${TESINA_EQUINOS_VENV:-}"
if [ -z "$VENV" ]; then
    if [ -d "$REPO/env" ]; then
        VENV="$REPO/env"
    else
        VENV="$HOME/.venvs/tesina-equinos"
    fi
fi

if [ -d "$VENV/Scripts" ]; then
    BIN_DIR="$VENV/Scripts"
else
    BIN_DIR="$VENV/bin"
fi

PY="$BIN_DIR/python"

# La raíz oficial es la del repositorio en el disco externo: es donde viven los artefactos
# buenos (223 videos, 4016 frames) y donde hay lugar de sobra. `$HOME/tesina_equinos` quedó
# con los artefactos anteriores (146 videos, 3360 frames) y no se usa más.
export TESINA_EQUINOS_RAIZ="${TESINA_EQUINOS_RAIZ:-$REPO/datos_equinos}"
export TESINA_EQUINOS_VIDEOS="${TESINA_EQUINOS_VIDEOS:-$HOME/videos_de_prueba}"

if [ ! -x "$PY" ]; then
    echo "No existe el entorno en $VENV."
    echo "Reconstruilo con las instrucciones de requirements-local.txt."
    exit 1
fi

cd "$REPO"

echo "Raíz de artefactos : $TESINA_EQUINOS_RAIZ"
echo "Carpeta de videos  : $TESINA_EQUINOS_VIDEOS"
echo

case "${1:-jupyter}" in
    estado)
        "$PY" -c "from equinos import etapas; etapas.diagnostico()"
        ;;
    reportes)
        "$PY" -c "from equinos import reporte; reporte.cobertura(); print(); reporte.calidad_recortes()"
        ;;
    anotar-que)
        "$PY" -c "from equinos import anotacion; anotacion.informe()"
        ;;
    anotar)
        # Genera una página con los frames embebidos y la abre en el navegador: dos clics por
        # frame, primero el ojo y después el hocico. Sólo pide los frames que COMPLETAN un clip
        # —los que están a uno o dos de la integridad—, porque anotar un frame de un clip al que
        # le faltan diez no suma nada.
        #
        # Va por el navegador y no por una ventana de OpenCV porque el cv2 de este entorno es el
        # headless que arrastra DeepLabCut: no tiene GUI, y cambiarlo arriesga el stack de DLC.
        "$PY" -c "from equinos import anotacion; anotacion.pagina(anotado_por='${USER}')"
        ;;
    anotar-cargar)
        # Incorpora el CSV que bajó el navegador. Después hay que re-derivar con etapa02.
        if [ -z "${2:-}" ]; then
            echo "Falta el CSV: ./correr_local.sh anotar-cargar ~/Descargas/keypoints_manuales.csv"
            exit 2
        fi
        "$PY" -c "import sys; from equinos import anotacion; anotacion.importar(sys.argv[1])" "$2"
        ;;
    entorno)
        "$PY" -c "from equinos import entorno; entorno.informe()"
        ;;
    etapa01)
        # El orquestador saltea la etapa 01 en cuanto sus artefactos existen: `etapas.estado()` no
        # compara contra el catálogo, así que material nuevo entra sólo corriendo el notebook
        # directamente. Con LIMPIAR_ETAPA=False el checkpoint hace que procese únicamente los
        # videos que todavía no están, en vez de las horas que cuesta empezar de cero.
        #
        # PISO_VIDEO_CORTO va explícito aunque coincida con el default del notebook. Es el
        # parámetro que decide cuántos EJEMPLARES de `Disgustado` entran al dataset, y su valor
        # tiene que estar a la vista en el comando que se corre, no sólo dentro de una celda:
        # con 2.0 quedaban 11 caballos de la clase minoritaria enteros afuera, con 1.0 entran 7.
        # El checkpoint lo incluye en su hash_config, así que cambiarlo re-segmenta todo solo.
        PISO_VIDEO_CORTO="${PISO_VIDEO_CORTO:-1.0}"
        MARCA="$(date +%Y%m%d_%H%M%S)"
        SALIDA="$TESINA_EQUINOS_RAIZ/ejecuciones/preparar_dataset_${MARCA}.ipynb"
        mkdir -p "$TESINA_EQUINOS_RAIZ/ejecuciones"
        echo "Ejecutando 01_preparar_dataset_equinos.ipynb -> $SALIDA"
        echo "  PISO_VIDEO_CORTO = $PISO_VIDEO_CORTO s"
        "$BIN_DIR/papermill" 01_preparar_dataset_equinos.ipynb "$SALIDA" \
            -p EJECUCION_AUTOMATICA True -p LIMPIAR_ETAPA False \
            -p PISO_VIDEO_CORTO "$PISO_VIDEO_CORTO" \
            -k tesina-equinos --cwd . --log-output
        ;;
    etapa02)
        # Sin LIMPIAR_ETAPA: en True borra el checkpoint y obliga a reinferir DeepLabCut, que acá
        # son horas. La corrida deja su copia ejecutada en ejecuciones/, con toda la salida.
        MARCA="$(date +%Y%m%d_%H%M%S)"
        SALIDA="$TESINA_EQUINOS_RAIZ/ejecuciones/roi_normalizacion_${MARCA}.ipynb"
        mkdir -p "$TESINA_EQUINOS_RAIZ/ejecuciones"
        echo "Ejecutando 02_roi_y_normalizacion.ipynb -> $SALIDA"
        "$BIN_DIR/papermill" 02_roi_y_normalizacion.ipynb "$SALIDA" \
            -p EJECUCION_AUTOMATICA True -p LIMPIAR_ETAPA False \
            -k tesina-equinos --cwd . --log-output
        ;;
    interfaz)
        # La interfaz del objetivo 4. Corre en esta máquina porque arrastra la cadena entera de
        # inferencia —YOLO y DeepLabCut— y sin GPU los 35 s por clip se vuelven varios minutos.
        echo "Abriendo la interfaz. Ctrl+C para cerrarla."
        exec "$BIN_DIR/streamlit" run app_inferencia.py
        ;;
    jupyter)
        # Jupyter sale del Python del sistema, no del venv: lo único que hace falta del entorno es
        # el kernel, que ya está registrado a nivel de usuario y por eso lo ve cualquier Jupyter.
        echo "Abriendo Jupyter Lab. Elegí el kernel: Tesina equinos (GPU local)"
        exec "$HOME/.local/bin/jupyter" lab
        ;;
    *)
        echo "Opción desconocida: $1"
        echo "Usá: estado | reportes | anotar | anotar-que | anotar-cargar | entorno | etapa01 | etapa02 | interfaz | jupyter"
        exit 2
        ;;
esac
