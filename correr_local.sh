#!/usr/bin/env bash
# Lanza el pipeline en la máquina local con las rutas correctas ya fijadas.
#
# Existe por un motivo concreto: si Jupyter arranca SIN estas variables, el pipeline no falla.
# `rutas` cae a `<repo>/datos_equinos` —vacía— y `catalogo` cae al espejo viejo de `<repo>/videos`,
# que no es el material bueno. Todo corre, no se queja, y produce números de otro dataset. Fijarlas
# a mano en cada sesión es exactamente la clase de paso que se olvida una vez y cuesta una tarde.
#
# Uso:
#   ./correr_local.sh              abre Jupyter Lab (elegir el kernel "Tesina equinos (GPU local)")
#   ./correr_local.sh estado       en qué estado está cada etapa, sin ejecutar nada
#   ./correr_local.sh reportes     cobertura y calidad de los recortes de la última corrida
#   ./correr_local.sh entorno      versiones instaladas y GPU visible
#   ./correr_local.sh etapa02      re-ejecuta el notebook 02 con papermill (LIMPIAR_ETAPA = False)
#
# Las rutas se pueden sobrescribir desde afuera:
#   TESINA_EQUINOS_RAIZ=/otra/raiz ./correr_local.sh estado

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${TESINA_EQUINOS_VENV:-$HOME/.venvs/tesina-equinos}"
PY="$VENV/bin/python"

export TESINA_EQUINOS_RAIZ="${TESINA_EQUINOS_RAIZ:-$HOME/tesina_equinos}"
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
    entorno)
        "$PY" -c "from equinos import entorno; entorno.informe()"
        ;;
    etapa02)
        # Sin LIMPIAR_ETAPA: en True borra el checkpoint y obliga a reinferir DeepLabCut, que acá
        # son horas. La corrida deja su copia ejecutada en ejecuciones/, con toda la salida.
        MARCA="$(date +%Y%m%d_%H%M%S)"
        SALIDA="$TESINA_EQUINOS_RAIZ/ejecuciones/roi_normalizacion_${MARCA}.ipynb"
        mkdir -p "$TESINA_EQUINOS_RAIZ/ejecuciones"
        echo "Ejecutando 02_roi_y_normalizacion.ipynb -> $SALIDA"
        "$VENV/bin/papermill" 02_roi_y_normalizacion.ipynb "$SALIDA" \
            -p EJECUCION_AUTOMATICA True -p LIMPIAR_ETAPA False \
            -k tesina-equinos --cwd . --log-output
        ;;
    jupyter)
        # Jupyter sale del Python del sistema, no del venv: lo único que hace falta del entorno es
        # el kernel, que ya está registrado a nivel de usuario y por eso lo ve cualquier Jupyter.
        echo "Abriendo Jupyter Lab. Elegí el kernel: Tesina equinos (GPU local)"
        exec "$HOME/.local/bin/jupyter" lab
        ;;
    *)
        echo "Opción desconocida: $1"
        echo "Usá: estado | reportes | entorno | etapa02 | jupyter"
        exit 2
        ;;
esac
