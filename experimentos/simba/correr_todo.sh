#!/usr/bin/env bash
# Reproduce TODO el experimento con un solo comando:  bash experimentos/simba/correr_todo.sh
# Requiere: ~/.venvs/tesina-equinos (+ shap, slicer, cloudpickle instalados con --no-deps) y
# ~/.venvs/simba (simba-uw-tf-dev) para el paso del extractor real. Solo LEE datos_equinos/ del repo
# principal (variable DATOS_EQUINOS para cambiar la ruta).
set -euo pipefail
AQUI="$(cd "$(dirname "$0")" && pwd)"; RAIZ="$(cd "$AQUI/../.." && pwd)"
PY=/home/sergio/.venvs/tesina-equinos/bin/python; SB=/home/sergio/.venvs/simba/bin/python
cd "$RAIZ"
PYTHONPATH="$AQUI/stub_tkinter" $SB "$AQUI/simba_real_intento.py"   # SimBA real (features de movimiento)
$PY "$AQUI/construir_dataset.py"                                    # features propias por clip
$PY "$AQUI/evaluar.py"                                              # folds, ablación, limpio, bootstrap
$PY "$AQUI/shap_rf.py"                                              # SHAP out-of-fold
$PY "$AQUI/inventario.py"                                           # inventario y compatibilidad
$PY "$AQUI/parsimonia_mil.py"                                       # tiempos/parámetros del MIL
$PY "$AQUI/figuras_y_resumen.py"                                    # figuras y resumen.json
$PY "$AQUI/test_features_simba.py"
