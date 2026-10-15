#!/usr/bin/env bash
# Regenera informe/cifras.tex, informe/cifras_lista.md y informe/figuras/*.{pdf,png}.
set -euo pipefail
AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TESINA_EQUINOS_RAIZ="${TESINA_EQUINOS_RAIZ:-$AQUI/../datos_equinos}"
PY="${PYTHON:-/home/sergio/.venvs/tesina-equinos/bin/python}"
"$PY" "$AQUI/generar_cifras.py"
"$PY" "$AQUI/generar_figuras.py"
