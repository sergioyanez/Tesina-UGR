"""Carga común para `generar_cifras.py` y `generar_figuras.py`.

Reutiliza `docs/informes/estado_actual_2026-10-08/datos.py` (lectura de los JSON de
`comparacion/resultados/` y verificaciones de coherencia con `comparacion/DECISION.md`) sin
modificarlo, para que `generar.py` de esa carpeta siga funcionando.
"""
import re
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "docs" / "informes" / "estado_actual_2026-10-08"))

import datos as base  # noqa: E402
from datos import CANDIDATOS, CLASES, NOMBRE, f1_pooled_54, f1_sesion, par  # noqa: E402,F401

CARPETA = Path(__file__).resolve().parent

FUS, RF, MIL, BASE = CANDIDATOS[0], CANDIDATOS[1], CANDIDATOS[2], CANDIDATOS[3]
PREFIJO = {FUS: "Fusion", RF: "RFPose", MIL: "MIL", BASE: "Base"}
PAR_PREFIJO = {FUS: "Fusion", RF: "RF", MIL: "MIL", BASE: "Base"}
CLASE_CORTA = {"Disgustado": "Disg", "Interesado": "Int", "Sereno": "Ser"}

MENOS_TEX = r"\ensuremath{-}"


def cargar():
    return base.cargar()


def plano(texto):
    """Normaliza espacios para buscar frases de DECISION.md que cortan línea."""
    return re.sub(r"\s+", " ", texto)


# --- formato LaTeX en español ---
def num(x, d=3):
    return f"{x:.{d}f}".replace(".", ",")


def sgn(x, d=3):
    return ("+" if x >= 0 else MENOS_TEX) + f"{abs(x):.{d}f}".replace(".", ",")


def intervalo(par_, d=3, signo=False):
    f = sgn if signo else num
    return f"[{f(par_[0], d)}; {f(par_[1], d)}]"


def pct(x, d=1):
    return f"{100 * x:.{d}f}".replace(".", ",") + r"\,\%"
