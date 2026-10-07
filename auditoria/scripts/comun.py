"""Utilidades compartidas por los scripts de la auditoría. SOLO LECTURA sobre el repositorio principal.

Los datos (`datos_equinos/`, gitignored) viven en el repo principal; este worktree sólo escribe
bajo `auditoria/`. Nunca se llama a funciones que escriban cachés en la raíz de datos
(`caracteristicas.extraer`, `rutas.inicializar`): las características se leen directo del .npz.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
AUD = AQUI.parent
WT = AUD.parent
REPO = Path(os.environ.get(
    "AUDITORIA_REPO",
    "/media/sergio/Expansion/UGR/Licenciatura en Ciencias de datos/cuatrimestre3/materias/"
    "seminario trabajo final/Tesina/expresiones_faciales_equinos"))
DATOS_REPO = REPO / "datos_equinos"
os.environ["TESINA_EQUINOS_RAIZ"] = str(DATOS_REPO)   # las funciones de equinos/ leen de aquí (solo lectura)
RES = DATOS_REPO / "resultados"
VIDEOS = Path(os.environ.get("TESINA_EQUINOS_VIDEOS", str(Path.home() / "videos_de_prueba")))
SALIDA = AUD / "datos"
FIGURAS = AUD / "figuras"
SALIDA.mkdir(exist_ok=True)
FIGURAS.mkdir(exist_ok=True)

# el código del worktree (equinos/) es el código vigente de main
if str(WT) not in sys.path:
    sys.path.insert(0, str(WT))

CLASES = ("Disgustado", "Interesado", "Sereno")
COLOR = {"Disgustado": "#c0392b", "Interesado": "#e0a800", "Sereno": "#2a7f9e"}


def leer_csv(ruta):
    with open(ruta, encoding="utf-8", errors="replace", newline="") as f:
        return list(csv.DictReader(f))


def leer_json(ruta):
    return json.loads(Path(ruta).read_text(encoding="utf-8"))


def guardar_json(obj, nombre):
    ruta = SALIDA / nombre
    ruta.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=_conv), encoding="utf-8")
    return ruta


def guardar_csv(filas, nombre, columnas=None):
    ruta = SALIDA / nombre
    columnas = columnas or list(filas[0].keys())
    with open(ruta, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columnas)
        w.writeheader()
        w.writerows(filas)
    return ruta


def _conv(o):
    try:
        import numpy as np
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:
        pass
    if isinstance(o, set):
        return sorted(o)
    return str(o)


def particion():
    return leer_json(RES / "particion_clips.json")["clips"]


def evaluacion_vigente():
    return leer_json(RES / "objetivo3_evaluacion_v2.json")


def lineas_de_base():
    return leer_json(RES / "objetivo3_lineas_de_base.json")


def caracteristicas_v2():
    """(orden_de_clips, arreglo de (16,512)) leídos directo del npz, sin tocar la caché."""
    import numpy as np
    meta = leer_json(RES / "caracteristicas_backbone_v2.json")
    datos = np.load(RES / "caracteristicas_backbone_v2.npz", allow_pickle=True)["caracteristicas"]
    return list(meta["clips"]), datos


def estilo():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 100, "savefig.dpi": 200, "font.size": 10, "axes.titlesize": 11,
        "axes.labelsize": 10, "axes.spines.top": False, "axes.spines.right": False,
        "font.family": "DejaVu Sans", "axes.grid": True, "grid.alpha": 0.25})
    return plt


def guardar_fig(fig, nombre):
    ruta = FIGURAS / nombre
    fig.savefig(ruta, dpi=200, bbox_inches="tight")
    return ruta


def fmt(x, d=3):
    return f"{x:.{d}f}".replace(".", ",")
