"""El modelo de fusión tardía pose + visual, listo para inferir.

Es el ganador de la comparación de modelos (`comparacion/DECISION.md`). Acá vive lo que la
inferencia necesita de él; el ajuste está en `comparacion/candidato_fusion.py` y el entrenamiento
final en `comparacion/entrenar_modelo_final.py`.

**Las features de pose no se reimplementan:** salen de `experimentos/simba/features_simba.py`
(`vector_del_clip`), la misma función con la que se entrenó. Duplicarla permitiría que las dos
divergieran sin que nada fallara, que es el error que ya ocurrió entre la etapa 02 y la interfaz.

**El artefacto es un paquete de objetos scikit-learn**, no la clase `FusionPoseVisual`: así la app
no depende de cómo se llamó el módulo que lo guardó.

scikit-learn, joblib y numpy se importan dentro de las funciones.
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

from equinos import rutas

CLASES = ("Disgustado", "Interesado", "Sereno")
FRAMES_POR_CLIP = 16

SUBCARPETA_MODELO = "fusion_pose_visual"
ARCHIVO_MODELO = "fusion_pose_visual.joblib"

RAIZ_REPO = Path(__file__).resolve().parent.parent
CARPETA_RESULTADOS = RAIZ_REPO / "comparacion" / "resultados"
ARCHIVO_DESARROLLO = "fusion_tardia_pose_visual.json"
ARCHIVO_RESERVA = "fusion_tardia_pose_visual__reserva.json"


def ruta_modelo() -> Path:
    """Dónde guarda y busca la app el paquete: junto a los pesos del MIL, bajo `modelos/`."""
    return rutas.carpeta("modelos") / SUBCARPETA_MODELO / ARCHIVO_MODELO


# --- paquete ------------------------------------------------------------------------------------

def guardar_paquete(paquete: dict, ruta: Path | None = None) -> Path:
    import joblib

    destino = Path(ruta) if ruta else ruta_modelo()
    destino.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(paquete, destino)
    return destino


def cargar_paquete(ruta: Path | None = None) -> dict:
    """Carga el paquete; avisa si la versión de scikit-learn no es la del entrenamiento."""
    import joblib
    import sklearn

    origen = Path(ruta) if ruta else ruta_modelo()
    if not origen.exists():
        raise FileNotFoundError(
            f"No existe {origen}. Generarlo con `python comparacion/entrenar_modelo_final.py`. "
            "El modelo no se versiona: vive bajo la raíz de datos.")
    paquete = joblib.load(origen)
    registrada = paquete.get("metadatos", {}).get("version_sklearn")
    if registrada != sklearn.__version__:
        warnings.warn(
            f"El modelo se entrenó con scikit-learn {registrada} y hay {sklearn.__version__}: "
            "las probabilidades pueden diferir. Regenerarlo con entrenar_modelo_final.py.",
            stacklevel=2)
    return paquete


# --- features -----------------------------------------------------------------------------------

def _simba():
    """`features_simba` y `datos` de `experimentos/simba`, sin duplicarlos.

    `datos.py` toma la raíz de DATOS_EQUINOS: se la alinea con la de `equinos.rutas`.
    """
    os.environ.setdefault("DATOS_EQUINOS", str(rutas.raiz()))
    carpeta = str(RAIZ_REPO / "experimentos" / "simba")
    if carpeta not in sys.path:
        sys.path.insert(0, carpeta)
    import datos
    import features_simba
    return features_simba, datos


def matriz_de_pose(filas_pose):
    """Pose `(16, 39, 3)` = (x, y, confianza), NaN donde falta, desde filas en formato largo.

    `orden` es la posición en la secuencia (0..15), como la deja `preparar_clip`.
    """
    import numpy as np

    _, datos = _simba()
    P = np.full((FRAMES_POR_CLIP, len(datos.BODYPARTS), 3), np.nan)
    for f in filas_pose:
        P[int(f["orden"]), datos.BP[f["bodypart"]]] = [
            float(f["x"]), float(f["y"]), float(f["confianza"])]
    return P


def features_de_pose(filas_pose, frames, fps):
    """Las 160 features «tipo SimBA» del clip: la misma función del entrenamiento."""
    import numpy as np

    features_simba, _ = _simba()
    frames = np.asarray(frames, dtype=float)
    if len(frames) != FRAMES_POR_CLIP:
        raise ValueError(f"Hacen falta {FRAMES_POR_CLIP} fotogramas y llegaron {len(frames)}.")
    return features_simba.vector_del_clip(matriz_de_pose(filas_pose), frames, float(fps))[0]


# --- predicción ---------------------------------------------------------------------------------

def probabilidades(paquete: dict, vector_pose, vectores_visuales) -> dict:
    """Probabilidades por clase de la fusión, y de cada rama.

    `vectores_visuales`: `(16, 512)`; el modelo usa su media, como en el entrenamiento.
    """
    import numpy as np

    clases = [str(c) for c in paquete["clases"]]
    w = float(paquete["peso"])
    p_pose = paquete["modelo_pose"].predict_proba(np.asarray(vector_pose, dtype=float)[None, :])[0]
    media = np.asarray(vectores_visuales, dtype=float).mean(axis=0)
    p_vis = paquete["modelo_visual"].predict_proba(media[None, :])[0]
    p = w * p_pose + (1 - w) * p_vis
    como_dict = lambda v: {c: float(x) for c, x in zip(clases, v)}  # noqa: E731
    return {"fusion": como_dict(p), "pose": como_dict(p_pose), "visual": como_dict(p_vis),
            "peso_pose": w}


# --- confiabilidad ------------------------------------------------------------------------------

def _leer(nombre: str, carpeta: Path | None = None) -> dict:
    ruta = (carpeta or CARPETA_RESULTADOS) / nombre
    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe {ruta}. Lo produce la comparación de modelos; sin él no hay de dónde "
            "sacar la confiabilidad que la interfaz tiene que mostrar.")
    return json.loads(ruta.read_text(encoding="utf-8"))


def precision_por_clase(carpeta: Path | None = None) -> dict:
    """De las veces que la fusión dijo una clase, cuántas acertó, fuera de muestra.

    Sale de la matriz de confusión sumada de las 20 evaluaciones de desarrollo (5 repeticiones ×
    4 folds por caballo). Cada clip se evaluó una vez por repetición, así que los conteos se
    dividen por R: la precisión es la misma, y los números absolutos quedan en clips.
    """
    d = _leer(ARCHIVO_DESARROLLO, carpeta)
    m, R = d["global"]["matriz_confusion_suma"], d["R"]
    salida = {}
    for j, clase in enumerate(d["clases"]):
        dijo = sum(m[i][j] for i in range(len(m)))
        acerto = m[j][j]
        salida[clase] = {"precision": acerto / dijo if dijo else float("nan"),
                         "acertadas": round(acerto / R), "veces_que_lo_dijo": round(dijo / R)}
    return salida


def precision_reserva(carpeta: Path | None = None) -> dict:
    """Lo mismo, medido en la reserva: 20 caballos / 68 clips que no se usaron para elegir nada.

    Es la medida más honesta y la más chica: cada precisión se apoya en pocas decenas de clips.
    """
    r = _leer(ARCHIVO_RESERVA, carpeta)
    m = r["metricas"]["matriz"]
    salida = {}
    for j, clase in enumerate(CLASES):
        dijo = sum(m[i][j] for i in range(len(m)))
        salida[clase] = {"precision": m[j][j] / dijo if dijo else float("nan"),
                         "acertadas": m[j][j], "veces_que_lo_dijo": dijo}
    return salida


def metricas_globales(carpeta: Path | None = None) -> dict:
    """F1 macro de desarrollo y de la reserva (único uso), y del subconjunto limpio."""
    d, r = _leer(ARCHIVO_DESARROLLO, carpeta), _leer(ARCHIVO_RESERVA, carpeta)
    return {
        "f1_macro_desarrollo": d["global"]["f1_macro"]["media"],
        "f1_macro_desarrollo_desvio": d["global"]["f1_macro"]["desvio"],
        "f1_limpio_desarrollo": d["limpio"]["f1_macro_oof_por_repeticion"]["media"],
        "f1_macro_reserva": r["metricas"]["f1_macro"],
        "accuracy_reserva": r["metricas"]["accuracy"],
        "f1_limpio_reserva": r["limpio_f1_macro"],
        "clips_reserva": r["clips"], "caballos_reserva": r["caballos"],
        "clips_desarrollo": sum(sum(f) for f in d["global"]["matriz_confusion_suma"]) // d["R"],
    }


def texto_de_confiabilidad(clase: str, precisiones=None, metricas=None, reserva=None) -> str:
    precisiones = precisiones or precision_por_clase()
    m = metricas or metricas_globales()
    p = precisiones[clase]
    r = (reserva or precision_reserva())[clase]
    return (
        f"Cuando este modelo predijo «{clase}», acertó {p['acertadas']} de "
        f"{p['veces_que_lo_dijo']} veces ({p['precision']:.0%}) en validación cruzada por caballo "
        f"({m['clips_desarrollo']} clips de desarrollo, 5 repeticiones), y {r['acertadas']} de "
        f"{r['veces_que_lo_dijo']} ({r['precision']:.0%}) en la reserva.\n"
        f"F1 macro {m['f1_macro_reserva']:.3f} sobre {m['clips_reserva']} clips de "
        f"{m['caballos_reserva']} caballos reservados (un solo uso) y "
        f"{m['f1_macro_desarrollo']:.3f} en desarrollo; {m['f1_limpio_reserva']:.3f} en la reserva "
        "si se miran sólo los clips filmados en condiciones parejas.\n"
        "Este resultado NO es un diagnóstico veterinario."
    )
