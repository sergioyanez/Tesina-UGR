"""Líneas de base del objetivo 3 sobre el conjunto vigente, y los intervalos que las comparan.

Un F1 macro de 0,74 no dice nada solo. Este módulo mide, **sobre exactamente los mismos folds por
caballo que el modelo**, lo que se obtiene sin el modelo:

  - sin mirar nada: clase mayoritaria y azar estratificado;
  - sin mirar la cara: metadatos de filmación (duración, fps, resolución, ventana) y la duración
    sola. Es el piso honesto: si la forma de filmar ya separa las clases, superar a la mayoritaria
    no prueba que el modelo lea la expresión;
  - mirando la cara, pero sin lo que el modelo agrega: un frame suelto, el promedio de los 16
    frames y la geometría sola. El promedio es el que pone a prueba la atención, el orden y el
    canal geométrico: si el MIL no le gana, ninguna de esas tres cosas está demostrada.

Las lineales usan scikit-learn con hiperparámetros **fijados a priori** (`C_ALTA_DIMENSION`,
`C_BAJA_DIMENSION`): barrerlos mirando el test las volvería optimistas, y barrerlos con validación
interna exigiría replicar todo el protocolo del MIL para un modelo de comparación. Fijos es lo
conservador para una línea de base. La estandarización se ajusta sólo con el train de cada fold.

Las métricas salen de `clasificacion.f1_macro`, la misma función que evalúa al modelo: una
comparación entre dos implementaciones de F1 no compararía modelos.

Los intervalos remuestrean **caballos**, no clips (`bootstrap_por_caballo`). Los clips de un
mismo animal no son independientes, y un bootstrap por clip daría intervalos más estrechos de lo
que la evidencia permite. Las diferencias se calculan pareadas, sobre la misma réplica.

scikit-learn y numpy se importan dentro de las funciones, así `import equinos` no los arrastra.
"""

from __future__ import annotations

import csv
import statistics
from collections import Counter

from equinos import clasificacion, rutas

C_ALTA_DIMENSION = 0.1      # entradas de 512 dimensiones: regularización fuerte
C_BAJA_DIMENSION = 1.0      # metadatos y geometría resumida: 1 a 18 dimensiones
UMBRAL_ALTA_DIMENSION = 50

SEMILLA_BOOTSTRAP = 1
REPLICAS_BOOTSTRAP = 2000
SIMULACIONES_AZAR = 2000

# El subconjunto sin rasgos de filmación delatores. Los 26 clips 640×480 son todos `Interesado`,
# los 8 de 2560×1440 son `Disgustado`, y 66 de los 85 clips de menos de 4,5 s son `Disgustado`.
RESOLUCION_MAYORITARIA = (1080, 1920)
DURACION_MINIMA_SIN_DELATORES = 5.0

COLUMNAS_METADATOS = ("duracion_seg", "fps", "ancho", "alto", "ventana_seg")


# --- entradas --------------------------------------------------------------------------------

def metadatos_de_filmacion(clips: list[dict]):
    """Matriz `(clips, 5)` con duración, fps, ancho, alto y ventana usada. Sin un solo píxel.

    La ventana (`span_seg`) es el tramo del clip del que salieron los 16 frames; se incluye porque
    define cuán juntos en el tiempo están los frames que ve el modelo.
    """
    import numpy as np

    carpeta = rutas.carpeta("resultados")
    with (carpeta / "inventario_clips.csv").open(encoding="utf-8") as archivo:
        inv_clips = {f["clip"]: f for f in csv.DictReader(archivo)}
    with (carpeta / "inventario_videos.csv").open(encoding="utf-8") as archivo:
        inv_videos = {f["archivo"]: f for f in csv.DictReader(archivo)}
    ventana = {}
    from equinos import particion
    with (carpeta / particion.ARCHIVO_FRAMES).open(encoding="utf-8", errors="replace") as archivo:
        for fila in csv.DictReader(archivo):
            ventana[fila["clip"]] = float(fila["span_seg"])

    filas = []
    for c in clips:
        clip = inv_clips[c["clip"]]
        video = inv_videos[clip["video_origen"]]
        filas.append([float(clip["duracion_seg"]), float(clip["fps"]),
                      float(video["ancho"]), float(video["alto"]), ventana[c["clip"]]])
    return np.array(filas, dtype=float)


def entradas(clips, vectores, indice, pose_filas) -> dict:
    """Las matrices de cada línea de base, en el orden de `clips`."""
    import numpy as np

    visual = np.stack([np.asarray(vectores[indice[c["clip"]]], dtype=np.float32) for c in clips])
    ds = clasificacion.MultimodalEquineDataset(clips, vectores, indice, pose_filas)
    geometria = np.stack([ds[i][1].numpy() for i in range(len(clips))])        # (n, 16, 9)
    meta = metadatos_de_filmacion(clips)
    central = visual.shape[1] // 2
    return {
        "visual_media": visual.mean(axis=1),
        "visual_un_frame": visual[:, central],
        "geometria": np.concatenate([geometria.mean(axis=1), geometria.std(axis=1)], axis=1),
        "metadatos": meta,
        "duracion": meta[:, :1],
    }


# --- modelos ---------------------------------------------------------------------------------

def predecir_lineal(X, y, fold, c=None):
    """Regresión logística fuera de muestra: para cada fold, entrena con los otros y predice ése."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if c is None:
        c = C_ALTA_DIMENSION if X.shape[1] > UMBRAL_ALTA_DIMENSION else C_BAJA_DIMENSION
    y, fold = np.asarray(y), np.asarray(fold)
    salida = np.empty(len(y), dtype=object)
    for k in sorted(set(fold.tolist())):
        entrena, prueba = fold != k, fold == k
        modelo = make_pipeline(StandardScaler(), LogisticRegression(
            C=c, max_iter=5000, class_weight="balanced"))
        modelo.fit(X[entrena], y[entrena])
        salida[prueba] = modelo.predict(X[prueba])
    return salida


def predecir_mayoritaria(y, fold):
    import numpy as np

    y, fold = np.asarray(y), np.asarray(fold)
    salida = np.empty(len(y), dtype=object)
    for k in sorted(set(fold.tolist())):
        salida[fold == k] = Counter(y[fold != k].tolist()).most_common(1)[0][0]
    return salida


def azar_estratificado(y, fold, simulaciones=SIMULACIONES_AZAR, semilla=0) -> dict:
    """Esperanza del F1 macro de sortear con las proporciones del train de cada fold."""
    import numpy as np

    y, fold = np.asarray(y), np.asarray(fold)
    rng = np.random.default_rng(semilla)
    folds = sorted(set(fold.tolist()))
    medias = []
    for _ in range(simulaciones):
        f1s = []
        for k in folds:
            clases, conteos = np.unique(y[fold != k], return_counts=True)
            sorteo = rng.choice(clases, size=int((fold == k).sum()), p=conteos / conteos.sum())
            f1s.append(clasificacion.f1_macro(y[fold == k].tolist(), sorteo.tolist()))
        medias.append(statistics.fmean(f1s))
    return {"f1_macro": statistics.fmean(medias), "f1_desvio": statistics.stdev(medias),
            "percentil_95": float(np.percentile(medias, 95)), "accuracy": None}


# --- métricas --------------------------------------------------------------------------------

def resumen(y, predichas, fold) -> dict:
    """F1 macro media ± desvío entre folds (como el modelo) y accuracy agrupada."""
    import numpy as np

    y, predichas, fold = np.asarray(y), np.asarray(predichas), np.asarray(fold)
    por_fold = [clasificacion.f1_macro(y[fold == k].tolist(), predichas[fold == k].tolist())
                for k in sorted(set(fold.tolist()))]
    media, desvio = clasificacion.resumen(por_fold)
    return {"f1_macro": media, "f1_desvio": desvio, "por_fold": por_fold,
            "f1_agrupado": clasificacion.f1_macro(y.tolist(), predichas.tolist()),
            "accuracy": float((y == predichas).mean())}


def bootstrap_por_caballo(y, predicciones: dict, caballos, referencia: str,
                          replicas=REPLICAS_BOOTSTRAP, semilla=SEMILLA_BOOTSTRAP) -> dict:
    """Intervalos del 95 % remuestreando caballos, y diferencias pareadas contra `referencia`.

    `predicciones` es `{nombre: arreglo de predicciones fuera de muestra}`, todas alineadas con `y`.
    Cada réplica sortea caballos con reposición y arrastra TODOS sus clips; el F1 es el agrupado
    de la réplica. La diferencia `referencia − otro` se calcula en la misma réplica.
    """
    import numpy as np

    y, caballos = np.asarray(y), np.asarray(caballos)
    unicos = np.unique(caballos)
    clips_de = {h: np.where(caballos == h)[0] for h in unicos}
    rng = np.random.default_rng(semilla)
    f1 = {n: [] for n in predicciones}
    acc = {n: [] for n in predicciones}
    for _ in range(replicas):
        idx = np.concatenate([clips_de[h] for h in rng.choice(unicos, len(unicos))])
        reales = y[idx].tolist()
        for nombre, pred in predicciones.items():
            p = np.asarray(pred)[idx]
            f1[nombre].append(clasificacion.f1_macro(reales, p.tolist()))
            acc[nombre].append(float((p == y[idx]).mean()))

    def ic(valores):
        return [float(np.percentile(valores, 2.5)), float(np.percentile(valores, 97.5))]

    salida = {"replicas": replicas, "semilla": semilla, "referencia": referencia,
              "caballos": len(unicos), "modelos": {}}
    ref = np.array(f1[referencia])
    for nombre in predicciones:
        registro = {"f1_ic95": ic(f1[nombre]), "accuracy_ic95": ic(acc[nombre])}
        if nombre != referencia:
            dif = ref - np.array(f1[nombre])
            registro["diferencia_referencia_menos_este"] = {
                "media": float(dif.mean()), "ic95": ic(dif),
                "incluye_cero": bool(np.percentile(dif, 2.5) <= 0 <= np.percentile(dif, 97.5))}
        salida["modelos"][nombre] = registro
    return salida


def sin_rasgos_delatores(clips, metadatos):
    """Máscara de los clips 1080×1920 de al menos 5 s: sin los rasgos que delatan la clase."""
    ancho, alto = RESOLUCION_MAYORITARIA
    return ((metadatos[:, 2] == ancho) & (metadatos[:, 3] == alto)
            & (metadatos[:, 0] >= DURACION_MINIMA_SIN_DELATORES))


def en_subconjunto(y, predichas, mascara) -> dict:
    import numpy as np

    y, predichas = np.asarray(y)[mascara], np.asarray(predichas)[mascara]
    return {"clips": int(mascara.sum()),
            "clips_por_clase": {c: int((y == c).sum()) for c in clasificacion.CLASES},
            "f1_macro": clasificacion.f1_macro(y.tolist(), predichas.tolist()),
            "accuracy": float((y == predichas).mean())}
