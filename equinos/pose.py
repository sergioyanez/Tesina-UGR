"""Selección de keypoints faciales sobre la tabla de pose completa.

Este módulo es el único lugar donde se decide *qué* keypoints se usan como par de referencia. El
notebook 02 lo invoca para derivar `keypoints_faciales.csv`, y el barrido lo invoca para comparar
candidatos. Que sea el mismo código en ambos casos es lo que hace que el barrido prediga la
cobertura real en vez de una aproximación.

La razón de existir del módulo es económica. La inferencia de pose es la operación más cara del
pipeline (~1 h), mientras que elegir el keypoint del hocico es una decisión analítica que se revisa
varias veces. Mientras la elección viviera dentro del notebook y la salida cruda se descartara, cada
revisión costaba una reinferencia, y ese costo desalentaba justamente el análisis que la haría
correcta. Con la pose persistida en formato largo, revisar la elección cuesta segundos.

Deliberadamente sin pandas ni numpy, como `reporte`: DeepLabCut fija `numpy<2` y deja el stack
científico de la sesión en un estado del que pandas no siempre se recupera. El barrido se corre
justo después de la inferencia, que es exactamente cuando el stack está en ese estado.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

from equinos import rutas

# Nombre del artefacto con la salida cruda del modelo de pose.
ARCHIVO_POSE = "pose_completa.csv"

# Columnas de la tabla de pose en formato largo. Largo y no ancho porque el conjunto de bodyparts
# lo fija el modelo, no el proyecto: en formato ancho, cambiar de modelo cambia el esquema de
# columnas y rompe todo lo que lea el archivo.
COLUMNAS_POSE = ("clip", "orden", "bodypart", "x", "y", "confianza")

# Candidatos a keypoint del hocico, en el orden histórico de precedencia. El orden se conserva sólo
# para poder reproducir el comportamiento anterior; NO es un criterio de selección: cuál usar se
# decide con `barrido()`.
CANDIDATOS_HOCICO = ("nose", "snout", "muzzle", "upper_jaw", "mouth")

# Subcadena que identifica a los keypoints oculares en el esquema del modelo.
PATRON_OJO = "eye"

# Valor de `bodypart_hocico` que pide elegir el más confiable en cada frame, igual que se hace con
# el ojo. Ojo: no es gratis geométricamente — ver `variacion_distancia()`.
MAS_CONFIABLE = "*"


def _a_float(valor) -> float:
    """Convierte a float devolviendo NaN ante vacío o texto no numérico."""
    try:
        return float(valor)
    except (TypeError, ValueError):
        return float("nan")


def _es_nan(valor: float) -> bool:
    return valor != valor


# --- lectura -------------------------------------------------------------------------------


def leer_pose(ruta: str | Path | None = None) -> list[dict]:
    """Lee la tabla de pose completa desde la raíz de proyecto."""
    ruta = Path(ruta) if ruta else rutas.carpeta("resultados") / ARCHIVO_POSE
    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe {ruta}. La produce la etapa roi_normalizacion; si la raíz de proyecto "
            "viene de una corrida anterior a este cambio, hay que re-ejecutar esa etapa."
        )
    with ruta.open(encoding="utf-8", errors="replace") as archivo:
        filas = list(csv.DictReader(archivo))

    if filas:
        faltantes = [c for c in COLUMNAS_POSE if c not in filas[0]]
        if faltantes:
            raise ValueError(
                f"{ruta} no tiene las columnas {faltantes}. Se esperaba la tabla de pose en "
                f"formato largo con {list(COLUMNAS_POSE)}."
            )
    return filas


def bodyparts_disponibles(filas: list[dict]) -> list[str]:
    """Bodyparts presentes en la tabla, en orden de aparición."""
    vistos: dict[str, None] = {}
    for fila in filas:
        vistos.setdefault(fila["bodypart"], None)
    return list(vistos)


def _agrupar_por_frame(filas: list[dict]) -> dict[tuple[str, int], dict[str, tuple]]:
    """Reorganiza la tabla larga en {(clip, orden): {bodypart: (x, y, confianza)}}."""
    frames: dict[tuple[str, int], dict[str, tuple]] = defaultdict(dict)
    for fila in filas:
        clave = (fila["clip"], int(fila["orden"]))
        frames[clave][fila["bodypart"]] = (
            _a_float(fila["x"]), _a_float(fila["y"]), _a_float(fila["confianza"])
        )
    return frames


# --- de la salida cruda de DeepLabCut al formato largo --------------------------------------
# El notebook 02 hace esto mismo en su Paso 12. Vive también acá porque la inferencia sobre un
# video suelto lo necesita, y elegir el individuo ES una decisión: con varios animales detectados,
# cuál se sigue cambia todos los keypoints y por lo tanto el recorte.


def elegir_individuo(tabla) -> str:
    """Con varios animales detectados, el de mayor confianza media.

    `video_inference_superanimal` devuelve diez `individuals` aunque haya un solo animal: los que
    no detectó vienen con confianza -1. Quedarse con el último, o mezclarlos, deja los keypoints
    en NaN sin que nada falle — y aguas abajo eso se ve como «no se encontraron keypoints», que
    culpa al video en vez de al código.
    """
    import numpy as np

    individuos = list(dict.fromkeys(tabla.columns.get_level_values("individuals")))
    if len(individuos) == 1:
        return individuos[0]

    mejor, mejor_conf = individuos[0], float("-inf")
    for individuo in individuos:
        columnas = [c for c in tabla.columns
                    if c[1] == individuo and c[-1] == "likelihood"]
        if not columnas:
            continue
        media = float(np.nanmean(tabla[columnas].to_numpy(dtype=float)))
        if media > mejor_conf:
            mejor, mejor_conf = individuo, media
    return mejor


def a_formato_largo(tabla, ordenes_e_indices, clip: str) -> list[dict]:
    """Traduce el DataFrame de DeepLabCut a filas `(clip, orden, bodypart, x, y, confianza)`.

    Emite TODOS los bodyparts, no sólo el par facial: es el mismo criterio del notebook 02, y es
    lo que después permitió medir las orejas sin reinferir.
    """
    individuo = elegir_individuo(tabla)
    n_filas = len(tabla)
    bodyparts = list(dict.fromkeys(tabla.columns.get_level_values("bodyparts")))
    scorer = tabla.columns.get_level_values("scorer")[0]

    series = {}
    for bodypart in bodyparts:
        sub = tabla.xs((scorer, individuo, bodypart), axis=1,
                       level=("scorer", "individuals", "bodyparts"))
        series[bodypart] = (sub["x"].to_numpy(dtype=float),
                            sub["y"].to_numpy(dtype=float),
                            sub["likelihood"].to_numpy(dtype=float))

    filas = []
    for orden, indice in ordenes_e_indices:
        # CAP_PROP_FRAME_COUNT puede sobreestimar; se acota al rango real de la tabla.
        i = int(min(max(int(indice), 0), n_filas - 1))
        for bodypart, (xs, ys, cs) in series.items():
            filas.append({"clip": clip, "orden": int(orden), "bodypart": bodypart,
                          "x": float(xs[i]), "y": float(ys[i]), "confianza": float(cs[i])})
    return filas


# --- selección del par de referencia -------------------------------------------------------


def candidatos_presentes(
    bodyparts: list[str], candidatos: tuple[str, ...] = CANDIDATOS_HOCICO
) -> list[str]:
    """Candidatos del listado que el modelo efectivamente expone, respetando su nombre real."""
    por_minuscula = {b.lower(): b for b in bodyparts}
    return [por_minuscula[c] for c in candidatos if c in por_minuscula]


def bodyparts_oculares(bodyparts: list[str], patron: str = PATRON_OJO) -> list[str]:
    return [b for b in bodyparts if patron in b.lower()]


def _mejor_de(candidatos: list[str], medidos: dict[str, tuple]) -> tuple:
    """Del conjunto de bodyparts, el de mayor confianza en este frame.

    Devuelve (nombre, x, y, confianza), o (None, nan, nan, -1.0) si ninguno está medido.
    """
    mejor = (None, float("nan"), float("nan"), -1.0)
    for nombre in candidatos:
        if nombre not in medidos:
            continue
        x, y, conf = medidos[nombre]
        if not _es_nan(conf) and conf > mejor[3]:
            mejor = (nombre, x, y, conf)
    return mejor


def seleccionar_par(
    filas: list[dict],
    bodypart_hocico: str | None = None,
    candidatos: tuple[str, ...] = CANDIDATOS_HOCICO,
    patron_ojo: str = PATRON_OJO,
) -> list[dict]:
    """Deriva el par ojo/hocico por frame a partir de la tabla de pose completa.

    El ojo se resuelve siempre tomando el más confiable del frame: en perfil sólo uno es visible, y
    cuál depende de hacia dónde mire el animal.

    El hocico admite tres modos:
      - un nombre de bodypart concreto: se usa ese y sólo ese;
      - `MAS_CONFIABLE`: el más confiable del frame entre los candidatos presentes;
      - `None`: el primer candidato del listado que el modelo exponga. Es el comportamiento
        histórico, y se conserva sólo para que la etapa siga corriendo sin parámetros. Elegir por
        posición en una lista escrita a mano no es un criterio: preferir `barrido()` y fijar el
        resultado.

    Devuelve filas con el esquema de `keypoints_faciales.csv`, incluidos los nombres de los
    bodyparts efectivamente usados, para que la elección quede auditable frame por frame.
    """
    bodyparts = bodyparts_disponibles(filas)
    ojos = bodyparts_oculares(bodyparts, patron_ojo)
    presentes = candidatos_presentes(bodyparts, candidatos)

    if not ojos:
        raise ValueError(
            f"Ningún bodypart contiene {patron_ojo!r}. Disponibles: {bodyparts}"
        )

    if bodypart_hocico == MAS_CONFIABLE:
        hocicos = presentes
    elif bodypart_hocico is not None:
        if bodypart_hocico not in bodyparts:
            raise ValueError(
                f"El bodypart {bodypart_hocico!r} no está en el modelo. Disponibles: {bodyparts}"
            )
        hocicos = [bodypart_hocico]
    else:
        hocicos = presentes[:1]

    if not hocicos:
        raise ValueError(
            f"Ninguno de los candidatos {list(candidatos)} está en el modelo. "
            f"Disponibles: {bodyparts}"
        )

    resultado = []
    for (clip, orden), medidos in sorted(_agrupar_por_frame(filas).items()):
        n_ojo, x_ojo, y_ojo, c_ojo = _mejor_de(ojos, medidos)
        n_hoc, x_hoc, y_hoc, c_hoc = _mejor_de(hocicos, medidos)
        resultado.append({
            "clip": clip,
            "orden": orden,
            "x_ojo": x_ojo, "y_ojo": y_ojo, "conf_ojo": max(c_ojo, 0.0),
            "x_hocico": x_hoc, "y_hocico": y_hoc, "conf_hocico": max(c_hoc, 0.0),
            "bodypart_ojo": n_ojo or "",
            "bodypart_hocico": n_hoc or "",
        })
    return resultado


# --- barrido de candidatos -----------------------------------------------------------------


def _alineable(fila: dict, umbral: float) -> bool:
    """Un frame se alinea si ambos keypoints superan el umbral y tienen coordenadas."""
    if _es_nan(fila["x_ojo"]) or _es_nan(fila["x_hocico"]):
        return False
    return fila["conf_ojo"] >= umbral and fila["conf_hocico"] >= umbral


def _metadatos_clips() -> dict[str, dict]:
    """Clase y caballo de cada clip, para desglosar la cobertura."""
    ruta = rutas.carpeta("resultados") / "inventario_clips.csv"
    if not ruta.exists():
        return {}
    with ruta.open(encoding="utf-8", errors="replace") as archivo:
        return {f["clip"]: f for f in csv.DictReader(archivo)}


def _cobertura(pares: list[dict], umbral: float, meta: dict[str, dict]) -> dict:
    total = len(pares)
    ok = sum(1 for f in pares if _alineable(f, umbral))

    por_clase: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    por_caballo: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for fila in pares:
        info = meta.get(fila["clip"], {})
        alineado = _alineable(fila, umbral)
        for grupo, clave in ((por_clase, info.get("clase")), (por_caballo, info.get("id_caballo"))):
            if clave:
                grupo[clave][0] += 1
                grupo[clave][1] += alineado

    return {
        "frames": total,
        "alineables": ok,
        "tasa": ok / total if total else 0.0,
        "por_clase": {k: v[1] / v[0] for k, v in sorted(por_clase.items())},
        "por_caballo": {k: v[1] / v[0] for k, v in sorted(por_caballo.items())},
    }


def variacion_distancia(
    filas: list[dict],
    candidatos_reales: list[str],
    patron_ojo: str = PATRON_OJO,
) -> dict:
    """Cuánto cambia la distancia ojo-hocico según qué candidato se use.

    Es el dato que decide si `MAS_CONFIABLE` es admisible para el hocico. La transformada de
    similitud lleva el ojo y el hocico a posiciones fijas del recorte, así que la escala del recorte
    queda determinada por la distancia entre ambos en la imagen. Si el hocico alterna entre
    candidatos que están en lugares distintos del eje de la cabeza, esa distancia cambia por razones
    anatómicas y no de pose, y el recorte se agranda o achica en consecuencia.

    Devuelve la distancia mediana a cada candidato y la dispersión relativa mediana dentro del mismo
    frame: `(max - min) / mediana`, calculada sólo sobre frames donde hay al menos dos candidatos
    medidos.
    """
    ojos = bodyparts_oculares(bodyparts_disponibles(filas), patron_ojo)
    distancias: dict[str, list[float]] = defaultdict(list)
    dispersiones: list[float] = []

    for medidos in _agrupar_por_frame(filas).values():
        _, x_ojo, y_ojo, c_ojo = _mejor_de(ojos, medidos)
        if _es_nan(x_ojo):
            continue

        del_frame = {}
        for nombre in candidatos_reales:
            if nombre not in medidos:
                continue
            x, y, _ = medidos[nombre]
            if _es_nan(x) or _es_nan(y):
                continue
            d = math.hypot(x - x_ojo, y - y_ojo)
            del_frame[nombre] = d
            distancias[nombre].append(d)

        if len(del_frame) >= 2:
            valores = sorted(del_frame.values())
            central = valores[len(valores) // 2]
            if central > 0:
                dispersiones.append((valores[-1] - valores[0]) / central)

    def _mediana(xs: list[float]) -> float:
        if not xs:
            return float("nan")
        ordenados = sorted(xs)
        return ordenados[len(ordenados) // 2]

    return {
        "distancia_mediana": {k: _mediana(v) for k, v in sorted(distancias.items())},
        "dispersion_relativa_mediana": _mediana(dispersiones),
        "frames_comparables": len(dispersiones),
    }


def barrido(
    umbral: float = 0.50,
    candidatos: tuple[str, ...] = CANDIDATOS_HOCICO,
    patron_ojo: str = PATRON_OJO,
    peores: int = 8,
) -> dict:
    """Compara los candidatos a hocico sobre la pose ya persistida, sin reinferir.

    Informa, para cada candidato y para la estrategia de máxima confianza por frame, qué proporción
    de frames quedaría alineable. Es la respuesta a la pregunta abierta del pipeline: la cobertura
    quedó en 59,5 % con `nose`, y el diagnóstico mostró que el hocico frena 359 frames contra 10 del
    ojo.
    """
    filas = leer_pose()
    if not filas:
        raise ValueError(f"{ARCHIVO_POSE} no tiene filas.")

    bodyparts = bodyparts_disponibles(filas)
    presentes = candidatos_presentes(bodyparts, candidatos)
    ausentes = [c for c in candidatos if c.lower() not in {b.lower() for b in bodyparts}]
    meta = _metadatos_clips()

    print("=" * 62)
    print("BARRIDO DE CANDIDATOS A KEYPOINT DEL HOCICO")
    print("=" * 62)
    print(f"Bodyparts en el modelo : {len(bodyparts)}")
    print(f"Candidatos presentes   : {presentes or 'ninguno'}")
    if ausentes:
        print(f"Candidatos ausentes    : {ausentes}  (se omiten, no es un error)")
    print(f"Umbral de confianza    : {umbral}")

    if not presentes:
        raise ValueError(
            f"Ninguno de los candidatos {list(candidatos)} está en el modelo. "
            f"Disponibles: {bodyparts}"
        )

    estrategias: list[tuple[str, str]] = [(b, b) for b in presentes]
    if len(presentes) > 1:
        estrategias.append((MAS_CONFIABLE, "más confiable por frame"))

    resultados = {}
    print(f"\n{'estrategia':<28} {'alineables':>12}   {'tasa':>7}")
    print("-" * 62)
    for clave, etiqueta in estrategias:
        pares = seleccionar_par(filas, clave, candidatos, patron_ojo)
        datos = _cobertura(pares, umbral, meta)
        resultados[clave] = datos
        print(f"{etiqueta:<28} {datos['alineables']:>5} / {datos['frames']:<4}   "
              f"{datos['tasa']:>7.1%}")

    mejor = max(resultados, key=lambda k: resultados[k]["tasa"])
    etiqueta_mejor = "más confiable por frame" if mejor == MAS_CONFIABLE else mejor

    if meta:
        datos = resultados[mejor]
        print(f"\nDesglose de la mejor estrategia ({etiqueta_mejor}):")
        for clase, tasa in datos["por_clase"].items():
            print(f"  {clase:<14} {tasa:6.1%}")
        peores_caballos = sorted(datos["por_caballo"].items(), key=lambda kv: kv[1])[:peores]
        print(f"\n  Peores {len(peores_caballos)} ejemplares:")
        for caballo, tasa in peores_caballos:
            print(f"    {caballo:<7} {tasa:6.1%}")

    geometria = variacion_distancia(filas, presentes, patron_ojo)
    print("\nEfecto geométrico de alternar candidatos:")
    for nombre, d in geometria["distancia_mediana"].items():
        print(f"  distancia mediana ojo-{nombre:<12} {d:8.1f} px")
    dispersion = geometria["dispersion_relativa_mediana"]
    if not _es_nan(dispersion):
        print(f"  dispersión relativa mediana dentro del frame: {dispersion:.1%} "
              f"({geometria['frames_comparables']} frames comparables)")
        print("  Esa dispersión se traslada a la escala del recorte normalizado.")

    print("\n" + "-" * 62)
    print(f"Mejor cobertura: {etiqueta_mejor} con {resultados[mejor]['tasa']:.1%}")
    if mejor == MAS_CONFIABLE and not _es_nan(dispersion) and dispersion >= 0.10:
        print("ATENCIÓN: gana la estrategia de máxima confianza, pero la dispersión de la\n"
              "distancia ojo-hocico supera el 10 %. Un recorte con escala inconsistente entre\n"
              "frames del mismo clip contamina la señal temporal del objetivo 3. Conviene\n"
              "preferir un bodypart fijo aunque cubra menos.")

    return {
        "bodyparts": bodyparts,
        "presentes": presentes,
        "ausentes": ausentes,
        "umbral": umbral,
        "resultados": resultados,
        "geometria": geometria,
        "mejor": mejor,
    }
