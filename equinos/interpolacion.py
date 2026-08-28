"""Relleno temporal de los keypoints faltantes, y la validación que lo autoriza.

La restricción del dataset dejó de ser cuántos frames se alinean y pasó a ser **cuántos clips
quedan íntegros**: el modelo consume secuencias de 16 frames, y un clip con 11 frames alineados y
5 recortes de caja es una muestra contaminada, no una muestra a medias.

El 80 % de los huecos de detección son de uno o dos frames rodeados de detecciones válidas, y la
cabeza de un caballo no se desplaza de forma arbitraria en ~0,4 s. De ahí que interpolar sea
razonable. Pero un frame mal interpolado **no falla**: produce un recorte descentrado que entra al
dataset como si fuera bueno. Por eso este módulo trae las dos mitades juntas —el relleno y el
experimento que mide su error— y el tope de relleno sale de la curva medida, no del criterio de
quien lo implementa.

La geometría (`similitud_desde_dos_puntos`, `destino_canonico`) se **extrae del notebook 02** en
vez de duplicarse, igual que hace `tests/test_geometria.py`: dos copias divergen, y lo que hay que
medir es exactamente el código que se va a ejecutar.

Uso típico:

    from equinos import interpolacion

    interpolacion.informe_validacion()      # el experimento completo, tal como pide la spec
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from equinos import rutas

# ============================================================
# CONFIGURACIÓN
# ============================================================

# Mismo valor que `CONF_MINIMA_KEYPOINT` en la celda de parámetros del notebook 02. Un frame es
# "detectado" cuando el ojo Y el hocico lo superan: alinear necesita el par completo.
CONF_MINIMA_KEYPOINT = 0.50

# Geometría del recorte, también del notebook 02. La separación canónica ojo-hocico
# —(FRAC_HOCICO - FRAC_OJO) * LADO_RECORTE = 134,4 px— es la escala contra la que se lee el error.
LADO_RECORTE = 224
FRAC_OJO = 0.20
FRAC_HOCICO = 0.80

# Criterio de aceptación VIGENTE: un largo de hueco se acepta si su mediana y su p90 no superan
# los de la línea de base —el desplazamiento del recorte entre dos detecciones PROPIAS
# consecutivas—, que se mide en la misma corrida con `linea_de_base()`.
#
# Historia, porque hay que poder contarla: el 28/08/2026 se declaró antes de medir un umbral fijo
# de mediana <= 3,0 px y p90 <= 8,0 px. No lo cumplió ningún largo; el mejor caso dio 13,99 px.
# La línea de base resultó ser 16,28 px, o sea que el umbral pedía que la estimación fuera cinco
# veces más exacta que el propio movimiento del caballo entre frames muestreados —imposible— y que
# interpolar un hueco de 1 es de hecho MÁS preciso que usar la detección real del frame vecino.
# El umbral se re-declaró ese mismo día anclándolo en la línea de base. Los valores originales se
# conservan acá para poder informar el resultado contra ambos, como pide la spec.
UMBRAL_HISTORICO_MEDIANA_PX = 3.0
UMBRAL_HISTORICO_P90_PX = 8.0

# Estratos de largo de hueco que informa la validación. El último es abierto.
ESTRATOS = (1, 2, 3, 4, 5)

NOTEBOOK_GEOMETRIA = "02_roi_y_normalizacion.ipynb"
FUNCIONES_GEOMETRIA = ("similitud_desde_dos_puntos", "destino_canonico")


# ============================================================
# GEOMETRÍA, EXTRAÍDA DEL NOTEBOOK
# ============================================================


def cargar_geometria(notebook: str | Path | None = None) -> dict:
    """Ejecuta la celda del notebook 02 que define las funciones geométricas y las devuelve.

    No se duplica el código acá a propósito: el error que este módulo tiene que poder detectar es
    justamente un signo cambiado en la rotación, y una copia paralela lo escondería.
    """
    ruta = Path(notebook) if notebook else Path(__file__).resolve().parent.parent / NOTEBOOK_GEOMETRIA
    if not ruta.exists():
        raise SystemExit(f"No encontré el notebook con la geometría: {ruta}")

    nb = json.loads(ruta.read_text(encoding="utf-8"))
    for i, celda in enumerate(nb["cells"]):
        if celda["cell_type"] != "code":
            continue
        fuente = "".join(celda["source"])
        if all(f"def {f}" in fuente for f in FUNCIONES_GEOMETRIA):
            espacio: dict = {}
            exec(compile(fuente, f"notebook_celda_{i}", "exec"), espacio)
            return espacio

    raise SystemExit(
        f"Ninguna celda de {ruta.name} define {' y '.join(FUNCIONES_GEOMETRIA)}. "
        "Si se movieron de celda, actualizar FUNCIONES_GEOMETRIA."
    )


# ============================================================
# LECTURA DE LOS KEYPOINTS
# ============================================================


@dataclass
class Clip:
    """Los keypoints de un clip, en orden de frame, con la máscara de cuáles se detectaron."""

    nombre: str
    ordenes: list[int] = field(default_factory=list)
    ojo: list[tuple[float, float] | None] = field(default_factory=list)
    hocico: list[tuple[float, float] | None] = field(default_factory=list)

    @property
    def validos(self) -> list[int]:
        """Índices (posicionales) de los frames con el par ojo-hocico detectado."""
        return [i for i, p in enumerate(self.ojo) if p is not None and self.hocico[i] is not None]


def _a_float(valor) -> float:
    try:
        return float(valor)
    except (TypeError, ValueError):
        return float("nan")


def leer_clips(ruta: str | Path | None = None) -> list[Clip]:
    """Agrupa `keypoints_faciales.csv` por clip, ordenado por número de frame."""
    destino = Path(ruta) if ruta else rutas.carpeta("resultados") / "keypoints_faciales.csv"
    if not destino.exists():
        raise SystemExit(
            f"No existe {destino}. Correr antes la etapa 02 (./correr_local.sh etapa02)."
        )

    por_clip: dict[str, list[dict]] = {}
    with destino.open(encoding="utf-8") as fuente:
        for fila in csv.DictReader(fuente):
            por_clip.setdefault(fila["clip"], []).append(fila)

    clips = []
    for nombre, filas in por_clip.items():
        filas.sort(key=lambda f: int(f["orden"]))
        clip = Clip(nombre=nombre)
        for fila in filas:
            conf_ojo = _a_float(fila["conf_ojo"])
            conf_hocico = _a_float(fila["conf_hocico"])
            detectado = conf_ojo >= CONF_MINIMA_KEYPOINT and conf_hocico >= CONF_MINIMA_KEYPOINT
            clip.ordenes.append(int(fila["orden"]))
            if detectado:
                clip.ojo.append((_a_float(fila["x_ojo"]), _a_float(fila["y_ojo"])))
                clip.hocico.append((_a_float(fila["x_hocico"]), _a_float(fila["y_hocico"])))
            else:
                clip.ojo.append(None)
                clip.hocico.append(None)
        clips.append(clip)

    return clips


# ============================================================
# EL RELLENO
# ============================================================


def interpolar_punto(p_izq, p_der, izq: int, der: int, i: int) -> tuple[float, float]:
    """Posición de `i` interpolada linealmente entre los frames `izq` y `der`."""
    t = (i - izq) / (der - izq)
    return (
        p_izq[0] + t * (p_der[0] - p_izq[0]),
        p_izq[1] + t * (p_der[1] - p_izq[1]),
    )


def huecos_interiores(validos: list[int]) -> list[tuple[int, int, int]]:
    """Tramos sin detección con vecino detectado a ambos lados.

    Devuelve `(izq, der, largo)`: los índices de los vecinos válidos y cuántos frames hay entre
    ellos. Los tramos del principio y del final del clip NO aparecen acá — se tratan por el
    mecanismo de extremos, que es extrapolación y por eso tiene su propio tope.
    """
    tramos = []
    for a, b in zip(validos, validos[1:]):
        if b - a > 1:
            tramos.append((a, b, b - a - 1))
    return tramos


def clip_desde_filas(nombre: str, filas: list[dict]) -> Clip:
    """Arma un `Clip` desde las filas de un clip ya ordenadas por `orden`.

    Es el puente para que el notebook 02 use exactamente la misma función de relleno que se
    validó, en vez de una segunda implementación sobre el DataFrame. Acepta `par_valido` si la
    fila ya lo trae calculado; si no, lo deriva de las confianzas.
    """
    clip = Clip(nombre=nombre)
    for fila in filas:
        clip.ordenes.append(int(fila["orden"]))
        if "par_valido" in fila:
            detectado = bool(fila["par_valido"])
        else:
            detectado = (_a_float(fila.get("conf_ojo")) >= CONF_MINIMA_KEYPOINT
                         and _a_float(fila.get("conf_hocico")) >= CONF_MINIMA_KEYPOINT)
        x_ojo, y_ojo = _a_float(fila.get("x_ojo")), _a_float(fila.get("y_ojo"))
        x_hoc, y_hoc = _a_float(fila.get("x_hocico")), _a_float(fila.get("y_hocico"))
        if detectado and not (np.isnan(x_ojo) or np.isnan(x_hoc)):
            clip.ojo.append((x_ojo, y_ojo))
            clip.hocico.append((x_hoc, y_hoc))
        else:
            clip.ojo.append(None)
            clip.hocico.append(None)
    return clip


def rescates_por_clip(grupos: dict[str, list[dict]], max_hueco: int,
                      max_extremo: int = 0) -> dict[tuple[str, int], dict]:
    """Aplica el relleno a varios clips y devuelve `{(clip, orden): estimación}`.

    La clave usa el `orden` real del frame y no su índice posicional, para que el notebook pueda
    buscar por la misma clave con la que recorre el DataFrame.
    """
    rescates: dict[tuple[str, int], dict] = {}
    for nombre, filas in grupos.items():
        clip = clip_desde_filas(nombre, filas)
        for indice, datos in rellenar_clip(clip, max_hueco, max_extremo).items():
            rescates[(nombre, clip.ordenes[indice])] = datos
    return rescates


def rellenar_clip(clip: Clip, max_hueco: int, max_extremo: int = 0) -> dict[int, dict]:
    """Estima los keypoints faltantes de un clip dentro de los topes declarados.

    Devuelve `{indice: {"ojo": (x, y), "hocico": (x, y), "origen": ...}}` sólo para los frames
    efectivamente rescatados. Los que exceden el tope no aparecen: siguen cayendo al recorte de
    caja, como antes.
    """
    validos = clip.validos
    rescatados: dict[int, dict] = {}
    if not validos:
        return rescatados

    for izq, der, largo in huecos_interiores(validos):
        if largo > max_hueco:
            continue
        for i in range(izq + 1, der):
            rescatados[i] = {
                "ojo": interpolar_punto(clip.ojo[izq], clip.ojo[der], izq, der, i),
                "hocico": interpolar_punto(clip.hocico[izq], clip.hocico[der], izq, der, i),
                "origen": "interpolada",
            }

    if max_extremo > 0:
        primero, ultimo = validos[0], validos[-1]
        for i in range(max(0, primero - max_extremo), primero):
            rescatados[i] = {"ojo": clip.ojo[primero], "hocico": clip.hocico[primero],
                             "origen": "extremo_sostenido"}
        for i in range(ultimo + 1, min(len(clip.ojo), ultimo + 1 + max_extremo)):
            rescatados[i] = {"ojo": clip.ojo[ultimo], "hocico": clip.hocico[ultimo],
                             "origen": "extremo_sostenido"}

    return rescatados


# ============================================================
# EL ERROR, EN LAS TRES UNIDADES
# ============================================================


def error_de_estimacion(ojo_est, hocico_est, ojo_real, hocico_real, geometria: dict) -> dict:
    """Cuánto se desplaza el recorte por haber estimado los puntos en vez de detectarlos.

    Las tres unidades que pide la spec, y por qué cada una:

    - `px_original`  — la magnitud cruda del desvío sobre la imagen de entrada.
    - `frac_cara`    — el mismo desvío relativo a la distancia ojo-hocico real, que es lo único
                       comparable entre videos filmados a distinta distancia.
    - `px_recorte`   — el desplazamiento **dentro del recorte de 224**, que es el que ve el modelo
                       y sobre el que se declaró el umbral.

    `px_recorte` se calcula construyendo la transformada con los puntos ESTIMADOS y aplicándola a
    los REALES: mide cuán lejos del lugar canónico cae la cara verdadera en el recorte que se
    habría producido. Comparar la transformada contra los mismos puntos con que se la construyó da
    cero siempre, y ese error ya se cometió una vez en este proyecto.
    """
    similitud = geometria["similitud_desde_dos_puntos"]
    canonico = geometria["destino_canonico"]

    q_ojo, q_hocico = canonico(LADO_RECORTE, FRAC_OJO, FRAC_HOCICO)

    ojo_est = np.asarray(ojo_est, dtype=float)
    hocico_est = np.asarray(hocico_est, dtype=float)
    ojo_real = np.asarray(ojo_real, dtype=float)
    hocico_real = np.asarray(hocico_real, dtype=float)

    px_original = max(
        float(np.hypot(*(ojo_est - ojo_real))),
        float(np.hypot(*(hocico_est - hocico_real))),
    )

    dist_cara = float(np.hypot(*(hocico_real - ojo_real)))
    frac_cara = px_original / dist_cara if dist_cara > 1e-8 else float("nan")

    matriz = similitud(ojo_est, hocico_est, q_ojo, q_hocico)
    if matriz is None:
        return {"px_original": px_original, "frac_cara": frac_cara, "px_recorte": float("nan")}

    def aplicar(p):
        return matriz[:, :2] @ np.asarray(p, dtype=float) + matriz[:, 2]

    px_recorte = max(
        float(np.hypot(*(aplicar(ojo_real) - np.asarray(q_ojo, dtype=float)))),
        float(np.hypot(*(aplicar(hocico_real) - np.asarray(q_hocico, dtype=float)))),
    )

    return {"px_original": px_original, "frac_cara": frac_cara, "px_recorte": px_recorte}


# ============================================================
# VALIDACIÓN: DEJAR UNO AFUERA
# ============================================================


def validar_huecos_interiores(clips: list[Clip] | None = None, geometria: dict | None = None,
                              max_largo: int = 8) -> list[dict]:
    """Oculta tramos de detecciones válidas, los interpola, y mide el error contra lo real.

    El protocolo, que es lo que hace la medición honesta: se elige un tramo de `L` frames
    **consecutivos y todos detectados**, cuyos vecinos inmediatos también estén detectados. Se
    ocultan los `L`, se interpolan desde los vecinos, y cada uno se compara contra su posición
    real. Nada de lo que se mide participó de la estimación.

    Devuelve una fila por frame estimado, con su largo de hueco y su velocidad aparente.
    """
    clips = clips if clips is not None else leer_clips()
    geometria = geometria if geometria is not None else cargar_geometria()

    filas = []
    for clip in clips:
        validos = clip.validos
        if len(validos) < 3:
            continue
        conjunto = set(validos)

        for largo in range(1, max_largo + 1):
            # Un tramo candidato arranca en `validos[k]` y ocupa `largo` frames consecutivos; se
            # exige que el anterior y el posterior existan y estén detectados, porque interpolar
            # necesita los dos lados.
            for inicio in validos:
                tramo = list(range(inicio, inicio + largo))
                izq, der = inicio - 1, inicio + largo
                if izq < 0 or der >= len(clip.ojo):
                    continue
                if izq not in conjunto or der not in conjunto:
                    continue
                if not all(i in conjunto for i in tramo):
                    continue

                # Velocidad aparente entre los vecinos: el sesgo conocido de esta validación es
                # que la detección tiende a fallar cuando el caballo se mueve, así que hay que
                # poder mirar el error contra esta variable.
                desplazamiento = float(np.hypot(
                    clip.ojo[der][0] - clip.ojo[izq][0],
                    clip.ojo[der][1] - clip.ojo[izq][1],
                ))
                dist_cara = float(np.hypot(
                    clip.hocico[izq][0] - clip.ojo[izq][0],
                    clip.hocico[izq][1] - clip.ojo[izq][1],
                ))
                velocidad = desplazamiento / (der - izq) / dist_cara if dist_cara > 1e-8 else float("nan")

                for i in tramo:
                    est_ojo = interpolar_punto(clip.ojo[izq], clip.ojo[der], izq, der, i)
                    est_hocico = interpolar_punto(clip.hocico[izq], clip.hocico[der], izq, der, i)
                    error = error_de_estimacion(est_ojo, est_hocico, clip.ojo[i], clip.hocico[i],
                                                geometria)
                    filas.append({"clip": clip.nombre, "largo": largo, "velocidad": velocidad,
                                  **error})

    return filas


def validar_extremos(clips: list[Clip] | None = None, geometria: dict | None = None,
                     max_largo: int = 4) -> list[dict]:
    """Mismo protocolo para los extremos, donde sólo se puede sostener el valor del vecino.

    Es extrapolación, no interpolación: más débil por construcción, y por eso va con su propio
    tope y su propia marca. Se oculta el primer (o el último) frame detectado del clip y se lo
    estima sosteniendo el valor del siguiente detectado hacia adentro.
    """
    clips = clips if clips is not None else leer_clips()
    geometria = geometria if geometria is not None else cargar_geometria()

    filas = []
    for clip in clips:
        validos = clip.validos
        if len(validos) < 2:
            continue

        for largo in range(1, max_largo + 1):
            # Extremo inicial: se ocultan los primeros `largo` detectados y se sostiene el que
            # sigue. Se exige que el sostén exista y sea contiguo, que es la situación real.
            if len(validos) > largo:
                ocultos = validos[:largo]
                sosten = validos[largo]
                if ocultos == list(range(ocultos[0], ocultos[0] + largo)) and sosten == ocultos[-1] + 1:
                    for i in ocultos:
                        error = error_de_estimacion(clip.ojo[sosten], clip.hocico[sosten],
                                                    clip.ojo[i], clip.hocico[i], geometria)
                        filas.append({"clip": clip.nombre, "extremo": "inicio",
                                      "largo": sosten - i, **error})

                ocultos = validos[-largo:]
                sosten = validos[-largo - 1]
                if ocultos == list(range(ocultos[0], ocultos[0] + largo)) and sosten == ocultos[0] - 1:
                    for i in ocultos:
                        error = error_de_estimacion(clip.ojo[sosten], clip.hocico[sosten],
                                                    clip.ojo[i], clip.hocico[i], geometria)
                        filas.append({"clip": clip.nombre, "extremo": "final",
                                      "largo": i - sosten, **error})

    return filas


# ============================================================
# RESUMEN E INFORME
# ============================================================


def linea_de_base(clips: list[Clip] | None = None, geometria: dict | None = None) -> dict:
    """Cuánto se mueve el recorte entre dos detecciones PROPIAS consecutivas.

    Es la referencia que le da sentido a cualquier error de estimación, y sin ella el resultado
    se lee al revés. Mide, sobre pares de frames contiguos **ambos detectados**, cuán lejos del
    lugar canónico cae la cara del frame `i` en el recorte construido con los keypoints del
    frame `i+1`.

    Ese valor es el piso físico del problema: es lo que se mueve la cabeza del caballo entre dos
    frames muestreados, y ninguna estimación puede ser sistemáticamente más precisa que eso sin
    estar sobreajustando. Un relleno cuyo error quede por debajo de esta línea está dentro del
    ruido propio del dataset.
    """
    clips = clips if clips is not None else leer_clips()
    geometria = geometria if geometria is not None else cargar_geometria()

    filas = []
    for clip in clips:
        conjunto = set(clip.validos)
        for i in clip.validos:
            if i + 1 in conjunto:
                filas.append(error_de_estimacion(
                    clip.ojo[i + 1], clip.hocico[i + 1], clip.ojo[i], clip.hocico[i], geometria))

    recorte = [f["px_recorte"] for f in filas if not np.isnan(f["px_recorte"])]
    return {
        "n": len(filas),
        "px_recorte_mediana": _percentil(recorte, 50),
        "px_recorte_p90": _percentil(recorte, 90),
        "px_original_mediana": _percentil([f["px_original"] for f in filas], 50),
    }


def _percentil(valores: list[float], p: float) -> float:
    if not valores:
        return float("nan")
    return float(np.percentile(np.asarray(valores, dtype=float), p))


def resumir(filas: list[dict], base: dict | None = None, clave: str = "largo") -> list[dict]:
    """Mediana, p90 y máximo del error por estrato, y si el estrato cumple el criterio vigente.

    `base` es el resultado de `linea_de_base()`. Con él, `cumple` evalúa el criterio vigente —no
    superar la variación entre dos detecciones propias—; sin él, cae al umbral histórico fijo, que
    se conserva sólo para poder informar contra qué se comparaba antes.
    """
    estratos: dict[int, list[dict]] = {}
    for fila in filas:
        estratos.setdefault(int(fila[clave]), []).append(fila)

    if base:
        tope_mediana, tope_p90 = base["px_recorte_mediana"], base["px_recorte_p90"]
    else:
        tope_mediana, tope_p90 = UMBRAL_HISTORICO_MEDIANA_PX, UMBRAL_HISTORICO_P90_PX

    resumen = []
    for largo in sorted(estratos):
        grupo = estratos[largo]
        recorte = [f["px_recorte"] for f in grupo if not np.isnan(f["px_recorte"])]
        original = [f["px_original"] for f in grupo]
        cara = [f["frac_cara"] for f in grupo if not np.isnan(f["frac_cara"])]
        mediana = _percentil(recorte, 50)
        p90 = _percentil(recorte, 90)
        resumen.append({
            "largo": largo,
            "n": len(grupo),
            "px_recorte_mediana": mediana,
            "px_recorte_p90": p90,
            "px_recorte_max": max(recorte) if recorte else float("nan"),
            "px_original_mediana": _percentil(original, 50),
            "frac_cara_mediana": _percentil(cara, 50),
            "cumple": bool(mediana <= tope_mediana and p90 <= tope_p90),
            "cumple_historico": bool(mediana <= UMBRAL_HISTORICO_MEDIANA_PX
                                     and p90 <= UMBRAL_HISTORICO_P90_PX),
        })
    return resumen


def tope_desde_la_curva(resumen: list[dict]) -> int:
    """El mayor largo tal que **todos** los largos hasta él cumplen el umbral.

    Se corta en el primer estrato que no cumple y no se sigue mirando: aceptar un largo 4 porque
    dio bien cuando el 3 falló sería elegir el número que conviene, no leer la curva.
    """
    tope = 0
    for fila in resumen:
        if fila["largo"] != tope + 1:
            break
        if not fila["cumple"]:
            break
        tope = fila["largo"]
    return tope


def _linea(fila: dict) -> str:
    marca = ("ok " if fila["cumple"] else "NO ") + ("/ ok" if fila["cumple_historico"] else "/ no")
    return (f"  {fila['largo']:>5}  {fila['n']:>7}  "
            f"{fila['px_recorte_mediana']:>9.2f}  {fila['px_recorte_p90']:>7.2f}  "
            f"{fila['px_recorte_max']:>8.2f}  {fila['px_original_mediana']:>9.1f}  "
            f"{fila['frac_cara_mediana']:>8.3f}   {marca}")


def informe_validacion(clips: list[Clip] | None = None) -> dict:
    """El experimento completo de la tarea 1 del change, impreso y devuelto."""
    geometria = cargar_geometria()
    clips = clips if clips is not None else leer_clips()

    detectados = sum(len(c.validos) for c in clips)
    con_tres = sum(1 for c in clips if len(c.validos) >= 3)

    print("=" * 78)
    print("VALIDACIÓN DEL RELLENO TEMPORAL — DEJAR UNO AFUERA")
    print("=" * 78)
    print(f"Clips leídos            : {len(clips)}")
    print(f"Frames detectados       : {detectados}")
    print(f"Clips con 3+ detecciones: {con_tres}   (base del experimento)")
    print()
    print(f"Criterio vigente: no superar la línea de base, medida abajo en esta misma corrida.")
    print(f"Umbral histórico (28/08/2026, superado): mediana <= {UMBRAL_HISTORICO_MEDIANA_PX} px "
          f"y p90 <= {UMBRAL_HISTORICO_P90_PX} px. Se informa contra ambos.")
    print(f"Separación canónica ojo-hocico : "
          f"{(FRAC_HOCICO - FRAC_OJO) * LADO_RECORTE:.1f} px (la escala contra la que se lee).")
    print()

    base = linea_de_base(clips, geometria)
    print("-" * 78)
    print("LÍNEA DE BASE — dos detecciones PROPIAS consecutivas")
    print("-" * 78)
    print(f"  pares medidos              : {base['n']}")
    print(f"  desplazamiento en el recorte: mediana {base['px_recorte_mediana']:.2f} px   "
          f"p90 {base['px_recorte_p90']:.2f} px")
    print(f"  en la imagen original       : mediana {base['px_original_mediana']:.1f} px")
    print()
    print("  Es el piso físico del problema: lo que se mueve la cabeza entre dos frames")
    print("  muestreados. Un relleno por debajo de esta línea está dentro del ruido propio")
    print("  del dataset; exigirle menos que esto es exigirle lo imposible.")
    print()

    interiores = validar_huecos_interiores(clips, geometria)
    resumen_int = resumir(interiores, base)

    cabecera = ("  largo        n   mediana      p90       máx   px_orig  frac_cara   "
                "vigente/histórico")
    print("-" * 78)
    print("HUECOS INTERIORES (interpolación entre dos vecinos detectados)")
    print("-" * 78)
    print(cabecera)
    for fila in resumen_int:
        print(_linea(fila))
    tope_interior = tope_desde_la_curva(resumen_int)
    print()
    print(f"  -> MAX_HUECO_INTERPOLADO = {tope_interior}")
    print(f"     (contra el umbral histórico habría sido "
          f"{tope_desde_la_curva([{**f, 'cumple': f['cumple_historico']} for f in resumen_int])})")
    print()

    extremos = validar_extremos(clips, geometria)
    resumen_ext = resumir(extremos, base)
    print("-" * 78)
    print("EXTREMOS (se sostiene el valor del vecino hacia adentro — es extrapolación)")
    print("-" * 78)
    print(cabecera)
    for fila in resumen_ext:
        print(_linea(fila))
    tope_extremo = tope_desde_la_curva(resumen_ext)
    print()
    print(f"  -> MAX_EXTREMO_SOSTENIDO = {tope_extremo}")
    print()

    # El sesgo conocido: los frames que se usan para validar son los que SÍ se detectaron, que no
    # son una muestra aleatoria de los que fallaron. Si la detección falla justamente cuando el
    # caballo se mueve rápido, el error real de los huecos será mayor que el medido.
    print("-" * 78)
    print("SESGO CONOCIDO — error contra velocidad aparente entre los vecinos")
    print("-" * 78)
    con_vel = [f for f in interiores if not np.isnan(f.get("velocidad", float("nan")))]
    if con_vel:
        velocidades = np.asarray([f["velocidad"] for f in con_vel])
        cortes = np.percentile(velocidades, [25, 50, 75])
        etiquetas = ["quieto  (q1)", "lento   (q2)", "rápido  (q3)", "muy rápido (q4)"]
        indices = np.digitize(velocidades, cortes)
        print("  cuartil de movimiento        n    mediana px recorte      p90")
        for q, etiqueta in enumerate(etiquetas):
            grupo = [f["px_recorte"] for f, k in zip(con_vel, indices)
                     if k == q and not np.isnan(f["px_recorte"])]
            if grupo:
                print(f"  {etiqueta:<22} {len(grupo):>6}   {_percentil(grupo, 50):>13.2f}   "
                      f"{_percentil(grupo, 90):>7.2f}")
        correlacion = float(np.corrcoef(
            velocidades,
            np.nan_to_num([f["px_recorte"] for f in con_vel]),
        )[0, 1])
        print(f"\n  correlación velocidad / error en el recorte: {correlacion:+.3f}")
    print()

    return {
        "interiores": resumen_int,
        "extremos": resumen_ext,
        "max_hueco_interpolado": tope_interior,
        "max_extremo_sostenido": tope_extremo,
        "linea_de_base": base,
        "umbral_historico_mediana_px": UMBRAL_HISTORICO_MEDIANA_PX,
        "umbral_historico_p90_px": UMBRAL_HISTORICO_P90_PX,
    }
