"""El conjunto de clips que entra al modelo, y la partición por ejemplar.

Este módulo no entrena ni evalúa nada: decide **qué clips entran** y **cómo se cortan**, y
persiste esa decisión. Existe separado porque la partición tiene que ser idéntica en todas las
corridas —las líneas de base, cada variante de cabeza temporal, la ablación— o los números dejan
de ser comparables entre sí, que es justamente lo que se quiere comparar.

**La unidad es el clip íntegro.** Un clip con frames mezclados —parte alineados por keypoints,
parte recortes de caja— es una muestra contaminada: a qué frame le toca cuál correlaciona con el
ejemplar, y el ejemplar predice la clase. Se exige que los 16 frames estén alineados.

**El grupo es el EJEMPLAR, no el par ejemplar-clase.** Hay caballos con clips íntegros en más de
una clase; agrupar por el par los pondría de los dos lados del corte. Eso no da error: da una
métrica optimista y falsa.

**`GroupKFold` no lleva `random_state` y está bien que no lo lleve.** Es determinista: reparte los
grupos buscando folds de tamaño parejo, sin sortear. Así que la reproducibilidad no depende de
fijar una semilla sino de que la entrada llegue siempre en el mismo orden, y por eso los grupos se
ordenan explícitamente antes de repartir. La partición se persiste igual, porque el día que cambie
el dataset los cortes van a cambiar y hay que poder saber sobre cuáles se midió cada resultado.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from equinos import rutas

# Frames que tiene un clip completo. Un clip con menos no es íntegro y no entra.
FRAMES_POR_CLIP = 16

# Folds de la validación cruzada. 4 y no 5: con 5 hay folds con apenas 2 clips de `Disgustado` en
# test, y una métrica calculada sobre dos muestras no dice nada. Verificado sobre los datos reales.
FOLDS = 4

ARCHIVO_PARTICION = "particion_clips.json"

# Orígenes de alineación que cuentan como detección propia del modelo de pose. La ablación del
# objetivo 3 entrena con y sin los estimados, y este es el filtro que la hace posible.
ORIGENES_DETECTADOS = ("keypoints",)


# El dataset vigente. `dataset_frames_alineados.csv` es el de la corrida vieja —muestreo de 16
# posiciones fijas— y se conserva para poder reproducir sus números; el v2 sale de la selección
# por ventana de `equinos.secuencia`. Cuál se lee es explícito y no depende de qué archivo exista.
ARCHIVO_FRAMES = "dataset_frames_alineados_v2.csv"


def _leer_frames(archivo: str | None = None) -> list[dict]:
    ruta = rutas.carpeta("resultados") / (archivo or ARCHIVO_FRAMES)
    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe {ruta}. La produce `python3 -m equinos.dataset`.")
    with ruta.open(encoding="utf-8", errors="replace") as archivo:
        return list(csv.DictReader(archivo))


def _es_si(valor) -> bool:
    return str(valor).strip().lower() in ("true", "1", "sí", "si")


def tabla_clips(solo_detectados: bool = False, archivo: str | None = None) -> list[dict]:
    """Un registro por clip íntegro, con lo que hace falta para particionar y evaluar.

    `solo_detectados` deja únicamente los clips cuyos 16 frames se alinearon por detección propia,
    sin interpolación ni anotación manual. Es el conjunto de la ablación: NO es "el bueno" y el
    otro "el malo" — es menos material, y sirve para contestar cuánto del resultado se apoya en
    posiciones estimadas.
    """
    por_clip: dict[str, list[dict]] = defaultdict(list)
    for fila in _leer_frames(archivo):
        por_clip[fila["clip"]].append(fila)

    clips = []
    for clip, filas in por_clip.items():
        alineados = [f for f in filas if _es_si(f["alineado"])]
        # Los 16, no "al menos 12". El 12..16 que estuvo vigente admitía clips donde hasta 4
        # frames eran recortes de CAJA sin alinear, y `caracteristicas.extraer` no distingue: el
        # backbone comía dos geometrías dentro de la misma secuencia. Con la selección por
        # ventana dejó de haber disyuntiva —los 16 salen de mediciones reales—, así que se vuelve
        # al criterio que el docstring y `tests/test_particion.py` siempre declararon.
        if len(alineados) != FRAMES_POR_CLIP or len(filas) != FRAMES_POR_CLIP:
            continue
        origenes = {f["origen_alineacion"] for f in alineados}
        if solo_detectados and not origenes <= set(ORIGENES_DETECTADOS):
            continue
        primera = filas[0]
        clips.append({
            "clip": clip,
            "clase": primera["clase"],
            "id_caballo": str(primera["id_caballo"]),
            "origenes": sorted(origenes),
            "solo_detectado": origenes <= set(ORIGENES_DETECTADOS),
        })

    # Orden estable: la partición no puede depender del orden en que el sistema de archivos
    # devolvió las filas.
    return sorted(clips, key=lambda c: c["clip"])


def composicion(clips: list[dict]) -> dict:
    """Clips y ejemplares por clase, y los ejemplares que aparecen en más de una clase."""
    por_clase: dict[str, set] = defaultdict(set)
    clips_por_clase: dict[str, int] = defaultdict(int)
    clases_por_caballo: dict[str, set] = defaultdict(set)
    for c in clips:
        por_clase[c["clase"]].add(c["id_caballo"])
        clips_por_clase[c["clase"]] += 1
        clases_por_caballo[c["id_caballo"]].add(c["clase"])

    multiclase = sorted(k for k, v in clases_por_caballo.items() if len(v) > 1)
    return {
        "clips": len(clips),
        "ejemplares": len(clases_por_caballo),
        "clips_por_clase": dict(clips_por_clase),
        "ejemplares_por_clase": {k: len(v) for k, v in por_clase.items()},
        "ejemplares_en_varias_clases": multiclase,
    }


def construir(clips: list[dict], folds: int = FOLDS) -> list[dict]:
    """Reparte los EJEMPLARES en `folds` grupos, y con ellos los clips.

    Réplica determinista de `GroupKFold`: los ejemplares se ordenan por cantidad de clips
    descendente y se van asignando al fold que menos clips tiene. Se implementa acá en vez de
    llamar a sklearn para que el módulo no dependa del stack científico —que DeepLabCut deja en
    un estado frágil— y para poder fijar el desempate por nombre de ejemplar, que es lo que
    vuelve la partición reproducible sin semilla.
    """
    if folds < 2:
        raise ValueError(f"Hacen falta al menos 2 folds, se pidieron {folds}.")

    clips_por_caballo: dict[str, list[str]] = defaultdict(list)
    for c in clips:
        clips_por_caballo[c["id_caballo"]].append(c["clip"])

    # Desempate por nombre: sin esto, dos ejemplares con la misma cantidad de clips podrían caer
    # en folds distintos según el orden de iteración del diccionario.
    orden = sorted(clips_por_caballo, key=lambda cab: (-len(clips_por_caballo[cab]), cab))

    carga = [0] * folds
    fold_de_caballo: dict[str, int] = {}
    for caballo in orden:
        destino = min(range(folds), key=lambda f: (carga[f], f))
        fold_de_caballo[caballo] = destino
        carga[destino] += len(clips_por_caballo[caballo])

    return [dict(c, fold=fold_de_caballo[c["id_caballo"]]) for c in clips]


def dividir_por_ejemplar(clips: list[dict], fraccion_validacion: float = 0.25) -> tuple[list, list]:
    """Parte un conjunto de clips en entrenamiento y validación SIN que un ejemplar cruce.

    Es el corte interno del fold: `evaluar` ya aisló un fold de test por ejemplar, y acá se vuelve
    a cortar lo que queda para poder elegir la época, el dropout y cualquier otra decisión sin
    mirar el test ni una vez. Un ejemplar que aparezca de los dos lados de ESTE corte no infla la
    métrica final —el test sigue limpio— pero sí infla la validación, y con ella se elige mal la
    época: el modelo parece seguir mejorando porque está reconociendo al caballo.

    El reparto reusa el criterio de `construir`: ejemplares ordenados por cantidad de clips y
    asignados al lado que menos tiene. Determinista, sin sorteo, así que dos corridas del mismo
    conjunto dan el mismo corte y las comparaciones entre configuraciones son limpias.
    """
    if not 0.0 < fraccion_validacion < 1.0:
        raise ValueError(f"La fracción de validación tiene que estar entre 0 y 1: "
                         f"{fraccion_validacion}")

    clips_por_caballo: dict[str, list[dict]] = defaultdict(list)
    for c in clips:
        clips_por_caballo[c["id_caballo"]].append(c)

    objetivo = len(clips) * fraccion_validacion
    orden = sorted(clips_por_caballo, key=lambda k: (-len(clips_por_caballo[k]), k))

    validacion_ids: set[str] = set()
    acumulado = 0
    for caballo in orden:
        if acumulado >= objetivo:
            break
        validacion_ids.add(caballo)
        acumulado += len(clips_por_caballo[caballo])

    entrenamiento = [c for c in clips if c["id_caballo"] not in validacion_ids]
    validacion = [c for c in clips if c["id_caballo"] in validacion_ids]

    cruce = {c["id_caballo"] for c in entrenamiento} & {c["id_caballo"] for c in validacion}
    assert not cruce, f"fuga de ejemplar entre train interno y validación: {sorted(cruce)}"
    assert entrenamiento and validacion, (
        f"un lado del corte interno quedó vacío: {len(entrenamiento)} train, "
        f"{len(validacion)} validación, sobre {len(clips_por_caballo)} ejemplares")
    return entrenamiento, validacion


def verificar(particionados: list[dict], folds: int = FOLDS) -> dict:
    """Comprueba lo que, si falla, no da error sino una métrica falsa.

    Devuelve un informe; levanta `AssertionError` sólo ante fuga de ejemplar, que es el fallo que
    invalida todo lo que venga después.
    """
    caballos_por_fold: dict[int, set] = defaultdict(set)
    clases_por_fold: dict[int, dict] = defaultdict(lambda: defaultdict(int))
    for c in particionados:
        caballos_por_fold[c["fold"]].add(c["id_caballo"])
        clases_por_fold[c["fold"]][c["clase"]] += 1

    for f in range(folds):
        otros = set().union(*(caballos_por_fold[g] for g in range(folds) if g != f)) \
            if folds > 1 else set()
        fuga = caballos_por_fold[f] & otros
        assert not fuga, f"fuga de ejemplar entre folds: {sorted(fuga)}"

    clases = sorted({c["clase"] for c in particionados})
    vacias = {f: [cl for cl in clases if clases_por_fold[f][cl] == 0] for f in range(folds)}
    return {
        "folds": folds,
        "clips_por_fold": {f: sum(clases_por_fold[f].values()) for f in range(folds)},
        "ejemplares_por_fold": {f: len(caballos_por_fold[f]) for f in range(folds)},
        "clases_por_fold": {f: dict(clases_por_fold[f]) for f in range(folds)},
        "clases_vacias": {f: v for f, v in vacias.items() if v},
        "sin_fuga": True,
    }


def guardar(particionados: list[dict], nombre: str = ARCHIVO_PARTICION) -> Path:
    ruta = rutas.carpeta("resultados") / nombre
    ruta.write_text(json.dumps(
        {"folds": FOLDS, "frames_por_clip": FRAMES_POR_CLIP,
         "clips": [{k: c[k] for k in ("clip", "clase", "id_caballo", "fold", "solo_detectado")}
                   for c in particionados]},
        ensure_ascii=False, indent=1), encoding="utf-8")
    return ruta


def cargar(nombre: str = ARCHIVO_PARTICION) -> list[dict]:
    """Lee la partición persistida. Es la única fuente admitida para comparar corridas."""
    ruta = rutas.carpeta("resultados") / nombre
    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe {ruta}. Construirla con `particion.construir()` y `particion.guardar()`; "
            "comparar dos corridas sobre particiones distintas no dice nada.")
    return json.loads(ruta.read_text(encoding="utf-8"))["clips"]


def informe() -> list[dict]:
    """Construye, verifica, persiste e imprime la partición. Es el punto de entrada del notebook."""
    clips = tabla_clips()
    comp = composicion(clips)

    print("=" * 66)
    print("CONJUNTO DE CLIPS Y PARTICIÓN POR EJEMPLAR")
    print("=" * 66)
    print(f"Clips íntegros : {comp['clips']}")
    print(f"Ejemplares     : {comp['ejemplares']}")
    print("\nPor clase:")
    for clase in sorted(comp["clips_por_clase"]):
        print(f"  {clase:<12} {comp['clips_por_clase'][clase]:>4} clips  "
              f"{comp['ejemplares_por_clase'][clase]:>3} ejemplares")

    multi = comp["ejemplares_en_varias_clases"]
    print(f"\nEjemplares con clips íntegros en MÁS DE UNA clase: {len(multi)}")
    print(f"  {', '.join(multi)}")
    print("  Por eso el grupo es el ejemplar y no el par ejemplar-clase: agrupar por el par los")
    print("  pondría de los dos lados del corte, y la métrica saldría optimista sin dar error.")

    solo_det = tabla_clips(solo_detectados=True)
    print(f"\nClips por detección propia (conjunto de la ablación): {len(solo_det)} de {comp['clips']}")

    particionados = construir(clips)
    info = verificar(particionados)
    print(f"\nPartición en {info['folds']} folds — sin fuga de ejemplar verificada")
    print(f"{'fold':<6}{'clips':>7}{'ejemplares':>12}   por clase")
    print("-" * 62)
    for f in range(info["folds"]):
        clases = info["clases_por_fold"][f]
        detalle = "  ".join(f"{k} {v}" for k, v in sorted(clases.items()))
        print(f"{f:<6}{info['clips_por_fold'][f]:>7}{info['ejemplares_por_fold'][f]:>12}   {detalle}")

    if info["clases_vacias"]:
        print(f"\nADVERTENCIA: hay folds sin alguna clase: {info['clases_vacias']}")
        print("Una métrica por clase sobre un fold vacío no existe. Revisar antes de seguir.")
    else:
        print("\nNinguna clase queda vacía en ningún fold.")

    ruta = guardar(particionados)
    print(f"\nPartición persistida en: {ruta}")
    print("Todas las corridas —líneas de base, variantes de cabeza, ablación— tienen que leer")
    print("ESTA partición. Comparar números medidos sobre cortes distintos no dice nada.")
    return particionados
