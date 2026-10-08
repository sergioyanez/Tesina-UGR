"""Protocolo común para comparar modelos de clasificación (ver `comparacion/PROTOCOLO.md`).

Todos los candidatos se miden con ESTE módulo, sobre las MISMAS particiones y con las MISMAS
métricas. Un candidato sólo implementa `ajustar` y `predecir` sobre ids de clip; el evaluador
reparte, mide y escribe `comparacion/resultados/<candidato>.json` con un esquema fijo.

Piezas:

  - **reserva** (`construir_reserva`, `cargar_reserva`): 15 % de los caballos, estratificada por
    clase dominante del caballo, semilla fija, persistida en `comparacion/reserva_caballos.json`
    con la huella del dataset. NINGUNA evaluación de desarrollo la toca. La única puerta es
    `evaluar_en_reserva`, que exige un flag explícito, registra cada uso y se niega a un segundo.
  - **particiones** (`construir_particiones`): R = 5 repeticiones de 4 folds por caballo
    (`StratifiedGroupKFold`, semillas fijas) sobre los caballos de desarrollo, en
    `comparacion/particiones.json`.
  - **evaluador** (`evaluar`): F1 macro, accuracy, P/R/F1 por clase, matrices de confusión, media
    ± desvío sobre las 20 evaluaciones y por repetición, subconjunto limpio, gap train−test.
  - **comparación** (`comparar`): IC95 bootstrap pareado, remuestreando CABALLOS (no clips).

scikit-learn y numpy se importan dentro de las funciones.
"""

from __future__ import annotations

import hashlib
import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Protocol, Sequence

CLASES = ("Disgustado", "Interesado", "Sereno")

RAIZ_REPO = Path(__file__).resolve().parent.parent
CARPETA = RAIZ_REPO / "comparacion"
ARCHIVO_RESERVA = CARPETA / "reserva_caballos.json"
ARCHIVO_PARTICIONES = CARPETA / "particiones.json"
ARCHIVO_USO_RESERVA = CARPETA / "reserva_uso.log"
CARPETA_RESULTADOS = CARPETA / "resultados"

# Decisión de Sergio (07/10/2026): 15 % de los caballos, una sola vez, con el ganador.
FRACCION_RESERVA = 0.15
SEMILLA_RESERVA = 20261007
REPETICIONES = 5
FOLDS = 4
SEMILLAS_PARTICION = (1101, 1102, 1103, 1104, 1105)
REPLICAS_BOOTSTRAP = 2000
SEMILLA_BOOTSTRAP = 7
ESQUEMA = "comparacion-v1"


# --- datos --------------------------------------------------------------------------------------

def huella_dataset(clips: Sequence[dict]) -> str:
    """Identifica el conjunto de clips (id, caballo, clase). Cambia si cambia el dataset."""
    partes = sorted(f"{c['clip']}|{c['id_caballo']}|{c['clase']}" for c in clips)
    return hashlib.sha256("\n".join(partes).encode()).hexdigest()[:16]


def _huella_json(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


class Datos:
    """Todo lo que un candidato puede necesitar, indexado por id de clip. Carga perezosa.

    `clips`: lista de `{clip, clase, id_caballo, ...}` (los 359 íntegros). `etiqueta(id)`,
    `caballo(id)`. `visual(id)`: (16, 512) del backbone congelado. `pose(ids)`: filas de pose de los
    16 frames elegidos (orden = posición). `meta(id)`: duración, fps, ancho, alto, ventana.
    Sólo lectura sobre la raíz de datos.
    """

    def __init__(self, clips: list[dict] | None = None):
        from equinos import particion

        self.clips = clips if clips is not None else particion.tabla_clips()
        self._por_id = {c["clip"]: c for c in self.clips}
        self.huella = huella_dataset(self.clips)
        self._visual = None
        self._pose = None
        self._meta = None

    def ids(self) -> list[str]:
        return [c["clip"] for c in self.clips]

    def etiqueta(self, i: str) -> str:
        return self._por_id[i]["clase"]

    def caballo(self, i: str) -> str:
        return self._por_id[i]["id_caballo"]

    def etiquetas(self, ids) -> list[str]:
        return [self.etiqueta(i) for i in ids]

    def visual(self, i: str):
        if self._visual is None:
            import numpy as np
            from equinos import caracteristicas, rutas
            carpeta = rutas.carpeta("resultados")
            nombres = json.loads((carpeta / caracteristicas.ARCHIVO_HUELLA).read_text("utf-8"))["clips"]
            arr = np.load(carpeta / caracteristicas.ARCHIVO_CACHE, allow_pickle=True)["caracteristicas"]
            self._visual = {n: np.asarray(a, dtype=np.float32) for n, a in zip(nombres, arr)}
        return self._visual[i]

    def pose(self, ids=None) -> list[dict]:
        if self._pose is None:
            import csv
            from equinos import dataset, rutas
            with (rutas.carpeta("resultados") / dataset.ARCHIVO_POSE_SELECCIONADA).open(
                    encoding="utf-8", errors="replace") as f:
                self._pose = list(csv.DictReader(f))
        if ids is None:
            return self._pose
        conjunto = set(ids)
        return [f for f in self._pose if f["clip"] in conjunto]

    def meta(self, i: str) -> dict:
        if self._meta is None:
            from equinos import lineas_de_base
            m = lineas_de_base.metadatos_de_filmacion(self.clips)
            self._meta = {c["clip"]: dict(zip(lineas_de_base.COLUMNAS_METADATOS, fila))
                          for c, fila in zip(self.clips, m)}
        return self._meta[i]

    def es_limpio(self, i: str) -> bool:
        """1080×1920 y duración ≥ 5 s (el subconjunto sin rasgos de filmación delatores)."""
        from equinos import lineas_de_base as lb
        m = self.meta(i)
        return (m["ancho"], m["alto"]) == lb.RESOLUCION_MAYORITARIA and \
            m["duracion_seg"] >= lb.DURACION_MINIMA_SIN_DELATORES


# --- reserva ------------------------------------------------------------------------------------

def clase_dominante_por_caballo(clips: Sequence[dict]) -> dict[str, str]:
    """Clase con más clips de cada caballo; el empate lo gana la clase más rara del dataset."""
    global_ = Counter(c["clase"] for c in clips)
    por_caballo: dict[str, Counter] = defaultdict(Counter)
    for c in clips:
        por_caballo[c["id_caballo"]][c["clase"]] += 1
    return {h: min(cnt, key=lambda k: (-cnt[k], global_[k], k)) for h, cnt in por_caballo.items()}


def construir_reserva(clips: Sequence[dict], fraccion: float = FRACCION_RESERVA,
                      semilla: int = SEMILLA_RESERVA) -> dict:
    """Sortea la reserva estratificando por clase dominante (restos por mayor resto fraccional)."""
    import random

    dom = clase_dominante_por_caballo(clips)
    estratos: dict[str, list[str]] = defaultdict(list)
    for h in sorted(dom):
        estratos[dom[h]].append(h)
    total = round(fraccion * len(dom))
    cuota = {k: fraccion * len(v) for k, v in estratos.items()}
    base = {k: int(cuota[k]) for k in estratos}
    faltan = total - sum(base.values())
    for k in sorted(estratos, key=lambda k: (-(cuota[k] - base[k]), k))[:faltan]:
        base[k] += 1
    rng = random.Random(semilla)
    reserva = []
    for k in sorted(estratos):
        reserva += rng.sample(estratos[k], base[k])
    reserva = sorted(reserva)
    desarrollo = sorted(set(dom) - set(reserva))
    cl = {c["clip"]: c for c in clips}

    def comp(caballos):
        s = set(caballos)
        cc = [c for c in clips if c["id_caballo"] in s]
        return {"caballos": len(s), "clips": len(cc),
                "clips_por_clase": {k: sum(1 for c in cc if c["clase"] == k) for k in CLASES},
                "caballos_por_clase_dominante": {k: sum(1 for h in s if dom[h] == k) for k in CLASES}}
    del cl
    cuerpo = {"fraccion": fraccion, "semilla": semilla, "huella_dataset": huella_dataset(clips),
              "criterio": "15 % de los caballos, estratificado por clase dominante del caballo",
              "reserva": reserva, "desarrollo": desarrollo,
              "composicion": {"reserva": comp(reserva), "desarrollo": comp(desarrollo)}}
    cuerpo["huella"] = _huella_json({k: cuerpo[k] for k in ("fraccion", "semilla", "huella_dataset",
                                                            "reserva", "desarrollo")})
    return cuerpo


def guardar_reserva(reserva: dict, ruta: Path | None = None) -> Path:
    ruta = ruta or ARCHIVO_RESERVA
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(reserva, ensure_ascii=False, indent=1), encoding="utf-8")
    return ruta


def cargar_reserva(ruta: Path | None = None, clips: Sequence[dict] | None = None) -> dict:
    """Lee la reserva persistida y verifica su huella (y la del dataset, si se pasan los clips)."""
    ruta = ruta or ARCHIVO_RESERVA
    r = json.loads(Path(ruta).read_text("utf-8"))
    esperada = _huella_json({k: r[k] for k in ("fraccion", "semilla", "huella_dataset",
                                               "reserva", "desarrollo")})
    if r["huella"] != esperada:
        raise ValueError(f"{ruta} fue modificada a mano: su huella no coincide.")
    if set(r["reserva"]) & set(r["desarrollo"]):
        raise ValueError("Reserva y desarrollo se solapan.")
    if clips is not None and huella_dataset(clips) != r["huella_dataset"]:
        raise ValueError("El dataset cambió desde que se persistió la reserva: regenerarla es una "
                         "decisión que hay que tomar y registrar, no un efecto colateral.")
    return r


def clips_de_desarrollo(clips: Sequence[dict], reserva: dict) -> list[dict]:
    fuera = set(reserva["reserva"])
    return [c for c in clips if c["id_caballo"] not in fuera]


# --- particiones --------------------------------------------------------------------------------

def construir_particiones(clips: Sequence[dict], reserva: dict, repeticiones: int = REPETICIONES,
                          folds: int = FOLDS, semillas: Sequence[int] = SEMILLAS_PARTICION) -> dict:
    """R repeticiones de `folds` folds por caballo sobre el desarrollo. Guarda caballo→fold."""
    import numpy as np
    from sklearn.model_selection import StratifiedGroupKFold

    dev = sorted(clips_de_desarrollo(clips, reserva), key=lambda c: c["clip"])
    y = np.array([c["clase"] for c in dev])
    g = np.array([c["id_caballo"] for c in dev])
    reps = []
    for r in range(repeticiones):
        sgkf = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=semillas[r])
        fold_de_caballo = {}
        for k, (_, te) in enumerate(sgkf.split(np.zeros(len(y)), y, g)):
            for h in set(g[te].tolist()):
                fold_de_caballo[h] = k
        reps.append({"repeticion": r, "semilla": int(semillas[r]),
                     "fold_de_caballo": dict(sorted(fold_de_caballo.items()))})
    cuerpo = {"R": repeticiones, "K": folds, "huella_dataset": huella_dataset(clips),
              "huella_reserva": reserva["huella"], "repeticiones": reps}
    cuerpo["huella"] = _huella_json({k: cuerpo[k] for k in ("R", "K", "huella_dataset",
                                                            "huella_reserva", "repeticiones")})
    return cuerpo


def guardar_particiones(p: dict, ruta: Path = ARCHIVO_PARTICIONES) -> Path:
    Path(ruta).write_text(json.dumps(p, ensure_ascii=False, indent=1), encoding="utf-8")
    return Path(ruta)


def cargar_particiones(ruta: Path = ARCHIVO_PARTICIONES) -> dict:
    p = json.loads(Path(ruta).read_text("utf-8"))
    esperada = _huella_json({k: p[k] for k in ("R", "K", "huella_dataset", "huella_reserva",
                                               "repeticiones")})
    if p["huella"] != esperada:
        raise ValueError(f"{ruta} fue modificado a mano: su huella no coincide.")
    return p


def folds_de(clips: Sequence[dict], particiones: dict):
    """Genera `(repeticion, fold, ids_train, ids_test)` para las R×K evaluaciones."""
    por_caballo = defaultdict(list)
    for c in sorted(clips, key=lambda c: c["clip"]):
        por_caballo[c["id_caballo"]].append(c["clip"])
    for rep in particiones["repeticiones"]:
        fdc = rep["fold_de_caballo"]
        usados = {h for h in por_caballo if h in fdc}
        for k in range(particiones["K"]):
            tr = [i for h in usados if fdc[h] != k for i in por_caballo[h]]
            te = [i for h in usados if fdc[h] == k for i in por_caballo[h]]
            yield rep["repeticion"], k, sorted(tr), sorted(te)


def verificar_particiones(clips: Sequence[dict], particiones: dict, reserva: dict) -> dict:
    """Aserciones: sin cruces de caballo, sin reserva, cobertura total, clases en cada test."""
    cab = {c["clip"]: c["id_caballo"] for c in clips}
    cls = {c["clip"]: c["clase"] for c in clips}
    res = set(reserva["reserva"])
    dev = {c["clip"] for c in clips_de_desarrollo(clips, reserva)}
    vacias = []
    for r, k, tr, te in folds_de(clips_de_desarrollo(clips, reserva), particiones):
        assert not ({cab[i] for i in tr} & {cab[i] for i in te}), f"cruce de caballo en ({r},{k})"
        assert not ({cab[i] for i in tr + te} & res), f"la reserva aparece en ({r},{k})"
        assert set(tr) | set(te) == dev and not set(tr) & set(te), f"cobertura en ({r},{k})"
        faltan = [c for c in CLASES if not any(cls[i] == c for i in te)]
        if faltan:
            vacias.append((r, k, faltan))
    return {"evaluaciones": particiones["R"] * particiones["K"], "folds_con_clase_ausente": vacias}


# --- candidato ----------------------------------------------------------------------------------

class Candidato(Protocol):
    """Contrato de un modelo. Todo se expresa en ids de clip; los datos vienen de `Datos`.

    `ajustar` recibe SÓLO ids de entrenamiento (el candidato puede partirlos como quiera para su
    validación interna, siempre por caballo). `predecir` devuelve una clase por id, en el mismo
    orden. Se instancia una vez por evaluación: no debe arrastrar estado entre folds.
    """

    nombre: str

    def ajustar(self, ids_train: list[str], datos: Datos) -> None: ...

    def predecir(self, ids: list[str], datos: Datos) -> list[str]: ...


# --- métricas -----------------------------------------------------------------------------------

def _metricas(y, p) -> dict:
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, labels=list(CLASES), zero_division=0)
    return {"f1_macro": float(f1.mean()), "accuracy": float(accuracy_score(y, p)),
            "precision": dict(zip(CLASES, map(float, pr))), "recall": dict(zip(CLASES, map(float, rc))),
            "f1": dict(zip(CLASES, map(float, f1))),
            "matriz": confusion_matrix(y, p, labels=list(CLASES)).tolist()}


def f1_macro(y, p, clases=CLASES) -> float:
    """F1 macro (average='macro', las tres clases con peso igual, 0 si no se predice una clase)."""
    from sklearn.metrics import f1_score
    return float(f1_score(y, p, labels=list(clases), average="macro", zero_division=0))


def _ms(valores) -> dict:
    v = [x for x in valores if x == x]
    return {"media": statistics.fmean(v) if v else float("nan"),
            "desvio": statistics.stdev(v) if len(v) > 1 else 0.0, "n": len(v)}


# --- evaluador ----------------------------------------------------------------------------------

def evaluar(fabrica: Callable[[], Candidato], nombre: str, descripcion: str = "",
            datos: Datos | None = None, guardar: bool = True, verbose: bool = True,
            carpeta_resultados: Path = CARPETA_RESULTADOS) -> dict:
    """Mide un candidato sobre las R×K evaluaciones de desarrollo y escribe el JSON."""
    datos = datos or Datos()
    reserva = cargar_reserva(clips=datos.clips)
    part = cargar_particiones()
    if part["huella_reserva"] != reserva["huella"] or part["huella_dataset"] != datos.huella:
        raise ValueError("Las particiones no corresponden a la reserva / dataset vigentes.")
    dev = clips_de_desarrollo(datos.clips, reserva)
    verificar_particiones(datos.clips, part, reserva)

    evaluaciones, oof = [], defaultdict(dict)
    for r, k, tr, te in folds_de(dev, part):
        cand = fabrica()
        cand.ajustar(list(tr), datos)
        p_te = list(cand.predecir(list(te), datos))
        p_tr = list(cand.predecir(list(tr), datos))
        if len(p_te) != len(te) or len(p_tr) != len(tr):
            raise ValueError("`predecir` debe devolver una clase por id.")
        m = _metricas(datos.etiquetas(te), p_te)
        m.update(repeticion=r, fold=k, n_train=len(tr), n_test=len(te),
                 f1_macro_train=f1_macro(datos.etiquetas(tr), p_tr))
        m["gap_f1"] = m["f1_macro_train"] - m["f1_macro"]
        evaluaciones.append(m)
        oof[r].update(dict(zip(te, p_te)))
        if verbose:
            print(f"  [{nombre}] rep {r} fold {k}: F1 {m['f1_macro']:.3f}  gap {m['gap_f1']:+.3f}",
                  flush=True)

    limpio_ids = {c["clip"] for c in dev if datos.es_limpio(c["clip"])}
    por_rep = []
    for r in range(part["R"]):
        ev = [e for e in evaluaciones if e["repeticion"] == r]
        ids_r = sorted(oof[r])
        y_r = datos.etiquetas(ids_r)
        p_r = [oof[r][i] for i in ids_r]
        lim = [i for i in ids_r if i in limpio_ids]
        por_rep.append({
            "repeticion": r, "semilla": part["repeticiones"][r]["semilla"],
            "f1_macro_folds": _ms([e["f1_macro"] for e in ev]),
            "accuracy_folds": _ms([e["accuracy"] for e in ev]),
            "f1_macro_oof": f1_macro(y_r, p_r),
            "f1_macro_oof_limpio": f1_macro(datos.etiquetas(lim), [oof[r][i] for i in lim])})
    conf = [[sum(e["matriz"][a][b] for e in evaluaciones) for b in range(3)] for a in range(3)]
    res = {
        "esquema": ESQUEMA, "candidato": nombre, "descripcion": descripcion,
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "huella_dataset": datos.huella, "huella_reserva": reserva["huella"],
        "huella_particiones": part["huella"], "R": part["R"], "K": part["K"],
        "uso_reserva": False, "clases": list(CLASES),
        "global": {
            "f1_macro": _ms([e["f1_macro"] for e in evaluaciones]),
            "accuracy": _ms([e["accuracy"] for e in evaluaciones]),
            "por_clase": {c: {"precision": _ms([e["precision"][c] for e in evaluaciones]),
                              "recall": _ms([e["recall"][c] for e in evaluaciones]),
                              "f1": _ms([e["f1"][c] for e in evaluaciones])} for c in CLASES},
            "f1_macro_train": _ms([e["f1_macro_train"] for e in evaluaciones]),
            "gap_train_test_f1": _ms([e["gap_f1"] for e in evaluaciones]),
            "matriz_confusion_suma": conf},
        "por_particion": por_rep,
        "limpio": {
            "criterio": "ancho 1080, alto 1920, duración >= 5 s",
            "clips": len(limpio_ids),
            "clips_por_clase": {c: sum(1 for i in limpio_ids if datos.etiqueta(i) == c) for c in CLASES},
            "f1_macro_oof_por_repeticion": _ms([p["f1_macro_oof_limpio"] for p in por_rep])},
        "evaluaciones": evaluaciones,
        "predicciones_oof": {str(r): oof[r] for r in sorted(oof)},
    }
    validar_esquema(res)
    if guardar:
        carpeta_resultados.mkdir(parents=True, exist_ok=True)
        (carpeta_resultados / f"{nombre}.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    return res


_CLAVES = ("esquema", "candidato", "descripcion", "fecha", "huella_dataset", "huella_reserva",
           "huella_particiones", "R", "K", "uso_reserva", "clases", "global", "por_particion",
           "limpio", "evaluaciones", "predicciones_oof")


def validar_esquema(res: dict) -> None:
    faltan = [k for k in _CLAVES if k not in res]
    if faltan or res["esquema"] != ESQUEMA:
        raise ValueError(f"Resultado fuera de esquema: faltan {faltan}")
    if len(res["evaluaciones"]) != res["R"] * res["K"]:
        raise ValueError("Cantidad de evaluaciones distinta de R×K.")


def cargar_resultado(nombre: str, carpeta: Path = CARPETA_RESULTADOS) -> dict:
    res = json.loads((Path(carpeta) / f"{nombre}.json").read_text("utf-8"))
    validar_esquema(res)
    return res


# --- comparación pareada ------------------------------------------------------------------------

def comparar(a: dict, b: dict, datos: Datos | None = None, subconjunto: str = "todo",
             replicas: int = REPLICAS_BOOTSTRAP, semilla: int = SEMILLA_BOOTSTRAP) -> dict:
    """IC95 bootstrap pareado por caballo de `F1(a) − F1(b)` (media sobre las R repeticiones).

    Cada réplica sortea caballos de desarrollo con reposición, arrastra todos sus clips y calcula,
    para cada candidato y cada repetición, el F1 macro sobre las predicciones fuera de muestra;
    luego promedia las R. Misma réplica para los dos: la diferencia es pareada. `subconjunto`:
    "todo" o "limpio" (1080×1920 y ≥ 5 s).
    """
    import numpy as np

    if a["huella_particiones"] != b["huella_particiones"]:
        raise ValueError("Los candidatos se midieron sobre particiones distintas.")
    datos = datos or Datos()
    reserva = cargar_reserva(clips=datos.clips)
    dev = clips_de_desarrollo(datos.clips, reserva)
    if subconjunto == "limpio":
        dev = [c for c in dev if datos.es_limpio(c["clip"])]
    elif subconjunto != "todo":
        raise ValueError("subconjunto debe ser 'todo' o 'limpio'")
    ids = sorted(c["clip"] for c in dev)
    cab = np.array([datos.caballo(i) for i in ids])
    y = np.array([CLASES.index(datos.etiqueta(i)) for i in ids])
    R = a["R"]
    pa = np.array([[CLASES.index(a["predicciones_oof"][str(r)][i]) for i in ids] for r in range(R)])
    pb = np.array([[CLASES.index(b["predicciones_oof"][str(r)][i]) for i in ids] for r in range(R)])
    unicos = np.unique(cab)
    de = {h: np.where(cab == h)[0] for h in unicos}

    def f1s(pred, idx):
        out = []
        for r in range(R):
            out.append(f1_macro([CLASES[j] for j in y[idx]], [CLASES[j] for j in pred[r][idx]]))
        return float(np.mean(out))

    rng = np.random.default_rng(semilla)
    fa, fb = [], []
    for _ in range(replicas):
        idx = np.concatenate([de[h] for h in rng.choice(unicos, len(unicos))])
        fa.append(f1s(pa, idx)); fb.append(f1s(pb, idx))
    fa, fb = np.array(fa), np.array(fb)
    d = fa - fb
    todo = np.arange(len(ids))
    ic = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]  # noqa: E731
    return {"a": a["candidato"], "b": b["candidato"], "subconjunto": subconjunto,
            "clips": len(ids), "caballos": int(len(unicos)), "replicas": replicas, "semilla": semilla,
            "f1_a": f1s(pa, todo), "f1_b": f1s(pb, todo),
            "f1_a_ic95": ic(fa), "f1_b_ic95": ic(fb),
            "diferencia_a_menos_b": float(f1s(pa, todo) - f1s(pb, todo)),
            "diferencia_ic95": ic(d), "incluye_cero": bool(np.percentile(d, 2.5) <= 0 <= np.percentile(d, 97.5))}


# --- reserva: la única puerta -------------------------------------------------------------------

def registrar_uso_reserva(candidato: str, motivo: str, ruta: Path = ARCHIVO_USO_RESERVA) -> None:
    with Path(ruta).open("a", encoding="utf-8") as f:
        f.write(json.dumps({"fecha": datetime.now().isoformat(timespec="seconds"),
                            "candidato": candidato, "motivo": motivo}, ensure_ascii=False) + "\n")


def usos_de_reserva(ruta: Path = ARCHIVO_USO_RESERVA) -> list[dict]:
    ruta = Path(ruta)
    if not ruta.exists():
        return []
    return [json.loads(l) for l in ruta.read_text("utf-8").splitlines() if l.strip()]


def evaluar_en_reserva(fabrica: Callable[[], Candidato], nombre: str, *,
                       uso_autorizado_por_sergio: bool = False, motivo: str = "",
                       datos: Datos | None = None, ruta_log: Path = ARCHIVO_USO_RESERVA,
                       carpeta_resultados: Path = CARPETA_RESULTADOS) -> dict:
    """Ajusta el candidato con TODO el desarrollo y lo mide una vez sobre la reserva.

    Exige `uso_autorizado_por_sergio=True` y un `motivo`; cada llamada que llega a medir queda en
    `reserva_uso.log`. Si el log ya registra un uso, se niega: la reserva se usa UNA sola vez, con
    el ganador (ver PROTOCOLO.md). No hay parámetro para saltear esa negativa.
    """
    if not uso_autorizado_por_sergio:
        raise PermissionError("La reserva no se toca sin uso_autorizado_por_sergio=True.")
    if not motivo.strip():
        raise ValueError("Hace falta un `motivo` para registrar el uso de la reserva.")
    previos = usos_de_reserva(ruta_log)
    if previos:
        raise PermissionError(f"La reserva ya se usó ({previos[0]['fecha']}, {previos[0]['candidato']}): "
                              "es de un solo uso.")
    datos = datos or Datos()
    reserva = cargar_reserva(clips=datos.clips)
    registrar_uso_reserva(nombre, motivo, ruta_log)
    dev = [c["clip"] for c in clips_de_desarrollo(datos.clips, reserva)]
    res_ids = sorted(c["clip"] for c in datos.clips if c["id_caballo"] in set(reserva["reserva"]))
    cand = fabrica()
    cand.ajustar(dev, datos)
    pred = list(cand.predecir(res_ids, datos))
    m = _metricas(datos.etiquetas(res_ids), pred)
    lim = [i for i, p in zip(res_ids, pred) if datos.es_limpio(i)]
    out = {"esquema": ESQUEMA + "-reserva", "candidato": nombre, "uso_reserva": True,
           "fecha": datetime.now().isoformat(timespec="seconds"), "huella_reserva": reserva["huella"],
           "caballos": len(reserva["reserva"]), "clips": len(res_ids), "metricas": m,
           "predicciones": dict(zip(res_ids, pred)),
           "limpio_f1_macro": f1_macro(datos.etiquetas(lim), [p for i, p in zip(res_ids, pred)
                                                              if datos.es_limpio(i)]) if lim else None}
    carpeta_resultados.mkdir(parents=True, exist_ok=True)
    (carpeta_resultados / f"{nombre}__reserva.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


# --- regenerar los archivos versionados ---------------------------------------------------------

def inicializar(forzar: bool = False) -> dict:
    """Crea `reserva_caballos.json` y `particiones.json` si no existen. No pisa nada sin `forzar`."""
    datos = Datos()
    if ARCHIVO_RESERVA.exists() and not forzar:
        reserva = cargar_reserva(clips=datos.clips)
    else:
        if usos_de_reserva():
            raise PermissionError("La reserva ya se usó: no se regenera.")
        reserva = construir_reserva(datos.clips)
        guardar_reserva(reserva)
    if ARCHIVO_PARTICIONES.exists() and not forzar:
        part = cargar_particiones()
    else:
        part = construir_particiones(datos.clips, reserva)
        guardar_particiones(part)
    info = verificar_particiones(datos.clips, part, reserva)
    return {"reserva": reserva["composicion"], "particiones": info}


if __name__ == "__main__":
    print(json.dumps(inicializar(), ensure_ascii=False, indent=1))
