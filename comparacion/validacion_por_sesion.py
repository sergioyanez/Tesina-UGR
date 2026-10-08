"""Validación dejando afuera una sesión de filmación (leave-session-out). Análisis complementario.

Declarado en `comparacion/PROTOCOLO.md` (enmienda del 08/10/2026). NO cambia la elección del ganador
ni toca la reserva: sólo usa los 291 clips de los 113 caballos de desarrollo.

Partición (fijada antes de correr):

  - Para cada sesión de filmación `s` con clips de desarrollo: test = clips de `s`; train = clips de
    desarrollo de las OTRAS sesiones, EXCLUYENDO los de cualquier caballo que aparezca en `s`. Así no
    hay fuga ni por sesión ni por caballo. Se registra cuántos clips de train se pierden por esa
    exclusión.
  - Si el train queda sin alguna de las tres clases, la sesión se saltea (y queda registrada).
  - Cada clip de desarrollo se predice a lo sumo una vez (en la sesión a la que pertenece).

Métrica principal: F1 macro AGRUPADO sobre todas las predicciones fuera de muestra (muchas sesiones
tienen pocos clips o una sola clase, así que un F1 por sesión no es estable). IC95 bootstrap por
caballo (2 000 réplicas, semilla 7), también para la diferencia pareada fusión − otro.

Se reutilizan `Datos`, `f1_macro`, la reserva y las constantes de `equinos.comparacion` sin modificarlas.
Uso: TESINA_EQUINOS_RAIZ=<datos> python comparacion/validacion_por_sesion.py [--plan]
"""
from __future__ import annotations

import csv
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

RAIZ_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ_REPO))
sys.path.insert(0, str(RAIZ_REPO / "comparacion"))

from equinos import comparacion  # noqa: E402
from equinos.comparacion import CLASES, f1_macro  # noqa: E402

ARCHIVO_SESIONES = RAIZ_REPO / "comparacion" / "sesiones_de_filmacion.csv"
ARCHIVO_RESULTADO = comparacion.CARPETA_RESULTADOS / "validacion_por_sesion.json"
ESQUEMA = "validacion-por-sesion-v1"


# --- mapeo clip -> video -> sesión --------------------------------------------------------------

def sesion_por_video(archivo: Path = ARCHIVO_SESIONES) -> dict[tuple[str, str], str]:
    """(clase, nombre del archivo de video) -> `sesion_confirmada`.

    La tabla de sesiones usa la ruta relativa (`Clase/[subcarpeta/]archivo`); el dataset registra
    `video_origen` (sólo el archivo) y la clase. La clave (clase, archivo) es única en los 274 videos.
    """
    mapa: dict[tuple[str, str], set[str]] = defaultdict(set)
    with Path(archivo).open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            mapa[(r["clase"], r["ruta"].split("/")[-1])].add(r["sesion_confirmada"])
    ambiguas = [k for k, v in mapa.items() if len(v) != 1 or "" in v]
    if ambiguas:
        raise ValueError(f"Videos sin sesión única confirmada: {ambiguas[:3]}")
    return {k: next(iter(v)) for k, v in mapa.items()}


def sesion_por_clip(clips, archivo_sesiones: Path = ARCHIVO_SESIONES) -> dict[str, str]:
    """clip -> sesión, vía `video_origen` de `dataset_frames_alineados_v2.csv` (lo que usa `particion`)."""
    from equinos import particion
    video = {}
    for fila in particion._leer_frames():
        video[fila["clip"]] = (fila["clase"], fila["video_origen"])
    mapa = sesion_por_video(archivo_sesiones)
    out = {}
    for c in clips:
        out[c["clip"]] = mapa[video[c["clip"]]]
    return out


# --- plan de la partición -----------------------------------------------------------------------

def plan_de_particion(ids, sesion: dict, caballo: dict, clase: dict) -> dict:
    """Plan leave-session-out sobre `ids` (clips de desarrollo). Función pura, sin datos pesados.

    Devuelve `{"evaluadas": [{sesion, test, train, perdidos, caballos}], "salteadas": [{sesion,
    motivo, ...}]}`; `perdidos` son los clips de otras sesiones que se excluyen del train por ser de
    un caballo presente en la sesión de test.
    """
    faltan = [i for i in ids if not sesion.get(i)]
    assert not faltan, f"clips de desarrollo sin sesión: {faltan[:3]}"
    por_sesion: dict[str, list[str]] = defaultdict(list)
    for i in sorted(ids):
        por_sesion[sesion[i]].append(i)
    evaluadas, salteadas = [], []
    for s in sorted(por_sesion):
        test = por_sesion[s]
        caballos_s = {caballo[i] for i in test}
        otros = [i for i in sorted(ids) if sesion[i] != s]
        perdidos = [i for i in otros if caballo[i] in caballos_s]
        train = [i for i in otros if caballo[i] not in caballos_s]
        faltantes = [k for k in CLASES if not any(clase[i] == k for i in train)]
        reg = {"sesion": s, "test": test, "train": train, "perdidos": perdidos,
               "caballos": sorted(caballos_s)}
        if faltantes:
            reg["motivo"] = f"el train queda sin la clase {', '.join(faltantes)}"
            salteadas.append(reg)
        else:
            evaluadas.append(reg)
    return {"evaluadas": evaluadas, "salteadas": salteadas}


def resumen_plan(plan: dict, clase: dict) -> dict:
    def fila(r):
        return {"sesion": r["sesion"], "n_test": len(r["test"]), "n_train": len(r["train"]),
                "clips_train_perdidos": len(r["perdidos"]), "caballos_test": len(r["caballos"]),
                "test_por_clase": {k: sum(1 for i in r["test"] if clase[i] == k) for k in CLASES},
                "motivo_salteo": r.get("motivo")}
    return {"evaluadas": [fila(r) for r in plan["evaluadas"]],
            "salteadas": [fila(r) for r in plan["salteadas"]],
            "clips_predichos": sum(len(r["test"]) for r in plan["evaluadas"]),
            "clips_sin_prediccion": sum(len(r["test"]) for r in plan["salteadas"]),
            "clips_train_perdidos_total": sum(len(r["perdidos"]) for r in plan["evaluadas"]),
            "clips_train_perdidos_media": (sum(len(r["perdidos"]) for r in plan["evaluadas"])
                                           / max(1, len(plan["evaluadas"])))}


def verificar_plan(plan: dict, ids, sesion: dict, caballo: dict, reserva: dict) -> None:
    """Aserciones: sin cruce de caballo ni de sesión, sin reserva, cada clip a lo sumo una vez."""
    res = set(reserva["reserva"])
    visto = Counter()
    for r in plan["evaluadas"]:
        te, tr = set(r["test"]), set(r["train"])
        assert not te & tr, f"clips repetidos en {r['sesion']}"
        assert not {caballo[i] for i in te} & {caballo[i] for i in tr}, f"cruce de caballo en {r['sesion']}"
        assert not {sesion[i] for i in te} & {sesion[i] for i in tr}, f"cruce de sesión en {r['sesion']}"
        assert not {caballo[i] for i in te | tr} & res, f"la reserva aparece en {r['sesion']}"
        assert te | tr | set(r["perdidos"]) == set(ids), f"cobertura en {r['sesion']}"
        visto.update(te)
    assert all(n == 1 for n in visto.values())
    salteados = {i for r in plan["salteadas"] for i in r["test"]}
    assert set(visto) | salteados == set(ids) and not set(visto) & salteados


# --- métricas e IC ------------------------------------------------------------------------------

def _f1_de_indices(y, p, idx):
    """F1 macro (zero_division=0) con numpy puro; equivale a `comparacion.f1_macro`."""
    import numpy as np
    m = np.bincount(y[idx] * 3 + p[idx], minlength=9).reshape(3, 3)
    tp = np.diag(m).astype(float)
    pred, real = m.sum(axis=0), m.sum(axis=1)
    den = pred + real
    f1 = np.divide(2 * tp, den, out=np.zeros(3), where=den > 0)
    return float(f1.mean())


def metricas_agrupadas(y, p) -> dict:
    return comparacion._metricas(y, p)


def bootstrap_por_caballo(ids, caballo, clase, predicciones: dict[str, dict], limpio: set,
                          replicas=comparacion.REPLICAS_BOOTSTRAP, semilla=comparacion.SEMILLA_BOOTSTRAP,
                          referencia="fusion_tardia_pose_visual") -> dict:
    """IC95 del F1 agrupado de cada candidato y de `referencia − otro`, remuestreando caballos.

    `predicciones[candidato][clip]`. Se calcula en todo lo predicho y en el subconjunto limpio.
    """
    import numpy as np

    out = {}
    for nombre_sub, sub in (("todo", None), ("limpio", limpio)):
        sel = sorted(i for i in ids if sub is None or i in sub)
        cab = np.array([caballo[i] for i in sel])
        y = np.array([CLASES.index(clase[i]) for i in sel])
        P = {n: np.array([CLASES.index(pr[i]) for i in sel]) for n, pr in predicciones.items()}
        unicos = np.unique(cab)
        de = {h: np.where(cab == h)[0] for h in unicos}
        rng = np.random.default_rng(semilla)
        F = {n: [] for n in P}
        for _ in range(replicas):
            idx = np.concatenate([de[h] for h in rng.choice(unicos, len(unicos))])
            for n in P:
                F[n].append(_f1_de_indices(y, P[n], idx))
        F = {n: np.array(v) for n, v in F.items()}
        todo = np.arange(len(sel))
        ic = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]  # noqa: E731
        punto = {n: _f1_de_indices(y, P[n], todo) for n in P}
        for n in P:
            assert abs(punto[n] - f1_macro([CLASES[j] for j in y], [CLASES[j] for j in P[n]])) < 1e-12
        sub_out = {"clips": len(sel), "caballos": int(len(unicos)),
                   "f1_agrupado": {n: {"valor": punto[n], "ic95": ic(F[n])} for n in P},
                   "diferencia_vs_" + referencia: {}}
        for n in P:
            if n == referencia or referencia not in P:
                continue
            d = F[referencia] - F[n]
            sub_out["diferencia_vs_" + referencia][n] = {
                "diferencia": punto[referencia] - punto[n], "ic95": ic(d),
                "incluye_cero": bool(np.percentile(d, 2.5) <= 0 <= np.percentile(d, 97.5))}
        out[nombre_sub] = sub_out
    return out


# --- ejecución ----------------------------------------------------------------------------------

def _fabricas():
    from candidato_fusion import FusionPoseVisual
    from candidato_mil import MILMultimodalCorregido
    from candidato_rf_pose import RFPoseSimba
    from correr_baseline import LogRegMediaFrames
    return [LogRegMediaFrames, RFPoseSimba, MILMultimodalCorregido, FusionPoseVisual]


def correr(fabricas=None, verbose=True, guardar=True) -> dict:
    datos = comparacion.Datos()
    reserva = comparacion.cargar_reserva(clips=datos.clips)
    dev = comparacion.clips_de_desarrollo(datos.clips, reserva)
    ids = sorted(c["clip"] for c in dev)
    ses = sesion_por_clip(dev)
    cab = {c["clip"]: c["id_caballo"] for c in dev}
    cls = {c["clip"]: c["clase"] for c in dev}
    plan = plan_de_particion(ids, ses, cab, cls)
    verificar_plan(plan, ids, ses, cab, reserva)
    resumen = resumen_plan(plan, cls)
    limpio = {i for i in ids if datos.es_limpio(i)}

    candidatos = {}
    for fabrica in (fabricas or _fabricas()):
        nombre = fabrica.nombre
        t0 = time.time()
        pred, por_sesion = {}, []
        for r in plan["evaluadas"]:
            cand = fabrica()
            cand.ajustar(list(r["train"]), datos)
            p = list(cand.predecir(list(r["test"]), datos))
            if len(p) != len(r["test"]):
                raise ValueError("`predecir` debe devolver una clase por id.")
            pred.update(dict(zip(r["test"], p)))
            acc = sum(a == cls[i] for a, i in zip(p, r["test"])) / len(p)
            por_sesion.append({"sesion": r["sesion"], "n_test": len(p), "accuracy": acc})
            if verbose:
                print(f"  [{nombre}] {r['sesion']}: n_test {len(p)} acc {acc:.2f} "
                      f"({time.time() - t0:.0f} s)", flush=True)
        orden = sorted(pred)
        y = [cls[i] for i in orden]
        p = [pred[i] for i in orden]
        lim = [i for i in orden if i in limpio]
        m = metricas_agrupadas(y, p)
        candidatos[nombre] = {
            "metricas": m,
            "f1_macro_limpio": f1_macro([cls[i] for i in lim], [pred[i] for i in lim]),
            "clips_limpios_predichos": len(lim),
            "por_sesion": por_sesion, "predicciones": pred,
            "segundos": round(time.time() - t0)}
        if verbose:
            print(f"[{nombre}] F1 agrupado {m['f1_macro']:.3f} | acc {m['accuracy']:.3f} "
                  f"| limpio {candidatos[nombre]['f1_macro_limpio']:.3f}", flush=True)

    preds = {n: c["predicciones"] for n, c in candidatos.items()}
    predichos = sorted(next(iter(preds.values())))
    ic = bootstrap_por_caballo(predichos, cab, cls, preds, limpio)
    res = {"esquema": ESQUEMA, "fecha": datetime.now().isoformat(timespec="seconds"),
           "huella_dataset": datos.huella, "huella_reserva": reserva["huella"],
           "uso_reserva": False, "clases": list(CLASES),
           "diseno": "leave-session-out sobre desarrollo; train excluye los caballos presentes en la sesión de test",
           "replicas_bootstrap": comparacion.REPLICAS_BOOTSTRAP, "semilla_bootstrap": comparacion.SEMILLA_BOOTSTRAP,
           "plan": resumen, "candidatos": candidatos, "intervalos": ic}
    if guardar:
        ARCHIVO_RESULTADO.parent.mkdir(parents=True, exist_ok=True)
        ARCHIVO_RESULTADO.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    return res


if __name__ == "__main__":
    if "--plan" in sys.argv:
        datos = comparacion.Datos()
        reserva = comparacion.cargar_reserva(clips=datos.clips)
        dev = comparacion.clips_de_desarrollo(datos.clips, reserva)
        ids = sorted(c["clip"] for c in dev)
        ses = sesion_por_clip(dev)
        cab = {c["clip"]: c["id_caballo"] for c in dev}
        cls = {c["clip"]: c["clase"] for c in dev}
        pl = plan_de_particion(ids, ses, cab, cls)
        verificar_plan(pl, ids, ses, cab, reserva)
        print(json.dumps(resumen_plan(pl, cls), ensure_ascii=False, indent=1))
    else:
        correr()
