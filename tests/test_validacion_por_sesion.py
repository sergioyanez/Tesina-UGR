"""Validación por sesión de filmación: partición sin fuga y cobertura (parte sintética y con datos).

Uso: TESINA_EQUINOS_RAIZ=<datos> python3 tests/test_validacion_por_sesion.py
(sin datos reales sólo corre la parte sintética)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "comparacion"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import validacion_por_sesion as vps  # noqa: E402

# Caballo h1 aparece en s1 y s2; h4 sólo tiene Sereno y está sólo en s3.
SESION = {"a": "s1", "b": "s1", "c": "s2", "d": "s2", "e": "s2", "f": "s3", "g": "s3", "h": "s3"}
CABALLO = {"a": "h1", "b": "h2", "c": "h1", "d": "h3", "e": "h5", "f": "h4", "g": "h6", "h": "h7"}
CLASE = {"a": "Sereno", "b": "Interesado", "c": "Disgustado", "d": "Sereno", "e": "Interesado",
         "f": "Disgustado", "g": "Sereno", "h": "Interesado"}
IDS = sorted(SESION)


def test_sin_cruce_de_caballo_ni_sesion_y_perdidos():
    plan = vps.plan_de_particion(IDS, SESION, CABALLO, CLASE)
    por = {r["sesion"]: r for r in plan["evaluadas"]}
    assert "s1" in por and por["s1"]["test"] == ["a", "b"]
    assert por["s1"]["perdidos"] == ["c"]  # h1 está en s1: su clip de s2 sale del train
    assert "c" not in por["s1"]["train"]
    for r in plan["evaluadas"]:
        assert not {CABALLO[i] for i in r["test"]} & {CABALLO[i] for i in r["train"]}
        assert not {SESION[i] for i in r["test"]} & {SESION[i] for i in r["train"]}
    vps.verificar_plan(plan, IDS, SESION, CABALLO, {"reserva": []})


def test_cada_clip_se_predice_una_vez_salvo_salteadas():
    plan = vps.plan_de_particion(IDS, SESION, CABALLO, CLASE)
    vistos = [i for r in plan["evaluadas"] for i in r["test"]]
    saltados = [i for r in plan["salteadas"] for i in r["test"]]
    assert sorted(vistos + saltados) == IDS and len(vistos) == len(set(vistos))


def test_sesion_que_deja_el_train_sin_clase_se_saltea():
    ses = {"a": "s1", "b": "s2", "c": "s3"}
    cab = {"a": "h1", "b": "h2", "c": "h3"}
    cls = {"a": "Sereno", "b": "Interesado", "c": "Disgustado"}
    plan = vps.plan_de_particion(sorted(ses), ses, cab, cls)
    assert not plan["evaluadas"] and len(plan["salteadas"]) == 3
    assert all("sin la clase" in r["motivo"] for r in plan["salteadas"])


def test_clip_sin_sesion_falla():
    try:
        vps.plan_de_particion(IDS, {**SESION, "a": ""}, CABALLO, CLASE)
    except AssertionError:
        return
    raise AssertionError("debía fallar por clip sin sesión")


def test_reserva_en_el_plan_falla():
    plan = vps.plan_de_particion(IDS, SESION, CABALLO, CLASE)
    try:
        vps.verificar_plan(plan, IDS, SESION, CABALLO, {"reserva": ["h6"]})
    except AssertionError:
        return
    raise AssertionError("debía detectar la reserva")


def test_f1_numpy_equivale_a_sklearn():
    import numpy as np
    from equinos.comparacion import CLASES, f1_macro
    rng = np.random.default_rng(0)
    y, p = rng.integers(0, 3, 50), rng.integers(0, 3, 50)
    esperado = f1_macro([CLASES[i] for i in y], [CLASES[i] for i in p])
    assert abs(vps._f1_de_indices(y, p, np.arange(50)) - esperado) < 1e-12


def test_con_datos_reales():
    from equinos import comparacion as cmp
    datos = cmp.Datos()
    reserva = cmp.cargar_reserva(clips=datos.clips)
    dev = cmp.clips_de_desarrollo(datos.clips, reserva)
    ids = sorted(c["clip"] for c in dev)
    ses = vps.sesion_por_clip(dev)
    assert len(ids) == 291 and all(ses[i] for i in ids)
    cab = {c["clip"]: c["id_caballo"] for c in dev}
    cls = {c["clip"]: c["clase"] for c in dev}
    plan = vps.plan_de_particion(ids, ses, cab, cls)
    vps.verificar_plan(plan, ids, ses, cab, reserva)
    res = set(reserva["reserva"])
    for r in plan["evaluadas"]:
        assert not {cab[i] for i in r["test"] + r["train"]} & res
    vistos = [i for r in plan["evaluadas"] for i in r["test"]]
    assert len(vistos) == len(set(vistos))
    assert len(vistos) + sum(len(r["test"]) for r in plan["salteadas"]) == 291


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            if n == "test_con_datos_reales":
                import os
                if not os.environ.get("TESINA_EQUINOS_RAIZ"):
                    print("OMITIDO", n, "(sin TESINA_EQUINOS_RAIZ)")
                    continue
            f()
            print("OK", n)
