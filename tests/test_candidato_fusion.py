"""Fusión pose + visual: elección del peso y contrato del candidato (parte sintética).

Uso: python3 tests/test_candidato_fusion.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "comparacion"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

import candidato_fusion as cf  # noqa: E402

CLASES = np.array(["Disgustado", "Interesado", "Sereno"])
y = np.tile(CLASES, 10)
bueno = np.eye(3)[np.tile([0, 1, 2], 10)] * 0.8 + 0.1
malo = np.eye(3)[np.tile([1, 2, 0], 10)] * 0.8 + 0.1


def test_peso_extremos():
    assert cf.elegir_peso(bueno, malo, y, CLASES)[0] >= 0.75
    assert cf.elegir_peso(malo, bueno, y, CLASES)[0] <= 0.25


def test_empate_prefiere_cercano_a_medio():
    w, _ = cf.elegir_peso(bueno, bueno, y, CLASES)
    assert w == 0.5


def test_contrato_y_pesos_validos():
    assert set(cf.PESOS) == {0.0, 0.25, 0.5, 0.75, 1.0}
    assert hasattr(cf.FusionPoseVisual, "ajustar") and hasattr(cf.FusionPoseVisual, "predecir")


def test_ajustar_predecir_sintetico():
    rng = np.random.default_rng(0)
    ids = [f"c{i}" for i in range(90)]
    clase = {i: CLASES[k % 3] for k, i in enumerate(ids)}
    cab = {i: f"h{k // 3}" for k, i in enumerate(ids)}
    vis = {i: rng.normal(size=(16, 8)) + (CLASES == clase[i]).argmax() for i in ids}
    pose = {i: rng.normal(size=5) + 2 * (CLASES == clase[i]).argmax() for i in ids}

    class D:
        etiquetas = staticmethod(lambda xs: [clase[x] for x in xs])
        caballo = staticmethod(lambda x: cab[x])
        visual = staticmethod(lambda x: vis[x])
    orig = cf.features_pose
    cf.features_pose = lambda xs, d: np.stack([pose[x] for x in xs])
    try:
        c = cf.FusionPoseVisual()
        c.ajustar(ids[:60], D)
        pred = c.predecir(ids[60:], D)
    finally:
        cf.features_pose = orig
    assert c.peso in cf.PESOS and len(pred) == 30
    assert np.mean(np.array(pred) == np.array([clase[i] for i in ids[60:]])) > 0.8


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f()
            print("OK", n)
