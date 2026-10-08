"""Carga de las cifras del informe «Estado actual y métricas».

Todas las cifras salen de los JSON de `comparacion/resultados/` (y de `comparacion/*.json`); este
módulo no calcula métricas nuevas: sólo lee, formatea y comprueba que los archivos sean coherentes
entre sí y con `comparacion/DECISION.md`.
"""
import json
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[3]
CMP = RAIZ / "comparacion"
RES = CMP / "resultados"

CANDIDATOS = ["fusion_tardia_pose_visual", "rf_pose_tipo_simba",
              "mil_multimodal_corregido", "baseline_logreg_media_frames"]
NOMBRE = {"fusion_tardia_pose_visual": "Fusión pose + visual",
          "rf_pose_tipo_simba": "RF de pose tipo SimBA",
          "mil_multimodal_corregido": "MIL multimodal corregido",
          "baseline_logreg_media_frames": "Baseline LogReg (media de frames)"}
CORTO = {"fusion_tardia_pose_visual": "Fusión\npose + visual",
         "rf_pose_tipo_simba": "RF de pose\ntipo SimBA",
         "mil_multimodal_corregido": "MIL\ncorregido",
         "baseline_logreg_media_frames": "Baseline\nLogReg"}
COLOR = {"fusion_tardia_pose_visual": "#1f4fa3", "rf_pose_tipo_simba": "#2a8c82",
         "mil_multimodal_corregido": "#c27c1a", "baseline_logreg_media_frames": "#8591a5"}
CLASES = ["Disgustado", "Interesado", "Sereno"]

MENOS = "−"


def n(x, d=3):
    """Número con coma decimal."""
    return f"{x:.{d}f}".replace(".", ",")


def sg(x, d=3):
    """Número con signo explícito (+ o −)."""
    return ("+" if x >= 0 else MENOS) + f"{abs(x):.{d}f}".replace(".", ",")


def pm(m, s, d=3):
    return f"{n(m, d)} ± {n(s, d)}"


def ic(par, d=3, signo=False):
    f = sg if signo else n
    return f"[{f(par[0], d)}; {f(par[1], d)}]"


def pct(x, d=0):
    return f"{100 * x:.{d}f}".replace(".", ",") + " %"


def _json(ruta):
    return json.loads(Path(ruta).read_text(encoding="utf-8"))


def cargar():
    D = {"cand": {c: _json(RES / f"{c}.json") for c in CANDIDATOS}}
    D["pareadas"] = _json(RES / "comparaciones_pareadas.json")
    D["reserva"] = _json(RES / "fusion_tardia_pose_visual__reserva.json")
    D["sesion"] = _json(RES / "validacion_por_sesion.json")
    D["particiones"] = _json(CMP / "particiones.json")
    D["reserva_caballos"] = _json(CMP / "reserva_caballos.json")
    D["epocas_mil"] = _json(RES / "mil_multimodal_corregido__epocas.json")
    D["decision"] = (CMP / "DECISION.md").read_text(encoding="utf-8")
    D["uso_reserva"] = (CMP / "reserva_uso.log").read_text(encoding="utf-8").strip().splitlines()

    # Coherencia de huellas entre todos los resultados.
    hp = {c["huella_particiones"] for c in D["cand"].values()}
    hr = {c["huella_reserva"] for c in D["cand"].values()}
    hd = {c["huella_dataset"] for c in D["cand"].values()}
    assert len(hp) == len(hr) == len(hd) == 1, "los candidatos no comparten huellas"
    D["huella_particiones"], D["huella_reserva"], D["huella_dataset"] = hp.pop(), hr.pop(), hd.pop()
    assert D["particiones"]["huella_reserva"] == D["huella_reserva"]
    assert D["particiones"]["huella_dataset"] == D["huella_dataset"]
    assert D["reserva"]["huella_reserva"] == D["huella_reserva"] == D["sesion"]["huella_reserva"]
    assert D["reserva_caballos"]["huella_dataset"] == D["huella_dataset"] == D["sesion"]["huella_dataset"]
    assert len(D["uso_reserva"]) == 1, "la reserva debe tener un único uso registrado"

    # Contraste con los números que DECISION.md declara en texto.
    dec = D["decision"]
    for c in CANDIDATOS:
        g = D["cand"][c]["global"]
        lim = D["cand"][c]["limpio"]["f1_macro_oof_por_repeticion"]
        fila = f"{pm(g['f1_macro']['media'], g['f1_macro']['desvio'])} | {pm(lim['media'], lim['desvio'])}"
        assert fila in dec, f"DECISION.md no coincide con el JSON de {c}: {fila}"
    r = D["reserva"]["metricas"]
    assert f"F1 macro **{n(r['f1_macro'])}**" in dec and f"Accuracy {n(r['accuracy'])}" in dec
    return D


def par(D, a, b, sub):
    """Entrada de comparaciones_pareadas para (a, b, subconjunto)."""
    for p in D["pareadas"]:
        if p["a"] == a and p["b"] == b and p["subconjunto"] == sub:
            return p
    raise KeyError((a, b, sub))


def f1_pooled_54(D, c, sub):
    """F1 fuera de muestra (media de las 5 repeticiones) con su IC95 bootstrap, del 5×4."""
    if c == CANDIDATOS[0]:
        p = par(D, c, CANDIDATOS[1], sub)
        return p["f1_a"], p["f1_a_ic95"]
    p = par(D, CANDIDATOS[0], c, sub)
    return p["f1_b"], p["f1_b_ic95"]


def f1_sesion(D, c, sub):
    x = D["sesion"]["intervalos"][sub]["f1_agrupado"][c]
    return x["valor"], x["ic95"]
