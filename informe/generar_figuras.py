"""Figuras del informe final (estilo APA 7): PDF vectorial + PNG 300 dpi en `informe/figuras/`.

Sin título dentro de la figura (va en el LaTeX), fuente serif, 10 pt, escala de grises con patrones,
coma decimal. Los datos salen de `comparacion/resultados/*.json` vía `_comun.py`.
Uso: python informe/generar_figuras.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import FuncFormatter

import _comun as c
from _comun import BASE, CLASES, FUS, MIL, RF, f1_pooled_54, f1_sesion, par

SALIDA = Path(__file__).resolve().parent / "figuras"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman", "STIXGeneral"],
    "mathtext.fontset": "stix", "font.size": 10, "axes.labelsize": 10, "xtick.labelsize": 10,
    "ytick.labelsize": 10, "legend.fontsize": 9.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
    "hatch.linewidth": 0.7,
})
ETIQ = {FUS: "Fusión\npose + visual", RF: "RF de pose\ntipo SimBA", MIL: "MIL\ncorregido", BASE: "Regresión logística\n(media de fotogramas)"}
ETIQ_LARGA = {FUS: "Fusión pose + visual", RF: "RF de pose tipo SimBA", MIL: "MIL multimodal corregido",
              BASE: "Regresión logística (línea de base)"}
# relleno gris + patrón: legibles en escala de grises
ESTILO = {FUS: ("#2b2b2b", None), RF: ("#6e6e6e", "///"), MIL: ("#a8a8a8", "xxx"), BASE: ("#e2e2e2", "...")}
CANDS = [FUS, RF, MIL, BASE]
COMA = FuncFormatter(lambda v, _: f"{v:.1f}".replace(".", ",").replace("-", "−"))


def coma(ax, eje="y", d=1):
    f = FuncFormatter(lambda v, _: f"{v:.{d}f}".replace(".", ",").replace("-", "−"))
    (ax.yaxis if eje == "y" else ax.xaxis).set_major_formatter(f)


def n3(v, d=3):
    return f"{v:.{d}f}".replace(".", ",")


def guardar(fig, nombre):
    SALIDA.mkdir(exist_ok=True)
    fig.savefig(SALIDA / f"{nombre}.pdf", bbox_inches="tight")
    fig.savefig(SALIDA / f"{nombre}.png", bbox_inches="tight", dpi=300, facecolor="white")
    plt.close(fig)


def barra(ax, x, h, ancho, cand, claro=False, **kw):
    fc, ht = ESTILO[cand]
    if claro:
        fc = "white"
    ax.bar(x, h, ancho, color=fc, hatch=ht, edgecolor="black", linewidth=0.8, **kw)


def fig_f1_candidatos(D):
    fig, ax = plt.subplots(figsize=(6.3, 3.6))
    w = 0.36
    for i, k in enumerate(CANDS):
        g = D["cand"][k]["global"]["f1_macro"]
        lim = D["cand"][k]["limpio"]["f1_macro_oof_por_repeticion"]
        for off, (m, s), claro in ((-w / 2 - 0.01, (g["media"], g["desvio"]), False),
                                   (w / 2 + 0.01, (lim["media"], lim["desvio"]), True)):
            barra(ax, i + off, m, w, k, claro)
            ax.errorbar(i + off, m, yerr=s, color="black", lw=1, capsize=3)
            ax.text(i + off, m + s + 0.015, n3(m), ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(4)); ax.set_xticklabels([ETIQ[k] for k in CANDS])
    ax.set_ylim(0, 1.05); coma(ax); ax.set_ylabel("F1 macro")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="#6e6e6e", edgecolor="black", label="Desarrollo completo (izquierda de cada par)"),
                       Patch(facecolor="white", edgecolor="black", label="Subconjunto limpio (derecha de cada par)")],
              loc="upper right", frameon=False, fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    fig.tight_layout(); guardar(fig, "f1_candidatos")


def fig_ic_pareados(D):
    rivales = [(FUS, RF), (FUS, MIL), (FUS, BASE)]
    series = [("Desarrollo 5×4 (todos los clips)", "o", True, "todo5"),
              ("Desarrollo 5×4 (subconjunto limpio)", "s", False, "limpio5"),
              ("Una sesión afuera (todos los clips)", "^", True, "sesion")]
    fig, ax = plt.subplots(figsize=(6.3, 3.4))
    sep = 0.24
    for i, (a, b) in enumerate(rivales):
        y0 = len(rivales) - 1 - i
        for k, (rot, mk, lleno, tipo) in enumerate(series):
            if tipo == "sesion":
                x = D["sesion"]["intervalos"]["todo"]["diferencia_vs_fusion_tardia_pose_visual"][b]
                d, (lo, hi) = x["diferencia"], x["ic95"]
            else:
                p = par(D, a, b, "todo" if tipo == "todo5" else "limpio")
                d, (lo, hi) = p["diferencia_a_menos_b"], p["diferencia_ic95"]
            y = y0 + (1 - k) * sep
            ax.plot([lo, hi], [y, y], color="black", lw=1.4)
            ax.plot([lo, lo], [y - .04, y + .04], color="black", lw=1.2); ax.plot([hi, hi], [y - .04, y + .04], color="black", lw=1.2)
            ax.plot(d, y, mk, color="black", mfc="black" if lleno else "white", mew=1.3, ms=6.5)
            ax.text(hi + 0.008, y, f"{d:+.3f} [{lo:+.3f}; {hi:+.3f}]".replace(".", ",").replace("-", "−"),
                    va="center", fontsize=8.5)
    ax.axvline(0, color="black", lw=1, ls="--")
    ax.set_yticks(range(3))
    ax.set_yticklabels([f"Fusión − {t}" for t in ("MIL corregido", "RF de pose", "")][:0] or
                       ["Fusión − línea\nde base", "Fusión − MIL\ncorregido", "Fusión − RF\nde pose"])
    ax.set_ylim(-0.5, 2.5); ax.set_xlim(-0.06, 0.55)
    coma(ax, "x", 2)
    ax.set_xlabel("Diferencia de F1 macro (IC95 bootstrap por caballo)")
    ax.legend(handles=[plt.Line2D([], [], color="black", marker=mk, mfc="black" if ll else "white", mew=1.3, ls="-", label=r_)
                       for r_, mk, ll, _ in series], loc="upper right", frameon=False, fontsize=8.5, bbox_to_anchor=(1.0, 1.02))
    fig.tight_layout(); guardar(fig, "ic_pareados")


def matriz(ax, M, cmax=None, fs=10):
    M = np.array(M)
    ax.imshow(M, cmap="Greys", vmin=0, vmax=cmax or M.max() * 1.25)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(int(M[i, j])), ha="center", va="center", fontsize=fs,
                    color="white" if M[i, j] > 0.55 * (cmax or M.max() * 1.25) else "black",
                    fontweight="bold" if i == j else "normal")
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(CLASES); ax.set_yticklabels(CLASES, rotation=90, va="center")
    ax.set_xlabel("Clase predicha"); ax.set_ylabel("Clase real")
    for s in ax.spines.values():
        s.set_visible(True); s.set_linewidth(0.8)
    ax.tick_params(length=0)


def fig_confusion_reserva(D):
    fig, ax = plt.subplots(figsize=(3.8, 3.4))
    matriz(ax, D["reserva"]["metricas"]["matriz"], cmax=40, fs=12)
    fig.tight_layout(); guardar(fig, "confusion_reserva")


def fig_confusion_desarrollo(D):
    fig, axes = plt.subplots(2, 2, figsize=(6.3, 6.1))
    mats = [np.array(D["cand"][k]["global"]["matriz_confusion_suma"]) for k in CANDS]
    cmax = max(m.max() for m in mats) * 1.15
    for ax, k, m, lab in zip(axes.flat, CANDS, mats, "abcd"):
        matriz(ax, m, cmax=cmax, fs=10)
        ax.set_title(f"({lab}) {ETIQ_LARGA[k]}", fontsize=10, loc="left")
        ax.set_xticklabels(CLASES, fontsize=9)
        ax.set_yticklabels(CLASES, fontsize=9, rotation=90, va="center")
    fig.tight_layout(h_pad=1.2); guardar(fig, "confusion_desarrollo")


def fig_f1_por_clase(D):
    fig, ax = plt.subplots(figsize=(6.3, 3.5))
    w = 0.2
    for i, k in enumerate(CANDS):
        for j, cl in enumerate(CLASES):
            d = D["cand"][k]["global"]["por_clase"][cl]["f1"]
            x = j + (i - 1.5) * (w + 0.015)
            barra(ax, x, d["media"], w, k, label=ETIQ_LARGA[k] if j == 0 else None)
            ax.errorbar(x, d["media"], yerr=d["desvio"], color="black", lw=0.9, capsize=2.5)
            ax.text(x, d["media"] + d["desvio"] + 0.012, n3(d["media"], 2), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(3)); ax.set_xticklabels(CLASES)
    ax.set_ylim(0, 1.3); ax.set_yticks(np.arange(0, 1.01, 0.2)); coma(ax); ax.set_ylabel("F1 de la clase")
    ax.legend(ncol=2, loc="upper center", frameon=False, fontsize=9)
    fig.tight_layout(); guardar(fig, "f1_por_clase")


def fig_sesion_vs_5x4(D):
    fig, ax = plt.subplots(figsize=(6.3, 3.6))
    w = 0.36
    for i, k in enumerate(CANDS):
        for off, (v, (lo, hi)), claro in ((-w / 2 - 0.01, f1_pooled_54(D, k, "todo"), False),
                                          (w / 2 + 0.01, f1_sesion(D, k, "todo"), True)):
            barra(ax, i + off, v, w, k, claro)
            ax.errorbar(i + off, v, yerr=[[v - lo], [hi - v]], color="black", lw=1, capsize=3)
            ax.text(i + off, hi + 0.015, n3(v), ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(4)); ax.set_xticklabels([ETIQ[k] for k in CANDS])
    ax.set_ylim(0, 1.2); ax.set_yticks(np.arange(0, 1.01, 0.2)); coma(ax); ax.set_ylabel("F1 macro agrupado")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="#6e6e6e", edgecolor="black", label="Validación 5×4 por caballo (izquierda)"),
                       Patch(facecolor="white", edgecolor="black", label="Dejando afuera una sesión (derecha)")],
              loc="upper right", frameon=False, fontsize=9)
    fig.tight_layout(); guardar(fig, "sesion_vs_5x4")


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(6.3, 7.2))
    ax.set_xlim(0, 10); ax.set_ylim(0.9, 15.0); ax.axis("off")

    def caja(x, y, w, h, texto, fc="white", fs=9.5, ls="-"):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.15",
                                    fc=fc, ec="black", lw=1.1, ls=ls))
        ax.text(x, y, texto, ha="center", va="center", fontsize=fs, linespacing=1.15)

    def flecha(p, q, rad=0.0):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=12, lw=1.1, color="black",
                                     connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))
    caja(5, 14.4, 5.6, 1.0, "Video de la expresión facial\n(clip de un caballo)", "#f0f0f0")
    caja(5, 12.7, 5.6, 1.0, "Corrección de orientación\n(rotación del video)")
    caja(5, 11.0, 6.4, 1.0, "Pose de la cabeza con DeepLabCut\n(todos los fotogramas)")
    caja(5, 9.3, 6.4, 1.0, "Selección de 16 fotogramas por clip", "#f0f0f0")
    for y0, y1 in ((13.9, 13.2), (12.2, 11.5), (10.5, 9.8)):
        flecha((5, y0), (5, y1))
    caja(2.6, 6.9, 4.4, 1.7, "Rasgos de pose\n(160 por clip)\n\nBosque aleatorio\n(rama de pose)", "#d9d9d9")
    caja(7.4, 6.9, 4.4, 1.7, "ResNet-18 (apariencia)\npor fotograma\n\nMedia de los 16 vectores +\nregresión logística", "#bdbdbd")
    flecha((4.0, 8.8), (2.6, 7.8), -0.05); flecha((6.0, 8.8), (7.4, 7.8), 0.05)
    ax.text(2.6, 5.35, "probabilidades\nde pose  $p_{pose}$", ha="center", va="center", fontsize=9)
    ax.text(7.4, 5.35, "probabilidades\nvisuales  $p_{vis}$", ha="center", va="center", fontsize=9)
    flecha((2.6, 6.0), (2.6, 5.75)); flecha((7.4, 6.0), (7.4, 5.75))
    caja(5, 3.6, 7.2, 1.2, "Promedio ponderado\n$p = w\\,p_{pose} + (1-w)\\,p_{vis}$,   $w = 0{,}75$")
    flecha((2.6, 4.95), (3.8, 4.25), 0.1); flecha((7.4, 4.95), (6.2, 4.25), -0.1)
    caja(5, 1.7, 6.0, 1.0, "Clase predicha\nDisgustado · Interesado · Sereno", "#f0f0f0")
    flecha((5, 3.0), (5, 2.2))
    fig.tight_layout(); guardar(fig, "pipeline")


def main():
    D = c.cargar()
    fig_f1_candidatos(D); fig_ic_pareados(D); fig_confusion_reserva(D); fig_confusion_desarrollo(D)
    fig_f1_por_clase(D); fig_sesion_vs_5x4(D); fig_pipeline()
    print("figuras en", SALIDA)


if __name__ == "__main__":
    main()
