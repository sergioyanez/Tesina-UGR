"""Figuras del informe, generadas con matplotlib a partir de los JSON de `comparacion/resultados/`."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from datos import (CANDIDATOS, CLASES, COLOR, CORTO, NOMBRE, f1_pooled_54, f1_sesion, n, par,
                   pct)

TINTA = "#1a2233"
GRIS = "#6b7385"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": "#9aa3b5",
    "axes.labelcolor": TINTA, "xtick.color": TINTA, "ytick.color": TINTA, "text.color": TINTA,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 8.5,
    "axes.titleweight": "bold", "figure.dpi": 100, "savefig.dpi": 200,
})


def _coma(fig):
    """Coma decimal en los ejes numéricos."""
    from matplotlib.ticker import FuncFormatter
    fmt = FuncFormatter(lambda v, _: f"{v:.1f}".replace(".", ","))
    for ax in fig.axes:
        if ax.get_images():
            continue
        if ax.get_ylim()[1] <= 1.1 and ax.get_ylim()[0] >= 0 or "F1" in ax.get_ylabel():
            ax.yaxis.set_major_formatter(fmt)
        if ax.get_xlabel().startswith("Diferencia"):
            ax.xaxis.set_major_formatter(fmt)


def _guardar(fig, ruta):
    _coma(fig)
    fig.savefig(ruta, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return ruta


def _etiquetas_x(ax):
    ax.set_xticks(range(len(CANDIDATOS)))
    ax.set_xticklabels([CORTO[c] for c in CANDIDATOS], fontsize=7.2)


def fig_f1_candidatos(D, ruta):
    """F1 macro fuera de muestra con IC95 por caballo: 5×4 (izquierda) y por sesión (derecha)."""
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.5), sharey=True)
    paneles = [("Validación 5×4 por caballo (desarrollo)", f1_pooled_54),
               ("Dejando afuera una sesión de filmación", f1_sesion)]
    ancho = 0.36
    for ax, (titulo, fuente) in zip(axes, paneles):
        for i, c in enumerate(CANDIDATOS):
            for j, (sub, desfase) in enumerate((("todo", -ancho / 2 - 0.01), ("limpio", ancho / 2 + 0.01))):
                v, (lo, hi) = fuente(D, c, sub)
                ax.bar(i + desfase, v, ancho, color=COLOR[c], alpha=1.0 if sub == "todo" else 0.42,
                       hatch=None if sub == "todo" else "////", edgecolor="white" if sub == "todo" else COLOR[c],
                       linewidth=0.6)
                ax.errorbar(i + desfase, v, yerr=[[v - lo], [hi - v]], color=TINTA, lw=1, capsize=2.5)
                ax.text(i + desfase, hi + 0.012, n(v), ha="center", va="bottom", fontsize=6.6,
                        color=TINTA)
        _etiquetas_x(ax)
        ax.set_title(titulo, loc="left")
        ax.set_ylim(0, 1.05)
        ax.yaxis.grid(True, color="#e3e7ee", lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("F1 macro (barras de error: IC95 por caballo)")
    leyenda = [Patch(facecolor="#555d70", label="Todos los clips"),
               Patch(facecolor="#555d70", alpha=0.42, hatch="////", edgecolor="#555d70",
                     label="Subconjunto limpio (1080×1920, ≥ 5 s)")]
    fig.legend(handles=leyenda, loc="lower center", ncol=2, frameon=False, fontsize=7.4,
               bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return _guardar(fig, ruta)


def _heat(ax, M, titulo, sufijo=""):
    M = np.array(M, dtype=float)
    P = M / M.sum(axis=1, keepdims=True)
    ax.imshow(P, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{int(M[i, j])}\n{pct(P[i, j])}", ha="center", va="center", fontsize=7.4,
                    color="white" if P[i, j] > 0.5 else TINTA)
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(CLASES, fontsize=7.2); ax.set_yticklabels(CLASES, fontsize=7.2, rotation=90,
                                                                  va="center")
    ax.set_xlabel("Clase predicha"); ax.set_ylabel("Clase real")
    ax.set_title(titulo, loc="left")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)


def fig_confusion_fusion(D, ruta):
    """Matriz de confusión de la fusión: suma de las 20 evaluaciones y reserva (único uso)."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3))
    M54 = D["cand"][CANDIDATOS[0]]["global"]["matriz_confusion_suma"]
    _heat(axes[0], M54, "Desarrollo: suma de 20 evaluaciones")
    _heat(axes[1], D["reserva"]["metricas"]["matriz"], "Reserva: 68 clips (único uso)")
    fig.text(0.5, -0.02, "En cada celda: cantidad de clips y porcentaje de la fila (clase real).",
             ha="center", fontsize=7, color=GRIS)
    fig.tight_layout()
    return _guardar(fig, ruta)


def fig_f1_por_clase(D, ruta):
    """F1 por clase de los cuatro candidatos: 5×4 (media ± desvío) y por sesión."""
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), sharey=True)
    ancho = 0.2
    for k, ax in enumerate(axes):
        for i, c in enumerate(CANDIDATOS):
            xs, vs, es = [], [], []
            for j, cl in enumerate(CLASES):
                if k == 0:
                    d = D["cand"][c]["global"]["por_clase"][cl]["f1"]
                    v, e = d["media"], d["desvio"]
                else:
                    v, e = D["sesion"]["candidatos"][c]["metricas"]["f1"][cl], None
                x = j + (i - 1.5) * (ancho + 0.015)
                ax.bar(x, v, ancho, color=COLOR[c], label=NOMBRE[c] if j == 0 else None)
                if e is not None:
                    ax.errorbar(x, v, yerr=e, color=TINTA, lw=0.9, capsize=2)
                ax.text(x, v + (e or 0) + 0.012, n(v, 2), ha="center", va="bottom", fontsize=6)
        ax.set_xticks(range(3)); ax.set_xticklabels(CLASES)
        ax.set_ylim(0, 1.08)
        ax.yaxis.grid(True, color="#e3e7ee", lw=0.6); ax.set_axisbelow(True)
    axes[0].set_title("5×4 (media ± desvío, 20 evaluaciones)", loc="left")
    axes[1].set_title("Por sesión (F1 agrupado)", loc="left")
    axes[0].set_ylabel("F1 de la clase")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=4, frameon=False,
               fontsize=7.2, bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return _guardar(fig, ruta)


def fig_diferencias(D, ruta):
    """Diferencia fusión − otro (IC95 bootstrap pareado por caballo), 5×4 y por sesión."""
    rivales = CANDIDATOS[1:]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    filas = [("5×4 · todo", "#1f3a6e", "o", True), ("5×4 · limpio", "#1f3a6e", "o", False),
             ("Sesión · todo", "#2a8c82", "s", True), ("Sesión · limpio", "#2a8c82", "s", False)]
    sep = 0.2
    for i, r in enumerate(rivales):
        y0 = len(rivales) - 1 - i
        for k, (rot, col, mk, lleno) in enumerate(filas):
            sub = "todo" if lleno else "limpio"
            if rot.startswith("5×4"):
                p = par(D, CANDIDATOS[0], r, sub)
                d, (lo, hi) = p["diferencia_a_menos_b"], p["diferencia_ic95"]
            else:
                x = D["sesion"]["intervalos"][sub]["diferencia_vs_fusion_tardia_pose_visual"][r]
                d, (lo, hi) = x["diferencia"], x["ic95"]
            y = y0 + (1.5 - k) * sep
            ax.plot([lo, hi], [y, y], color=col, lw=1.5, solid_capstyle="round")
            ax.plot(d, y, mk, color=col, mfc=col if lleno else "white", mew=1.4, ms=5.5)
            ax.text(hi + 0.006, y, f"{d:+.3f} [{lo:+.3f}; {hi:+.3f}]".replace(".", ","), va="center",
                    fontsize=6.3, color=TINTA)
    ax.axvline(0, color="#b3402e", lw=1, ls="--")
    ax.set_yticks(range(len(rivales)))
    ax.set_yticklabels([f"Fusión − {NOMBRE[r].split(' (')[0]}" for r in reversed(rivales)], fontsize=7.4)
    ax.set_xlim(-0.08, 0.42)
    ax.set_xlabel("Diferencia de F1 macro (a la derecha de la línea, gana la fusión)")
    ax.xaxis.grid(True, color="#e3e7ee", lw=0.6); ax.set_axisbelow(True)
    leyenda = [plt.Line2D([], [], color="#1f3a6e", marker="o", ls="", label="5×4, todos los clips"),
               plt.Line2D([], [], color="#1f3a6e", marker="o", mfc="white", mew=1.4, ls="",
                          label="5×4, subconjunto limpio"),
               plt.Line2D([], [], color="#2a8c82", marker="s", ls="", label="Por sesión, todos los clips"),
               plt.Line2D([], [], color="#2a8c82", marker="s", mfc="white", mew=1.4, ls="",
                          label="Por sesión, subconjunto limpio")]
    ax.legend(handles=leyenda, loc="upper right", frameon=False, fontsize=6.8)
    fig.tight_layout()
    return _guardar(fig, ruta)


def fig_repeticiones(D, ruta):
    """F1 fuera de muestra de cada una de las 5 repeticiones (misma partición para todos)."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True)
    for ax, (clave, titulo) in zip(axes, [("f1_macro_oof", "Todos los clips"),
                                          ("f1_macro_oof_limpio", "Subconjunto limpio")]):
        for c in CANDIDATOS:
            v = [p[clave] for p in D["cand"][c]["por_particion"]]
            ax.plot(range(1, 6), v, "-o", color=COLOR[c], ms=3.5, lw=1.3, label=NOMBRE[c])
        ax.set_xticks(range(1, 6)); ax.set_xlabel("Repetición (partición de caballos en 4 folds)")
        ax.set_title(titulo, loc="left"); ax.set_ylim(0.45, 0.95)
        ax.yaxis.grid(True, color="#e3e7ee", lw=0.6); ax.set_axisbelow(True)
    axes[0].set_ylabel("F1 macro fuera de muestra")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=4, frameon=False,
               fontsize=6.8, bbox_to_anchor=(0.5, -0.07))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return _guardar(fig, ruta)


def generar_todas(D, carpeta: Path):
    carpeta.mkdir(parents=True, exist_ok=True)
    return {"f1": fig_f1_candidatos(D, carpeta / "fig1_f1_candidatos.png"),
            "conf": fig_confusion_fusion(D, carpeta / "fig2_confusion_fusion.png"),
            "clase": fig_f1_por_clase(D, carpeta / "fig3_f1_por_clase.png"),
            "dif": fig_diferencias(D, carpeta / "fig4_diferencias.png"),
            "rep": fig_repeticiones(D, carpeta / "fig5_repeticiones.png")}
