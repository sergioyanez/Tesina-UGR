"""Diagramas: pipeline video -> DeepLabCut -> features -> clasificador, y esquema de folds."""
from comun import *
plt = estilo()
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

def caja(ax, x, y, w, h, txt, c="#dfeaf2", ec="#1b3f5a", fs=8.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=c, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", wrap=True)

def flecha(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12, lw=1.3, color="#444"))

fig, ax = plt.subplots(figsize=(11.5, 5.2)); ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(-0.8, 6)
ax.grid(False)
caja(ax, 0.1, 2.4, 1.6, 1.2, "Video\n(274; clips de\n5–7 s, 30 fps)", c="#eeeeee")
caja(ax, 2.1, 2.4, 1.9, 1.2, "DeepLabCut 3.0.1\nSuperAnimal-Quadruped\n(39 landmarks, 2D)", c="#fde9c9", ec="#a86a00", bold=True)
caja(ax, 4.4, 2.4, 1.9, 1.2, "Ventana de 16 frames\ncon ojo–hocico\nválido (conf ≥ 0,50)", c="#eeeeee")
flecha(ax, 1.7, 3, 2.1, 3); flecha(ax, 4.0, 3, 4.4, 3)
# rama MIL
caja(ax, 6.7, 4.2, 1.8, 1.1, "Alineado por\nojo–hocico, recorte\n224×343 (JPEG)", c="#eeeeee")
caja(ax, 8.8, 4.2, 1.6, 1.1, "ResNet-18\ncongelada\n512 / frame", c="#dfeaf2")
caja(ax, 10.6, 4.2, 1.3, 1.1, "MIL + atención\n(+ 9 geom.)\n132 355 par.", c="#dfeaf2", bold=True)
flecha(ax, 6.3, 3.3, 6.7, 4.7); flecha(ax, 8.5, 4.75, 8.8, 4.75); flecha(ax, 10.4, 4.75, 10.6, 4.75)
ax.text(6.7, 5.55, "MODELO VIGENTE (visual)", fontsize=9, fontweight="bold", color="#1b3f5a")
# rama SimBA
caja(ax, 6.7, 0.7, 1.8, 1.1, "Landmarks\nnormalizados\n(escala del clip)", c="#eeeeee")
caja(ax, 8.8, 0.7, 1.6, 1.1, "160 features\n«tipo SimBA»\n(código propio)", c="#e3f1dc", ec="#2f6b1f", bold=True)
caja(ax, 10.6, 0.7, 1.3, 1.1, "Random Forest\n+ SHAP", c="#e3f1dc", ec="#2f6b1f", bold=True)
flecha(ax, 6.3, 2.7, 6.7, 1.25); flecha(ax, 8.5, 1.25, 8.8, 1.25); flecha(ax, 10.4, 1.25, 10.6, 1.25)
ax.text(6.7, 1.95, "EXPERIMENTO COMPARATIVO (solo geometría)", fontsize=9, fontweight="bold", color="#2f6b1f")
caja(ax, 10.6, 2.55, 1.3, 0.9, "Clase:\nSereno / Interesado /\nDisgustado", c="#fdf3d0", fs=8)
flecha(ax, 11.25, 4.2, 11.25, 3.45); flecha(ax, 11.25, 1.8, 11.25, 2.55)
ax.text(0.1, -0.5, "DLC: video → landmarks.   SimBA (real): landmarks → features → clasificador.   Ninguno cumple ambos roles.", fontsize=8.5, color="#a86a00")
fig.savefig(FIGURAS / "fig_diagrama_pipeline.png", dpi=200, bbox_inches="tight"); plt.close(fig)

fig, ax = plt.subplots(figsize=(11, 4.6)); ax.axis("off"); ax.grid(False); ax.set_xlim(0, 100); ax.set_ylim(-2.5, 12)
ax.text(0, 11.2, "133 caballos / 359 clips; 4 folds agrupados por id_caballo (≈ 33 caballos de test por fold)", fontsize=10, fontweight="bold")
cols = {"te": "#c0392b", "tr": "#2a7f9e", "va": "#e0a800"}
for k in range(4):
    y = 8.4 - k * 2.3
    ax.text(0, y + 0.55, f"Fold {k}", fontsize=9, va="center")
    # el test ocupa la posición k
    for j in range(4):
        x0 = 10 + j * 22
        if j == k:
            ax.add_patch(Rectangle((x0, y), 21, 1.1, fc=cols["te"], ec="white"))
            ax.text(x0 + 10.5, y + 0.55, "TEST (≈ 90 clips, 1 sola vez)", ha="center", va="center", color="white", fontsize=8)
        else:
            ax.add_patch(Rectangle((x0, y), 21, 1.1, fc=cols["tr"], ec="white"))
            ax.text(x0 + 10.5, y + 0.55, "train_val", ha="center", va="center", color="white", fontsize=8)
ax.add_patch(Rectangle((10, -1.0), 6, 0.9, fc=cols["tr"])); ax.text(17, -0.55, "train interno ≈ 93 cab. / 199 clips", va="center", fontsize=8.5)
ax.add_patch(Rectangle((58, -1.0), 6, 0.9, fc=cols["va"])); ax.text(65, -0.55, "validación interna: 7 caballos (≈ 70 clips)", va="center", fontsize=8.5)
ax.text(10, -1.9, "Dentro de cada fold, `train_val` se corta otra vez por caballo en train interno y validación interna.", fontsize=8.5)
fig.savefig(FIGURAS / "fig_diagrama_folds.png", dpi=200, bbox_inches="tight"); plt.close(fig)
print("ok")
