"""Calibración de las probabilidades que muestra la interfaz (modelos de fold sobre su test, fuera de muestra)."""
import numpy as np
import torch
from comun import *
from equinos import clasificacion as CL
from equinos import pose

clips = particion(); orden, X = caracteristicas_v2(); indice = {c: i for i, c in enumerate(orden)}
pose_filas = pose.leer_pose(RES / "pose_seleccionada_v2.csv")
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
P, Y, F = [], [], []
for k, (tr, te) in enumerate(CL.folds_por_ejemplar(clips)):
    m = CL.MILFusionModel(dropout=0.3).to(dev)
    m.load_state_dict(torch.load(DATOS_REPO / "modelos/objetivo3_v2" / f"mil_v2_fold{k}.pt", map_location=dev, weights_only=True)); m.eval()
    dl = torch.utils.data.DataLoader(CL.MultimodalEquineDataset(te, X, indice, pose_filas), batch_size=32, shuffle=False, collate_fn=CL.collate_fn)
    with torch.no_grad():
        for vv, vg, mask, _, lab in dl:
            P.append(torch.softmax(m(vv.to(dev), vg.to(dev), mask.to(dev)), 1).cpu().numpy()); Y.append(lab.numpy()); F += [k] * len(lab)
P = np.concatenate(P); Y = np.concatenate(Y)
pred = P.argmax(1); conf = P.max(1); ok = (pred == Y).astype(float)
# concordancia con la evaluación vigente
ev = evaluacion_vigente()
idx = {c: i for i, c in enumerate(CL.CLASES)}
ev_pred = np.array([idx[p["predicha"]] for p in ev["predicciones_fuera_de_muestra"]])
bins = np.linspace(0, 1, 11); ece = 0.0; tabla = []
for a, b in zip(bins[:-1], bins[1:]):
    s = (conf > a) & (conf <= b)
    if s.sum():
        ece += s.mean() * abs(ok[s].mean() - conf[s].mean())
        tabla.append({"rango": f"({a:.1f},{b:.1f}]", "n": int(s.sum()), "confianza_media": float(conf[s].mean()), "acierto": float(ok[s].mean())})
nll = float(-np.log(P[np.arange(len(Y)), Y] + 1e-12).mean())
brier = float(((P - np.eye(3)[Y]) ** 2).sum(1).mean())
res = {"clips": len(Y), "predicciones_iguales_a_la_evaluacion_vigente": int((pred == ev_pred).sum()),
       "confianza_media": float(conf.mean()), "accuracy": float(ok.mean()), "ECE_10_bins": float(ece), "NLL": nll,
       "NLL_azar_uniforme": float(np.log(3)), "Brier": brier, "tabla": tabla,
       "pct_clips_con_confianza_mayor_0.9": float((conf > .9).mean() * 100),
       "acierto_con_confianza_mayor_0.9": float(ok[conf > .9].mean()) if (conf > .9).any() else None}
guardar_json(res, "calibracion.json")
plt = estilo()
fig, ax = plt.subplots(figsize=(4.8, 4.2))
ax.plot([0, 1], [0, 1], "k--", lw=1)
ax.plot([t["confianza_media"] for t in tabla], [t["acierto"] for t in tabla], "o-", color="#1b3f5a")
for t in tabla:
    ax.text(t["confianza_media"], t["acierto"] + .03, str(t["n"]), fontsize=7, ha="center")
ax.set_xlabel("Confianza media del modelo"); ax.set_ylabel("Acierto observado"); ax.set_title(f"Calibración (ECE = {ece:.3f})".replace(".", ","))
ax.set_xlim(.3, 1.02); ax.set_ylim(0, 1.02)
fig.tight_layout(); guardar_fig(fig, "fig_calibracion.png"); plt.close(fig)
print(json.dumps({k: v for k, v in res.items() if k != "tabla"}, ensure_ascii=False, indent=1)); [print(t) for t in tabla]
