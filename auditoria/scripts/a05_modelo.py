"""Arquitectura: parámetros (instanciando los modelos), dimensionalidad frente al nº de muestras,
e invariancia al orden de los frames (devolución de Juan: «¿espacio-temporal o solo procesamiento
del video?»). Usa los pesos guardados de cada fold (solo lectura)."""
import numpy as np
import torch
from torchvision import models

from comun import *
from equinos import clasificacion as CL
from equinos import pose

def cuenta(m):
    tot = sum(p.numel() for p in m.parameters())
    ent = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return tot, ent

mil = CL.MILFusionModel(dropout=0.3)
tot, ent = cuenta(mil)
partes = {"geom_mlp (9→64→128)": cuenta(mil.geom_mlp)[0], "attention (640→128→1)": cuenta(mil.attention)[0],
          "classifier (640→64→3)": cuenta(mil.classifier)[0]}
resnet = models.resnet18(weights=None)
extractor = torch.nn.Sequential(*list(resnet.children())[:-1])
tot_r, _ = cuenta(extractor)
for p_ in extractor.parameters():
    p_.requires_grad = False
_, ent_r = cuenta(extractor)
visual_puro = None
arq = {"MIL multimodal: parametros_totales": tot, "MIL multimodal: parametros_entrenables": ent,
       "por_componente": partes, "pct_atencion": 100 * partes["attention (640→128→1)"] / tot,
       "resnet18_sin_fc: parametros_totales": tot_r, "resnet18_sin_fc: entrenables_tras_congelar": ent_r,
       "pipeline_total_parametros": tot + tot_r, "pipeline_entrenables": ent + ent_r,
       "guia_dice_132355_entrenables": ent == 132355,
       "dimensiones_por_clip": {"visuales_por_frame": 512, "geometricas_por_frame": 9, "frames": 16,
                                "numero_de_valores_por_clip": 16 * (512 + 9)}}
clips = particion()
n = len(clips); cab = len({c["id_caballo"] for c in clips})
arq["muestras"] = {"clips": n, "caballos": cab, "clips_train_por_fold_aprox": 270, "clips_train_interno": 199,
                   "caballos_train_interno": 93}
arq["razones"] = {"parametros_entrenables_por_clip_train_interno": ent / 199,
                  "parametros_entrenables_por_caballo_train_interno": ent / 93,
                  "dimension_de_entrada_por_clip_sobre_clips_train": 16 * 521 / 199,
                  "dimension_de_entrada_vector_por_frame_sobre_clips_train": 521 / 199}

# ---- invariancia al orden ----------------------------------------------------------------
orden, X = caracteristicas_v2()
indice = {c: i for i, c in enumerate(orden)}
pose_filas = pose.leer_pose(RES / "pose_seleccionada_v2.csv")
folds = CL.folds_por_ejemplar(clips)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rng = np.random.default_rng(0)
res = {"clips_evaluados": 0, "predicciones_cambiadas_con_orden_barajado": 0,
       "max_dif_abs_logits": 0.0, "max_dif_abs_logits_con_orden_invertido": 0.0, "por_fold": []}
for i, (tr, te) in enumerate(folds):
    m = CL.MILFusionModel(dropout=0.3).to(dev)
    m.load_state_dict(torch.load(DATOS_REPO / "modelos/objetivo3_v2" / f"mil_v2_fold{i}.pt",
                                 map_location=dev, weights_only=True)); m.eval()
    ds = CL.MultimodalEquineDataset(te, X, indice, pose_filas)
    cambios = 0; mx = 0.0; mxi = 0.0
    for j in range(len(ds)):
        vv, vg, _ = ds[j]
        vv, vg = vv.unsqueeze(0).to(dev), vg.unsqueeze(0).to(dev)
        mask = torch.ones(1, vv.shape[1], dtype=torch.bool, device=dev)
        with torch.no_grad():
            l0 = m(vv, vg, mask)
            perm = torch.as_tensor(rng.permutation(vv.shape[1]), device=dev)
            l1 = m(vv[:, perm], vg[:, perm], mask)
            inv = torch.arange(vv.shape[1] - 1, -1, -1, device=dev)
            l2 = m(vv[:, inv], vg[:, inv], mask)
        cambios += int(l0.argmax() != l1.argmax())
        mx = max(mx, float((l0 - l1).abs().max())); mxi = max(mxi, float((l0 - l2).abs().max()))
    res["clips_evaluados"] += len(ds); res["predicciones_cambiadas_con_orden_barajado"] += cambios
    res["max_dif_abs_logits"] = max(res["max_dif_abs_logits"], mx)
    res["max_dif_abs_logits_con_orden_invertido"] = max(res["max_dif_abs_logits_con_orden_invertido"], mxi)
    res["por_fold"].append({"fold": i, "clips": len(ds), "cambios": cambios, "max_dif_logits": mx})
res["conclusion"] = ("El MIL es invariante al orden de los frames por construcción: la salida es una suma "
                     "ponderada por softmax de instancias que se procesan independientemente; no hay "
                     "codificación de posición ni recurrencia. Es 'espacial + agregación de conjunto'.")
guardar_json({"arquitectura": arq, "invariancia_al_orden": res}, "arquitectura.json")
print(json.dumps({"arquitectura": arq, "invariancia": res}, ensure_ascii=False, indent=1))
