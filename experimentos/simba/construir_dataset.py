"""Dataset tabular por CLIP: features tipo SimBA (código propio) + trazabilidad.

Unidad y conjunto: los 359 clips íntegros del MIL vigente (`particion_clips.json`), 133 caballos.
Las columnas `meta_*` y `fold`/`clip_id`/`video`/`caballo`/`clase` son TRAZABILIDAD; ninguna entra
al clasificador (los nombres de features no tienen esos prefijos; ver `columnas.json`).

Uso: python experimentos/simba/construir_dataset.py
"""
import json

import numpy as np
import pandas as pd

import datos
import features_simba as F


def main():
    clips = datos.clips_con_fold()
    P = datos.pose_por_clip(clips)
    T = datos.tiempos_por_clip()
    meta = datos.metadatos(clips)
    video = datos.video_origen(clips)
    filas, escalas = [], []
    for c in clips:
        frames, fps, _ = T[c["clip"]]
        v, nombres, grupos, esc = F.vector_del_clip(P[c["clip"]], frames, fps)
        filas.append(v)
        escalas.append(esc)
    X = np.stack(filas)
    df = pd.DataFrame(X, columns=nombres)
    limpio = (meta[:, 2] == 1080) & (meta[:, 3] == 1920) & (meta[:, 0] >= 5.0)
    traza = pd.DataFrame({
        "clip_id": [c["clip"] for c in clips], "video": video,
        "caballo": [c["id_caballo"] for c in clips], "clase": [c["clase"] for c in clips],
        "fold": [c["fold"] for c in clips],
        "meta_duracion_seg": meta[:, 0], "meta_fps": meta[:, 1], "meta_ancho": meta[:, 2],
        "meta_alto": meta[:, 3], "meta_escala_cabeza_px": escalas, "meta_subconjunto_limpio": limpio,
    })
    out = pd.concat([traza, df], axis=1)
    out.to_csv(datos.SALIDA / "dataset_tabular_simba.csv", index=False)
    (datos.SALIDA / "columnas.json").write_text(json.dumps(
        {"features": nombres, "grupos": dict(zip(nombres, grupos)),
         "trazabilidad": list(traza.columns)}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(out.shape, "features:", len(nombres), "limpio:", int(limpio.sum()))
    print("NaN global en features: %.3f" % np.isnan(X).mean())


if __name__ == "__main__":
    main()
