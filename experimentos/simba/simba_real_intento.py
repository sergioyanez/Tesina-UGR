"""Intento de usar el extractor REAL de SimBA (UserDefinedFeatureExtractor) sobre los landmarks.

Corre en el venv aparte `~/.venvs/simba` (simba-uw-tf-dev). Construye en un directorio temporal un
proyecto SimBA con los 39 bodyparts de DLC como body-part config definido por el usuario, escribe
los landmarks de N clips en el formato interno de SimBA (`<bp>_x, <bp>_y, <bp>_p`) y llama al
extractor. Registra qué pasó en `resultados/simba_real_intento.json`. SimBA necesita `tkinter`
para importarse (no instalado en este equipo, sin sudo): se usa un stub vacío de tkinter, que sólo
afecta a la GUI, no a los extractores.

Uso: PYTHONPATH=<stub_tkinter> ~/.venvs/simba/bin/python experimentos/simba/simba_real_intento.py
"""
import json
import os
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import datos  # noqa: E402

N_CLIPS = int(os.environ.get('N_CLIPS', '0'))   # 0 = todos los clips
registro = {"simba_instalado": True, "pasos": []}
try:
    import simba
    registro["simba_version_paquete"] = "simba-uw-tf-dev 5.6.1"
    from simba.utils.config_creator import ProjectConfigCreator
    from simba.feature_extractors.feature_extractor_user_defined import UserDefinedFeatureExtractor

    base = tempfile.mkdtemp(prefix="simba_real_")
    ProjectConfigCreator(project_path=base, project_name="equinos", target_list=["Sereno", "Interesado", "Disgustado"],
                         pose_estimation_bp_cnt="user_defined", body_part_config_idx=0, animal_cnt=1, file_type="csv")
    pf = Path(base) / "equinos" / "project_folder"
    (pf / "logs/measures/pose_configs/bp_names/project_bp_names.csv").write_text(
        "\n".join(datos.BODYPARTS) + "\n", encoding="utf-8")
    registro["pasos"].append("proyecto SimBA creado con 39 body-parts definidos por el usuario")

    clips = datos.clips_con_fold()
    if N_CLIPS:
        clips = clips[:N_CLIPS]
    registro['n_clips_probados'] = len(clips)
    meta = datos.metadatos(clips)
    P = datos.pose_por_clip(clips)
    T = datos.tiempos_por_clip()
    info = []
    for k, c in enumerate(clips):
        a = P[c["clip"]]
        cols = {}
        for i, bp in enumerate(datos.BODYPARTS):
            cols[f"{bp}_x"], cols[f"{bp}_y"], cols[f"{bp}_p"] = a[:, i, 0], a[:, i, 1], a[:, i, 2]
        df = pd.DataFrame(cols)
        nombre = c["clip"].rsplit(".", 1)[0]
        df.to_csv(pf / "csv/outlier_corrected_movement_location" / f"{nombre}.csv")
        info.append({"Video": nombre, "fps": T[c["clip"]][1], "Resolution_width": int(meta[k, 2]),
                     "Resolution_height": int(meta[k, 3]), "Distance_in_mm": 0, "pixels/mm": 1.0})
    pd.DataFrame(info).to_csv(pf / "logs/video_info.csv", index=False)
    registro["pasos"].append("landmarks escritos como <bp>_x/<bp>_y/<bp>_p, 16 filas por clip")

    ext = UserDefinedFeatureExtractor(config_path=str(pf / "project_config.ini"))
    ext.run()
    salida = sorted((pf / "csv/features_extracted").glob("*.csv"))
    registro["pasos"].append(f"extractor ejecutado; archivos: {len(salida)}")
    if salida:
        d = pd.read_csv(salida[0])
        registro["columnas_extraidas_por_clip"] = int(d.shape[1])
        registro["filas_por_clip"] = int(d.shape[0])
        registro["columnas_no_pose"] = [x for x in d.columns if not x.endswith(("_x", "_y", "_p")) and not x.startswith("Unnamed")]
        # agregación por clip de las 59 columnas por frame propias de SimBA (no pose)
        filas, nombres = [], None
        for c in clips:
            d = pd.read_csv(pf / "csv/features_extracted" / (c["clip"].rsplit(".", 1)[0] + ".csv"))
            cols = [x for x in d.columns if not x.endswith(("_x", "_y", "_p")) and not x.startswith("Unnamed")]
            v = d[cols]
            agg = pd.concat([v.mean().add_suffix("__media"), v.std(ddof=0).add_suffix("__desvio"),
                             v.max().add_suffix("__max")])
            filas.append(agg)
        out = pd.DataFrame(filas)
        out.insert(0, "clip_id", [c["clip"] for c in clips])
        out.to_csv(datos.SALIDA / "dataset_tabular_simba_real.csv", index=False)
        registro["columnas_agregadas"] = int(out.shape[1] - 1)
    registro["resultado"] = "EJECUTADO"
except Exception as e:  # el fallo es un resultado
    registro["resultado"] = "FALLO"
    registro["error"] = f"{type(e).__name__}: {e}"
    registro["traza"] = traceback.format_exc()[-1500:]

(datos.SALIDA / "simba_real_intento.json").write_text(json.dumps(registro, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps({k: v for k, v in registro.items() if k != "traza"}, ensure_ascii=False, indent=1))
