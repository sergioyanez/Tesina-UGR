"""Reproducibilidad: dependencias fijadas frente a las instaladas, semillas, configuración,
scripts, checkpoints, trazabilidad (registro de experimentos) y estado del repositorio."""
import importlib.metadata as md
import re
import subprocess

from comun import *

req = (WT / "requirements-local.txt").read_text(encoding="utf-8")
fijados = {}
for linea in req.splitlines():
    m = re.match(r"^([A-Za-z0-9_.\-]+)==([^\s#]+)", linea.strip())
    if m:
        fijados[m.group(1).lower().replace("_", "-")] = m.group(2)
inst = {}
for k, v in fijados.items():
    try:
        inst[k] = md.version(k)
    except md.PackageNotFoundError:
        inst[k] = None
dif = {k: {"requirements": fijados[k], "instalada": inst[k]} for k in fijados if inst[k] != fijados[k]}
src_clas = (WT / "equinos" / "clasificacion.py").read_text(encoding="utf-8")
src_part = (WT / "equinos" / "particion.py").read_text(encoding="utf-8")
semillas = {
    "SEMILLA global (clasificacion.SEMILLA)": re.search(r"SEMILLA = (\d+)", src_clas).group(1),
    "torch.manual_seed / random.seed / np.random.seed / cuda.manual_seed_all en entrenar_mil_multimodal": all(
        s in src_clas for s in ("torch.manual_seed(semilla)", "random.seed(semilla)", "np.random.seed(semilla)", "manual_seed_all")),
    "generador propio del DataLoader de train": "generador.manual_seed(semilla)" in src_clas,
    "cudnn.deterministic / use_deterministic_algorithms": ("deterministic" in src_clas),
    "partición sin azar (determinista por orden y desempate por nombre)": ("import random" not in src_part and "numpy" not in src_part),
    "semilla del bootstrap (lineas_de_base)": "SEMILLA_BOOTSTRAP = 1",
}
def git(*a):
    return subprocess.run(["git", *a], cwd=WT, capture_output=True, text=True).stdout.strip()
estado = {
    "commit_base": git("rev-parse", "--short", "main"), "rama_de_la_auditoria": git("rev-parse", "--abbrev-ref", "HEAD"),
    "ramas": git("branch", "-a").splitlines(),
    "gitignore_datos": "datos_equinos" in (WT / ".gitignore").read_text(encoding="utf-8") if (WT / ".gitignore").exists() else None,
}
arch = {
    "evaluacion_vigente": (RES / "objetivo3_evaluacion_v2.json").exists(),
    "lineas_de_base": (RES / "objetivo3_lineas_de_base.json").exists(),
    "pesos_de_los_4_folds": all((DATOS_REPO / "modelos/objetivo3_v2" / f"mil_v2_fold{i}.pt").exists() for i in range(4)),
    "modelo_de_produccion_y_procedencia": (DATOS_REPO / "modelos/objetivo3_v2/mil_multimodal_produccion.json").exists(),
    "huella_de_caracteristicas": leer_json(RES / "caracteristicas_backbone_v2.json")["huella"],
    "huella_registrada_en_el_modelo_de_produccion": leer_json(DATOS_REPO / "modelos/objetivo3_v2/mil_multimodal_produccion.json")["huella_caracteristicas"],
    "scripts_de_evaluacion_en_main": sorted(p.name for p in (WT / "scripts").glob("*.py")),
    "scripts_pyc_sin_fuente_en_scripts/__pycache__": sorted(p.name for p in (WT / "scripts" / "__pycache__").glob("*.pyc")) if (WT / "scripts/__pycache__").exists() else [],
    "docs_registro_de_experimentos": (WT / "docs/registro_de_experimentos.md").exists(),
    "tests": sorted(p.name for p in (WT / "tests").glob("test_*.py")),
}
# huella basada en mtime: ¿es robusta?
src_car = (WT / "equinos" / "caracteristicas.py").read_text(encoding="utf-8")
fragilidad = {"la huella de la caché usa st_mtime del CSV": "st_mtime" in src_car,
              "consecuencia": "copiar o tocar dataset_frames_alineados_v2.csv invalida la caché y obliga a re-extraer (y escribe en la raíz de datos); no cambia resultados porque el backbone es determinista, pero rompe la trazabilidad por huella"}
ram = {}
# ramas laterales con scripts de experimentos
for rama in ("feat/enriquecer-canal-geometrico", "feat/validacion-cruzada-anidada-NestedCV", "feat/finetuning-visual"):
    ram[rama] = {"scripts": [x for x in git("ls-tree", "-r", "--name-only", rama, "scripts/").splitlines()],
                 "ultimo_commit": git("log", "-1", "--format=%h %ad %s", "--date=short", rama)}
guardar_json({"dependencias": {"fijadas": len(fijados), "diferencias_con_lo_instalado": dif, "instaladas_clave": {k: inst.get(k) for k in ("torch", "numpy", "scikit-learn", "deeplabcut", "opencv-python", "streamlit")}},
              "semillas": semillas, "estado_git": estado, "artefactos": arch, "fragilidad_cache": fragilidad, "ramas_laterales": ram}, "reproducibilidad.json")
print(json.dumps({"dif": dif, "semillas": semillas, "arch": {k: v for k, v in arch.items() if k != "tests"}, "ram": ram, "frag": fragilidad}, ensure_ascii=False, indent=1))
