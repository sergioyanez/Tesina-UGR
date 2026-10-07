"""Reproduce toda la auditoría: ejecuta los scripts en orden y regenera datos/, figuras/ y secciones/.

Uso (desde la raíz del worktree, con el entorno del proyecto):

    ~/.venvs/tesina-equinos/bin/python auditoria/scripts/correr_todo.py            # completo (~45 min con GPU)
    ~/.venvs/tesina-equinos/bin/python auditoria/scripts/correr_todo.py --rapido   # omite lo caro y reutiliza sus salidas

Lo caro: hash de 274 videos (a02), re-entrenamientos (a04, a15) y DeepLabCut sobre videos (a09).
Solo lee `datos_equinos/` del repositorio principal (AUDITORIA_REPO puede apuntar a otra copia).
"""
import subprocess
import sys
import time
from pathlib import Path

AQUI = Path(__file__).resolve().parent
CAROS = {"a02_duplicados.py", "a04_reentrenar.py", "a15_particiones_alternativas.py", "a09_orientacion_inferencia.py", "a09b_resumen.py"}
ORDEN = [
    "a01_dataset.py", "a02_duplicados.py", "a03_balance_folds_leakage.py", "a04_reentrenar.py", "a05_modelo.py",
    "a06_metricas.py", "a07_early_stopping_overfitting.py", "a08_entrenamiento_vs_inferencia.py",
    "a09_orientacion_inferencia.py", "a09b_resumen.py", "a10_reproducibilidad.py", "a11_identidades.py",
    "a12_app_streamlit.py", "a13_seleccion_frames.py", "a15_particiones_alternativas.py", "a14_repetibilidad.py",
    "a17_confusores.py", "a18_calibracion.py", "a19_identidad.py", "a20_validacion_interna.py", "a16_ensamblar.py"]


def main():
    rapido = "--rapido" in sys.argv
    tiempos = {}
    for s in ORDEN:
        if rapido and s in CAROS:
            print(f"== {s}: omitido (--rapido)")
            continue
        t0 = time.time()
        print(f"== {s} …", flush=True)
        r = subprocess.run([sys.executable, str(AQUI / s)], cwd=AQUI, capture_output=True, text=True)
        tiempos[s] = time.time() - t0
        if r.returncode != 0:
            print(r.stdout[-1500:], r.stderr[-3000:])
            raise SystemExit(f"FALLÓ {s}")
        print(f"   ok ({tiempos[s]:.0f} s)")
    print("Listo:", {k: round(v) for k, v in tiempos.items()})


if __name__ == "__main__":
    main()
