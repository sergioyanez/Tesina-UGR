"""Tarea 4.3 — cada cuánto la interfaz puede responder, medido sobre videos crudos.

Es el número que decide si el entregable sirve en la práctica, y por eso se mide en vez de
estimarse. El diseño anticipó que la negativa sería el resultado más probable, a partir del 38 %
de clips íntegros del propio dataset; acá se comprueba sobre la ENTRADA REAL de la interfaz, que
es un video crudo y no un clip ya segmentado.

**Sesgo declarado:** los videos de la muestra contribuyeron al conjunto de entrenamiento. Eso NO
afecta la tasa de negativa, que es una propiedad del preprocesamiento y no del modelo, pero sí
haría optimista cualquier medición de acierto — por eso acá no se informa ninguna.

Uso:
    python3 openspec/changes/interfaz-de-inferencia/tasa_de_negativa.py [por_clase]
"""

import csv
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path


def _raiz_del_repo():
    for candidata in Path(__file__).resolve().parents:
        if (candidata / "equinos" / "__init__.py").exists():
            return candidata
    raise RuntimeError("No se encontró la raíz del repositorio desde " + __file__)


sys.path.insert(0, str(_raiz_del_repo()))

from equinos import inferencia, rutas  # noqa: E402

SEMILLA = 42


def main(por_clase=4):
    rutas.inicializar(verbose=False)
    with (rutas.carpeta("resultados") / "inventario_videos.csv").open(encoding="utf-8") as f:
        videos = list(csv.DictReader(f))

    por_clase_dict = defaultdict(list)
    for v in videos:
        por_clase_dict[v["clase"]].append(v)

    azar = random.Random(SEMILLA)
    muestra = []
    for clase in sorted(por_clase_dict):
        candidatos = sorted(por_clase_dict[clase], key=lambda v: v["archivo"])
        muestra.extend(azar.sample(candidatos, min(por_clase, len(candidatos))))

    print("=" * 78)
    print("TASA DE NEGATIVA DE LA INTERFAZ")
    print("=" * 78)
    print(f"{len(muestra)} videos crudos, {por_clase} por clase, semilla {SEMILLA}.")
    print("Cada uno cuesta ~35 s.\n")

    motivos = Counter()
    for video in muestra:
        ruta = Path(video["ruta"])
        if not ruta.is_absolute():
            ruta = rutas.resolver(video["ruta"])
        resultado = inferencia.clasificar_video(ruta)
        if resultado.hay_prediccion:
            motivos["RESPONDIÓ"] += 1
            estado = f"predijo {resultado.clase}"
        else:
            motivos[resultado.motivo] += 1
            estado = f"negativa: {resultado.motivo}"
        print(f"  {video['clase']:<11} {video['archivo'][:40]:<42} {estado}")

    respondio = motivos["RESPONDIÓ"]
    print(f"\n{'=' * 78}")
    print(f"Respondió en {respondio} de {len(muestra)} ({respondio / len(muestra):.0%}).")
    print("\nMotivos de negativa:")
    for motivo, veces in motivos.most_common():
        if motivo == "RESPONDIÓ":
            continue
        print(f"  {motivo:<26} {veces:>3}  — {inferencia.MOTIVOS[motivo]}")
    print("\nRecordar el sesgo declarado en el encabezado de este archivo.")
    return motivos


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
