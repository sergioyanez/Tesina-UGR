"""Tarea 4.1 — la inferencia tiene que reproducir el conjunto sobre el que se midió el modelo.

Es la verificación que sostiene todo lo que la interfaz afirma. Si la cadena de inferencia
difiere del preprocesamiento evaluado, aunque sea en un umbral o en un paso de compresión, las
entradas son de otra distribución y **la confiabilidad informada deja de aplicar sin que nada
falle**. No hay excepción ni número raro que lo delate: simplemente deja de ser cierto.

Se comprueba pasando por la cadena completa clips que YA están en el dataset y comparando sus
características contra la caché. Sólo clips alineados por detección propia: los que tienen frames
interpolados no se pueden reproducir, porque la inferencia no interpola a propósito.

Uso:
    python3 openspec/changes/interfaz-de-inferencia/verificar_inferencia.py [cuántos]
"""

import csv
import sys
from pathlib import Path


def _raiz_del_repo():
    for candidata in Path(__file__).resolve().parents:
        if (candidata / "equinos" / "__init__.py").exists():
            return candidata
    raise RuntimeError("No se encontró la raíz del repositorio desde " + __file__)


sys.path.insert(0, str(_raiz_del_repo()))

from equinos import caracteristicas, inferencia, particion, rutas  # noqa: E402


def main(cuantos=3):
    import numpy as np

    rutas.inicializar(verbose=False)
    clips = particion.cargar()
    orden, X = caracteristicas.cargar()
    indice = {c: i for i, c in enumerate(orden)}
    with (rutas.carpeta("resultados") / "inventario_clips.csv").open(encoding="utf-8") as archivo:
        meta = {c["clip"]: c for c in csv.DictReader(archivo)}

    candidatos = [c for c in clips if c["solo_detectado"]][:cuantos]
    print("=" * 78)
    print("VERIFICACIÓN DE LA CADENA DE INFERENCIA")
    print("=" * 78)
    print(f"{len(candidatos)} clips del propio dataset, alineados por detección propia.")
    print("Cada uno cuesta ~35 s: la mayor parte es DeepLabCut.\n")

    exactos = 0
    for clip in candidatos:
        ruta = rutas.resolver(meta[clip["clip"]]["ruta_clip"])
        preparado = inferencia.preparar_clip(ruta)
        if not isinstance(preparado, list):
            print(f"  {clip['clip'][:44]:<46} NEGATIVA: {preparado.motivo}")
            continue
        diferencia = np.abs(inferencia._caracteristicas_de(preparado) - X[indice[clip["clip"]]])
        exacto = diferencia.max() == 0
        exactos += exacto
        print(f"  {clip['clip'][:44]:<46} dif_max {diferencia.max():.6f}  "
              f"{'EXACTO' if exacto else 'DIFIERE'}")

    print(f"\n{exactos} de {len(candidatos)} reproducen la caché exactamente.")
    if exactos != len(candidatos):
        print("\nALGUNO DIFIERE. La interfaz está midiendo otra distribución y la confiabilidad")
        print("que informa no aplica. Revisar antes de mostrarle un resultado a nadie.")
        return 1

    print("\nLa cadena de inferencia es idéntica a la que produjo el conjunto evaluado.")
    print("Lo que la interfaz informa como confiabilidad corresponde a su entrada.")
    print()
    print("NOTA sobre la tarea 4.2. Comparar la predicción contra la que dio el fold en que el")
    print("clip era test NO valida nada: el modelo de producción vio ese clip al entrenarse, así")
    print("que lo acierta por memorización. Los dos modelos son distintos y difieren con razón.")
    print("Lo que valida la cadena es la igualdad de características de arriba.")
    return 0


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 3))
