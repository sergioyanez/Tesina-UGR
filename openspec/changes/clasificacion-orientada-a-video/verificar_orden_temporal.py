"""Tarea 1.2 — el frame `k` del tensor ES el archivo que el CSV declara para `orden = k`.

Un desorden temporal no rompe nada: no lanza excepciones, no cambia una forma, y todos los
números siguen saliendo. Pero **arruinaría la conclusión central del objetivo 3**: si los frames
llegaran barajados de fábrica, «permutar el orden no cambia las predicciones» sería trivialmente
cierto y no diría nada sobre la hipótesis del trabajo.

Por eso esta verificación es parte del resultado y no higiene opcional.

**Sin tolerancia absoluta.** Extraer un frame solo y extraerlo dentro de un lote de 16 da vectores
que difieren en ~1e-3: cuDNN elige algoritmos distintos según el tamaño del lote. Comparar contra
un épsilon fijo mediría esa diferencia numérica y no la correspondencia. Lo que se verifica es
**discriminativo**: que el vector recién extraído del archivo de `orden = k` se parezca a la fila
`k` del tensor MÁS que a cualquiera de las otras 15. Eso no depende de ninguna tolerancia.

Uso:
    python3 openspec/changes/clasificacion-orientada-a-video/verificar_orden_temporal.py
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from equinos import caracteristicas, rutas  # noqa: E402

# Clips a verificar: primero, medio y último del orden persistido.
POSICIONES_CLIP = (0, 70, -1)
POSICIONES_FRAME = (0, 1, 7, 14, 15)


def main():
    import numpy as np
    import torch

    rutas.inicializar(verbose=False)
    base = rutas.dir_trabajo("frames_alineados")
    ruta_csv = rutas.carpeta("resultados") / "dataset_frames_alineados.csv"
    with ruta_csv.open(encoding="utf-8", errors="replace") as archivo:
        filas = list(csv.DictReader(archivo))
    por_clip: dict[str, list] = {}
    for fila in filas:
        por_clip.setdefault(fila["clip"], []).append(fila)

    orden, X = caracteristicas.cargar()
    extractor = caracteristicas._cargar_backbone(
        torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    dispositivo = next(extractor.parameters()).device

    print("=" * 78)
    print("VERIFICACIÓN DEL ORDEN TEMPORAL DEL TENSOR")
    print("=" * 78)
    print("Criterio: el vector del archivo de `orden = k` debe parecerse a la fila k del tensor")
    print("MÁS que a las otras 15. Sin tolerancia absoluta.\n")

    fallos = 0
    for posicion in POSICIONES_CLIP:
        clip = orden[posicion]
        indice = orden.index(clip)
        frames = sorted(por_clip[clip], key=lambda f: int(f["orden"]))
        if [int(f["orden"]) for f in frames] != list(range(16)):
            print(f"  {clip}: el CSV no declara los órdenes 0..15")
            fallos += 1
            continue

        print(f"  {clip}")
        for k in POSICIONES_FRAME:
            entrada = caracteristicas._tensor_del_clip(
                [base / frames[k]["ruta_frame"]], dispositivo)
            with torch.no_grad():
                vector = extractor(entrada).flatten(1)[0].cpu().numpy()

            distancias = np.linalg.norm(X[indice] - vector, axis=1)
            mas_parecido = int(np.argmin(distancias))
            propia = distancias[k]
            ajena = float(np.min(np.delete(distancias, k)))
            ok = mas_parecido == k
            fallos += not ok
            print(f"    orden={k:<3} {frames[k]['ruta_frame'].split('/')[-1]:<15}"
                  f"más parecido a la fila {mas_parecido:<3}"
                  f"d(propia)={propia:7.4f}  d(mejor ajena)={ajena:7.4f}  "
                  f"{'OK' if ok else 'FALLA'}")

    print("\n" + "=" * 78)
    if fallos:
        print(f"FALLARON {fallos} verificaciones: el tensor NO está en el orden del CSV.")
        print("Todo resultado sobre información temporal queda invalidado hasta corregirlo.")
        sys.exit(1)
    print("El tensor está en el orden que declara el CSV.")
    print("La prueba del orden del bloque 3 mide lo que dice medir.")


if __name__ == "__main__":
    main()
