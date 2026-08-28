"""¿Cuántas veces se ve la oreja en los clips que van a entrenar, y por qué cuando no?

Este script no decide nada del modelo: mide una LIMITACIÓN declarada antes de entrenar, para que
si `Disgustado` no se separa, la explicación candidata esté medida de antemano y no inventada
después para justificar el resultado.

Separa dos causas que se confunden con facilidad:

  - **fuera del recorte**  — la oreja se detectó pero el encuadre la dejó afuera. Es lo que el
    change `encuadre-que-incluye-las-orejas` atacó, y lo bajó del 59,8 % al 7,3 %.
  - **no detectada**       — el modelo de pose no la encontró: oclusión, desenfoque, o la oreja
    apuntando fuera de cámara. El encuadre NO puede moverlo, porque la detección ocurre sobre el
    frame original, antes de recortar.

ADVERTENCIA DE INTERPRETACIÓN, y es la que más importa: "no detectada" NO es "no visible". El
clasificador consume píxeles, no keypoints, así que una oreja puede estar perfectamente visible en
el recorte sin que DeepLabCut la haya detectado. Los porcentajes de "dentro" son por lo tanto una
COTA INFERIOR de cuántas veces el modelo puede ver la oreja, no una medición de lo que ve.

Uso:
    python3 openspec/changes/clasificacion-orientada-a-video/disponibilidad_orejas.py
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from equinos import encuadre, rutas  # noqa: E402
from equinos.encuadre import CONF_MINIMA_KEYPOINT, CampoDeVision  # noqa: E402

# El encuadre vigente, el mismo que declara la celda de parámetros del notebook 02.
CAMPO = CampoDeVision(1.00, 0.30, 0.75)
ANCHO = 224


def _clips_integros():
    filas = list(csv.DictReader(
        (rutas.carpeta("resultados") / "dataset_frames_alineados.csv").open(encoding="utf-8")))
    por_clip = defaultdict(list)
    for f in filas:
        por_clip[f["clip"]].append(f)
    return {c: fs for c, fs in por_clip.items()
            if sum(x["alineado"] == "True" for x in fs) == 16}


def _estado(medibles, geo, clip, orden):
    m = medibles.get((clip, str(orden)))
    if not m:
        return "sin par propio"
    detectada = False
    for lado in ("left", "right"):
        base = m["orejas"].get(lado + "_earbase")
        punta = m["orejas"].get(lado + "_earend")
        if (base is None or punta is None
                or base[2] < CONF_MINIMA_KEYPOINT or punta[2] < CONF_MINIMA_KEYPOINT):
            continue
        detectada = True
        if (encuadre.dentro(encuadre.proyectar(base[:2], *m["ojo"], *m["hocico"], geo), geo)
                and encuadre.dentro(encuadre.proyectar(punta[:2], *m["ojo"], *m["hocico"], geo), geo)):
            return "dentro"
    return "fuera del recorte" if detectada else "no detectada"


def main():
    rutas.inicializar(verbose=False)
    geo = encuadre.destino(CAMPO, ANCHO)
    medibles = {(f["clip"], f["orden"]): f for f in encuadre._frames_medibles()}
    integros = _clips_integros()

    total = defaultdict(int)
    por_clase = defaultdict(lambda: defaultdict(int))
    for clip, filas in integros.items():
        clase = filas[0]["clase"]
        for fila in filas:
            e = _estado(medibles, geo, clip, fila["orden"])
            total[e] += 1
            por_clase[clase][e] += 1

    n = sum(total.values())
    print(f"Clips íntegros: {len(integros)}   ({n} frames)\n")
    orden = ("dentro", "fuera del recorte", "no detectada", "sin par propio")
    print(f'{"":<14}' + "".join(f"{k:>20}" for k in orden))
    print("-" * (14 + 20 * len(orden)))
    print(f'{"TOTAL":<14}' + "".join(f"{total[k] / n:>19.1%}" for k in orden))
    for clase in ("Disgustado", "Interesado", "Sereno"):
        d = por_clase[clase]
        m = sum(d.values())
        print(f'{clase:<14}' + "".join(f"{d[k] / m:>19.1%}" for k in orden))

    print("\nLa oreja a lo largo del clip — de sus 16 frames, ¿en cuántos se lee?")
    conteo = {c: sum(1 for f in fs if _estado(medibles, geo, c, f["orden"]) == "dentro")
              for c, fs in integros.items()}
    clase_de = {c: fs[0]["clase"] for c, fs in integros.items()}
    print(f'{"umbral":<22}{"clips":>7}{"Disgustado":>12}{"Interesado":>12}{"Sereno":>9}')
    print("-" * 62)
    for etiqueta, minimo in (("al menos 1 de 16", 1), ("al menos 8 de 16", 8),
                             ("al menos 12 de 16", 12), ("los 16", 16)):
        sel = [c for c, v in conteo.items() if v >= minimo]
        cuenta = defaultdict(int)
        for c in sel:
            cuenta[clase_de[c]] += 1
        print(f'{etiqueta:<22}{len(sel):>7}{cuenta["Disgustado"]:>12}'
              f'{cuenta["Interesado"]:>12}{cuenta["Sereno"]:>9}')

    print("\nRecordar: 'no detectada' no es 'no visible'. Ver el encabezado de este archivo.")


if __name__ == "__main__":
    main()
