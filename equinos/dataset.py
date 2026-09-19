"""Construye el conjunto de frames alineados a partir de la pose completa.

Es la etapa barata: consume `pose_todos_los_frames.csv` —lo que costó las horas de GPU— y produce
los recortes que después lee el backbone. Decide QUÉ frames entran llamando a
`equinos.alineado.preparar_secuencia`, que es la misma función que usa la interfaz: por eso el
dataset y la inferencia no pueden divergir aunque mañana cambie el criterio.

**El descarte se clasifica, no se cuenta.** Un informe que diga "se perdieron 139 clips" no sirve
para arreglar nada; uno que diga cuántos se perdieron por cada motivo dice dónde está el cuello de
botella. Por eso cada clip rechazado queda anotado con su código.

Uso:

    python3 -m equinos.dataset            # reconstruye todo
    python3 -m equinos.dataset --limite 5 # una prueba corta
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from equinos import alineado, rutas, secuencia

# El dataset nuevo. NO pisa `dataset_frames_alineados.csv`: mientras la app siga sirviendo al
# modelo viejo, ese archivo describe lo que ese modelo comió, y pisarlo dejaría a la interfaz
# informando una confiabilidad derivada de datos que ya no existen.
ARCHIVO_DATASET = "dataset_frames_alineados_v2.csv"
ARCHIVO_RECHAZOS = "dataset_rechazos_v2.json"
CARPETA_FRAMES = "frames_alineados_v2"

# La pose de los frames ELEGIDOS, con `orden` renumerado a la posición en la secuencia (0..15).
#
# No es una comodidad: `MultimodalEquineDataset` busca la geometría con `pose_por_clip[clip][orden]`
# recorriendo `range(seq_len)`, o sea que interpreta `orden` como posición en la secuencia. El
# archivo de la pasada de GPU usa `orden` = índice de frame del video, así que pasárselo directo
# hace que el canal geométrico lea los frames 0..15 del video mientras el visual usa los que la
# ventana eligió. No falla: devuelve la geometría de OTROS frames y el modelo entrena con los dos
# canales mirando momentos distintos.
ARCHIVO_POSE_SELECCIONADA = "pose_seleccionada_v2.csv"

COLUMNAS = ("clase", "id_caballo", "video_origen", "clip", "frame", "ruta_frame", "orden",
            "alineado", "origen_alineacion", "span_seg", "validos", "fps")

# Calidad del JPEG intermedio. El backbone ve la imagen DESPUÉS de este viaje, así que el valor
# es parte de la distribución de entrada y no un detalle de almacenamiento: la inferencia replica
# el mismo viaje en memoria. Es el que venía usando la etapa 02.
CALIDAD_JPEG = 95


def _inventario() -> dict[str, dict]:
    ruta = rutas.carpeta("resultados") / "inventario_clips.csv"
    if not ruta.exists():
        raise FileNotFoundError(f"No existe {ruta}. La produce la etapa 01.")
    with ruta.open(encoding="utf-8", errors="replace", newline="") as archivo:
        return {f["clip"]: f for f in csv.DictReader(archivo)}


def construir(limite: int | None = None, verbose: bool = True) -> dict:
    """Elige, recorta y escribe los frames de cada clip. Devuelve el informe de aprovechamiento."""
    import cv2

    inventario = _inventario()
    p, geo = alineado.parametros(), alineado.geometria()
    destino_geo = alineado.destino_canonico(p)

    ruta_pose = rutas.carpeta("resultados") / "pose_todos_los_frames.csv"
    if not ruta_pose.exists():
        raise FileNotFoundError(
            f"No existe {ruta_pose}. La produce `python3 -m equinos.reproceso`, que es la pasada "
            "de GPU: sin ella no hay pose de todos los frames y la selección no tiene de dónde "
            "elegir.")
    if verbose:
        print(f"Leyendo la pose de {ruta_pose.name}…", flush=True)
    pose = alineado.pose_por_clip(ruta_pose)

    base = rutas.dir_trabajo(CARPETA_FRAMES)
    salida = rutas.carpeta("resultados")
    clips = sorted(pose)
    if limite is not None:
        clips = clips[:limite]

    filas, filas_pose, rechazos = [], [], []
    aceptados = 0
    for i, clip in enumerate(clips, 1):
        meta = inventario.get(clip)
        if meta is None:
            rechazos.append({"clip": clip, "motivo": "SIN_INVENTARIO"})
            continue

        total, fps = int(meta["frames"]), float(meta["fps"])
        prep = alineado.preparar_secuencia(pose[clip], total=total, fps=fps, p=p, geo=geo,
                                           destino=destino_geo, minimo=secuencia.FRAMES_POR_CLIP)
        if not prep.hay_secuencia:
            rechazos.append({"clip": clip, "clase": meta["clase"], "motivo": prep.motivo,
                             "validos": prep.validos, "frames": total})
            continue

        pose_por_orden: dict[int, list[dict]] = {}
        for fila in pose[clip]:
            pose_por_orden.setdefault(int(fila["orden"]), []).append(fila)

        captura = cv2.VideoCapture(str(rutas.resolver(meta["ruta_clip"])))
        carpeta = base / meta["clase"] / Path(clip).stem
        carpeta.mkdir(parents=True, exist_ok=True)

        escritos = []
        for posicion, (orden, matriz) in enumerate(zip(prep.ordenes, prep.matrices)):
            captura.set(cv2.CAP_PROP_POS_FRAMES, int(orden))
            ok, imagen = captura.read()
            if not ok or imagen is None:
                continue
            recorte = alineado.recortar(imagen, matriz, prep.destino)
            nombre = f"{posicion:02d}_f{orden:05d}.jpg"
            cv2.imwrite(str(carpeta / nombre), recorte,
                        [int(cv2.IMWRITE_JPEG_QUALITY), CALIDAD_JPEG])
            escritos.append((posicion, orden, f"{meta['clase']}/{Path(clip).stem}/{nombre}"))
        captura.release()

        # Un frame que no se pudo leer del video deja la secuencia corta. Se descarta el clip
        # entero: una secuencia de 15 entra igual al modelo y nadie se entera.
        if len(escritos) != secuencia.FRAMES_POR_CLIP:
            rechazos.append({"clip": clip, "clase": meta["clase"], "motivo": "LECTURA_INCOMPLETA",
                             "leidos": len(escritos), "frames": total})
            continue

        for posicion, orden, _ in escritos:
            for fila in pose_por_orden.get(orden, ()):
                filas_pose.append({"clip": clip, "orden": posicion,
                                   "bodypart": fila["bodypart"], "x": fila["x"],
                                   "y": fila["y"], "confianza": fila["confianza"]})

        for posicion, orden, ruta_frame in escritos:
            filas.append({
                "clase": meta["clase"], "id_caballo": meta["id_caballo"],
                "video_origen": meta["video_origen"], "clip": clip, "frame": orden,
                "ruta_frame": ruta_frame, "orden": posicion,
                "alineado": "True", "origen_alineacion": "keypoints",
                "span_seg": round(prep.span_seg, 3), "validos": prep.validos,
                "fps": round(fps, 3),
            })
        aceptados += 1

        if verbose and i % 40 == 0:
            print(f"  {i}/{len(clips)} clips — {aceptados} aceptados", flush=True)

    ruta_csv = salida / ARCHIVO_DATASET
    with ruta_csv.open("w", encoding="utf-8", newline="") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=COLUMNAS)
        escritor.writeheader()
        escritor.writerows(filas)

    ruta_pose_sel = salida / ARCHIVO_POSE_SELECCIONADA
    with ruta_pose_sel.open("w", encoding="utf-8", newline="") as archivo:
        escritor = csv.DictWriter(
            archivo, fieldnames=("clip", "orden", "bodypart", "x", "y", "confianza"))
        escritor.writeheader()
        escritor.writerows(filas_pose)

    motivos: dict[str, int] = {}
    for r in rechazos:
        motivos[r["motivo"]] = motivos.get(r["motivo"], 0) + 1
    informe = {
        "clips_candidatos": len(clips),
        "clips_aceptados": aceptados,
        "clips_rechazados": len(rechazos),
        "porcentaje_aprovechado": round(100 * aceptados / max(1, len(clips)), 1),
        "motivos_de_rechazo": motivos,
        "frames_escritos": len(filas),
        "dataset": str(ruta_csv),
        "pose_seleccionada": str(ruta_pose_sel),
    }
    (salida / ARCHIVO_RECHAZOS).write_text(
        json.dumps({"informe": informe, "rechazos": rechazos}, ensure_ascii=False, indent=1),
        encoding="utf-8")

    if verbose:
        print("\n" + "=" * 60)
        print(f"clips candidatos  : {informe['clips_candidatos']}")
        print(f"clips aceptados   : {informe['clips_aceptados']}")
        print(f"clips rechazados  : {informe['clips_rechazados']}")
        print(f"aprovechamiento   : {informe['porcentaje_aprovechado']} %")
        print("\nmotivos de rechazo:")
        for motivo, cuantos in sorted(motivos.items(), key=lambda kv: -kv[1]):
            print(f"  {motivo:<22} {cuantos:>4}")
        print(f"\ndataset en: {ruta_csv}")
    return informe


def main() -> None:
    import argparse

    analizador = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    analizador.add_argument("--limite", type=int, default=None)
    argumentos = analizador.parse_args()

    rutas.inicializar(verbose=False)
    construir(limite=argumentos.limite)


if __name__ == "__main__":
    main()
