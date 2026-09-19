"""La pasada cara: pose de **todos** los frames de cada clip, persistida una sola vez.

Existe porque la etapa 02 tiraba lo que más costaba. Corría DeepLabCut sobre el clip entero y
después se quedaba únicamente con las filas de los 16 índices que `np.linspace` había fijado de
antemano, sin mirar cuáles tenían keypoints. El resto de la inferencia —el 90 % de los frames,
las cuatro horas y media de GPU— se descartaba sin escribirse en ningún lado. Con `pose_completa.csv`
teniendo exactamente 16 órdenes por clip, cambiar la selección de frames obligaba a pagar la GPU
de nuevo.

Por eso esta etapa hace una cosa sola y la deja escrita: **la pose de cada frame de cada clip.**
Todo lo que viene después —elegir los 16 con `equinos.secuencia`, alinear, extraer características,
particionar, entrenar— se recalcula en minutos sobre este archivo. Si mañana se cambia el criterio
de selección, el costo es de minutos y no de horas.

Es reanudable: cada clip terminado se anota en el progreso, y volver a lanzarla retoma donde
quedó. Una corrida de 4,5 h que hay que reiniciar entera ante el primer error no es una corrida,
es una apuesta.

Uso:

    python3 -m equinos.reproceso                 # procesa lo que falte
    python3 -m equinos.reproceso --limite 10     # una prueba corta
"""

from __future__ import annotations

import contextlib
import csv
import io
import json
import shutil
import time
from pathlib import Path

from equinos import rutas

# La pose de todos los frames. NO pisa `pose_completa.csv`, que es el muestreo de 16 de la corrida
# vieja: mientras no se re-genere el dataset, ese archivo sigue siendo el que describe al modelo
# desplegado, y pisarlo dejaría a la interfaz informando sobre datos que ya no existen.
ARCHIVO_POSE = "pose_todos_los_frames.csv"
ARCHIVO_PROGRESO = "pose_todos_los_frames_progreso.jsonl"

COLUMNAS = ("clip", "orden", "bodypart", "x", "y", "confianza")

CLASES = ("Disgustado", "Interesado", "Sereno")


# DeepLabCut deja junto al video de entrada una copia ETIQUETADA, con el mismo `.mp4` y el nombre
# del modelo pegado al final. Una corrida vieja dejó una de esas en la carpeta de clips, y sin
# este filtro entra al dataset como si fuera un clip más: mismo caballo, misma clase, pero con los
# keypoints dibujados encima de la imagen que después ve el backbone. No falla en ningún lado.
SUFIJOS_DE_DEEPLABCUT = ("superanimal", "_labeled", "_before_adapt", "_full")


def es_artefacto(nombre: str) -> bool:
    """Si el archivo es una salida de DeepLabCut y no un clip del dataset."""
    return any(marca in nombre for marca in SUFIJOS_DE_DEEPLABCUT)


def clips_en_disco() -> list[tuple[str, Path]]:
    """`(clase, ruta)` de cada clip, en orden estable.

    El orden no puede depender de lo que devuelva el sistema de archivos: una corrida reanudada
    tiene que recorrer lo mismo que la que se interrumpió.
    """
    encontrados = []
    for clase in CLASES:
        carpeta = rutas.carpeta("clips") / clase
        if not carpeta.exists():
            continue
        encontrados.extend((clase, p) for p in sorted(carpeta.glob("*.mp4"))
                           if not es_artefacto(p.name))
    return encontrados


def _progreso(ruta: Path) -> dict[str, dict]:
    if not ruta.exists():
        return {}
    hecho = {}
    for linea in ruta.read_text(encoding="utf-8").splitlines():
        if not linea.strip():
            continue
        try:
            registro = json.loads(linea)
        except json.JSONDecodeError:
            continue          # una línea a medio escribir por un corte: se reprocesa el clip
        hecho[registro["clip"]] = registro
    return hecho


def _sanear(ruta_pose: Path, completados: set[str], verbose: bool = True) -> None:
    """Descarta las filas de clips que no llegaron a anotarse como terminados.

    Un corte —Ctrl-C, un OOM, la notebook que se suspende— deja las filas del clip en curso ya
    escritas pero sin su registro de progreso. Al reanudar, ese clip se procesa de nuevo y sus
    filas quedarían DUPLICADAS en el CSV: la pose del mismo frame dos veces, que aguas abajo se
    convierte en un clip con el doble de ordenes y una seleccion de frames sin sentido. No falla
    en ningun lado, que es exactamente el modo de error que este proyecto ya conoce.
    """
    if not ruta_pose.exists():
        return

    import tempfile

    descartadas = 0
    with ruta_pose.open(encoding="utf-8", newline="") as entrada:
        lector = csv.reader(entrada)
        try:
            cabecera = next(lector)
        except StopIteration:
            return
        temporal = tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", newline="", delete=False, dir=str(ruta_pose.parent))
        escritor = csv.writer(temporal)
        escritor.writerow(cabecera)
        for fila in lector:
            if fila and fila[0] in completados:
                escritor.writerow(fila)
            else:
                descartadas += 1
        temporal.close()

    if descartadas:
        Path(temporal.name).replace(ruta_pose)
        if verbose:
            print(f"corrida anterior interrumpida: se descartaron {descartadas} filas de un clip "
                  f"a medio escribir", flush=True)
    else:
        Path(temporal.name).unlink(missing_ok=True)


def _limpiar_auxiliares(video: Path) -> None:
    """DeepLabCut deja un .h5, un .pickle, un .json y un video etiquetado por cada entrada.

    El video etiquetado pesa casi como el original. Sin limpiarlos, procesar 397 clips duplica
    el dataset en disco.
    """
    for archivo in video.parent.glob(f"{video.stem}_*"):
        if archivo.name != video.name:
            archivo.unlink(missing_ok=True)


def procesar(limite: int | None = None, verbose: bool = True) -> dict:
    """Corre la pose sobre cada clip pendiente y va escribiendo sus filas.

    Devuelve el resumen de la corrida. Los clips que fallan quedan anotados con su motivo: para
    el informe importa tanto cuántos fallan como por qué, y un fallo silenciado se vuelve un
    clip ausente que nadie sabe explicar.
    """
    import cv2
    import deeplabcut

    from equinos import pose as sel_pose

    salida = rutas.carpeta("resultados")
    ruta_pose = salida / ARCHIVO_POSE
    ruta_progreso = salida / ARCHIVO_PROGRESO
    trabajo = rutas.dir_trabajo("pose_completa")

    hecho = _progreso(ruta_progreso)
    _sanear(ruta_pose, set(hecho), verbose=verbose)
    todos = clips_en_disco()
    pendientes = [(c, p) for c, p in todos if p.name not in hecho]
    if limite is not None:
        pendientes = pendientes[:limite]

    if verbose:
        print(f"clips en disco: {len(todos)} | ya procesados: {len(hecho)} | "
              f"a procesar ahora: {len(pendientes)}", flush=True)

    nuevo = not ruta_pose.exists()
    archivo = ruta_pose.open("a", encoding="utf-8", newline="")
    escritor = csv.writer(archivo)
    if nuevo:
        escritor.writerow(COLUMNAS)

    t0 = time.time()
    for i, (clase, origen) in enumerate(pendientes, 1):
        registro = {"clip": origen.name, "clase": clase}
        destino = trabajo / origen.name
        try:
            captura = cv2.VideoCapture(str(origen))
            total = int(captura.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = captura.get(cv2.CAP_PROP_FPS) or 30.0
            captura.release()
            registro.update(frames=total, fps=round(float(fps), 3))

            if total <= 0:
                registro["motivo"] = "ARCHIVO_ILEGIBLE"
                raise _Saltear

            shutil.copy2(origen, destino)
            with contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(io.StringIO()):
                resultado = deeplabcut.video_inference_superanimal(
                    [str(destino)], superanimal_name="superanimal_quadruped",
                    model_name="hrnet_w32", detector_name="fasterrcnn_resnet50_fpn_v2")

            if not isinstance(resultado, dict) or not resultado:
                registro["motivo"] = "POSE_VACIA"
                raise _Saltear

            tabla = next(iter(resultado.values()))
            filas = sel_pose.a_formato_largo(
                tabla, list(enumerate(range(total))), origen.name)
            if not filas:
                registro["motivo"] = "SIN_FILAS"
                raise _Saltear

            for f in filas:
                escritor.writerow((f["clip"], f["orden"], f["bodypart"],
                                   _redondear(f["x"], 2), _redondear(f["y"], 2),
                                   _redondear(f["confianza"], 4)))
            archivo.flush()
            registro.update(motivo="OK", filas=len(filas),
                            ordenes=len({int(f["orden"]) for f in filas}))

        except _Saltear:
            pass
        except Exception as error:                        # noqa: BLE001
            registro["motivo"] = f"ERROR_{type(error).__name__}"
            registro["detalle"] = str(error)[:300]
        finally:
            _limpiar_auxiliares(destino)
            destino.unlink(missing_ok=True)

        with ruta_progreso.open("a", encoding="utf-8") as p:
            p.write(json.dumps(registro, ensure_ascii=False) + "\n")

        if verbose:
            transcurrido = time.time() - t0
            resta = transcurrido / i * (len(pendientes) - i)
            print(f"[{i}/{len(pendientes)}] {origen.name}: {registro['motivo']} "
                  f"({registro.get('ordenes', 0)} órdenes) — "
                  f"{transcurrido/60:.0f} min transcurridos, ~{resta/60:.0f} min restantes",
                  flush=True)

    archivo.close()

    final = _progreso(ruta_progreso)
    motivos: dict[str, int] = {}
    for r in final.values():
        motivos[r.get("motivo", "?")] = motivos.get(r.get("motivo", "?"), 0) + 1
    resumen = {"clips_en_disco": len(todos), "procesados": len(final), "motivos": motivos,
               "pose": str(ruta_pose), "progreso": str(ruta_progreso)}
    if verbose:
        print("\n" + "=" * 60)
        print(f"clips procesados: {len(final)} de {len(todos)}")
        for motivo, cuantos in sorted(motivos.items(), key=lambda kv: -kv[1]):
            print(f"  {motivo:<22} {cuantos:>4}")
        print(f"\npose escrita en: {ruta_pose}")
    return resumen


class _Saltear(Exception):
    """Corta el procesamiento de un clip sin que cuente como error inesperado."""


def _redondear(valor, decimales: int) -> str:
    try:
        f = float(valor)
    except (TypeError, ValueError):
        return ""
    return "" if f != f else f"{f:.{decimales}f}"          # f != f detecta NaN


def main() -> None:
    import argparse

    analizador = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    analizador.add_argument("--limite", type=int, default=None,
                            help="procesa a lo sumo esta cantidad de clips pendientes")
    argumentos = analizador.parse_args()

    rutas.inicializar(verbose=False)
    procesar(limite=argumentos.limite)


if __name__ == "__main__":
    main()
