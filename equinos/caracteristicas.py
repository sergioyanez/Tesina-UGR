"""Características por frame extraídas con el backbone congelado, y su caché.

El backbone es un extractor **congelado**: no se entrena, así que para un frame dado siempre
devuelve el mismo vector. Eso es lo que vuelve legítimo cachearlo, y lo que hace que comparar
variantes de cabeza temporal cueste segundos en vez de minutos — son 141 × 16 vectores que no
cambian entre configuraciones.

**Por qué el backbone va congelado.** Con 141 clips íntegros de 73 ejemplares, descongelarlo pone
los parámetros entrenables en el orden de 10⁷ y el sobreajuste es inmediato. Congelado, lo
entrenable queda en la cabeza, del orden de 10⁴–10⁵. No es una elección de arquitectura por
mérito: es lo que 141 muestras permiten sostener.

**Consecuencia que hay que tener presente: con las características cacheadas NO se puede
aumentar.** La aumentación transforma los píxeles, así que cambiaría el vector y obligaría a
volver a pasar el backbone en cada época. Para las líneas de base y para comparar cabezas la
caché es lo correcto —son comparaciones entre sí, sobre exactamente la misma entrada—. Si el
entrenamiento final va a usar aumentación, ese paso tiene que re-extraer, y hay que decirlo en
vez de comparar un modelo aumentado contra cabezas medidas sin aumentar.

La caché se invalida sola si cambia lo que la determina: el backbone, la geometría del recorte o
el conjunto de clips. Reanudar con una caché de otra corrida produce números que no corresponden
a ningún dataset.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from equinos import particion, rutas

# Backbone y capa de salida. ResNet-18 preentrenada en ImageNet, sin su clasificador: la salida
# del pooling adaptativo son 512 dimensiones por frame, y el pooling adaptativo es lo que permite
# que el recorte no cuadrado (224x343) entre sin reescalarlo ni deformarlo.
BACKBONE = "resnet18"
PESOS = "IMAGENET1K_V1"
DIMENSIONES = 512

# Normalización de ImageNet. Tiene que ser la misma con la que se entrenaron los pesos, o las
# características salen de una distribución distinta de la que el backbone espera.
MEDIA_IMAGENET = (0.485, 0.456, 0.406)
DESVIO_IMAGENET = (0.229, 0.224, 0.225)

# De dónde salen los recortes y su índice. Van juntos a propósito: leer el CSV de una corrida y
# los JPEG de otra produce vectores de frames que no son los que el CSV dice, sin que nada falle.
# `dataset_frames_alineados.csv` y `frames_alineados/` son los de la corrida vieja —muestreo de 16
# posiciones fijas— y se conservan para poder reproducir sus números.
ARCHIVO_FRAMES = "dataset_frames_alineados_v2.csv"
CARPETA_FRAMES = "frames_alineados_v2"

ARCHIVO_CACHE = "caracteristicas_backbone_v2.npz"
ARCHIVO_HUELLA = "caracteristicas_backbone_v2.json"

ARCHIVO_AUMENTADAS = "caracteristicas_aumentadas.npz"
ARCHIVO_AUMENTADAS_HUELLA = "caracteristicas_aumentadas.json"

# Rangos de la aumentación. Salen de la propuesta de la rama `hector`, que es la parte de ese
# trabajo que sí vale portar: los parámetros se sortean UNA VEZ POR CLIP y se aplican idénticos a
# sus 16 frames. Aplicarlos por frame destruiría la continuidad temporal del gesto, que es
# justamente la señal que el modelo secuencial tiene que aprovechar.
ROTACION_GRADOS = 10.0
BRILLO = (0.85, 1.15)
CONTRASTE = (0.85, 1.15)
PROB_ESPEJO = 0.5


def huella(clips: list[dict]) -> str:
    """Identifica de qué corrida son estas características.

    Entra todo lo que las determina: el backbone, sus pesos, y el conjunto exacto de clips con su
    orden. NO entra la geometría del recorte de forma explícita porque no hace falta — si el
    encuadre cambia, los JPEG cambian y las características también, pero el conjunto de clips
    podría ser idéntico. Por eso se guarda además la fecha de modificación del CSV del dataset,
    que sí se reescribe cuando se re-derivan los recortes.
    """
    csv_dataset = rutas.carpeta("resultados") / ARCHIVO_FRAMES
    partes = [BACKBONE, PESOS, str(DIMENSIONES),
              str(int(csv_dataset.stat().st_mtime)) if csv_dataset.exists() else "sin-csv"]
    partes += [c["clip"] for c in clips]
    return hashlib.sha256("|".join(partes).encode("utf-8")).hexdigest()[:16]


def _cargar_backbone(dispositivo):
    import torch
    from torchvision import models

    pesos = getattr(models.ResNet18_Weights, PESOS)
    red = models.resnet18(weights=pesos)
    extractor = torch.nn.Sequential(*list(red.children())[:-1])   # sin la capa clasificadora
    for parametro in extractor.parameters():
        parametro.requires_grad = False
    return extractor.eval().to(dispositivo)


def _tensor_del_clip(rutas_frames, dispositivo):
    import numpy as np
    import torch
    from PIL import Image

    imagenes = []
    for ruta in rutas_frames:
        with Image.open(ruta) as imagen:
            arreglo = np.asarray(imagen.convert("RGB"), dtype=np.float32) / 255.0
        imagenes.append(arreglo)
    lote = torch.from_numpy(np.stack(imagenes)).permute(0, 3, 1, 2)   # (T, C, H, W)
    media = torch.tensor(MEDIA_IMAGENET).view(1, 3, 1, 1)
    desvio = torch.tensor(DESVIO_IMAGENET).view(1, 3, 1, 1)
    return ((lote - media) / desvio).to(dispositivo)


def extraer(clips: list[dict] | None = None, lote: int = 8, verbose: bool = True):
    """Extrae y cachea las características de todos los frames de cada clip.

    Devuelve `(orden_de_clips, arreglo)` con forma `(n_clips, 16, 512)`.
    """
    import numpy as np
    import torch

    clips = clips if clips is not None else particion.cargar()
    marca = huella(clips)

    destino = rutas.carpeta("resultados") / ARCHIVO_CACHE
    metadatos = rutas.carpeta("resultados") / ARCHIVO_HUELLA
    if destino.exists() and metadatos.exists():
        previo = json.loads(metadatos.read_text(encoding="utf-8"))
        if previo.get("huella") == marca:
            if verbose:
                print(f"Características ya extraídas y vigentes ({destino.name}).")
            datos = np.load(destino, allow_pickle=True)
            return list(previo["clips"]), datos["caracteristicas"]
        if verbose:
            print("La caché de características corresponde a otra corrida; se descarta.")

    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = _cargar_backbone(dispositivo)
    base = rutas.dir_trabajo(CARPETA_FRAMES)

    # Rutas de los frames de cada clip, en el orden en que el modelo los va a consumir.
    import csv as _csv
    ruta_csv = rutas.carpeta("resultados") / ARCHIVO_FRAMES
    with ruta_csv.open(encoding="utf-8", errors="replace") as archivo:
        filas = [f for f in _csv.DictReader(archivo)]
    por_clip: dict[str, list] = {}
    for fila in filas:
        por_clip.setdefault(fila["clip"], []).append(fila)

    orden_clips, vectores = [], []
    if verbose:
        print(f"Extrayendo con {BACKBONE}/{PESOS} congelado en {dispositivo} — "
              f"{len(clips)} clips")
    for indice, clip in enumerate(clips):
        frames = sorted(por_clip[clip["clip"]], key=lambda f: int(f["orden"]))
        rutas_frames = [base / f["ruta_frame"] for f in frames]
        entrada = _tensor_del_clip(rutas_frames, dispositivo)
        with torch.no_grad():
            salida = extractor(entrada).flatten(1)
        orden_clips.append(clip["clip"])
        vectores.append(salida.cpu().numpy().astype(np.float32))
        if verbose and (indice + 1) % 40 == 0:
            print(f"  {indice + 1}/{len(clips)}")

    arreglo = np.empty(len(vectores), dtype=object)
    arreglo[:] = vectores
    np.savez_compressed(destino, caracteristicas=arreglo)
    metadatos.write_text(json.dumps(
        {"huella": marca, "backbone": BACKBONE, "pesos": PESOS,
         "dimensiones": DIMENSIONES, "forma": "variable", "clips": orden_clips},
        ensure_ascii=False, indent=1), encoding="utf-8")
    if verbose:
        print(f"Guardadas en {destino}  forma {arreglo.shape}")
    return orden_clips, arreglo


def _transformar_clip(lote, generador):
    """Aumentación TEMPORALMENTE COHERENTE: un sorteo por clip, aplicado a sus 16 frames.

    `lote` llega ya normalizado, con forma (T, C, H, W). Las transformaciones geométricas y
    fotométricas se aplican sobre el tensor normalizado; para brillo y contraste eso equivale a
    un reescalado afín de la señal, que es lo que se quiere.
    """
    import torch
    from torchvision.transforms import functional as TF

    if torch.rand((), generator=generador).item() < PROB_ESPEJO:
        lote = TF.hflip(lote)

    angulo = (torch.rand((), generator=generador).item() * 2 - 1) * ROTACION_GRADOS
    if abs(angulo) > 1e-6:
        lote = TF.rotate(lote, angulo)

    brillo = BRILLO[0] + torch.rand((), generator=generador).item() * (BRILLO[1] - BRILLO[0])
    contraste = CONTRASTE[0] + torch.rand((), generator=generador).item() * (CONTRASTE[1] - CONTRASTE[0])
    media = lote.mean(dim=(1, 2, 3), keepdim=True)
    return (lote - media) * contraste + media * brillo


def extraer_aumentadas(vistas: int = 20, semilla: int = 42, clips=None, verbose: bool = True):
    """Genera `vistas` versiones aumentadas de cada clip y extrae sus características.

    Se pre-generan y se cachean en vez de aumentar sobre la marcha porque el backbone está
    congelado: una pasada completa cuesta ~11 s, así que 20 vistas son ~4 minutos UNA vez, contra
    repetir esa pasada en cada época. Y quedan reproducibles, que sobre la marcha no quedarían.

    Devuelve `(orden_de_clips, arreglo)` con forma `(n_clips, vistas, 16, 512)`.
    """
    import numpy as np
    import torch

    clips = clips if clips is not None else particion.cargar()
    marca = f"{huella(clips)}-aum{vistas}-s{semilla}"
    destino = rutas.carpeta("resultados") / ARCHIVO_AUMENTADAS
    metadatos = rutas.carpeta("resultados") / ARCHIVO_AUMENTADAS_HUELLA
    if destino.exists() and metadatos.exists():
        previo = json.loads(metadatos.read_text(encoding="utf-8"))
        if previo.get("huella") == marca:
            if verbose:
                print(f"Características aumentadas ya vigentes ({destino.name}).")
            return list(previo["clips"]), np.load(destino, allow_pickle=True)["caracteristicas"]

    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = _cargar_backbone(dispositivo)
    base = rutas.dir_trabajo(CARPETA_FRAMES)
    generador = torch.Generator().manual_seed(semilla)

    import csv as _csv
    with (rutas.carpeta("resultados") / ARCHIVO_FRAMES).open(
            encoding="utf-8", errors="replace") as archivo:
        filas = list(_csv.DictReader(archivo))
    por_clip: dict[str, list] = {}
    for fila in filas:
        por_clip.setdefault(fila["clip"], []).append(fila)

    if verbose:
        print(f"Extrayendo {vistas} vistas aumentadas por clip en {dispositivo} "
              f"({len(clips)} clips)")
    orden_clips, todas = [], []
    for indice, clip in enumerate(clips):
        frames = sorted(por_clip[clip["clip"]], key=lambda f: int(f["orden"]))
        entrada = _tensor_del_clip([base / f["ruta_frame"] for f in frames], dispositivo)
        del_clip = []
        for _ in range(vistas):
            aumentado = _transformar_clip(entrada.cpu(), generador).to(dispositivo)
            with torch.no_grad():
                del_clip.append(extractor(aumentado).flatten(1).cpu().numpy().astype(np.float32))
        orden_clips.append(clip["clip"])
        todas.append(np.stack(del_clip))
        if verbose and (indice + 1) % 40 == 0:
            print(f"  {indice + 1}/{len(clips)}")

    arreglo = np.empty(len(todas), dtype=object)
    arreglo[:] = todas
    np.savez_compressed(destino, caracteristicas=arreglo)
    metadatos.write_text(json.dumps(
        {"huella": marca, "vistas": vistas, "semilla": semilla,
         "forma": "variable", "clips": orden_clips},
        ensure_ascii=False, indent=1), encoding="utf-8")
    if verbose:
        print(f"Guardadas en {destino}  forma {arreglo.shape}")
    return orden_clips, arreglo


def cargar_aumentadas():
    import numpy as np
    destino = rutas.carpeta("resultados") / ARCHIVO_AUMENTADAS
    metadatos = rutas.carpeta("resultados") / ARCHIVO_AUMENTADAS_HUELLA
    if not destino.exists():
        raise FileNotFoundError(f"No existe {destino}. Generarla con `extraer_aumentadas()`.")
    previo = json.loads(metadatos.read_text(encoding="utf-8"))
    return list(previo["clips"]), np.load(destino, allow_pickle=True)["caracteristicas"]


def cargar():
    """Lee la caché, verificando que corresponda al conjunto de clips vigente."""
    import numpy as np

    destino = rutas.carpeta("resultados") / ARCHIVO_CACHE
    metadatos = rutas.carpeta("resultados") / ARCHIVO_HUELLA
    if not destino.exists() or not metadatos.exists():
        raise FileNotFoundError(
            f"No existe {destino}. Generarla con `caracteristicas.extraer()`.")
    previo = json.loads(metadatos.read_text(encoding="utf-8"))
    esperada = huella(particion.cargar())
    if previo.get("huella") != esperada:
        raise ValueError(
            "La caché de características no corresponde al conjunto de clips vigente. "
            "Volver a extraer: usarla igual daría números que no son de ningún dataset.")
    return list(previo["clips"]), np.load(destino, allow_pickle=True)["caracteristicas"]
