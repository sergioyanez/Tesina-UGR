"""Verificación del entorno de ejecución, en Colab y en la máquina local.

El pipeline corre en dos entornos con formas distintas de conseguir sus dependencias:

- **Colab**: la máquina es efímera, así que cada sesión instala todo desde la celda de arranque.
- **Local**: el entorno está construido de antemano y pinneado en `requirements-local.txt`. Una
  celda que instale ahí es peor que inútil: `deeplabcut` fija `numpy<2` y bajaría el numpy del
  intérprete que se le indique, rompiendo lo que hubiera alrededor.

Este módulo es lo que va en lugar de la instalación cuando no hay que instalar: **verifica**. Un
entorno incompleto tiene que fallar en el arranque de la etapa y no cuarenta minutos después, con
un traceback de otra cosa —que fue exactamente lo que pasó con `numpy.dtype size changed`.

Deliberadamente sin dependencias fuera de la biblioteca estándar, y sin importar numpy, pandas ni
torch al importarse: se usa `importlib.metadata`, que lee las versiones instaladas del disco sin
cargar los módulos. Importar este archivo no puede tener efectos sobre el stack que verifica.
"""

from __future__ import annotations

import platform
import sys
from importlib import metadata

from equinos import rutas

# ============================================================
# REQUISITOS
# ============================================================
# Se declaran RANGOS, no las versiones exactas: los pines exactos viven en `requirements-local.txt`
# y son el entorno que se verificó, mientras que acá está lo que el pipeline realmente necesita
# para no romperse. Verificar contra el pin exacto haría fallar entornos que funcionan
# perfectamente, y un chequeo que grita sin motivo se termina ignorando.
#
# La clave es el nombre de DISTRIBUCIÓN (el de pip), que no siempre es el de import:
# `opencv-python` se importa como `cv2`.
#
# El tope de cada rango no es prudencia: es una incompatibilidad conocida.
#   numpy  <2      lo fija deeplabcut, y arrastra a todo lo compilado contra numpy
#   scipy  <1.14   es la última serie que convive con numpy 1.x
#   matplotlib <3.9 lo fija deeplabcut

REQUISITOS: dict[str, tuple[str | None, str | None]] = {
    "numpy": ("1.18", "2.0"),
    "scipy": ("1.9", "1.14"),
    "pandas": ("2.2", "3.0"),
    "matplotlib": ("3.3", "3.9"),
    "opencv-python": ("4.5", None),
    "tqdm": (None, None),
    "scikit-learn": ("1.0", None),
    "torch": ("2.0", None),
    "torchvision": (None, None),
    "ultralytics": ("8.0", None),
    "deeplabcut": ("3.0", None),
}

# Qué necesita cada etapa. Verificar de más obliga a instalar de más: la etapa 01 sólo corta
# videos y no tiene por qué exigir DeepLabCut para arrancar.
PILAS: dict[str, tuple[str, ...]] = {
    "basica": ("numpy", "pandas", "opencv-python", "matplotlib", "tqdm", "scikit-learn"),
    "vision": (
        "numpy", "scipy", "pandas", "opencv-python", "matplotlib", "tqdm",
        "torch", "torchvision", "ultralytics", "deeplabcut",
    ),
    # La etapa 03 consume los frames ya recortados y las características del backbone congelado:
    # NO necesita detector ni modelo de pose. Es la única etapa que no arrastra la restricción de
    # `numpy<2` que impone DeepLabCut, y exigirle esas dependencias obligaría a instalar de más.
    "modelado": ("numpy", "pandas", "matplotlib", "scikit-learn", "torch", "torchvision"),
}

INSTRUCCION_LOCAL = (
    "Entorno local incompleto. Reconstruilo desde el repositorio:\n"
    "  uv venv --python 3.10 ~/.venvs/tesina-equinos\n"
    "  VIRTUAL_ENV=~/.venvs/tesina-equinos uv pip install -r requirements-local.txt \\\n"
    "      --extra-index-url https://download.pytorch.org/whl/cu121\n"
    "Y asegurate de que el notebook esté corriendo con el kernel 'Tesina equinos (GPU local)'."
)


# ============================================================
# COMPARACIÓN DE VERSIONES
# ============================================================


def _clave(version: str) -> tuple[int, ...]:
    """Versión como tupla de enteros comparable, ignorando sufijos como `+cu121` o `rc1`.

    No pretende implementar PEP 440: sólo se usa para comparar contra los topes de REQUISITOS,
    que son números de serie. `2.5.1+cu121` y `2.5.1` tienen que dar lo mismo, que es justo lo
    que un `str` comparado lexicográficamente no hace.
    """
    numeros = []
    for parte in version.split("+")[0].split("."):
        digitos = ""
        for caracter in parte:
            if not caracter.isdigit():
                break
            digitos += caracter
        if not digitos:
            break
        numeros.append(int(digitos))
    return tuple(numeros) or (0,)


def _en_rango(version: str, minimo: str | None, tope: str | None) -> bool:
    actual = _clave(version)
    if minimo is not None and actual < _clave(minimo):
        return False
    if tope is not None and actual >= _clave(tope):
        return False
    return True


def version_de(paquete: str) -> str | None:
    """Versión instalada de una distribución, o None si no está. No importa el módulo."""
    try:
        return metadata.version(paquete)
    except metadata.PackageNotFoundError:
        return None


# ============================================================
# VERIFICACIÓN
# ============================================================


def verificar(pila: str = "basica", gpu: bool = False, verbose: bool = True) -> dict[str, str]:
    """Comprueba que el entorno tenga lo que la etapa necesita, y devuelve qué versiones hay.

    Falla con `SystemExit` si falta un paquete o si su versión está fuera del rango compatible.
    Es a propósito: una etapa que arranca con el stack roto consume una hora de cómputo para
    terminar en un error que no dice qué pasó.

    `gpu=True` además informa si hay GPU y cuál. No falla si no la hay —el pipeline corre en CPU,
    sólo que muy despacio— pero lo dice fuerte, porque descubrirlo por el tiempo transcurrido es
    caro.
    """
    if pila not in PILAS:
        raise ValueError(f"Pila desconocida: {pila!r}. Opciones: {', '.join(PILAS)}")

    presentes: dict[str, str] = {}
    faltantes: list[str] = []
    incompatibles: list[str] = []

    for paquete in PILAS[pila]:
        minimo, tope = REQUISITOS[paquete]
        version = version_de(paquete)

        if version is None:
            faltantes.append(paquete)
            continue

        presentes[paquete] = version

        if not _en_rango(version, minimo, tope):
            esperado = f">={minimo}" if minimo else ""
            esperado += (", " if esperado and tope else "") + (f"<{tope}" if tope else "")
            incompatibles.append(f"{paquete} {version} (se necesita {esperado})")

    if verbose:
        print(" | ".join(f"{nombre} {v}" for nombre, v in presentes.items()))

    if gpu and verbose:
        informe_gpu()

    if faltantes or incompatibles:
        detalle = ""
        if faltantes:
            detalle += f"\nFaltan: {', '.join(faltantes)}"
        if incompatibles:
            detalle += "\nVersiones incompatibles:\n  " + "\n  ".join(incompatibles)
        instruccion = (
            "En Colab: reejecutá la celda de instalación de dependencias del Paso 0 y reiniciá "
            "el entorno."
            if rutas.en_colab()
            else INSTRUCCION_LOCAL
        )
        raise SystemExit(f"El entorno no sirve para ejecutar esta etapa.{detalle}\n\n{instruccion}")

    return presentes


def informe_gpu(verbose: bool = True) -> str | None:
    """Nombre de la GPU visible para torch, o None. Importa torch sólo si se lo llama."""
    try:
        import torch
    except ImportError:
        if verbose:
            print("GPU: torch no está instalado")
        return None

    if torch.cuda.is_available():
        nombre = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        if verbose:
            print(f"GPU: {nombre} ({vram:.1f} GB) | torch {torch.__version__}")
        return nombre

    if verbose:
        print(f"GPU: NO disponible — torch {torch.__version__} corriendo en CPU.")
        print("  El procesamiento de miles de frames va a ser MUY lento.")
        if rutas.en_colab():
            print("  Entorno de ejecución -> Cambiar tipo de entorno -> GPU")
        else:
            # La causa real que costó una sesión: el torch instalado pedía una CUDA más nueva
            # que la del driver, y el mensaje de torch queda sepultado entre warnings.
            print("  Verificá que el torch instalado corresponda a la CUDA del driver:")
            print("    nvidia-smi                     -> CUDA Version del driver")
            print(f"    torch.version.cuda = {getattr(torch.version, 'cuda', '?')}   -> la que pide torch")
            print("  Un torch +cu130 contra un driver CUDA 12.2 deja la GPU invisible.")
    return None


def informe(verbose: bool = True) -> dict[str, str]:
    """Resumen del entorno para dejar asentado en el notebook ejecutado.

    Queda guardado en la copia de `ejecuciones/`, que es lo que después permite responder "¿con
    qué versiones se obtuvo este número?" sin tener que confiar en la memoria de nadie.
    """
    datos = {
        "entorno": "Colab" if rutas.en_colab() else "local",
        "python": platform.python_version(),
        "ejecutable": sys.executable,
        "sistema": f"{platform.system()} {platform.release()}",
    }
    for paquete in REQUISITOS:
        version = version_de(paquete)
        if version:
            datos[paquete] = version

    if verbose:
        ancho = max(len(k) for k in datos)
        for clave, valor in datos.items():
            print(f"{clave.ljust(ancho)} : {valor}")
        informe_gpu()

    return datos
