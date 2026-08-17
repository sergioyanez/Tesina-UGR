"""Resolución de las rutas del proyecto, en Colab y fuera de Colab.

El pipeline corre en dos entornos con disposiciones de disco distintas:

- **Colab**: la raíz vive en Google Drive montado, para que los artefactos sobrevivan al cierre de
  la sesión. El disco local (`/content`) se usa sólo como área de trabajo rápida.
- **Local**: la raíz es un directorio del disco, y no hay nada que montar.

Los notebooks no declaran en cuál de los dos están: `inicializar()` lo detecta. Ese es el punto del
módulo — el mismo código de un notebook corre en la máquina de desarrollo (donde se puede verificar
la geometría sin GPU) y en Colab (donde se ejecuta de verdad).
"""

from __future__ import annotations

import os
from pathlib import Path

# ============================================================
# CONFIGURACIÓN
# ============================================================

# Nombre de la carpeta de proyecto dentro de MyDrive. Cambiarlo acá lo cambia en todo el pipeline.
NOMBRE_PROYECTO = "tesina_equinos"

PUNTO_MONTAJE_DRIVE = Path("/content/drive")
RAIZ_EN_DRIVE = PUNTO_MONTAJE_DRIVE / "MyDrive" / NOMBRE_PROYECTO

# Variable de entorno para sobrescribir la raíz sin tocar el código (útil en pruebas y en local).
VARIABLE_ENTORNO_RAIZ = "TESINA_EQUINOS_RAIZ"

# Subcarpetas de artefactos. Se crean al inicializar si no existen.
SUBCARPETAS = (
    "videos",       # datos crudos, una carpeta por clase
    "clips",        # clips de 5-7 s
    "resultados",   # CSV de inventarios y datasets
    "cache",        # artefactos numerosos empaquetados
    "checkpoints",  # estado de reanudación por etapa
    "ejecuciones",  # copias ejecutadas de los notebooks
    "modelos",      # pesos entrenados
    "codigo",       # copia del paquete, para el bootstrap en Colab
)

# Área de trabajo rápida. En Colab es el disco local de la sesión: efímero pero sin la latencia por
# archivo de Drive montado, que es lo que vuelve impracticable escribir miles de JPEG sueltos.
TRABAJO_EN_COLAB = Path("/content/trabajo_equinos")

# Rutas absolutas que usaban las versiones anteriores de los notebooks, y el subdirectorio de la
# raíz al que corresponden ahora. Permite que un CSV viejo, con rutas de `/content`, siga siendo
# legible en lugar de reportar 92 clips faltantes.
_PREFIJOS_HEREDADOS = {
    "/content/clips_equinos": "clips",
    "/content/frames_equinos": "frames_equinos",
    "/content/frames_alineados": "frames_alineados",
    "/content/resultados_equinos": "resultados",
    "/content/dataset_equinos_videos": "videos",
}

# Estado del módulo. Se fija una sola vez por sesión, en inicializar().
_raiz: Path | None = None


# ============================================================
# DETECCIÓN DE ENTORNO
# ============================================================


def en_colab() -> bool:
    """Indica si el código corre dentro de Google Colab."""
    try:
        import google.colab  # noqa: F401
    except ImportError:
        return False
    return True


def raiz_por_defecto() -> Path:
    """Raíz que corresponde al entorno actual, sin montar nada ni crear carpetas."""
    desde_entorno = os.environ.get(VARIABLE_ENTORNO_RAIZ)
    if desde_entorno:
        return Path(desde_entorno).expanduser().resolve()
    if en_colab():
        return RAIZ_EN_DRIVE
    # Fuera de Colab: junto al repo, nunca en /tmp (el disco raíz de la máquina de desarrollo
    # está casi lleno y los artefactos de video pesan gigabytes).
    return (Path(__file__).resolve().parent.parent / "datos_equinos").resolve()


def montar_drive(verbose: bool = True) -> None:
    """Monta Google Drive si hace falta. Sin efecto fuera de Colab."""
    if not en_colab():
        return
    if (PUNTO_MONTAJE_DRIVE / "MyDrive").exists():
        if verbose:
            print(f"Drive ya montado en {PUNTO_MONTAJE_DRIVE}")
        return
    from google.colab import drive

    drive.mount(str(PUNTO_MONTAJE_DRIVE))


# ============================================================
# INICIALIZACIÓN
# ============================================================


def inicializar(raiz: str | Path | None = None, verbose: bool = True) -> Path:
    """Deja el proyecto listo para usarse y devuelve la raíz resuelta.

    Monta Drive si corresponde, fija la raíz, verifica que se pueda escribir en ella y crea las
    subcarpetas que falten. Es idempotente: llamarla dos veces en la misma sesión no hace daño.

    Falla de inmediato si la raíz no admite escritura, en lugar de caer a una ubicación alternativa.
    Un pipeline que escribe en un lugar distinto del que cree es peor que uno que no arranca.
    """
    global _raiz

    montar_drive(verbose=verbose)

    destino = Path(raiz).expanduser() if raiz is not None else raiz_por_defecto()

    try:
        destino.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise RuntimeError(
            f"No se pudo crear la raíz de proyecto en {destino}.\n"
            f"Causa: {error}\n"
            "Si estás en Colab, verificá que Drive se haya montado y que la cuenta tenga espacio."
        ) from error

    if not os.access(destino, os.W_OK):
        raise RuntimeError(
            f"La raíz de proyecto {destino} existe pero no admite escritura.\n"
            "Si es una carpeta compartida de Drive, pedí permiso de editor o cambiá "
            f"{VARIABLE_ENTORNO_RAIZ} a una carpeta propia."
        )

    for nombre in SUBCARPETAS:
        (destino / nombre).mkdir(exist_ok=True)

    _raiz = destino.resolve()

    if verbose:
        print(f"Entorno         : {'Colab' if en_colab() else 'local'}")
        print(f"Raíz de proyecto: {_raiz}")
        print(f"Área de trabajo : {dir_trabajo()}")

    return _raiz


def raiz() -> Path:
    """Raíz de proyecto vigente. Inicializa con los valores por defecto si hiciera falta."""
    if _raiz is None:
        return inicializar(verbose=False)
    return _raiz


def carpeta(nombre: str) -> Path:
    """Devuelve una subcarpeta de artefactos, creándola si no existe."""
    if nombre not in SUBCARPETAS:
        raise ValueError(f"Subcarpeta desconocida: {nombre!r}. Opciones: {', '.join(SUBCARPETAS)}")
    destino = raiz() / nombre
    destino.mkdir(parents=True, exist_ok=True)
    return destino


def dir_trabajo(nombre: str = "") -> Path:
    """Área de trabajo rápida y efímera, para artefactos numerosos.

    Lo que se escriba acá NO sobrevive a la sesión: hay que persistirlo con `cache.empaquetar`.
    """
    base = TRABAJO_EN_COLAB if en_colab() else (raiz() / "trabajo")
    destino = base / nombre if nombre else base
    destino.mkdir(parents=True, exist_ok=True)
    return destino


# ============================================================
# RUTAS EN LOS CSV
# ============================================================
# Los CSV guardan rutas RELATIVAS a la raíz. Guardar rutas absolutas fue exactamente la causa de que
# el notebook 02 no encontrara los clips del 01: `/content/clips_equinos/...` no apunta a nada en
# una sesión nueva. Relativas, el mismo CSV sirve en Colab, en local y en otra cuenta de Drive.


def relativizar(ruta: str | Path) -> str:
    """Convierte una ruta absoluta del proyecto en la ruta relativa que se guarda en los CSV."""
    ruta = Path(ruta)
    if not ruta.is_absolute():
        return ruta.as_posix()
    try:
        return ruta.resolve().relative_to(raiz()).as_posix()
    except ValueError:
        # Fuera de la raíz: se conserva absoluta antes que romper la trazabilidad en silencio.
        return ruta.as_posix()


def resolver(ruta: str | Path) -> Path:
    """Reconstruye la ruta absoluta de un artefacto a partir de lo guardado en un CSV.

    Acepta también las rutas absolutas de `/content` que escribían las versiones anteriores de los
    notebooks, para que un CSV viejo siga siendo utilizable.
    """
    ruta = Path(ruta)

    if not ruta.is_absolute():
        return raiz() / ruta

    if ruta.exists():
        return ruta

    texto = ruta.as_posix()
    for prefijo, subcarpeta in _PREFIJOS_HEREDADOS.items():
        if texto.startswith(prefijo + "/"):
            return raiz() / subcarpeta / texto[len(prefijo) + 1 :]

    return ruta
