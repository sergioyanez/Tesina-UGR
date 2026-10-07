"""Qué videos pertenecen a cada clase, y desde qué carpetas de Drive salen.

Hasta ahora la clase de un video era el nombre de su carpeta contenedora, y cualquier subcarpeta
quedaba afuera. Esa regla se rompió cuando el material etiquetado empezó a vivir también en
subcarpetas de trabajo: el 20/08/2026, `Disgustado` —la clase minoritaria, la que más falta hace—
tenía 10 de sus videos dentro de `Seleccionar cabeza?`, invisibles para el inventario.

La regla nueva es explícita en vez de estructural: **por cada clase se declara qué subcarpetas
suman**, y esa lista NO es la misma en todas. `Disgustado` suma dos, `Interesado` y `Sereno` una.
Se declara acá, en un solo lugar, y no en la celda de un notebook: la composición del dataset es
una decisión de etiquetado que hay que poder auditar y citar en el informe, no un detalle de
implementación.

Lo que NO está declarado sigue quedando afuera, pero **se informa** con nombre y conteo. Ese fue
el punto de la regla anterior y se conserva: una carpeta de trabajo que entra callada al dataset
es una etiqueta inventada, y una que sale callada es material perdido. Ninguna de las dos avisa.

Uso típico:

    from equinos import catalogo

    videos_por_clase = catalogo.catalogar()          # {'Disgustado': [Path, ...], ...}
    inventario = catalogo.inventariar()              # lo mismo, con el detalle de qué entró
    catalogo.informe(inventario)                     # y su impresión legible
"""

from __future__ import annotations

import os
import re
import unicodedata
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from equinos import rutas

# ============================================================
# CONFIGURACIÓN
# ============================================================

EXTENSIONES_VIDEO = (".mp4", ".avi", ".mov", ".mkv", ".webm")

# Las subcarpetas que suman a cada clase, además de los videos sueltos en la raíz de la clase.
#
# Deliberadamente NO es una lista única compartida: cada categoría declara la suya. Suponer una
# estructura común es lo que obligaba a elegir entre dejar afuera los 10 videos de `Disgustado` o
# dejar entrar carpetas que en otra clase no corresponden.
#
# `Separar cabeza?` (en `Sereno`) NO está declarada a propósito: es material de trabajo sin
# decisión de etiqueta tomada. Si mañana la tiene, se agrega acá y se re-ejecuta la etapa.
SUBCARPETAS_POR_CLASE: dict[str, tuple[str, ...]] = {
    "Disgustado": ("Seleccionar cabeza?", "Se tapa una parte de la cara"),
    "Interesado": ("Seleccionar cabeza?",),
    "Sereno": ("Seleccionar cabeza?",),
}

# Carpeta de Drive que contiene las tres carpetas de clase. Está en "Compartidos conmigo", así que
# Colab sólo la ve si hay un ACCESO DIRECTO a ella en Mi unidad (Drive -> clic derecho ->
# Organizar -> Agregar acceso directo). Eso no copia ni mueve nada.
NOMBRE_CARPETA_COMPARTIDA = "Videos de prueba"

# Permite apuntar a otra carpeta sin tocar el código: útil en local, donde el espejo del dataset
# no está en la raíz de proyecto.
VARIABLE_ENTORNO_VIDEOS = "TESINA_EQUINOS_VIDEOS"

# Profundidad máxima al buscar la carpeta compartida bajo Mi unidad. Un `rglob("*")` sobre Drive
# montado recorre la cuenta entera archivo por archivo, y cada entrada es una llamada de red.
PROFUNDIDAD_BUSQUEDA = 3


# ============================================================
# COMPARACIÓN DE NOMBRES DE CARPETA
# ============================================================
# El mismo directorio no se llama igual en todos lados. En Drive es literalmente
# `Seleccionar cabeza?`; en una copia bajada como ZIP el signo de pregunta queda saneado a
# `Seleccionar cabeza_`, porque no es un carácter válido en todos los sistemas de archivos.
# Comparar literalmente haría que el código ande en Colab y falle callado en la copia local
# —o al revés—, que es exactamente el tipo de divergencia que no se nota hasta que faltan videos.

_NO_ALFANUMERICO = re.compile(r"[^0-9a-z]+")


def clave(nombre: str) -> str:
    """Forma canónica de un nombre de carpeta, para compararlo sin depender de su grafía.

    Ignora mayúsculas, acentos y todo signo: `Seleccionar cabeza?`, `seleccionar cabeza_` y
    `Selección cabeza` colapsan al mismo valor.
    """
    sin_acentos = "".join(
        c for c in unicodedata.normalize("NFKD", nombre) if not unicodedata.combining(c)
    )
    return _NO_ALFANUMERICO.sub(" ", sin_acentos.casefold()).strip()


def _subdirectorio(base: Path, nombre: str) -> Path | None:
    """Subdirectorio de `base` cuyo nombre equivale a `nombre`. None si no existe."""
    buscada = clave(nombre)
    # La coincidencia exacta primero: si está, no hace falta listar el directorio entero.
    directo = base / nombre
    if directo.is_dir():
        return directo
    try:
        candidatos = sorted(p for p in base.iterdir() if p.is_dir())
    except OSError:
        return None
    for candidato in candidatos:
        if clave(candidato.name) == buscada:
            return candidato
    return None


def carpeta_de_clase(base: str | Path, etiqueta: str) -> Path | None:
    """Carpeta de una clase dentro de `base`, sin depender de su grafía. None si no está."""
    return _subdirectorio(Path(base), etiqueta)


def _videos_en(directorio: Path, recursivo: bool) -> list[Path]:
    """Archivos de video de un directorio, ordenados por ruta."""
    entradas = directorio.rglob("*") if recursivo else directorio.iterdir()
    return sorted(
        ruta for ruta in entradas
        if ruta.is_file() and ruta.suffix.lower() in EXTENSIONES_VIDEO
    )


# ============================================================
# RAÍZ DE LOS VIDEOS
# ============================================================


def _tiene_clases(directorio: Path, clases) -> bool:
    """Indica si un directorio contiene al menos una de las carpetas de clase."""
    if not directorio.is_dir():
        return False
    return any(_subdirectorio(directorio, nombre) is not None for nombre in clases)


def _candidatos() -> Iterator[Path]:
    """Carpetas donde puede estar el dataset, en orden de preferencia y evaluadas de a una.

    La búsqueda amplia bajo Mi unidad va última y sólo se ejecuta si ninguna de las rutas
    directas sirvió: cada entrada que recorre sobre Drive montado es una llamada de red.
    """
    if rutas.en_colab():
        mi_unidad = rutas.PUNTO_MONTAJE_DRIVE / "MyDrive"
        # La compartida primero: es la fuente de verdad del material etiquetado. La carpeta de
        # la raíz de proyecto puede tener una copia parcial de una corrida anterior, y ganarle
        # significaría inventariar un dataset viejo sin que nada lo indique.
        yield mi_unidad / NOMBRE_CARPETA_COMPARTIDA
        yield rutas.carpeta("videos")
        yield from _buscar_compartida(mi_unidad)
    else:
        yield rutas.carpeta("videos")


def raiz_videos(base: str | Path | None = None, clases=None) -> Path:
    """Carpeta que contiene las tres carpetas de clase.

    Resuelve en cascada, y en cada paso exige que el candidato **efectivamente** contenga alguna
    carpeta de clase. Una carpeta vacía que existe es el peor resultado posible: el inventario
    sale en cero y las etapas siguientes procesan nada sin fallar.

      1. `base`, si se pasa explícitamente (lo que inyecta el orquestador).
      2. La variable de entorno `TESINA_EQUINOS_VIDEOS`.
      3. En Colab, el acceso directo a `Videos de prueba` en Mi unidad.
      4. `<raiz de proyecto>/videos`.
    """
    clases = tuple(clases if clases is not None else SUBCARPETAS_POR_CLASE)

    if base:
        destino = Path(base).expanduser()
        if not destino.is_dir():
            raise SystemExit(f"La carpeta de videos indicada no existe: {destino}")
        return destino

    desde_entorno = os.environ.get(VARIABLE_ENTORNO_VIDEOS)
    if desde_entorno:
        destino = Path(desde_entorno).expanduser()
        if not destino.is_dir():
            raise SystemExit(
                f"{VARIABLE_ENTORNO_VIDEOS} apunta a {destino}, que no existe."
            )
        return destino

    revisados = []
    for candidato in _candidatos():
        revisados.append(candidato)
        if _tiene_clases(candidato, clases):
            return candidato

    detalle = "\n".join(f"  - {c}" for c in revisados)
    raise SystemExit(
        f"No encontré ninguna carpeta con las clases {', '.join(clases)}.\n"
        f"Revisé:\n{detalle}\n\n"
        f"La carpeta '{NOMBRE_CARPETA_COMPARTIDA}' está en 'Compartidos conmigo', y Colab no monta\n"
        "esa sección como una ruta. Hay que agregarle un acceso directo en Mi unidad:\n"
        "  Drive -> Compartidos conmigo -> clic derecho -> Organizar -> Agregar acceso directo.\n"
        "Eso NO copia ni mueve los videos.\n\n"
        f"Alternativa: exportar {VARIABLE_ENTORNO_VIDEOS} con la ruta, o pasarla como parámetro."
    )


def _buscar_compartida(mi_unidad: Path) -> list[Path]:
    """Carpetas llamadas como la compartida, buscadas a profundidad acotada bajo Mi unidad."""
    buscada = clave(NOMBRE_CARPETA_COMPARTIDA)
    hallazgos: list[Path] = []
    niveles = [mi_unidad]
    for _ in range(PROFUNDIDAD_BUSQUEDA):
        siguiente: list[Path] = []
        for directorio in niveles:
            try:
                hijos = [p for p in directorio.iterdir() if p.is_dir()]
            except OSError:
                continue
            for hijo in hijos:
                if clave(hijo.name) == buscada:
                    hallazgos.append(hijo)
                else:
                    siguiente.append(hijo)
        niveles = siguiente
    return hallazgos


# ============================================================
# CATÁLOGO
# ============================================================


@dataclass
class Inventario:
    """Resultado de recorrer las carpetas: qué entró, de dónde, y qué quedó afuera."""

    base: Path
    videos: dict[str, list[Path]] = field(default_factory=dict)
    # clase -> nombre real de la carpeta aportante -> cantidad de videos que aportó.
    aportes: dict[str, dict[str, int]] = field(default_factory=dict)
    # clase -> subcarpetas declaradas que no existen en el disco.
    declaradas_ausentes: dict[str, list[str]] = field(default_factory=dict)
    # clase -> videos que están bajo la clase pero en subcarpetas NO declaradas.
    excluidos: dict[str, list[Path]] = field(default_factory=dict)
    # clases declaradas cuya carpeta no aparece bajo la base.
    clases_ausentes: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return sum(len(v) for v in self.videos.values())


def inventariar(
    base: str | Path | None = None,
    subcarpetas_por_clase: dict[str, tuple[str, ...]] | None = None,
) -> Inventario:
    """Recorre las carpetas de clase y devuelve el catálogo con su detalle de procedencia."""
    mapa = subcarpetas_por_clase if subcarpetas_por_clase is not None else SUBCARPETAS_POR_CLASE
    raiz = raiz_videos(base, clases=mapa.keys())

    inventario = Inventario(base=raiz)

    for etiqueta, subcarpetas in mapa.items():
        carpeta_clase = _subdirectorio(raiz, etiqueta)
        if carpeta_clase is None:
            inventario.clases_ausentes.append(etiqueta)
            inventario.videos[etiqueta] = []
            inventario.aportes[etiqueta] = {}
            continue

        # Los videos sueltos en la raíz de la clase, sin recursión: lo que hay más abajo entra
        # sólo si su subcarpeta está declarada.
        encontrados = _videos_en(carpeta_clase, recursivo=False)
        aportes = {carpeta_clase.name: len(encontrados)}
        ausentes: list[str] = []

        for nombre in subcarpetas:
            subcarpeta = _subdirectorio(carpeta_clase, nombre)
            if subcarpeta is None:
                ausentes.append(nombre)
                continue
            # Recursivo acá sí: si una subcarpeta declarada tiene otra adentro, su contenido
            # ya recibió la misma decisión de etiqueta que la subcarpeta que la contiene.
            de_la_subcarpeta = _videos_en(subcarpeta, recursivo=True)
            encontrados.extend(de_la_subcarpeta)
            aportes[f"{carpeta_clase.name}/{subcarpeta.name}"] = len(de_la_subcarpeta)

        inventario.videos[etiqueta] = sorted(encontrados)
        inventario.aportes[etiqueta] = aportes
        if ausentes:
            inventario.declaradas_ausentes[etiqueta] = ausentes

        # Todo lo que está bajo la clase y no entró. No es un error —hay carpetas de trabajo
        # legítimas— pero tiene que quedar dicho: es la diferencia entre descartar y perder.
        catalogados = set(inventario.videos[etiqueta])
        afuera = [r for r in _videos_en(carpeta_clase, recursivo=True) if r not in catalogados]
        if afuera:
            inventario.excluidos[etiqueta] = afuera

    return inventario


def catalogar(
    base: str | Path | None = None,
    subcarpetas_por_clase: dict[str, tuple[str, ...]] | None = None,
) -> dict[str, list[Path]]:
    """Diccionario `{clase: [rutas de video]}`, con las subcarpetas declaradas ya incorporadas."""
    return inventariar(base, subcarpetas_por_clase).videos


# ============================================================
# INFORME Y VALIDACIÓN
# ============================================================


def informe(inventario: Inventario, ejemplos: int = 3) -> Inventario:
    """Imprime de dónde salió cada video y qué quedó afuera. Devuelve el inventario, para encadenar."""
    print(f"Carpeta de videos: {inventario.base}")
    print(f"Total catalogado : {inventario.total} videos\n")

    for etiqueta, encontrados in inventario.videos.items():
        print(f"{etiqueta}: {len(encontrados)} videos")
        for origen, cantidad in inventario.aportes.get(etiqueta, {}).items():
            print(f"    {cantidad:>4}  {origen}")
        for nombre in inventario.declaradas_ausentes.get(etiqueta, []):
            print(f"       -  {nombre}  (declarada, no existe en el disco)")

    if inventario.clases_ausentes:
        print(f"\nCLASES SIN CARPETA: {', '.join(inventario.clases_ausentes)}")

    if inventario.excluidos:
        total = sum(len(v) for v in inventario.excluidos.values())
        print(f"\nAFUERA — {total} videos en subcarpetas no declaradas:")
        for etiqueta, afuera in inventario.excluidos.items():
            por_carpeta: dict[str, list[Path]] = {}
            for ruta in afuera:
                por_carpeta.setdefault(ruta.parent.name, []).append(ruta)
            for carpeta_nombre, archivos in sorted(por_carpeta.items()):
                print(f"  {etiqueta} / {carpeta_nombre!r}: {len(archivos)} videos")
                for ruta in archivos[:ejemplos]:
                    print(f"      {ruta.name}")
                if len(archivos) > ejemplos:
                    print(f"      ... y {len(archivos) - ejemplos} más")
        print("  Si alguno debía entrar, agregá su subcarpeta a SUBCARPETAS_POR_CLASE.")

    return inventario


def verificar(inventario: Inventario) -> None:
    """Corta la etapa ante las dos formas de catálogo inservible. No devuelve nada: falla o pasa.

    Un inventario vacío o con nombres repetidos entre clases no rompe nada por sí mismo; rompe
    tres etapas más adelante, con el síntoma desplazado y la causa ya fuera de vista.
    """
    if inventario.total == 0:
        raise SystemExit(
            f"No hay ningún video bajo {inventario.base}.\n"
            f"Esperaba una carpeta por clase ({', '.join(SUBCARPETAS_POR_CLASE)}) con los videos\n"
            "adentro, o dentro de las subcarpetas declaradas en catalogo.SUBCARPETAS_POR_CLASE.\n"
            "Un inventario vacío haría que las etapas siguientes procesen cero clips sin avisar."
        )

    # El nombre del archivo es la clave del checkpoint y la base del nombre de cada clip, que a su
    # vez es la clave de unión con el notebook 02. Dos videos homónimos en clases distintas
    # producirían clips con el mismo nombre y dos filas con la misma clave y distinta etiqueta:
    # las etapas siguientes cruzarían frames de una clase con los de la otra, sin fallar.
    #
    # Con las subcarpetas sumando al catálogo esto dejó de ser hipotético: la misma toma puede
    # estar suelta en la raíz de la clase y también dentro de `Seleccionar cabeza?`.
    procedencias: dict[str, list[Path]] = {}
    for etiqueta, encontrados in inventario.videos.items():
        for ruta in encontrados:
            procedencias.setdefault(ruta.name, []).append(ruta)

    colisiones = {n: r for n, r in procedencias.items() if len(r) > 1}
    if colisiones:
        detalle = "\n".join(
            f"  {nombre}\n" + "\n".join(f"      {rutas.relativizar(r)}" for r in encontradas)
            for nombre, encontradas in sorted(colisiones.items())
        )
        raise SystemExit(
            f"Hay {len(colisiones)} nombre(s) de archivo repetidos en el catálogo:\n{detalle}\n\n"
            "El nombre del archivo es la clave del checkpoint y la base del nombre de los clips,\n"
            "que es la clave de unión con el notebook 02. Dejarlo pasar mezclaría frames bajo la\n"
            "misma clave, sin error visible y con la etiqueta equivocada.\n\n"
            "Si son la misma toma duplicada entre la raíz de la clase y una subcarpeta, borrá una\n"
            "copia. Si son tomas distintas, renombrá una."
        )
