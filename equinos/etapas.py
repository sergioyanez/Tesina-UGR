"""Registro de las etapas del pipeline y diagnóstico de su estado.

Es la fuente de verdad de qué produce cada notebook y de qué depende. El orquestador la consulta
para decidir qué ejecutar y qué saltear; el mismo predicado alimenta el diagnóstico que se puede
pedir al abrir una sesión, sin ejecutar nada.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from equinos import cache, rutas

# Estados posibles de una etapa
PENDIENTE = "pendiente"
PARCIAL = "parcial"
COMPLETA = "completa"


@dataclass(frozen=True)
class Etapa:
    """Una etapa del pipeline y los artefactos por los que se la reconoce terminada."""

    nombre: str
    notebook: str
    descripcion: str
    depende_de: tuple[str, ...] = ()
    # Rutas relativas a la raíz de proyecto que deben existir para considerarla completa.
    artefactos: tuple[str, ...] = ()
    # Nombres de caché comprimida que deben estar disponibles.
    caches: tuple[str, ...] = ()
    # Etapas cuyo trabajo se mide por clip y por lo tanto llevan checkpoint.
    lleva_checkpoint: bool = False


PIPELINE: dict[str, Etapa] = {
    "preparar_dataset": Etapa(
        nombre="preparar_dataset",
        notebook="01_preparar_dataset_equinos.ipynb",
        descripcion="Inventario, segmentación en clips de 5-7 s y extracción de 16 frames por clip",
        artefactos=(
            "resultados/inventario_videos.csv",
            "resultados/inventario_clips.csv",
            "resultados/dataset_frames.csv",
        ),
        caches=("frames_equinos",),
        lleva_checkpoint=True,
    ),
    "roi_normalizacion": Etapa(
        nombre="roi_normalizacion",
        notebook="02_roi_y_normalizacion.ipynb",
        descripcion="Detección del equino, keypoints faciales, ROI y normalización de pose",
        depende_de=("preparar_dataset",),
        artefactos=(
            "resultados/pose_completa.csv",
            "resultados/keypoints_faciales.csv",
            "resultados/dataset_frames_alineados.csv",
            "resultados/reporte_cobertura.csv",
        ),
        caches=("frames_alineados",),
        lleva_checkpoint=True,
    ),
}


def orden_de_ejecucion() -> list[str]:
    """Etapas ordenadas de modo que cada una vaya después de aquellas de las que depende."""
    ordenadas: list[str] = []
    visitando: set[str] = set()

    def visitar(nombre: str) -> None:
        if nombre in ordenadas:
            return
        if nombre in visitando:
            raise ValueError(f"Ciclo de dependencias en el pipeline, alrededor de {nombre!r}")
        visitando.add(nombre)
        for dependencia in PIPELINE[nombre].depende_de:
            if dependencia not in PIPELINE:
                raise ValueError(f"{nombre!r} depende de {dependencia!r}, que no está registrada")
            visitar(dependencia)
        visitando.discard(nombre)
        ordenadas.append(nombre)

    for nombre in PIPELINE:
        visitar(nombre)
    return ordenadas


def _progreso_checkpoint(nombre: str) -> tuple[int, str | None]:
    """Unidades ya procesadas según el checkpoint, y cuándo se actualizó."""
    ruta = rutas.carpeta("checkpoints") / f"{nombre}.json"
    if not ruta.exists():
        return 0, None
    try:
        datos = json.loads(ruta.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return 0, None
    return len(datos.get("unidades_completas", [])), datos.get("actualizado")


def _artefacto_valido(ruta) -> bool:
    """Un artefacto cuenta como producido sólo si además tiene contenido.

    La existencia sola es un criterio demasiado débil: un CSV con nada más que el encabezado
    aparece igual que uno con 1472 filas, y haría pasar por completa una etapa que no procesó
    nada. En un pipeline de tesina ese falso positivo es caro, porque el error no se manifiesta
    acá sino mucho después, como métricas calculadas sobre un dataset vacío.
    """
    if not ruta.exists():
        return False
    if ruta.stat().st_size == 0:
        return False
    if ruta.suffix.lower() == ".csv":
        try:
            with ruta.open(encoding="utf-8", errors="replace") as archivo:
                next(archivo, None)              # encabezado
                return next(archivo, None) is not None   # al menos una fila de datos
        except OSError:
            return False
    return True


def estado(nombre: str) -> dict:
    """Estado de una etapa, sin ejecutarla ni modificar ningún artefacto."""
    etapa = PIPELINE[nombre]

    faltan_artefactos = [a for a in etapa.artefactos if not _artefacto_valido(rutas.raiz() / a)]
    faltan_caches = [c for c in etapa.caches if not cache.esta_disponible(c)]
    procesadas, actualizado = _progreso_checkpoint(nombre)

    if not faltan_artefactos and not faltan_caches:
        situacion = COMPLETA
    elif procesadas > 0:
        situacion = PARCIAL
    else:
        situacion = PENDIENTE

    return {
        "etapa": nombre,
        "estado": situacion,
        "notebook": etapa.notebook,
        "unidades_procesadas": procesadas,
        "actualizado": actualizado,
        "artefactos_faltantes": faltan_artefactos,
        "caches_faltantes": faltan_caches,
    }


def diagnostico(imprimir: bool = True) -> list[dict]:
    """Estado de todas las etapas, en orden de ejecución."""
    reporte = [estado(nombre) for nombre in orden_de_ejecucion()]

    if imprimir:
        simbolos = {COMPLETA: "[x]", PARCIAL: "[~]", PENDIENTE: "[ ]"}
        print(f"Raíz de proyecto: {rutas.raiz()}\n")
        for fila in reporte:
            print(f"{simbolos[fila['estado']]} {fila['etapa']:<20} {fila['estado']}")
            if fila["estado"] == PARCIAL:
                print(f"      {fila['unidades_procesadas']} unidades procesadas "
                      f"(última actualización: {fila['actualizado']})")
            for artefacto in fila["artefactos_faltantes"]:
                print(f"      falta artefacto: {artefacto}")
            for nombre_cache in fila["caches_faltantes"]:
                print(f"      falta caché: {nombre_cache}")
        print()

    return reporte


def limpiar(nombre: str, incluir_artefactos: bool = True, verbose: bool = True) -> None:
    """Invalida una etapa de forma explícita y dirigida.

    Es la única vía por la que el pipeline borra artefactos. Ningún notebook borra nada al arrancar:
    la versión anterior del 01 hacía `rmtree` de los clips en su primer paso, lo que volvía imposible
    reanudar nada porque re-ejecutarlo destruía el trabajo de las etapas siguientes.
    """
    etapa = PIPELINE[nombre]

    for ruta in (
        rutas.carpeta("checkpoints") / f"{nombre}.json",
        rutas.carpeta("checkpoints") / f"{nombre}_parcial.csv",
    ):
        if ruta.exists():
            ruta.unlink()
            if verbose:
                print(f"Borrado: {ruta}")

    for nombre_cache in etapa.caches:
        cache.limpiar(nombre_cache, verbose=verbose)

    if incluir_artefactos:
        for artefacto in etapa.artefactos:
            ruta = rutas.raiz() / artefacto
            if ruta.exists():
                ruta.unlink()
                if verbose:
                    print(f"Borrado: {ruta}")

    dependientes = [n for n, e in PIPELINE.items() if nombre in e.depende_de]
    if dependientes and verbose:
        print(f"\nAtención: {', '.join(dependientes)} dependen de {nombre!r} "
              "y sus resultados quedaron desactualizados.")
