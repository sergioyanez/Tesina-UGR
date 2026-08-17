"""Lectura de los resultados del preprocesamiento facial.

Vive en el paquete y no dentro de una celda para que se pueda pedir desde cualquier notebook con
una línea, y para que el criterio de "cobertura aceptable" esté escrito en un solo lugar en vez de
recalculado a ojo en cada consulta.

Deliberadamente sin pandas: DeepLabCut fija numpy<2 y deja el stack científico de la sesión en un
estado del que pandas no siempre se recupera. Estos CSV tienen 1472 filas; la biblioteca estándar
alcanza de sobra.
"""

from __future__ import annotations

import csv
from pathlib import Path

from equinos import rutas

# Frames que el Bloque 3 extrae por clip. Si cambia allá, cambia acá.
FRAMES_POR_CLIP = 16


def _leer(nombre: str) -> list[dict]:
    ruta = rutas.carpeta("resultados") / nombre
    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe {ruta}. Hay que ejecutar la etapa roi_normalizacion."
        )
    with ruta.open(encoding="utf-8", errors="replace") as archivo:
        return list(csv.DictReader(archivo))


def _es_si(valor) -> bool:
    return str(valor).strip().lower() in ("true", "1", "si", "sí")


def cobertura(umbral: float = 0.70, peores: int = 8) -> dict:
    """Imprime el reporte de cobertura y devuelve sus números.

    `umbral` es la tasa de alineación por debajo de la cual un ejemplar se considera problemático.
    Por defecto coincide con COBERTURA_MINIMA_ACEPTABLE del notebook 02.
    """
    frames = _leer("dataset_frames_alineados.csv")
    if not frames:
        raise ValueError("dataset_frames_alineados.csv no tiene filas.")

    clips = _leer("inventario_clips.csv")
    esperados = len(clips) * FRAMES_POR_CLIP

    generados = len(frames)
    alineados = sum(1 for f in frames if _es_si(f["alineado"]))

    print("=" * 58)
    print("COBERTURA DEL PREPROCESAMIENTO FACIAL")
    print("=" * 58)
    print(f"Clips procesados : {len(clips)}")
    print(f"Frames generados : {generados} de {esperados} ({generados / esperados:.1%})")
    print(f"Alineados por kp : {alineados} de {generados} ({alineados / generados:.1%})")

    por_clase: dict[str, list[int]] = {}
    for fila in frames:
        acumulado = por_clase.setdefault(fila["clase"], [0, 0])
        acumulado[0] += 1
        acumulado[1] += _es_si(fila["alineado"])

    print("\nPor clase:")
    for clase, (total, ok) in sorted(por_clase.items()):
        print(f"  {clase:<14} {total:>5} frames   {ok / total:6.1%} alineados")

    caballos = sorted(_leer("reporte_cobertura.csv"), key=lambda f: float(f["tasa"]))
    problematicos = [f for f in caballos if float(f["tasa"]) < umbral]

    print(f"\nPeores {min(peores, len(caballos))} ejemplares:")
    for fila in caballos[:peores]:
        marca = "  <-- bajo el umbral" if float(fila["tasa"]) < umbral else ""
        print(f"  {fila['id_caballo']:<7} {fila.get('clase', ''):<14} "
              f"{float(fila['tasa']):6.1%}   {fila['frames']:>4} frames{marca}")

    tasa_global = alineados / generados
    print("\n" + "-" * 58)
    if tasa_global >= umbral and not problematicos:
        print(f"El pipeline automático es utilizable: {tasa_global:.1%} >= {umbral:.0%}")
    elif tasa_global >= umbral:
        cuantos = len(problematicos)
        print(f"Cobertura global aceptable ({tasa_global:.1%}), pero "
              f"{cuantos} ejemplar{'es' if cuantos != 1 else ''} "
              f"qued{'an' if cuantos != 1 else 'a'} por debajo del {umbral:.0%}.")
        print("Decidir qué hacer con ellos ANTES de entrenar: un ejemplar mal alineado")
        print("mete ruido sistemático, no aleatorio.")
    else:
        print(f"Cobertura global insuficiente: {tasa_global:.1%} < {umbral:.0%}")
        print("Hay que ajustar umbrales o pasar a anotación manual de las ROI.")

    return {
        "clips": len(clips),
        "frames_esperados": esperados,
        "frames_generados": generados,
        "frames_alineados": alineados,
        "tasa_global": tasa_global,
        "por_clase": {c: {"frames": t, "alineados": o, "tasa": o / t}
                      for c, (t, o) in por_clase.items()},
        "ejemplares_bajo_umbral": [f["id_caballo"] for f in problematicos],
    }
