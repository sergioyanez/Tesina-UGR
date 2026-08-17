"""Persistencia de los artefactos numerosos del pipeline.

Google Drive montado tiene una latencia por archivo que domina cuando hay miles de archivos chicos:
escribir los ~1472 JPEG de frames tarda más que escribir los 2.7 GB de clips. El patrón es entonces
generar en el disco local de la sesión, rápido y efímero, y persistir un único `.zip` a Drive.

Sin compresión (`ZIP_STORED`): los JPEG ya vienen comprimidos, así que deflate gastaría CPU sobre
1472 archivos para no ganar espacio.
"""

from __future__ import annotations

import json
import os
import zipfile
from datetime import datetime
from pathlib import Path

from equinos import rutas


def _ruta_zip(etapa: str) -> Path:
    return rutas.carpeta("cache") / f"{etapa}.zip"


def _ruta_manifiesto(etapa: str) -> Path:
    return rutas.carpeta("cache") / f"{etapa}.json"


def empaquetar(directorio: str | Path, etapa: str, hash_config: str | None = None) -> Path:
    """Empaqueta un directorio del área de trabajo en la caché persistente.

    Devuelve la ruta del zip. Escribe de forma atómica: un corte durante la subida deja el zip
    anterior intacto en lugar de uno truncado.
    """
    directorio = Path(directorio)
    if not directorio.is_dir():
        raise FileNotFoundError(f"No existe el directorio a empaquetar: {directorio}")

    archivos = sorted(p for p in directorio.rglob("*") if p.is_file())
    if not archivos:
        raise ValueError(
            f"El directorio {directorio} no tiene archivos. Empaquetar una caché vacía la haría "
            "pasar por válida y las etapas siguientes leerían cero frames sin enterarse."
        )

    destino = _ruta_zip(etapa)
    temporal = destino.with_suffix(".zip.tmp")

    with zipfile.ZipFile(temporal, "w", compression=zipfile.ZIP_STORED) as zf:
        for archivo in archivos:
            zf.write(archivo, archivo.relative_to(directorio).as_posix())

    os.replace(temporal, destino)

    _ruta_manifiesto(etapa).write_text(
        json.dumps(
            {
                "etapa": etapa,
                "n_archivos": len(archivos),
                "hash_config": hash_config,
                "creado": datetime.now().isoformat(timespec="seconds"),
                "bytes": destino.stat().st_size,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"[{etapa}] caché guardada: {len(archivos)} archivos, "
          f"{destino.stat().st_size / 1e6:.0f} MB -> {destino}")
    return destino


def manifiesto(etapa: str) -> dict | None:
    """Manifiesto de la caché, o None si no existe o es ilegible."""
    ruta = _ruta_manifiesto(etapa)
    if not ruta.exists():
        return None
    try:
        return json.loads(ruta.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def esta_disponible(etapa: str, hash_config: str | None = None) -> bool:
    """Indica si hay una caché utilizable para la etapa.

    Si se pasa `hash_config`, la caché generada con otra configuración se considera no utilizable:
    reusarla mezclaría resultados de dos corridas distintas.
    """
    info = manifiesto(etapa)
    if info is None or not _ruta_zip(etapa).exists():
        return False
    if hash_config is not None and info.get("hash_config") != hash_config:
        return False
    return True


def expandir(etapa: str, directorio: str | Path, hash_config: str | None = None) -> Path:
    """Expande la caché de una etapa al área de trabajo y devuelve el directorio.

    Verifica que la cantidad de archivos coincida con la registrada. La verificación es por conteo y
    no por hash del contenido: hashear 1472 archivos en cada arranque costaría más de lo que vale, y
    el único modo de corrupción plausible acá —un zip truncado por un corte durante la subida— se
    detecta igual con el conteo.
    """
    if not esta_disponible(etapa, hash_config):
        raise FileNotFoundError(
            f"No hay caché válida para la etapa {etapa!r} en {rutas.carpeta('cache')}. "
            "Hay que ejecutar la etapa que la produce."
        )

    directorio = Path(directorio)
    directorio.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(_ruta_zip(etapa)) as zf:
        zf.extractall(directorio)

    esperados = manifiesto(etapa)["n_archivos"]
    obtenidos = sum(1 for p in directorio.rglob("*") if p.is_file())
    if obtenidos != esperados:
        raise RuntimeError(
            f"Caché de {etapa!r} inconsistente: el manifiesto declara {esperados} archivos y se "
            f"expandieron {obtenidos}. Probablemente el zip quedó truncado. "
            f"Borrar {_ruta_zip(etapa)} y volver a ejecutar la etapa."
        )

    print(f"[{etapa}] caché expandida: {obtenidos} archivos en {directorio}")
    return directorio


def limpiar(etapa: str, verbose: bool = True) -> None:
    """Borra la caché de una etapa y su manifiesto."""
    for ruta in (_ruta_zip(etapa), _ruta_manifiesto(etapa)):
        if ruta.exists():
            ruta.unlink()
            if verbose:
                print(f"Borrado: {ruta}")
