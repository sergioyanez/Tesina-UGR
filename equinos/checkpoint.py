"""Estado reanudable de las etapas largas del pipeline.

El notebook 02 procesa 92 clips a razón de ~45 s cada uno: alrededor de una hora. Colab corta la
sesión por límite de tiempo o por inactividad, y sin puntos de guardado una interrupción a los 50
minutos no deja nada aprovechable. Ya ocurrió.

La unidad de checkpoint es el **clip**, que es como ya itera el notebook. Acota la pérdida ante un
corte a menos de un minuto de trabajo. Bajar a frame multiplicaría por 16 las escrituras a Drive sin
ganancia real.

Riesgo que este módulo existe para evitar
-----------------------------------------
Reanudar con una configuración distinta de la que generó el checkpoint produce un artefacto que
mezcla dos corridas: no falla, no avisa, y devuelve números plausibles y equivocados. Por eso el
checkpoint registra su configuración y se invalida solo, en vez de confiar en que quien ejecuta se
acuerde de borrarlo.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

from equinos import rutas


def calcular_hash(config: dict, claves: Sequence[str]) -> str:
    """Huella de los parámetros que afectan el resultado de una etapa.

    Sólo entran las `claves` declaradas por la etapa, no toda la configuración: así cambiar el
    tamaño de una figura de inspección visual no invalida una hora de cómputo.
    """
    faltantes = [c for c in claves if c not in config]
    if faltantes:
        raise KeyError(
            f"La etapa declara los parámetros {faltantes} pero no están en la configuración. "
            "Si un parámetro dejó de existir, hay que sacarlo de la lista de claves relevantes."
        )
    relevante = {c: config[c] for c in sorted(claves)}
    texto = json.dumps(relevante, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()[:16]


class Checkpoint:
    """Progreso persistido de una etapa, más el CSV parcial con sus resultados."""

    def __init__(self, etapa: str, config: dict, claves_relevantes: Sequence[str]):
        self.etapa = etapa
        self.claves_relevantes = list(claves_relevantes)
        self.config = {c: config[c] for c in claves_relevantes if c in config}
        self.hash_config = calcular_hash(config, claves_relevantes)
        self.unidades_completas: list[str] = []
        self.actualizado: str | None = None

    # --- ubicación de los archivos ---------------------------------------------------

    @property
    def ruta_estado(self) -> Path:
        return rutas.carpeta("checkpoints") / f"{self.etapa}.json"

    @property
    def ruta_parcial(self) -> Path:
        return rutas.carpeta("checkpoints") / f"{self.etapa}_parcial.csv"

    # --- ciclo de vida ---------------------------------------------------------------

    def cargar(self, verbose: bool = True) -> "Checkpoint":
        """Lee el estado persistido, descartándolo si no corresponde a esta configuración."""
        if not self.ruta_estado.exists():
            return self

        try:
            datos = json.loads(self.ruta_estado.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as error:
            # Un checkpoint ilegible se trata como inexistente: reprocesar cuesta tiempo, usar
            # datos corruptos cuesta un resultado inválido.
            if verbose:
                print(f"[{self.etapa}] checkpoint ilegible ({error}); se reprocesa desde cero.")
            return self

        if datos.get("hash_config") != self.hash_config:
            cambiadas = self._claves_cambiadas(datos.get("config", {}))
            if verbose:
                print(f"[{self.etapa}] la configuración cambió; se descarta el checkpoint previo.")
                for clave, (antes, ahora) in cambiadas.items():
                    print(f"    {clave}: {antes!r} -> {ahora!r}")
            self.limpiar(verbose=False)
            return self

        self.unidades_completas = list(datos.get("unidades_completas", []))
        self.actualizado = datos.get("actualizado")
        if verbose and self.unidades_completas:
            print(f"[{self.etapa}] checkpoint válido: {len(self.unidades_completas)} unidades ya procesadas.")
        return self

    def _claves_cambiadas(self, config_previa: dict) -> dict[str, tuple]:
        cambios = {}
        for clave in self.claves_relevantes:
            antes, ahora = config_previa.get(clave, "<ausente>"), self.config.get(clave)
            if antes != ahora:
                cambios[clave] = (antes, ahora)
        return cambios

    def guardar(self) -> None:
        """Persiste el estado de forma atómica.

        Se escribe a un temporal y se renombra encima. Un corte a mitad de la escritura deja intacto
        el checkpoint anterior, en vez de un JSON truncado que sería peor que no tener ninguno.
        """
        datos = {
            "etapa": self.etapa,
            "hash_config": self.hash_config,
            "config": self.config,
            "unidades_completas": self.unidades_completas,
            "actualizado": datetime.now().isoformat(timespec="seconds"),
        }
        destino = self.ruta_estado
        temporal = destino.with_suffix(".json.tmp")
        temporal.write_text(
            json.dumps(datos, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
        )
        os.replace(temporal, destino)

    def limpiar(self, verbose: bool = True) -> None:
        """Borra el estado y el CSV parcial de esta etapa."""
        for ruta in (self.ruta_estado, self.ruta_parcial):
            if ruta.exists():
                ruta.unlink()
                if verbose:
                    print(f"Borrado: {ruta}")
        self.unidades_completas = []
        self.actualizado = None

    # --- trabajo ---------------------------------------------------------------------

    def pendientes(self, universo: Iterable[str]) -> list[str]:
        """Unidades del universo que todavía no se procesaron, en el orden recibido."""
        hechas = set(self.unidades_completas)
        return [u for u in universo if u not in hechas]

    def esta_completo(self, universo: Iterable[str]) -> bool:
        return not self.pendientes(universo)

    def marcar(self, unidad: str, filas: Sequence[dict] | None = None) -> None:
        """Registra una unidad como terminada, anexando sus filas al CSV parcial.

        El orden importa: primero las filas, después el estado. Si el proceso muere en el medio, el
        peor caso es reprocesar una unidad —inofensivo—; al revés, la unidad quedaría marcada como
        hecha sin sus datos, y el resultado saldría incompleto sin avisar.
        """
        if filas:
            self._anexar(filas)
        if unidad not in self.unidades_completas:
            self.unidades_completas.append(unidad)
        self.guardar()

    def _anexar(self, filas: Sequence[dict]) -> None:
        import pandas as pd

        nuevo = not self.ruta_parcial.exists()
        pd.DataFrame(list(filas)).to_csv(
            self.ruta_parcial, mode="a", header=nuevo, index=False
        )

    def filas(self):
        """DataFrame con todo lo acumulado hasta ahora. Vacío si no hay nada."""
        import pandas as pd

        if not self.ruta_parcial.exists():
            return pd.DataFrame()
        return pd.read_csv(self.ruta_parcial)


def abrir(etapa: str, config: dict, claves_relevantes: Sequence[str], verbose: bool = True) -> Checkpoint:
    """Atajo: construye el checkpoint de una etapa y carga su estado previo si es válido."""
    return Checkpoint(etapa, config, claves_relevantes).cargar(verbose=verbose)
