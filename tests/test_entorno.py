"""Pruebas de la verificación del entorno de ejecución.

Lo que está en juego acá es un chequeo que tiene que fallar cuando corresponde y **sólo** cuando
corresponde. Un verificador que no detecta un numpy 2 deja pasar la etapa y el error aparece una
hora más tarde disfrazado de otra cosa; uno que grita sin motivo se termina desactivando, y
entonces no verifica nada.

El caso que más importa es la comparación de versiones: `torch` reporta `2.5.1+cu121` y `numpy`
reporta `1.26.4`. Comparados como texto, "1.9" > "1.26", que es la clase de error silencioso que
hace pasar por bueno un entorno roto.

Uso:
    python3 tests/test_entorno.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from equinos import entorno  # noqa: E402

fallos = []


def check(descripcion, condicion, detalle=""):
    if condicion:
        print(f"  OK    {descripcion}")
    else:
        print(f"  FALLA {descripcion}" + (f" -> {detalle}" if detalle else ""))
        fallos.append(descripcion)


# ============================================================
print("[1] Las versiones se comparan como números, no como texto")
# ============================================================
check("1.26 es posterior a 1.9", entorno._clave("1.26.4") > entorno._clave("1.9"))
check("el sufijo local de torch no altera el orden",
      entorno._clave("2.5.1+cu121") == entorno._clave("2.5.1"))
check("un sufijo de prerelease no rompe la comparación",
      entorno._clave("1.14.0rc1") >= entorno._clave("1.14"))
check("una versión sin números no explota", entorno._clave("desconocida") == (0,))


# ============================================================
print("\n[2] Los rangos abiertos y cerrados se respetan")
# ============================================================
check("numpy 1.26.4 entra en [1.18, 2.0)", entorno._en_rango("1.26.4", "1.18", "2.0"))
check("numpy 2.2.6 NO entra en [1.18, 2.0)", not entorno._en_rango("2.2.6", "1.18", "2.0"))
check("scipy 1.13.1 entra en [1.9, 1.14)", entorno._en_rango("1.13.1", "1.9", "1.14"))
check("scipy 1.15.3 NO entra en [1.9, 1.14)", not entorno._en_rango("1.15.3", "1.9", "1.14"))
check("torch 2.5.1+cu121 supera el mínimo 2.0", entorno._en_rango("2.5.1+cu121", "2.0", None))
check("torch 1.13 NO supera el mínimo 2.0", not entorno._en_rango("1.13", "2.0", None))
check("sin mínimo ni tope, cualquier versión vale", entorno._en_rango("0.0.1", None, None))


# ============================================================
print("\n[3] Toda pila declarada es verificable")
# ============================================================
# Un paquete listado en una pila pero ausente de REQUISITOS haría estallar `verificar` con un
# KeyError recién en ejecución, dentro de la etapa. Se detecta acá, que cuesta un milisegundo.
for pila, paquetes in entorno.PILAS.items():
    faltan = [p for p in paquetes if p not in entorno.REQUISITOS]
    check(f"la pila '{pila}' sólo nombra paquetes con requisito declarado",
          not faltan, f"sin requisito: {faltan}")

try:
    entorno.verificar("inexistente", verbose=False)
except ValueError:
    check("una pila desconocida falla con ValueError", True)
except SystemExit:
    check("una pila desconocida falla con ValueError", False, "lanzó SystemExit")
else:
    check("una pila desconocida falla con ValueError", False, "no falló")


# ============================================================
print("\n[4] version_de lee del disco sin importar el módulo")
# ============================================================
# El paquete de referencia se toma del propio intérprete y no se hardcodea: un venv creado con
# `uv` no trae `pip`, así que dar por sentado cualquier nombre concreto convierte la prueba en
# una afirmación sobre cómo se armó el entorno, que no es lo que se quiere verificar acá.
from importlib import metadata as _metadata  # noqa: E402

_instalado = next((d.metadata["Name"] for d in _metadata.distributions()), None)
check("un paquete instalado devuelve versión",
      _instalado is not None and entorno.version_de(_instalado) is not None,
      f"referencia: {_instalado}")
check("un paquete ausente devuelve None, no explota",
      entorno.version_de("paquete-que-no-existe-en-ningun-lado") is None)
# El módulo se importa antes que numpy y no debe arrastrarlo: si lo hiciera, en Colab quedaría
# numpy cargado en memoria antes de la celda que lo reemplaza en disco, que es el origen exacto
# de "numpy.dtype size changed".
check("importar `entorno` no carga numpy ni torch",
      "numpy" not in sys.modules and "torch" not in sys.modules)


# ============================================================
print("\n[5] La pila básica del intérprete actual")
# ============================================================
# No se afirma que pase: este test tiene que poder correrse desde el Python del sistema, donde
# a propósito NO está el entorno del pipeline. Lo que se verifica es que el veredicto sea
# coherente con lo que hay instalado, y que cuando falla lo haga con instrucciones.
faltantes = [p for p in entorno.PILAS["basica"] if entorno.version_de(p) is None]
try:
    entorno.verificar("basica", verbose=False)
except SystemExit as error:
    check("si falla, el mensaje dice cómo reconstruir el entorno",
          "requirements-local.txt" in str(error) or "Colab" in str(error))
    print(f"        (este intérprete no cumple la pila básica: {sys.executable})")
else:
    check("si pasa, es porque no faltaba ninguno", not faltantes)


print()
if fallos:
    print(f"{len(fallos)} prueba(s) fallaron:")
    for f in fallos:
        print(f"  - {f}")
    sys.exit(1)

print("Todas las pruebas pasaron.")
