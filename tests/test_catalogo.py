"""Pruebas del catálogo de videos por clase.

Se prueba con un árbol de archivos falso —archivos vacíos con extensión de video— porque lo que
está en juego no es el contenido de ningún video sino **qué archivo cuenta como de qué clase**.
Ese es el tipo de error que no rompe nada: produce un dataset con menos material del que hay, o
con material de una carpeta de trabajo entrando como etiqueta, y ambos se ven igual de bien.

Uso:
    python3 tests/test_catalogo.py
"""
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from equinos import catalogo  # noqa: E402

fallos = []


def check(descripcion, condicion, detalle=""):
    if condicion:
        print(f"  OK    {descripcion}")
    else:
        print(f"  FALLA {descripcion}" + (f" -> {detalle}" if detalle else ""))
        fallos.append(descripcion)


BANCO = Path(tempfile.mkdtemp(prefix="equinos_catalogo_"))

SUBCARPETAS = {
    "Disgustado": ("Seleccionar cabeza?", "Se tapa una parte de la cara"),
    "Interesado": ("Seleccionar cabeza?",),
    "Sereno": ("Seleccionar cabeza?",),
}


def crear(base, *relativas):
    """Crea archivos vacíos en las rutas indicadas, con sus directorios."""
    for relativa in relativas:
        destino = base / relativa
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.touch()


def nombres(inventario, etiqueta):
    return sorted(r.name for r in inventario.videos[etiqueta])


# ============================================================
print("\n[1] Forma canónica de los nombres de carpeta")
# ============================================================
# En Drive la carpeta es 'Seleccionar cabeza?'; en una copia bajada como ZIP, 'Seleccionar
# cabeza_'. Comparar literalmente haría andar el código en un entorno y fallar callado en el otro.

check("el signo de pregunta y el guión bajo colapsan",
      catalogo.clave("Seleccionar cabeza?") == catalogo.clave("Seleccionar cabeza_"))
check("las mayúsculas no importan",
      catalogo.clave("SELECCIONAR CABEZA?") == catalogo.clave("seleccionar cabeza"))
check("los acentos no importan",
      catalogo.clave("Selección") == catalogo.clave("Seleccion"))
check("nombres distintos siguen siendo distintos",
      catalogo.clave("Seleccionar cabeza?") != catalogo.clave("Separar cabeza?"))


# ============================================================
print("\n[2] Cada clase suma las subcarpetas que declara, y sólo esas")
# ============================================================
base = BANCO / "arbol"
crear(
    base,
    "Disgustado/a.mp4",
    "Disgustado/Seleccionar cabeza_/b.mp4",           # declarada, grafía saneada
    "Disgustado/Se tapa una parte de la cara/c.mp4",  # declarada sólo para esta clase
    "Disgustado/Descartados/x.mp4",                   # no declarada
    "Interesado/d.mp4",
    "Interesado/Seleccionar cabeza?/e.mp4",           # declarada, grafía de Drive
    "Interesado/Se tapa una parte de la cara/y.mp4",  # NO declarada para Interesado
    "Sereno/f.mp4",
    "Sereno/Separar cabeza_/z.mp4",                   # no declarada en ninguna clase
)

inv = catalogo.inventariar(base=base, subcarpetas_por_clase=SUBCARPETAS)

check("Disgustado suma sus dos subcarpetas", nombres(inv, "Disgustado") == ["a.mp4", "b.mp4", "c.mp4"],
      nombres(inv, "Disgustado"))
check("Interesado suma sólo la suya", nombres(inv, "Interesado") == ["d.mp4", "e.mp4"],
      nombres(inv, "Interesado"))
check("una subcarpeta declarada en otra clase NO entra acá",
      "y.mp4" not in nombres(inv, "Interesado"))
check("Sereno no suma una subcarpeta que no declaró", nombres(inv, "Sereno") == ["f.mp4"],
      nombres(inv, "Sereno"))
check("el total es la suma de las tres", inv.total == 6, inv.total)


# ============================================================
print("\n[3] Lo que queda afuera se informa, no desaparece")
# ============================================================
excluidos = {e: sorted(r.name for r in v) for e, v in inv.excluidos.items()}
check("los videos de una subcarpeta no declarada se listan",
      excluidos == {"Disgustado": ["x.mp4"], "Interesado": ["y.mp4"], "Sereno": ["z.mp4"]},
      excluidos)
check("una subcarpeta declarada que no existe se informa",
      inv.declaradas_ausentes.get("Sereno") == ["Seleccionar cabeza?"],
      inv.declaradas_ausentes)
check("el aporte de cada carpeta queda registrado",
      inv.aportes["Disgustado"]["Disgustado/Seleccionar cabeza_"] == 1,
      inv.aportes["Disgustado"])


# ============================================================
print("\n[4] Carpetas hermanas que no son de clase no entran")
# ============================================================
# La regla anterior tomaba la clase del nombre de la carpeta contenedora sobre un recorrido
# recursivo, así que una descarga parcial de Drive llamada 'Sereno-20260819T000125Z-1-001/Sereno'
# entraba entera como clase Sereno, sin aviso. Acá se comprueba que ya no.
crear(base, "Sereno-20260819T000125Z-1-001/Sereno/intruso.mp4")

inv2 = catalogo.inventariar(base=base, subcarpetas_por_clase=SUBCARPETAS)
check("una carpeta de descarga con una subcarpeta homónima no aporta nada",
      "intruso.mp4" not in nombres(inv2, "Sereno"), nombres(inv2, "Sereno"))


# ============================================================
print("\n[5] Sólo se cuentan archivos con extensión de video")
# ============================================================
crear(base, "Sereno/notas.txt", "Sereno/miniatura.jpg")
inv3 = catalogo.inventariar(base=base, subcarpetas_por_clase=SUBCARPETAS)
check("los archivos que no son video se ignoran", nombres(inv3, "Sereno") == ["f.mp4"],
      nombres(inv3, "Sereno"))


# ============================================================
print("\n[6] Clase declarada sin carpeta en el disco")
# ============================================================
parcial = BANCO / "parcial"
crear(parcial, "Sereno/f.mp4")
inv4 = catalogo.inventariar(base=parcial, subcarpetas_por_clase=SUBCARPETAS)
check("se informa qué clases no tienen carpeta",
      sorted(inv4.clases_ausentes) == ["Disgustado", "Interesado"], inv4.clases_ausentes)
check("la clase ausente queda con lista vacía, no ausente del diccionario",
      inv4.videos["Disgustado"] == [])


# ============================================================
print("\n[7] verificar() corta ante un catálogo inservible")
# ============================================================
vacia = BANCO / "vacia"
crear(vacia, "Sereno/notas.txt")

try:
    catalogo.verificar(catalogo.inventariar(base=vacia, subcarpetas_por_clase=SUBCARPETAS))
except SystemExit:
    check("un catálogo sin ningún video corta la etapa", True)
else:
    check("un catálogo sin ningún video corta la etapa", False, "no lanzó SystemExit")

# El mismo archivo suelto en la raíz de la clase y adentro de una subcarpeta declarada: dos rutas,
# un solo nombre. El nombre es la clave del checkpoint y la base del nombre de los clips.
duplicada = BANCO / "duplicada"
crear(duplicada, "Sereno/f.mp4", "Sereno/Seleccionar cabeza_/f.mp4")

try:
    catalogo.verificar(catalogo.inventariar(base=duplicada, subcarpetas_por_clase=SUBCARPETAS))
except SystemExit as error:
    check("un nombre repetido entre la clase y su subcarpeta corta la etapa", True)
    check("el mensaje nombra el archivo en conflicto", "f.mp4" in str(error))
else:
    check("un nombre repetido entre la clase y su subcarpeta corta la etapa", False,
          "no lanzó SystemExit")

# Un catálogo sano no debe cortar.
try:
    catalogo.verificar(inv3)
except SystemExit as error:
    check("un catálogo sano pasa la verificación", False, str(error))
else:
    check("un catálogo sano pasa la verificación", True)


# ============================================================
print("\n[8] catalogar() devuelve el diccionario pedido")
# ============================================================
resultado = catalogo.catalogar(base=base, subcarpetas_por_clase=SUBCARPETAS)
check("es un dict de clase a lista de rutas",
      isinstance(resultado, dict) and sorted(resultado) == ["Disgustado", "Interesado", "Sereno"]
      and all(isinstance(r, Path) for v in resultado.values() for r in v))
try:
    catalogo.catalogar(base=BANCO / "no_existe", subcarpetas_por_clase=SUBCARPETAS)
except SystemExit:
    check("una base inexistente falla con un mensaje, no con un catálogo vacío", True)
else:
    check("una base inexistente falla con un mensaje, no con un catálogo vacío", False,
          "no lanzó SystemExit")


shutil.rmtree(BANCO, ignore_errors=True)

print()
if fallos:
    print(f"{len(fallos)} prueba(s) fallaron:")
    for f in fallos:
        print(f"  - {f}")
    sys.exit(1)

print("Todas las pruebas pasaron.")
