"""Pruebas del conjunto de clips y de la partición por ejemplar.

Lo que hay que poder detectar acá no es una excepción: es una partición que **parece** bien hecha
y produce una métrica optimista. Los tres fallos que no dan error son:

  - que un ejemplar caiga en dos folds — el modelo memoriza el caballo y la métrica sube;
  - que se agrupe por el par ejemplar-clase en vez de por el ejemplar, que es la forma sutil del
    mismo fallo, porque hay 10 caballos con clips íntegros en más de una clase;
  - que entre un clip a medio alinear, que mezcla recortes de keypoints con recortes de caja
    dentro de la misma muestra.

Ninguno rompe nada. Los tres inflan el resultado.

No requieren GPU, ni internet, ni los CSV de una corrida: los casos se construyen a mano.

Uso:
    python3 tests/test_particion.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from equinos import particion as pa  # noqa: E402

fallos = []


def check(nombre, condicion, detalle=""):
    if condicion:
        print(f"  OK    {nombre}")
    else:
        print(f"  FALLA {nombre}  {detalle}")
        fallos.append(nombre)


def clips_de(spec):
    """spec: lista de (clip, clase, caballo). Todos íntegros y detectados."""
    return [{"clip": c, "clase": cl, "id_caballo": cab,
             "origenes": ["keypoints"], "solo_detectado": True}
            for c, cl, cab in spec]


print("=== construir: el grupo es el EJEMPLAR, no el par ejemplar-clase ===")

# El caballo 07 tiene clips en dos clases. Si se agrupara por el par, sus clips podrían caer en
# folds distintos: el mismo animal de los dos lados del corte.
mixto = clips_de([("a", "Sereno", "07"), ("b", "Disgustado", "07"),
                  ("c", "Sereno", "08"), ("d", "Interesado", "09"),
                  ("e", "Sereno", "10"), ("f", "Disgustado", "11")])
p = pa.construir(mixto, folds=2)
folds_de_07 = {c["fold"] for c in p if c["id_caballo"] == "07"}
check("los clips de un ejemplar en dos clases caen en UN solo fold",
      len(folds_de_07) == 1, f"cayeron en {folds_de_07}")

check("todos los clips reciben fold", all("fold" in c for c in p))
check("no se pierde ni se duplica ningún clip",
      sorted(c["clip"] for c in p) == ["a", "b", "c", "d", "e", "f"])


print("\n=== verificar: la fuga de ejemplar tiene que ser un error, no un número raro ===")

info = pa.verificar(p, folds=2)
check("una partición correcta pasa la verificación", info["sin_fuga"])
check("informa clips por fold", sum(info["clips_por_fold"].values()) == 6)

# Fuga construida a mano: el mismo caballo en dos folds.
con_fuga = [dict(c) for c in p]
con_fuga[0]["fold"] = 0
con_fuga[1]["fold"] = 1
con_fuga[0]["id_caballo"] = con_fuga[1]["id_caballo"] = "99"
try:
    pa.verificar(con_fuga, folds=2)
    check("la fuga de ejemplar levanta AssertionError", False, "no levantó nada")
except AssertionError as error:
    check("la fuga de ejemplar levanta AssertionError", "99" in str(error), str(error))


print("\n=== construir: determinismo, que es lo que vuelve comparables dos corridas ===")

a = pa.construir(mixto, folds=2)
b = pa.construir(list(reversed(mixto)), folds=2)
check("el resultado no depende del orden de entrada",
      {c["clip"]: c["fold"] for c in a} == {c["clip"]: c["fold"] for c in b})
check("dos construcciones seguidas dan lo mismo",
      {c["clip"]: c["fold"] for c in a}
      == {c["clip"]: c["fold"] for c in pa.construir(mixto, folds=2)})

# El desempate por nombre importa: dos ejemplares con la misma cantidad de clips no pueden quedar
# a merced del orden de iteración del diccionario.
empatados = clips_de([("x", "Sereno", "02"), ("y", "Sereno", "01"),
                      ("z", "Sereno", "03"), ("w", "Sereno", "04")])
uno = {c["clip"]: c["fold"] for c in pa.construir(empatados, folds=2)}
otro = {c["clip"]: c["fold"] for c in pa.construir(list(reversed(empatados)), folds=2)}
check("con ejemplares empatados el desempate es estable", uno == otro, f"{uno} vs {otro}")

check("reparte la carga entre folds",
      sorted(pa.verificar(pa.construir(empatados, folds=2), folds=2)["clips_por_fold"].values())
      == [2, 2])

try:
    pa.construir(mixto, folds=1)
    check("menos de 2 folds levanta ValueError", False, "no levantó nada")
except ValueError:
    check("menos de 2 folds levanta ValueError", True)


print("\n=== composicion: los ejemplares en varias clases hay que poder nombrarlos ===")

comp = pa.composicion(mixto)
check("cuenta clips", comp["clips"] == 6)
check("cuenta ejemplares distintos", comp["ejemplares"] == 5)
check("detecta al ejemplar que está en dos clases",
      comp["ejemplares_en_varias_clases"] == ["07"], comp["ejemplares_en_varias_clases"])
check("los ejemplares por clase NO suman el total",
      sum(comp["ejemplares_por_clase"].values()) > comp["ejemplares"])


print("\n=== clases vacías en un fold: se informan, no se esconden ===")

# Una clase con un solo ejemplar no puede estar en los dos folds.
raro = clips_de([("a", "Sereno", "01"), ("b", "Sereno", "02"),
                 ("c", "Disgustado", "03")])
info_raro = pa.verificar(pa.construir(raro, folds=2), folds=2)
check("un fold sin la clase minoritaria queda informado",
      any("Disgustado" in v for v in info_raro["clases_vacias"].values()),
      info_raro["clases_vacias"])


print("\n=== tabla_clips: qué entra y qué no ===")

FRAMES = []
def agregar(clip, clase, caballo, alineados, origen="keypoints"):
    for i in range(pa.FRAMES_POR_CLIP):
        FRAMES.append({"clip": clip, "clase": clase, "id_caballo": caballo,
                       "orden": str(i), "alineado": str(i < alineados),
                       "origen_alineacion": origen if i < alineados else "caja"})

agregar("completo", "Sereno", "01", 16)
agregar("casi", "Sereno", "02", 15)                      # 15 de 16: NO entra
agregar("interpolado", "Disgustado", "03", 16, "interpolada")
agregar("manual", "Disgustado", "04", 16, "manual")
agregar("corto", "Sereno", "05", 12)                     # 12 de 16: tampoco entra

pa._leer_frames = lambda archivo=None: FRAMES              # noqa: SLF001

todos = pa.tabla_clips()
check("un clip con los 16 frames alineados entra",
      "completo" in [c["clip"] for c in todos])
check("un clip con 15 de 16 NO entra — mezclaría dos geometrías en una muestra",
      "casi" not in [c["clip"] for c in todos], [c["clip"] for c in todos])
# El criterio estuvo en `12 <= n <= 16` por un tiempo, y eso dejaba entrar clips con hasta 4
# recortes de CAJA sin alinear. `caracteristicas.extraer` no filtra por `alineado`, así que el
# backbone comía dos geometrías dentro de la misma secuencia sin que nada lo marcara.
check("un clip con 12 de 16 tampoco entra",
      "corto" not in [c["clip"] for c in todos], [c["clip"] for c in todos])
check("los clips rescatados por interpolación entran", "interpolado" in [c["clip"] for c in todos])

detectados = pa.tabla_clips(solo_detectados=True)
check("la ablación deja sólo los de detección propia",
      [c["clip"] for c in detectados] == ["completo"], [c["clip"] for c in detectados])
check("y marca el origen en la tabla completa",
      {c["clip"]: c["solo_detectado"] for c in todos}
      == {"completo": True, "interpolado": False, "manual": False})


print("\n" + "=" * 52)
if fallos:
    print(f"FALLARON {len(fallos)}: {fallos}")
    sys.exit(1)
print("TODAS LAS PRUEBAS PASARON")
