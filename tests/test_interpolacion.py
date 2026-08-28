"""Pruebas del relleno temporal de keypoints y de su validación.

Lo que hay que poder detectar acá no es una excepción: es que el relleno estime mal en silencio,
o que la medición del error dé bien porque se está comparando algo contra sí mismo. Ese error ya
se cometió una vez en el proyecto —el Paso 16 informaba 1e-13 px midiendo la transformada contra
los mismos puntos con que se la había construido— y es el que estas pruebas vigilan.

No requieren GPU, ni los CSV de una corrida: los casos se construyen a mano.

Uso:
    python3 tests/test_interpolacion.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from equinos import interpolacion as it  # noqa: E402

fallos = []


def check(nombre, condicion, detalle=""):
    if condicion:
        print(f"  OK    {nombre}")
    else:
        print(f"  FALLA {nombre}  {detalle}")
        fallos.append(nombre)


def clip_con(mascara, paso=(10.0, 4.0), origen=(100.0, 200.0), sep=(0.0, 60.0)):
    """Clip sintético donde la cabeza se mueve en línea recta a velocidad constante.

    Sobre un movimiento lineal la interpolación tiene que ser exacta: es el único caso donde se
    conoce la respuesta, y por eso es el que hace falta para verificar la aritmética.
    """
    clip = it.Clip(nombre="sintetico")
    for i, hay in enumerate(mascara):
        clip.ordenes.append(i)
        if hay:
            ojo = (origen[0] + paso[0] * i, origen[1] + paso[1] * i)
            clip.ojo.append(ojo)
            clip.hocico.append((ojo[0] + sep[0], ojo[1] + sep[1]))
        else:
            clip.ojo.append(None)
            clip.hocico.append(None)
    return clip


print("=== huecos_interiores ===")

clip = clip_con([1, 0, 1])
check("un hueco de 1 entre dos válidos", it.huecos_interiores(clip.validos) == [(0, 2, 1)])

clip = clip_con([1, 0, 0, 0, 1])
check("un hueco de 3 se informa con su largo", it.huecos_interiores(clip.validos) == [(0, 4, 3)])

clip = clip_con([0, 0, 1, 1, 0, 0])
check("los extremos NO son huecos interiores", it.huecos_interiores(clip.validos) == [])

clip = clip_con([1, 0, 1, 0, 1])
check("dos huecos separados", it.huecos_interiores(clip.validos) == [(0, 2, 1), (2, 4, 1)])

check("clip sin detecciones no tiene huecos interiores",
      it.huecos_interiores(clip_con([0, 0, 0]).validos) == [])


print("\n=== interpolar_punto ===")

check("el punto medio de un hueco de 1 es el promedio",
      it.interpolar_punto((0, 0), (10, 20), 0, 2, 1) == (5.0, 10.0))
check("respeta la posición dentro de un hueco largo",
      it.interpolar_punto((0, 0), (40, 0), 0, 4, 1) == (10.0, 0.0))
check("coincide con el extremo izquierdo en t=0",
      it.interpolar_punto((3, 7), (9, 1), 0, 3, 0) == (3.0, 7.0))


print("\n=== rellenar_clip: el tope se respeta ===")

clip = clip_con([1, 0, 1])
rescatados = it.rellenar_clip(clip, max_hueco=1)
check("rescata el hueco de 1 cuando el tope es 1", set(rescatados) == {1})
check("sobre movimiento lineal la interpolación es exacta",
      np.allclose(rescatados[1]["ojo"], (110.0, 204.0)),
      f"dio {rescatados[1]['ojo']}")
check("marca el origen como interpolada", rescatados[1]["origen"] == "interpolada")

clip = clip_con([1, 0, 0, 1])
check("un hueco de 2 NO se rescata con tope 1", it.rellenar_clip(clip, max_hueco=1) == {})
check("un hueco de 2 SÍ se rescata con tope 2",
      set(it.rellenar_clip(clip, max_hueco=2)) == {1, 2})

check("con tope 0 no se rescata nada",
      it.rellenar_clip(clip_con([1, 0, 1]), max_hueco=0) == {})

clip = clip_con([0, 0, 1, 1, 0, 0])
sin_extremos = it.rellenar_clip(clip, max_hueco=5, max_extremo=0)
check("los extremos no se tocan si su tope es 0", sin_extremos == {})

con_extremos = it.rellenar_clip(clip, max_hueco=5, max_extremo=1)
check("con tope de extremo 1 se sostienen los dos bordes contiguos",
      set(con_extremos) == {1, 4})
check("el extremo se marca aparte, no como interpolada",
      all(v["origen"] == "extremo_sostenido" for v in con_extremos.values()))
check("el extremo sostiene el valor del vecino, no lo interpola",
      con_extremos[1]["ojo"] == clip.ojo[2])

check("clip sin ninguna detección no rescata nada",
      it.rellenar_clip(clip_con([0, 0, 0, 0]), max_hueco=5, max_extremo=2) == {})


print("\n=== error_de_estimacion: la medición no puede dar cero sola ===")

geometria = it.cargar_geometria()

ojo, hocico = (100.0, 100.0), (100.0, 400.0)

exacto = it.error_de_estimacion(ojo, hocico, ojo, hocico, geometria)
check("estimación idéntica a la real da error ~0",
      exacto["px_recorte"] < 1e-6 and exacto["px_original"] < 1e-6)

# La prueba que importa: si la estimación está corrida, el error DEBE aparecer. Una medición que
# compara la transformada contra los puntos con que se la construyó daría cero también acá.
corrido = it.error_de_estimacion((130.0, 100.0), hocico, ojo, hocico, geometria)
check("una estimación corrida 30 px produce error distinto de cero",
      corrido["px_recorte"] > 1.0, f"dio {corrido['px_recorte']:.4f} px")
check("el error crudo refleja el desplazamiento aplicado",
      abs(corrido["px_original"] - 30.0) < 1e-6, f"dio {corrido['px_original']:.4f}")
check("el error relativo usa la distancia ojo-hocico real",
      abs(corrido["frac_cara"] - 30.0 / 300.0) < 1e-9)

mas_corrido = it.error_de_estimacion((160.0, 100.0), hocico, ojo, hocico, geometria)
check("un desvío mayor produce un error mayor",
      mas_corrido["px_recorte"] > corrido["px_recorte"])

check("puntos de origen coincidentes no rompen la medición",
      np.isnan(it.error_de_estimacion(ojo, ojo, ojo, hocico, geometria)["px_recorte"]))


print("\n=== tope_desde_la_curva: se corta en el primer estrato que falla ===")

curva = [{"largo": 1, "cumple": True}, {"largo": 2, "cumple": True},
         {"largo": 3, "cumple": False}, {"largo": 4, "cumple": True}]
check("no acepta un largo posterior a uno que falló", it.tope_desde_la_curva(curva) == 2)
check("si falla el primero el tope es 0",
      it.tope_desde_la_curva([{"largo": 1, "cumple": False}]) == 0)
check("una curva vacía da tope 0", it.tope_desde_la_curva([]) == 0)
check("acepta todos los estratos si todos cumplen",
      it.tope_desde_la_curva([{"largo": 1, "cumple": True},
                              {"largo": 2, "cumple": True}]) == 2)


print("\n=== validar_huecos_interiores: no mide nada contra sí mismo ===")

# Movimiento lineal: la interpolación es exacta, así que el error medido tiene que ser ~0. Si
# diera algo distinto, el protocolo estaría interpolando desde el frame que dice ocultar.
recta = [clip_con([1] * 8)]
filas = it.validar_huecos_interiores(recta, geometria, max_largo=3)
check("sobre movimiento lineal el error medido es ~0",
      max(f["px_recorte"] for f in filas) < 1e-6,
      f"máx {max(f['px_recorte'] for f in filas):.4f}")
check("estratifica por largo de hueco", sorted({f["largo"] for f in filas}) == [1, 2, 3])

# Movimiento NO lineal: la interpolación tiene que errar. Si diera cero acá, la validación sería
# incapaz de rechazar nada.
curvo = it.Clip(nombre="curvo")
for i in range(8):
    curvo.ordenes.append(i)
    curvo.ojo.append((100.0 + 20.0 * i, 200.0 + 15.0 * i * i))
    curvo.hocico.append((100.0 + 20.0 * i, 260.0 + 15.0 * i * i))
filas_curvas = it.validar_huecos_interiores([curvo], geometria, max_largo=2)
check("sobre movimiento acelerado el error medido es > 0",
      min(f["px_recorte"] for f in filas_curvas) > 0.0)

check("un clip con menos de 3 detecciones no aporta filas",
      it.validar_huecos_interiores([clip_con([1, 1, 0, 0])], geometria) == [])


print("\n=== 2.5 geometría del recorte rescatado, contra un caso conocido ===")

# El caso conocido: la cabeza se mueve en línea recta y falta el frame del medio. Sus keypoints
# interpolados son exactamente el punto medio, y el recorte construido con ellos tiene que dejar
# esos puntos EN la posición canónica — la misma que tendría un frame detectado.
#
# Esta prueba existe por `calidad_recortes()`, cuya primera versión tenía la rotación traspuesta
# y erraba 240 px sin romper nada. Un recorte mal construido no lanza excepciones.
clip = clip_con([1, 0, 1])
rescate = it.rellenar_clip(clip, max_hueco=1)[1]

destino = (it.GEOMETRIA_RECORTE["dest_ojo"], it.GEOMETRIA_RECORTE["dest_hocico"])
M = geometria["similitud_desde_dos_puntos"](rescate["ojo"], rescate["hocico"], *destino)


def aplicar(matriz, punto):
    return matriz[:, :2] @ np.asarray(punto, dtype=float) + matriz[:, 2]


check("el ojo estimado cae en su posición canónica",
      np.allclose(aplicar(M, rescate["ojo"]), destino[0], atol=1e-9))
check("el hocico estimado cae en su posición canónica",
      np.allclose(aplicar(M, rescate["hocico"]), destino[1], atol=1e-9))

# Y la contraparte, que es la que da valor a la anterior: la posición REAL del frame oculto —que
# el relleno no vio— también tiene que caer cerca, porque el movimiento es lineal.
real_ojo = (110.0, 204.0)
check("sobre movimiento lineal la posición real cae en la canónica",
      np.allclose(aplicar(M, real_ojo), destino[0], atol=1e-9))

# El recorte tiene que quedar dentro del lienzo de 224: si la escala saliera invertida, los
# puntos canónicos caerían fuera y el recorte sería basura sin que nada falle.
for punto in destino:
    check(f"el destino {punto} está dentro del recorte "
          f"{it.GEOMETRIA_RECORTE['ancho']}x{it.GEOMETRIA_RECORTE['alto']}",
          0 <= punto[0] <= it.GEOMETRIA_RECORTE["ancho"]
          and 0 <= punto[1] <= it.GEOMETRIA_RECORTE["alto"])

check("la separación canónica es la declarada",
      abs((destino[1][1] - destino[0][1]) - it.GEOMETRIA_RECORTE["px_por_d"]) < 1e-9)


print("\n=== clip_desde_filas / rescates_por_clip: el puente con el notebook ===")

filas = [
    {"orden": 0, "par_valido": True, "x_ojo": 100.0, "y_ojo": 200.0,
     "x_hocico": 100.0, "y_hocico": 260.0},
    {"orden": 1, "par_valido": False, "x_ojo": float("nan"), "y_ojo": float("nan"),
     "x_hocico": float("nan"), "y_hocico": float("nan")},
    {"orden": 2, "par_valido": True, "x_ojo": 120.0, "y_ojo": 208.0,
     "x_hocico": 120.0, "y_hocico": 268.0},
]
puente = it.clip_desde_filas("c", filas)
check("respeta par_valido de la fila", puente.validos == [0, 2])

rescates = it.rescates_por_clip({"c": filas}, max_hueco=1)
check("la clave usa el orden real del frame, no el índice", list(rescates) == [("c", 1)])
check("el punto rescatado es el medio de sus vecinos",
      np.allclose(rescates[("c", 1)]["ojo"], (110.0, 204.0)))

sin_par_valido = [{k: v for k, v in f.items() if k != "par_valido"} | {"conf_ojo": 0.9, "conf_hocico": 0.9}
                  for f in filas]
sin_par_valido[1]["conf_ojo"] = 0.1
check("sin par_valido lo deriva de las confianzas",
      it.clip_desde_filas("c", sin_par_valido).validos == [0, 2])

check("con topes en 0 el puente no rescata nada",
      it.rescates_por_clip({"c": filas}, max_hueco=0, max_extremo=0) == {})


print("\n" + "=" * 62)
if fallos:
    print(f"FALLARON {len(fallos)}: {fallos}")
    sys.exit(1)
print("TODAS LAS PRUEBAS PASARON")
