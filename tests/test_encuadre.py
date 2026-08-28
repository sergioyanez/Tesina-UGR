"""Pruebas del campo de visión del recorte facial.

Lo que hay que poder detectar acá no es una excepción: es que el encuadre quede mal medido en
silencio. Un signo cambiado en el ángulo de la transformada no rompe nada —los dos puntos de
anclaje salen bien igual, porque son los que la definen— y sólo se manifiesta como una métrica de
orejas o de relleno equivocada, que es exactamente el número con el que se va a decidir el
encuadre. Ese error ya ocurrió una vez en el proyecto: el Paso 16 informaba 1e-13 px porque medía
la transformada contra los mismos puntos con que se la había construido.

Por eso las pruebas de proyección usan puntos que NO son el ojo ni el hocico.

No requieren GPU, ni internet, ni los CSV de una corrida: los casos se construyen a mano.

Uso:
    python3 tests/test_encuadre.py
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from equinos.encuadre import (  # noqa: E402
    CAMPO_VIGENTE,
    CampoDeVision,
    dentro,
    destino,
    evaluar,
    proyectar,
    relleno,
)

fallos = []


def check(nombre, condicion, detalle=""):
    if condicion:
        print(f"  OK    {nombre}")
    else:
        print(f"  FALLA {nombre}  {detalle}")
        fallos.append(nombre)


# Encuadre de juguete con números redondos, para que las cuentas de las pruebas se puedan hacer a
# mano: 1 d de aire arriba, 1 d abajo, y 1,5 d de ancho total. Extensión axial 3 d, aspecto 2,0.
JUGUETE = CampoDeVision(aire_sobre_ojo=1.0, aire_bajo_hocico=1.0, semiancho=0.75)


print("=== CampoDeVision: geometría declarada ===")

check("extensión axial = arriba + 1 + abajo", abs(JUGUETE.extension_axial - 3.0) < 1e-12,
      f"dio {JUGUETE.extension_axial}")
check("aspecto = axial / (2 * semiancho)", abs(JUGUETE.aspecto - 2.0) < 1e-12,
      f"dio {JUGUETE.aspecto}")
check("frac_ojo = aire_arriba / axial", abs(JUGUETE.frac_ojo - 1 / 3) < 1e-12)
check("frac_hocico = (aire_arriba + 1) / axial", abs(JUGUETE.frac_hocico - 2 / 3) < 1e-12)

# El encuadre vigente tiene que reproducir exactamente los FRAC_OJO/FRAC_HOCICO históricos, o el
# barrido no estaría comparando contra la línea de base que dice comparar.
check("CAMPO_VIGENTE reproduce FRAC_OJO = 0,20", abs(CAMPO_VIGENTE.frac_ojo - 0.20) < 1e-12,
      f"dio {CAMPO_VIGENTE.frac_ojo}")
check("CAMPO_VIGENTE reproduce FRAC_HOCICO = 0,80",
      abs(CAMPO_VIGENTE.frac_hocico - 0.80) < 1e-12, f"dio {CAMPO_VIGENTE.frac_hocico}")
check("CAMPO_VIGENTE es cuadrado", abs(CAMPO_VIGENTE.aspecto - 1.0) < 1e-12,
      f"dio {CAMPO_VIGENTE.aspecto}")


print("\n=== destino: el alto se deriva, no se elige ===")

g = destino(JUGUETE, ancho=100)
check("alto = ancho * aspecto", g["alto"] == 200, f"dio {g['alto']}")
check("el ojo cae sobre el eje vertical central", abs(g["dest_ojo"][0] - 50.0) < 1e-12)
check("el hocico cae sobre el eje vertical central", abs(g["dest_hocico"][0] - 50.0) < 1e-12)
check("el ojo queda a frac_ojo del borde superior",
      abs(g["dest_ojo"][1] - 200 / 3) < 1e-9, f"dio {g['dest_ojo'][1]}")
check("d ocupa alto/axial píxeles", abs(g["px_por_d"] - 200 / 3) < 1e-9,
      f"dio {g['px_por_d']}")
check("el semiancho realizado coincide con el declarado",
      abs(g["semiancho_real"] - 0.75) < 1e-9, f"dio {g['semiancho_real']}")

g224 = destino(CAMPO_VIGENTE, ancho=224)
check("el encuadre vigente da un destino de 224x224", (g224["ancho"], g224["alto"]) == (224, 224),
      f"dio {g224['ancho']}x{g224['alto']}")
check("el encuadre vigente da d -> 134 px", abs(g224["px_por_d"] - 224 * 0.60) < 0.5,
      f"dio {g224['px_por_d']}")

try:
    destino(JUGUETE, ancho=0)
    check("ancho no positivo levanta ValueError", False, "no levantó nada")
except ValueError:
    check("ancho no positivo levanta ValueError", True)


print("\n=== proyectar: el par de anclaje y, sobre todo, lo que NO es el par ===")

# Cabeza vertical: ojo arriba, hocico 400 px abajo. d = 400.
OJO, HOCICO = (500.0, 300.0), (500.0, 700.0)

p_ojo = proyectar(OJO, *OJO, *HOCICO, g)
p_hoc = proyectar(HOCICO, *OJO, *HOCICO, g)
check("el ojo aterriza en su posición canónica",
      math.dist(p_ojo, g["dest_ojo"]) < 1e-9, f"dio {p_ojo}")
check("el hocico aterriza en su posición canónica",
      math.dist(p_hoc, g["dest_hocico"]) < 1e-9, f"dio {p_hoc}")

# El punto medio del eje: la prueba que sí puede fallar, porque no define la transformada.
medio = proyectar((500.0, 500.0), *OJO, *HOCICO, g)
esperado_medio = (g["dest_ojo"][0], (g["dest_ojo"][1] + g["dest_hocico"][1]) / 2)
check("el punto medio del eje cae en el punto medio canónico",
      math.dist(medio, esperado_medio) < 1e-9, f"dio {medio}, esperado {esperado_medio}")

# Un punto POR ENCIMA del ojo, que es donde viven las orejas. A 1 d sobre el ojo debe caer justo
# en el borde superior del recorte, porque el campo declara 1 d de aire arriba.
arriba = proyectar((500.0, 300.0 - 400.0), *OJO, *HOCICO, g)
check("un punto a 1 d sobre el ojo cae en el borde superior",
      abs(arriba[1] - 0.0) < 1e-9, f"dio y={arriba[1]}")

# Un punto lateral a 0,75 d del eje debe caer en el borde derecho.
lateral = proyectar((500.0 + 0.75 * 400.0, 300.0), *OJO, *HOCICO, g)
check("un punto a 0,75 d del eje cae en el borde lateral",
      abs(lateral[0] - g["ancho"]) < 1e-9, f"dio x={lateral[0]}")

# Invariancia a la pose: rotar y escalar la cabeza en el origen no puede mover nada en el destino.
# Es la propiedad que define la normalización, y la que un signo mal escrito rompe.
peor = 0.0
for grados in range(0, 360, 7):
    for escala in (0.3, 1.0, 3.7):
        rad = math.radians(grados)
        ojo_r = (1000.0, 1000.0)
        hoc_r = (1000.0 + 400 * escala * math.cos(rad), 1000.0 + 400 * escala * math.sin(rad))
        # El mismo punto físico: a 1 d sobre el ojo siguiendo el eje de la cabeza.
        punto_r = (1000.0 - 400 * escala * math.cos(rad), 1000.0 - 400 * escala * math.sin(rad))
        q = proyectar(punto_r, *ojo_r, *hoc_r, g)
        peor = max(peor, math.dist(q, (g["dest_ojo"][0], 0.0)))
check("la proyección es invariante a rotación y escala de la cabeza", peor < 1e-6,
      f"peor desvío {peor:.2e} px")

check("puntos de anclaje degenerados devuelven None",
      proyectar((0.0, 0.0), 10.0, 10.0, 10.0, 10.0, g) is None)


print("\n=== dentro: qué entra y qué no ===")

check("un punto interior está dentro", dentro((50.0, 100.0), g))
check("el origen está dentro", dentro((0.0, 0.0), g))
check("un punto por encima del recorte está fuera", not dentro((50.0, -1.0), g))
check("un punto por debajo del recorte está fuera", not dentro((50.0, 200.0), g))
check("un punto a la derecha del recorte está fuera", not dentro((100.0, 100.0), g))
check("None está fuera", not dentro(None, g))

# La prueba que le da sentido al change: con el campo de visión vigente, una oreja a 0,81 d sobre
# el ojo —la mediana medida— queda FUERA; con un campo que declara 1 d de aire arriba, entra.
oreja = (500.0, 300.0 - 0.81 * 400.0)
g_vigente = destino(CAMPO_VIGENTE, ancho=224)
check("con el encuadre vigente la oreja mediana queda fuera",
      not dentro(proyectar(oreja, *OJO, *HOCICO, g_vigente), g_vigente))
check("con 1 d de aire arriba la oreja mediana entra",
      dentro(proyectar(oreja, *OJO, *HOCICO, g), g))


print("\n=== relleno: cuánto del recorte se inventa ===")

# Cabeza en el medio de un frame enorme: el recorte entra entero. Ojo con la ubicación — el campo
# de juguete declara 1 d (400 px) de aire por encima del ojo, así que una cabeza con el ojo a 300
# px del borde superior NO entra entera, y la prueba mediría otra cosa.
CENTRO_OJO, CENTRO_HOCICO = (5000.0, 5000.0), (5000.0, 5400.0)
check("recorte enteramente dentro del frame da relleno 0",
      relleno(*CENTRO_OJO, *CENTRO_HOCICO, g, 10000, 10000) == 0.0,
      f"dio {relleno(*CENTRO_OJO, *CENTRO_HOCICO, g, 10000, 10000)}")

# El mismo recorte contra un frame de 1x1: no entra nada.
check("recorte enteramente fuera del frame da relleno 1",
      relleno(*CENTRO_OJO, *CENTRO_HOCICO, g, 1, 1) == 1.0,
      f"dio {relleno(*CENTRO_OJO, *CENTRO_HOCICO, g, 1, 1)}")

# Cabeza pegada al borde superior: parte del aire de arriba cae fuera, pero no todo el recorte.
parcial = relleno(500.0, 50.0, 500.0, 450.0, g, 1080, 1920)
check("cabeza contra el borde da relleno parcial", 0.0 < parcial < 1.0, f"dio {parcial}")

# Más campo de visión no puede dar MENOS relleno a igualdad de todo lo demás: es la compensación
# que este change acepta a cambio de las orejas, y tiene que estar bien medida.
angosto = destino(CampoDeVision(0.3, 0.3, 0.5), ancho=224)
amplio = destino(CampoDeVision(1.5, 1.5, 0.5), ancho=224)
r_angosto = relleno(500.0, 400.0, 500.0, 600.0, angosto, 1080, 1920)
r_amplio = relleno(500.0, 400.0, 500.0, 600.0, amplio, 1080, 1920)
check("ampliar el campo axial no reduce el relleno", r_amplio >= r_angosto,
      f"angosto {r_angosto:.3f} vs amplio {r_amplio:.3f}")

check("puntos de anclaje degenerados dan relleno NaN",
      relleno(10.0, 10.0, 10.0, 10.0, g, 100, 100) != relleno(10.0, 10.0, 10.0, 10.0, g, 100, 100))


print("\n=== evaluar: el vector de oreja, que es la métrica que decide ===")

# Un frame de juguete: cabeza vertical, d = 400, frame grande para que no haya relleno.
# La oreja izquierda tiene base a 0,5 d y punta a 1,5 d sobre el ojo; la derecha, base a 0,5 d y
# punta a 0,6 d. Con 1 d de aire arriba entra la oreja derecha entera y de la izquierda sólo la
# base: el conteo suelto de keypoints da 3/4, pero el vector legible da 1 de 1 frame.
FRAME = {
    "clip": "juguete", "orden": "0",
    "ojo": (5000.0, 5000.0), "hocico": (5000.0, 5400.0),
    "ancho_frame": 10000, "alto_frame": 10000,
    "orejas": {
        "left_earbase": (5000.0, 5000.0 - 0.5 * 400, 0.9),
        "left_earend": (5000.0, 5000.0 - 1.5 * 400, 0.9),
        "right_earbase": (5000.0, 5000.0 - 0.5 * 400, 0.9),
        "right_earend": (5000.0, 5000.0 - 0.6 * 400, 0.9),
    },
}
r = evaluar(JUGUETE, ancho=100, medibles=[FRAME])
check("cuenta los cuatro keypoints de oreja como medidos", r["orejas_medidas"] == 4,
      f"dio {r['orejas_medidas']}")
check("el conteo suelto de keypoints da 3 de 4", abs(r["orejas_dentro"] - 0.75) < 1e-12,
      f"dio {r['orejas_dentro']}")
check("el vector de oreja es legible en el frame", abs(r["vector_legible"] - 1.0) < 1e-12,
      f"dio {r['vector_legible']}")
check("no hay relleno con la cabeza en el medio de un frame enorme",
      r["relleno_p50"] == 0.0, f"dio {r['relleno_p50']}")

# La asimetría que motiva el change, aislada: un frame donde entran las DOS bases y ninguna de
# las dos puntas. El conteo suelto de keypoints informa 50 %, y el vector de oreja informa 0 %,
# que es lo correcto — sin punta no se puede leer hacia dónde apunta la oreja. Es la razón por la
# que el criterio se ancla en el vector y no en el conteo suelto.
SOLO_BASES = dict(FRAME, orejas={
    "left_earbase": (5000.0, 5000.0 - 0.5 * 400, 0.9),
    "left_earend": (5000.0, 5000.0 - 1.5 * 400, 0.9),
    "right_earbase": (5000.0, 5000.0 - 0.5 * 400, 0.9),
    "right_earend": (5000.0, 5000.0 - 1.5 * 400, 0.9),
})
r_bases = evaluar(JUGUETE, ancho=100, medibles=[SOLO_BASES])
check("con sólo las bases dentro, el conteo suelto informa 50 %",
      abs(r_bases["orejas_dentro"] - 0.5) < 1e-12, f"dio {r_bases['orejas_dentro']}")
check("con sólo las bases dentro, el vector de oreja informa 0 %",
      r_bases["vector_legible"] == 0.0, f"dio {r_bases['vector_legible']}")

# Con el encuadre vigente, que muestra 0,33 d por encima del ojo, no entra ni siquiera la base:
# la base mediana medida está a 0,47 d. Las dos métricas dan cero, y por eso el change existe.
r_vig = evaluar(CAMPO_VIGENTE, ancho=224, medibles=[FRAME])
check("con el encuadre vigente no entra ninguna oreja",
      r_vig["orejas_dentro"] == 0.0 and r_vig["vector_legible"] == 0.0,
      f"kp {r_vig['orejas_dentro']}, vector {r_vig['vector_legible']}")

# Una oreja sin punta detectada no cuenta como evaluable: no se puede premiar ni castigar a un
# encuadre por un vector que el modelo de pose nunca midió.
sin_punta = dict(FRAME, orejas={
    "left_earbase": FRAME["orejas"]["left_earbase"],
    "right_earbase": FRAME["orejas"]["right_earbase"],
})
check("un frame sin puntas detectadas no entra al denominador del vector",
      evaluar(JUGUETE, 100, [sin_punta])["frames_con_vector"] == 0)

# Un keypoint por debajo del umbral de confianza se ignora en las dos métricas.
flojo = dict(FRAME, orejas={k: (v[0], v[1], 0.1) for k, v in FRAME["orejas"].items()})
r_flojo = evaluar(JUGUETE, 100, [flojo])
check("los keypoints de baja confianza no se cuentan",
      r_flojo["orejas_medidas"] == 0 and r_flojo["frames_con_vector"] == 0)


print("\n" + "=" * 52)
if fallos:
    print(f"FALLARON {len(fallos)}: {fallos}")
    sys.exit(1)
print("TODAS LAS PRUEBAS PASARON")
