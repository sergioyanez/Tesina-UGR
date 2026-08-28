"""Pruebas de las funciones geométricas de 02_roi_y_normalizacion.ipynb.

Las funciones se **extraen del propio notebook** en lugar de duplicarse acá.
Así no hay dos copias que puedan divergir: lo que se prueba es exactamente el
código que se va a ejecutar en Colab.

No requieren GPU, internet ni pesos preentrenados: la geometría es pura, y es
justamente donde se esconden los errores silenciosos (un signo cambiado en una
rotación no rompe nada, solo produce recortes mal alineados en silencio).

Uso:
    python3 tests/test_geometria.py
"""
import json
import sys
from pathlib import Path

import numpy as np

NOTEBOOK = Path(__file__).resolve().parent.parent / "02_roi_y_normalizacion.ipynb"

FUNCIONES = ["suavizar_cajas", "rellenar_faltantes", "similitud_desde_dos_puntos",
             "roi_desde_keypoints"]


def cargar_geometria_del_notebook():
    """Busca la celda que define las funciones geométricas y la ejecuta."""
    if not NOTEBOOK.exists():
        raise SystemExit(f"No se encontró el notebook: {NOTEBOOK}")

    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))

    for i, celda in enumerate(nb["cells"]):
        if celda["cell_type"] != "code":
            continue
        fuente = "".join(celda["source"])
        if all(f"def {f}" in fuente for f in FUNCIONES):
            espacio = {}
            exec(compile(fuente, f"notebook_celda_{i}", "exec"), espacio)
            print(f"Geometría extraída de la celda {i} del notebook.\n")
            return espacio

    raise SystemExit("Ninguna celda del notebook define las funciones geométricas: "
                     + ", ".join(FUNCIONES))


G = cargar_geometria_del_notebook()
suavizar_cajas = G["suavizar_cajas"]
rellenar_faltantes = G["rellenar_faltantes"]
similitud_desde_dos_puntos = G["similitud_desde_dos_puntos"]
roi_desde_keypoints = G["roi_desde_keypoints"]

fallos = []


def check(nombre, condicion, detalle=""):
    if condicion:
        print(f"  OK    {nombre}")
    else:
        print(f"  FALLA {nombre}  {detalle}")
        fallos.append(nombre)


print("=== similitud_desde_dos_puntos ===")

# La propiedad que define la función: debe mandar p1->q1 y p2->q2 exactamente.
rng = np.random.default_rng(42)
peor = 0.0
for _ in range(2000):
    p1, p2, q1, q2 = rng.uniform(-500, 500, (4, 2))
    if np.hypot(*(p2 - p1)) < 1e-6:
        continue
    M = similitud_desde_dos_puntos(p1, p2, q1, q2)
    if M is None:
        continue
    ap1 = M[:, :2] @ p1 + M[:, 2]
    ap2 = M[:, :2] @ p2 + M[:, 2]
    peor = max(peor, np.hypot(*(ap1 - q1)), np.hypot(*(ap2 - q2)))
check("mapea los dos puntos exactamente (2000 casos aleatorios)",
      peor < 1e-6, f"error máximo {peor:.2e}")

# Debe ser una similitud real: preservar ángulos y escalar de forma uniforme.
M = similitud_desde_dos_puntos((0, 0), (10, 0), (5, 5), (5, 25))
A = M[:, :2]
check("preserva ortogonalidad (A^T A proporcional a la identidad)",
      np.allclose(A.T @ A, (A.T @ A)[0, 0] * np.eye(2)))
check("escala uniforme correcta (10 -> 20 es factor 2)",
      abs(np.sqrt((A.T @ A)[0, 0]) - 2.0) < 1e-9)

M = similitud_desde_dos_puntos((0, 0), (1, 0), (0, 0), (0, 1))
check("rotación de 90 grados", np.allclose(M[:, :2], [[0, -1], [1, 0]]))

check("devuelve None si los puntos de origen coinciden",
      similitud_desde_dos_puntos((3, 3), (3, 3), (0, 0), (1, 1)) is None)

print("\n=== alineación canónica (el uso real) ===")

# Tres poses muy distintas de la misma cabeza. Tras alinear, el ojo y el hocico
# deben caer SIEMPRE en el mismo lugar del recorte. Eso es la normalización.
#
# El destino canónico ya no lo calcula el notebook sino `equinos.encuadre.destino`, porque lo
# usan el notebook Y el barrido de encuadres, y dos copias que pueden divergir es lo que este
# proyecto evita. Sus propias pruebas están en tests/test_encuadre.py.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from equinos import encuadre  # noqa: E402

CAMPO = encuadre.CampoDeVision(1.00, 0.30, 0.75)   # el encuadre adoptado
GEO = encuadre.destino(CAMPO, 224)
ANCHO, ALTO = GEO["ancho"], GEO["alto"]
dest_ojo, dest_hocico = GEO["dest_ojo"], GEO["dest_hocico"]

poses = [
    ((100.0, 200.0), (140.0, 300.0)),    # perfil, cabeza chica
    ((800.0, 400.0), (600.0, 380.0)),    # girada casi horizontal, invertida
    ((500.0, 100.0), (520.0, 700.0)),    # frontal, cabeza grande
]
maximo = 0.0
for ojo, hocico in poses:
    M = similitud_desde_dos_puntos(ojo, hocico, dest_ojo, dest_hocico)
    o = M[:, :2] @ np.array(ojo) + M[:, 2]
    h = M[:, :2] @ np.array(hocico) + M[:, 2]
    maximo = max(maximo, np.hypot(*(o - dest_ojo)), np.hypot(*(h - dest_hocico)))
check("tres poses muy distintas caen en la misma posición canónica",
      maximo < 1e-6, f"error máximo {maximo:.2e}")
check("el destino queda dentro del recorte",
      0 < dest_ojo[0] < ANCHO and 0 < dest_hocico[0] < ANCHO
      and 0 < dest_ojo[1] < ALTO and 0 < dest_hocico[1] < ALTO)

# El recorte dejó de ser cuadrado, y la prueba que reemplaza a "es cuadrado" es la que de verdad
# importaba: que la transformada NO deforme. Lo que impide la deformación es la isotropía de la
# similitud, no la forma del destino — confundir ambas cosas es lo que ató históricamente el
# encuadre a un cuadrado.
check("el destino es más alto que ancho", ALTO > ANCHO, f"{ANCHO}x{ALTO}")

peor_razon = 0.0
for ojo, hocico in poses:
    M = similitud_desde_dos_puntos(ojo, hocico, dest_ojo, dest_hocico)
    # Dos segmentos cualesquiera del frame de origen, ninguno alineado con el eje ojo-hocico.
    a, b, c = np.array([10.0, 20.0]), np.array([310.0, 45.0]), np.array([70.0, 500.0])
    lleva = lambda p: M[:, :2] @ p + M[:, 2]
    razon_origen = np.hypot(*(b - a)) / np.hypot(*(c - a))
    razon_destino = np.hypot(*(lleva(b) - lleva(a))) / np.hypot(*(lleva(c) - lleva(a)))
    peor_razon = max(peor_razon, abs(razon_origen - razon_destino))
check("un destino rectangular no deforma: la razón entre distancias se conserva",
      peor_razon < 1e-9, f"peor desvío {peor_razon:.2e}")

# La escala aplicada tiene que ser idéntica en los dos ejes. Es la definición de isotropía, y es
# lo único que separa "recorte rectangular" de "cara estirada".
M = similitud_desde_dos_puntos(*poses[0], dest_ojo, dest_hocico)
escala_x = np.hypot(M[0, 0], M[1, 0])
escala_y = np.hypot(M[0, 1], M[1, 1])
check("la escala es la misma en ambos ejes", abs(escala_x - escala_y) < 1e-9,
      f"x {escala_x:.6f} vs y {escala_y:.6f}")

print("\n=== suavizar_cajas ===")

base = [(10.0, 10.0, 100.0, 100.0)] * 9
base[4] = (500.0, 500.0, 900.0, 900.0)          # detección aislada disparatada
suave = suavizar_cajas(base, ventana=5)
check("elimina el outlier aislado", suave[4] == (10.0, 10.0, 100.0, 100.0),
      f"quedó {suave[4]}")
check("no corre los bordes de las cajas estables",
      suave[0] == (10.0, 10.0, 100.0, 100.0) and suave[-1] == (10.0, 10.0, 100.0, 100.0))

# Un movimiento sostenido NO debe ser aplastado por el suavizado
rampa = [(float(i), float(i), float(i + 50), float(i + 50)) for i in range(20)]
suave = suavizar_cajas(rampa, ventana=5)
error = max(abs(suave[i][0] - rampa[i][0]) for i in range(2, 18))
check("conserva el movimiento sostenido (no aplasta la rampa)", error < 1e-9,
      f"desvío {error}")

check("lista vacía no rompe", suavizar_cajas([], 5) == [])
check("todo None devuelve todo None", suavizar_cajas([None] * 5, 5) == [None] * 5)
check("ignora los None al calcular la mediana",
      suavizar_cajas([(0.0, 0.0, 10.0, 10.0), None, (0.0, 0.0, 10.0, 10.0)], 3)[1]
      == (0.0, 0.0, 10.0, 10.0))

print("\n=== rellenar_faltantes ===")

c, n = rellenar_faltantes([None, (1.0, 1.0, 2.0, 2.0), None])
check("rellena con la caja válida más cercana",
      c[0] == (1.0, 1.0, 2.0, 2.0) and c[2] == (1.0, 1.0, 2.0, 2.0) and n == 2)
check("sin ninguna detección válida devuelve -1", rellenar_faltantes([None, None])[1] == -1)
check("sin huecos no rellena nada", rellenar_faltantes([(1.0, 1.0, 2.0, 2.0)])[1] == 0)

print("\n=== roi_desde_keypoints ===")

roi = roi_desde_keypoints([(100, 100), (200, 140)], margen=0.0)
check("sin aspecto declarado la ROI sigue siendo cuadrada",
      abs((roi[2] - roi[0]) - (roi[3] - roi[1])) < 1e-9)

# El camino de degradación por caja tiene que producir la MISMA forma que el camino alineado. Si
# saliera cuadrada y se reescalara al destino rectangular, esos frames quedarían estirados y el
# dataset mezclaría dos geometrías: una diferencia sistemática entre frames que después se puede
# aprender como atajo, porque a qué frame le toca cuál correlaciona con el ejemplar.
roi_asp = roi_desde_keypoints([(100, 100), (200, 140)], margen=0.0, aspecto=CAMPO.aspecto)
check("con aspecto declarado la ROI lo respeta",
      abs((roi_asp[3] - roi_asp[1]) / (roi_asp[2] - roi_asp[0]) - CAMPO.aspecto) < 1e-9,
      f"dio {(roi_asp[3] - roi_asp[1]) / (roi_asp[2] - roi_asp[0]):.4f}")
check("la ROI con aspecto sigue cubriendo todos los keypoints",
      roi_asp[0] <= 100 and roi_asp[1] <= 100 and roi_asp[2] >= 200 and roi_asp[3] >= 140)
check("la ROI con aspecto tiene la misma forma que el recorte de destino",
      abs((roi_asp[3] - roi_asp[1]) / (roi_asp[2] - roi_asp[0]) - ALTO / ANCHO) < 0.01)
check("cubre todos los keypoints",
      roi[0] <= 100 and roi[1] <= 100 and roi[2] >= 200 and roi[3] >= 140)
check("el margen agranda la caja",
      (roi_desde_keypoints([(100, 100), (200, 140)], margen=0.5)[2]
       - roi_desde_keypoints([(100, 100), (200, 140)], margen=0.5)[0]) > (roi[2] - roi[0]))

roi_c = roi_desde_keypoints([(10, 10), (50, 50)], margen=2.0, forma=(1920, 1080))
check("se recorta al borde de la imagen",
      roi_c[0] >= 0 and roi_c[1] >= 0 and roi_c[2] <= 1080 and roi_c[3] <= 1920)
check("ignora los keypoints None",
      roi_desde_keypoints([(100, 100), None, (200, 140)], margen=0.0) == roi)
check("sin keypoints devuelve None", roi_desde_keypoints([None, None]) is None)

print("\n" + "=" * 52)
if fallos:
    print(f"FALLARON {len(fallos)}: {fallos}")
    sys.exit(1)
print("TODAS LAS PRUEBAS PASARON")
