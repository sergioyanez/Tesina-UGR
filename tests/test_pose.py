"""Pruebas de la selección de keypoints faciales sobre la tabla de pose completa.

Corren sin GPU, sin internet y sin pesos preentrenados, con una tabla de pose sintética. Es
exactamente lo que se puede verificar fuera de Colab, y alcanza: lo que se prueba acá es criterio
de selección y geometría, no inferencia.

El fallo peligroso que cubren estas pruebas es silencioso. Si la selección del hocico toma un
candidato equivocado o descarta un frame que debería alinear, nada se rompe: sale un dataset con
menos frames o con recortes de escala inconsistente, y el problema recién se manifiesta como
métricas raras en el objetivo 3, a semanas de distancia.

Uso:
    python3 tests/test_pose.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from equinos import pose  # noqa: E402

fallos = []


def check(descripcion, condicion, detalle=""):
    if condicion:
        print(f"  OK    {descripcion}")
    else:
        print(f"  FALLA {descripcion}" + (f" -> {detalle}" if detalle else ""))
        fallos.append(descripcion)


def fila(clip, orden, bodypart, x, y, conf):
    return {"clip": clip, "orden": orden, "bodypart": bodypart,
            "x": x, "y": y, "confianza": conf}


# ============================================================
print("\n1. Selección del ojo: el más confiable de cada frame")
# ============================================================
# En perfil sólo un ojo es visible, y cuál depende de hacia dónde mire el animal, así que la
# elección tiene que rehacerse frame por frame y no fijarse para todo el clip.

tabla = [
    # frame 0: gana el izquierdo
    fila("a.mp4", 0, "left_eye",  10, 10, 0.90),
    fila("a.mp4", 0, "right_eye", 99, 99, 0.10),
    fila("a.mp4", 0, "nose",      10, 50, 0.80),
    # frame 1: gana el derecho, sobre el MISMO clip
    fila("a.mp4", 1, "left_eye",  99, 99, 0.05),
    fila("a.mp4", 1, "right_eye", 20, 10, 0.85),
    fila("a.mp4", 1, "nose",      20, 50, 0.75),
]

pares = pose.seleccionar_par(tabla)
check("elige el ojo izquierdo cuando es el más confiable",
      pares[0]["bodypart_ojo"] == "left_eye", pares[0]["bodypart_ojo"])
check("cambia de ojo dentro del mismo clip",
      pares[1]["bodypart_ojo"] == "right_eye", pares[1]["bodypart_ojo"])
check("registra las coordenadas del ojo elegido",
      (pares[1]["x_ojo"], pares[1]["y_ojo"]) == (20.0, 10.0))


# ============================================================
print("\n2. Candidato ausente: se informa y se sigue")
# ============================================================
# El listado de candidatos es una conjetura sobre el esquema del modelo. Que falte uno es lo
# esperable, no un error.

disponibles = pose.bodyparts_disponibles(tabla)
presentes = pose.candidatos_presentes(disponibles)
check("detecta sólo los candidatos que el modelo expone", presentes == ["nose"], str(presentes))
check("no falla cuando faltan candidatos del listado", len(pares) == 2)

sin_hocico = [f for f in tabla if f["bodypart"] != "nose"]
try:
    pose.seleccionar_par(sin_hocico)
    check("falla explícitamente si NINGÚN candidato existe", False, "no levantó excepción")
except ValueError as e:
    check("falla explícitamente si NINGÚN candidato existe", "candidatos" in str(e).lower())

try:
    pose.seleccionar_par(tabla, bodypart_hocico="inventado")
    check("rechaza un bodypart que no está en el modelo", False, "no levantó excepción")
except ValueError:
    check("rechaza un bodypart que no está en el modelo", True)


# ============================================================
print("\n3. Modos de selección del hocico")
# ============================================================

tabla2 = [
    fila("b.mp4", 0, "left_eye",  0,  0, 0.90),
    fila("b.mp4", 0, "nose",      0, 40, 0.20),   # poco confiable
    fila("b.mp4", 0, "upper_jaw", 0, 30, 0.88),   # confiable
]

fijo_nose = pose.seleccionar_par(tabla2, bodypart_hocico="nose")[0]
fijo_jaw = pose.seleccionar_par(tabla2, bodypart_hocico="upper_jaw")[0]
mas_conf = pose.seleccionar_par(tabla2, bodypart_hocico=pose.MAS_CONFIABLE)[0]
legado = pose.seleccionar_par(tabla2)[0]

check("el modo fijo respeta el bodypart pedido aunque sea el peor",
      fijo_nose["bodypart_hocico"] == "nose" and fijo_nose["conf_hocico"] == 0.20)
check("el modo fijo puede pedir el otro candidato",
      fijo_jaw["bodypart_hocico"] == "upper_jaw")
check("el modo MAS_CONFIABLE toma el de mayor confianza",
      mas_conf["bodypart_hocico"] == "upper_jaw", mas_conf["bodypart_hocico"])
check("sin parámetro reproduce el comportamiento histórico (primer candidato del listado)",
      legado["bodypart_hocico"] == "nose", legado["bodypart_hocico"])


# ============================================================
print("\n4. Frames sin ningún candidato válido")
# ============================================================
# El modelo puede no ver la cara. Ese frame no se alinea, pero no puede tumbar la corrida ni
# colarse como alineable.

tabla3 = [
    fila("c.mp4", 0, "left_eye", 5, 5, 0.95),
    fila("c.mp4", 0, "nose",    "", "", ""),      # sin coordenadas
    fila("c.mp4", 1, "left_eye", 5, 5, 0.95),
    fila("c.mp4", 1, "nose",     5, 40, 0.95),
]

pares3 = pose.seleccionar_par(tabla3)
check("un frame sin coordenadas de hocico no se alinea",
      not pose._alineable(pares3[0], 0.50))
check("y el frame completo del mismo clip sí se alinea",
      pose._alineable(pares3[1], 0.50))
check("la confianza nunca sale negativa", all(p["conf_hocico"] >= 0.0 for p in pares3))


# ============================================================
print("\n5. Empate de confianzas: la elección es determinista")
# ============================================================
# Con dos candidatos empatados, cuál gane es arbitrario pero NO puede variar entre corridas: si
# variara, dos ejecuciones sobre los mismos datos darían recortes distintos.

tabla4 = [
    fila("d.mp4", 0, "left_eye",  0,  0, 0.90),
    fila("d.mp4", 0, "nose",      0, 40, 0.70),
    fila("d.mp4", 0, "upper_jaw", 0, 30, 0.70),
]
elecciones = {pose.seleccionar_par(tabla4, pose.MAS_CONFIABLE)[0]["bodypart_hocico"]
              for _ in range(5)}
check("el empate se resuelve siempre igual", len(elecciones) == 1, str(elecciones))


# ============================================================
print("\n6. Efecto geométrico de alternar candidatos")
# ============================================================
# Es el dato que decide si MAS_CONFIABLE es admisible: la escala del recorte queda determinada
# por la distancia ojo-hocico, así que alternar entre puntos anatómicos distintos la mueve.

geo = pose.variacion_distancia(tabla4, ["nose", "upper_jaw"])
check("mide la distancia a cada candidato",
      geo["distancia_mediana"]["nose"] == 40.0
      and geo["distancia_mediana"]["upper_jaw"] == 30.0,
      str(geo["distancia_mediana"]))
check("la dispersión relativa detecta candidatos anatómicamente distintos",
      abs(geo["dispersion_relativa_mediana"] - (40 - 30) / 40) < 1e-9,
      str(geo["dispersion_relativa_mediana"]))

# Candidatos que caen casi en el mismo punto: alternarlos sería inocuo.
tabla5 = [
    fila("e.mp4", 0, "left_eye",  0,  0, 0.90),
    fila("e.mp4", 0, "nose",      0, 40, 0.70),
    fila("e.mp4", 0, "upper_jaw", 0, 39, 0.70),
]
geo2 = pose.variacion_distancia(tabla5, ["nose", "upper_jaw"])
check("candidatos casi coincidentes dan dispersión chica",
      geo2["dispersion_relativa_mediana"] < 0.05,
      str(geo2["dispersion_relativa_mediana"]))


# ============================================================
print("\n7. Trazabilidad: el CSV de keypoints se reconstruye desde la pose")
# ============================================================
# Es el requisito que hace que persistir la pose sirva de algo: si la selección no fuera
# reproducible desde la tabla, el artefacto no probaría nada sobre la corrida que lo produjo.

esquema = {"clip", "orden", "x_ojo", "y_ojo", "conf_ojo",
           "x_hocico", "y_hocico", "conf_hocico", "bodypart_ojo", "bodypart_hocico"}
check("la salida trae el esquema de keypoints_faciales.csv",
      set(pares[0]) == esquema, str(set(pares[0]) ^ esquema))
check("hay una fila por frame, no por bodypart", len(pares) == 2)
check("dos derivaciones seguidas dan lo mismo",
      pose.seleccionar_par(tabla) == pose.seleccionar_par(tabla))
check("las filas salen ordenadas por clip y orden",
      [(p["clip"], p["orden"]) for p in pares] == sorted((p["clip"], p["orden"]) for p in pares))


# ============================================================
print("\n" + "=" * 52)
if fallos:
    print(f"FALLARON {len(fallos)}: {fallos}")
    sys.exit(1)
print("TODAS LAS PRUEBAS PASARON")
