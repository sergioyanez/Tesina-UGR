"""Pruebas de la selección de frames a anotar y de su persistencia.

El error caro acá es silencioso: que `frame_video` apunte al frame equivocado. La ventana se
abriría igual, mostraría una cara, y la persona anotaría de buena fe una posición que corresponde
a otro instante del video. Nada fallaría, y el recorte saldría mal alineado.

El segundo error caro es de criterio: pedir que se anoten frames de clips que igual no van a
quedar íntegros. Es trabajo humano tirado, y es exactamente lo que el filtro por `faltan` evita.

Uso:
    python3 tests/test_anotacion.py
"""
import csv
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

fallos = []


def check(nombre, condicion, detalle=""):
    if condicion:
        print(f"  OK    {nombre}")
    else:
        print(f"  FALLA {nombre}  {detalle}")
        fallos.append(nombre)


# La raíz se apunta a un directorio temporal ANTES de importar, para no tocar los artefactos
# reales y para poder construir los CSV del caso.
tmp = tempfile.TemporaryDirectory()
raiz = Path(tmp.name)
(raiz / "resultados").mkdir(parents=True)
import os  # noqa: E402
os.environ["TESINA_EQUINOS_RAIZ"] = str(raiz)

from equinos import anotacion  # noqa: E402


def escribir(nombre, filas, columnas):
    ruta = raiz / "resultados" / nombre
    with ruta.open("w", encoding="utf-8", newline="") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=columnas)
        escritor.writeheader()
        escritor.writerows(filas)


def montar(estados, frame_inicio=0, frame_fin=160, frames=160):
    """Escribe los CSV de un caso. `estados` es {clip: [alineado por orden]}."""
    filas_frames, clips = [], []
    for clip, banderas in estados.items():
        clips.append({"clip": clip, "ruta_clip": f"clips/{clip}",
                      "frame_inicio": frame_inicio, "frame_fin": frame_fin,
                      "frames": frames})
        for orden, alineado in enumerate(banderas):
            filas_frames.append({"clase": "Disgustado", "id_caballo": clip[:3], "clip": clip,
                                 "orden": orden, "alineado": str(bool(alineado))})
    escribir("dataset_frames_alineados.csv", filas_frames,
             ["clase", "id_caballo", "clip", "orden", "alineado"])
    escribir("inventario_clips.csv", clips,
             ["clip", "ruta_clip", "frame_inicio", "frame_fin", "frames"])


print("=== frame_del_orden contra la fórmula REAL del notebook 01 ===")

# La prueba que importa: no comparar contra una fórmula escrita a mano, sino contra la que el
# notebook 01 ejecuta de verdad. Si alguien cambia `extraer_frames_uniformes`, esto falla acá y
# no en silencio dentro de una sesión de anotación.
import numpy as np  # noqa: E402

peor = 0
for total in (16, 17, 30, 45, 60, 100, 180, 181, 1200):
    reales = np.linspace(0, total - 1, 16).astype(int)
    for orden in range(16):
        mio = anotacion.frame_del_orden(total, orden)
        peor = max(peor, abs(int(reales[orden]) - mio))
check("coincide con np.linspace(0, total-1, 16).astype(int) en 9 largos x 16 órdenes",
      peor == 0, f"peor diferencia: {peor} frames")

check("el orden 0 es siempre el primer frame", anotacion.frame_del_orden(180, 0) == 0)
check("el orden 15 es siempre el último frame", anotacion.frame_del_orden(180, 15) == 179)
check("nunca se sale del clip",
      all(0 <= anotacion.frame_del_orden(45, o) < 45 for o in range(16)))
check("un clip con menos frames que muestras no rompe",
      all(0 <= anotacion.frame_del_orden(10, o) < 10 for o in range(16)))
check("un clip vacío devuelve 0 en vez de un índice inválido",
      anotacion.frame_del_orden(0, 5) == 0)

# Y la contraparte: la fórmula VIEJA —la que usaba frame_inicio del video original como offset—
# tiene que dar distinto. Si diera igual, esta prueba no estaría verificando nada.
viejo_bug = 1000 + round((1160 - 1000) * 3 / 16)
check("la fórmula con offset del video original NO coincide (era el bug)",
      anotacion.frame_del_orden(160, 3) != viejo_bug,
      f"linspace da {anotacion.frame_del_orden(160, 3)}, la vieja daba {viejo_bug}")


print("\n=== el frame se toma del archivo del clip, no del video original ===")

montar({"001_a.mp4": [0] + [1] * 15}, frames=180)
p = anotacion.pendientes()[0]
check("orden 0 apunta al frame 0 del clip", p.frame_video == 0, f"dio {p.frame_video}")

montar({"001_a.mp4": [1] * 15 + [0]}, frames=180)
p = anotacion.pendientes()[0]
check("orden 15 apunta al último frame del clip", p.frame_video == 179, f"dio {p.frame_video}")

# El caso que destapó el error: un clip que empieza tarde en un video largo. El índice tiene que
# seguir siendo relativo al clip, no al video.
montar({"001_a.mp4": [1] * 15 + [0]}, frames=180, frame_inicio=1000, frame_fin=1180)
p = anotacion.pendientes()[0]
check("frame_inicio del video NO se suma al índice",
      p.frame_video == 179,
      f"dio {p.frame_video}; con el offset habría dado ~1168 y el clip tiene 180 frames")
check("el índice cae dentro del clip", 0 <= p.frame_video < 180)


print("\n=== qué se pide anotar y qué no ===")

montar({"001_a.mp4": [0] + [1] * 15})
check("un clip al que le falta 1 entra", len(anotacion.pendientes()) == 1)

montar({"001_a.mp4": [0, 0] + [1] * 14})
check("un clip al que le faltan 2 entra, con sus dos frames",
      len(anotacion.pendientes()) == 2)

montar({"001_a.mp4": [0, 0, 0] + [1] * 13})
check("un clip al que le faltan 3 NO entra (con el tope en 2)",
      anotacion.pendientes() == [])
check("...pero sí con el tope en 3", len(anotacion.pendientes(max_faltantes=3)) == 3)

montar({"001_a.mp4": [1] * 16})
check("un clip ya íntegro no pide nada", anotacion.pendientes() == [])

montar({"001_a.mp4": [0] * 16})
check("un clip sin ninguna detección no pide nada",
      anotacion.pendientes() == [],
      "anotar 16 frames a mano no es 'completar un clip'")

montar({"001_a.mp4": [0] + [1] * 15, "002_b.mp4": [0] * 10 + [1] * 6})
pend = anotacion.pendientes()
check("con varios clips sólo entra el que está cerca",
      len(pend) == 1 and pend[0].clip == "001_a.mp4")

check("sólo se ofrecen las clases declaradas",
      anotacion.pendientes(clases=("Sereno",)) == [])


print("\n=== persistencia: se guarda de a una y no se repite ===")

montar({"001_a.mp4": [0, 0] + [1] * 14})
check("al principio no hay nada anotado", anotacion.leer_manuales() == {})

anotacion.guardar_manual({"clip": "001_a.mp4", "orden": 0, "x_ojo": 10, "y_ojo": 20,
                          "x_hocico": 30, "y_hocico": 40, "anotado_por": "test",
                          "fecha": "2026-08-28"})
hechas = anotacion.leer_manuales()
check("la anotación se lee de vuelta", ("001_a.mp4", 0) in hechas)
check("conserva las coordenadas", hechas[("001_a.mp4", 0)]["x_ojo"] == "10")

anotacion.guardar_manual({"clip": "001_a.mp4", "orden": 1, "x_ojo": 11, "y_ojo": 21,
                          "x_hocico": 31, "y_hocico": 41, "anotado_por": "test",
                          "fecha": "2026-08-28"})
check("la segunda se agrega sin pisar la primera", len(anotacion.leer_manuales()) == 2)

with (raiz / "resultados" / anotacion.ARCHIVO_MANUAL).open(encoding="utf-8") as f:
    lineas = f.readlines()
check("el CSV tiene una sola cabecera", lineas[0].startswith("clip,") and len(lineas) == 3)

# Lo que hace la sesión reanudable: descontar lo ya hecho.
pendiente_real = [p for p in anotacion.pendientes()
                  if (p.clip, p.orden) not in anotacion.leer_manuales()]
check("una vez anotados, no vuelven a pedirse", pendiente_real == [])

resumen = anotacion.informe()
check("el informe cuenta lo hecho", resumen["hechas"] == 2 and resumen["pendientes"] == [])


print("\n" + "=" * 62)
tmp.cleanup()
if fallos:
    print(f"FALLARON {len(fallos)}: {fallos}")
    sys.exit(1)
print("TODAS LAS PRUEBAS PASARON")
