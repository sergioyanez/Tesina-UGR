"""Protocolo de comparación: reserva disjunta, particiones sin cruces, validación interna por caballo.

Parte sintética (siempre) + parte sobre la reserva y las particiones versionadas (si hay datos).

Uso:
    python3 tests/test_comparacion.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from equinos import comparacion as cmp, particion  # noqa: E402

fallos = []


def check(nombre, condicion, detalle=""):
    if condicion:
        print(f"  OK    {nombre}")
    else:
        print(f"  FALLA {nombre}  {detalle}")
        fallos.append(nombre)


def sintetico(n_caballos=60):
    import random
    rng = random.Random(3)
    clips = []
    for h in range(n_caballos):
        dom = ("Disgustado", "Interesado", "Sereno")[h % 3]
        for j in range(rng.randint(1, 6)):
            clase = dom if rng.random() < 0.8 else rng.choice(cmp.CLASES)
            clips.append({"clip": f"{h:03d}_{j}.mp4", "clase": clase, "id_caballo": f"{h:03d}"})
    return clips


clips = sintetico()
print("\nVALIDACIÓN INTERNA ESTRATIFICADA POR CABALLO")
for fr in (0.25, 0.2):
    tr, va = particion.dividir_estratificada(clips, fr)
    check(f"sin caballos en ambos lados (fracción {fr})",
          not ({c['id_caballo'] for c in tr} & {c['id_caballo'] for c in va}))
    check("tr + va = todos, sin repetidos",
          sorted(c["clip"] for c in tr + va) == sorted(c["clip"] for c in clips))
    check("las tres clases están en validación", {c["clase"] for c in va} == set(cmp.CLASES))
tr1, va1 = particion.dividir_estratificada(clips)
tr2, va2 = particion.dividir_estratificada(list(reversed(clips)))
check("determinista y no depende del orden de entrada",
      [c["clip"] for c in va1] == [c["clip"] for c in va2])
_, vv = particion.dividir_validacion(clips, modo="mas_clips")
_, vh = particion.dividir_por_ejemplar(clips)
check("el modo histórico sigue disponible y reproduce dividir_por_ejemplar",
      [c["clip"] for c in vv] == [c["clip"] for c in vh])
try:
    particion.dividir_validacion(clips, modo="otro")
    check("modo desconocido levanta ValueError", False)
except ValueError:
    check("modo desconocido levanta ValueError", True)

print("\nRESERVA (sintética)")
res = cmp.construir_reserva(clips)
check("reserva y desarrollo son disjuntos", not set(res["reserva"]) & set(res["desarrollo"]))
check("reserva + desarrollo = todos los caballos",
      set(res["reserva"]) | set(res["desarrollo"]) == {c["id_caballo"] for c in clips})
check("15 % de los caballos", len(res["reserva"]) == round(0.15 * 60), len(res["reserva"]))
check("determinista", cmp.construir_reserva(clips)["reserva"] == res["reserva"])
dev = cmp.clips_de_desarrollo(clips, res)
check("ningún clip de la reserva está en desarrollo",
      not {c["id_caballo"] for c in dev} & set(res["reserva"]))

print("\nPARTICIONES (sintéticas)")
part = cmp.construir_particiones(clips, res)
info = cmp.verificar_particiones(clips, part, res)
check("5 repeticiones × 4 folds = 20 evaluaciones", info["evaluaciones"] == 20 and part["R"] == 5)
vistos = set()
for r, k, tr, te in cmp.folds_de(dev, part):
    cab = {c["clip"]: c["id_caballo"] for c in clips}
    assert not {cab[i] for i in tr} & {cab[i] for i in te}
    vistos.add((r, k))
check("sin cruces de caballo en ninguna de las 20", len(vistos) == 20)
check("la reserva no aparece en ningún fold",
      all(not {cab[i] for i in tr + te} & set(res["reserva"]) for _, _, tr, te in cmp.folds_de(dev, part)))
check("las repeticiones difieren entre sí",
      part["repeticiones"][0]["fold_de_caballo"] != part["repeticiones"][1]["fold_de_caballo"])

print("\nPUERTA DE LA RESERVA")
class Dummy:
    nombre = "dummy"
    def ajustar(self, ids, datos): self.c = datos.etiqueta(ids[0])
    def predecir(self, ids, datos): return [self.c] * len(ids)

class DatosSint(cmp.Datos):
    def __init__(self, c): super().__init__(clips=c)
    def es_limpio(self, i): return True

tmp = Path(tempfile.mkdtemp(prefix="test_cmp_"))
log, rj = tmp / "uso.log", tmp / "r.json"
cmp.guardar_reserva(res, rj)
orig = cmp.ARCHIVO_RESERVA
cmp.ARCHIVO_RESERVA = rj
d = DatosSint(clips)
for kw, exc in (({}, PermissionError), ({"uso_autorizado_por_sergio": True}, ValueError)):
    try:
        cmp.evaluar_en_reserva(Dummy, "dummy", datos=d, ruta_log=log, carpeta_resultados=tmp, **kw)
        check(f"sin flag/motivo se niega ({kw})", False)
    except exc:
        check(f"sin flag/motivo se niega ({kw})", True)
check("las negativas no dejan rastro en el log", cmp.usos_de_reserva(log) == [])
out = cmp.evaluar_en_reserva(Dummy, "dummy", uso_autorizado_por_sergio=True, motivo="prueba",
                             datos=d, ruta_log=log, carpeta_resultados=tmp)
check("con el flag mide sobre los caballos de la reserva",
      out["uso_reserva"] and out["caballos"] == len(res["reserva"]))
check("el uso queda en el log", len(cmp.usos_de_reserva(log)) == 1)
try:
    cmp.evaluar_en_reserva(Dummy, "dummy", uso_autorizado_por_sergio=True, motivo="otra",
                           datos=d, ruta_log=log, carpeta_resultados=tmp)
    check("un segundo uso se niega", False)
except PermissionError:
    check("un segundo uso se niega", True)
cmp.ARCHIVO_RESERVA = orig

print("\nARCHIVOS VERSIONADOS (comparacion/)")
try:
    datos = cmp.Datos()
except Exception as e:  # sin datos del proyecto
    print(f"  (sin datos reales: se omite — {type(e).__name__})")
else:
    R = cmp.cargar_reserva(clips=datos.clips)
    P = cmp.cargar_particiones()
    check("la huella de la reserva y del dataset coinciden", R["huella_dataset"] == datos.huella)
    check("reserva = 15 % de 133 caballos", abs(len(R["reserva"]) / 133 - 0.15) < 0.01, len(R["reserva"]))
    check("reserva y desarrollo disjuntos", not set(R["reserva"]) & set(R["desarrollo"]))
    check("particiones ligadas a esta reserva", P["huella_reserva"] == R["huella"])
    v = cmp.verificar_particiones(datos.clips, P, R)
    check("20 evaluaciones sin cruces ni reserva ni clases ausentes",
          v["evaluaciones"] == 20 and not v["folds_con_clase_ausente"], v)
    check("la reserva nunca se usó en el repositorio", cmp.usos_de_reserva() == [])

print("\n" + "=" * 52)
if fallos:
    print(f"FALLARON {len(fallos)}: {fallos}")
    sys.exit(1)
print("TODAS LAS PRUEBAS PASARON")
