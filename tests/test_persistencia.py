"""Pruebas del paquete `equinos`: rutas, checkpointing, caché y estado de las etapas.

Todo lo que se prueba acá corre sin GPU, sin internet y sin pesos preentrenados, que es
exactamente lo que se puede verificar fuera de Colab. Y es donde vive el fallo peligroso: un
checkpoint que se reanuda con la configuración equivocada no rompe nada, simplemente devuelve un
resultado que mezcla dos corridas y parece correcto.

Uso:
    python3 tests/test_persistencia.py
"""
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from equinos import cache, checkpoint, etapas, rutas  # noqa: E402

fallos = []


def check(descripcion, condicion, detalle=""):
    if condicion:
        print(f"  OK    {descripcion}")
    else:
        print(f"  FALLA {descripcion}" + (f" -> {detalle}" if detalle else ""))
        fallos.append(descripcion)


BANCO = Path(tempfile.mkdtemp(prefix="equinos_pruebas_"))


def raiz_limpia(nombre):
    """Raíz de proyecto nueva y vacía, para que cada bloque parta de cero."""
    destino = BANCO / nombre
    if destino.exists():
        shutil.rmtree(destino)
    return rutas.inicializar(raiz=destino, verbose=False)


# ============================================================
print("=== rutas ===")
# ============================================================

raiz = raiz_limpia("rutas")

check("la raíz se crea", raiz.exists())
check("crea todas las subcarpetas de artefactos",
      all((raiz / s).is_dir() for s in rutas.SUBCARPETAS),
      f"faltan {[s for s in rutas.SUBCARPETAS if not (raiz / s).is_dir()]}")
check("fuera de Colab no intenta montar Drive", not rutas.en_colab())
check("carpeta() devuelve la subcarpeta pedida", rutas.carpeta("clips") == raiz / "clips")

try:
    rutas.carpeta("inexistente")
    check("carpeta() rechaza nombres desconocidos", False, "no lanzó excepción")
except ValueError:
    check("carpeta() rechaza nombres desconocidos", True)

# relativizar y resolver son inversas: es la propiedad de la que depende que un CSV escrito en una
# sesión siga siendo legible en otra.
absoluta = raiz / "clips" / "Sereno" / "013_clip_01.mp4"
relativa = rutas.relativizar(absoluta)
check("relativizar quita la raíz", relativa == "clips/Sereno/013_clip_01.mp4", relativa)
check("resolver(relativizar(x)) == x", rutas.resolver(relativa) == absoluta)
check("relativizar usa separadores POSIX", "\\" not in relativa)

fuera = Path("/algun/lugar/afuera.mp4")
check("una ruta fuera de la raíz se conserva absoluta", rutas.relativizar(fuera) == fuera.as_posix())

# Compatibilidad con los CSV que escribían las versiones anteriores de los notebooks.
heredada = rutas.resolver("/content/clips_equinos/Sereno/013_clip_01.mp4")
check("resolver traduce las rutas heredadas de /content",
      heredada == raiz / "clips" / "Sereno" / "013_clip_01.mp4", str(heredada))

# Este es el escenario que rompía el pipeline: un CSV escrito en una sesión, leído en otra.
otra_raiz = raiz_limpia("rutas_otra_sesion")
check("el mismo CSV resuelve contra la raíz vigente",
      rutas.resolver(relativa) == otra_raiz / "clips" / "Sereno" / "013_clip_01.mp4")

# Raíz sin permiso de escritura: debe fallar, no buscarse otro lugar en silencio.
solo_lectura = BANCO / "solo_lectura"
solo_lectura.mkdir(exist_ok=True)
os.chmod(solo_lectura, 0o500)
try:
    rutas.inicializar(raiz=solo_lectura / "proyecto", verbose=False)
    # En exFAT y como root los permisos no se aplican; ahí la prueba no es concluyente.
    check("raíz sin escritura falla", not os.access(solo_lectura, os.W_OK),
          "el sistema de archivos no aplica permisos POSIX; prueba no concluyente")
except RuntimeError as error:
    check("raíz sin escritura falla con mensaje claro", "escritura" in str(error).lower()
          or "no se pudo crear" in str(error).lower(), str(error))
finally:
    os.chmod(solo_lectura, 0o700)


# ============================================================
print("\n=== checkpoint ===")
# ============================================================

raiz_limpia("checkpoint")

CONFIG = {"CONF_MINIMA_DETECCION": 0.5, "MARGEN_ROI": 0.45, "TAMANIO_FIGURA": 12}
RELEVANTES = ["CONF_MINIMA_DETECCION", "MARGEN_ROI"]
UNIVERSO = [f"clip_{i:02d}" for i in range(10)]

cp = checkpoint.abrir("prueba", CONFIG, RELEVANTES, verbose=False)
check("un checkpoint nuevo no tiene unidades hechas", cp.unidades_completas == [])
check("todas las unidades están pendientes", cp.pendientes(UNIVERSO) == UNIVERSO)
check("no está completo", not cp.esta_completo(UNIVERSO))

for unidad in UNIVERSO[:6]:
    cp.marcar(unidad, [{"clip": unidad, "valor": 1}])

check("el estado se persiste en disco", cp.ruta_estado.exists())
check("el CSV parcial se persiste", cp.ruta_parcial.exists())

# Simula la sesión nueva tras el corte: mismo código, proceso distinto.
cp2 = checkpoint.abrir("prueba", CONFIG, RELEVANTES, verbose=False)
check("reanuda con las unidades ya hechas", len(cp2.unidades_completas) == 6)
check("sólo quedan pendientes las que faltan", cp2.pendientes(UNIVERSO) == UNIVERSO[6:],
      str(cp2.pendientes(UNIVERSO)))
check("las filas acumuladas se recuperan", len(cp2.filas()) == 6, str(len(cp2.filas())))

for unidad in cp2.pendientes(UNIVERSO):
    cp2.marcar(unidad, [{"clip": unidad, "valor": 1}])
check("al terminar queda completo", cp2.esta_completo(UNIVERSO))
check("no duplica filas al reanudar", len(cp2.filas()) == 10, str(len(cp2.filas())))
check("no duplica unidades al remarcar",
      (cp2.marcar("clip_00"), len(cp2.unidades_completas))[1] == 10)

# Invalidación: el punto entero del módulo.
config_cambiada = dict(CONFIG, MARGEN_ROI=0.60)
cp3 = checkpoint.abrir("prueba", config_cambiada, RELEVANTES, verbose=False)
check("cambiar un parámetro relevante descarta el checkpoint", cp3.unidades_completas == [])
check("cambiar un parámetro relevante borra el CSV parcial", not cp3.ruta_parcial.exists())

# Y lo simétrico: no invalidar por algo que no afecta el resultado.
raiz_limpia("checkpoint_irrelevante")
cp4 = checkpoint.abrir("prueba", CONFIG, RELEVANTES, verbose=False)
for unidad in UNIVERSO[:4]:
    cp4.marcar(unidad, [{"clip": unidad}])
cp5 = checkpoint.abrir("prueba", dict(CONFIG, TAMANIO_FIGURA=99), RELEVANTES, verbose=False)
check("cambiar un parámetro irrelevante conserva el checkpoint",
      len(cp5.unidades_completas) == 4, str(len(cp5.unidades_completas)))

# Declarar una clave que no existe es un error del programador, y debe doler ahora y no en Colab
# a los 40 minutos de cómputo.
try:
    checkpoint.abrir("prueba", CONFIG, ["PARAMETRO_QUE_NO_EXISTE"], verbose=False)
    check("declarar una clave inexistente falla", False, "no lanzó excepción")
except KeyError:
    check("declarar una clave inexistente falla", True)

# Un checkpoint corrupto no debe arrastrar datos basura al resultado.
cp5.ruta_estado.write_text("{esto no es json", encoding="utf-8")
cp6 = checkpoint.abrir("prueba", CONFIG, RELEVANTES, verbose=False)
check("un checkpoint ilegible se trata como inexistente", cp6.unidades_completas == [])

# Escritura atómica: el temporal no debe quedar tirado ni pisar al definitivo.
raiz_limpia("checkpoint_atomico")
cp7 = checkpoint.abrir("prueba", CONFIG, RELEVANTES, verbose=False)
cp7.marcar("clip_00")
check("la escritura no deja archivos temporales",
      not list(cp7.ruta_estado.parent.glob("*.tmp")))
check("el JSON persistido es legible y coherente",
      json.loads(cp7.ruta_estado.read_text())["unidades_completas"] == ["clip_00"])


# ============================================================
print("\n=== cache ===")
# ============================================================

raiz = raiz_limpia("cache")

origen = BANCO / "frames_origen"
if origen.exists():
    shutil.rmtree(origen)
(origen / "Sereno" / "clip_01").mkdir(parents=True)
(origen / "Interesado" / "clip_02").mkdir(parents=True)
for i in range(5):
    (origen / "Sereno" / "clip_01" / f"frame_{i:03d}.jpg").write_bytes(b"jpeg-falso" + bytes([i]))
for i in range(3):
    (origen / "Interesado" / "clip_02" / f"frame_{i:03d}.jpg").write_bytes(b"otro" + bytes([i]))

check("sin caché no está disponible", not cache.esta_disponible("frames_prueba"))

cache.empaquetar(origen, "frames_prueba", hash_config="abc123")
check("la caché queda disponible", cache.esta_disponible("frames_prueba"))
check("el manifiesto cuenta los archivos", cache.manifiesto("frames_prueba")["n_archivos"] == 8,
      str(cache.manifiesto("frames_prueba")))
check("el manifiesto guarda el hash de configuración",
      cache.manifiesto("frames_prueba")["hash_config"] == "abc123")
check("no queda el zip temporal", not list((raiz / "cache").glob("*.tmp")))

destino = BANCO / "frames_destino"
if destino.exists():
    shutil.rmtree(destino)
cache.expandir("frames_prueba", destino)
check("expandir restituye la cantidad de archivos",
      sum(1 for p in destino.rglob("*") if p.is_file()) == 8)
check("expandir preserva la estructura de carpetas",
      (destino / "Sereno" / "clip_01" / "frame_000.jpg").exists())
check("expandir preserva el contenido",
      (destino / "Sereno" / "clip_01" / "frame_004.jpg").read_bytes() == b"jpeg-falso" + bytes([4]))

check("una caché de otra configuración no se considera disponible",
      not cache.esta_disponible("frames_prueba", hash_config="otro_hash"))
check("la misma configuración sí", cache.esta_disponible("frames_prueba", hash_config="abc123"))

# Zip truncado: el modo de corrupción real cuando se corta la sesión durante la subida.
manifiesto = raiz / "cache" / "frames_prueba.json"
datos = json.loads(manifiesto.read_text())
datos["n_archivos"] = 99
manifiesto.write_text(json.dumps(datos))
destino2 = BANCO / "frames_destino2"
try:
    cache.expandir("frames_prueba", destino2)
    check("un conteo que no coincide se detecta", False, "expandió sin protestar")
except RuntimeError as error:
    check("un conteo que no coincide se detecta", "inconsistente" in str(error).lower())

# Empaquetar un directorio vacío haría pasar por válida una caché sin datos.
vacio = BANCO / "vacio"
vacio.mkdir(exist_ok=True)
try:
    cache.empaquetar(vacio, "vacia")
    check("empaquetar un directorio vacío falla", False, "no lanzó excepción")
except ValueError:
    check("empaquetar un directorio vacío falla", True)

cache.limpiar("frames_prueba", verbose=False)
check("limpiar borra zip y manifiesto", not cache.esta_disponible("frames_prueba"))


# ============================================================
print("\n=== etapas ===")
# ============================================================

raiz = raiz_limpia("etapas")

orden = etapas.orden_de_ejecucion()
check("el orden respeta las dependencias",
      orden.index("preparar_dataset") < orden.index("roi_normalizacion"), str(orden))
check("todas las etapas aparecen una sola vez", len(orden) == len(set(orden)) == len(etapas.PIPELINE))
check("toda dependencia declarada está registrada",
      all(d in etapas.PIPELINE for e in etapas.PIPELINE.values() for d in e.depende_de))

estado = etapas.estado("preparar_dataset")
check("sin artefactos la etapa está pendiente", estado["estado"] == etapas.PENDIENTE)
check("informa qué artefactos faltan", len(estado["artefactos_faltantes"]) == 3,
      str(estado["artefactos_faltantes"]))
check("informa qué cachés faltan", estado["caches_faltantes"] == ["frames_equinos"])

# Con checkpoint a medias pero sin salidas: parcial.
cp = checkpoint.abrir("preparar_dataset", {"a": 1}, ["a"], verbose=False)
cp.marcar("clip_00")
check("con progreso y sin salidas la etapa está parcial",
      etapas.estado("preparar_dataset")["estado"] == etapas.PARCIAL)
check("informa cuántas unidades lleva",
      etapas.estado("preparar_dataset")["unidades_procesadas"] == 1)

# Un CSV con sólo el encabezado NO alcanza: existe pero no contiene nada.
for artefacto in etapas.PIPELINE["preparar_dataset"].artefactos:
    (raiz / artefacto).parent.mkdir(parents=True, exist_ok=True)
    (raiz / artefacto).write_text("clase,clip\n")
cache.empaquetar(origen, "frames_equinos")
check("un CSV con sólo encabezado no cuenta como artefacto producido",
      etapas.estado("preparar_dataset")["estado"] != etapas.COMPLETA,
      str(etapas.estado("preparar_dataset")))
check("un artefacto vacío se reporta como faltante",
      len(etapas.estado("preparar_dataset")["artefactos_faltantes"]) == 3)

# Con contenido real: completa.
for artefacto in etapas.PIPELINE["preparar_dataset"].artefactos:
    (raiz / artefacto).write_text("clase,clip\nSereno,013_clip_01\n")
check("con todas las salidas la etapa está completa",
      etapas.estado("preparar_dataset")["estado"] == etapas.COMPLETA,
      str(etapas.estado("preparar_dataset")))

check("el diagnóstico cubre todas las etapas", len(etapas.diagnostico(imprimir=False)) == len(etapas.PIPELINE))
check("el diagnóstico no modifica artefactos",
      (etapas.diagnostico(imprimir=False), etapas.estado("preparar_dataset")["estado"])[1]
      == etapas.COMPLETA)

etapas.limpiar("preparar_dataset", verbose=False)
check("limpiar deja la etapa pendiente",
      etapas.estado("preparar_dataset")["estado"] == etapas.PENDIENTE)
check("limpiar borra el checkpoint",
      not (raiz / "checkpoints" / "preparar_dataset.json").exists())
check("limpiar no toca las otras etapas",
      etapas.estado("roi_normalizacion")["estado"] == etapas.PENDIENTE)


# ============================================================
shutil.rmtree(BANCO, ignore_errors=True)

print("\n" + "=" * 52)
if fallos:
    print(f"FALLARON {len(fallos)}: {fallos}")
    sys.exit(1)
print("TODAS LAS PRUEBAS PASARON")
