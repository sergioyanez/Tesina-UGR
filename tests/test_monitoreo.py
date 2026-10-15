"""Monitoreo continuo: ventanas, recuadro, video real de punta a punta y la app en dos columnas.

Necesita las variables del proyecto (ver `tests/test_entorno.py`):
    TESINA_EQUINOS_RAIZ=<datos_equinos> TESINA_EQUINOS_VIDEOS=~/videos_de_prueba
    python tests/test_monitoreo.py

La parte con video real corre pose + modelo (un par de minutos). Sin videos, se omite.
"""
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

fallos = []


def check(nombre, condicion, detalle=""):
    if condicion:
        print(f"  OK    {nombre}")
    else:
        print(f"  FALLA {nombre}  {detalle}")
        fallos.append(nombre)


from equinos import inferencia, monitoreo  # noqa: E402

print("\nVENTANAS")
v = monitoreo.generar_ventanas(300, 30.0)
check("ventanas de 3 s con paso de 1 s", len(v) == 8 and v[1][0] - v[0][0] == 30, len(v))
check("16 frames por ventana, dentro de la ventana",
      all(len(f) == 16 and ini <= f[0] and f[-1] == fin for ini, fin, f in v))
check("video corto (< 3 s): una sola ventana", len(monitoreo.generar_ventanas(60, 30.0)) == 1)
check("menos de 16 frames: ninguna", monitoreo.generar_ventanas(10, 30.0) == [])

print("\nRECUADRO")
filas = [{"bodypart": "nose", "x": 100, "y": 200, "confianza": 0.9},
         {"bodypart": "left_eye", "x": 200, "y": 100, "confianza": 0.9},
         {"bodypart": "left_earend", "x": 900, "y": 900, "confianza": 0.2},   # poca confianza
         {"bodypart": "withers", "x": 5, "y": 5, "confianza": 0.99}]         # no es de la cabeza
check("rodea sólo la cabeza confiable, con 10 % de margen",
      monitoreo.calcular_bounding_box(filas) == (90, 90, 210, 210), monitoreo.calcular_bounding_box(filas))
check("sin puntos de cabeza: None", monitoreo.calcular_bounding_box([filas[3]]) is None)

print("\nUN VIDEO REAL, DE PUNTA A PUNTA (pose una sola vez)")
videos = Path(os.environ.get("TESINA_EQUINOS_VIDEOS", "~/videos_de_prueba")).expanduser()
candidatos = sorted(videos.glob("*/*.mp4"))[:1]
if not candidatos:
    print("  (sin videos de prueba: se omite)")
for ruta in candidatos:
    llamadas = []
    original = inferencia._pose_completa
    inferencia._pose_completa = lambda *a, **k: (llamadas.append(1), original(*a, **k))[1]
    salida = Path(tempfile.mkdtemp(prefix="test_monitoreo_")) / "salida.mp4"
    info = {"conservar": True}
    try:
        t0 = time.time()
        r = inferencia.clasificar_video(ruta, info=info)
        t1 = time.time()
        if not r.hay_prediccion:
            print(f"  (negativa con {ruta.name}: {r.motivo}; se sigue con el video igual)")
        ventanas = monitoreo.procesar_video_continuo(str(ruta), str(salida), previo=info)
        t2 = time.time()
    finally:
        inferencia._pose_completa = original
    print(f"        clasificar {t1 - t0:.1f} s, video {t2 - t1:.1f} s, total {t2 - t0:.1f} s")
    check("DeepLabCut corrió una sola vez", len(llamadas) == 1, len(llamadas))
    check("hay al menos una ventana clasificada", len(ventanas) >= 1, len(ventanas))
    check("la etiqueta de cada ventana es una clase conocida",
          all(w["clase"] in ("Sereno", "Interesado", "Disgustado") for w in ventanas))
    check("se escribió el mp4", salida.exists() and salida.stat().st_size > 1000)
    check("no queda el intermedio mp4v", not list(salida.parent.glob("*_mp4v.mp4")))
    sonda = json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=codec_name,pix_fmt,nb_frames", "-of", "json", str(salida)],
        capture_output=True, text=True, check=True).stdout)["streams"][0]
    check("es H.264 yuv420p (reproducible en el navegador)",
          sonda["codec_name"] == "h264" and sonda["pix_fmt"] == "yuv420p", sonda)
    import cv2
    cap = cv2.VideoCapture(str(salida))
    ok, _ = cap.read()
    cap.release()
    check("OpenCV puede leer el primer fotograma", ok)
    import shutil
    shutil.rmtree(info["carpeta"], ignore_errors=True)
    shutil.rmtree(salida.parent, ignore_errors=True)

print("\nLA APP: DOS COLUMNAS, VIDEO A LA DERECHA (AppTest)")
from streamlit.testing.v1 import AppTest  # noqa: E402

fuente = (RAIZ / "app_inferencia.py").read_text(encoding="utf-8")
cambios = [('archivo = st.file_uploader("Video de la toma", type=FORMATOS)', "archivo = _Falso()"),
           ("st.video(archivo)", "pass"),
           ('st.button("Analizar", type="primary")', "True"),
           ("inferencia.clasificar_video(ruta, progreso=avisar, info=info)", "_clasif(info)"),
           ("from equinos.monitoreo import procesar_video_continuo",
            "procesar_video_continuo = _procesar")]
for viejo, nuevo in cambios:
    assert viejo in fuente, viejo
    fuente = fuente.replace(viejo, nuevo)
PRELUDIO = '''
import sys
sys.path.insert(0, %r)
from pathlib import Path
from equinos import inferencia as _inf
class _Falso:
    name = "x.mp4"
    def getbuffer(self):
        return b""
_d = {"Sereno": 0.7, "Interesado": 0.2, "Disgustado": 0.1}
PRED = _inf.Prediccion("Sereno", _d, 0.8, 8, 10, "t", 16, [], [], modelo="fusion",
                       detalle={"pose": _d, "visual": _d, "peso_pose": 0.75, "precision_reserva": 0.8})
NEG = _inf.Negativa("SIN_KEYPOINTS", "No se pudo.", 0)
RESULTADO = %s
CON_POSE = %s
FALLA = %s
def _clasif(info):
    if CON_POSE:
        info["filas_pose"] = [{"orden": 0}]
    return RESULTADO
def _procesar(ruta, ruta_out, progreso=None, previo=None):
    if FALLA:
        raise ValueError("sin caballo")
    Path(ruta_out).write_bytes(b"\\x00" * 2000)
    return [] if FALLA is None else [{"clase": "Sereno"}]
'''


def correr(resultado, con_pose, falla):
    app = AppTest.from_string(PRELUDIO % (str(RAIZ), resultado, con_pose, falla) + fuente,
                              default_timeout=60)
    app.run()
    return app


app = correr("PRED", True, False)
check("sin excepciones", not app.exception, [e.value for e in app.exception])
check("no hay selector de modo", not app.radio)
check("hay dos columnas al nivel de la página", len(app.columns) >= 2)
# AppTest lista las columnas en el orden del árbol: primero la de la izquierda (con sus anidadas),
# después la de la derecha y al final las de «Cómo llegó a esta respuesta», que van debajo.
_con_video = [i for i, c in enumerate(app.columns)
              if any("Video con la expresión" in str(m.value) for m in c.markdown)]
check("el video marcado va en la columna de la derecha, no en la izquierda",
      len(_con_video) == 1 and _con_video[0] > 0, _con_video)
check("«Cómo llegó a esta respuesta» va debajo, fuera de las columnas",
      any("Cómo llegó a esta respuesta" in str(m.value) for m in app.markdown)
      and not any("Cómo llegó a esta respuesta" in str(m.value)
                  for c in app.columns for m in c.markdown))
check("la respuesta va en la izquierda",
      any("Sereno" in str(m.value) for m in app.columns[0].markdown))
check("el pie del video explica las ventanas de 3 s",
      any("3 segundos" in c.value for c in app.caption))

app = correr("PRED", True, True)
check("si el video falla: sin excepciones y st.info a la derecha",
      not app.exception and any("No se pudo generar" in i.value for i in app.info),
      [e.value for e in app.exception])
check("si el video falla, la respuesta sigue a la izquierda",
      any("Sereno" in str(m.value) for m in app.columns[0].markdown))

app = correr("NEG", False, False)
check("negativa sin pose: sin excepciones, aviso a la derecha y la negativa visible",
      not app.exception and any("No se pudo generar" in i.value for i in app.info)
      and any("No se puede clasificar" in w.value for w in app.warning),
      [e.value for e in app.exception])

print()
if fallos:
    print(f"{len(fallos)} FALLAS: {fallos}")
    sys.exit(1)
print("Todo bien.")
