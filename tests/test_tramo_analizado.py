"""La predicción informa el tramo analizado; la app ya no lo muestra.

Necesita las variables del proyecto (ver `tests/test_entorno.py`):
    TESINA_EQUINOS_RAIZ=<repo>/datos_equinos TESINA_EQUINOS_VIDEOS=~/videos_de_prueba
    python tests/test_tramo_analizado.py

La parte con video real usa un video de `TESINA_EQUINOS_VIDEOS` (pose + modelo: tarda un minuto).
Si no hay videos, se omite y se avisa.
"""
import os
import re
import sys
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


from equinos import inferencia, secuencia  # noqa: E402

TOL = 0.1

print("\nEL CÁLCULO DEL TRAMO")
t = inferencia.tramo_analizado([60 + 5 * i for i in range(16)], total=360, fps=30.0)
check("inicio y fin salen de los índices y el fps",
      abs(t["inicio_seg"] - 2.0) < 1e-9 and abs(t["fin_seg"] - 136 / 30) < 1e-9, t)
check("duración total = fotogramas / fps", abs(t["duracion_seg"] - 12.0) < 1e-9, t)
t = inferencia.tramo_analizado(list(range(300, 316)), total=316, fps=30.0)
check("el fin no pasa de la duración", t["fin_seg"] <= t["duracion_seg"] + 1e-9, t)

print("\nUN VIDEO REAL, DE PUNTA A PUNTA")
videos = Path(os.environ.get("TESINA_EQUINOS_VIDEOS", "~/videos_de_prueba")).expanduser()
candidatos = sorted(videos.glob("*/*.mp4"))[:1]
if not candidatos:
    print("  (sin videos de prueba: se omite)")
for ruta in candidatos:
    for modelo in ("fusion", "mil"):
        r = inferencia.clasificar_video(ruta, modelo=modelo)
        if not r.hay_prediccion:
            print(f"  (negativa con {ruta.name}: {r.motivo}; se omite {modelo})")
            continue
        tr = r.tramo
        check(f"[{modelo}] trae el tramo", tr is not None)
        check(f"[{modelo}] 0 <= inicio < fin <= duración",
              0 <= tr["inicio_seg"] < tr["fin_seg"] <= tr["duracion_seg"] + 1e-9, tr)
        check(f"[{modelo}] fin - inicio <= ventana máxima + tolerancia",
              tr["fin_seg"] - tr["inicio_seg"] <= secuencia.VENTANA_MAXIMA_SEG + TOL, tr)
        print(f"        {ruta.name}: {tr}")

print("\nLA APP DIBUJA LA SECCIÓN (AppTest, sin subir archivo)")
from streamlit.testing.v1 import AppTest  # noqa: E402

fuente = (RAIZ / "app_inferencia.py").read_text(encoding="utf-8")
for viejo in ('archivo = st.file_uploader("Video de la toma", type=FORMATOS)',
              "st.video(archivo)", 'st.button("Analizar", type="primary")',
              "inferencia.clasificar_video(ruta, progreso=avisar, info=info)"):
    assert viejo in fuente, viejo
fuente = (fuente
          .replace('archivo = st.file_uploader("Video de la toma", type=FORMATOS)',
                   "archivo = _Falso()")
          .replace("st.video(archivo)", "pass")
          .replace('st.button("Analizar", type="primary")', "True")
          .replace("inferencia.clasificar_video(ruta, progreso=avisar, info=info)", "PRED"))
PRELUDIO = '''
import sys
sys.path.insert(0, %r)
from equinos import inferencia as _inf
class _Falso:
    name = "x.mp4"
    def getbuffer(self):
        return b""
_pr = {c: {"precision": 0.8, "acertadas": 8, "veces_que_lo_dijo": 10} for c in ("Sereno", "Interesado", "Disgustado")}
PRED = _inf.Prediccion("Sereno", {"Sereno": 0.7, "Interesado": 0.2, "Disgustado": 0.1}, 0.8, 8, 10, "t", 16, [], [],
                       modelo="fusion",
                       detalle={"pose": {"Sereno": 0.7, "Interesado": 0.2, "Disgustado": 0.1},
                                "visual": {"Sereno": 0.6, "Interesado": 0.3, "Disgustado": 0.1},
                                "peso_pose": 0.75, "precision_reserva": _pr["Sereno"]},
                       tramo=%r)
'''


def correr(tramo):
    app = AppTest.from_string(PRELUDIO % (str(RAIZ), tramo) + fuente, default_timeout=60)
    app.run()
    textos = [m.value for m in app.markdown]
    return app, textos


# La app ya no muestra el tramo, la confiabilidad en palabras ni el recuadro de cómo se probó:
# se quitaron a pedido (15/10). Acá se comprueba que no vuelvan y que la pantalla no falle.
for tramo in ({"inicio_seg": 2.4, "fin_seg": 8.9, "duracion_seg": 12.0}, None):
    app, textos = correr(tramo)
    check(f"sin excepciones (tramo={tramo})", not app.exception, [e.value for e in app.exception])
    check("no muestra el tramo", not any("Se analizó el tramo" in x for x in textos))
    check("no muestra «¿Cuánto confiar?»", not any("Cuánto confiar" in x for x in textos))
    check("no muestra el recuadro de cómo se probó",
          not any("Cómo se probó" in e.label for e in app.expander))
    check("las barras sólo dicen clase y porcentaje",
          all("acierta" not in str(b.proto) for b in app.get("progress")))

print()
if fallos:
    print(f"{len(fallos)} FALLAS: {fallos}")
    sys.exit(1)
print("Todo bien.")
