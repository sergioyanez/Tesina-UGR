"""Contrato de la fusión en la inferencia, sin video, sin GPU y sin el modelo entrenado.

Se vigila:
  - que la fusión sea el modelo por omisión y el MIL siga siendo una opción explícita;
  - que se niegue cuando no hay 16 frames, igual que el MIL;
  - que la salida conserve la interfaz (clase, tres probabilidades que suman 1, confiabilidad);
  - que la confiabilidad salga de los resultados de la comparación y no de la del MIL;
  - que el paquete se guarde y se cargue con sus metadatos, y avise si cambia sklearn.

Uso: python tests/test_fusion_inferencia.py
"""
import inspect
import sys
import tempfile
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from equinos import fusion, inferencia as inf  # noqa: E402

fallos = []


def check(nombre, condicion, detalle=""):
    if condicion:
        print(f"  OK    {nombre}")
    else:
        print(f"  FALLA {nombre}  {detalle}")
        fallos.append(nombre)


CLASES = ["Disgustado", "Interesado", "Sereno"]


class Falso:
    """Un modelo sklearn de juguete: siempre reparte la probabilidad igual."""

    def __init__(self, favorita):
        self.favorita = CLASES.index(favorita)

    def predict_proba(self, X):
        p = np.full((len(X), 3), 0.1)
        p[:, self.favorita] = 0.8
        return p


def paquete_falso(pose, visual, peso=0.75):
    return {"modelo_pose": Falso(pose), "modelo_visual": Falso(visual), "peso": peso,
            "clases": CLASES, "metadatos": {"version_sklearn": __import__("sklearn").__version__}}


def pose_16():
    filas = []
    for orden in range(16):
        for bp, x, y in (("left_eye", 10.0 + orden, 10.0), ("right_eye", 20.0 + orden, 10.0),
                         ("nose", 15.0 + orden, 40.0), ("left_earbase", 5.0, 5.0),
                         ("left_earend", 4.0, -10.0)):
            filas.append({"clip": "x.mp4", "orden": orden, "bodypart": bp, "x": x, "y": y,
                          "confianza": 0.9})
    return filas


PRECISIONES = {c: {"precision": 0.8, "acertadas": 8, "veces_que_lo_dijo": 10} for c in CLASES}
RESERVA = {c: {"precision": 0.5, "acertadas": 5, "veces_que_lo_dijo": 10} for c in CLASES}
METRICAS = {"clips_desarrollo": 291, "f1_macro_reserva": 0.834, "clips_reserva": 68, "caballos_reserva": 20,
            "f1_macro_desarrollo": 0.868, "f1_limpio_reserva": 0.767}
VIS = np.ones((16, 512), dtype=np.float32)
FRAMES = np.arange(16) * 5 + 100

print("=== el modelo por omisión es la fusión ===")
firma = inspect.signature(inf.clasificar_video)
check("clasificar_video usa la fusión por omisión", firma.parameters["modelo"].default == "fusion")
check("el MIL sigue siendo una opción", set(inf.MODELOS) == {"fusion", "mil"})
try:
    inf.clasificar_video("no_importa.mp4", modelo="otro")
    check("un modelo desconocido levanta ValueError", False, "no levantó")
except ValueError:
    check("un modelo desconocido levanta ValueError", True)

print("\n=== negarse cuando no hay 16 frames ===")
for n in (0, 8, 15, 17):
    r = inf.predecir_con_fusion(np.ones((n, 512)), pose_16(), np.arange(n), 30.0,
                                PRECISIONES, paquete_falso("Sereno", "Sereno"), METRICAS, RESERVA)
    check(f"con {n} frames se niega", not r.hay_prediccion)

print("\n=== la salida conserva la interfaz ===")
r = inf.predecir_con_fusion(VIS, pose_16(), FRAMES, 30.0, PRECISIONES,
                            paquete_falso("Disgustado", "Sereno"), METRICAS, RESERVA)
check("predice", r.hay_prediccion)
check("la clase es la de la mezcla ponderada", r.clase == "Disgustado", r.clase)
check("trae las tres probabilidades", set(r.probabilidades) == set(CLASES))
check("suman 1", abs(sum(r.probabilidades.values()) - 1) < 1e-9)
esperada = 0.75 * 0.8 + 0.25 * 0.1
check("w pondera la pose", abs(r.probabilidades["Disgustado"] - esperada) < 1e-9,
      r.probabilidades)
check("trae la precisión de su clase", (r.precision, r.acertadas, r.veces_que_lo_dijo) == (0.8, 8, 10))
check("se identifica como fusión", r.modelo == "fusion")
check("sin atención (es del MIL)", r.atencion == [] and r.clases_por_frame == [])
check("trae lo que dijo cada rama", set(r.detalle) == {"pose", "visual", "peso_pose", "precision_reserva"})
check("el texto informa el F1 de la reserva y que no es diagnóstico",
      "0.834" in r.texto_confiabilidad and "NO es un diagnóstico veterinario" in r.texto_confiabilidad,
      r.texto_confiabilidad)
check("el texto declara la precisión y el tamaño de la reserva",
      "5 de 10" in r.texto_confiabilidad and "291 clips" in r.texto_confiabilidad)
check("la reserva real es la registrada (Sereno 30 de 31)",
      (fusion.precision_reserva()["Sereno"]["acertadas"],
       fusion.precision_reserva()["Sereno"]["veces_que_lo_dijo"]) == (30, 31))
check("frames alineados", r.frames_alineados == 16)
w0 = inf.predecir_con_fusion(VIS, pose_16(), FRAMES, 30.0, PRECISIONES,
                             paquete_falso("Disgustado", "Sereno", peso=0.0), METRICAS, RESERVA)
check("con w=0 manda la rama visual", w0.clase == "Sereno", w0.clase)

print("\n=== el MIL conserva su salida (campos nuevos con valores por omisión) ===")
from equinos.inferencia import Prediccion  # noqa: E402
mil = Prediccion("Sereno", {}, 0.7, 1, 2, "t", 16, [0.1] * 16, ["Sereno"] * 16)
check("Prediccion del MIL: modelo 'mil' por omisión", mil.modelo == "mil" and mil.detalle is None)

print("\n=== la confiabilidad sale de la comparación ===")
prec = fusion.precision_por_clase()
check("tres clases con precisión en [0, 1]",
      set(prec) == set(CLASES) and all(0 <= v["precision"] <= 1 for v in prec.values()), prec)
check("los conteos son consistentes",
      all(v["acertadas"] <= v["veces_que_lo_dijo"] for v in prec.values()))
met = fusion.metricas_globales()
check("el F1 de la reserva es el registrado (único uso)", abs(met["f1_macro_reserva"] - 0.8337) < 1e-3, met)
check("no coincide con el del MIL", abs(met["f1_macro_desarrollo"] - 0.740) > 0.05)

print("\n=== el paquete se guarda, se carga y avisa si cambia sklearn ===")
with tempfile.TemporaryDirectory() as d:
    destino = fusion.guardar_paquete(paquete_falso("Sereno", "Sereno"), Path(d) / "m" / "p.joblib")
    check("se guarda", destino.exists())
    with warnings.catch_warnings(record=True) as avisos:
        warnings.simplefilter("always")
        cargado = fusion.cargar_paquete(destino)
    check("se carga sin avisos si coincide la versión", not avisos, [str(a.message) for a in avisos])
    check("conserva el peso y las clases", cargado["peso"] == 0.75 and cargado["clases"] == CLASES)
    malo = paquete_falso("Sereno", "Sereno")
    malo["metadatos"]["version_sklearn"] = "0.0.1"
    fusion.guardar_paquete(malo, Path(d) / "malo.joblib")
    with warnings.catch_warnings(record=True) as avisos:
        warnings.simplefilter("always")
        fusion.cargar_paquete(Path(d) / "malo.joblib")
    check("avisa si la versión de sklearn difiere", any("0.0.1" in str(a.message) for a in avisos))
    try:
        fusion.cargar_paquete(Path(d) / "no_existe.joblib")
        check("un paquete ausente levanta FileNotFoundError", False)
    except FileNotFoundError as e:
        check("un paquete ausente levanta FileNotFoundError", "entrenar_modelo_final" in str(e))

print("\n=== features de pose ===")
v = fusion.features_de_pose(pose_16(), FRAMES, 30.0)
check("son 160", v.shape == (160,))
try:
    fusion.features_de_pose(pose_16(), FRAMES[:10], 30.0)
    check("rechaza menos de 16 fotogramas", False)
except ValueError:
    check("rechaza menos de 16 fotogramas", True)

print("\n" + "=" * 52)
if fallos:
    print(f"FALLARON {len(fallos)}: {fallos}")
    sys.exit(1)
print("TODAS LAS PRUEBAS PASARON")
