"""Pruebas del contrato de la inferencia: qué devuelve, y cuándo se niega.

Lo que se vigila acá no son excepciones. Son dos cosas que salen mal en silencio:

  - **que el sistema prediga cuando no debería.** El modelo consume clips íntegros; sobre un clip
    a medio alinear devolvería una etiqueta cuya confiabilidad nadie midió;
  - **que la inferencia use umbrales distintos de los del preprocesamiento evaluado.** Si divergen,
    las entradas son de otra distribución y la confiabilidad informada deja de aplicar sin que
    nada falle.

No requieren GPU, ni video, ni el modelo entrenado: el modelo se inyecta.

Uso:
    python3 tests/test_inferencia.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from equinos import inferencia as inf  # noqa: E402

fallos = []


def check(nombre, condicion, detalle=""):
    if condicion:
        print(f"  OK    {nombre}")
    else:
        print(f"  FALLA {nombre}  {detalle}")
        fallos.append(nombre)


CLASES = ["Disgustado", "Interesado", "Sereno"]
PRECISIONES = {
    "Disgustado": {"precision": 7 / 21, "acertadas": 7, "veces_que_lo_dijo": 21},
    "Interesado": {"precision": 49 / 62, "acertadas": 49, "veces_que_lo_dijo": 62},
    "Sereno": {"precision": 44 / 58, "acertadas": 44, "veces_que_lo_dijo": 58},
}


def modelo_falso(clase_ganadora, margen=4.0):
    """Un MIL de juguete que siempre gana con `clase_ganadora`.

    Cumple el contrato que `predecir_desde_caracteristicas` usa —`forward(..., return_attn)` y
    `logits_por_instancia`— sin depender de pesos entrenados ni de GPU. Lo que se prueba acá es
    el contrato de la inferencia, no lo que el modelo aprendió.
    """
    import torch
    import torch.nn as nn

    class MILDeJuguete(nn.Module):
        def __init__(self):
            super().__init__()
            # Un parámetro aunque no se use: `predecir_desde_caracteristicas` averigua el
            # dispositivo con `next(modelo.parameters())`, y un módulo sin parámetros lo rompe.
            self.marca = nn.Parameter(torch.zeros(1))

        def forward(self, v_vis, v_geom, attn_mask, return_attn=False):
            lotes, largo = v_vis.shape[0], v_vis.shape[1]
            logits = torch.zeros(lotes, len(CLASES))
            logits[:, CLASES.index(clase_ganadora)] = margen
            if return_attn:
                return logits, torch.full((lotes, largo), 1.0 / largo)
            return logits

        def logits_por_instancia(self, v_vis, v_geom):
            lotes, largo = v_vis.shape[0], v_vis.shape[1]
            salida = torch.zeros(lotes, largo, len(CLASES))
            salida[:, :, CLASES.index(clase_ganadora)] = margen
            return salida

    return MILDeJuguete().eval(), list(CLASES)


def pose_falsa(cuantos):
    """Pose mínima con `orden` renumerado a la posición en la secuencia, como espera el modelo."""
    filas = []
    for orden in range(cuantos):
        for bodypart, x, y in (("left_eye", 10.0, 10.0), ("nose", 10.0, 30.0),
                               ("left_earbase", 5.0, 5.0)):
            filas.append({"clip": "prueba.mp4", "orden": orden, "bodypart": bodypart,
                          "x": x, "y": y, "confianza": 0.9})
    return filas


print("=== negarse cuando no hay 16 frames alineados ===")

# 15 y 17 también: el criterio es EXACTAMENTE 16, igual que en `particion.tabla_clips`.
for cuantos in (0, 1, 8, 12, 15, 17):
    r = inf.predecir_desde_caracteristicas(
        np.ones((cuantos, 4), dtype=np.float32), pose_falsa(cuantos),
        PRECISIONES, modelo_falso("Sereno"))
    check(f"con {cuantos} frames se niega", not r.hay_prediccion, f"devolvió {type(r).__name__}")
    if not r.hay_prediccion:
        check(f"  y el motivo es ALINEACION_INSUFICIENTE ({cuantos})",
              r.motivo == "ALINEACION_INSUFICIENTE", r.motivo)
        check(f"  y dice cuántos alineó ({cuantos})", str(cuantos) in r.detalle, r.detalle)

check("la negativa lleva la cuenta de frames alineados",
      inf.predecir_desde_caracteristicas(
          np.ones((9, 4), dtype=np.float32), pose_falsa(9),
          PRECISIONES, modelo_falso("Sereno")).frames_alineados == 9)


print("\n=== la predicción trae SIEMPRE la confiabilidad de su clase ===")

vectores = np.ones((16, 4), dtype=np.float32)
POSE = pose_falsa(16)
r = inf.predecir_desde_caracteristicas(vectores, POSE, PRECISIONES, modelo_falso("Disgustado"))
check("con 16 frames predice", r.hay_prediccion)
check("la clase es la que gana el voto", r.clase == "Disgustado", r.clase)
check("trae la precisión de esa clase", abs(r.precision - 7 / 21) < 1e-12)
check("trae el numerador y el denominador", (r.acertadas, r.veces_que_lo_dijo) == (7, 21))
check("trae las TRES probabilidades", set(r.probabilidades) == set(CLASES), r.probabilidades)
# 1e-6 y no 1e-9: el softmax del MIL corre en `float32` y la suma cierra en 0,99999993. La
# tolerancia de 1e-9 venía del clasificador lineal anterior, que trabajaba en `float64`.
check("las probabilidades suman 1", abs(sum(r.probabilidades.values()) - 1) < 1e-6,
      (r.probabilidades, sum(r.probabilidades.values())))
check("informa 16 frames alineados", r.frames_alineados == 16)

texto = r.texto_confiabilidad
check("el texto dice cuántas veces acertó", "7 de 21" in texto, texto)
check("el texto dice que no alcanzó el criterio", "NO alcanzó el criterio" in texto)
check("el texto dice que no es diagnóstico", "NO es un diagnóstico veterinario" in texto)

# La clase de menor precisión es la que más importa que traiga su número.
peor = inf.predecir_desde_caracteristicas(vectores, POSE, PRECISIONES,
                                         modelo_falso("Disgustado"))
check("la clase de PEOR precisión igual trae su confiabilidad", peor.precision < 0.4)

otra = inf.predecir_desde_caracteristicas(vectores, POSE, PRECISIONES,
                                         modelo_falso("Interesado"))
check("y cambia con la clase predicha", abs(otra.precision - 49 / 62) < 1e-12)


print("\n=== el contrato de las dos salidas ===")

check("Prediccion.hay_prediccion es True", r.hay_prediccion is True)
check("Negativa.hay_prediccion es False",
      inf.predecir_desde_caracteristicas(
          np.ones((3, 4), dtype=np.float32), pose_falsa(3),
          PRECISIONES, modelo_falso("Sereno")).hay_prediccion is False)
check("todos los motivos tienen texto", all(inf.MOTIVOS.values()))
check("los motivos cubren las causas distinguidas",
      set(inf.MOTIVOS) == {"SIN_VIDEO", "VIDEO_CORTO", "VIDEO_LARGO", "SIN_DETECCION",
                           "SIN_KEYPOINTS", "ALINEACION_INSUFICIENTE", "VENTANA_CORTA"},
      sorted(inf.MOTIVOS))
check("hay un motivo para el video demasiado largo, que es un caso propio",
      "VIDEO_LARGO" in inf.MOTIVOS)
check("el límite de duración es explícito y no está enterrado en el código",
      isinstance(inf.DURACION_MAXIMA_SEG, float) and inf.DURACION_MAXIMA_SEG > 7)


print("\n=== los umbrales salen de la etapa 02, no de acá ===")

p = inf.parametros_etapa02()
for clave in ("CONF_MINIMA_DETECCION", "CONF_MINIMA_KEYPOINT", "AIRE_SOBRE_OJO",
              "AIRE_BAJO_HOCICO", "SEMIANCHO_LATERAL", "ANCHO_RECORTE"):
    check(f"lee {clave} del notebook", clave in p, sorted(p))

campo, ancho = inf.campo_de_vision(p)
check("el campo de visión coincide con los parámetros declarados",
      (campo.aire_sobre_ojo, campo.aire_bajo_hocico, campo.semiancho)
      == (p["AIRE_SOBRE_OJO"], p["AIRE_BAJO_HOCICO"], p["SEMIANCHO_LATERAL"]))
check("el ancho del recorte también", ancho == p["ANCHO_RECORTE"])

geo = inf.geometria_etapa02()
check("extrae la geometría del notebook en vez de reimplementarla",
      all(f in geo for f in inf.FUNCIONES_GEOMETRIA), sorted(inf.FUNCIONES_GEOMETRIA))

# Si alguien mueve la celda `parameters`, esto tiene que gritar y no seguir con valores propios.
try:
    inf.parametros_etapa02(notebook="no_existe.ipynb")
    check("un notebook ausente levanta FileNotFoundError", False, "no levantó")
except FileNotFoundError:
    check("un notebook ausente levanta FileNotFoundError", True)


print("\n" + "=" * 52)
if fallos:
    print(f"FALLARON {len(fallos)}: {fallos}")
    sys.exit(1)
print("TODAS LAS PRUEBAS PASARON")
