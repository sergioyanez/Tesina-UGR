"""Prueba de humo de features_simba con una pose sintética. Uso: python experimentos/simba/test_features_simba.py"""
import numpy as np

import datos
import features_simba as F

P = np.full((16, 39, 3), np.nan); P[..., 2] = 0.9
def poner(n, x, y): P[:, datos.BP[n], 0] = x; P[:, datos.BP[n], 1] = y
poner("left_eye", 100, 100); poner("right_eye", 100, 100); poner("nose", 200, 100)       # escala = 100 px
poner("left_earbase", 100, 80); poner("left_earend", 100, 20)                              # oreja de 60 px, perpendicular al eje
poner("right_earbase", 100, 80); poner("right_earend", 100, 20)
poner("upper_jaw", 190, 110); poner("lower_jaw", 190, 130)
v, nombres, grupos, esc = F.vector_del_clip(P, np.arange(16) * 10.0, 30.0)
d = dict(zip(nombres, v))
assert abs(esc - 100) < 1e-9
assert abs(d["oreja_largo_media__media"] - 0.6) < 1e-9
assert abs(d["oreja_angulo_media__media"] - 90) < 1e-6
assert abs(d["boca_apertura_mandibulas__media"] - 0.2) < 1e-9
assert abs(d["cabeza_velocidad__media"]) < 1e-12 and d["disp_orejas_ambas"] == 1.0
P[:5, datos.BP["nose"], 2] = 0.1                       # baja confianza -> ausente, no coordenada inventada
v2, n2, *_ = F.vector_del_clip(P, np.arange(16) * 10.0, 30.0)
assert dict(zip(n2, v2))["disp_hocico"] == 11 / 16
print("ok")
