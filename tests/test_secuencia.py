"""Pruebas de la selección de frames que comparten la etapa 02 y la inferencia.

Lo que hay que poder detectar acá tampoco es una excepción. Una selección mal hecha **no falla**:
devuelve 16 índices y el modelo predice igual. Los fallos silenciosos que se prueban son:

  - que la secuencia salte entre momentos desconectados del video, que rompe el supuesto de la
    cabeza de atención sin que nada avise;
  - que se elija un frame sin medición, que entra al backbone como un recorte descentrado;
  - que la ventana abarque más tiempo del que abarcaba al entrenar, que cambia la distribución
    de entrada dejando intacta la confiabilidad que se informa;
  - que se devuelvan índices repetidos o desordenados, que degrada la secuencia a algo más corto
    sin que el conteo lo delate.

No requieren GPU, ni internet, ni los CSV de una corrida: los casos se construyen a mano.

Uso:
    python3 tests/test_secuencia.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from equinos import secuencia as sq  # noqa: E402

fallos = []


def check(nombre, condicion, detalle=""):
    if condicion:
        print(f"  OK    {nombre}")
    else:
        print(f"  FALLA {nombre}  {detalle}")
        fallos.append(nombre)


print("\nRACHAS")
check("frames contiguos son una sola racha", sq.rachas([3, 4, 5, 6], 1) == [(3, 6)])
check("un hueco de 1 no corta la racha cuando el tope es 1",
      sq.rachas([3, 4, 6, 7], 1) == [(3, 7)], sq.rachas([3, 4, 6, 7], 1))
check("un hueco de 2 sí la corta", sq.rachas([3, 4, 7, 8], 1) == [(3, 4), (7, 8)],
      sq.rachas([3, 4, 7, 8], 1))
check("con tope 0 cualquier hueco corta", sq.rachas([3, 4, 6], 0) == [(3, 4), (6, 6)])
check("sin válidos no hay rachas", sq.rachas([], 1) == [])
check("los duplicados no inventan tramos", sq.rachas([3, 3, 4], 1) == [(3, 4)])


print("\nELECCIÓN DENTRO DE LA VENTANA")
s = sq.elegir_frames(range(100), total=100, fps=30.0)
check("de una ventana larga saca exactamente 16", len(s.ordenes) == 16, s.ordenes)
check("y los devuelve en orden temporal estricto",
      all(a < b for a, b in zip(s.ordenes, s.ordenes[1:])), s.ordenes)
check("sin repetir ninguno", len(set(s.ordenes)) == len(s.ordenes))
check("todos los elegidos son frames medidos", set(s.ordenes) <= set(range(100)))

# Lo que motivó el cambio de criterio: un tramo corto y DENSO contra uno largo y más ralo.
# Maximizar válidos elegiría el denso —16 frames casi simultáneos—; hay que elegir el largo.
denso = list(range(0, 20))                          # 20 válidos en 0,67 s
ralo = list(range(100, 280, 3))                     # 60 válidos repartidos en 6 s
s = sq.elegir_frames(denso + ralo, total=300, fps=30.0)
check("prefiere la ventana larga y no la más densa", min(s.ordenes) >= 100, s.ordenes)
check("y cubre varios segundos, no una fracción", s.span_seg > 3.0, round(s.span_seg, 2))

# Un agujero mayor al tope parte la ventana: no se puede coser lo que está desconectado.
izq = list(range(0, 40))
der = list(range(100, 140))                         # 2 s de nada en el medio, a 30 fps
s = sq.elegir_frames(izq + der, total=200, fps=30.0, hueco_maximo_seg=0.5)
check("un agujero largo corta la ventana", s.ordenes and max(s.ordenes) - min(s.ordenes) < 60,
      (min(s.ordenes), max(s.ordenes)))
check("y no mezcla los dos lados del agujero",
      set(s.ordenes) <= set(izq) or set(s.ordenes) <= set(der), s.ordenes)

# Un agujero por debajo del tope NO corta: es la tolerancia que hace falta para no perder clips.
con_hueco = list(range(0, 30)) + list(range(40, 70))   # 0,33 s sin medición, a 30 fps
s = sq.elegir_frames(con_hueco, total=100, fps=30.0, hueco_maximo_seg=0.5)
check("un agujero corto no corta la ventana", s.span_seg > 1.5, round(s.span_seg, 2))
check("y ningún frame elegido cae en el agujero",
      not (set(s.ordenes) & set(range(30, 40))), sorted(set(s.ordenes) & set(range(30, 40))))

print("\nANCLAJE QUE PREFIERE EL FRAME MEJOR MEDIDO")
validos = list(range(0, 160))
# Confianza alta salvo en las posiciones múltiplo de 10, que son las que el reparto uniforme
# elegiría primero: sin preferencia caen todas las borrosas, con preferencia no cae ninguna.
conf = {f: (0.51 if f % 10 == 0 else 0.95) for f in validos}
sin_pref = sq.elegir_frames(validos, total=160, fps=30.0)
con_pref = sq.elegir_frames(validos, total=160, fps=30.0, confianzas=conf)
flojos_sin = sum(1 for f in sin_pref.ordenes if conf[f] < 0.6)
flojos_con = sum(1 for f in con_pref.ordenes if conf[f] < 0.6)
check("sin preferencia caen los frames pegados al umbral", flojos_sin > 0, flojos_sin)
check("con preferencia se evitan", flojos_con < flojos_sin, (flojos_sin, flojos_con))
check("y siguen siendo 16 en orden estricto",
      len(con_pref.ordenes) == 16 and all(a < b for a, b in zip(con_pref.ordenes, con_pref.ordenes[1:])),
      con_pref.ordenes)
check("la ventana no se mueve: cubre los mismos segundos",
      abs(con_pref.span_seg - sin_pref.span_seg) < 0.5,
      (round(sin_pref.span_seg, 2), round(con_pref.span_seg, 2)))

# Si TODO alrededor es malo, no se va lejos a buscar: la tolerancia es un techo duro.
conf_plano = {f: 0.51 for f in validos}
s_plano = sq.elegir_frames(validos, total=160, fps=30.0, confianzas=conf_plano)
check("sin nada mejor cerca, el reparto queda igual que sin preferencia",
      s_plano.ordenes == sin_pref.ordenes, (s_plano.ordenes, sin_pref.ordenes))


print("\nTECHO TEMPORAL DE LA VENTANA")
# 15 s a 30 fps con todo válido: la ventana tiene que recortarse a 7 s.
s = sq.elegir_frames(range(450), total=450, fps=30.0, ventana_maxima_seg=7.0)
check("una ventana de 15 s se recorta al techo de 7 s", s.span_seg <= 7.0 + 1e-9, s.span_seg)
check("y aun así devuelve los 16 frames", len(s.ordenes) == 16, len(s.ordenes))
# Un clip corto del piso de 1 s entra entero: el techo no puede estorbar ahí.
s = sq.elegir_frames(range(45), total=45, fps=30.0)
check("un clip de 1,5 s entra entero sin recorte", len(s.ordenes) == 16, s.ordenes)
# El techo va en segundos, así que a 60 fps tiene que recortar al doble de frames que a 30.
a30 = sq.elegir_frames(range(600), total=600, fps=30.0)
a60 = sq.elegir_frames(range(600), total=600, fps=60.0)
check("el techo es temporal y no en frames: a 60 fps abarca los mismos segundos",
      abs(a30.span_seg - a60.span_seg) < 0.1 and a60.ordenes[-1] > a30.ordenes[-1],
      (round(a30.span_seg, 2), round(a60.span_seg, 2), a30.ordenes[-1], a60.ordenes[-1]))


print("\nNEGATIVAS")
s = sq.elegir_frames([], total=100)
check("sin frames válidos devuelve SIN_KEYPOINTS", s.motivo == "SIN_KEYPOINTS", s.motivo)
check("y no devuelve órdenes", s.ordenes == [])
check("hay_seleccion es falso", not s.hay_seleccion)

s = sq.elegir_frames(list(range(8)), total=100)
check("una ventana de 8 no alcanza el piso de 12", s.motivo == "VENTANA_CORTA", s.motivo)

# 30 válidos separados por un segundo cada uno: no hay ventana, hay 30 momentos sueltos.
s = sq.elegir_frames(list(range(0, 900, 30)), total=900, fps=30.0)
check("frames separados por un segundo no forman una secuencia",
      s.motivo == "VENTANA_CORTA", (s.motivo, s.racha))
check("y el informe dice cuántos válidos había", s.validos == 30, s.validos)

# Justo en el piso: 12 contiguos tienen que entrar, con 12 frames y no 16.
s = sq.elegir_frames(list(range(12)), total=100)
check("una ventana de exactamente 12 entra", s.hay_seleccion, s.motivo)
check("y aporta 12 frames, no 16", len(s.ordenes) == 12, s.ordenes)


print("\nPREDICADO DE VALIDEZ")
filas = [
    {"orden": 0, "x_ojo": 1, "x_hocico": 2, "conf_ojo": 0.9, "conf_hocico": 0.8},
    {"orden": 1, "x_ojo": 1, "x_hocico": 2, "conf_ojo": 0.4, "conf_hocico": 0.9},
    {"orden": 2, "x_ojo": "", "x_hocico": 2, "conf_ojo": 0.9, "conf_hocico": 0.9},
    {"orden": 3, "x_ojo": 1, "x_hocico": 2, "conf_ojo": 0.5, "conf_hocico": 0.5},
]
check("exige los dos puntos por encima del umbral",
      sq.frames_validos(filas, 0.5) == [0, 3], sq.frames_validos(filas, 0.5))
check("una coordenada vacía invalida el frame aunque la confianza alcance",
      2 not in sq.frames_validos(filas, 0.5))


print("\n" + "=" * 52)
if fallos:
    print(f"FALLARON {len(fallos)}: {fallos}")
    sys.exit(1)
print("TODAS LAS PRUEBAS PASARON")
