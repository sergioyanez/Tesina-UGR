"""Pruebas de las métricas, las líneas de base y la confiabilidad que ve el usuario.

Lo que se vigila acá no son excepciones: son números que salen mal y parecen bien.

  - una métrica mal promediada da un resultado plausible y arruina toda comparación;
  - confundir precisión con recall cambia lo que la interfaz le dice a una persona: `Interesado`
    tiene recall 0,88 y precisión 0,79, y `Sereno` tiene recall 0,69 y precisión 0,76 —al revés—.
    En `Disgustado` coinciden por casualidad de esta matriz, así que ilustrar la distinción con
    esa clase no probaría nada;
  - un texto de confiabilidad escrito a mano diverge del modelo el día que se re-evalúa.

No requieren GPU, ni internet, ni los CSV de una corrida.

Uso:
    python3 tests/test_clasificacion.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from equinos import clasificacion as cl  # noqa: E402

fallos = []


def check(nombre, condicion, detalle=""):
    if condicion:
        print(f"  OK    {nombre}")
    else:
        print(f"  FALLA {nombre}  {detalle}")
        fallos.append(nombre)


print("=== métricas: casos donde el promedio mal hecho pasa desapercibido ===")

# Predicción perfecta.
reales = ["Disgustado", "Interesado", "Sereno"]
check("predicción perfecta da F1 macro 1", abs(cl.f1_macro(reales, reales) - 1.0) < 1e-12)
check("predicción perfecta da balanced accuracy 1",
      abs(cl.balanced_accuracy(reales, reales) - 1.0) < 1e-12)

# El caso que motiva no usar accuracy global: predecir siempre la mayoritaria.
reales = ["Sereno"] * 45 + ["Interesado"] * 40 + ["Disgustado"] * 15
siempre_sereno = ["Sereno"] * 100
f1 = cl.f1_macro(reales, siempre_sereno)
exactitud = sum(1 for r, p in zip(reales, siempre_sereno) if r == p) / len(reales)
check("predecir siempre la mayoritaria da 45 % de exactitud global", abs(exactitud - 0.45) < 1e-12)
check("y sin embargo su F1 macro es bajísimo", f1 < 0.25, f"dio {f1:.3f}")
check("su balanced accuracy es 1/3", abs(cl.balanced_accuracy(reales, siempre_sereno) - 1 / 3) < 1e-12)
check("el recall de la clase nunca predicha es 0",
      cl.recall_de(reales, siempre_sereno, "Disgustado") == 0.0)

# El F1 macro NO pondera por frecuencia: una clase chica pesa igual que una grande.
reales = ["Sereno"] * 90 + ["Disgustado"] * 10
casi = ["Sereno"] * 90 + ["Interesado"] * 10           # falla toda la clase chica
check("fallar entera la clase chica hunde el F1 macro aunque sean 10 de 100",
      cl.f1_macro(reales, casi) < 0.4, f"dio {cl.f1_macro(reales, casi):.3f}")

check("una clase ausente del conjunto no rompe la balanced accuracy",
      cl.balanced_accuracy(["Sereno", "Sereno"], ["Sereno", "Sereno"]) == 1.0)
check("el recall de una clase ausente es NaN, no 0",
      cl.recall_de(["Sereno"], ["Sereno"], "Disgustado") != cl.recall_de(["Sereno"], ["Sereno"], "Disgustado"))


print("\n=== resumen: media y desvío entre folds ===")

media, desvio = cl.resumen([0.6, 0.7, 0.8])
check("media correcta", abs(media - 0.7) < 1e-12)
check("informa desvío", desvio > 0)
check("un solo valor da desvío 0 y no error", cl.resumen([0.5]) == (0.5, 0.0))
check("los NaN se ignoran", cl.resumen([0.5, float("nan"), 0.7])[0] == 0.6)


print("\n=== precisión por clase: lo que ve el usuario ===")

# Construido a mano para que precisión y recall sean claramente distintos.
predicciones = (
    [{"real": "Disgustado", "predicha": "Disgustado"}] * 7
    + [{"real": "Interesado", "predicha": "Disgustado"}] * 2
    + [{"real": "Sereno", "predicha": "Disgustado"}] * 12
    + [{"real": "Disgustado", "predicha": "Interesado"}] * 5
    + [{"real": "Interesado", "predicha": "Interesado"}] * 49
    + [{"real": "Sereno", "predicha": "Interesado"}] * 8
    + [{"real": "Disgustado", "predicha": "Sereno"}] * 9
    + [{"real": "Interesado", "predicha": "Sereno"}] * 5
    + [{"real": "Sereno", "predicha": "Sereno"}] * 44
)
pr = cl.precision_por_clase(predicciones)
check("precisión de Disgustado = 7/21", (pr["Disgustado"]["acertadas"],
                                         pr["Disgustado"]["veces_que_lo_dijo"]) == (7, 21))
check("precisión de Interesado = 49/62", (pr["Interesado"]["acertadas"],
                                          pr["Interesado"]["veces_que_lo_dijo"]) == (49, 62))
check("precisión de Sereno = 44/58", (pr["Sereno"]["acertadas"],
                                      pr["Sereno"]["veces_que_lo_dijo"]) == (44, 58))

# LA distinción que la interfaz no puede confundir. Ojo: para `Disgustado` los dos números
# coinciden por casualidad de esta matriz —el modelo predijo la clase 21 veces y hay 21 reales—,
# así que ilustrarla con esa clase no probaría nada. Se prueba donde la diferencia existe.
reales = [p["real"] for p in predicciones]
predichas = [p["predicha"] for p in predicciones]
check("en Interesado, precisión (0,79) y recall (0,88) difieren",
      abs(pr["Interesado"]["precision"] - cl.recall_de(reales, predichas, "Interesado")) > 0.05,
      f"precisión {pr['Interesado']['precision']:.3f} vs recall {cl.recall_de(reales, predichas, 'Interesado'):.3f}")
check("en Sereno la precisión (0,76) es MAYOR que el recall (0,69)",
      pr["Sereno"]["precision"] - cl.recall_de(reales, predichas, "Sereno") > 0.05,
      f"precisión {pr['Sereno']['precision']:.3f} vs recall {cl.recall_de(reales, predichas, 'Sereno'):.3f}")
check("en Disgustado coinciden, y por eso no sirve de ejemplo",
      abs(pr["Disgustado"]["precision"] - cl.recall_de(reales, predichas, "Disgustado")) < 1e-9)

check("una clase que el modelo nunca predice da precisión NaN",
      cl.precision_por_clase([{"real": "Sereno", "predicha": "Sereno"}])["Disgustado"]["precision"]
      != cl.precision_por_clase([{"real": "Sereno", "predicha": "Sereno"}])["Disgustado"]["precision"])


print("\n=== texto de confiabilidad: los tres hechos que no pueden faltar ===")

texto = cl.texto_de_confiabilidad("Disgustado", pr)
check("dice cuántas veces acertó de cuántas", "7 de 21" in texto, texto)
check("dice el porcentaje", "33%" in texto, texto)
# El texto traía `F1 macro 0,754 contra 0,691 exigido` escrito a mano: el número de otro modelo
# medido en otra corrida. Ahora sale de `metricas_globales()`, del mismo archivo del que salen las
# precisiones por clase, así que lo que se prueba es que informe el F1 del modelo vigente.
check("informa el F1 del modelo que está corriendo",
      f"{cl.metricas_globales()['f1_macro']:.3f}" in texto, texto)
check("dice que no es un diagnóstico veterinario", "NO es un diagnóstico veterinario" in texto)
check("nombra la clase predicha", "Disgustado" in texto)


print("\n=== líneas de base ===")

entrenamiento = ([{"clase": "Sereno", "id_caballo": "01"}] * 5
                 + [{"clase": "Disgustado", "id_caballo": "02"}])
prueba = [{"clase": "Disgustado", "id_caballo": "99"}]

predichas, extra = cl.predecir_mayoritaria(entrenamiento, prueba)
check("la mayoritaria predice la clase más frecuente del entrenamiento", predichas == ["Sereno"])

predichas, extra = cl.predecir_por_identidad(entrenamiento, prueba)
check("un caballo nunca visto cae al voto mayoritario", predichas == ["Sereno"])
check("y queda contado como 'sin ver'", extra["sin_ver"] == 1)

visto = [{"clase": "Disgustado", "id_caballo": "02"}]
predichas, extra = cl.predecir_por_identidad(entrenamiento, visto)
check("un caballo visto usa SU clase, no la mayoritaria", predichas == ["Disgustado"])
check("y no cuenta como 'sin ver'", extra["sin_ver"] == 0)


print("\n=== particiones ===")

clips = [{"clase": "Sereno", "id_caballo": "01", "fold": 0},
         {"clase": "Sereno", "id_caballo": "02", "fold": 1},
         {"clase": "Disgustado", "id_caballo": "03", "fold": 0}]
folds = cl.folds_por_ejemplar(clips)
check("hay tantas particiones como folds", len(folds) == 2)
check("cada clip de test está fuera de su entrenamiento",
      all(not (set(id(c) for c in ent) & set(id(c) for c in pru)) for ent, pru in folds))
check("los folds ingenuos NO respetan el ejemplar (por eso no se usan)",
      len(cl.folds_ingenuos(clips, n=2)) == 2)


print("\n" + "=" * 52)
if fallos:
    print(f"FALLARON {len(fallos)}: {fallos}")
    sys.exit(1)
print("TODAS LAS PRUEBAS PASARON")
