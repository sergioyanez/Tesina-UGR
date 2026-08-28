"""Las tres líneas de base del objetivo 3, sobre la partición persistida.

  1. clase mayoritaria
  2. sólo `id_caballo`, sin mirar la imagen
  3. un frame a la vez con el backbone congelado, y voto — sin información temporal


Sin ellas, un F1 macro de 0,55 no significa nada: puede ser un modelo que aprendió algo o uno que
descubrió la clase mayoritaria. Se corren ANTES que el modelo, y el criterio de aceptación del
objetivo 3 se ancla en sus números.

La de identidad es la menos habitual y la más importante de las dos. Mide cuánta señal hay en el
confounder que este proyecto viene combatiendo desde el principio: si la clase está determinada
por el individuo, cualquier modelo que mire la imagen puede estar aprendiendo eso mismo por otra
vía, y ningún resultado sobre este material distinguiría expresión de ejemplar.

**Se mide con DOS protocolos, y la comparación entre ambos es el resultado.**

  - Con `GroupKFold` por ejemplar —el protocolo real del trabajo— un caballo de test nunca se vio
    en entrenamiento, así que el modelo de identidad no tiene con qué predecirlo y cae al voto
    mayoritario. Se espera que empate con la línea de base trivial. Eso NO es un defecto de la
    medición: es la prueba de que la partición neutraliza el confounder.
  - Con una partición ingenua por clip —la que haría alguien que no agrupa— el mismo caballo cae
    de los dos lados, y el modelo de identidad predice sin mirar una sola imagen. Ese número es
    la medida del confounder, y es el que justifica todo el protocolo.

Uso:
    python3 openspec/changes/clasificacion-orientada-a-video/lineas_de_base.py
"""

import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from equinos import caracteristicas, particion, rutas  # noqa: E402

SEMILLA = 42
CLASES = ("Disgustado", "Interesado", "Sereno")


# --- métricas ------------------------------------------------------------------------------
# Se implementan acá, en vez de importar sklearn, para que quede a la vista qué se está
# promediando. La accuracy global NO se calcula: con 15 % / 40 % / 45 %, un clasificador que
# nunca prediga `Disgustado` llega al 85 % sin haber aprendido nada.

def f1_macro(reales, predichas, clases=CLASES):
    """Media no ponderada del F1 por clase. Trata a las tres clases como igual de importantes."""
    f1s = []
    for c in clases:
        vp = sum(1 for r, p in zip(reales, predichas) if r == c and p == c)
        fp = sum(1 for r, p in zip(reales, predichas) if r != c and p == c)
        fn = sum(1 for r, p in zip(reales, predichas) if r == c and p != c)
        precision = vp / (vp + fp) if (vp + fp) else 0.0
        recall = vp / (vp + fn) if (vp + fn) else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if (precision + recall) else 0.0)
    return sum(f1s) / len(f1s)


def balanced_accuracy(reales, predichas, clases=CLASES):
    """Media del recall por clase, sobre las clases realmente presentes."""
    recalls = []
    for c in clases:
        presentes = [p for r, p in zip(reales, predichas) if r == c]
        if presentes:
            recalls.append(sum(1 for p in presentes if p == c) / len(presentes))
    return sum(recalls) / len(recalls) if recalls else float("nan")


def recall_de(reales, predichas, clase):
    presentes = [p for r, p in zip(reales, predichas) if r == clase]
    return sum(1 for p in presentes if p == clase) / len(presentes) if presentes else float("nan")


# --- modelos que no miran la imagen ---------------------------------------------------------

def predecir_mayoritaria(entrenamiento, prueba):
    mayoritaria = Counter(c["clase"] for c in entrenamiento).most_common(1)[0][0]
    return [mayoritaria] * len(prueba), 0


def predecir_por_identidad(entrenamiento, prueba):
    """Predice la clase más frecuente del MISMO caballo en entrenamiento.

    Devuelve también cuántas veces tuvo que caer al voto mayoritario por no haber visto nunca a
    ese caballo. Ese contador es la mitad del resultado: si es el 100 %, el modelo de identidad
    no está usando la identidad para nada.
    """
    clases_por_caballo = defaultdict(Counter)
    for c in entrenamiento:
        clases_por_caballo[c["id_caballo"]][c["clase"]] += 1
    mayoritaria = Counter(c["clase"] for c in entrenamiento).most_common(1)[0][0]

    predichas, sin_ver = [], 0
    for c in prueba:
        visto = clases_por_caballo.get(c["id_caballo"])
        if visto:
            predichas.append(visto.most_common(1)[0][0])
        else:
            predichas.append(mayoritaria)
            sin_ver += 1
    return predichas, sin_ver


def predecir_por_frame_y_votar(entrenamiento, prueba, vectores, indice):
    """Clasifica CADA frame por separado con el backbone congelado, y vota entre los 16.

    Es la línea de base que pone a prueba la hipótesis del trabajo. Usa los 16 frames, pero es
    **invariante a su orden**: barajar el clip no cambia una sola predicción. Por eso aísla
    exactamente lo que el modelo secuencial tiene que aportar —la evolución temporal— y no
    cualquier otra ventaja del backbone o de tener más frames.

    Cabeza lineal sobre las características congeladas, con pesos por clase balanceados: la
    métrica es F1 macro, así que dejar que `Sereno` domine la pérdida mediría otra cosa.
    """
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    def aplanar(clips):
        filas, etiquetas, de_clip = [], [], []
        for c in clips:
            v = vectores[indice[c["clip"]]]           # (16, 512)
            filas.append(v)
            etiquetas.extend([c["clase"]] * len(v))
            de_clip.extend([c["clip"]] * len(v))
        return np.concatenate(filas), np.array(etiquetas), np.array(de_clip)

    X_ent, y_ent, _ = aplanar(entrenamiento)
    X_pru, _, clip_de_frame = aplanar(prueba)

    escalador = StandardScaler().fit(X_ent)
    cabeza = LogisticRegression(max_iter=2000, class_weight="balanced")
    cabeza.fit(escalador.transform(X_ent), y_ent)
    por_frame = cabeza.predict(escalador.transform(X_pru))

    # Voto mayoritario entre los frames de cada clip. El desempate es por orden alfabético de
    # clase, fijo, para que el resultado no dependa del orden de iteración.
    votos = defaultdict(Counter)
    for clip, prediccion in zip(clip_de_frame, por_frame):
        votos[clip][prediccion] += 1
    predichas = [sorted(votos[c["clip"]].items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
                 for c in prueba]
    return predichas, 0


# --- protocolos de partición ----------------------------------------------------------------

def folds_por_ejemplar(clips):
    """El protocolo real: GroupKFold por `id_caballo`, leído de la partición persistida."""
    n = max(c["fold"] for c in clips) + 1
    return [([c for c in clips if c["fold"] != f], [c for c in clips if c["fold"] == f])
            for f in range(n)]


def folds_ingenuos(clips, n=4, semilla=SEMILLA):
    """El protocolo que NO hay que usar: reparte clips al azar, sin agrupar por ejemplar.

    Existe sólo para medir cuánto sube la métrica por la fuga, que es lo que justifica el
    protocolo real. No se usa para reportar ningún resultado del trabajo.
    """
    mezclados = list(clips)
    random.Random(semilla).shuffle(mezclados)
    asignados = [(c, i % n) for i, c in enumerate(mezclados)]
    return [([c for c, f in asignados if f != k], [c for c, f in asignados if f == k])
            for k in range(n)]


def evaluar(nombre, modelo, particiones):
    filas = []
    for i, (entrenamiento, prueba) in enumerate(particiones):
        reales = [c["clase"] for c in prueba]
        predichas, sin_ver = modelo(entrenamiento, prueba)
        filas.append({
            "fold": i,
            "n": len(prueba),
            "f1_macro": f1_macro(reales, predichas),
            "balanced_accuracy": balanced_accuracy(reales, predichas),
            "recall_disgustado": recall_de(reales, predichas, "Disgustado"),
            "sin_ver": sin_ver,
        })
    return {"nombre": nombre, "folds": filas}


def _resumen(valores):
    media = statistics.fmean(valores)
    desvio = statistics.stdev(valores) if len(valores) > 1 else 0.0
    return media, desvio


def imprimir(resultado):
    filas = resultado["folds"]
    f1_m, f1_d = _resumen([f["f1_macro"] for f in filas])
    ba_m, ba_d = _resumen([f["balanced_accuracy"] for f in filas])
    rd = [f["recall_disgustado"] for f in filas if f["recall_disgustado"] == f["recall_disgustado"]]
    rd_m, rd_d = _resumen(rd) if rd else (float("nan"), float("nan"))
    sin_ver = sum(f["sin_ver"] for f in filas)
    total = sum(f["n"] for f in filas)

    print(f"\n{resultado['nombre']}")
    print(f"  F1 macro           {f1_m:.3f} ± {f1_d:.3f}")
    print(f"  Balanced accuracy  {ba_m:.3f} ± {ba_d:.3f}")
    print(f"  Recall Disgustado  {rd_m:.3f} ± {rd_d:.3f}")
    if sin_ver:
        print(f"  Cayó al voto mayoritario en {sin_ver} de {total} clips ({sin_ver / total:.0%}) "
              f"por no haber visto nunca a ese caballo")
    print("  por fold: " + "  ".join(f"F1 {f['f1_macro']:.3f}" for f in filas))
    return {"f1_macro": (f1_m, f1_d), "balanced_accuracy": (ba_m, ba_d)}


def main():
    rutas.inicializar(verbose=False)
    clips = particion.cargar()
    print("=" * 70)
    print("LÍNEAS DE BASE QUE NO MIRAN LA IMAGEN")
    print("=" * 70)
    print(f"{len(clips)} clips íntegros, {len({c['id_caballo'] for c in clips})} ejemplares")
    reparto = Counter(c["clase"] for c in clips)
    print("Reparto: " + "  ".join(f"{k} {v} ({v / len(clips):.0%})"
                                  for k, v in sorted(reparto.items())))

    print("\n" + "-" * 70)
    print("PROTOCOLO REAL — GroupKFold por ejemplar (el del trabajo)")
    print("-" * 70)
    por_ejemplar = folds_por_ejemplar(clips)
    mayoritaria = imprimir(evaluar("1. Clase mayoritaria", predecir_mayoritaria, por_ejemplar))
    identidad = imprimir(evaluar("2. Sólo id_caballo, sin mirar la imagen",
                                 predecir_por_identidad, por_ejemplar))

    orden_clips, vectores = caracteristicas.cargar()
    indice = {clip: i for i, clip in enumerate(orden_clips)}
    un_frame = imprimir(evaluar(
        "3. Un frame a la vez + voto, sin información temporal",
        lambda ent, pru: predecir_por_frame_y_votar(ent, pru, vectores, indice),
        por_ejemplar))
    print("     (usa los 16 frames pero es invariante a su orden: barajar el clip no cambia")
    print("      ninguna predicción. Aísla lo que el modelo secuencial tiene que aportar.)")

    print("\n" + "-" * 70)
    print("PROTOCOLO INGENUO — reparto al azar por clip, SIN agrupar por ejemplar")
    print("-" * 70)
    print("No se usa para reportar nada del trabajo. Mide cuánto sube la métrica por la fuga,")
    print("que es lo que justifica agrupar por ejemplar.")
    ingenuo = imprimir(evaluar("2b. Sólo id_caballo, partición ingenua",
                               predecir_por_identidad, folds_ingenuos(clips)))


    print("\n" + "=" * 70)
    print("LECTURA")
    print("=" * 70)
    delta = ingenuo["f1_macro"][0] - identidad["f1_macro"][0]
    print(f"La identidad del ejemplar predice la clase con F1 macro "
          f"{ingenuo['f1_macro'][0]:.3f} cuando la partición no agrupa por caballo, y "
          f"{identidad['f1_macro'][0]:.3f} cuando sí.")
    print(f"Diferencia: {delta:+.3f}. Eso es la fuga que la partición por ejemplar elimina, y el")
    print("motivo por el que ningún número de este trabajo se mide sobre cortes por clip.")
    print()
    piso = max(mayoritaria["f1_macro"][0], identidad["f1_macro"][0], un_frame["f1_macro"][0])
    print(f"El piso contra el que hay que leer el modelo es F1 macro {piso:.3f}, "
          f"que lo fija la línea de base de un frame.")
    print()
    print("HIPÓTESIS DEL TRABAJO: la expresión es un evento dinámico y la evolución temporal es")
    print("lo que permite clasificarla. Queda a prueba contra ese 3: si el modelo secuencial no")
    print("le gana, la hipótesis no queda sostenida por la evidencia de este dataset, y eso hay")
    print("que escribirlo tal cual.")
    return {"mayoritaria": mayoritaria, "identidad": identidad,
            "un_frame": un_frame, "ingenuo": ingenuo, "piso": piso}


if __name__ == "__main__":
    main()
