"""Clasificación de la expresión a partir del clip: métricas, líneas de base y modelo final.

Este módulo contiene lo que el notebook 03 necesita para reproducir el objetivo 3. Vive acá y no
en el directorio del change porque el change se archiva, y un notebook que importe desde
`openspec/changes/...` se rompe el día que eso pasa.

**El modelo final es el clasificador de un frame con voto.** Se lo diseñó como línea de base para
poner a prueba la hipótesis del trabajo —que la evolución temporal es lo que permite clasificar— y
terminó ganándole a las 18 configuraciones probadas. Es **invariante al orden**: barajar los
frames del clip no cambia una predicción. Ese hecho es el resultado central del objetivo 3, y no
un detalle de implementación.

La accuracy global no se calcula en ningún lado. Con el reparto de este dataset —`Disgustado`
15 %, `Interesado` 40 %, `Sereno` 45 %— un clasificador que nunca prediga la clase minoritaria
llega al 85 % sin haber aprendido nada.
"""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict

from equinos import rutas

CLASES = ("Disgustado", "Interesado", "Sereno")
SEMILLA = 42

# Criterio de aceptación declarado en el change, ANTES de entrenar: la línea de base más alta
# (un frame, 0,626) más un desvío entre folds (0,065). No se toca a posteriori.
CRITERIO_ACEPTACION = 0.691


# --- métricas -------------------------------------------------------------------------------
# Se implementan acá y no se importan de sklearn para que quede a la vista qué se promedia.

def f1_macro(reales, predichas, clases=CLASES):
    """Media NO ponderada del F1 por clase: las tres pesan igual, aunque no sean igual de
    frecuentes. Es la métrica principal del objetivo 3."""
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
    """Media del recall por clase, sobre las clases presentes en el conjunto evaluado."""
    recalls = [r for r in (recall_de(reales, predichas, c) for c in clases) if r == r]
    return sum(recalls) / len(recalls) if recalls else float("nan")


def recall_de(reales, predichas, clase):
    presentes = [p for r, p in zip(reales, predichas) if r == clase]
    return sum(1 for p in presentes if p == clase) / len(presentes) if presentes else float("nan")


def resumen(valores):
    """Media y desvío. El desvío ENTRE FOLDS se informa siempre: con 3 a 7 clips de la clase
    minoritaria por fold, una media sola oculta cuánto se mueve el resultado."""
    limpios = [v for v in valores if v == v]
    if not limpios:
        return float("nan"), float("nan")
    return statistics.fmean(limpios), (statistics.stdev(limpios) if len(limpios) > 1 else 0.0)


# --- particiones ----------------------------------------------------------------------------

def folds_por_ejemplar(clips):
    """El protocolo del trabajo: los folds persistidos, que agrupan por `id_caballo`."""
    n = max(c["fold"] for c in clips) + 1
    return [([c for c in clips if c["fold"] != f], [c for c in clips if c["fold"] == f])
            for f in range(n)]


def folds_ingenuos(clips, n=4, semilla=SEMILLA):
    """El protocolo que NO se usa: reparte clips al azar, sin agrupar por ejemplar.

    Existe únicamente para medir la fuga que el protocolo real elimina. Ningún resultado del
    trabajo se reporta sobre estos folds.
    """
    import random

    mezclados = list(clips)
    random.Random(semilla).shuffle(mezclados)
    asignados = [(c, i % n) for i, c in enumerate(mezclados)]
    return [([c for c, f in asignados if f != k], [c for c, f in asignados if f == k])
            for k in range(n)]


# --- modelos --------------------------------------------------------------------------------

def predecir_mayoritaria(entrenamiento, prueba, guardar_en=None, **_):
    mayoritaria = Counter(c["clase"] for c in entrenamiento).most_common(1)[0][0]
    return [mayoritaria] * len(prueba), {"sin_ver": 0}


def predecir_por_identidad(entrenamiento, prueba, guardar_en=None, **_):
    """Predice la clase más frecuente del MISMO caballo en entrenamiento.

    Con partición por ejemplar el caballo de test nunca se vio, así que cae al voto mayoritario y
    empata con la línea de base trivial. Eso no es un defecto: es la demostración de que la
    partición neutraliza el confounder de individuo. El contador `sin_ver` lo hace explícito.
    """
    por_caballo = defaultdict(Counter)
    for c in entrenamiento:
        por_caballo[c["id_caballo"]][c["clase"]] += 1
    mayoritaria = Counter(c["clase"] for c in entrenamiento).most_common(1)[0][0]

    predichas, sin_ver = [], 0
    for c in prueba:
        visto = por_caballo.get(c["id_caballo"])
        if visto:
            predichas.append(visto.most_common(1)[0][0])
        else:
            predichas.append(mayoritaria)
            sin_ver += 1
    return predichas, {"sin_ver": sin_ver}


def predecir_un_frame(entrenamiento, prueba, vectores, indice, guardar_en=None):
    """EL MODELO FINAL. Clasifica cada frame por separado y vota entre los 16 del clip.

    Usa todos los frames pero es invariante a su orden. Cabeza lineal sobre las características
    congeladas, con pesos por clase balanceados porque la métrica es F1 macro.
    """
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    def aplanar(clips):
        filas, etiquetas, de_clip = [], [], []
        for c in clips:
            v = vectores[indice[c["clip"]]]
            filas.append(v)
            etiquetas.extend([c["clase"]] * len(v))
            de_clip.extend([c["clip"]] * len(v))
        return np.concatenate(filas), np.array(etiquetas), np.array(de_clip)

    X_ent, y_ent, _ = aplanar(entrenamiento)
    X_pru, _, clip_de_frame = aplanar(prueba)

    escalador = StandardScaler().fit(X_ent)
    modelo = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEMILLA)
    modelo.fit(escalador.transform(X_ent), y_ent)

    # Voto mayoritario, con desempate alfabético fijo para no depender del orden de iteración.
    votos = defaultdict(Counter)
    for clip, pred in zip(clip_de_frame, modelo.predict(escalador.transform(X_pru))):
        votos[clip][pred] += 1
    predichas = [sorted(votos[c["clip"]].items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
                 for c in prueba]

    if guardar_en is not None:
        guardar_en.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            guardar_en, coeficientes=modelo.coef_, intercepto=modelo.intercept_,
            clases=np.array(modelo.classes_), media=escalador.mean_, escala=escalador.scale_)
    return predichas, {"sin_ver": 0}


# --- evaluación -----------------------------------------------------------------------------

def evaluar(modelo, particiones, etiqueta="", pesos_en=None, **kwargs):
    """Validación cruzada. Devuelve métricas por fold y predicciones FUERA DE MUESTRA.

    `pesos_en` es una función `fold -> ruta` para persistir el modelo de cada fold. Los modelos
    que no guardan pesos —las líneas de base triviales— ignoran el argumento.
    """
    por_fold, fuera_de_muestra, sin_ver = [], [], 0
    for i, (entrenamiento, prueba) in enumerate(particiones):
        if not prueba or not entrenamiento:
            continue
        predichas, extra = modelo(entrenamiento, prueba,
                                  guardar_en=(pesos_en(i) if pesos_en else None), **kwargs)
        reales = [c["clase"] for c in prueba]
        sin_ver += extra.get("sin_ver", 0)
        por_fold.append({
            "fold": i, "n": len(prueba),
            "f1_macro": f1_macro(reales, predichas),
            "balanced_accuracy": balanced_accuracy(reales, predichas),
            "recall_disgustado": recall_de(reales, predichas, "Disgustado"),
        })
        fuera_de_muestra.extend(
            {"clip": c["clip"], "id_caballo": c["id_caballo"], "real": c["clase"], "predicha": p}
            for c, p in zip(prueba, predichas))

    return {
        "etiqueta": etiqueta,
        "clips": sum(f["n"] for f in por_fold),
        "f1_macro": resumen([f["f1_macro"] for f in por_fold]),
        "balanced_accuracy": resumen([f["balanced_accuracy"] for f in por_fold]),
        "recall_disgustado": resumen([f["recall_disgustado"] for f in por_fold]),
        "sin_ver": sin_ver,
        "por_fold": por_fold,
        "predicciones": fuera_de_muestra,
    }


def imprimir(resultado):
    r = resultado
    print(f"\n{r['etiqueta']}")
    print(f"  F1 macro           {r['f1_macro'][0]:.3f} ± {r['f1_macro'][1]:.3f}")
    print(f"  Balanced accuracy  {r['balanced_accuracy'][0]:.3f} ± {r['balanced_accuracy'][1]:.3f}")
    print(f"  Recall Disgustado  {r['recall_disgustado'][0]:.3f} ± {r['recall_disgustado'][1]:.3f}")
    if r["sin_ver"]:
        print(f"  Cayó al voto mayoritario en {r['sin_ver']} de {r['clips']} clips "
              f"({r['sin_ver'] / r['clips']:.0%}): no había visto nunca a ese caballo")
    print("  por fold: " + "  ".join(f"{f['f1_macro']:.3f}" for f in r["por_fold"]))
    return r


def matriz_de_confusion(predicciones, clases=CLASES):
    print("\nMATRIZ DE CONFUSIÓN — fuera de muestra, agregada sobre los folds")
    encabezado = "real / predicha"
    print(f"{encabezado:<16}" + "".join(f"{c:>13}" for c in clases) + f"{'total':>8}")
    for real in clases:
        fila = [sum(1 for p in predicciones if p["real"] == real and p["predicha"] == pred)
                for pred in clases]
        print(f"{real:<16}" + "".join(f"{v:>13}" for v in fila) + f"{sum(fila):>8}")
    print(f"{'total':<16}" + "".join(
        f"{sum(1 for p in predicciones if p['predicha'] == c):>13}" for c in clases))


def errores_por_ejemplar(predicciones, peores=10):
    """Si los fallos se concentran en pocos caballos, eso es información sobre el DATASET."""
    print("\nERRORES POR EJEMPLAR")
    por_caballo = defaultdict(lambda: [0, 0])
    for p in predicciones:
        por_caballo[p["id_caballo"]][0] += 1
        por_caballo[p["id_caballo"]][1] += (p["real"] != p["predicha"])
    fallan = {k: v for k, v in por_caballo.items() if v[1]}
    total = sum(v[1] for v in por_caballo.values())
    if not total:
        print("  sin errores")
        return
    ranking = sorted(fallan.items(), key=lambda kv: (-kv[1][1], kv[0]))
    acumulado = sum(v[1] for _, v in ranking[:peores])
    print(f"  {len(fallan)} de {len(por_caballo)} ejemplares fallan alguna vez "
          f"({total} errores en total)")
    print(f"  los {min(peores, len(ranking))} peores concentran {acumulado} ({acumulado / total:.0%})")
    print("  " + "  ".join(f"{c}:{v[1]}/{v[0]}" for c, v in ranking[:peores]))


def contra_el_criterio(resultado, criterio=CRITERIO_ACEPTACION):
    """Contrasta contra el criterio declarado ANTES de entrenar, y lo informa aunque no se cumpla.

    Que se informe igual es la condición con la que se declaró: un criterio que sólo se reporta
    cuando se cumple no es un criterio.
    """
    medio = resultado["f1_macro"][0]
    cumple = medio > criterio
    print(f"\n  criterio declarado (tarea 2.4) : {criterio:.3f}")
    print(f"  F1 macro obtenido              : {medio:.3f} ± {resultado['f1_macro'][1]:.3f}")
    print(f"  {'CUMPLE' if cumple else 'NO CUMPLE'}  (diferencia {medio - criterio:+.3f})")
    return cumple


def ruta_pesos(fold):
    """Pesos de un fold de validación cruzada. Respaldan la métrica; NO se despliegan."""
    return rutas.carpeta("modelos") / "objetivo3" / f"un_frame_fold{fold}.npz"


# --- confiabilidad que se le muestra al usuario ----------------------------------------------

ARCHIVO_EVALUACION = "objetivo3_evaluacion.json"


def precision_por_clase(predicciones=None):
    """De cuántas veces que el modelo dijo una clase, cuántas acertó.

    **Es el número que hay que mostrarle a quien usa la interfaz**, y no el recall. El recall
    contesta «de los `Disgustado` reales, ¿cuántos encontró?», que le sirve a quien evalúa el
    modelo. Quien recibe una predicción se hace la pregunta inversa: «dijo esto, ¿cuántas veces
    acierta cuando lo dice?». Son números distintos y confundirlos engaña.

    Se deriva de las predicciones FUERA DE MUESTRA de la validación cruzada. No se escribe a
    mano: escrita a mano diverge el día que se re-evalúe, y quedaría una interfaz informando una
    confiabilidad que ya no corresponde a ningún modelo.
    """
    import json

    if predicciones is None:
        ruta = rutas.carpeta("resultados") / ARCHIVO_EVALUACION
        if not ruta.exists():
            raise FileNotFoundError(
                f"No existe {ruta}. La produce la evaluación del objetivo 3; sin ella no hay de "
                "dónde sacar la confiabilidad que la interfaz tiene que mostrar.")
        predicciones = json.loads(ruta.read_text(encoding="utf-8"))["predicciones_fuera_de_muestra"]

    salida = {}
    for clase in CLASES:
        dijo = [p for p in predicciones if p["predicha"] == clase]
        acerto = sum(1 for p in dijo if p["real"] == clase)
        salida[clase] = {
            "precision": acerto / len(dijo) if dijo else float("nan"),
            "acertadas": acerto,
            "veces_que_lo_dijo": len(dijo),
        }
    return salida


def texto_de_confiabilidad(clase, precisiones=None):
    """El texto que acompaña a toda predicción. Vive acá, en un solo lugar.

    Si estuviera repetido en la interfaz, cambiar el modelo dejaría el texto viejo en pantalla.
    """
    precisiones = precisiones or precision_por_clase()
    p = precisiones[clase]
    return (
        f"Cuando este modelo predijo «{clase}», acertó {p['acertadas']} de "
        f"{p['veces_que_lo_dijo']} veces ({p['precision']:.0%}) en validación cruzada.\n"
        f"El modelo NO alcanzó el criterio declarado para considerarse útil "
        f"(F1 macro 0,626 contra 0,691 exigido).\n"
        f"Este resultado NO es un diagnóstico veterinario."
    )


# --- modelo de producción -------------------------------------------------------------------

ARCHIVO_PRODUCCION = "un_frame_produccion.npz"
ARCHIVO_PROCEDENCIA = "un_frame_produccion.json"


def ruta_produccion():
    return rutas.carpeta("modelos") / "objetivo3" / ARCHIVO_PRODUCCION


def entrenar_produccion(clips, vectores, indice, metrica_validacion=None, verbose=True):
    """Entrena sobre TODOS los clips íntegros y persiste el modelo que va a usar la interfaz.

    **Su métrica no se mide acá.** Es la de validación cruzada, y no se vuelve a calcular sobre
    los datos con los que se entrenó. Entrenar con todo mejora el modelo, pero no hay forma
    honesta de medir cuánto sin dejar material afuera; informar una exactitud calculada sobre el
    conjunto de entrenamiento sería informar cuánto memorizó.

    No se despliega ninguno de los modelos de fold: cada uno vio 3 de 4 partes del material y
    elegir cuál sería arbitrario. Promediarlos tampoco: sería un ensamble, y su métrica no es la
    que se reportó.
    """
    import json

    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    filas, etiquetas = [], []
    for c in clips:
        v = vectores[indice[c["clip"]]]
        filas.append(v)
        etiquetas.extend([c["clase"]] * len(v))
    X = np.concatenate(filas)
    y = np.array(etiquetas)

    escalador = StandardScaler().fit(X)
    modelo = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEMILLA)
    modelo.fit(escalador.transform(X), y)

    destino = ruta_produccion()
    destino.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destino, coeficientes=modelo.coef_, intercepto=modelo.intercept_,
        clases=np.array(modelo.classes_), media=escalador.mean_, escala=escalador.scale_)

    # Procedencia: de qué conjunto salió este modelo. Sin esto, dentro de dos meses no hay forma
    # de saber si los pesos corresponden al dataset vigente o a una corrida anterior.
    procedencia = {
        "modelo": "un frame + voto, cabeza lineal sobre backbone congelado",
        "clips": len(clips),
        "ejemplares": len({c["id_caballo"] for c in clips}),
        "frames_de_entrenamiento": int(len(y)),
        "clases": sorted(set(y.tolist())),
        "semilla": SEMILLA,
        "huella_caracteristicas": _huella_caracteristicas(),
        "metrica_validacion_cruzada": metrica_validacion,
        "advertencia": (
            "La métrica de este modelo es la de validación cruzada, no una calculada sobre el "
            "conjunto con el que se entrenó. No alcanza el criterio declarado de "
            f"{CRITERIO_ACEPTACION}."),
    }
    (destino.parent / ARCHIVO_PROCEDENCIA).write_text(
        json.dumps(procedencia, ensure_ascii=False, indent=1), encoding="utf-8")

    if verbose:
        print(f"Modelo de producción entrenado sobre {len(clips)} clips "
              f"({procedencia['ejemplares']} ejemplares, {len(y)} frames)")
        print(f"  pesos      : {destino}")
        print(f"  procedencia: {destino.parent / ARCHIVO_PROCEDENCIA}")
    return modelo, escalador, procedencia


def _huella_caracteristicas():
    """De qué corrida del preprocesamiento salen las características que entrenaron el modelo."""
    import json

    ruta = rutas.carpeta("resultados") / "caracteristicas_backbone.json"
    if not ruta.exists():
        return None
    return json.loads(ruta.read_text(encoding="utf-8")).get("huella")


def cargar_produccion():
    """Carga el modelo de producción para inferir. Devuelve `(coef, intercepto, clases, escalador)`."""
    import numpy as np

    ruta = ruta_produccion()
    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe {ruta}. Entrenarlo con `clasificacion.entrenar_produccion()`. "
            "Los pesos por fold NO se despliegan: cada uno vio sólo 3 de 4 partes del material.")
    d = np.load(ruta, allow_pickle=False)
    return d["coeficientes"], d["intercepto"], [str(c) for c in d["clases"]], (d["media"], d["escala"])
