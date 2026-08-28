"""Modelo final del objetivo 3: entrenamiento por fold, evaluación, ablación y errores.

**El modelo elegido es el clasificador de un frame con voto.** No es el que se esperaba elegir.
Se lo diseñó como línea de base para poner a prueba la hipótesis del trabajo, y terminó siendo el
mejor de las 18 configuraciones probadas — también el más simple y el más estable. Elegir otro
porque «es el que corresponde a un trabajo sobre video» sería elegir contra la evidencia.

Cabeza lineal sobre las características del backbone congelado, entrenada a nivel de frame y
agregada por voto mayoritario entre los 16 frames del clip. **Es invariante al orden**: barajar el
clip no cambia una predicción. Eso es exactamente lo que hace que la hipótesis quede sin sostén.

Uso:
    python3 openspec/changes/clasificacion-orientada-a-video/evaluacion_final.py
"""

import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from equinos import caracteristicas, particion, rutas  # noqa: E402

CLASES = ("Disgustado", "Interesado", "Sereno")
SEMILLA = 42

# Criterio declarado en la tarea 2.4, ANTES de entrenar. No se toca.
CRITERIO = 0.691
LINEA_DE_BASE_UN_FRAME = 0.626


def _f1_macro(reales, predichas):
    f1s = []
    for c in CLASES:
        vp = sum(1 for r, p in zip(reales, predichas) if r == c and p == c)
        fp = sum(1 for r, p in zip(reales, predichas) if r != c and p == c)
        fn = sum(1 for r, p in zip(reales, predichas) if r == c and p != c)
        pr = vp / (vp + fp) if (vp + fp) else 0.0
        rc = vp / (vp + fn) if (vp + fn) else 0.0
        f1s.append(2 * pr * rc / (pr + rc) if (pr + rc) else 0.0)
    return sum(f1s) / len(f1s)


def _balanced_accuracy(reales, predichas):
    recalls = []
    for c in CLASES:
        presentes = [p for r, p in zip(reales, predichas) if r == c]
        if presentes:
            recalls.append(sum(1 for p in presentes if p == c) / len(presentes))
    return sum(recalls) / len(recalls) if recalls else float("nan")


def _recall(reales, predichas, clase):
    presentes = [p for r, p in zip(reales, predichas) if r == clase]
    return sum(1 for p in presentes if p == clase) / len(presentes) if presentes else float("nan")


def entrenar_fold(entrenamiento, prueba, vectores, indice, guardar_en=None):
    """Entrena a nivel de frame y agrega por voto. Devuelve las predicciones por clip."""
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

    X_e, y_e, _ = aplanar(entrenamiento)
    X_p, _, clip_de_frame = aplanar(prueba)

    escalador = StandardScaler().fit(X_e)
    modelo = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEMILLA)
    modelo.fit(escalador.transform(X_e), y_e)

    votos = defaultdict(Counter)
    for clip, pred in zip(clip_de_frame, modelo.predict(escalador.transform(X_p))):
        votos[clip][pred] += 1
    predichas = [sorted(votos[c["clip"]].items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
                 for c in prueba]

    if guardar_en is not None:
        guardar_en.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            guardar_en, coeficientes=modelo.coef_, intercepto=modelo.intercept_,
            clases=np.array(modelo.classes_), media=escalador.mean_, escala=escalador.scale_)
    return predichas


def evaluar(clips, vectores, indice, etiqueta, guardar=False):
    """Validación cruzada completa. Devuelve métricas por fold y predicciones fuera de muestra."""
    folds = sorted({c["fold"] for c in clips})
    por_fold, fuera_de_muestra = [], []
    destino = rutas.carpeta("modelos") / "objetivo3"

    for f in folds:
        ent = [c for c in clips if c["fold"] != f]
        pru = [c for c in clips if c["fold"] == f]
        if not pru or not ent:
            continue
        ruta = (destino / f"un_frame_fold{f}.npz") if guardar else None
        predichas = entrenar_fold(ent, pru, vectores, indice, ruta)
        reales = [c["clase"] for c in pru]
        por_fold.append({
            "fold": f, "n": len(pru),
            "f1_macro": _f1_macro(reales, predichas),
            "balanced_accuracy": _balanced_accuracy(reales, predichas),
            "recall_disgustado": _recall(reales, predichas, "Disgustado"),
        })
        fuera_de_muestra.extend(
            {"clip": c["clip"], "id_caballo": c["id_caballo"], "real": c["clase"], "predicha": p}
            for c, p in zip(pru, predichas))

    f1 = [r["f1_macro"] for r in por_fold]
    ba = [r["balanced_accuracy"] for r in por_fold]
    rd = [r["recall_disgustado"] for r in por_fold if r["recall_disgustado"] == r["recall_disgustado"]]
    return {
        "etiqueta": etiqueta,
        "clips": len(clips),
        "ejemplares": len({c["id_caballo"] for c in clips}),
        "f1_macro": (statistics.fmean(f1), statistics.stdev(f1) if len(f1) > 1 else 0.0),
        "balanced_accuracy": (statistics.fmean(ba), statistics.stdev(ba) if len(ba) > 1 else 0.0),
        "recall_disgustado": (statistics.fmean(rd), statistics.stdev(rd) if len(rd) > 1 else 0.0),
        "por_fold": por_fold,
        "predicciones": fuera_de_muestra,
    }


def imprimir_resumen(r):
    print(f"\n{r['etiqueta']}  —  {r['clips']} clips, {r['ejemplares']} ejemplares")
    print(f"  F1 macro           {r['f1_macro'][0]:.3f} ± {r['f1_macro'][1]:.3f}")
    print(f"  Balanced accuracy  {r['balanced_accuracy'][0]:.3f} ± {r['balanced_accuracy'][1]:.3f}")
    print(f"  Recall Disgustado  {r['recall_disgustado'][0]:.3f} ± {r['recall_disgustado'][1]:.3f}")
    print("  por fold: " + "  ".join(
        f"f{p['fold']} {p['f1_macro']:.3f} (n={p['n']})" for p in r["por_fold"]))


def matriz_de_confusion(predicciones):
    print("\nMATRIZ DE CONFUSIÓN — predicciones FUERA DE MUESTRA, agregadas sobre los 4 folds")
    encabezado = "real / predicha"
    print(f"{encabezado:<16}" + "".join(f"{c:>13}" for c in CLASES) + f'{"total":>8}')
    for real in CLASES:
        fila = [sum(1 for p in predicciones if p["real"] == real and p["predicha"] == pred)
                for pred in CLASES]
        print(f"{real:<16}" + "".join(f"{v:>13}" for v in fila) + f"{sum(fila):>8}")
    print(f'{"total":<16}' + "".join(
        f'{sum(1 for p in predicciones if p["predicha"] == c):>13}' for c in CLASES))


def errores_por_ejemplar(predicciones, peores=10):
    print("\nERRORES POR EJEMPLAR — ¿se concentran en pocos caballos?")
    por_caballo = defaultdict(lambda: [0, 0])
    for p in predicciones:
        por_caballo[p["id_caballo"]][0] += 1
        por_caballo[p["id_caballo"]][1] += (p["real"] != p["predicha"])
    fallan = {k: v for k, v in por_caballo.items() if v[1]}
    total_err = sum(v[1] for v in por_caballo.values())
    print(f"  {len(fallan)} de {len(por_caballo)} ejemplares tienen al menos un error "
          f"({total_err} errores en total)")
    ranking = sorted(fallan.items(), key=lambda kv: (-kv[1][1], kv[0]))
    acumulado = sum(v[1] for _, v in ranking[:peores])
    print(f"  los {min(peores, len(ranking))} peores concentran {acumulado} de {total_err} "
          f"errores ({acumulado / total_err:.0%})")
    print("  " + "  ".join(f"{c}:{v[1]}/{v[0]}" for c, v in ranking[:peores]))


def main():
    rutas.inicializar(verbose=False)
    clips = particion.cargar()
    orden, vectores = caracteristicas.cargar()
    indice = {c: i for i, c in enumerate(orden)}

    print("=" * 78)
    print("MODELO FINAL DEL OBJETIVO 3 — un frame + voto, sobre backbone congelado")
    print("=" * 78)
    print("Elegido por evidencia entre 18 configuraciones. Es INVARIANTE AL ORDEN.")

    completo = evaluar(clips, vectores, indice, "COMPLETO (141 clips íntegros)", guardar=True)
    imprimir_resumen(completo)

    print("\n" + "-" * 78)
    print("CONTRA EL CRITERIO DECLARADO ANTES DE ENTRENAR (tarea 2.4)")
    print("-" * 78)
    medio = completo["f1_macro"][0]
    print(f"  criterio declarado : {CRITERIO:.3f}")
    print(f"  F1 macro obtenido  : {medio:.3f} ± {completo['f1_macro'][1]:.3f}")
    print(f"  {'CUMPLE' if medio > CRITERIO else 'NO CUMPLE'}  "
          f"(diferencia {medio - CRITERIO:+.3f})")

    matriz_de_confusion(completo["predicciones"])

    print("\n" + "-" * 78)
    print("ABLACIÓN POR origen_alineacion (tarea 4.4)")
    print("-" * 78)
    print("Sólo los clips cuyos 16 frames se alinearon por detección propia del modelo de pose,")
    print("sin interpolación ni anotación manual. MISMOS folds, para que sea comparable.")
    detectados = [c for c in clips if c["solo_detectado"]]
    reparto = Counter(c["clase"] for c in detectados)
    por_fold = Counter(c["fold"] for c in detectados)
    print(f"  reparto: " + "  ".join(f"{k} {v}" for k, v in sorted(reparto.items())))
    print(f"  clips por fold: {dict(sorted(por_fold.items()))}")
    ablacion = evaluar(detectados, vectores, indice, "ABLACIÓN (sólo detección propia)")
    imprimir_resumen(ablacion)
    print(f"\n  Diferencia contra el completo: "
          f"{ablacion['f1_macro'][0] - completo['f1_macro'][0]:+.3f} de F1 macro, "
          f"con {len(detectados)} clips contra {len(clips)}.")

    errores_por_ejemplar(completo["predicciones"])

    salida = rutas.carpeta("resultados") / "objetivo3_evaluacion.json"
    salida.write_text(json.dumps(
        {"completo": {k: v for k, v in completo.items() if k != "predicciones"},
         "ablacion": {k: v for k, v in ablacion.items() if k != "predicciones"},
         "criterio": CRITERIO, "cumple": bool(medio > CRITERIO),
         "predicciones_fuera_de_muestra": completo["predicciones"]},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nResultados guardados en {salida}")
    print(f"Pesos por fold en {rutas.carpeta('modelos') / 'objetivo3'}")


if __name__ == "__main__":
    main()
