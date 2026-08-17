"""Lectura de los resultados del preprocesamiento facial.

Vive en el paquete y no dentro de una celda para que se pueda pedir desde cualquier notebook con
una línea, y para que el criterio de "cobertura aceptable" esté escrito en un solo lugar en vez de
recalculado a ojo en cada consulta.

Deliberadamente sin pandas: DeepLabCut fija numpy<2 y deja el stack científico de la sesión en un
estado del que pandas no siempre se recupera. Estos CSV tienen 1472 filas; la biblioteca estándar
alcanza de sobra.
"""

from __future__ import annotations

import csv
from pathlib import Path

from equinos import rutas

# Frames que el Bloque 3 extrae por clip. Si cambia allá, cambia acá.
FRAMES_POR_CLIP = 16


def _leer(nombre: str) -> list[dict]:
    ruta = rutas.carpeta("resultados") / nombre
    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe {ruta}. Hay que ejecutar la etapa roi_normalizacion."
        )
    with ruta.open(encoding="utf-8", errors="replace") as archivo:
        return list(csv.DictReader(archivo))


def _es_si(valor) -> bool:
    return str(valor).strip().lower() in ("true", "1", "si", "sí")


def cobertura(umbral: float = 0.70, peores: int = 8) -> dict:
    """Imprime el reporte de cobertura y devuelve sus números.

    `umbral` es la tasa de alineación por debajo de la cual un ejemplar se considera problemático.
    Por defecto coincide con COBERTURA_MINIMA_ACEPTABLE del notebook 02.
    """
    frames = _leer("dataset_frames_alineados.csv")
    if not frames:
        raise ValueError("dataset_frames_alineados.csv no tiene filas.")

    clips = _leer("inventario_clips.csv")
    esperados = len(clips) * FRAMES_POR_CLIP

    generados = len(frames)
    alineados = sum(1 for f in frames if _es_si(f["alineado"]))

    print("=" * 58)
    print("COBERTURA DEL PREPROCESAMIENTO FACIAL")
    print("=" * 58)
    print(f"Clips procesados : {len(clips)}")
    print(f"Frames generados : {generados} de {esperados} ({generados / esperados:.1%})")
    print(f"Alineados por kp : {alineados} de {generados} ({alineados / generados:.1%})")

    por_clase: dict[str, list[int]] = {}
    for fila in frames:
        acumulado = por_clase.setdefault(fila["clase"], [0, 0])
        acumulado[0] += 1
        acumulado[1] += _es_si(fila["alineado"])

    print("\nPor clase:")
    for clase, (total, ok) in sorted(por_clase.items()):
        print(f"  {clase:<14} {total:>5} frames   {ok / total:6.1%} alineados")

    caballos = sorted(_leer("reporte_cobertura.csv"), key=lambda f: float(f["tasa"]))
    problematicos = [f for f in caballos if float(f["tasa"]) < umbral]

    print(f"\nPeores {min(peores, len(caballos))} ejemplares:")
    for fila in caballos[:peores]:
        marca = "  <-- bajo el umbral" if float(fila["tasa"]) < umbral else ""
        print(f"  {fila['id_caballo']:<7} {fila.get('clase', ''):<14} "
              f"{float(fila['tasa']):6.1%}   {fila['frames']:>4} frames{marca}")

    tasa_global = alineados / generados
    print("\n" + "-" * 58)
    if tasa_global >= umbral and not problematicos:
        print(f"El pipeline automático es utilizable: {tasa_global:.1%} >= {umbral:.0%}")
    elif tasa_global >= umbral:
        cuantos = len(problematicos)
        print(f"Cobertura global aceptable ({tasa_global:.1%}), pero "
              f"{cuantos} ejemplar{'es' if cuantos != 1 else ''} "
              f"qued{'an' if cuantos != 1 else 'a'} por debajo del {umbral:.0%}.")
        print("Decidir qué hacer con ellos ANTES de entrenar: un ejemplar mal alineado")
        print("mete ruido sistemático, no aleatorio.")
    else:
        print(f"Cobertura global insuficiente: {tasa_global:.1%} < {umbral:.0%}")
        print("Hay que ajustar umbrales o pasar a anotación manual de las ROI.")

    return {
        "clips": len(clips),
        "frames_esperados": esperados,
        "frames_generados": generados,
        "frames_alineados": alineados,
        "tasa_global": tasa_global,
        "por_clase": {c: {"frames": t, "alineados": o, "tasa": o / t}
                      for c, (t, o) in por_clase.items()},
        "ejemplares_bajo_umbral": [f["id_caballo"] for f in problematicos],
    }


def _a_float(valor) -> float:
    try:
        return float(valor)
    except (TypeError, ValueError):
        return float("nan")


def confianzas(umbrales=(0.20, 0.30, 0.40, 0.50, 0.60), piso_por_ejemplar: float = 0.50) -> list[dict]:
    """Qué cobertura daría cada umbral de confianza, sin volver a inferir nada.

    La etapa de keypoints se diseñó para NO depender de ningún umbral nuestro: DeepLabCut ya dejó
    en `keypoints_faciales.csv` las coordenadas y su likelihood para los 1472 frames, y el filtrado
    por CONF_MINIMA_KEYPOINT ocurre después. Por eso barrer el umbral cuesta segundos de lectura de
    CSV en vez de la hora de inferencia.

    Un frame cuenta como alineable si el ojo Y el hocico superan el umbral y tienen coordenadas,
    que es exactamente la condición `par_valido` del Paso 11.

    `piso_por_ejemplar` es la cobertura mínima que se le exige a cada caballo: el promedio global
    puede verse bien mientras un ejemplar concreto queda inutilizable, y eso mete sesgo
    sistemático en lugar de ruido.
    """
    kp = _leer("keypoints_faciales.csv")
    if not kp:
        raise ValueError("keypoints_faciales.csv no tiene filas.")

    caballo = {c["clip"]: c["id_caballo"] for c in _leer("inventario_clips.csv")}
    clase = {c["clip"]: c["clase"] for c in _leer("inventario_clips.csv")}

    # Un par vale lo que valga su keypoint más flojo: ambos deben superar el umbral.
    pares = []
    for fila in kp:
        ojo, hocico = _a_float(fila["conf_ojo"]), _a_float(fila["conf_hocico"])
        tiene_coords = _a_float(fila["x_ojo"]) == _a_float(fila["x_ojo"]) \
            and _a_float(fila["x_hocico"]) == _a_float(fila["x_hocico"])
        peor = min(ojo, hocico) if tiene_coords else float("nan")
        pares.append((fila["clip"], peor))

    validos = [p for _, p in pares if p == p]
    validos.sort()
    print(f"Frames con keypoints: {len(pares)}   con coordenadas: {len(validos)}")

    if validos:
        print("\nDistribución del keypoint más flojo de cada par:")
        for pct in (10, 25, 50, 75, 90):
            print(f"  p{pct:<3} {validos[min(len(validos) - 1, pct * len(validos) // 100)]:.3f}")

    print(f"\n{'umbral':>7} {'global':>8} {'ejemplares bajo el piso':>26}")
    print("-" * 44)

    resultados = []
    for umbral in umbrales:
        por_caballo: dict[str, list[int]] = {}
        por_clase: dict[str, list[int]] = {}
        for clip, peor in pares:
            ok = peor == peor and peor >= umbral
            for mapa, llave in ((por_caballo, caballo.get(clip, "?")),
                                (por_clase, clase.get(clip, "?"))):
                acc = mapa.setdefault(llave, [0, 0])
                acc[0] += 1
                acc[1] += ok

        total = sum(a[0] for a in por_caballo.values())
        alineables = sum(a[1] for a in por_caballo.values())
        flojos = sorted(c for c, (t, o) in por_caballo.items() if o / t < piso_por_ejemplar)

        print(f"{umbral:>7.2f} {alineables / total:>7.1%} {len(flojos):>4}  {' '.join(flojos[:6])}")

        resultados.append({
            "umbral": umbral,
            "tasa_global": alineables / total,
            "por_clase": {c: o / t for c, (t, o) in por_clase.items()},
            "ejemplares_bajo_piso": flojos,
        })

    print("\nBajar el umbral no cuesta reinferir, pero un keypoint poco confiable puede estar")
    print("mal puesto: antes de fijar uno, mirar los recortes del Paso 16 con ese valor.")
    return resultados


def cuello_de_botella(umbral: float = 0.50) -> dict:
    """Cuál de los dos keypoints limita la alineación: el ojo o el hocico.

    La alineación exige que AMBOS superen el umbral, así que basta con que uno falle. Saber cuál
    cambia por completo la respuesta: si el que falla es el hocico, cabe probar otro bodypart
    —DeepLabCut expone `nose` y `upper_jaw`— y eso sí requiere reinferir, pero es un cambio de
    configuración. Si fallan los dos a la vez, el modelo directamente no está viendo la cara y no
    hay umbral ni bodypart que lo arregle.
    """
    kp = _leer("keypoints_faciales.csv")
    caballo = {c["clip"]: c["id_caballo"] for c in _leer("inventario_clips.csv")}

    solo_ojo = solo_hocico = ambos = ninguno = sin_coords = 0
    conf_ojo, conf_hocico = [], []
    fallas: dict[str, list[int]] = {}

    for fila in kp:
        ojo, hocico = _a_float(fila["conf_ojo"]), _a_float(fila["conf_hocico"])
        x_ojo, x_hocico = _a_float(fila["x_ojo"]), _a_float(fila["x_hocico"])

        if x_ojo != x_ojo or x_hocico != x_hocico:
            sin_coords += 1
            continue

        conf_ojo.append(ojo)
        conf_hocico.append(hocico)

        ok_ojo, ok_hocico = ojo >= umbral, hocico >= umbral
        if ok_ojo and ok_hocico:
            ambos += 1
        elif ok_ojo:
            solo_ojo += 1
        elif ok_hocico:
            solo_hocico += 1
        else:
            ninguno += 1

        acc = fallas.setdefault(caballo.get(fila["clip"], "?"), [0, 0, 0])
        acc[0] += 1
        acc[1] += not ok_ojo
        acc[2] += not ok_hocico

    total = ambos + solo_ojo + solo_hocico + ninguno
    if not total:
        raise ValueError("Ningún frame con coordenadas.")

    def mediana(valores):
        ordenados = sorted(v for v in valores if v == v)
        return ordenados[len(ordenados) // 2] if ordenados else float("nan")

    print(f"Frames con coordenadas: {total}   sin coordenadas: {sin_coords}")
    print(f"\nConfianza mediana   ojo {mediana(conf_ojo):.3f}   hocico {mediana(conf_hocico):.3f}")
    print(f"\nCon umbral {umbral:.2f}:")
    print(f"  ambos superan       {ambos:>5}  ({ambos / total:.1%})   <- alineables")
    print(f"  sólo el ojo         {solo_ojo:>5}  ({solo_ojo / total:.1%})   lo frena el hocico")
    print(f"  sólo el hocico      {solo_hocico:>5}  ({solo_hocico / total:.1%})   lo frena el ojo")
    print(f"  ninguno             {ninguno:>5}  ({ninguno / total:.1%})   no ve la cara")

    print("\nPor ejemplar (% de frames donde falla cada keypoint):")
    for nombre in sorted(fallas, key=lambda n: -max(fallas[n][1], fallas[n][2]) / fallas[n][0]):
        n, f_ojo, f_hocico = fallas[nombre]
        print(f"  {nombre:<6} ojo {f_ojo / n:6.1%}   hocico {f_hocico / n:6.1%}   ({n} frames)")

    print("\n" + "-" * 58)
    if ninguno > max(solo_ojo, solo_hocico):
        print("Domina 'ninguno': el modelo de pose no está encontrando la cara. Ni bajar el")
        print("umbral ni cambiar de bodypart lo resuelve; hace falta anotación manual.")
    elif solo_ojo > solo_hocico:
        print("El cuello de botella es el HOCICO. Vale probar el otro bodypart disponible")
        print("(`nose` vs `upper_jaw`) antes de anotar a mano.")
    else:
        print("El cuello de botella es el OJO, que en perfil cerrado suele quedar oculto.")
        print("Revisar si esos clips son tomas de perfil.")

    return {"ambos": ambos, "solo_ojo": solo_ojo, "solo_hocico": solo_hocico,
            "ninguno": ninguno, "sin_coordenadas": sin_coords}
