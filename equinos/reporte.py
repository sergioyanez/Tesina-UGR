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


def frames_perdidos(umbral: float = 0.50, peores: int = 10) -> dict:
    """Frames que la detección extrajo pero que nunca llegaron a recorte.

    Hay dos pérdidas distintas y conviene no confundirlas. Una es la de alineación: el frame se
    genera pero sin normalizar pose, porque el par ojo/hocico no supera el umbral. Esa la mide
    `cobertura()`. La otra es anterior y más cara: el frame **no se genera en absoluto**, porque
    no hay caja de detección utilizable —ni propia ni rellenada por el suavizado— o porque el
    recorte resultante es degenerado.

    La segunda se pierde antes de que los keypoints importen, así que ningún ajuste de umbral ni
    cambio de bodypart la recupera. Y no se ve en la tasa que reporta `cobertura()`, que divide
    por los frames generados y no por los esperados: el 19/08/2026 esa distinción era 68,1 %
    contra 51,9 %.

    Este reporte los agrupa por video, por clase y por extensión de archivo, que es lo que
    permite distinguir "el detector falla parejo" de "hay un subconjunto de videos que el
    pipeline no digiere".
    """
    kp = _leer("keypoints_faciales.csv")
    alineados = _leer("dataset_frames_alineados.csv")
    clips = {c["clip"]: c for c in _leer("inventario_clips.csv")}

    try:
        videos = {v["archivo"]: v for v in _leer("inventario_videos.csv")}
    except FileNotFoundError:
        videos = {}

    generados = {(f["clip"], str(f["orden"])) for f in alineados}
    esperados = [(f["clip"], str(f["orden"]), f) for f in kp]

    perdidos = [(clip, orden, fila) for clip, orden, fila in esperados
                if (clip, orden) not in generados]

    if not esperados:
        raise ValueError("keypoints_faciales.csv no tiene filas.")

    # De los perdidos, cuáles tenían el par facial en condiciones. Son los que más duelen: el
    # modelo de pose los resolvió bien y se descartaron igual, en la etapa anterior.
    recuperables = 0
    for _, _, fila in perdidos:
        ojo, hocico = _a_float(fila["conf_ojo"]), _a_float(fila["conf_hocico"])
        x_ojo, x_hoc = _a_float(fila["x_ojo"]), _a_float(fila["x_hocico"])
        if x_ojo == x_ojo and x_hoc == x_hoc and ojo >= umbral and hocico >= umbral:
            recuperables += 1

    print("=" * 62)
    print("FRAMES PERDIDOS ANTES DEL RECORTE")
    print("=" * 62)
    print(f"Frames con detección : {len(esperados)}")
    print(f"Frames generados     : {len(generados)}")
    print(f"Perdidos             : {len(perdidos)}  ({len(perdidos) / len(esperados):.1%})")
    print(f"  de ellos, con par ojo/hocico válido: {recuperables}")
    print("  (esos se habrían alineado; se perdieron por la caja, no por los keypoints)")

    def _agrupar(clave_de):
        acumulado: dict[str, list[int]] = {}
        for clip, orden, _ in esperados:
            clave = clave_de(clip)
            if clave is None:
                continue
            fila = acumulado.setdefault(clave, [0, 0])
            fila[0] += 1
            fila[1] += (clip, orden) not in generados
        return acumulado

    def _origen(clip):
        return clips.get(clip, {}).get("video_origen")

    def _extension(clip):
        origen = _origen(clip)
        return Path(origen).suffix.lower() if origen else None

    por_extension = _agrupar(_extension)
    print("\nPor extensión de archivo:")
    for ext, (total, faltan) in sorted(por_extension.items(),
                                       key=lambda kv: -kv[1][1] / max(kv[1][0], 1)):
        print(f"  {ext:<6} {faltan:>5} / {total:<5} perdidos   {faltan / total:6.1%}")

    por_clase = _agrupar(lambda c: clips.get(c, {}).get("clase"))
    print("\nPor clase:")
    for clase, (total, faltan) in sorted(por_clase.items()):
        print(f"  {clase:<14} {faltan:>5} / {total:<5} perdidos   {faltan / total:6.1%}")

    por_video = _agrupar(_origen)
    con_perdida = [(v, t, f) for v, (t, f) in por_video.items() if f]
    con_perdida.sort(key=lambda r: (-r[2] / r[1], -r[2]))

    print(f"\nPeores {min(peores, len(con_perdida))} videos de origen "
          f"({len(con_perdida)} con alguna pérdida, de {len(por_video)}):")
    for video, total, faltan in con_perdida[:peores]:
        info = videos.get(video, {})
        forma = f"{info.get('ancho', '?')}x{info.get('alto', '?')}" if info else ""
        print(f"  {faltan:>4}/{total:<4} {faltan / total:6.1%}  {forma:<11} {video[:44]}")

    completos = [v for v, t, f in con_perdida if f == t]
    if completos:
        print(f"\n{len(completos)} videos perdieron TODOS sus frames. Son los que hay que mirar "
              "primero:\n  no es que el detector falle a veces, es que no ve nada en ellos.")

    return {
        "esperados": len(esperados),
        "generados": len(generados),
        "perdidos": len(perdidos),
        "recuperables": recuperables,
        "por_extension": {k: {"total": t, "perdidos": f} for k, (t, f) in por_extension.items()},
        "por_clase": {k: {"total": t, "perdidos": f} for k, (t, f) in por_clase.items()},
        "videos_totalmente_perdidos": completos,
    }


def _perdidos_por_video() -> dict[str, tuple[int, int]]:
    """{video_origen: (frames_con_deteccion, frames_perdidos)}, leído de los artefactos."""
    kp = _leer("keypoints_faciales.csv")
    alineados = _leer("dataset_frames_alineados.csv")
    clips = {c["clip"]: c for c in _leer("inventario_clips.csv")}

    generados = {(f["clip"], str(f["orden"])) for f in alineados}
    acumulado: dict[str, list[int]] = {}
    for fila in kp:
        origen = clips.get(fila["clip"], {}).get("video_origen")
        if not origen:
            continue
        cuenta = acumulado.setdefault(origen, [0, 0])
        cuenta[0] += 1
        cuenta[1] += (fila["clip"], str(fila["orden"])) not in generados
    return {k: (v[0], v[1]) for k, v in acumulado.items()}


def orientacion_videos(peores: int = 12) -> dict:
    """Rotación declarada en los metadatos de cada video, cruzada con los frames perdidos.

    Los videos de celular guardan la orientación como metadato en vez de rotar los píxeles.
    OpenCV NO la aplica al leer: un video grabado en vertical se entrega acostado. Un detector
    entrenado sobre imágenes derechas no reconoce un caballo girado 90 grados, así que no hay
    caja, y sin caja el frame no se recorta ni llega al dataset. El síntoma es exactamente el que
    dejó 6 videos con el 100 % de sus frames perdidos el 19/08/2026.

    Si el cruce muestra que la pérdida se concentra en los videos con rotación distinta de cero,
    la corrección es barata: `cv2.CAP_PROP_ORIENTATION_AUTO` hace que OpenCV entregue el frame ya
    rotado, y hay que activarlo donde se leen los videos de origen.

    cv2 se importa acá adentro y no arriba: este módulo tiene que poder importarse aunque el
    stack de numpy esté en el estado que deja DeepLabCut.
    """
    import cv2

    videos = _leer("inventario_videos.csv")
    perdidos = _perdidos_por_video()

    filas = []
    for video in videos:
        ruta = rutas.resolver(video["ruta"])
        if not ruta.exists():
            continue
        cap = cv2.VideoCapture(str(ruta))
        rotacion = cap.get(cv2.CAP_PROP_ORIENTATION_META) if cap.isOpened() else float("nan")
        cap.release()

        total, faltan = perdidos.get(video["archivo"], (0, 0))
        filas.append({
            "archivo": video["archivo"],
            "rotacion": int(rotacion) if rotacion == rotacion else None,
            "ancho": video.get("ancho", "?"),
            "alto": video.get("alto", "?"),
            "frames": total,
            "perdidos": faltan,
        })

    if not filas:
        raise ValueError("No se pudo abrir ningún video del inventario.")

    print("=" * 62)
    print("ORIENTACIÓN DE LOS VIDEOS Y PÉRDIDA DE FRAMES")
    print("=" * 62)
    print(f"Videos inspeccionados: {len(filas)}")

    por_rotacion: dict = {}
    for f in filas:
        acc = por_rotacion.setdefault(f["rotacion"], [0, 0, 0])
        acc[0] += 1
        acc[1] += f["frames"]
        acc[2] += f["perdidos"]

    print(f"\n{'rotación':<10}{'videos':>8}{'frames':>9}{'perdidos':>10}{'tasa':>9}")
    print("-" * 62)
    for rot, (n, total, faltan) in sorted(por_rotacion.items(),
                                          key=lambda kv: (kv[0] is None, kv[0])):
        etiqueta = "sin dato" if rot is None else f"{rot}°"
        tasa = f"{faltan / total:.1%}" if total else "-"
        print(f"{etiqueta:<10}{n:>8}{total:>9}{faltan:>10}{tasa:>9}")

    rotados = [f for f in filas if f["rotacion"]]
    derechos = [f for f in filas if f["rotacion"] == 0]

    def _tasa(grupo):
        total = sum(f["frames"] for f in grupo)
        return sum(f["perdidos"] for f in grupo) / total if total else 0.0

    print("\nVideos con más pérdida:")
    for f in sorted(filas, key=lambda f: (-(f["perdidos"] / f["frames"]) if f["frames"] else 0,
                                          -f["perdidos"]))[:peores]:
        if not f["perdidos"]:
            break
        rot = "sin dato" if f["rotacion"] is None else f"{f['rotacion']}°"
        print(f"  {f['perdidos']:>4}/{f['frames']:<4} {f['perdidos'] / f['frames']:6.1%}  "
              f"rot {rot:<9} {f['archivo'][:40]}")

    print("\n" + "-" * 62)
    if rotados and derechos:
        print(f"Rotados  ({len(rotados):>3} videos): {_tasa(rotados):6.1%} de pérdida")
        print(f"Derechos ({len(derechos):>3} videos): {_tasa(derechos):6.1%} de pérdida")
        if _tasa(rotados) > 2 * max(_tasa(derechos), 0.01):
            print("\nLa pérdida se concentra en los videos rotados: OpenCV entrega el frame")
            print("acostado y el detector no reconoce al equino. Corrección: activar")
            print("CAP_PROP_ORIENTATION_AUTO donde se leen los videos de origen.")
    elif not rotados:
        print("Ningún video declara rotación: la pérdida tiene otra causa.")
        print("Siguiente sospecha: mirar un frame de los videos que perdieron todo.")

    return {"videos": filas, "por_rotacion": {str(k): v for k, v in por_rotacion.items()}}


def prueba_orientacion(archivo: str | None = None) -> dict:
    """Qué hace realmente OpenCV con un video rotado, en ESTE entorno.

    No es una pregunta retórica: el comportamiento cambia entre versiones y backends de OpenCV.
    En 4.10 local la rotación se aplica sola; el inventario producido en Colab sugiere que ahí
    no. Y la diferencia importa mucho, porque si se rota un frame que OpenCV ya rotó, el dataset
    entero queda al revés — un error peor que el original, y silencioso.

    Compara, sobre un video con rotación declarada, las dimensiones almacenadas contra la forma
    real del frame entregado, con la auto-orientación apagada y encendida.
    """
    import cv2

    videos = _leer("inventario_videos.csv")
    por_archivo = {v["archivo"]: v for v in videos}

    if archivo is None:
        for v in videos:
            ruta = rutas.resolver(v["ruta"])
            if not ruta.exists():
                continue
            cap = cv2.VideoCapture(str(ruta))
            rot = cap.get(cv2.CAP_PROP_ORIENTATION_META) if cap.isOpened() else 0
            cap.release()
            if rot:
                archivo = v["archivo"]
                break

    if archivo is None or archivo not in por_archivo:
        raise ValueError(
            "No se encontró ningún video con rotación declarada en el inventario. "
            "Pasar un nombre de archivo explícito si se quiere inspeccionar uno concreto."
        )

    video = por_archivo[archivo]
    ruta = rutas.resolver(video["ruta"])

    print("=" * 62)
    print("COMPORTAMIENTO DE OPENCV ANTE UN VIDEO ROTADO")
    print("=" * 62)
    print(f"OpenCV        : {cv2.__version__}")
    print(f"Video         : {archivo}")
    print(f"Inventario    : {video.get('ancho')} x {video.get('alto')}  (ancho x alto)")

    medidas = {}
    for etiqueta, auto in (("auto OFF", 0), ("auto ON", 1)):
        cap = cv2.VideoCapture(str(ruta))
        if not cap.isOpened():
            raise ValueError(f"No se pudo abrir {ruta}")
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, auto)
        meta = cap.get(cv2.CAP_PROP_ORIENTATION_META)
        prop = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        ok, frame = cap.read()
        cap.release()
        forma = (frame.shape[1], frame.shape[0]) if ok else None
        medidas[etiqueta] = {"meta": meta, "propiedades": prop, "frame": forma}
        print(f"\n{etiqueta}")
        print(f"  rotación declarada : {meta:.0f}")
        print(f"  propiedades        : {prop[0]} x {prop[1]}")
        print(f"  frame entregado    : {forma[0]} x {forma[1]}" if forma else "  frame: no se pudo leer")

    off, on = medidas["auto OFF"]["frame"], medidas["auto ON"]["frame"]

    print("\n" + "-" * 62)
    if off != on:
        print("La bandera SÍ cambia el frame entregado. Activar CAP_PROP_ORIENTATION_AUTO")
        print("al leer los videos de origen alcanza para corregirlo.")
        veredicto = "bandera_efectiva"
    elif off and off[0] > off[1]:
        print("La bandera NO cambia nada y el frame sale APAISADO pese a la rotación")
        print("declarada: OpenCV entrega los píxeles como están almacenados. Hay que rotar")
        print("a mano con cv2.rotate() según el metadato.")
        veredicto = "rotar_a_mano"
    else:
        print("La bandera no cambia nada, pero el frame ya sale en vertical: OpenCV aplica")
        print("la rotación por su cuenta. La pérdida de frames tiene otra causa y NO hay que")
        print("rotar nada — hacerlo dejaría el dataset al revés.")
        veredicto = "ya_rotado"

    return {"archivo": archivo, "opencv": cv2.__version__,
            "medidas": medidas, "veredicto": veredicto}


def calidad_recortes(lado: int = 224, frac_ojo: float = 0.34, frac_hocico: float = 0.74,
                     peores: int = 12) -> dict:
    """Calidad geométrica de los recortes alineados, sin abrir una sola imagen.

    Que la alineación sea numéricamente exacta no dice nada sobre si sirve. La transformada de
    similitud lleva el par ojo/hocico DETECTADO a su posición canónica con error de punto
    flotante, aunque el modelo haya puesto el hocico sobre una oreja: el error queda en cero
    igual, porque se mide contra los mismos puntos que se usaron para construirla.

    Dos cosas sí son medibles desde la geometría, y son las que separan un recorte utilizable de
    uno que sólo parece estarlo:

    - **Relleno.** `warpAffine` completa con negro lo que cae fuera del frame de origen. Si la
      cabeza está contra el borde, o el animal lejos y la escala se agranda, el cuadrado canónico
      se sale y el recorte queda mayormente vacío.
    - **Distancia ojo-hocico en el origen.** Fija la escala. Una distancia muy chica significa
      que la cara ocupaba pocos píxeles y el recorte es una ampliación de casi nada; una absurda
      —comparable al cuadro entero— delata que los keypoints no están sobre la misma cabeza.
    """
    import math

    kp = {(f["clip"], str(f["orden"])): f for f in _leer("keypoints_faciales.csv")}
    alineados = [f for f in _leer("dataset_frames_alineados.csv") if _es_si(f["alineado"])]
    clips = {c["clip"]: c for c in _leer("inventario_clips.csv")}
    videos = {v["archivo"]: v for v in _leer("inventario_videos.csv")}

    dest_ojo = (lado * 0.5, lado * frac_ojo)
    dest_hocico = (lado * 0.5, lado * frac_hocico)
    dist_destino = dest_hocico[1] - dest_ojo[1]

    # Muestreo del cuadrado canónico: se lleva cada punto al origen con la transformada inversa
    # y se cuenta cuántos caen fuera del frame. Aproximar con una grilla alcanza y evita tener
    # que calcular el área exacta de un cuadrado rotado contra un rectángulo.
    paso = 8
    rejilla = [(i * lado / (paso - 1), j * lado / (paso - 1))
               for i in range(paso) for j in range(paso)]

    filas = []
    for fila in alineados:
        clave = (fila["clip"], str(fila["orden"]))
        punto = kp.get(clave)
        origen = clips.get(fila["clip"], {}).get("video_origen")
        info = videos.get(origen, {})
        if not punto or not info:
            continue

        try:
            ancho, alto = int(info["ancho"]), int(info["alto"])
        except (KeyError, ValueError):
            continue

        x_ojo, y_ojo = _a_float(punto["x_ojo"]), _a_float(punto["y_ojo"])
        x_hoc, y_hoc = _a_float(punto["x_hocico"]), _a_float(punto["y_hocico"])
        if x_ojo != x_ojo or x_hoc != x_hoc:
            continue

        dist_origen = math.hypot(x_hoc - x_ojo, y_hoc - y_ojo)
        if dist_origen <= 0:
            continue

        escala = dist_destino / dist_origen

        # Rotación que la transformada directa aplica: lleva el eje ojo->hocico del origen a la
        # vertical del recorte. La inversa usa su traspuesta. Escribirla al revés no falla ni
        # desalinea el par —los dos puntos de anclaje salen bien igual— y sólo se manifiesta
        # como un relleno mal medido, así que se verificó punto por punto contra el par conocido.
        theta = math.pi / 2 - math.atan2(y_hoc - y_ojo, x_hoc - x_ojo)
        cos_t, sin_t = math.cos(theta), math.sin(theta)

        fuera = 0
        for qx, qy in rejilla:
            dx, dy = (qx - dest_ojo[0]) / escala, (qy - dest_ojo[1]) / escala
            px = x_ojo + dx * cos_t + dy * sin_t
            py = y_ojo - dx * sin_t + dy * cos_t
            if not (0 <= px < ancho and 0 <= py < alto):
                fuera += 1

        filas.append({
            "clip": fila["clip"],
            "orden": fila["orden"],
            "video_origen": origen,
            "id_caballo": fila.get("id_caballo", "?"),
            "clase": fila.get("clase", "?"),
            "relleno": fuera / len(rejilla),
            "dist_origen": dist_origen,
            "diagonal": math.hypot(ancho, alto),
        })

    if not filas:
        raise ValueError("No se pudo evaluar ningún recorte alineado.")

    print("=" * 62)
    print("CALIDAD GEOMÉTRICA DE LOS RECORTES ALINEADOS")
    print("=" * 62)
    print(f"Recortes evaluados: {len(filas)}")

    for etiqueta, minimo in (("nada", 0.0), ("algo (>10 %)", 0.10),
                             ("mucho (>30 %)", 0.30), ("mayormente vacío (>50 %)", 0.50)):
        n = sum(1 for f in filas if f["relleno"] > minimo) if minimo else \
            sum(1 for f in filas if f["relleno"] == 0)
        print(f"  relleno {etiqueta:<26} {n:>5}  ({n / len(filas):5.1%})")

    proporciones = sorted(f["dist_origen"] / f["diagonal"] for f in filas)
    def _pct(p):
        return proporciones[min(int(p * len(proporciones)), len(proporciones) - 1)]
    print("\nDistancia ojo-hocico como fracción de la diagonal del cuadro:")
    for p in (0.05, 0.25, 0.50, 0.75, 0.95):
        print(f"  p{int(p * 100):<3} {_pct(p):.3f}")

    sospechosos = [f for f in filas if f["relleno"] > 0.30 or
                   f["dist_origen"] / f["diagonal"] < 0.02 or
                   f["dist_origen"] / f["diagonal"] > 0.60]
    print(f"\nRecortes sospechosos: {len(sospechosos)} ({len(sospechosos) / len(filas):.1%})")
    print("  (relleno > 30 %, o distancia ojo-hocico < 2 % o > 60 % de la diagonal)")

    por_caballo: dict = {}
    for f in filas:
        acc = por_caballo.setdefault(f["id_caballo"], [0, 0])
        acc[0] += 1
        acc[1] += f in sospechosos if False else 0
    for f in sospechosos:
        por_caballo[f["id_caballo"]][1] += 1

    ranking = sorted(((c, t, s) for c, (t, s) in por_caballo.items() if s),
                     key=lambda r: -r[2] / r[1])
    if ranking:
        print(f"\nPeores {min(peores, len(ranking))} ejemplares:")
        for caballo, total, malos in ranking[:peores]:
            print(f"  {caballo:<7} {malos:>4}/{total:<4} {malos / total:6.1%} sospechosos")

    return {"evaluados": len(filas), "sospechosos": len(sospechosos),
            "por_caballo": {c: {"total": t, "sospechosos": s}
                            for c, (t, s) in por_caballo.items()}}
