"""Qué frames forman la secuencia que ve el modelo. **Un solo lugar, para las dos etapas.**

Lo usan la generación del dataset (etapa 02) y la inferencia de la interfaz. Que vivan acá y no
duplicados en cada lado no es prolijidad: la etapa 02 y `equinos.inferencia` ya habían divergido
—la primera interpolaba huecos y rescataba frames con recorte de caja, la segunda no—, y el
síntoma de esa divergencia es una interfaz que rechaza videos que el conjunto de entrenamiento
habría aceptado, informando una confiabilidad medida sobre otra distribución. No falla: miente.

**El problema que resuelve.** La versión anterior elegía 16 posiciones con `np.linspace` sobre el
clip entero y recién después miraba si esos frames tenían keypoints. Un parpadeo, un giro de
cabeza o un desenfoque que cayera en una de las 16 posiciones costaba el frame, y cinco de esos
costaban el clip completo —habiendo, en un clip de 5 s a 30 fps, otros 134 frames ya procesados
por DeepLabCut que nadie miraba. Medido sobre el conjunto propio: con las posiciones fijas sólo
82 de 393 clips tenían los 16 frames válidos.

**Lo que hace en cambio.** Busca la **ventana temporal contigua más larga** —de a lo sumo
`VENTANA_MAXIMA_SEG`— que contenga al menos 16 frames con par válido y en la que no haya ningún
tramo de más de `HUECO_MAXIMO_SEG` sin una sola medición. Dentro de esa ventana reparte 16
posiciones parejo y ancla cada una al frame válido más cercano, así que los 16 frames que entran
al modelo son mediciones reales y no posiciones inventadas.

**Por qué la ventana y no la racha de válidos.** La primera versión de esto buscaba la racha más
larga de frames válidos *consecutivos*, admitiendo huecos de 1. Medido sobre el material: aceptaba
el 87 % de los clips pero con una ventana mediana de 2,33 s, contra los 6,00 s que abarcaba el
muestreo anterior. Eran 16 frames casi simultáneos. La hipótesis del trabajo es que la evolución
temporal es lo que permite clasificar, así que una secuencia de medio segundo no es una muestra
más corta: es una muestra de otra cosa. Exigir contigüidad de la VENTANA en vez de contigüidad de
los válidos sube la aceptación al 93 % y devuelve la mediana a 5,74 s.

**Qué evita igual.** El tope de hueco sigue impidiendo lo que importaba: que la secuencia se arme
con fragmentos temporalmente desconectados. Medio segundo sin ninguna medición usable es el punto
donde deja de ser creíble que los dos lados pertenezcan al mismo gesto; más que eso corta la
ventana. No es el `MAX_HUECO_INTERPOLADO` de la etapa 02 —ese autoriza a inventar una coordenada,
y acá no se inventa ninguna— sino la tolerancia para decidir hasta dónde llega un mismo momento.

**La ventana tiene techo.** La etapa 01 corta clips de 5 a 7 s (`DURACION_MAXIMA = 7.0`), con un
piso de 1 s para videos cortos. Un video de inferencia de 15 s cuya ventana válida abarcara los
15 s daría 16 frames separados por casi un segundo: otra distribución, con la confiabilidad
medida dejando de aplicar sin que nada falle.
"""

from __future__ import annotations

from typing import NamedTuple

# Frames que consume el modelo. Es la forma de su entrada, no un parámetro.
FRAMES_POR_CLIP = 16

# Piso de frames para que un clip cuente como íntegro. Es el mismo `12 <= n <= 16` que aplican
# `equinos.particion.tabla_clips` y `equinos.inferencia.preparar_clip`: se declara acá para que
# los tres lean el mismo número en vez de repetirlo.
MINIMO_FRAMES = 12

# Techo temporal de la ventana, en segundos: la `DURACION_MAXIMA` de la etapa 01.
VENTANA_MAXIMA_SEG = 7.0

# Tramo máximo, en segundos, que la ventana puede atravesar sin una sola medición válida. Es lo
# que impide que la secuencia junte dos momentos desconectados. Va en segundos y no en frames a
# propósito: el material mezcla 30 y 60 fps, y "diez frames" significa dos cosas distintas según
# el clip, mientras que medio segundo de cara no medida significa lo mismo en los dos.
HUECO_MAXIMO_SEG = 0.5

# Cuánto puede correrse una posición de su lugar ideal para caer en un frame mejor medido, en
# segundos. Existe porque el umbral de confianza es un piso, no una garantía: DeepLabCut devuelve
# el par por encima de 0,50 en frames con motion blur donde la cara no se distingue, y medido
# sobre el material el 6 % de los frames elegidos quedaba pegado al umbral. Un frame nítido y uno
# borroso en la misma posición temporal no son datos equivalentes.
#
# Es chico a propósito. La tolerancia efectiva se recorta además a un tercio del paso entre
# posiciones, así que dos posiciones contiguas no pueden cruzarse y el reparto sigue siendo
# parejo: se corrige cuál frame, nunca dónde.
TOLERANCIA_ANCLAJE_SEG = 0.10


class Seleccion(NamedTuple):
    """Los frames elegidos, y por qué si no se pudo.

    `ordenes` son índices de frame del video, estrictamente crecientes. `motivo` es `None` cuando
    la selección sirve; si no, trae el código de la negativa y `ordenes` viene vacío.
    """

    ordenes: list[int]
    motivo: str | None
    validos: int
    racha: int
    span_seg: float

    @property
    def hay_seleccion(self) -> bool:
        return self.motivo is None


def rachas(validos: list[int], max_hueco: int) -> list[tuple[int, int]]:
    """Tramos `[inicio, fin]` de frames válidos separados por huecos de a lo sumo `max_hueco`.

    `max_hueco` se cuenta en frames faltantes entre dos válidos consecutivos: con 1, la secuencia
    `[3, 4, 6, 7]` es una sola racha `(3, 7)` —falta el 5— y `[3, 4, 7]` son dos, porque entre el
    4 y el 7 faltan dos.
    """
    if not validos:
        return []
    orden = sorted(set(validos))
    tramos = []
    inicio = anterior = orden[0]
    for actual in orden[1:]:
        if actual - anterior - 1 > max_hueco:
            tramos.append((inicio, anterior))
            inicio = actual
        anterior = actual
    tramos.append((inicio, anterior))
    return tramos


def mejor_ventana(validos: list[int], *, tope_frames: int, hueco_frames: int,
                  n: int) -> tuple[int, int] | None:
    """La ventana contigua más larga con al menos `n` válidos y sin agujeros mayores a `hueco_frames`.

    Se maximiza el LARGO y no la cantidad de válidos adentro. La diferencia importa: maximizar la
    cantidad elige el tramo donde DeepLabCut anduvo mejor, que suele ser medio segundo de cara
    quieta, y devuelve 16 frames casi idénticos. Maximizar el largo elige el tramo que más se
    parece a lo que el modelo vio al entrenarse, que es un gesto desplegándose en varios segundos.

    Empates: se queda con la ventana más temprana. Es arbitrario, pero es determinista, y la
    reproducibilidad vale más acá que la preferencia.
    """
    if len(validos) < n:
        return None

    mejor = None
    for i in range(len(validos) - n + 1):
        j = i
        while j + 1 < len(validos):
            if validos[j + 1] - validos[j] - 1 > hueco_frames:
                break
            if validos[j + 1] - validos[i] + 1 > tope_frames:
                break
            j += 1
        if j - i + 1 >= n:
            largo = validos[j] - validos[i]
            if mejor is None or largo > mejor[2]:
                mejor = (validos[i], validos[j], largo)
    return (mejor[0], mejor[1]) if mejor else None


def _anclar(objetivo: float, disponibles: list[int], usados: set[int],
            confianzas: dict[int, float] | None = None, tolerancia: float = 0.0) -> int | None:
    """El frame válido sin usar que mejor sirve para la posición `objetivo`.

    Sin `confianzas` es el más cercano, desempatando hacia el más temprano. Con `confianzas`, entre
    los que caen a menos de `tolerancia` del objetivo elige el mejor medido, y sólo si ninguno cae
    dentro de la tolerancia vuelve al más cercano. Así la corrección nunca mueve una posición más
    de lo declarado: si al lado no hay nada mejor, se queda con lo que había.
    """
    candidatos = [f for f in disponibles if f not in usados]
    if not candidatos:
        return None
    if not confianzas or tolerancia <= 0:
        return min(candidatos, key=lambda f: (abs(f - objetivo), f))

    cercanos = [f for f in candidatos if abs(f - objetivo) <= tolerancia]
    if not cercanos:
        return min(candidatos, key=lambda f: (abs(f - objetivo), f))
    return max(cercanos, key=lambda f: (confianzas.get(f, 0.0), -abs(f - objetivo), -f))


def elegir_frames(validos, *, total: int, fps: float = 30.0, n: int = FRAMES_POR_CLIP,
                  minimo: int = MINIMO_FRAMES,
                  ventana_maxima_seg: float = VENTANA_MAXIMA_SEG,
                  hueco_maximo_seg: float = HUECO_MAXIMO_SEG,
                  confianzas: dict[int, float] | None = None,
                  tolerancia_seg: float = TOLERANCIA_ANCLAJE_SEG) -> Seleccion:
    """Elige `n` frames dentro de la mejor ventana temporal, en orden temporal estricto.

    `validos` son los índices de frame con par ojo-hocico por encima del umbral. La función NO
    conoce umbrales ni confianzas: quién es válido lo decide quien llama, con el criterio de la
    etapa 02, y acá sólo se elige entre los que ya pasaron.

    Si ninguna ventana reúne `n` válidos, afloja hasta `minimo` antes de rendirse. Por debajo de
    eso devuelve una `Seleccion` con motivo y sin órdenes, que es lo que las dos etapas convierten
    en descarte del clip o en negativa de la interfaz.
    """
    validos = sorted(set(int(v) for v in validos if 0 <= int(v) < total))
    if not validos:
        return Seleccion([], "SIN_KEYPOINTS", 0, 0, 0.0)

    fps = fps if fps and fps > 0 else 30.0
    tope_frames = max(n, int(round(ventana_maxima_seg * fps)))
    hueco_frames = max(1, int(round(hueco_maximo_seg * fps)))

    ventana = cuantos = None
    for objetivo in range(n, minimo - 1, -1):
        ventana = mejor_ventana(validos, tope_frames=tope_frames,
                                hueco_frames=hueco_frames, n=objetivo)
        if ventana is not None:
            cuantos = objetivo
            break

    if ventana is None:
        return Seleccion([], "VENTANA_CORTA", len(validos), 0, 0.0)

    inicio, fin = ventana
    en_ventana = [f for f in validos if inicio <= f <= fin]

    # Objetivos repartidos parejo sobre la ventana, cada uno anclado a una medición real. Se ancla
    # en vez de interpolar porque dentro de la ventana siempre hay un frame medido cerca: inventar
    # una coordenada teniendo una medida al lado es peor dato, no más dato.
    paso = (fin - inicio) / (cuantos - 1) if cuantos > 1 else 0.0
    # Un tercio del paso es el techo: más que eso deja de ser un reparto parejo, y dos posiciones
    # contiguas podrían cruzarse y romper el orden temporal.
    tolerancia = min(tolerancia_seg * fps, paso / 3) if confianzas else 0.0
    usados: set[int] = set()
    elegidos = []
    for i in range(cuantos):
        ancla = _anclar(inicio + paso * i, en_ventana, usados, confianzas, tolerancia)
        if ancla is None:
            break
        usados.add(ancla)
        elegidos.append(ancla)

    elegidos.sort()
    if len(elegidos) < minimo:
        return Seleccion([], "VENTANA_CORTA", len(validos), len(en_ventana), 0.0)

    span = (elegidos[-1] - elegidos[0] + 1) / fps
    return Seleccion(elegidos, None, len(validos), len(en_ventana), span)


def frames_validos(pares, umbral: float) -> list[int]:
    """Los `orden` cuyo par ojo-hocico supera `umbral` en ambos puntos.

    `pares` son las filas que devuelve `equinos.pose.seleccionar_par`. Es el mismo predicado que
    la etapa 02 escribe como `par_valido`, y está acá para que la inferencia no lo reescriba.
    """
    validos = []
    for p in pares:
        if p.get("x_ojo") in (None, "") or p.get("x_hocico") in (None, ""):
            continue
        try:
            if (float(p.get("conf_ojo") or 0) >= umbral
                    and float(p.get("conf_hocico") or 0) >= umbral):
                validos.append(int(p["orden"]))
        except (TypeError, ValueError):
            continue
    return validos


def confianzas_del_par(pares, umbral: float) -> dict[int, float]:
    """`{orden: confianza del punto más flojo}` para los frames válidos.

    Se toma el mínimo entre ojo y hocico porque alinear necesita los dos: la calidad del frame la
    fija el peor de los dos puntos, no su promedio.
    """
    salida = {}
    for p in pares:
        if p.get("x_ojo") in (None, "") or p.get("x_hocico") in (None, ""):
            continue
        try:
            ojo, hocico = float(p.get("conf_ojo") or 0), float(p.get("conf_hocico") or 0)
        except (TypeError, ValueError):
            continue
        if ojo >= umbral and hocico >= umbral:
            salida[int(p["orden"])] = min(ojo, hocico)
    return salida
