"""Campo de visión del recorte facial, y barrido de encuadres candidatos.

Este módulo es el único lugar donde se decide *cuánta cara* entra en el recorte. El notebook 02 lo
invoca para derivar las posiciones canónicas del ojo y del hocico, y el barrido lo invoca para
comparar candidatos. Que sea el mismo código en ambos casos es lo que hace que el barrido prediga
el recorte real en vez de una aproximación.

**Por qué el encuadre se declara como campo de visión y no como dos fracciones.** Históricamente el
recorte se parametrizaba con `FRAC_OJO` y `FRAC_HOCICO` sobre un lado único y cuadrado. Esa
parametrización ata el campo vertical al lateral: no se puede pedir más aire arriba sin pedir más a
los costados. Y el déficit medido del encuadre es asimétrico —la punta de oreja está a 0,81 `d` por
encima del ojo en la mediana y el recorte vigente muestra 0,33 `d`, mientras que lateralmente sobra
campo—, así que la parametrización vieja ocultaba el problema por construcción.

Acá el encuadre son tres números en unidades de la **distancia ojo-hocico `d`**: aire por encima
del ojo, aire por debajo del hocico y semiancho lateral. `FRAC_OJO` y `FRAC_HOCICO` se derivan de
ellos, no al revés.

El recorte resultante NO es cuadrado, y eso no deforma la cara: la deformación la impide la
**isotropía de la transformada de similitud**, no la forma del destino. La forma del destino
determina el campo de visión. Confundir ambas cosas es lo que ató el encuadre a un cuadrado.

**Por qué se puede barrer sin reinferir.** `pose_completa.csv` guarda los 39 bodyparts de cada
frame, incluidos `left_earbase`, `left_earend` y sus simétricos, aunque la etapa 02 sólo use el par
facial. Proyectar esos keypoints por la transformada de un encuadre candidato contesta cuántas
orejas entrarían, sin abrir una imagen ni pagar la hora de inferencia.

Deliberadamente sin pandas ni numpy, como `pose` y `reporte`: DeepLabCut fija `numpy<2` y deja el
stack científico de la sesión en un estado del que pandas no siempre se recupera. El barrido se
corre justo después de la etapa 02, que es exactamente cuando el stack está en ese estado.
"""

from __future__ import annotations

import math
from typing import NamedTuple

from equinos import reporte

# Keypoints de oreja del esquema de superanimal_quadruped. La punta (`earend`) es la que define la
# orientación de la oreja, que es el rasgo etológicamente informativo; la base sola no alcanza para
# leer si la oreja está echada hacia atrás.
OREJAS = ("left_earbase", "left_earend", "right_earbase", "right_earend")

# Confianza mínima para creerle a un keypoint de oreja. Mismo umbral que usa la etapa 02 para el
# par facial: no hay motivo para exigirle más ni menos a un keypoint que al otro.
CONF_MINIMA_KEYPOINT = 0.50

# Lado del muestreo del recorte para estimar el relleno. Mismo criterio que
# `reporte.calidad_recortes`: se lleva cada punto de la rejilla al origen con la transformada
# inversa y se cuenta cuántos caen fuera del frame. Aproximar con una rejilla evita calcular el
# área exacta de un rectángulo rotado contra otro, y a 8x8 el error es despreciable frente a la
# dispersión entre frames.
PASO_REJILLA = 8


class CampoDeVision(NamedTuple):
    """Encuadre en unidades de la distancia ojo-hocico `d`.

    El marco de la cabeza tiene el ojo en 0 y el hocico en 1 sobre el eje axial, y el eje lateral
    perpendicular. El recorte abarca `[-aire_sobre_ojo, 1 + aire_bajo_hocico]` en axial y
    `[-semiancho, +semiancho]` en lateral.
    """

    aire_sobre_ojo: float
    aire_bajo_hocico: float
    semiancho: float

    @property
    def extension_axial(self) -> float:
        return self.aire_sobre_ojo + 1.0 + self.aire_bajo_hocico

    @property
    def aspecto(self) -> float:
        """Alto sobre ancho que el campo de visión exige al recorte de destino."""
        return self.extension_axial / (2.0 * self.semiancho)

    @property
    def frac_ojo(self) -> float:
        return self.aire_sobre_ojo / self.extension_axial

    @property
    def frac_hocico(self) -> float:
        return (self.aire_sobre_ojo + 1.0) / self.extension_axial


# El encuadre vigente al 28/08/2026, expresado como campo de visión. Sale de FRAC_OJO=0,20 y
# FRAC_HOCICO=0,80 sobre un cuadrado: la separación es 0,60, así que el recorte abarca 1/0,60 veces
# la distancia ojo-hocico y el ojo queda a 0,20/0,60 de `d` del borde superior.
CAMPO_VIGENTE = CampoDeVision(0.20 / 0.60, 0.20 / 0.60, 0.50 / 0.60)


def destino(campo: CampoDeVision, ancho: int = 224) -> dict:
    """Geometría del recorte de destino para un campo de visión dado.

    El alto NO se elige: se deriva del aspecto que el campo exige, con el ancho fijo. Elegir los
    dos por separado deformaría la cara o cambiaría el campo sin querer.

    Como el alto se redondea a un entero de píxeles, el campo realmente representado difiere
    apenas del pedido. Se devuelve el **realizado**, no el pedido, para que lo que se informe sea
    lo que el recorte hace.
    """
    if ancho <= 0:
        raise ValueError(f"El ancho del recorte debe ser positivo, se recibió {ancho}.")

    alto = int(round(ancho * campo.aspecto))
    dest_ojo = (ancho * 0.5, alto * campo.frac_ojo)
    dest_hocico = (ancho * 0.5, alto * campo.frac_hocico)

    # Píxeles del recorte que ocupa la distancia ojo-hocico. Es la escala de la transformada, y
    # también la unidad en la que están medidos los topes del relleno temporal.
    px_por_d = dest_hocico[1] - dest_ojo[1]

    # Campo realizado tras el redondeo del alto: el semiancho que efectivamente cubre el recorte.
    semiancho_real = (ancho * 0.5) / px_por_d if px_por_d else float("nan")

    return {
        "ancho": ancho,
        "alto": alto,
        "dest_ojo": dest_ojo,
        "dest_hocico": dest_hocico,
        "px_por_d": px_por_d,
        "semiancho_real": semiancho_real,
        "aire_sobre_ojo_real": dest_ojo[1] / px_por_d if px_por_d else float("nan"),
    }


def _transformada(x_ojo, y_ojo, x_hocico, y_hocico, geo):
    """Escala y ángulo que llevan el eje ojo-hocico del origen a la vertical del recorte.

    Se escribe una sola vez y se usa en los dos sentidos: hacia el destino para proyectar los
    keypoints de oreja, y hacia el origen para muestrear el relleno. Dos copias con el signo del
    ángulo escrito al revés no fallarían —los dos puntos de anclaje salen bien igual— y sólo se
    manifestarían como una métrica mal medida.
    """
    dist_origen = math.hypot(x_hocico - x_ojo, y_hocico - y_ojo)
    if dist_origen <= 0:
        return None

    escala = geo["px_por_d"] / dist_origen
    theta = math.pi / 2 - math.atan2(y_hocico - y_ojo, x_hocico - x_ojo)
    return escala, theta, dist_origen


def proyectar(punto, x_ojo, y_ojo, x_hocico, y_hocico, geo):
    """Lleva un punto del frame de origen a coordenadas del recorte."""
    t = _transformada(x_ojo, y_ojo, x_hocico, y_hocico, geo)
    if t is None:
        return None
    escala, theta, _ = t
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    dx, dy = punto[0] - x_ojo, punto[1] - y_ojo
    return (geo["dest_ojo"][0] + escala * (dx * cos_t - dy * sin_t),
            geo["dest_ojo"][1] + escala * (dx * sin_t + dy * cos_t))


def dentro(punto, geo) -> bool:
    """¿El punto cae dentro del recorte?"""
    if punto is None:
        return False
    return 0 <= punto[0] < geo["ancho"] and 0 <= punto[1] < geo["alto"]


def relleno(x_ojo, y_ojo, x_hocico, y_hocico, geo, ancho_frame, alto_frame) -> float:
    """Fracción del recorte que `warpAffine` completaría con píxeles inventados.

    Se muestrea el recorte con una rejilla y se lleva cada punto al frame de origen con la
    transformada inversa. Los que caen fuera del frame son los que el borde replicado inventa.

    Medir sobre el rectángulo y no sobre su caja envolvente importa: el rectángulo va rotado
    —el eje ojo-hocico está a 24 grados de la vertical del cuadro en la mediana— y su caja
    envolvente sobrestima el relleno de forma sistemática.
    """
    t = _transformada(x_ojo, y_ojo, x_hocico, y_hocico, geo)
    if t is None:
        return float("nan")
    escala, theta, _ = t
    cos_t, sin_t = math.cos(theta), math.sin(theta)

    ancho, alto = geo["ancho"], geo["alto"]
    dest_ojo = geo["dest_ojo"]

    fuera = total = 0
    for i in range(PASO_REJILLA):
        for j in range(PASO_REJILLA):
            qx = i * (ancho - 1) / (PASO_REJILLA - 1)
            qy = j * (alto - 1) / (PASO_REJILLA - 1)
            dx, dy = (qx - dest_ojo[0]) / escala, (qy - dest_ojo[1]) / escala
            px = x_ojo + dx * cos_t + dy * sin_t
            py = y_ojo - dx * sin_t + dy * cos_t
            total += 1
            if not (0 <= px < ancho_frame and 0 <= py < alto_frame):
                fuera += 1
    return fuera / total


# --- barrido -------------------------------------------------------------------------------


def _frames_medibles() -> list[dict]:
    """Frames con par ojo-hocico DETECTADO, con sus orejas y las dimensiones de su video.

    Sólo detectados: medir el encuadre sobre posiciones interpoladas o anotadas a mano mezclaría
    dos fuentes de error, y lo que se está comparando es geometría, no cobertura.
    """
    from equinos import pose

    orejas_por_frame: dict[tuple[str, str], dict[str, tuple[float, float, float]]] = {}
    for fila in pose.leer_pose():
        if fila["bodypart"] not in OREJAS:
            continue
        try:
            punto = (float(fila["x"]), float(fila["y"]), float(fila["confianza"]))
        except (TypeError, ValueError):
            continue
        clave = (fila["clip"], str(fila["orden"]))
        orejas_por_frame.setdefault(clave, {})[fila["bodypart"]] = punto

    clips = {c["clip"]: c for c in reporte._leer("inventario_clips.csv")}
    videos = {v["archivo"]: v for v in reporte._leer("inventario_videos.csv")}

    medibles = []
    for fila in reporte._leer("keypoints_faciales.csv"):
        try:
            x_ojo, y_ojo = float(fila["x_ojo"]), float(fila["y_ojo"])
            x_hoc, y_hoc = float(fila["x_hocico"]), float(fila["y_hocico"])
        except (TypeError, ValueError):
            continue                      # sin par detectado

        origen = clips.get(fila["clip"], {}).get("video_origen")
        info = videos.get(origen, {})
        try:
            ancho_frame, alto_frame = int(info["ancho"]), int(info["alto"])
        except (KeyError, ValueError):
            continue

        medibles.append({
            "clip": fila["clip"],
            "orden": str(fila["orden"]),
            "ojo": (x_ojo, y_ojo),
            "hocico": (x_hoc, y_hoc),
            "ancho_frame": ancho_frame,
            "alto_frame": alto_frame,
            "orejas": orejas_por_frame.get((fila["clip"], str(fila["orden"])), {}),
        })
    return medibles


def _percentil(valores: list[float], p: float) -> float:
    if not valores:
        return float("nan")
    ordenados = sorted(valores)
    return ordenados[min(int(p * len(ordenados)), len(ordenados) - 1)]


def evaluar(campo: CampoDeVision, ancho: int = 224, medibles: list[dict] | None = None) -> dict:
    """Mide un encuadre candidato sobre la corrida vigente, sin escribir una sola imagen.

    Devuelve la inclusión de keypoints de oreja y la distribución del relleno. Ambas métricas
    tienen sesgos conocidos que hay que informar junto a ellas — ver el docstring de `barrido`.
    """
    if medibles is None:
        medibles = _frames_medibles()
    geo = destino(campo, ancho)

    rellenos: list[float] = []
    orejas_medidas = orejas_dentro = 0
    con_vector = vector_legible = 0

    for frame in medibles:
        x_ojo, y_ojo = frame["ojo"]
        x_hoc, y_hoc = frame["hocico"]

        valor = relleno(x_ojo, y_ojo, x_hoc, y_hoc, geo,
                        frame["ancho_frame"], frame["alto_frame"])
        if valor == valor:
            rellenos.append(valor)

        for nombre, (ox, oy, conf) in frame["orejas"].items():
            if conf < CONF_MINIMA_KEYPOINT:
                continue
            orejas_medidas += 1
            if dentro(proyectar((ox, oy), x_ojo, y_ojo, x_hoc, y_hoc, geo), geo):
                orejas_dentro += 1

        # Vector de oreja: base Y punta de una misma oreja, las dos dentro del recorte. Es la
        # métrica que decide, y no el conteo suelto de keypoints: lo que informa el estado del
        # animal es la ORIENTACIÓN de la oreja —echada hacia atrás, aplanada—, y eso se lee del
        # vector base->punta. Una base sin su punta no dice hacia dónde apunta la oreja, así que
        # contar keypoints sueltos sobrestima cuánta señal queda dentro del cuadro.
        hay = legible = False
        for lado in ("left", "right"):
            base = frame["orejas"].get(lado + "_earbase")
            punta = frame["orejas"].get(lado + "_earend")
            if (base is None or punta is None
                    or base[2] < CONF_MINIMA_KEYPOINT or punta[2] < CONF_MINIMA_KEYPOINT):
                continue
            hay = True
            if (dentro(proyectar(base[:2], x_ojo, y_ojo, x_hoc, y_hoc, geo), geo)
                    and dentro(proyectar(punta[:2], x_ojo, y_ojo, x_hoc, y_hoc, geo), geo)):
                legible = True
        con_vector += hay
        vector_legible += legible

    return {
        "campo": campo,
        "geometria": geo,
        "frames": len(medibles),
        "orejas_medidas": orejas_medidas,
        "orejas_dentro": orejas_dentro / orejas_medidas if orejas_medidas else float("nan"),
        "frames_con_vector": con_vector,
        "vector_legible": vector_legible / con_vector if con_vector else float("nan"),
        "relleno_p50": _percentil(rellenos, 0.50),
        "relleno_p90": _percentil(rellenos, 0.90),
        "relleno_mayor_25": (sum(1 for r in rellenos if r > 0.25) / len(rellenos)
                             if rellenos else float("nan")),
        "sin_relleno": (sum(1 for r in rellenos if r == 0) / len(rellenos)
                        if rellenos else float("nan")),
    }


def barrido(candidatos: dict[str, CampoDeVision], ancho: int = 224) -> list[dict]:
    """Compara encuadres candidatos e imprime la tabla de decisión.

    **Dos sesgos conocidos, que deben acompañar cualquier número que salga de acá:**

    1. La inclusión de orejas se mide sólo sobre frames donde la oreja se detectó con confianza
       suficiente. Una oreja fuera del cuadro tiende a no detectarse, así que el denominador
       excluye justo los peores casos. Favorece a TODOS los encuadres por igual: sirve para
       comparar, no para afirmar un nivel absoluto.
    2. El relleno se estima muestreando el recorte con una rejilla de 8x8. Es exacto salvo por la
       discretización, que afecta a todos los encuadres por igual.

    Ninguno de los dos se corrige con más aritmética sobre los mismos keypoints. La evidencia
    independiente es mirar los recortes.
    """
    medibles = _frames_medibles()
    print("=" * 96)
    print("BARRIDO DE ENCUADRES")
    print("=" * 96)
    print(f"Frames con par ojo-hocico detectado: {len(medibles)}\n")

    cabecera = (f'{"Encuadre":<28}{"destino":>9}{"d->px":>7}{"vector oreja":>14}'
                f'{"kp orejas":>11}{"relleno p50":>13}{"p90":>7}{">25%":>8}')
    print(cabecera)
    print("-" * len(cabecera))

    resultados = []
    for nombre, campo in candidatos.items():
        r = evaluar(campo, ancho, medibles)
        g = r["geometria"]
        forma = "{}x{}".format(g["ancho"], g["alto"])
        print(f'{nombre:<28}{forma:>9}{g["px_por_d"]:>7.0f}'
              f'{r["vector_legible"]:>14.1%}{r["orejas_dentro"]:>11.1%}'
              f'{r["relleno_p50"]:>13.1%}{r["relleno_p90"]:>7.1%}{r["relleno_mayor_25"]:>8.1%}')
        resultados.append(r)

    print("\n'vector oreja' es la métrica que decide: base Y punta de una misma oreja dentro del")
    print("recorte. El conteo suelto de keypoints la sobrestima, porque una base sin su punta no")
    print("dice hacia dónde apunta la oreja, que es lo que informa el estado del animal.")

    print("\nSesgos de estas métricas: ver el docstring de equinos.encuadre.barrido.")
    return resultados
