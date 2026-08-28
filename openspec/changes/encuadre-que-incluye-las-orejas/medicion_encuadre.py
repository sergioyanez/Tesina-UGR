"""Barrido de encuadres candidatos y hoja de contacto, para decidir el encuadre del change.

La lógica reutilizable vive en `equinos.encuadre`; acá quedan sólo los candidatos concretos que
se compararon y el armado de la hoja de contacto. Que el barrido use el mismo código que va a
usar el notebook 02 es lo que hace que la tabla prediga el recorte real y no una aproximación.

Uso:
    python3 openspec/changes/encuadre-que-incluye-las-orejas/medicion_encuadre.py [salida.png]

Los sesgos de las métricas están documentados en `equinos.encuadre.barrido`. El más importante:
la inclusión de orejas se mide sólo donde el modelo de pose detectó la oreja, y una oreja fuera
del cuadro tiende a no detectarse. Favorece a todos los encuadres por igual, así que sirve para
comparar y no para afirmar un nivel absoluto. La evidencia independiente de ese sesgo es la hoja
de contacto, que es la razón de que este script también la genere.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from equinos import encuadre, rutas  # noqa: E402
from equinos.encuadre import CAMPO_VIGENTE, CampoDeVision  # noqa: E402

ANCHO = 224

# Los candidatos comparados, en unidades de la distancia ojo-hocico `d`:
# (aire por encima del ojo, aire por debajo del hocico, semiancho lateral).
CANDIDATOS = {
    "Vigente 0,33/0,33/0,83": CAMPO_VIGENTE,
    "hector  0,45/0,36/0,91": CampoDeVision(0.25 / 0.55, 0.20 / 0.55, 0.50 / 0.55),
    "C       0,80/0,30/0,75": CampoDeVision(0.80, 0.30, 0.75),
    "D       1,00/0,30/0,75": CampoDeVision(1.00, 0.30, 0.75),
    "E       1,20/0,30/0,75": CampoDeVision(1.20, 0.30, 0.75),
    "F       1,20/0,25/0,70": CampoDeVision(1.20, 0.25, 0.70),
    "G       1,40/0,25/0,70": CampoDeVision(1.40, 0.25, 0.70),
    "A       1,00/0,40/0,85": CampoDeVision(1.00, 0.40, 0.85),
}

# Los encuadres que van a la hoja de contacto, y el alto del lienzo que los contiene a todos.
EN_LA_HOJA = ("Vigente 0,33/0,33/0,83", "C       0,80/0,30/0,75",
              "D       1,00/0,30/0,75", "A       1,00/0,40/0,85")
ALTO_LIENZO = 430
EJEMPLARES = 6


def hoja_de_contacto(salida: Path, clase: str = "Disgustado") -> None:
    """Los mismos frames bajo cada encuadre, un ejemplar por fila.

    `Disgustado` por defecto: es la clase donde el rasgo importa y la que menos ejemplares tiene,
    así que es donde un encuadre malo cuesta más caro.
    """
    import csv

    import cv2
    import numpy as np

    res = rutas.carpeta("resultados")

    def leer(nombre):
        with (res / nombre).open(encoding="utf-8", errors="replace") as archivo:
            return list(csv.DictReader(archivo))

    kp = {(f["clip"], f["orden"]): f for f in leer("keypoints_faciales.csv")}
    meta = {c["clip"]: c for c in leer("inventario_clips.csv")}

    por_clip: dict = {}
    for fila in leer("dataset_frames_alineados.csv"):
        if fila["clase"] == clase:
            por_clip.setdefault(fila["clip"], []).append(fila)

    # Sólo clips íntegros: un clip a medio alinear no representa lo que el modelo va a consumir.
    integros = {c: fs for c, fs in por_clip.items()
                if sum(f["alineado"] == "True" for f in fs) == 16}

    elegidos, vistos = [], set()
    for clip in sorted(integros):
        caballo = integros[clip][0]["id_caballo"]
        if caballo in vistos:
            continue
        # Un frame del medio del clip, con par DETECTADO: mirar un frame interpolado mezclaría
        # la calidad del encuadre con la de la estimación.
        for fila in sorted(integros[clip], key=lambda r: abs(int(r["orden"]) - 8)):
            punto = kp.get((clip, fila["orden"]))
            if punto and punto["x_ojo"] and fila["origen_alineacion"] == "keypoints":
                elegidos.append((clip, fila, punto))
                vistos.add(caballo)
                break
        if len(elegidos) == EJEMPLARES:
            break

    filas_hoja = []
    for clip, fila, punto in elegidos:
        captura = cv2.VideoCapture(str(rutas.resolver(meta[clip]["ruta_clip"])))
        total = int(captura.get(cv2.CAP_PROP_FRAME_COUNT))
        indice = int(np.linspace(0, total - 1, 16)[int(fila["orden"])])
        captura.set(cv2.CAP_PROP_POS_FRAMES, indice)
        ok, frame = captura.read()
        captura.release()
        if not ok:
            continue

        x_ojo, y_ojo = float(punto["x_ojo"]), float(punto["y_ojo"])
        x_hoc, y_hoc = float(punto["x_hocico"]), float(punto["y_hocico"])
        dist = math.hypot(x_hoc - x_ojo, y_hoc - y_ojo)

        celdas = []
        for nombre in EN_LA_HOJA:
            geo = encuadre.destino(CANDIDATOS[nombre], ANCHO)
            escala = geo["px_por_d"] / dist
            theta = math.pi / 2 - math.atan2(y_hoc - y_ojo, x_hoc - x_ojo)
            cos_t, sin_t = escala * math.cos(theta), escala * math.sin(theta)
            matriz = np.array([
                [cos_t, -sin_t, geo["dest_ojo"][0] - (cos_t * x_ojo - sin_t * y_ojo)],
                [sin_t, cos_t, geo["dest_ojo"][1] - (sin_t * x_ojo + cos_t * y_ojo)],
            ])
            recorte = cv2.warpAffine(frame, matriz, (geo["ancho"], geo["alto"]),
                                     borderMode=cv2.BORDER_REPLICATE)
            lienzo = np.full((ALTO_LIENZO, ANCHO, 3), 30, np.uint8)
            lienzo[:geo["alto"]] = recorte[:ALTO_LIENZO]
            cv2.putText(lienzo, nombre.split()[0], (4, ALTO_LIENZO - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
            celdas.append(lienzo)

        tira = np.hstack(celdas)
        cv2.putText(tira, f'{fila["id_caballo"]}  {clip[:34]}', (4, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)
        filas_hoja.append(tira)

    if not filas_hoja:
        print("No se pudo armar la hoja de contacto: ningún clip íntegro legible.")
        return

    cv2.imwrite(str(salida), np.vstack(filas_hoja))
    print(f"\nHoja de contacto: {salida}  ({len(filas_hoja)} ejemplares de {clase})")


def main():
    rutas.inicializar()
    encuadre.barrido(CANDIDATOS, ANCHO)
    if len(sys.argv) > 1:
        hoja_de_contacto(Path(sys.argv[1]))
    else:
        print("\nPara generar la hoja de contacto, pasar una ruta de salida .png como argumento.")


if __name__ == "__main__":
    main()
