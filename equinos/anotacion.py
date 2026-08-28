"""Anotación manual del par ojo-hocico, para los frames que el pipeline no pudo alinear.

Existe porque quedó una franja muy chica de trabajo humano con un rendimiento desproporcionado.
Al 28/08/2026 hay **14 clips de `Disgustado` a uno o dos frames de estar completos**, 23 frames en
total: anotarlos lleva la clase de 10 a 19 ejemplares con al menos un clip íntegro, que es la
métrica que decide si el dataset admite una partición por caballo. Filmar más caballos sigue
haciendo falta, pero no es lo que está bloqueando ahora.

Lo que se anota son **dos puntos por frame**, no una caja ni una etiqueta: dónde está el ojo y
dónde está el hocico. Con esos dos puntos el pipeline arma la misma transformada de similitud que
usa para cualquier frame detectado, así que no hay un segundo camino de construcción — es la misma
regla del change `interpolacion-temporal-de-keypoints`.

Bajar `CONF_MINIMA_KEYPOINT` no es una alternativa a esto: de esos 23 frames, ninguno tiene las
dos confianzas por encima de 0,4 y siete no tienen coordenadas en absoluto. No hay nada que
rescatar aflojando el umbral.

**Se anota en el navegador, no con una ventana de OpenCV.** El `cv2` de este entorno viene sin
GUI —es el headless que arrastra DeepLabCut— y reemplazarlo por la variante con GTK es
exactamente el tipo de cambio que rompe el stack de DLC por un rato de comodidad. La página se
genera con los frames embebidos, se abre desde el disco y no necesita instalar nada.

Uso:

    ./correr_local.sh anotar          # genera la página y dice cómo abrirla
    ./correr_local.sh anotar-que      # qué falta, sin generar nada
    ./correr_local.sh anotar-cargar <csv>   # incorpora el CSV que bajó el navegador

    from equinos import anotacion
    anotacion.pendientes()            # los frames que faltan
    anotacion.pagina()                # arma el HTML
    anotacion.importar("~/Descargas/keypoints_manuales.csv")
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from equinos import rutas

# ============================================================
# CONFIGURACIÓN
# ============================================================

# Frames que el Bloque 3 extrae por clip. Si cambia en el notebook 01, cambia acá.
FRAMES_POR_CLIP = 16

# Distancia máxima al clip íntegro que se considera "vale la pena anotar". Con 2, un clip al que
# le faltan hasta dos frames entra; con más, el trabajo humano crece rápido y el rendimiento por
# frame cae, porque un clip al que le faltan 6 necesita 6 aciertos para sumar uno.
MAX_FALTANTES = 2

# Clases a las que se les ofrece anotación. Es deliberadamente restrictivo: `Interesado` y
# `Sereno` ya tienen 35 y 36 ejemplares utilizables, y anotarles frames agranda la brecha en vez
# de cerrarla.
CLASES = ("Disgustado",)

ARCHIVO_MANUAL = "keypoints_manuales.csv"
COLUMNAS = ("clip", "orden", "x_ojo", "y_ojo", "x_hocico", "y_hocico", "anotado_por", "fecha")

# Alto máximo de la ventana de anotación, en píxeles de pantalla. Los frames son de 1080x1920 y
# no entran enteros; se muestran a escala y las coordenadas se devuelven al tamaño original.
ALTO_VENTANA = 900


# ============================================================
# QUÉ FALTA ANOTAR
# ============================================================


@dataclass
class Pendiente:
    """Un frame que vale la pena anotar, con todo lo necesario para abrirlo."""

    clase: str
    id_caballo: str
    clip: str
    orden: int
    ruta_clip: str
    frames_del_clip: int
    faltan_en_el_clip: int

    @property
    def frame_video(self) -> int:
        """Índice del frame DENTRO DEL ARCHIVO DEL CLIP, no del video original."""
        return frame_del_orden(self.frames_del_clip, self.orden)


def _leer(nombre: str) -> list[dict]:
    ruta = rutas.carpeta("resultados") / nombre
    if not ruta.exists():
        raise SystemExit(f"No existe {ruta}. Correr antes las etapas 01 y 02.")
    with ruta.open(encoding="utf-8", errors="replace") as archivo:
        return list(csv.DictReader(archivo))


def _es_si(valor) -> bool:
    return str(valor).strip().lower() in ("true", "1", "si", "sí")


def frame_del_orden(total_frames: int, orden: int, num_frames: int = FRAMES_POR_CLIP) -> int:
    """A qué frame del clip corresponde el `orden`-ésimo frame muestreado.

    Replica exactamente `extraer_frames_uniformes()` del notebook 01:

        indices = np.linspace(0, total_frames - 1, num_frames).astype(int)

    Sin numpy, para no arrastrar la dependencia: `linspace` reparte `num_frames` puntos
    equiespaciados entre 0 y `total-1` inclusive, y `astype(int)` trunca.

    **Esto NO son coordenadas del video original.** El notebook extrae los frames del archivo del
    CLIP ya recortado, así que el índice es relativo a ese archivo. Usar `frame_inicio` del
    inventario como offset apunta fuera del clip y hace que se anote un frame equivocado sin que
    nada falle — que es la razón por la que esta función existe y está probada aparte.
    """
    if total_frames <= 0 or num_frames <= 0:
        return 0
    if num_frames == 1:
        return 0
    return int((total_frames - 1) * orden / (num_frames - 1))


def pendientes(max_faltantes: int = MAX_FALTANTES, clases=CLASES) -> list[Pendiente]:
    """Frames sin alinear que pertenecen a clips cerca de estar íntegros.

    El criterio no es "los frames que fallaron" —son 1576— sino los que **completan un clip**.
    Anotar un frame de un clip al que le faltan diez no suma nada: el clip sigue sin servir.
    """
    frames = _leer("dataset_frames_alineados.csv")
    clips_meta = {c["clip"]: c for c in _leer("inventario_clips.csv")}

    # clip -> [total, alineados]
    estado: dict[str, list[int]] = {}
    for fila in frames:
        acumulado = estado.setdefault(fila["clip"], [0, 0])
        acumulado[0] += 1
        acumulado[1] += _es_si(fila["alineado"])

    salida = []
    for fila in frames:
        if fila["clase"] not in clases or _es_si(fila["alineado"]):
            continue
        total, alineados = estado[fila["clip"]]
        faltan = total - alineados
        if not 1 <= faltan <= max_faltantes:
            continue
        meta = clips_meta.get(fila["clip"])
        if meta is None:
            continue
        salida.append(Pendiente(
            clase=fila["clase"],
            id_caballo=fila["id_caballo"],
            clip=fila["clip"],
            orden=int(fila["orden"]),
            ruta_clip=meta["ruta_clip"],
            frames_del_clip=int(meta["frames"]),
            faltan_en_el_clip=faltan,
        ))

    salida.sort(key=lambda p: (p.id_caballo, p.clip, p.orden))
    return salida


# ============================================================
# LO YA ANOTADO
# ============================================================


def ruta_manual() -> Path:
    return rutas.carpeta("resultados") / ARCHIVO_MANUAL


def leer_manuales() -> dict[tuple[str, int], dict]:
    """Anotaciones ya hechas, indexadas por (clip, orden)."""
    ruta = ruta_manual()
    if not ruta.exists():
        return {}
    with ruta.open(encoding="utf-8") as archivo:
        return {(f["clip"], int(f["orden"])): f for f in csv.DictReader(archivo)}


def guardar_manual(fila: dict) -> None:
    """Agrega una anotación al CSV, creándolo con su cabecera si hace falta.

    Se escribe de a una y en el momento, no al final: una sesión de anotación se interrumpe —el
    teléfono, el sueño, un frame que no se entiende— y perder veinte clics porque el proceso se
    cerró mal es la clase de fricción que hace que la tarea no se termine nunca.
    """
    ruta = ruta_manual()
    ruta.parent.mkdir(parents=True, exist_ok=True)
    existe = ruta.exists()
    with ruta.open("a", encoding="utf-8", newline="") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=list(COLUMNAS))
        if not existe:
            escritor.writeheader()
        escritor.writerow({k: fila.get(k, "") for k in COLUMNAS})


def informe(max_faltantes: int = MAX_FALTANTES) -> dict:
    """Qué falta anotar y cuánto rendiría, sin abrir ninguna ventana."""
    falta = pendientes(max_faltantes)
    hechas = leer_manuales()
    sin_hacer = [p for p in falta if (p.clip, p.orden) not in hechas]

    clips = {p.clip for p in falta}
    caballos = {p.id_caballo for p in falta}

    print("=" * 62)
    print("ANOTACIÓN MANUAL DEL PAR OJO-HOCICO")
    print("=" * 62)
    print(f"Clips a un máximo de {max_faltantes} frames de estar íntegros: {len(clips)}")
    print(f"Frames a anotar    : {len(falta)}   en {len(caballos)} ejemplares")
    print(f"Ya anotados        : {len(falta) - len(sin_hacer)}")
    print(f"Pendientes         : {len(sin_hacer)}")

    if sin_hacer:
        print()
        print(f"  {'caballo':<8} {'clip':<46} {'orden':>5} {'frame':>7}")
        for p in sin_hacer:
            print(f"  {p.id_caballo:<8} {p.clip[:46]:<46} {p.orden:>5} {p.frame_video:>7}")

    return {"total": len(falta), "hechas": len(falta) - len(sin_hacer),
            "pendientes": sin_hacer, "clips": len(clips), "caballos": len(caballos)}


# ============================================================
# LA SESIÓN INTERACTIVA
# ============================================================

AYUDA = """
  clic izquierdo   marca el punto (primero el OJO, después el HOCICO)
  u                deshace el último punto de este frame
  ENTER            confirma y pasa al siguiente (necesita los dos puntos)
  s                saltea este frame: la cara no se ve, está tapada o fuera de cuadro
  q                guarda lo hecho y sale
"""


def anotar(max_faltantes: int = MAX_FALTANTES, anotado_por: str = "") -> dict:
    """Sesión de anotación: dos clics por frame, ojo y después hocico.

    Reanudable: los frames ya anotados no se vuelven a pedir. Cada anotación se guarda en el
    momento, así que salir con `q` —o que se cierre la ventana— no pierde nada.

    Lo que NO hace, a propósito: no corrige una anotación previa. Para rehacer una, borrar su
    fila de `keypoints_manuales.csv`. Editar sobre la marcha invita a "mejorar" puntos viejos
    hasta que la anotación deja de ser reproducible.
    """
    import datetime

    import cv2

    falta = [p for p in pendientes(max_faltantes) if (p.clip, p.orden) not in leer_manuales()]
    if not falta:
        print("No queda nada por anotar.")
        return {"anotados": 0, "salteados": 0}

    print(f"{len(falta)} frames por anotar.")
    print(AYUDA)

    puntos: list[tuple[int, int]] = []
    escala = 1.0

    def al_hacer_clic(evento, x, y, banderas, parametros):
        if evento == cv2.EVENT_LBUTTONDOWN and len(puntos) < 2:
            puntos.append((int(x / escala), int(y / escala)))

    ventana = "Anotar: ojo y despues hocico"
    cv2.namedWindow(ventana, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(ventana, al_hacer_clic)

    anotados = salteados = 0
    try:
        for indice, p in enumerate(falta, 1):
            ruta = rutas.resolver(p.ruta_clip)
            captura = cv2.VideoCapture(str(ruta))
            # El total se lee del archivo, no del inventario: es el mismo valor que usó el
            # notebook 01 para muestrear, y si por lo que sea divergieran, el que manda es el
            # archivo que se está por mostrar.
            total = int(captura.get(cv2.CAP_PROP_FRAME_COUNT))
            indice_frame = frame_del_orden(total, p.orden)
            captura.set(cv2.CAP_PROP_POS_FRAMES, indice_frame)
            ok, frame = captura.read()
            captura.release()

            if total != p.frames_del_clip:
                print(f"  aviso: {p.clip} tiene {total} frames y el inventario dice "
                      f"{p.frames_del_clip}; se usa el del archivo")

            if not ok:
                print(f"  [{indice}/{len(falta)}] no se pudo leer {p.clip} frame {indice_frame}")
                salteados += 1
                continue

            alto, ancho = frame.shape[:2]
            escala = min(1.0, ALTO_VENTANA / alto)
            puntos.clear()

            while True:
                vista = cv2.resize(frame, None, fx=escala, fy=escala) if escala < 1.0 else frame.copy()

                for i, (px, py) in enumerate(puntos):
                    color = (0, 220, 255) if i == 0 else (255, 120, 0)
                    centro = (int(px * escala), int(py * escala))
                    cv2.circle(vista, centro, 6, color, 2)
                    cv2.putText(vista, "ojo" if i == 0 else "hocico",
                                (centro[0] + 10, centro[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                if len(puntos) == 2:
                    cv2.line(vista,
                             (int(puntos[0][0] * escala), int(puntos[0][1] * escala)),
                             (int(puntos[1][0] * escala), int(puntos[1][1] * escala)),
                             (0, 255, 0), 2)

                falta_marcar = "OJO" if not puntos else ("HOCICO" if len(puntos) == 1 else "ENTER para confirmar")
                cabecera = (f"[{indice}/{len(falta)}] caballo {p.id_caballo}  "
                            f"orden {p.orden}  ->  marcar {falta_marcar}")
                cv2.rectangle(vista, (0, 0), (vista.shape[1], 34), (0, 0, 0), -1)
                cv2.putText(vista, cabecera, (10, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1)

                cv2.imshow(ventana, vista)
                tecla = cv2.waitKey(20) & 0xFF

                if tecla == ord("u") and puntos:
                    puntos.pop()
                elif tecla == ord("s"):
                    print(f"  [{indice}/{len(falta)}] salteado: {p.clip} orden {p.orden}")
                    salteados += 1
                    break
                elif tecla == ord("q"):
                    print("\nSalida pedida. Lo anotado ya está guardado.")
                    return {"anotados": anotados, "salteados": salteados}
                elif tecla in (13, 10) and len(puntos) == 2:
                    guardar_manual({
                        "clip": p.clip, "orden": p.orden,
                        "x_ojo": puntos[0][0], "y_ojo": puntos[0][1],
                        "x_hocico": puntos[1][0], "y_hocico": puntos[1][1],
                        "anotado_por": anotado_por,
                        "fecha": datetime.date.today().isoformat(),
                    })
                    anotados += 1
                    print(f"  [{indice}/{len(falta)}] {p.clip} orden {p.orden}: "
                          f"ojo {puntos[0]}  hocico {puntos[1]}")
                    break
    finally:
        cv2.destroyAllWindows()

    print(f"\nAnotados: {anotados}   salteados: {salteados}")
    print(f"Guardado en: {ruta_manual()}")
    print("Para que entren al dataset hay que re-derivar los recortes: ./correr_local.sh etapa02")
    return {"anotados": anotados, "salteados": salteados}


# ============================================================
# ANOTACIÓN EN EL NAVEGADOR
# ============================================================
# El camino principal. Genera una página autocontenida con los frames embebidos como data URI:
# se abre desde el disco, se marca con el mouse y baja un CSV que después se importa.
#
# Por qué no una ventana de OpenCV: el `cv2` de este entorno es el headless que arrastra
# DeepLabCut y no tiene GUI compilada. Cambiarlo por la variante con GTK arriesga el stack que ya
# funciona —DLC fija numpy<2 y el entorno costó reconstruirlo— por un rato de comodidad. El
# navegador ya está instalado y no le debe nada al venv.

# Alto al que se embeben los frames. Más que suficiente para marcar con precisión: el error de
# un clic a esta escala es de ~2 px en coordenadas originales, contra los ~60 px que se mueve la
# cara entre dos frames muestreados.
ALTO_EMBEBIDO = 1000

_PLANTILLA = """<!doctype html>
<meta charset="utf-8">
<title>Anotar ojo y hocico</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font-family: system-ui, sans-serif; margin: 0; padding: 1.5rem;
         background: #12100f; color: #ece7e2; }}
  header {{ display: flex; gap: 1.5rem; align-items: baseline; flex-wrap: wrap;
           margin-bottom: 1rem; }}
  h1 {{ font-size: 1.1rem; margin: 0; font-weight: 600; }}
  .avance {{ font-variant-numeric: tabular-nums; color: #b5aca4; }}
  .instruccion {{ background: #1d1917; border-left: 3px solid #c9762f; padding: .7rem 1rem;
                 margin-bottom: 1rem; font-size: .9rem; }}
  .marco {{ position: relative; display: inline-block; line-height: 0; }}
  img {{ max-height: 82vh; cursor: crosshair; border: 1px solid #38312c; }}
  .lupa {{ position: fixed; right: 1.5rem; top: 6rem; width: 240px; height: 240px;
          border: 1px solid #4a423c; border-radius: 4px; background-repeat: no-repeat;
          display: none; pointer-events: none; }}
  .lupa::after {{ content: ""; position: absolute; left: 50%; top: 50%; width: 21px;
                 height: 21px; margin: -11px 0 0 -11px; border: 1px solid #ffb703;
                 border-radius: 50%; }}
  .punto {{ position: absolute; width: 14px; height: 14px; margin: -7px 0 0 -7px;
           border-radius: 50%; border: 2px solid; pointer-events: none;
           box-shadow: 0 0 0 1px rgba(0,0,0,.75); }}
  .ojo {{ border-color: #ffb703; }}
  .hocico {{ border-color: #45b3e0; }}
  .barra {{ display: flex; gap: .6rem; margin-top: 1rem; flex-wrap: wrap; }}
  button {{ font: inherit; padding: .5rem 1rem; border-radius: 4px; border: 1px solid #4a423c;
           background: #241f1c; color: #ece7e2; cursor: pointer; }}
  button:hover {{ background: #2f2825; }}
  button.principal {{ background: #c9762f; border-color: #c9762f; color: #1a1512;
                     font-weight: 600; }}
  button:disabled {{ opacity: .4; cursor: not-allowed; }}
  #salida {{ width: 100%; height: 11rem; margin-top: 1rem; font-family: ui-monospace, monospace;
            font-size: .78rem; background: #1a1613; color: #ece7e2; border: 1px solid #38312c;
            border-radius: 4px; padding: .6rem; display: none; }}
  .listo {{ text-align: center; padding: 3rem 1rem; }}
</style>

<header>
  <h1>Anotar ojo y hocico</h1>
  <span class="avance" id="avance"></span>
</header>

<div class="instruccion" id="instruccion"></div>
<div id="zona"></div>

<div class="barra">
  <button id="deshacer">Deshacer punto</button>
  <button id="saltear">Saltear (la cara no se ve)</button>
  <button id="siguiente" class="principal">Confirmar y seguir</button>
  <button id="bajar">Bajar CSV con lo hecho</button>
</div>

<textarea id="salida" readonly></textarea>

<script>
const FRAMES = {datos};
let i = 0;
const hechas = [];
let puntos = [];

const zona = document.getElementById("zona");
const avance = document.getElementById("avance");
const instruccion = document.getElementById("instruccion");

// Se arma la estructura UNA vez por frame. Antes esta funcion recreaba el innerHTML en cada
// clic: la imagen se reconstruia entera y la referencia al <img> quedaba obsoleta, asi que un
// segundo clic rapido llegaba al elemento viejo —rect de ancho 0— y guardaba NaN sin avisar.
function montarFrame() {{
  const f = FRAMES[i];
  zona.innerHTML = '<div class="marco"><img id="im" src="' + f.img + '"></div>' +
                   '<div class="lupa" id="lupa"></div>';
  const im = document.getElementById("im");
  const marco = zona.querySelector(".marco");
  const lupa = document.getElementById("lupa");

  function relativo(ev) {{
    const r = im.getBoundingClientRect();
    if (!r.width || !r.height) return null;   // imagen sin layout: se ignora el clic
    return [(ev.clientX - r.left) / r.width, (ev.clientY - r.top) / r.height];
  }}

  im.onclick = (ev) => {{
    if (puntos.length >= 2) return;
    const rel = relativo(ev);
    if (!rel) return;
    // Coordenadas en la resolucion ORIGINAL del frame, no en la que se muestra.
    puntos.push([Math.round(rel[0] * f.ancho), Math.round(rel[1] * f.alto)]);
    marcar();
  }};

  // La imagen de la lupa se asigna UNA vez: hacerlo en cada mousemove reasigna un data URI de
  // ~150 KB por evento y satura el hilo.
  lupa.style.backgroundImage = "url(" + f.img + ")";
  im.onmousemove = (ev) => {{
    const rel = relativo(ev);
    if (!rel) return;
    const r = im.getBoundingClientRect();
    lupa.style.display = "block";
    lupa.style.backgroundSize = (r.width * 4) + "px " + (r.height * 4) + "px";
    lupa.style.backgroundPosition = (120 - rel[0] * r.width * 4) + "px " +
                                    (120 - rel[1] * r.height * 4) + "px";
  }};
  im.onmouseleave = function () {{ lupa.style.display = "none"; }};

  marcar();
}}

// Redibuja solo los marcadores y el texto. No toca la imagen.
function marcar() {{
  const f = FRAMES[i];
  const marco = zona.querySelector(".marco");
  if (!marco) return;
  [...marco.querySelectorAll(".punto")].forEach(d => d.remove());

  // Los marcadores van en PORCENTAJE del marco. En pixeles calculados desde el rect, una imagen
  // que todavia no decodifico mide 0 y todos los puntos caen en la esquina superior izquierda.
  puntos.forEach((p, k) => {{
    const d = document.createElement("div");
    d.className = "punto " + (k === 0 ? "ojo" : "hocico");
    d.style.left = (100 * p[0] / f.ancho) + "%";
    d.style.top = (100 * p[1] / f.alto) + "%";
    marco.appendChild(d);
  }});

  avance.textContent = "frame " + (i + 1) + " de " + FRAMES.length +
                       "  ·  caballo " + f.caballo + "  ·  " + f.clip;
  instruccion.textContent = puntos.length === 0
    ? "Marcá el OJO (círculo amarillo)"
    : (puntos.length === 1 ? "Ahora el HOCICO (círculo azul)"
                           : "Listo — ENTER para confirmar y seguir");
}}

function pintar() {{
  if (i >= FRAMES.length) {{
    zona.innerHTML = '<div class="listo"><h2>Listo: ' + hechas.length +
      ' frames anotados</h2><p>Bajá el CSV y después importálo con<br>' +
      '<code>./correr_local.sh anotar-cargar ~/Descargas/keypoints_manuales.csv</code></p></div>';
    instruccion.textContent = "";
    avance.textContent = hechas.length + " de " + FRAMES.length;
    mostrarCsv();
    return;
  }}
  montarFrame();
}}

function mostrarCsv() {{
  const salida = document.getElementById("salida");
  salida.style.display = "block";
  salida.value = csv();
}}

function csv() {{
  const cab = "clip,orden,x_ojo,y_ojo,x_hocico,y_hocico,anotado_por,fecha\\n";
  return cab + hechas.map(h => h.join(",")).join("\\n") + (hechas.length ? "\\n" : "");
}}

document.getElementById("deshacer").onclick = () => {{ puntos.pop(); marcar(); }};
document.getElementById("saltear").onclick = () => {{ puntos = []; i++; pintar(); }};
document.getElementById("siguiente").onclick = () => {{
  if (puntos.length !== 2) return;
  const f = FRAMES[i];
  hechas.push([f.clip, f.orden, puntos[0][0], puntos[0][1], puntos[1][0], puntos[1][1],
               "{quien}", "{fecha}"]);
  puntos = []; i++; pintar();
}};
document.getElementById("bajar").onclick = () => {{
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([csv()], {{type: "text/csv"}}));
  a.download = "keypoints_manuales.csv";
  a.click();
  mostrarCsv();
}};
document.addEventListener("keydown", (ev) => {{
  if (ev.key === "Enter") document.getElementById("siguiente").click();
  if (ev.key === "u") document.getElementById("deshacer").click();
  if (ev.key === "s") document.getElementById("saltear").click();
}});

pintar();
</script>
"""


def pagina(destino: str | Path | None = None, max_faltantes: int = MAX_FALTANTES,
           anotado_por: str = "") -> Path:
    """Genera la página de anotación con los frames pendientes embebidos.

    Devuelve la ruta del HTML. Los frames van como data URI para que el archivo se pueda abrir
    o mover solo, sin depender de rutas relativas a los clips.
    """
    import base64
    import datetime

    import cv2

    falta = [p for p in pendientes(max_faltantes) if (p.clip, p.orden) not in leer_manuales()]
    if not falta:
        raise SystemExit("No queda nada por anotar.")

    entradas = []
    ilegibles = []
    for p in falta:
        captura = cv2.VideoCapture(str(rutas.resolver(p.ruta_clip)))
        total = int(captura.get(cv2.CAP_PROP_FRAME_COUNT))
        captura.set(cv2.CAP_PROP_POS_FRAMES, frame_del_orden(total, p.orden))
        ok, frame = captura.read()
        captura.release()
        if not ok or frame is None:
            ilegibles.append(f"{p.clip} orden {p.orden}")
            continue

        alto, ancho = frame.shape[:2]
        escala = min(1.0, ALTO_EMBEBIDO / alto)
        chico = cv2.resize(frame, None, fx=escala, fy=escala) if escala < 1.0 else frame
        ok, jpg = cv2.imencode(".jpg", chico, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if not ok:
            ilegibles.append(f"{p.clip} orden {p.orden} (no se pudo codificar)")
            continue

        entradas.append({
            "clip": p.clip, "orden": p.orden, "caballo": p.id_caballo,
            "ancho": ancho, "alto": alto,
            "img": "data:image/jpeg;base64," + base64.b64encode(jpg.tobytes()).decode("ascii"),
        })

    import json as _json
    html = _PLANTILLA.format(
        datos=_json.dumps(entradas),
        quien=anotado_por.replace('"', ""),
        fecha=datetime.date.today().isoformat(),
    )

    ruta = Path(destino) if destino else rutas.carpeta("resultados") / "anotar.html"
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(html, encoding="utf-8")

    print(f"Página generada: {ruta}")
    print(f"  {len(entradas)} frames embebidos · {ruta.stat().st_size / 1e6:.1f} MB")
    if ilegibles:
        print(f"  {len(ilegibles)} frames no se pudieron leer y quedaron afuera:")
        for x in ilegibles:
            print(f"    {x}")
    print()
    print("Abrila con:")
    print(f"  xdg-open '{ruta}'")
    print()
    print("Marcá ojo y hocico en cada frame, bajá el CSV al final, y después:")
    print("  ./correr_local.sh anotar-cargar ~/Descargas/keypoints_manuales.csv")
    return ruta


def importar(csv_bajado: str | Path) -> dict:
    """Incorpora el CSV que produjo la página al archivo de anotaciones del proyecto.

    No pisa lo ya anotado: una fila cuyo (clip, orden) ya existe se informa y se descarta. Para
    rehacer una anotación hay que borrar su fila a mano, que es deliberado — corregir en silencio
    hace que el dataset dependa de en qué orden se importaron los archivos.
    """
    origen = Path(csv_bajado).expanduser()
    if not origen.exists():
        raise SystemExit(f"No existe {origen}")

    existentes = leer_manuales()
    nuevas = repetidas = 0

    with origen.open(encoding="utf-8") as archivo:
        for fila in csv.DictReader(archivo):
            clave = (fila["clip"], int(fila["orden"]))
            if clave in existentes:
                repetidas += 1
                continue
            guardar_manual(fila)
            existentes[clave] = fila
            nuevas += 1

    print(f"Importadas {nuevas} anotaciones en {ruta_manual()}")
    if repetidas:
        print(f"  {repetidas} ya estaban y se descartaron (borrar a mano para rehacerlas)")
    print()
    print("Para que entren al dataset: ./correr_local.sh etapa02")
    return {"nuevas": nuevas, "repetidas": repetidas}
