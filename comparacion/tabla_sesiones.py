"""Tabla de sesiones de filmación (una fila por video) para un futuro leave-session-out.

Fecha/hora: primero del nombre del archivo (formatos de celular y de la app de captura); si el
nombre no la trae, `creation_time` de ffprobe (puede ser la fecha de re-exportación y no la de
filmación: la columna `fuente_fecha` lo distingue). `sesion_propuesta` agrupa por día y por
cercanía temporal (corte con más de 120 min entre videos consecutivos del mismo día). `lugar` y
`sesion_confirmada` quedan VACÍAS para que las complete Sergio. Sólo lectura sobre los videos.
"""
import csv, json, os, re, subprocess, sys
from datetime import datetime, timedelta
from pathlib import Path

WT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WT))

RES = Path(os.environ["TESINA_EQUINOS_RAIZ"]) / "resultados"
BASE = Path.home() / "videos_de_prueba"
CORTE_MIN = 120

PATRONES = [
    (re.compile(r"(?<!\d)(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})(?!\d)"), "ymdhms_guiones"),
    (re.compile(r"(?<!\d)(\d{4})(\d{2})(\d{2})[-_](\d{2})(\d{2})(\d{2})(?!\d)"), "yyyymmdd_hhmmss"),
    (re.compile(r"(?i)vid[-_](\d{4})(\d{2})(\d{2})[-_](\d{2})(\d{2})(\d{2})\d*"), "vid"),
]


def fecha_de_nombre(nombre):
    for rx, _ in PATRONES:
        m = rx.search(nombre)
        if m:
            try:
                return datetime(*map(int, m.groups()))
            except ValueError:
                continue
    return None


def fecha_ffprobe(ruta):
    try:
        out = subprocess.run(["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format",
                              "-show_streams", str(ruta)], capture_output=True, text=True, timeout=60).stdout
        j = json.loads(out)
    except Exception:
        return None
    tags = [j.get("format", {}).get("tags", {})] + [s.get("tags", {}) for s in j.get("streams", [])]
    for t in tags:
        v = {k.lower(): x for k, x in t.items()}.get("creation_time")
        if v:
            try:
                return datetime.fromisoformat(v.replace("Z", "+00:00")).replace(tzinfo=None)
            except ValueError:
                pass
    return None


def rotacion(ruta):
    import cv2
    c = cv2.VideoCapture(str(ruta)); r = float(c.get(cv2.CAP_PROP_ORIENTATION_META)); c.release()
    return r


def main():
    vids = list(csv.DictReader((RES / "inventario_videos.csv").open(encoding="utf-8", errors="replace")))
    filas = []
    for v in vids:
        ruta = Path(v["ruta"])
        dt, fuente = fecha_de_nombre(v["archivo"]), "nombre"
        if dt is None:
            dt = fecha_ffprobe(ruta); fuente = "ffprobe" if dt else "ninguna"
        try:
            rel = ruta.relative_to(BASE).as_posix()
        except ValueError:
            rel = ruta.as_posix()
        filas.append({"ruta": rel, "clase": v["clase"],
                      "caballo": (re.match(r"^(\d+)", v["archivo"]) or [None, "desconocido"])[1],
                      "fecha_hora": dt.isoformat(sep=" ") if dt else "", "fuente_fecha": fuente,
                      "ancho": v["ancho"], "alto": v["alto"], "fps": round(float(v["fps"]), 3),
                      "duracion_seg": round(float(v["duracion_seg"]), 2),
                      "rotacion": int(rotacion(ruta)) if ruta.exists() else "",
                      "_dt": dt})
    # sesiones: por día y cercanía temporal
    con = sorted((f for f in filas if f["_dt"]), key=lambda f: f["_dt"])
    n, prev = 0, None
    for f in con:
        if prev is None or f["_dt"].date() != prev.date() or f["_dt"] - prev > timedelta(minutes=CORTE_MIN):
            n += 1
        f["sesion_propuesta"] = f"S{n:03d}_{f['_dt'].date().isoformat()}"
        prev = f["_dt"]
    for f in filas:
        f.setdefault("sesion_propuesta", "sin_fecha")
        f["lugar"] = ""; f["sesion_confirmada"] = ""
    cols = ["ruta", "clase", "caballo", "fecha_hora", "fuente_fecha", "ancho", "alto", "fps",
            "duracion_seg", "rotacion", "sesion_propuesta", "lugar", "sesion_confirmada"]
    filas.sort(key=lambda f: (f["sesion_propuesta"], f["ruta"]))
    with (WT / "comparacion" / "sesiones_de_filmacion.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(filas)
    ses = {f["sesion_propuesta"] for f in filas} - {"sin_fecha"}
    res = {"videos": len(filas), "fecha_del_nombre": sum(f["fuente_fecha"] == "nombre" for f in filas),
           "fecha_de_ffprobe": sum(f["fuente_fecha"] == "ffprobe" for f in filas),
           "sin_fecha_deducible": sum(f["fuente_fecha"] == "ninguna" for f in filas),
           "sesiones_propuestas": len(ses), "corte_min": CORTE_MIN}
    print(json.dumps(res, indent=1))
    (WT / "comparacion" / "sesiones_resumen.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
