#!/usr/bin/env python3
"""Regenera el PDF «Estado actual y métricas del proyecto de tesina».

    python3 docs/informes/estado_actual_2026-10-08/generar.py [--sin-copiar]

Pasos: lee las cifras de `comparacion/resultados/*.json` (y falla si contradicen `DECISION.md`), genera las figuras
con matplotlib, arma `informe.html` con `estilo.css` y lo convierte a PDF con WeasyPrint. El PDF queda en esta
carpeta y, salvo `--sin-copiar`, se copia a la carpeta de entregas (`ENTREGA_DIR`, por omisión
`<repo>/../material para presentar/2da entrega`).

Requisitos: Python 3 con `weasyprint` (probado con 69.0) y `matplotlib`. No necesita el entorno del proyecto ni GPU.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))

import contenido  # noqa: E402
import datos  # noqa: E402
import figuras  # noqa: E402
from observaciones import observaciones  # noqa: E402

NOMBRE_PDF = "Estado_Actual_y_Metricas_Proyecto_Tesina_Equinos.pdf"
FECHA = "08/10/2026"
# Resaltados del PDF del docente (extraídos con PyMuPDF: anotaciones de tipo Highlight).
N_RESALTADOS = 59
# El IC95 de la reserva figura en DECISION.md (no en el JSON); se recalculó con el bootstrap por caballo del protocolo.
IC_RESERVA = (0.698, 0.925)


def _git(*a):
    return subprocess.run(["git", "-C", str(datos.RAIZ), *a], capture_output=True, text=True).stdout.strip()


def metadatos(D):
    assert "[0,698; 0,925]" in D["decision"], "el IC de la reserva ya no coincide con DECISION.md"
    carpeta = datos.RAIZ / "openspec" / "changes"
    abiertos = [p.name for p in carpeta.iterdir() if p.is_dir() and p.name != "archive"]
    assert not abiertos, f"hay changes abiertos: {abiertos}"
    archivados = sorted(p.name for p in (carpeta / "archive").iterdir() if p.is_dir())
    oct_ = [a for a in archivados if a.startswith("2026-10-0")]
    uso = json.loads(D["uso_reserva"][0])
    anio, mes, dia = uso["fecha"][:10].split("-")
    obs = observaciones(D, IC_RESERVA)
    cuenta = lambda e: sum(o["estado"] == e for o in obs)  # noqa: E731
    return {
        "commit": _git("rev-parse", "--short", "main"),
        "fecha_commit": _git("log", "-1", "main", "--format=%cd", "--date=format:%d/%m/%Y"),
        "fecha": FECHA, "ic_reserva": IC_RESERVA, "fecha_uso": f"{dia}/{mes}/{anio}", "uso": uso,
        "n_tests": len(list((datos.RAIZ / "tests").glob("test_*.py"))),
        "n_archivados": len(archivados), "n_arch_oct": len(oct_), "archivados_oct": oct_,
        "observaciones": obs, "n_obs": len(obs), "n_res": cuenta("Resuelta"),
        "n_par": cuenta("Atendida en parte"), "n_ab": cuenta("Abierta"), "n_resaltados": N_RESALTADOS,
    }


def main():
    copiar = "--sin-copiar" not in sys.argv
    D = datos.cargar()
    meta = metadatos(D)
    figs = figuras.generar_todas(D, AQUI / "figuras")
    contenido._titulos.clear()
    contenido._c.update(h1=0, h2=0)
    cuerpo = contenido.construir(D, figs, meta)
    cuerpo = cuerpo.replace("@@INDICE@@", contenido.indice())
    css = (AQUI / "estilo.css").read_text(encoding="utf-8")
    html = f"""<!DOCTYPE html>
<html lang="es"><head><meta charset="utf-8">
<title>Estado actual y métricas del proyecto de tesina — Expresiones faciales en equinos</title>
<meta name="author" content="Héctor R. Liceaga · Sergio M. Yañez">
<style>{css}</style></head><body>
{cuerpo}
</body></html>"""
    # Las figuras se referencian por nombre de archivo: el HTML vive junto a su carpeta de figuras.
    html = html.replace('src="fig', 'src="figuras/fig')
    ruta_html = AQUI / "informe.html"
    ruta_html.write_text(html, encoding="utf-8")

    from weasyprint import HTML
    ruta_pdf = AQUI / NOMBRE_PDF
    HTML(filename=str(ruta_html), base_url=str(AQUI) + "/").write_pdf(str(ruta_pdf))
    print("PDF:", ruta_pdf)
    if copiar:
        entrega = Path(os.environ.get("ENTREGA_DIR", datos.RAIZ.parent / "material para presentar" / "2da entrega"))
        entrega.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ruta_pdf, entrega / NOMBRE_PDF)
        print("Copiado a:", entrega / NOMBRE_PDF)


if __name__ == "__main__":
    main()
