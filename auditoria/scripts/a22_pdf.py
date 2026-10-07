"""Genera auditoria/condiciones_cumplidas.pdf a partir de auditoria/secciones/NN_*.md (ya ensamblados).
Requiere el venv ~/.venvs/pdf-auditoria (markdown) y el ejecutable `weasyprint`:
    ~/.venvs/pdf-auditoria/bin/python auditoria/scripts/a22_pdf.py
"""
import re
import subprocess
from pathlib import Path

import markdown

AUD = Path(__file__).resolve().parent.parent
SEC = AUD / "secciones"
SALIDA = AUD / "condiciones_cumplidas.pdf"

CSS = """
@page { size: A4; margin: 17mm 14mm 17mm 14mm;
  @bottom-center { content: counter(page) " / " counter(pages); font-size: 8pt; color: #666; }
  @top-right { content: "Auditoría técnica y metodológica — Tesina equinos"; font-size: 7.5pt; color: #888; } }
@page portada { margin: 0; @bottom-center { content: none; } @top-right { content: none; } }
html { font-family: 'DejaVu Sans', 'Liberation Sans', sans-serif; font-size: 8.8pt; line-height: 1.38; color: #1c1c1c; }
.portada { page: portada; height: 296mm; overflow: hidden; background: #12304a; color: white; padding: 45mm 22mm 0 22mm; box-sizing: border-box; page-break-after: always; }
.portada h1 { font-size: 28pt; line-height: 1.15; margin: 0 0 8mm 0; border: none; color: white; page-break-before: avoid; }
.portada .sub { font-size: 12pt; margin-bottom: 12mm; color: #cfe2f0; }
.portada table { color: white; font-size: 10pt; border: none; }
.portada td { border: none !important; background: none !important; padding: 1.5mm 3mm 1.5mm 0; color: white; vertical-align: top; }
.portada table { width: auto; }
.portada .pie { position: absolute; bottom: 20mm; font-size: 9pt; color: #9fb9cc; }
h1 { font-size: 16pt; color: #12304a; border-bottom: 2px solid #12304a; padding-bottom: 2mm; page-break-before: always; margin-top: 0; }
h2 { font-size: 12pt; color: #1b5e7a; margin-top: 6mm; page-break-after: avoid; }
h3 { font-size: 10pt; color: #1b5e7a; page-break-after: avoid; }
p { margin: 1.6mm 0; text-align: left; }
table { border-collapse: collapse; width: 100%; margin: 2.5mm 0; font-size: 7.4pt; line-height: 1.25; table-layout: auto; page-break-inside: auto; }
th { background: #12304a; color: white; text-align: left; padding: 1.2mm 1.6mm; font-weight: 600; }
td { border: 0.4pt solid #b8c4cc; padding: 1mm 1.6mm; vertical-align: top; overflow-wrap: break-word; }
th code { background: transparent; color: white; }
figure { page-break-inside: avoid; margin: 2mm 0; }
figure figcaption { font-size: 7.6pt; color: #444; text-align: center; margin: 0.8mm 0 3mm 0; }
span.nw { white-space: nowrap; }
tr { page-break-inside: avoid; }
table { page-break-inside: avoid; }
table.larga { page-break-inside: auto; }
tbody tr:nth-child(even) td { background: #f3f7fa; }
code { font-family: 'DejaVu Sans Mono', monospace; font-size: 0.86em; background: #eef2f5; padding: 0 0.6mm; overflow-wrap: anywhere; word-break: break-all; }
pre { background: #eef2f5; padding: 2mm; font-size: 7.2pt; white-space: pre-wrap; border-left: 2pt solid #1b5e7a; page-break-inside: avoid; }
pre code { background: none; word-break: normal; }
img { max-width: 100%; max-height: 205mm; display: block; margin: 2mm auto 0 auto; }
img + em { display: block; font-size: 7.6pt; color: #444; text-align: center; margin: 0.8mm 0 3mm 0; }
blockquote { border-left: 2.5pt solid #e0a800; margin: 2mm 0; padding: 0.5mm 3mm; background: #fdf8e6; font-size: 8.3pt; }
ul, ol { margin: 1.2mm 0 1.2mm 4.5mm; padding-left: 3mm; }
li { margin: 0.5mm 0; }
.saltopagina { page-break-after: always; }
.toc { font-size: 9pt; }
.toc h1 { page-break-before: avoid; }
.toc a { color: #1c1c1c; text-decoration: none; }
.toc li { list-style: none; margin: 0.7mm 0; }
.toc li.n1 { margin-top: 2.2mm; font-weight: bold; }
.toc li.n2 { margin-left: 5mm; font-size: 8.3pt; }
.toc a::after { content: leader('.') target-counter(attr(href), page); }
.resumen table { font-size: 7.6pt; }
"""


def main():
    md = markdown.Markdown(extensions=["tables", "fenced_code", "toc", "sane_lists"], extension_configs={"toc": {"permalink": False}})
    cuerpos, toc = [], []
    for i, p in enumerate(sorted(SEC.glob("[0-9][0-9]_*.md"))):
        md.reset()
        html = md.convert(p.read_text(encoding="utf-8"))
        # tabla de contenidos: h1 y h2
        for m in re.finditer(r'<h([12]) id="([^"]+)">(.*?)</h\1>', html):
            nivel, ident, tit = m.group(1), m.group(2), re.sub(r"<[^>]+>", "", m.group(3))
            if i == 0 and nivel == "1":
                tit = "Resumen ejecutivo y veredicto"
            toc.append((nivel, f"{i:02d}-{ident}", tit))
        html = re.sub(r"<table>.*?</table>", lambda m: m.group(0).replace("<table>", '<table class="larga">', 1) if m.group(0).count("<tr>") > 9 else m.group(0), html, flags=re.S)
        html = re.sub(r"<p>(<img [^>]+>)</p>\s*<p><em>(.*?)</em></p>", r"<figure>\1<figcaption>\2</figcaption></figure>", html, flags=re.S)
        for est in ("CUMPLIDO PARCIALMENTE", "NO VERIFICABLE", "NO CUMPLIDO", "CUMPLIDO"):
            html = re.sub(r"<td>(<strong>)?%s(</strong>)?</td>" % est, lambda m, e=est: "<td><span class='nw'>%s</span></td>" % e.replace(" ", "<br>"), html)
        html = re.sub(r"<td>(R\d+)</td>", r"<td><span class='nw'>\1</span></td>", html)
        html = re.sub(r'id="([^"]+)"', lambda m: f'id="{i:02d}-{m.group(1)}"', html)
        clase = "resumen" if i == 0 else "cap"
        cuerpos.append(f'<section class="{clase}">{html}</section>')
    items = "".join(f'<li class="n{n}"><a href="#{ident}">{t}</a></li>' for n, ident, t in toc if not (n == "2" and ident.startswith("00-") ))
    hoy = "07 de octubre de 2026"
    portada = f"""<div class="portada"><h1>Condiciones cumplidas</h1>
<div class="sub">Auditoría técnica y metodológica del proyecto de clasificación automática de expresiones faciales en equinos, contrastada con el plan de trabajo corregido (mayo de 2026), con experimento comparativo de features de pose «tipo SimBA»</div>
<table><tr><td>Proyecto</td><td>Detección Automática de Expresiones Faciales en Equinos mediante Aprendizaje Profundo en Secuencias de Video</td></tr>
<tr><td>Integrantes</td><td>Héctor R. Liceaga · Sergio M. Yañez</td></tr>
<tr><td>Referencia</td><td>docs/plan_de_trabajo_corregido.pdf</td></tr>
<tr><td>Estado auditado</td><td>rama main, commit caf5564 (07/10/2026), artefactos de datos_equinos/ y experimentos/simba/</td></tr>
<tr><td>Fecha</td><td>{hoy}</td></tr></table>
<div class="pie">Todas las cifras provienen de scripts reproducibles en auditoria/scripts/ (solo lectura sobre los datos).</div></div>"""
    indice = f'<div class="toc"><h1>Índice</h1><ul>{items}</ul></div>'
    doc = f'<!doctype html><html lang="es"><head><meta charset="utf-8"><title>Condiciones cumplidas</title><style>{CSS}</style></head><body>{portada}{indice}{"".join(cuerpos)}</body></html>'
    (AUD / "_condiciones_cumplidas.html").write_text(doc, encoding="utf-8")
    r = subprocess.run(["weasyprint", "-q", "--base-url", str(SEC) + "/", str(AUD / "_condiciones_cumplidas.html"), str(SALIDA)], capture_output=True, text=True)
    print(r.stdout[-500:], r.stderr[-1500:])
    print("PDF:", SALIDA, SALIDA.exists())


if __name__ == "__main__":
    main()
