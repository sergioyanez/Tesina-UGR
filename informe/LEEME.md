# Informe final: cifras y figuras generadas

Las cifras del texto y las figuras salen de los JSON de `comparacion/resultados/`; no se escriben a mano.

## Regenerar

    informe/generar.sh

Usa el venv `~/.venvs/tesina-equinos` (otro intérprete: `PYTHON=... informe/generar.sh`) y `TESINA_EQUINOS_RAIZ=<repo>/datos_equinos` (ya definido por defecto). Produce:

- `cifras.tex`: una macro `\newcommand` por cifra (formato español, coma decimal). Lista legible con valor y fuente: `cifras_lista.md`.
- `figuras/*.pdf` (vectorial) y `figuras/*.png` (300 dpi). Títulos y notas sugeridos: `figuras/LEEME.md`.

El script de cifras verifica (assert) que los valores coinciden con los que cita `comparacion/DECISION.md`; si los JSON y la decisión divergen, falla. Si cambian los resultados, regenerar y volver a subir.

## Qué subir a Overleaf

`informe_final_tesina_equinos.tex`, `referencias.bib`, `cifras.tex` y `figuras/*.pdf`. El `.tex` debe incluir `\input{cifras}` en el preámbulo (ya está). Compilar con pdfLaTeX + Biber.

## Uso de las macros

Escribir `\FusionDevF{}` (con llaves vacías, para no perder el espacio siguiente). Convenciones: `Fusion`, `RFPose`, `MIL`, `Base` = candidato; `Dev` = desarrollo 5×4; `Limpio` = subconjunto limpio; `Agrup` = F1 agrupado; `Sesion` = una sesión afuera; `Reserva`; sufijos `DE` (desvío), `IC` (IC95), `Acc` (porcentaje), `AccDec` (decimal); clases `Disg`, `Int`, `Ser`; `Dif<A><B>` = A − B. Los signos menos son `\ensuremath{-}`.
