## Why

El MIL vigente combina una ResNet-18 congelada con 9 features geométricas. No se sabía cuánta señal
discriminativa hay SOLO en la geometría y el movimiento de los landmarks de DeepLabCut. Contestarlo da
una línea de base interpretable y liviana, y ubica al MIL frente a una alternativa sin red visual.

## What Changes

- Experimento ADICIONAL y aislado en `experimentos/simba/`: features «tipo SimBA» con código propio (el extractor
  real de SimBA para un animal solo produce movimiento), dataset tabular por clip, Random Forest, ablación,
  subconjunto limpio, SHAP out-of-fold, bootstrap por caballo y parsimonia.
- No se modifica `equinos/` ni el MIL vigente. No se re-corre DeepLabCut.
- Resultados persistidos en `experimentos/simba/resultados/` con `resumen.json` (12 preguntas).

## Capabilities

### New Capabilities
- `experimento-comparativo-geometrico`: requisitos de honestidad de nombre, mismos folds, denominadores, controles de confusores e incertidumbre del experimento.

### Modified Capabilities
(ninguna)

## Impact

Solo archivos nuevos bajo `experimentos/simba/` y este change. Dependencias nuevas solo en entornos aparte o sin
tocar numpy (`shap`, `slicer`, `cloudpickle` con `--no-deps`; `simba-uw-tf-dev` en `~/.venvs/simba`).
No hay spec delta sobre capacidades existentes; el delta agregado es de una capacidad nueva.
