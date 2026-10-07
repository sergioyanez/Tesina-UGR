## Why

La auditoría del 25/09/2026 (`material para presentar/2da entrega/condiciones_cumplidas.pdf`)
encontró que el modelo vigente —MIL multimodal, F1 macro 0,740 ± 0,052 sobre 359 clips de 133
caballos— se reporta sin las líneas de base que la spec `clasificacion-de-expresiones` exige
tener al lado, y que al medirlas sobre los mismos folds una regresión logística sobre el
**promedio** de las características congeladas de los 16 frames da 0,741 ± 0,020. La misma
auditoría midió que los metadatos de filmación (duración, fps, resolución) predicen la clase con
F1 0,618 sin mirar la cara, que el modelo sobreajusta el train interno (F1 ≈ 0,98) sin que el
proyecto lo registre, que el metadato del modelo desplegado apunta a las características de otro
conjunto, y que la figura 4 y el notebook 03 describen modelos que ya no son el vigente. Todo eso
se corrige sin datos nuevos, y hay que corregirlo antes de defender el objetivo 3 ante el
tribunal.

Contribuye al **objetivo específico 3** (clasificación orientada a video) y, por la
trazabilidad del modelo que sirve la interfaz, al **objetivo específico 4**.

## What Changes

- Nuevo módulo `equinos/lineas_de_base.py`: líneas de base sobre los mismos folds que el modelo
  —clase mayoritaria, aleatorio estratificado, visual estático (1 frame y media de 16), geometría
  sola, metadatos de filmación y duración sola— y bootstrap por caballo para intervalos de
  confianza y diferencias entre modelos.
- Nuevo `scripts/evaluar_lineas_de_base.py`: persiste `objetivo3_lineas_de_base.json` con las
  líneas de base, el MIL vigente al lado, los intervalos y el control de confusores (rendimiento
  del MIL y del baseline estático restringido a clips 1080×1920 de al menos 5 s).
- `entrenar_mil_multimodal` registra también, por época, pérdida, F1 y accuracy del train
  interno evaluado sin dropout. No cambia ninguna decisión ni ninguna predicción: sólo agrega
  columnas a la curva. `objetivo3_evaluacion_v2.json` se regenera y pasa a incluir el gap.
- `clasificacion._huella_caracteristicas` lee la huella de las características vigentes (`_v2`)
  y no la del conjunto de 153 clips. Se regenera `mil_multimodal_produccion.json`.
- `04_figuras_tesis.ipynb` lee la evaluación vigente para la figura 4 y agrega la figura de
  curvas de entrenamiento (figura 5).
- `03_clasificacion.ipynb` se reescribe para el protocolo vigente y se re-ejecuta: su texto y
  sus salidas pasan a describir el mismo modelo.
- Documentación nueva: `docs/registro_de_experimentos.md` (qué JSON sale de qué rama, commit y
  script), `docs/protocolo_de_etiquetado.md` (lo que se sabe del etiquetado y lo que falta
  declarar) y `docs/analisis_sobreajuste.md`.

## Capabilities

### New Capabilities

Ninguna.

### Modified Capabilities

- `clasificacion-de-expresiones`: la exigencia de líneas de base pasa a incluir una línea de base
  visual estática y una de metadatos de filmación, sobre el conjunto vigente; se agregan
  requisitos de registro del sobreajuste y de trazabilidad del modelo desplegado.

## Impact

- Código: `equinos/clasificacion.py`, `equinos/lineas_de_base.py` (nuevo),
  `scripts/evaluar_lineas_de_base.py` (nuevo), `tests/test_lineas_de_base.py` (nuevo),
  `tests/test_clasificacion.py`.
- Notebooks: `03_clasificacion.ipynb`, `04_figuras_tesis.ipynb`.
- Artefactos regenerados: `objetivo3_evaluacion_v2.json` (mismas predicciones, más columnas),
  `mil_multimodal_produccion.{pt,json}` (mismos pesos si el entrenamiento es determinista; se
  verifica), `objetivo3_lineas_de_base.json` (nuevo), `figuras/figura4_*`, `figuras/figura5_*`.
- Dependencias: `scikit-learn`, ya pinneado en `requirements-local.txt`; se importa de forma
  perezosa sólo en `lineas_de_base`.
- La interfaz no cambia de comportamiento: sigue leyendo `objetivo3_evaluacion_v2.json`.
