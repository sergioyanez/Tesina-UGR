# 14. Reproducibilidad y trazabilidad

(Respuesta a la pregunta de la devolución docente: «¿cómo está la repetibilidad?»)

## 14.1 Se reproduce lo publicado

Con el código vigente y la semilla 42 (`scripts/a04_reentrenar.py`, GPU RTX 3070 Laptop, torch 2.5.1+cu121):

| Verificación con el código vigente y semilla 42 | Resultado |
|---|---|
| Predicciones fuera de muestra idénticas a las guardadas | 359/359 |
| Curvas de F1 de validación idénticas (4 folds) | sí |
| Curvas de F1 de train idénticas (4 folds) | sí |
| Pesos idénticos bit a bit (folds 0–3) | sí, sí, sí, sí |
| Tiempo de re-entrenamiento (4 folds, GPU) | 145 s |

La re-ejecución de esta auditoría reprodujo, **bit a bit**, las predicciones, las curvas y los pesos de los 4 folds. (Nota: el código **no** activa `cudnn.deterministic` ni `use_deterministic_algorithms`; la igualdad exacta se observa en esta máquina, sin garantía en otro hardware o versión de CUDA.)

## 14.2 Repetibilidad con otras semillas y otras particiones

| Semilla | F1 macro (media de 4 folds) | DE entre folds | Accuracy | Mejor época por fold (0–3) |
|---|---|---|---|---|
| 42 | 0,740 | 0,052 | 0,744 | 28, 26, 20, 40 |
| 43 | 0,751 | 0,049 | 0,760 | 44, 28, 21, 27 |
| 44 | 0,730 | 0,065 | 0,738 | 18, 23, 16, 22 |
| 45 | 0,726 | 0,068 | 0,733 | 50, 24, 14, 19 |
| 46 | 0,763 | 0,074 | 0,766 | 13, 15, 18, 22 |
| 47 | 0,727 | 0,055 | 0,733 | 24, 30, 21, 21 |
| **Media ± DE entre semillas** | **0,739 ± 0,015** | 0,061 |  | rango 13–50 |

![Figura 14.1. F1 macro del MIL con 6 semillas sobre la misma partición (izquierda), mejor época por semilla y fold (centro) y MIL frente a la media de 16 frames en 4 asignaciones alternativas de caballos a folds (derecha).](../figuras/fig_repetibilidad.png)

*Figura 14.1. F1 macro del MIL con 6 semillas sobre la misma partición (izquierda), mejor época por semilla y fold (centro) y MIL frente a la media de 16 frames en 4 asignaciones alternativas de caballos a folds (derecha).*

- Con la partición vigente, **la semilla mueve el F1 macro en un rango de 0,036** (de 0,726 a 0,763; DE entre semillas 0,015). La semilla 42 es representativa (media de 6 semillas 0,739) y no fue elegida por su valor: la semilla ya figuraba en el código.
- **La mejor época es mucho más variable que el F1**: de 13 a 50 entre semillas y folds. Es el síntoma de una validación interna pequeña y ruidosa (§5.3).
- Con **otras asignaciones de caballos a folds** (4 particiones, semilla 42) el MIL da 0,744–0,758 (§10.2), lo que sitúa el 0,740 de la partición vigente en el extremo bajo del rango posible. La variabilidad debida a la partición y a la semilla (≈ ±0,015) es del mismo orden que la diferencia con el baseline estático, de modo que **no deben interpretarse diferencias de F1 menores a ≈ 0,03 entre variantes**.
- Una repetición del protocolo completo con varias particiones (p. ej., 5 × 4 folds) es la forma correcta de estimar la variabilidad; el proyecto reporta una sola.

## 14.3 Entorno, semillas y configuración

| Elemento | Estado | Evidencia |
|---|---|---|
| Dependencias fijadas | `requirements-local.txt` con versiones exactas (torch 2.5.1+cu121, numpy 1.26.4, scikit-learn 1.7.2, deeplabcut 3.0.1…). Las versiones instaladas coinciden salvo `tensorboard` (fijado pero no instalado y no usado) | `datos/reproducibilidad.json` |
| Semillas | `SEMILLA = 42`; `torch`, `random`, `numpy`, `cuda` y generador del `DataLoader` fijados; el bootstrap usa semilla 1; la partición no usa azar | `clasificacion.py`, `lineas_de_base.py`, `particion.py` |
| Configuración | Hiperparámetros **escritos en los scripts** (`DROPOUT = 0.3`, `PESOS = "raiz"` en `evaluar_v2.py`; `paciencia=10`, `max_epocas=100` como valores por defecto), sin archivo de configuración | `scripts/evaluar_v2.py` |
| Scripts del modelo vigente | En `main`: `scripts/evaluar_v2.py`, `scripts/evaluar_lineas_de_base.py`; pruebas: 15 archivos `tests/test_*.py`, **15/15 pasan** (`test_equivalencia.py` vuelve a pasar tras restituir `CONF_MINIMA_KEYPOINT = 0,50`) | ejecución de esta auditoría |
| Checkpoints | `mil_v2_fold{0..3}.pt` y `mil_multimodal_produccion.pt` con su procedencia; la **huella de las características** registrada (`806a6f478408ab0b`) **coincide** con la de la caché vigente (error de la auditoría anterior, corregido) | `datos/reproducibilidad.json` |
| Pipeline completo desde los videos | Documentado (`docs/informe_mejora_pipeline_mil.md` §8): ≈ 3 h de GPU de pose + minutos para el resto; depende de que `~/videos_de_prueba` exista | **No se re-ejecutó** la pasada de pose (horas de GPU) |

## 14.4 Trazabilidad de experimentos

- **Resuelta en gran parte desde el 25/09**: `docs/registro_de_experimentos.md` indica para cada JSON de `resultados/` su código, rama, commit y comando, y separa el modelo vigente de las variantes no adoptadas. Las tres ramas con los scripts de variantes (`feat/enriquecer-canal-geometrico`, `feat/validacion-cruzada-anidada-NestedCV`, `feat/finetuning-visual`) existen y contienen los scripts mencionados (verificado con `git ls-tree`); **siguen sin fusionarse**, por lo que esos resultados sólo se reproducen cambiando de rama, con código incompatible entre sí.
- **Fragilidad de la caché de características**: `caracteristicas.huella` incluye la fecha de modificación (`st_mtime`) del CSV del dataset. Copiar o tocar ese archivo invalida la caché y obliga a re-extraer y a **escribir** en la raíz de datos; no cambia los resultados (el backbone es determinista) pero rompe el control por huella. Para sus lecturas esta auditoría leyó el `.npz` directamente.
- **Documentación desactualizada** (§12.4): cifras viejas en `openspec/project.md`, `inferencia.py` y `app_inferencia.py`.
- **Cifras históricas**: ver §9.5; sólo `objetivo3_evaluacion_v2.json` es vigente.
