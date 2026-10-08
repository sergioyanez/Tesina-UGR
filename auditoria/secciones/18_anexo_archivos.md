# 18. Anexo: archivos analizados y producidos

## 18.1 Documentos de entrada

- `docs/plan_de_trabajo_corregido.pdf` (leído completo).
- `material para presentar/2da entrega/Devolucion_Juan_resumen_de_condiciones_cumplidas_o_no .pdf` (con las observaciones del docente) y `condiciones_cumplidas_2026-09-25.pdf` (auditoría anterior; comparadas por diferencia de texto).
- `openspec/project.md`, `openspec/specs/`, `openspec/changes/archive/2026-10-07-mejoras-prioridad-alta/`, `openspec/changes/experimento-comparativo-simba/`.
- `docs/registro_de_experimentos.md`, `docs/protocolo_de_etiquetado.md`, `docs/analisis_sobreajuste.md`, `docs/informe_mejora_pipeline_mil.md`.
- `material para presentar/informe/informe_final_tesina_equinos.tex` y `referencias.bib` (borrador; sólo lectura).

## 18.2 Código revisado (rama `main` @ `caf5564`)

`equinos/`: `particion.py`, `clasificacion.py`, `caracteristicas.py`, `secuencia.py`, `alineado.py`, `dataset.py`, `inferencia.py`, `lineas_de_base.py`, `reproceso.py`, `pose.py`, `encuadre.py`, `rutas.py`; `scripts/evaluar_v2.py`, `scripts/evaluar_lineas_de_base.py`; `app_inferencia.py`; `requirements-local.txt`; notebooks `01_preparar_dataset_equinos.ipynb` y `02_roi_y_normalizacion.ipynb` (parámetros y clips); los 15 archivos de `tests/`, ejecutados (15/15 pasan).

Ramas laterales consultadas con `git ls-tree` y `git log`: `feat/enriquecer-canal-geometrico`, `feat/validacion-cruzada-anidada-NestedCV`, `feat/finetuning-visual`.

## 18.3 Experimento SimBA (rama `feat/auditoria-y-experimento-simba`, sólo lectura)

`experimentos/simba/`: `README.md`, `features_simba.py`, `construir_dataset.py`, `evaluar.py`, `shap_rf.py`, `inventario.py`, `parsimonia_mil.py`, `simba_real_intento.py`, `figuras_y_resumen.py`, `test_features_simba.py`, `correr_todo.sh`; `resultados/` (`resumen.json`, `evaluacion.json`, `shap.json`, `inventario_anatomico.json`, `compatibilidad_dlc_simba.json`, `parsimonia_mil.json`, `simba_real_intento.json`, `columnas.json`, `dataset_tabular_simba.csv`, `dataset_tabular_simba_real.csv`, `oof_predicciones.csv`, `shap_top20_global.csv`, `shap_valores_oof.npz`, figuras `comparacion_modelos`, `matrices_confusion`, `gaps_sobreajuste`, `importancias`, `shap_resumen_*`); `verificacion/` (`README.md`, `verificacion.json`, `verificar.py`).

## 18.4 Artefactos de datos leídos (solo lectura, `datos_equinos/`)

`resultados/`: `inventario_videos.csv`, `inventario_clips.csv`, `dataset_frames_alineados_v2.csv`, `dataset_rechazos_v2.json`, `pose_seleccionada_v2.csv`, `pose_todos_los_frames.csv`, `particion_clips.json`, `caracteristicas_backbone_v2.{json,npz}`, `objetivo3_evaluacion_v2.json`, `objetivo3_lineas_de_base.json`, `objetivo3_evaluacion.json`, `objetivo3_{canal,nested,finetuning}_*.json`, `keypoints_manuales.csv`; `modelos/objetivo3_v2/mil_v2_fold{0..3}.pt`, `mil_multimodal_produccion.{pt,json}`; `trabajo/frames_alineados_v2/` (5 744 recortes); `clips/` (clips recodificados); y los videos fuente de `~/videos_de_prueba` (274 archivos).

## 18.5 Scripts de la auditoría (`auditoria/scripts/`)

| Script | Qué produce |
|---|---|
| `comun.py` | Rutas (solo lectura sobre el repositorio principal) y utilidades |
| `a01_dataset.py` | Embudo, clips y caballos por clase, folds, duración, consistencia de nombres, figuras 3.x |
| `a02_duplicados.py` | SHA-256 de videos, firma perceptual de video, dHash/MD5 de recortes, solapamientos |
| `a03_balance_folds_leakage.py` | Balance con criterio declarado, composición de test/train interno/validación, aserciones de fuga, comparación con `GroupKFold` |
| `a04_reentrenar.py` | Re-entrenamiento con el código vigente: reproducción bit a bit y 6 semillas |
| `a05_modelo.py` | Conteo de parámetros, dimensionalidad, invariancia al orden |
| `a06_metricas.py` | Métricas, estabilidad, matrices, baselines, bootstrap, independencia, cifras históricas |
| `a07_early_stopping_overfitting.py` | Tabla de épocas, brechas, test alterado, gap del baseline lineal, curvas |
| `a08_entrenamiento_vs_inferencia.py` | Efecto de recodificar clips en las características y predicciones |
| `a09_orientacion_inferencia.py`, `a09b_resumen.py` | Rotación declarada en el dataset; `clasificar_video` sobre videos del dataset |
| `a10_reproducibilidad.py` | Dependencias, semillas, ramas laterales, huellas |
| `a11_identidades.py` | Cribado de posibles identificadores duplicados |
| `a12_app_streamlit.py` | La interfaz arranca (sin navegador) y cita las métricas vigentes |
| `a13_seleccion_frames.py` | Ventana de 16 frames por clase |
| `a14_repetibilidad.py`, `a15_particiones_alternativas.py` | Variabilidad por semilla y por partición |
| `a17_confusores.py` | Rotación, contenedor, códec y fecha como predictores sin mirar la cara |
| `a18_calibracion.py`, `a19_identidad.py`, `a20_validacion_interna.py` | Calibración, efecto de agrupar por caballo, composición de la validación interna |
| `a21_diagramas.py` | Diagramas del pipeline y de los folds |
| `a16_ensamblar.py` | Genera los capítulos con tablas desde los JSON (incluye las de `experimentos/simba/`) |
| `a22_pdf.py` | Convierte los capítulos en `auditoria/condiciones_cumplidas.pdf` (WeasyPrint, venv `~/.venvs/pdf-auditoria`) |
| `correr_todo.py` | Reproduce todo, en orden |

Los números del texto salen de `auditoria/datos/*.json|csv`; las figuras (PNG, 200 dpi) están en `auditoria/figuras/`.

## 18.6 Limitaciones de esta auditoría

- No se re-ejecutó DeepLabCut sobre los videos (horas de GPU): la pose de las pruebas de §12 es la persistida.
- No se re-extrajeron las características desde las imágenes (se leyó el `.npz` vigente).
- La muestra de la prueba de extremo a extremo de la interfaz es pequeña (indicativa).
- La fecha de sesión se infiere del nombre de archivo; no hay metadato de establecimiento ni de cámara por video.
- No pudo verificarse con los archivos disponibles: identidad real de los caballos, raza, dispositivo, iluminación, composición del entrenamiento de SuperAnimal-Quadruped.
