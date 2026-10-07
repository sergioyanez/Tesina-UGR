# 1. Plan de trabajo: requisitos y matriz de cumplimiento

Documento de referencia: `docs/plan_de_trabajo_corregido.pdf` (mayo de 2026), leído completo. Estado auditado: rama `main`, commit `caf5564` (07/10/2026), incluidas las fusiones `feat/mejoras-prioridad-alta` y `chore/deuda-menor-y-observador-unico`, y los artefactos de `datos_equinos/resultados/`. Todo lo que sigue se **re-verificó contra el estado actual**; no se copió ningún número de la auditoría del 25/09.

## 1.1 Qué dice el plan

| Elemento | Contenido (fiel al plan) |
|---|---|
| Problema | La evaluación del estado emocional depende de la observación humana (subjetiva, lenta); las imágenes estáticas «pierden información crucial» porque las expresiones son «eventos dinámicos» |
| Objetivo general | «Construir un sistema informático para la detección automatizada de expresiones faciales en caballos mediante el procesamiento de secuencias espacio-temporales» |
| Objetivos específicos | (1) Base de datos de videos cortos (5–7 s) de rostros de equinos. (2) Pipeline de preprocesamiento: extracción de fotogramas clave y detección de la ROI. (3) Técnicas de clasificación de aprendizaje automático orientadas a video. (4) Aplicación o interfaz para usar el modelo |
| Hipótesis | Es factible desarrollar un modelo de aprendizaje profundo que clasifique expresiones faciales asociadas a estados emocionales a partir de secuencias de video cortas, «logrando un nivel de precisión útil para la asistencia de expertos humanos» |
| Metodología | (1) Investigación tecnológica (estado del arte espacio-temporal). (2) Preprocesamiento: limpieza, extracción y procesamiento de fotogramas, rostros alineados a lo largo de los clips. (3) Entrenamiento y selección: pruebas de concepto con modelos candidatos, selección por métricas. (4) Validación final con validación cruzada y análisis de resultados para evitar el sobreajuste |
| Datos y técnicas | Bases de video propias e institucionales; Colab, Python, PyTorch, TensorFlow, OpenCV, DeepLabCut |
| Evaluación | «Métricas de rendimiento» y validación cruzada; **el plan no fija una métrica ni un umbral numérico** |
| Resultados esperados / impacto | Herramienta no invasiva y objetiva que asista a veterinarios y cuidadores; metodología extrapolable a otras especies |
| Cronograma e índice | Entrega final proyectada para septiembre; índice: Introducción, Etología, Técnicas, Desarrollo experimental, Resultados y discusión, Conclusiones, Bibliografía, Anexos |

## 1.2 Requisitos verificables derivados (R1–R18)

Se conserva la lista de la auditoría del 25/09 **menos el antiguo R8** («la evolución temporal aporta robustez»), que se **retira** por indicación de la devolución docente (§1.5, punto 3). La numeración de esta tabla es consecutiva; el cruce con la numeración anterior está en §1.4.

| # | Requisito | Origen en el plan |
|---|---|---|
| R1 | Base de datos de videos cortos (5–7 s) de rostros de equinos | OE1, Resumen |
| R2 | Pipeline de preprocesamiento con extracción de fotogramas y detección de ROI | OE2 |
| R3 | Rostros alineados a lo largo de los clips | Metodología 2 |
| R4 | Clasificación con técnicas de aprendizaje automático orientadas a video | OE3 |
| R5 | Aplicación o interfaz para usar el modelo | OE4 |
| R6 | Sistema informático de detección automatizada de punta a punta | Objetivo general |
| R7 | Procesamiento de secuencias espacio-temporales / análisis de la dinámica | Objetivo general, Resumen |
| R8 | Analizar diversas arquitecturas candidatas (pruebas de concepto) | Resumen, Metodología 3 |
| R9 | Identificar la arquitectura óptima para características espacio-temporales | Resumen |
| R10 | Seleccionar la arquitectura en base a métricas de rendimiento | Metodología 3 |
| R11 | Validación final con validación cruzada | Metodología 4 |
| R12 | Análisis de resultados para evitar el sobreajuste | Metodología 4 |
| R13 | Relevamiento del estado del arte | Metodología 1 |
| R14 | Nutrirse de EquiFACS / metodologías etológicas (criterio de etiquetado) | Antecedentes |
| R15 | Precisión «útil para la asistencia de expertos» | Hipótesis |
| R16 | Herramienta objetiva que asista a veterinarios y cuidadores | Resumen, Impacto |
| R17 | Uso de los recursos previstos (PyTorch, OpenCV, DeepLabCut, Colab, Python) | Recursos |
| R18 | Informe final según el índice propuesto, con entrega proyectada para septiembre | Cronograma, Índice |

## 1.3 Matriz de cumplimiento

<!--MATRIZ-->
| # | Requisito | Evidencia (archivo concreto) | Estado | Justificación |
|---|---|---|---|---|
| R1 | Base de datos de videos cortos (5–7 s) | `resultados/inventario_videos.csv` (274 videos), `inventario_clips.csv` (393 clips), `particion_clips.json` (359 válidos, 133 caballos) | CUMPLIDO PARCIALMENTE | La base existe, es propia y se amplió. Pero el 33,4 % de los clips válidos (120/359) dura menos de 5 s: `Disgustado` tiene mediana 3,2 s (§3.4). El piso de 1 s para videos cortos es una decisión del proyecto, no del plan |
| R2 | Pipeline de preprocesamiento (fotogramas y ROI) | `equinos/pose.py`, `secuencia.py`, `alineado.py`, `encuadre.py`; notebooks 01–02; `resultados/dataset_rechazos_v2.json` (91,3 % de clips aprovechados) | CUMPLIDO | DeepLabCut sobre todos los frames, ventana temporal con anclaje, ROI alineada por el eje ojo–hocico con orejas; rechazos clasificados por motivo; 15/15 archivos de prueba pasan |
| R3 | Rostros alineados a lo largo de los clips | `resultados/dataset_frames_alineados_v2.csv` (5 744/5 744 frames `alineado = True`, `origen = keypoints`); `equinos/alineado.py`; `tests/test_equivalencia.py` | CUMPLIDO | Transformada de similitud por frame a una pose canónica; los 359 clips tienen los 16 frames medidos por el detector (ninguno interpolado) |
| R4 | Clasificación orientada a video | `equinos/clasificacion.py::MILFusionModel`; `resultados/objetivo3_evaluacion_v2.json` | CUMPLIDO | La unidad de decisión es el clip (16 frames) con atención sobre los frames. Es orientada a video en su unidad, no en el orden temporal (R7) |
| R5 | Aplicación o interfaz | `app_inferencia.py`; `equinos/inferencia.py`; `datos/app_streamlit.json`; `datos/orientacion_inferencia.json` | CUMPLIDO | La interfaz arranca sin excepciones, lee sus métricas del JSON vigente y la cadena completa produjo 16 predicciones y 0 negativas explicadas sobre 16 videos del dataset. Debilidades de robustez en §12.3 y §12.5 |
| R6 | Sistema de punta a punta | `equinos/inferencia.py::clasificar_video`; `datos_equinos/modelos/objetivo3_v2/mil_multimodal_produccion.pt` y su `.json` de procedencia | CUMPLIDO | Existe la cadena video → pose → ventana → alineado → ResNet-18 → MIL → clase y confiabilidad medida; la huella de las características del modelo de producción coincide con la caché vigente |
| R7 | Procesamiento espacio-temporal | `clasificacion.MILFusionModel.forward`; `datos/arquitectura.json` (prueba de permutación) | CUMPLIDO PARCIALMENTE | Procesa secuencias de 16 frames, pero es invariante al orden: barajar o invertir los frames no cambió ninguna de las 359 predicciones. Es espacial + agregación de conjunto, no temporal |
| R8 | Analizar diversas arquitecturas candidatas | `docs/registro_de_experimentos.md`; `resultados/objetivo3_*.json` (12 evaluaciones); `openspec/changes/archive/2026-08-28-clasificacion-orientada-a-video/` | CUMPLIDO | Al menos 10 variantes bajo partición por caballo (un frame con voto, promedio, atención, GRU, MIL, MIL visual, canal enriquecido, *fine-tuning*, reentrenado, CV anidada) con resultados y conclusión de cada una |
| R9 | Arquitectura óptima para características espacio-temporales | `resultados/objetivo3_lineas_de_base.json`; `datos/metricas.json`; `datos/particiones_alternativas.json` | CUMPLIDO PARCIALMENTE | Las diferencias entre las mejores variantes (0,740–0,755) caen dentro del desvío entre folds y de las semillas (±0,015); la media lineal de 16 frames iguala al MIL (0,741; dif. −0,005, IC [−0,068; +0,058]). No se puede afirmar que el MIL sea «óptimo»; tampoco que una arquitectura espacio-temporal lo sea. Un RF sobre features de pose rinde 0,841 en el original (0,750 frente a 0,705 en el limpio, no concluyente): ver cap. 17 |
| R10 | Selección por métricas | `docs/informe_mejora_pipeline_mil.md` §2.6; `equinos/clasificacion.py` | CUMPLIDO | La regla de decisión está declarada y usa la validación interna, no el test. Salvedad: la selección de variantes se hizo sobre los mismos folds (§6.4) |
| R11 | Validación cruzada final | `equinos/particion.py`; `resultados/particion_clips.json`; `objetivo3_evaluacion_v2.json` | CUMPLIDO | 4 folds agrupados por caballo con predicciones fuera de muestra para los 359 clips; 0 cruces verificados por código |
| R12 | Análisis para evitar el sobreajuste | `docs/analisis_sobreajuste.md`; `objetivo3_evaluacion_v2.json` (`curva`, `sobreajuste`); `datos/early_stopping_sobreajuste.json` | CUMPLIDO | Hay controles (*early stopping*, dropout, *weight decay*, backbone congelado) y análisis explícito del gap con las curvas de train y validación, desde el 25/09 (antes PARCIAL). El análisis encuentra sobreajuste MODERADO y lo documenta; no se afirma su ausencia |
| R13 | Relevamiento del estado del arte | `../material para presentar/informe/informe_final_tesina_equinos.tex` y `referencias.bib` (39 referencias); `openspec/project.md` | CUMPLIDO PARCIALMENTE | El borrador del informe (fuera del repositorio versionado) contiene un marco con EquiFACS, MIL, DeepLabCut, transferencia y reconocimiento de dolor equino. No hay un relevamiento formal comparativo de modelos espacio-temporales en el repositorio |
| R14 | Criterio etológico (EquiFACS) | `docs/protocolo_de_etiquetado.md` §3–§5; `docs/formulario_etiquetado_2026-09-28.docx` | CUMPLIDO PARCIALMENTE | Antes NO CUMPLIDO. Ahora hay autoría (veterinaria codificadora certificada de EquiFACS), criterio por clase con unidades de acción y etogramas, y exclusiones. Resta: **un único observador no ciego**, sin acuerdo medido sobre el dataset (kappa 0,99 sólo en un piloto con imágenes), sin codificación EquiFACS por video |
| R15 | Precisión «útil para expertos» | `clasificacion.CRITERIO_ACEPTACION` (0,691); `objetivo3_evaluacion_v2.json` | NO VERIFICABLE | El plan no define «útil». El proyecto declaró un criterio interno (0,691 antes de entrenar; 0,761 re-declarado después, y así consta); ninguno es un umbral clínico. No hubo validación con veterinarios |
| R16 | Herramienta objetiva que asista a veterinarios y cuidadores | `app_inferencia.py`; `openspec/changes/archive/*/prueba_con_persona_ajena.md` | NO VERIFICABLE | La herramienta existe y declara que no es un diagnóstico; su utilidad para el público objetivo no se midió (n = 1 en la prueba de usabilidad del proyecto) |
| R17 | Recursos previstos | `requirements-local.txt` (torch 2.5.1, opencv 4.11, deeplabcut 3.0.1); notebooks con celdas de Colab | CUMPLIDO | Se usan PyTorch, OpenCV, DeepLabCut, Python y Colab/GPU local. TensorFlow figura en el plan como alternativa y no se usó |
| R18 | Informe final según el índice, entrega en septiembre | `../material para presentar/informe/informe_final_tesina_equinos.tex` y `.pdf` (borrador del 07/10); `openspec/project.md` («entrega: 29/10/2026») | CUMPLIDO PARCIALMENTE | Hay un borrador con la estructura del índice (Introducción, Etología, Técnicas, Desarrollo, Resultados, Conclusiones, Bibliografía, Anexos) y cifras coherentes con esta auditoría; la entrega se reprogramó del plan (septiembre) al 29/10/2026 |
<!--/MATRIZ-->

## 1.4 Conteo y cambios respecto de la auditoría del 25/09

| Estado | Requisitos |
|---|---|
| CUMPLIDO | 10 |
| CUMPLIDO PARCIALMENTE | 6 |
| NO VERIFICABLE | 2 |
| NO CUMPLIDO | 0 |
| **Total** | 18 |

| Auditoría | Requisitos | Cumplidos | Parciales | No verificables | No cumplidos |
|---|---:|---:|---:|---:|---:|
| 25/09/2026 (estado anterior) | 19 | 9 | 4 | 4 | 2 |
| **Esta auditoría (07/10/2026)** | **18** | **10** | **6** | **2** | **0** |

Cambios, uno por uno:

| Requisito (nuevo ← antiguo) | Antes | Ahora | Motivo del cambio |
|---|---|---|---|
| (retirado) ← R8 «la evolución temporal aporta robustez» | NO CUMPLIDO | — | Retirado a pedido del docente: no es un requisito del plan sino una motivación del problema |
| R12 ← R13 sobreajuste | PARCIAL | CUMPLIDO | Se registra el gap y las curvas por época; hay análisis específico |
| R14 ← R15 EquiFACS | NO CUMPLIDO | PARCIAL | Protocolo de etiquetado y formulario (28/09) |
| R13 ← R14 estado del arte | NO VERIFICABLE | PARCIAL | El borrador del informe y su bibliografía son verificables aunque estén fuera del repositorio |
| R18 ← R19 informe final | NO VERIFICABLE | PARCIAL | Borrador con el índice del plan; entrega reprogramada |
| R8 ← R9, R9 ← R10, R10 ← R11, R11 ← R12 | iguales | iguales | Re-verificados |
| R1, R2, R3, R4, R5, R6, R7, R15, R16, R17 | iguales | iguales | Re-verificados (R3: ya no hay comprobación fallida en `test_equivalencia.py`) |

**Lectura:** el estado mejoró por trabajo real (sobreajuste, etiquetado, líneas de base, trazabilidad) y por la reformulación docente de un requisito (R8). Los dos requisitos que siguen sin poder verificarse (R15, R16) dependen de una validación con expertos que el proyecto no tiene prevista.

**Efecto del experimento SimBA (capítulo 17).** Se reconsideró cada requisito: **R9** se mantiene CUMPLIDO PARCIALMENTE (el experimento agrega evidencia de que el MIL no es la mejor opción disponible —0,841 frente a 0,740 en el original, no concluyente en el limpio—, lo que refuerza que no hay arquitectura demostrada óptima); R7 no cambia (las features resumen la dinámica, no modelan el tiempo); R14, R15 y R16 no cambian (SimBA no aporta unidades de acción, acuerdo interobservador ni validación veterinaria). El plan no tiene un requisito de interpretabilidad. **Los conteos permanecen en 10 / 6 / 2 / 0.**

## 1.5 Atención a la devolución docente

Las observaciones del docente sobre la auditoría del 25/09 son cuatro, intercaladas en el PDF `Devolucion_Juan_resumen_de_condiciones_cumplidas_o_no .pdf`. Se identificaron por diferencia de texto contra el PDF original y por su posición en la página. Cómo se atiende cada una:

| # | Observación del docente (textual) | Dónde se apuntaba | Cómo se atiende en esta auditoría |
|---|---|---|---|
| 1 | «¿Cómo está la repetibilidad?» | Resumen ejecutivo, fortalezas («reproducible bit a bit») | Se **midió** en lugar de afirmarse: (a) misma partición y semilla: 359/359 predicciones, curvas y pesos idénticos; (b) 6 semillas: F1 macro 0,739 ± 0,015 (rango 0,726–0,763), con la mejor época entre 13 y 50; (c) 4 asignaciones alternativas de caballos a folds: MIL 0,744–0,758. Capítulo 14, tablas 14.1–14.2 |
| 2 | «Me queda la duda si es espacio temporal o no porque es invariante al orden que se procesan los frames. Tal vez sea solo procesamiento del video» | R7 | **Tiene razón.** Se verificó numéricamente (0/359 predicciones cambian al barajar o invertir los frames; diferencia máxima de logits 3,8·10⁻⁶). El sistema procesa video (16 frames de un clip) pero **no modela la dinámica temporal**. R7 queda CUMPLIDO PARCIALMENTE y el texto del informe debe decir «agregación espacial de varios fotogramas», no «espacio-temporal». §2.1, §11.1 |
| 3 | «No lo estamos haciendo, sacar» | Antiguo R8 («la evolución temporal aporta robustez») | **Se retira** de la matriz y del conteo (18 requisitos en lugar de 19). La observación empírica se conserva en §2.4 como hallazgo, no como requisito |
| 4 | «¿Usaron el DeepLabCut con Horse-30? subconjunto experto específico para caballos» | Fila «Detección / pose» de la tabla entrenamiento–inferencia | **No.** Se usa el modelo preentrenado generalista `superanimal_quadruped` (red `hrnet_w32`, detector `fasterrcnn_resnet50_fpn_v2`) sin ajuste con datos de caballos; no hay referencias a Horse-10/Horse-30 en el repositorio. Cuánto de ese modelo se entrenó con caballos no pudo verificarse con los archivos disponibles. Se corrigió además la fila de la tabla: ya no dice «Detector/pose: SÍ igual» sin matices (§12.1, §12.3) |

Además de las cuatro observaciones puntuales, el PDF del docente es la **primera versión** del informe del 25/09 (antes de la corrección de misma fecha sobre las ramas laterales y del intervalo MIL − metadatos). Esta auditoría usa los valores corregidos: diferencia MIL − metadatos **+0,121, IC 95 % [+0,031; +0,211]**, calculada contra la regresión logística de metadatos (0,618) y no contra el Random Forest, que era la comparación del texto que leyó el docente ([+0,098; +0,276]).
