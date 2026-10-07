# Auditoría técnica y metodológica — Tesina «Detección automática de expresiones faciales en equinos»

**Fecha:** 07 de octubre de 2026 · **Estado auditado:** rama `main`, commit `caf5564`, más las fusiones `feat/mejoras-prioridad-alta` y `chore/deuda-menor-y-observador-unico` · **Referencia:** `docs/plan_de_trabajo_corregido.pdf` · **Antecedente:** auditoría del 25/09/2026 y devolución docente.

**Naturaleza.** Auditoría crítica. Cada cifra sale de un script de `auditoria/scripts/` que lee los artefactos del repositorio (sólo lectura) y escribe en `auditoria/datos/`; lo que no pudo comprobarse se declara como tal. Nada se copió de la auditoría anterior sin recalcularlo.

## Resumen ejecutivo

| Aspecto | Resultado de la auditoría |
|---|---|
| **Modelo y dato vigente** | MIL multimodal (`MILFusionModel`: ResNet-18 congelada + 9 variables geométricas + atención), 132 355 parámetros entrenables. 359 clips · 133 caballos · 244 videos de origen (393 clips candidatos, 91,3 % válidos). Clases: `Disgustado` 103, `Interesado` 118, `Sereno` 138 |
| **Rendimiento (evaluación vigente: `objetivo3_evaluacion_v2.json`)** | **F1 macro 0,740 ± 0,052** (media ± DE de 4 folds por caballo; IC 95 % bootstrap por caballo [0,677; 0,797]); accuracy 0,744 (267/359); balanced accuracy 0,740 |
| **Contra baselines (mismos folds)** | Aleatorio 0,327 · mayoritaria 0,183 · metadatos de filmación 0,618 · un frame 0,679 · **media de 16 frames + regresión logística 0,741 ± 0,020** (diferencia con el MIL −0,005, IC 95 % [−0,068; +0,058]) |
| **Cumplimiento del plan** | De 18 requisitos (se retira el antiguo R8 por indicación docente): **10 cumplidos, 6 parciales, 2 no verificables, 0 no cumplidos** (el 25/09: 9 / 4 / 4 / 2 de 19) |
| **Validación** | 4 folds agrupados por caballo; corte interno por caballo; *early stopping* sobre F1 macro de validación (paciencia 10); test evaluado una sola vez. **Verificado con un experimento**: re-entrenar con las etiquetas del test permutadas deja épocas, curvas y pesos idénticos |
| **Fuga de información** | **No se observan evidencias de fuga** por caballo, video, clip ni frame (0 cruces; 12 aserciones). Riesgo residual: sesión y lugar (el grupo es el caballo, no la sesión), selección de variantes no anidada, un posible caballo con dos identificadores (`123`/`169`, no verificable) |
| **Balance** | RAZONABLEMENTE BALANCEADO en clips (1,34 : 1) y BALANCEADO en caballos (1,15 : 1), con criterio declarado antes de calcular; 5 de los 12 subconjuntos internos de los folds son moderada o fuertemente desbalanceados (validación interna del fold 0: 65,7 % `Sereno`) |
| **Sobreajuste** | **MODERADO**: F1 train 0,979 / validación 0,712 / test 0,740 (gap train−test 0,24); la parada temprana mantiene el test igual o mayor que la validación en 4/4 folds, pero las probabilidades están sobreconfiadas (confianza 0,824 frente a acierto 0,744). Un modelo lineal memoriza igual (train 1,00) |
| **Repetibilidad** (pregunta del docente) | Semilla 42: 359/359 predicciones, curvas y pesos idénticos bit a bit. Otras 5 semillas: F1 0,726–0,763 (media 0,739 ± 0,015); la mejor época varía de 13 a 50. Otras 4 particiones de caballos: MIL 0,744–0,758 |
| **¿Es espacio-temporal?** (pregunta del docente) | **No**: invariante al orden (0 de 359 predicciones cambian al barajar o invertir los frames). Es una agregación de varios fotogramas; el requisito R7 queda parcial |
| **¿DeepLabCut con Horse-30?** (pregunta del docente) | **No**: modelo generalista `superanimal_quadruped` (`hrnet_w32` + `fasterrcnn_resnet50_fpn_v2`) sin ajuste con caballos |
| **Resultado SimBA (experimento adicional)** | {{SIMBA_RESULTADO_TITULAR}} |

## Hallazgos principales

1. **La metodología de evaluación es sólida y verificable**; el 0,74 es una estimación creíble del rendimiento sobre **caballos nuevos de esta población y este tipo de captura**, con ±0,06 de IC y ±0,015 entre semillas.
2. **La arquitectura no demuestra valor agregado**: la media lineal de las características de 16 frames rinde igual (0,741), con 2,6 veces menos dispersión entre folds. En 4 particiones alternativas el MIL es 0,003–0,021 mejor (media +0,015); no alcanza para afirmar una ventaja.
3. **El contexto de captura contiene señal**: sin mirar la cara, la duración sola da 0,544, los metadatos 0,618 y la fecha de sesión 0,535 (0,605 combinando técnicos y fecha). La duración del clip determina la ventana de los 16 frames (0,19 s entre frames en `Disgustado`, 0,39 s en las otras). Sobre el subconjunto 1080×1920 y ≥ 5 s el MIL conserva 0,705.
4. **Hallazgos nuevos respecto de la auditoría anterior** (no estaban en ella): (a) la **validación interna son los 7 caballos con más clips** (≈ 10 clips c/u frente a 2,1 en el train interno); (b) los **clips se recodifican en `mp4v`** y la interfaz decodifica el video original: en la muestra medida cambia la predicción en 2 de 28 clips, y al correr la interfaz sobre 16 videos del propio dataset (con el modelo ya entrenado sobre ellos) la predicción coincidió con la etiqueta sólo en 8; (c) **la inferencia no activa la orientación del video** que sí se aplicó al crear el dataset: 69 de 274 videos (25 %) declaran rotación y originan el 35 % de los clips, y esa rotación está asociada a la clase (1/110 de `Disgustado`, 47/93 de `Sereno`); (d) la partición **no coincide con `sklearn.GroupKFold`** aunque el docstring dice «réplica»; (e) la **calibración**: ECE 0,081.
5. **Las etiquetas** tienen ahora criterio documentado (EquiFACS, etogramas) pero **un único observador no ciego**; el acuerdo sobre el dataset no se midió y el techo de rendimiento se desconoce.
6. **Debilidades abiertas**: validación externa, utilidad para veterinarios (R15, R16), ramas de variantes sin fusionar, documentación con cifras viejas.

## Cómo se atendió la devolución docente

| Observación | Atención |
|---|---|
| «¿Cómo está la repetibilidad?» | Medida: bit a bit con la semilla 42; ±0,015 entre semillas; 0,740–0,758 entre 5 particiones de caballos (DE 0,008; cap. 14) |
| «¿Espacio temporal o sólo procesamiento del video?» | Confirmado: no es temporal; R7 parcial (cap. 2, 11) |
| «No lo estamos haciendo, sacar» | Antiguo R8 retirado de la matriz (cap. 1) |
| «¿DeepLabCut con Horse-30?» | No; modelo generalista, sin ajuste (cap. 12) |

## Veredicto

El proyecto **cumple su objetivo central** (un sistema que clasifica expresiones de caballos en video con evaluación honesta por individuo) y su metodología es de **confiabilidad ALTA**; la **confiabilidad estadística es MEDIA**, la **generalización externa no está evaluada**, y **la hipótesis queda parcialmente sustentada**: factibilidad de clasificar, sí; valor del aprendizaje profundo y de la modelización temporal, no demostrado; utilidad para expertos, no verificable. Las mejoras de prioridad alta (cap. 16) son: reformular la contribución, corregir el eslabón entrenamiento–inferencia, controlar la sesión como confusor, repetir la validación cruzada, rehacer la validación interna y llevar las limitaciones de etiquetado al informe.

{{SIMBA_CONCLUSION_RESUMEN}}
