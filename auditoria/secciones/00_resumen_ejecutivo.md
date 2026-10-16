# Auditoría técnica y metodológica — Tesina «Detección automática de expresiones faciales en equinos»

**Fecha:** 07 de octubre de 2026 · **Estado auditado:** rama `main`, commit `cd66fe9`, más las fusiones `feat/mejoras-prioridad-alta` y `chore/deuda-menor-y-observador-unico` · **Referencia:** `docs/plan_de_trabajo_corregido.pdf` · **Antecedente:** auditoría del 25/09/2026 y devolución docente.

**Naturaleza.** Auditoría crítica. Cada cifra sale de un script de `auditoria/scripts/` que lee los artefactos del repositorio (sólo lectura) y escribe en `auditoria/datos/`; lo que no pudo comprobarse se declara como tal. Nada se copió de la auditoría anterior sin recalcularlo.

## Resumen ejecutivo

| Ítem | Resultado |
|---|---|
| **Modelo vigente** | MIL multimodal (`MILFusionModel`): ResNet-18 congelada + 9 variables geométricas + atención; 132 355 parámetros entrenables |
| **Datos** | **133 caballos · 359 clips** · 244 videos de origen (393 clips candidatos, 91,3 % válidos) |
| **Distribución por clase** | `Disgustado` 103 (28,7 %) · `Interesado` 118 (32,9 %) · `Sereno` 138 (38,4 %) |
| **F1 macro (vigente)** | **0,740 ± 0,052** (media ± DE de 4 folds por caballo; IC 95 % [0,677; 0,797]); folds 0,718 / 0,769 / 0,677 / 0,795 |
| **Accuracy** | 0,744 (267/359) · balanced accuracy 0,740 |
| **Baselines (mismos folds)** | azar 0,327 · mayoritaria 0,183 · metadatos 0,618 · **media de 16 frames + LogReg 0,741 ± 0,020** (MIL − estática −0,005, IC [−0,068; +0,058]) |
| **Plan de trabajo** | 18 requisitos: **10 cumplidos · 6 parciales · 2 no verificables · 0 no cumplidos** (25/09: 9 / 4 / 4 / 2 de 19) |
| **Sobreajuste** | **MODERADO**: F1 train 0,979 / val 0,712 / test 0,740; el test no está inflado, las probabilidades sí están sobreconfiadas |
| **Balance** | RAZONABLEMENTE BALANCEADO (clips 1,34:1); BALANCEADO (caballos 1,15:1); folds y validaciones internas más desbalanceados |
| **Fuga de información** | No se observan evidencias (0 cruces de caballo, video, clip ni frame; test verificado como independiente); riesgo residual de sesión/lugar |
| **Repetibilidad** | Semilla 42 bit a bit; 6 semillas 0,739 ± 0,015 |
| **Experimento SimBA** | RF sobre 160 «features tipo SimBA» (código propio, no SimBA real): **F1 macro 0,841 ± 0,003** (5 semillas) frente al MIL 0,740 (+0,11 [0,04; 0,18]); en el subconjunto limpio 0,750 frente a 0,705 (no concluyente); deconfundido por filmación 0,777; alerta de confusor (R² 0,53 con la duración). SimBA real sólo da movimiento (0,720). Cap. 17 |
| **Fortalezas** | (1) Separación por caballo verificada por código; (2) test intacto y reproducibilidad exacta; (3) reporte riguroso con IC por caballo y baselines |
| **Debilidades** | (1) El MIL no supera a un promedio lineal y no es temporal; (2) confusores de captura (duración, contenedor, rotación, sesión); (3) eslabón entrenamiento–inferencia y etiquetas de un único observador |
| **Mejoras de prioridad alta** | Reformular la contribución; corregir el eslabón entrenamiento–inferencia; controlar sesión y filmación; repetir la CV (≥ 5 particiones); rehacer la validación interna; incorporar SimBA-RF como complemento; arreglar las 9 geométricas; declarar limitaciones de etiquetado |
| **Veredicto** | Metodología de confiabilidad ALTA; estadística MEDIA; generalización externa no evaluada; hipótesis **parcialmente sustentada** |

<div class="saltopagina"></div>

## Hallazgos principales

1. **La metodología de evaluación es sólida y verificable**; el 0,74 es una estimación creíble del rendimiento sobre **caballos nuevos de esta población y este tipo de captura**, con ±0,06 de IC y ±0,015 entre semillas.
2. **La arquitectura no demuestra valor agregado**: la media lineal de las características de 16 frames rinde igual (0,741), con 2,6 veces menos dispersión entre folds. En 4 particiones alternativas el MIL es 0,003–0,021 mejor (media +0,015); no alcanza para afirmar una ventaja.
3. **El contexto de captura contiene señal**: sin mirar la cara, la duración sola da 0,544, los metadatos 0,618 y la fecha de sesión 0,535 (0,605 combinando técnicos y fecha). La duración del clip determina la ventana de los 16 frames (0,19 s entre frames en `Disgustado`, 0,39 s en las otras). Sobre el subconjunto 1080×1920 y ≥ 5 s el MIL conserva 0,705.
4. **Hallazgos nuevos respecto de la auditoría anterior** (no estaban en ella): (a) la **validación interna son los 7 caballos con más clips** (≈ 10 clips c/u frente a 2,1 en el train interno); (b) los **clips se recodifican en `mp4v`** y la interfaz decodifica el video original: en la muestra medida cambia la predicción en 2 de 28 clips, y al correr la interfaz sobre 16 videos del propio dataset (con el modelo ya entrenado sobre ellos) la predicción coincidió con la etiqueta sólo en 8; (c) **la inferencia no activa la orientación del video** que sí se aplicó al crear el dataset: 69 de 274 videos (25 %) declaran rotación y originan el 35 % de los clips, y esa rotación está asociada a la clase (1/110 de `Disgustado`, 47/93 de `Sereno`); (d) la partición **no coincide con `sklearn.GroupKFold`** aunque el docstring dice «réplica»; (e) la **calibración**: ECE 0,081.
5. **Las etiquetas** tienen ahora criterio documentado (EquiFACS, etogramas) pero **un único observador no ciego**; el acuerdo sobre el dataset no se midió y el techo de rendimiento se desconoce.
6. **El canal geométrico del MIL es casi vacío**: las 9 variables llevan un solo grado de libertad (inclinación de la cabeza) y sus máscaras de validez valen 1 siempre (confianza > 0, no ≥ 0,50).
7. **Debilidades abiertas**: validación externa, utilidad para veterinarios (R15, R16), ramas de variantes sin fusionar, documentación con cifras viejas.

## Cómo se atendió la devolución docente

| Observación | Atención |
|---|---|
| «¿Cómo está la repetibilidad?» | Medida: bit a bit con la semilla 42; ±0,015 entre semillas; 0,740–0,758 entre 5 particiones de caballos (DE 0,008; cap. 14) |
| «¿Espacio temporal o sólo procesamiento del video?» | Confirmado: no es temporal; R7 parcial (cap. 2, 11) |
| «No lo estamos haciendo, sacar» | Antiguo R8 retirado de la matriz (cap. 1) |
| «¿DeepLabCut con Horse-30?» | No; modelo generalista, sin ajuste (cap. 12) |

## Veredicto

El proyecto **cumple su objetivo central** (un sistema que clasifica expresiones de caballos en video con evaluación honesta por individuo) y su metodología es de **confiabilidad ALTA**; la **confiabilidad estadística es MEDIA**, la **generalización externa no está evaluada**, y **la hipótesis queda parcialmente sustentada**: factibilidad de clasificar, sí; valor del aprendizaje profundo y de la modelización temporal, no demostrado; utilidad para expertos, no verificable. Las mejoras de prioridad alta (cap. 16) son: reformular la contribución, corregir el eslabón entrenamiento–inferencia, controlar la sesión como confusor, repetir la validación cruzada, rehacer la validación interna y llevar las limitaciones de etiquetado al informe.

**Experimento SimBA.** Con la misma pose, los mismos 16 frames y los mismos folds, un Random Forest sobre features geométricas y de movimiento calculadas con código propio supera al MIL en el conjunto original (0,841 frente a 0,740), pero la ventaja se reduce y deja de ser concluyente al controlar la filmación. Conviene incorporarlo como **modelo complementario e interpretable**, no como sustituto, y **controlar la sesión y la filmación** antes de atribuir el rendimiento a la expresión. Ninguno de los dos enfoques resuelve EquiFACS, el acuerdo interobservador ni la validación veterinaria. Los conteos de requisitos no cambian.
