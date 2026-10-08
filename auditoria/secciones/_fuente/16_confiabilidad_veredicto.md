# 16. Confiabilidad, puntos fuertes y débiles, mejoras y veredicto

## 16.1 Confiabilidad por dimensión

| Dimensión | Nivel | Justificación |
|---|---|---|
| **Metodológica** | **ALTA** | Separación por caballo en la validación cruzada y en la interna, con aserciones; test usado una sola vez (verificado con test alterado); métrica adecuada y verificada contra sklearn; líneas de base sobre los mismos folds. Resta: validación interna de 7 caballos prolíficos y desbalanceada; selección de variantes no anidada |
| **Estadística** | **MEDIA** | IC 95 % del F1 macro de ±0,06 (0,677–0,797) con bootstrap por caballo; 4 folds; tamaño efectivo ≈ 133–236; diferencias menores a ≈ 0,03 entre variantes no son interpretables. Suficiente para afirmar «muy por encima del azar y de los metadatos»; insuficiente para distinguir arquitecturas |
| **Generalización** | **MEDIA** (a caballos nuevos de la misma población) / **NO EVALUADA** (externa) | Un solo observador, una sola filmadora, una región; contexto de captura asociado a la clase (hasta 0,605 de F1 sin mirar la cara); diferencias entre entrenamiento e inferencia (recodificación, orientación) |
| **Reproducibilidad** | **ALTA** (modelo vigente) / **MEDIA** (variantes) | 359/359 predicciones y pesos idénticos bit a bit; ±0,015 entre semillas; entorno fijado; las variantes dependen de ramas sin fusionar |
| **Consistencia entre folds** | **MEDIA** | F1 0,677–0,795, sin folds colapsados; DE 0,052 (2,6 × la del baseline lineal); la mejor época varía 20–40 entre folds y 13–50 entre semillas |

## 16.2 Puntos fuertes

1. **Separación por caballo verificada por código** y cuantificada: sin agrupar, la identidad sola daría F1 0,613; agrupando, 0,183 (+0,43 de F1 espurio que el protocolo elimina).
2. **Test intacto**: la prueba del test alterado muestra que época, curvas y pesos no dependen de las etiquetas de test.
3. **Reproducibilidad exacta** del modelo vigente (predicciones, curvas y pesos), con 6 semillas medidas y entorno fijado.
4. **Rigor de reporte**: F1 macro verificado, métricas por clase, matrices de confusión, IC por caballo, líneas de base sobre los mismos folds y registro de los resultados negativos (hipótesis temporal no sostenida, *fine-tuning* y CV anidada no adoptados, criterio declarado antes de medir).
5. **Backbone congelado y extractor preentrenado**, apropiado para 359 clips; el pipeline de pose y de alineado es el mismo código para entrenamiento e inferencia.
6. **Etiquetado con criterio documentado** por una especialista (EquiFACS), con clases excluidas por falta de acuerdo en el piloto.
7. **Honestidad de la interfaz**: declara que no es un diagnóstico, cita las métricas del JSON vigente (no escritas a mano) y devuelve negativas explicadas.

## 16.3 Puntos débiles

1. **Sin valor agregado demostrado del MIL multimodal** frente a la media lineal de 16 frames (0,740 frente a 0,741; IC de la diferencia [−0,068; +0,058]); el único respaldo a una ventaja pequeña (+0,015 en 4 particiones alternativas) es débil.
2. **No es un modelo espacio-temporal**: invariante al orden (R7); la hipótesis dinámica del plan no se sostiene con los datos.
3. **Confusores de captura**: duración, contenedor, rotación y fecha de sesión predicen la clase sin mirar la cara (0,535–0,618); la **ventana de 16 frames depende de la duración y por lo tanto de la clase**.
4. **Etiquetas de un único observador no ciego**, con criterio que usa contexto; acuerdo no medido sobre el dataset; techo de rendimiento desconocido.
5. **Validación interna defectuosa**: 7 caballos con más clips (≈ 10 clips c/u frente a 2,1 en el train interno), desbalanceada (hasta 4,18:1); elección de época ruidosa.
6. **Eslabón entrenamiento–inferencia**: clips recodificados `mp4v` contra video original (cambia 7 % de las predicciones en la muestra medida) y orientación de video no tratada en la inferencia; la confiabilidad mostrada no necesariamente aplica a videos nuevos: en la prueba de extremo a extremo la interfaz coincidió con la etiqueta en 8 de 16 videos del propio dataset, con el modelo ya entrenado sobre ellos.
7. **Sobreajuste moderado**: F1 train 0,98; probabilidades sobreconfiadas (ECE 0,081; confianza 0,824 frente a acierto 0,744).
8. **Sin validación externa ni con usuarios**; un solo establecimiento regional; 133 caballos.
9. **El canal geométrico del MIL es casi vacío**: las 9 variables llevan un solo grado de libertad (inclinación de la cabeza) y sus máscaras de validez valen 1 siempre (se calculan con confianza > 0 y no ≥ 0,50) (§17.4).
10. **Deuda de trazabilidad y documentación**: ramas sin fusionar, huella dependiente de `mtime`, tres archivos con cifras viejas, configuración escrita en los scripts, partición descrita como «réplica de `GroupKFold`» sin serlo exactamente.

## 16.4 Mejoras recomendadas

**Prioridad ALTA** (antes de dar el modelo y el informe por cerrados)

| # | Mejora | Esfuerzo | Por qué |
|---|---|---|---|
| A1 | **Redactar la contribución con precisión**: el rendimiento proviene de un extractor preentrenado y de la agregación de varios frames; ni la atención ni el orden temporal están demostrados; quitar «espacio-temporal» donde describe al modelo; evaluar adoptar la media lineal por parsimonia si el criterio de los autores lo permite | Texto | Respuesta directa a la devolución docente y al empate 0,740/0,741 |
| A2 | **Investigar y corregir el eslabón entrenamiento–inferencia** (coincidencia de 8/16 en la prueba de la interfaz): activar `CAP_PROP_ORIENTATION_AUTO` en `inferencia.preparar_clip` y verificar que DeepLabCut lea con la misma orientación; medir (o eliminar) el efecto de la recodificación `mp4v`; extender `test_equivalencia.py` a video original | Bajo–medio | Una interfaz que acepta videos del usuario sin esta garantía puede informar una confiabilidad que no aplica |
| A3 | **Controlar los confusores de sesión/contexto**: reportar rendimiento por estratos de sesión (fecha), con partición *leave-one-session-out* o agrupada por sesión, y/o restringir a un subconjunto homogéneo de captura | Medio | Es la mayor amenaza a la validez que el protocolo actual no controla |
| A4 | **Repetir la validación cruzada** con ≥ 5 particiones y semillas y reportar la media de repeticiones con IC (`scripts/a15_particiones_alternativas.py` ya lo hace) | Bajo | La variabilidad por partición/semilla (±0,015) es del orden de las diferencias que se discuten |
| A5 | **Reemplazar la validación interna** por una muestra aleatoria de caballos estratificada por clase, o por una CV interna, y comprobar la época y el F1 | Medio | Hoy son los 7 caballos con más clips |
| A6 | **Declarar en el informe las limitaciones de etiquetado** (un observador no ciego, criterio contextual, techo desconocido) y las de captura (un establecimiento regional, un filmador) | Texto | El protocolo ya las documenta; falta que lleguen al informe |
| A7 | **Incorporar SimBA-RF como modelo complementario** (liviano, sin GPU salvo DLC, interpretable) y como línea de base geométrica fuerte; no sustituir al MIL ni presentar su ventaja como definitiva | Bajo | 0,841 ± 0,003 frente a 0,740; limpio no concluyente (cap. 17) |
| A8 | **Controlar sesión y filmación** en ambos modelos (residualizar duración/resolución, estratificar, partición por sesión) y reportar el piso deconfundido (0,777 para SimBA-RF) | Medio | Las features codifican duración (R² 0,53) y resolución (AUC 0,73); el MIL comparte el confusor |
| A9 | **Arreglar la validez de las 9 geométricas**: máscaras con `confianza ≥ 0,50`, rotar al eje de la cabeza, ampliar a orejas/boca (o reemplazarlas por las features del cap. 17) | Bajo | Hoy llevan un solo grado de libertad y máscaras constantes |

**Prioridad MEDIA**

1. Calibrar las probabilidades (*temperature scaling* sobre validación) o mostrar sólo la precisión medida por clase.
2. Reducir la capacidad de la cabeza (62 % de los parámetros está en la atención) y compararla bajo el mismo protocolo.
3. Revisar candidatos a error de etiqueta o de identidad: los 9 caballos con fallos del 100 %, y el par `123`/`169`.
4. Pasar los hiperparámetros a un archivo de configuración y fijar `cudnn.deterministic`.
5. Reemplazar la huella por `mtime` por una huella de contenido; actualizar las cifras viejas de `inferencia.py`, `app_inferencia.py` y `openspec/project.md`.
6. Fusionar o archivar con etiqueta las tres ramas de variantes, o dejar en `main` un índice con sus *hashes*.

**Prioridad BAJA** (trabajo futuro)

1. Validación externa: otro establecimiento, otra cámara, otro observador.
2. Segunda anotación ciega de una muestra (≈ 60 clips, kappa de Cohen) para estimar el techo de las etiquetas.
3. Evaluación de utilidad con veterinarios (R15, R16).
4. Modelos temporales (unidades de acción, flujo óptico) sólo si un experimento controlado muestra un aporte del orden.

## 16.5 Veredicto cuantificado

| Afirmación | Veredicto | Cifra |
|---|---|---|
| El sistema clasifica expresiones en caballos no vistos mejor que el azar | **Sustentada** | F1 macro 0,740 (IC 95 % 0,677–0,797) frente a 0,327 |
| Lo hace mejor que usar sólo la forma de filmar (metadatos) | **Sustentada** | +0,121 (IC 95 % +0,031; +0,211); la señal del contexto sola llega a 0,605 |
| Lo hace mejor que un modelo lineal sobre las características de los frames | **No sustentada** | −0,005 (IC 95 % −0,068; +0,058) |
| Modela la dinámica temporal | **No** | 0 de 359 predicciones cambian al permutar los frames |
| Es útil para expertos | **No verificable** | Sin validación con veterinarios; precisión por clase 0,73–0,76 |
| El resultado es repetible | **Sí** (con semilla 42, bit a bit); **±0,015** entre semillas | 6 semillas: 0,726–0,763 |
| Rendimiento esperable donde la filmación no delata la clase | ≈ 0,70 | 0,705 en 236 clips (28 de `Disgustado`) |

| Indicador de calidad | Resultado |
|---|---|
| Requisitos del plan | 10 cumplidos, 6 parciales, 2 no verificables, 0 no cumplidos (de 18) |
| Buenas prácticas | 11 cumplen, 7 parciales, 4 no (de 22) |
| Condiciones de un buen modelo | 6 cumplen, 8 parciales, 2 no evaluadas (de 16) |
| Sobreajuste | MODERADO |
| Balance | RAZONABLEMENTE BALANCEADO (clips 1,34:1) / BALANCEADO (caballos 1,15:1) |
| Fuga de información | No se observan evidencias (0 cruces de caballo, video, clip ni frame); riesgo residual de sesión y de selección no anidada |

**Conclusión.** El proyecto construyó un clasificador de tres clases con una metodología de separación por individuo sólida, verificada y reproducible. La estimación 0,74 de F1 macro es **creíble para caballos nuevos de esta población y este tipo de captura**, con un tope de incertidumbre de ±0,06 (IC) y ±0,015 (semillas). No está demostrado que la arquitectura profunda, la atención, el canal geométrico o el orden temporal aporten sobre un promedio lineal de las características de varios frames; el contexto de captura contiene una parte de la señal; y la utilidad para expertos, la validez externa y el techo impuesto por las etiquetas siguen sin conocerse.

**Efecto del experimento SimBA sobre el veredicto.** Un Random Forest sobre 160 features de pose «tipo SimBA» (código propio) obtiene 0,841 ± 0,003 frente al 0,740 del MIL, y 0,750 frente a 0,705 en el subconjunto limpio (no concluyente). El resultado **matiza** la conclusión anterior: la señal discriminativa está en la geometría facial (cara 0,848, orejas 0,784, cuello 0,461) y no depende de una red visual profunda; pero **comparte el confusor de filmación** (R² 0,53 con la log-duración; piso deconfundido 0,777), de modo que ni el 0,74 ni el 0,84 son estimaciones libres de contexto. La confiabilidad estadística y la de generalización siguen en los niveles indicados.
