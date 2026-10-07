# 2. Objetivo general frente a hipótesis

El plan separa dos cosas que suelen confundirse: un **objetivo** (construir algo) y una **hipótesis** (afirmar algo que puede ser falso). Se evalúan por separado, con los datos vigentes (359 clips, 133 caballos, 4 folds agrupados por caballo).

## 2.1 Objetivo general: «construir un sistema informático para la detección automatizada de expresiones faciales en caballos mediante el procesamiento de secuencias espacio-temporales»

| Verificación | Resultado | Evidencia |
|---|---|---|
| Existe una cadena completa video → clase | **Sí** | `equinos/inferencia.py::clasificar_video` (pose DeepLabCut → selección de ventana → alineado → ResNet-18 → MIL); `app_inferencia.py` renderiza sin excepciones (`datos/app_streamlit.json`) |
| La cadena se ejecuta sobre videos reales | **Sí, con salvedades** | 16 predicciones y 0 negativas sobre 16 videos del propio dataset (§12.5) |
| Entrenamiento real (no simulado) | **Sí** | 4 modelos por fold + modelo de producción; re-entrenados con el código vigente: 359/359 predicciones y pesos idénticos bit a bit con la semilla 42 (`datos/reentrenamiento_semillas.json`) |
| Evaluación real fuera de muestra | **Sí** | Cada uno de los 359 clips se predice con un modelo que no vio a su caballo; el test se evalúa una sola vez (§6, §7) |
| Clases bien definidas y consistentes | **Sí** | `CLASES = (Disgustado, Interesado, Sereno)` idénticas en CSV, JSON y código; criterio por clase documentado en `docs/protocolo_de_etiquetado.md` (un único observador) |
| «Secuencias espacio-temporales» | **Parcial** | El modelo recibe 16 frames pero es **invariante al orden**: barajar o invertir los frames no cambió ninguna de las 359 predicciones (diferencia máxima de logits 3,8·10⁻⁶; `datos/arquitectura.json`). Procesa una *secuencia de video* como un conjunto de imágenes; no modela dinámica temporal |

**Conclusión sobre el objetivo.** Un sistema que clasifica video en tres clases existe, se entrenó y se evaluó con una metodología de separación por individuo. El adjetivo «espacio-temporales» del objetivo no se cumple en sentido estricto, y así debe redactarse (respuesta a la duda del docente; §1.5, punto 2).

## 2.2 Comparación con baselines triviales (mismos folds)

Valores recalculados por esta auditoría sobre los **mismos 4 folds** que usa el modelo (`datos/metricas.json`).

| Modelo / línea de base | F1 macro (media ± DE entre folds) | Nota | Accuracy |
|---|---|---|---|
| Azar uniforme (1/3 por clase), esperanza | 0,327 | P95 sim. 0,367 | ≈ 0,333 |
| Azar estratificado (proporciones del train), esperanza | 0,326 | P95 sim. 0,366 | — |
| Clase mayoritaria (Sereno en los 4 folds) | 0,183 ± 0,033 |  | 0,384 |
| Solo duración del clip (LogReg) | 0,544 ± 0,035 |  | 0,532 |
| Metadatos de filmación (LogReg) | 0,618 ± 0,034 |  | 0,635 |
| Geometría sola, media y desvío (LogReg) | 0,607 ± 0,033 |  | 0,618 |
| Un frame central (LogReg) | 0,679 ± 0,024 |  | 0,688 |
| **Media de los 16 frames (LogReg)** | **0,741 ± 0,020** |  | 0,749 |
| **MIL multimodal (vigente)** | **0,740 ± 0,052** |  | 0,744 |

- El F1 macro esperado de una clasificación **aleatoria uniforme** es 0,327 (simulación de 4 000 repeticiones sobre los folds reales; percentil 95: 0,367), próximo al 1/3 teórico; no es exactamente 1/3 porque el F1 macro de una clase no predicha o muy rara se trunca a 0 en los folds pequeños.
- La **mayoritaria** predice `Sereno` en los 4 folds: accuracy 0,384 y F1 macro 0,183. La accuracy sola es engañosa con este reparto (38 % de `Sereno`), por eso la métrica principal es el F1 macro.
- El modelo (0,740) supera ampliamente a ambas. **Superarlas es una condición necesaria y débil**: la comparación informativa es con las líneas que miran la imagen (capítulo 10).

## 2.3 Hipótesis: «es factible desarrollar un modelo basado en aprendizaje profundo capaz de identificar y clasificar automáticamente expresiones faciales en equinos a partir de secuencias de video cortas, logrando un nivel de precisión útil para la asistencia de expertos humanos»

La hipótesis tiene cuatro componentes que no tienen la misma evidencia:

| Componente | Evidencia | Veredicto |
|---|---|---|
| **Factibilidad de clasificar** tres expresiones en caballos no vistos | F1 macro 0,740 (IC 95 % por caballo [0,677; 0,797]) contra 0,327 del azar y 0,183 de la mayoritaria; por encima de los metadatos de filmación (0,618; diferencia IC 95 % [+0,031; +0,211]) | **Respaldada** para este conjunto, con un tope por contexto de filmación (0,556–0,605 sin mirar la cara, §10.3) |
| **«Aprendizaje profundo»** como medio necesario | Una regresión logística sobre la media de las 16 características de ResNet-18 obtiene 0,741 ± 0,020; diferencia MIL − estática −0,005, IC 95 % [−0,068; +0,058] | **No demostrada**: lo que aporta es el *extractor preentrenado*; la cabeza profunda (atención + geometría) no mejora a un modelo lineal en la partición vigente |
| **«A partir de secuencias de video cortas»** | 16 frames superan a 1 frame (+0,059, IC 95 % [+0,001; +0,116]); el orden de los frames no influye (0/359 cambios) | **Parcial**: la *información de varios instantes* ayuda, la *dinámica* no está demostrada ni modelada |
| **«Precisión útil para la asistencia de expertos»** | El plan no define «útil»; precisión por clase 0,73–0,76 (1 de cada 4 predicciones erra); no hubo validación con veterinarios; la probabilidad que ve el usuario está sobreconfiada (confianza media 0,824 frente a acierto 0,744) | **No verificable** con los archivos disponibles |

**Conclusión sobre la hipótesis.** *Parcialmente sustentada*: no se observan evidencias en contra de la factibilidad de una clasificación automática por encima del azar y de los metadatos; sí se observa que el valor agregado de la arquitectura profunda y de la modelización temporal no está respaldado, y que la utilidad para expertos no fue evaluada. Un efecto de contexto de filmación (duración, contenedor, sesión, rotación) explica una parte del rendimiento que no se puede cuantificar con precisión (§10.3).

## 2.4 Hipótesis implícita del problema («las expresiones son eventos dinámicos»)

La frase de la *Problemática observada* del plan («capturar la evolución temporal … resulta fundamental para obtener una clasificación robusta») es una **motivación**, no un requisito del plan. En respuesta a la devolución docente se la retira de la matriz de cumplimiento (§1.5, punto 3). Se conserva como **observación empírica**: con los datos de este proyecto, el orden temporal no aportó en ninguna de las variantes medidas (atención con frames barajados da predicciones idénticas; GRU rinde peor con el orden real; `docs/registro_de_experimentos.md`, `openspec/project.md`).
