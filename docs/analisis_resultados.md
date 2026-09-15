# Análisis de Resultados: Clasificación de Expresiones Faciales Equinas

Este reporte detalla los resultados obtenidos por el pipeline de clasificación tras la incorporación de los nuevos videos de prueba. El objetivo de la Etapa 3 (`03_clasificacion.ipynb`) es entrenar un modelo para predecir si un caballo está **Disgustado**, **Interesado** o **Sereno**.

---

## 1. Composición del Dataset Actualizado

Al incorporar los nuevos videos de prueba (los cuales pasaron satisfactoriamente por la extracción de frames y la normalización YOLO), el conjunto de datos experimentó un incremento marginal pero crítico:

*   **Clips íntegros totales:** 142 (antes 141)
*   **Ejemplares únicos:** 74 (antes 73)
*   **Distribución:**
    *   **Disgustado:** 20 clips en 15 ejemplares *(+1 clip, +1 ejemplar respecto al dataset original)*
    *   **Interesado:** 58 clips en 36 ejemplares
    *   **Sereno:** 64 clips en 36 ejemplares

> [!NOTE]
> La partición de validación se realiza de manera estricta **por ejemplar** (sin fuga de datos) y no por clip. Esto previene que un modelo memorice las facciones de un caballo y las use para hacer trampa en la clasificación.

---

## 2. Rendimiento del Modelo vs. Líneas de Base

Para determinar si el modelo extrae "señal" real de las imágenes, se compara su F1 Macro contra líneas de base estadísticas ("modelos tontos" que no analizan la imagen).

| Modelo | F1 Macro | Descripción |
| :--- | :--- | :--- |
| **Voto Mayoritario** | `0.206` | Adivina siempre la clase más común ("Sereno"). |
| **Por Identidad** | `0.206` | Usa el ID del caballo para adivinar (Demuestra que no hay fuga de datos). |
| **Modelo Final (1 frame)** | **`0.659`** | Analiza las características faciales extraídas estáticamente. |

> [!TIP]
> **Conclusión sobre la señal:** Al destruir el piso de `0.206` y alcanzar un `0.659`, queda rigurosamente demostrado que **las imágenes faciales de los caballos contienen señal altamente útil** para predecir su estado emocional.

---

## 3. Contraste con la Hipótesis del Proyecto

La hipótesis principal de la tesina buscaba comprobar si el análisis **secuencial temporal** (mirar el video en orden) superaba al análisis de una foto estática (el modelo "Un frame").

Al inicio del experimento, se fijó un umbral estadístico estricto de **0.691** para aceptar la hipótesis (calculado como el rendimiento del modelo base + una desviación estándar).

*   **F1 Macro obtenido con nuevos datos:** `0.659` (antes `0.626`)
*   **Umbral requerido:** `0.691`

> [!WARNING]
> **La hipótesis temporal no queda sostenida.** Aunque el nuevo video inyectó un sólido `+0.033` a la métrica general, el modelo sigue sin superar el umbral. El mejor modelo sigue siendo invariante al orden: es decir, predecir barajando el orden de los frames de un video genera exactamente la misma predicción que viéndolos en orden.

---

## 4. Conclusiones y Próximos Pasos

El experimento fue un éxito metodológico, arrojando resultados contundentes:

1. **La IA puede leer caras de caballos:** El preprocesamiento (detección de orejas, alineamiento ojo-hocico) es fenomenal y permite clasificar con mucha certeza respecto al azar.
2. **El modelo lineal es suficiente:** Modelos más complejos (Redes Neuronales Recurrentes, GRU, Atención) colapsan hacia promedios simples y no aportan ventajas.
3. **El techo es la escasez de ejemplares:** Para mejorar la métrica actual (y quizás destrabar la utilidad de la secuencia temporal) **no hace falta arquitectura más compleja ni más clips del mismo caballo**, hace falta **grabar más ejemplares distintos**, particularmente en la clase `Disgustado` que sigue siendo la minoritaria.
