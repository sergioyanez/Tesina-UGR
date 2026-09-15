# Análisis de Resultados: Clasificación de Expresiones Faciales Equinas

Este reporte detalla los resultados obtenidos en la Etapa 3 (`03_clasificacion.ipynb`) tras la incorporación de los nuevos videos de prueba y el reprocesamiento completo del pipeline (extracción de clips, alineación facial y normalización de pose). 

El objetivo técnico consiste en entrenar y evaluar un clasificador triclase (**Disgustado**, **Interesado**, **Sereno**) bajo un protocolo riguroso de validación por ejemplar.

---

## 1. Composición del Dataset Actualizado

Con la incorporación de los nuevos videos y el descarte de clips que no cumplieron el criterio estricto de integridad (16 frames alineados por clip), la base consolidada cuenta con:

* **Clips íntegros totales:** 147 *(+6 clips respecto al dataset histórico de 141)*
* **Ejemplares únicos:** 79 *(+6 ejemplares respecto a los 73 originales)*
* **Distribución por clase:**
  * **Disgustado:** 25 clips en 20 ejemplares *(crecimiento real de +4 clips y +5 ejemplares sobre los 21/15 iniciales)*
  * **Interesado:** 58 clips en 36 ejemplares *(+2 clips y +1 ejemplar)*
  * **Sereno:** 64 clips en 36 ejemplares *(sin variaciones)*

> [!NOTE]
> **Partición por ejemplar:** 10 de los 79 caballos presentan clips íntegros en más de una clase (`178`, `218`, `230`, `237`, `37`, `54`, `55`, `906`, `907`, `910`). Agrupar por el par ejemplar-clase o por clip provocaría fuga de datos, inflando la métrica artificialmente. Por ende, la partición se realiza estrictamente por ejemplar en 4 folds estratificados.

---

## 2. Rendimiento vs. Líneas de Base y Control de Fuga

Para verificar que el modelo extrae patrones visuales genuinos y no artefactos estadísticos o identitarios, se evalúa contra tres líneas de base sobre los mismos folds:

| Modelo / Enfoque | F1 Macro | Balanced Accuracy | Recall Disgustado | Descripción / Diagnóstico |
| :--- | :---: | :---: | :---: | :--- |
| **Clase Mayoritaria** | `0.183 ± 0.021` | `0.333 ± 0.000` | `0.000 ± 0.000` | Predice siempre la clase dominante (*Sereno*). |
| **Por Identidad (sin imagen)** | `0.183 ± 0.021` | `0.333 ± 0.000` | `0.000 ± 0.000` | Cae al 100% al voto mayoritario: el caballo de test nunca fue visto. |
| **Partición Ingenua (diagnóstico de fuga)** | `0.422 ± 0.105` | `0.478 ± 0.073` | `0.098 ± 0.122` | Particiona por clip sin agrupar por caballo. |

> [!IMPORTANT]
> **Fuga eliminada:** La diferencia de **`+0.239` en F1 Macro** entre la partición ingenua y la de identidad mide la magnitud del *confounder* de individuo. Dado que la gran mayoría de los caballos solo aparecen en una clase, reconocer al individuo equivaldría a clasificar la expresión. El protocolo adoptado neutraliza totalmente este sesgo.

---

## 3. Modelo Final (1 frame + voto temporal)

El modelo adoptado utiliza un extractor convolucional congelado (ResNet-18 preentrenada en ImageNet, 512 dimensiones) con pooling adaptativo para recortes faciales de 224×343, seguido de una cabeza de clasificación lineal agregada por voto mayoritario sobre los 16 frames del clip.

* **F1 Macro:** **`0.649 ± 0.092`** *(supera ampliamente la línea de base trivial de `0.183` y mejora el `0.626` del dataset original)*
* **Balanced Accuracy:** `0.664 ± 0.070`
* **Recall Disgustado:** **`0.568 ± 0.047`** *(mejora sustancial en estabilidad y cobertura respecto al `0.473 ± 0.284` previo)*
* **Desempeño por fold (F1 Macro):**
  * Fold 0: `0.556`
  * Fold 1: `0.599`
  * Fold 2: `0.676`
  * Fold 3: `0.764`

---

## 4. Matriz de Confusión (Evaluación Fuera de Muestra)

Agregada sobre los 4 folds de validación cruzada (147 clips):

| Real \ Predicha | Disgustado | Interesado | Sereno | Total Real | Recall Clase |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Disgustado** | **14** | 4 | 7 | 25 | **56.0%** |
| **Interesado** | 3 | **48** | 7 | 58 | **82.8%** |
| **Sereno** | 14 | 12 | **38** | 64 | **59.4%** |
| **Total Predicha** | 31 | 64 | 52 | 147 | — |

### Observaciones clave:
1. **Interesado:** Es la clase más discriminable y consistente, alcanzando un 82.8% de precisión por clase.
2. **Disgustado:** Se identifican correctamente 14 de los 25 clips. Las confusiones ocurren principalmente hacia *Sereno* (7 casos).
3. **Sereno:** Manifiesta alta dispersión; se confunde en proporciones similares tanto con *Disgustado* (14 clips) como con *Interesado* (12 clips), sugiriendo que la postura neutra o serena comparte microexpresiones intermedias.

---

## 5. Ablación y Diagnóstico de Errores por Ejemplar

### Ablación por origen de alineación (82 clips de detección directa)
Al evaluar únicamente los clips cuyos 16 frames provinieron de detección directa de keypoints (sin interpolación temporal ni caja manual):
* **F1 Macro:** `0.527 ± 0.050` (`-0.122` frente al conjunto completo).
* **Recall Disgustado:** Cae a `0.167 ± 0.289` debido a la escasez de muestras puras (7 clips en 4 folds).
* **Conclusión:** El preprocesamiento de seguimiento temporal y normalización geométrica del Objetivo 2 introduce señal robusta y resulta imprescindible para viabilizar el entrenamiento.

### Concentración de errores
* **30 de los 79 ejemplares** cometen al menos un error (47 fallas en total).
* **Los 10 peores ejemplares concentran el 57% de los errores (27/47):**
  * Caballo `906`: 5 fallas sobre 9 clips.
  * Caballo `13`: 4 fallas sobre 4 clips.
  * Caballo `237`: 4 fallas sobre 6 clips.
  * Caballos `56` y `58`: 3 fallas sobre 4 clips cada uno.

---

## 6. Contraste con la Hipótesis del Trabajo

La hipótesis técnica planteaba que el modelado de dependencias temporales secuenciales superaría significativamente el análisis estático frame a frame.

* **Criterio formal de aceptación (declarado a priori):** `0.691` F1 Macro.
* **Resultado del modelo final:** `0.649 ± 0.092`.
* **Resultado formal:** **NO CUMPLE** (diferencia de `-0.042`).

> [!WARNING]
> **La hipótesis secuencial temporal no queda sostenida por la evidencia empírica:**
> 1. Las arquitecturas secuenciales dedicadas (GRU, Mecanismos de Atención Temporal) obtuvieron rendimientos inferiores (`0.580` a `0.632`) frente al clasificador lineal estático con agregación por voto (`0.649`).
> 2. En las pruebas de permutación temporal, barajar aleatoriamente los 16 frames produjo predicciones con variación nula (`+0.000` en atención): las capas aprendieron pesos prácticamente uniformes, colapsando a un promedio simple.

---

## 7. Conclusiones Generales

1. **Existe señal facial inequívoca:** El salto desde la línea base (`0.183`) hasta `0.649` demuestra empíricamente que la región facial normalizada contiene información determinante sobre el estado emocional del equino.
2. **Suficiencia del modelo estático:** Para secuencias breves (5–7 segundos), la agregación de características estáticas es superior y menos propensa a sobreajuste que modelos recurrentes complejos.
3. **Cuello de botella en diversidad de individuos:** El límite actual del rendimiento radica en el número de ejemplares únicos (especialmente en clases minoritarias), y no en la capacidad de la arquitectura neuronal.
