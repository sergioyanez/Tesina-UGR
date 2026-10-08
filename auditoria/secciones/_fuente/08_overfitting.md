# 8. Sobreajuste

## 8.1 Evidencia disponible

Los historiales por época **existen en el artefacto vigente**: `objetivo3_evaluacion_v2.json`, `por_fold[*].curva`, con `loss_train`, `loss_train_eval` (train interno sin dropout, comparable con la pérdida de validación), `f1_train`, `acc_train`, `loss_val`, `f1_val`, `acc_val`. No fue necesario re-entrenar para obtenerlos; sí se re-entrenó para comprobar que el código vigente reproduce esas curvas (idénticas con la semilla 42) y para medir la variabilidad entre semillas (§14).

{{FIG:fig_curvas_por_fold.png|Figura 8.1. Pérdida, F1 macro y accuracy de train (interno, sin dropout) y validación por época, para cada fold. La línea ámbar es la época elegida con la validación; la línea punteada negra es el F1 de test en esa época.}}

## 8.2 Brechas

{{TABLA:gaps}}

{{TABLA:perdida}}

**Lectura de los números:**

- **El modelo memoriza el train interno en los 4 folds**: F1 train 0,97–0,99 (accuracy 0,97–1,00) contra 0,64–0,77 en validación. El gap train−validación es 0,267 en promedio (0,228–0,329) y train−test 0,240 (0,186–0,292); ambos son varias veces mayores que la dispersión entre folds del F1 (DE 0,052) y que el ruido de las semillas (DE 0,015).
- **La pérdida de validación ya está subiendo cuando se elige la época.** Su mínimo se alcanza 3–11 épocas antes que el máximo de F1 en 3 de los 4 folds y en la misma época en el fold 2 (épocas 19, 15, 20, 37 frente a 28, 26, 20, 40), y al final de la ejecución es 0,18–0,35 mayor que en su mínimo: el modelo gana aciertos mientras pierde calibración. Confirmado con las probabilidades de test: confianza media **{{VAL:calibracion.json:confianza_media}}** frente a acierto **{{VAL:calibracion.json:accuracy}}** (ECE {{VAL:calibracion.json:ECE_10_bins}}); el 44,6 % de los clips recibe confianza > 0,9 y acierta el 89,4 % de ellos (§11.4).
- **El F1 de validación cae 0,05–0,07 en las 10 épocas posteriores a la mejor** (parte de la caída es regresión a la media tras elegir un máximo ruidoso).
- **La estimación de test no está inflada**: test ≥ validación en 4/4 folds.
- **La memorización no es exclusiva de la cabeza profunda.** El baseline lineal sobre la media de los 16 frames también alcanza **F1 train 1,000** en los 4 folds (brecha train−test 0,259, equivalente a la del MIL: 0,240) y generaliza igual. La brecha es en buena medida consecuencia de la dimensionalidad (512 características por frame; ≈ 199 clips y ≈ 93 caballos de entrenamiento), no de la complejidad temporal.

## 8.3 Clasificación

**Escala (declarada antes de aplicarla, sin umbrales numéricos arbitrarios):** se clasifica por *consecuencias observables*, no por el tamaño de un gap.

| Categoría | Condición observable |
|---|---|
| SIN EVIDENCIA | Train y validación no se separan por más que el ruido entre folds/semillas |
| LEVE | Hay separación, pero la métrica de validación no se degrada con más entrenamiento y las probabilidades están calibradas |
| MODERADO | Hay separación clara en todos los folds, la **estimación de test no está sesgada** (la parada temprana la contiene) y aparecen señales de pérdida de calidad *distintas del F1* (calibración, pérdida de validación creciente) |
| IMPORTANTE | La métrica de validación o de test se degrada materialmente con el entrenamiento, la mejor época es de las primeras, o un modelo de menor capacidad rinde claramente más |
| INSUFICIENTE | No hay historial o validación independiente con que juzgarlo |

**Resultado: MODERADO.**

- Contra «SIN EVIDENCIA» y «LEVE»: la separación (0,27) es 5 veces el desvío entre folds y está en 4/4 folds; la pérdida de validación sube y la confianza excede el acierto.
- Contra «IMPORTANTE»: el test acompaña a la validación (≥ en 4/4), la mejor época no es temprana (20–40), y un modelo lineal con la misma brecha rinde lo mismo (0,741), de modo que **no hay evidencia de que un modelo de menor capacidad generalice mejor** en este conjunto.
- **Evidencia contradictoria que debe mostrarse:** (a) un gap train−test de 0,24 es grande en términos absolutos y, por convención de muchos autores, se llamaría *importante*; la clasificación MODERADO depende de que el efecto sobre la estimación de test es nulo gracias a la parada temprana. (b) El F1 de validación cae después de la mejor época. (c) La interfaz muestra probabilidades sobreconfiadas.
- **No debe afirmarse que no hay sobreajuste.** La redacción correcta: *«el modelo sobreajusta el conjunto de entrenamiento; la parada temprana sobre una validación separada por caballo mantiene la estimación de test estable y consistente con la validación; las probabilidades emitidas están sobreconfiadas»*.

{{FIG:fig_calibracion.png|Figura 8.2. Calibración de las probabilidades del modelo de cada fold sobre su test (fuera de muestra); los números son clips por intervalo.}}
