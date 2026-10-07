# Análisis de sobreajuste del modelo vigente

Modelo: MIL multimodal de `scripts/evaluar_v2.py` (359 clips, 133 caballos, 4 folds por caballo,
validación interna de 7 caballos por fold, *early stopping* sobre F1 macro de validación,
paciencia 10). Fuente de todos los números: `datos_equinos/resultados/objetivo3_evaluacion_v2.json`
(campos `por_fold[*].curva` y `resumen.sobreajuste`). Figura: `figuras/figura5_curvas_sobreajuste`
(`04_figuras_tesis.ipynb`).

## Qué se registra

Hasta el 18/09 la curva guardaba la pérdida de train (la del paso de entrenamiento, con dropout),
la pérdida de validación y el F1 de validación. El gap no se podía leer. Desde este cambio cada
época registra además, para el **train interno evaluado sin dropout** con los pesos del final de
la época: `loss_train_eval`, `f1_train` y `acc_train`, y `acc_val`.

Medir el train no altera el entrenamiento: se hace con un cargador con generador propio
(`clasificacion.cargador_de_medicion`), porque iterar un `DataLoader` sin generador saca una
semilla del RNG global aunque no baraje. Verificado de dos formas:

- `tests/test_sobreajuste.py` comprueba que el RNG global y el generador del train quedan
  intactos, con un control que muestra el consumo sin generador propio;
- la re-ejecución del 25/09 dio **359/359 predicciones iguales**, curvas de validación idénticas y
  **pesos idénticos bit a bit** en los 4 folds y en el modelo de producción.

## Resultados en la época elegida

| Fold | Mejor época | Épocas corridas | F1 train | F1 validación | F1 test | Gap train−val | Gap train−test | Época de mínima pérdida de val. |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 28 | 38 | 0,973 | 0,700 | 0,718 | 0,273 | 0,254 | 19 |
| 1 | 26 | 36 | 0,994 | 0,766 | 0,769 | 0,228 | 0,226 | 15 |
| 2 | 20 | 30 | 0,969 | 0,641 | 0,677 | 0,329 | 0,292 | 20 |
| 3 | 40 | 50 | 0,981 | 0,742 | 0,795 | 0,238 | 0,186 | 37 |
| **Media** | 28,5 | 38,5 | **0,979** | **0,712** | **0,740** | **0,267** | **0,240** | 22,8 |

## Lectura

No existe un umbral universal que declare sobreajuste; los gaps se leen junto con el resto:

1. **El modelo memoriza el train interno.** F1 de 0,97 a 0,99, con una pérdida que cae hasta ~0,05.
   Con 132 355 parámetros entrenables y ~199 clips (~93 caballos) por fold, es lo esperable.
2. **La pérdida de validación sube** a partir de la época 15–37, mientras el F1 de validación
   todavía mejora: el modelo gana aciertos y pierde calibración. Las probabilidades que muestra la
   interfaz son, por lo tanto, más confiadas de lo que justifican sus aciertos.
3. **La estimación de test no está inflada.** En los cuatro folds el F1 de test es igual o mayor
   que el de validación. La época se elige sin mirar el test —en el fold 3 el test tenía su máximo
   en otra época, y el procedimiento no la eligió—, así que 0,740 es una estimación honesta sobre
   caballos no vistos.
4. **La elección de época es ruidosa**: 20 a 40 entre folds, con validaciones de 7 caballos y
   F1 que salta ±0,05 entre épocas consecutivas. La rama `feat/validacion-cruzada-anidada-NestedCV`
   probó promediar curvas internas y no lo resolvió (`docs/registro_de_experimentos.md`).

**Clasificación: sobreajuste moderado, contenido por el *early stopping*.** La formulación
correcta para la tesis es «el modelo sobreajusta el conjunto de entrenamiento; el *early stopping*
sobre una validación separada por caballo mantiene la estimación de test estable y consistente con
la validación». No debe afirmarse que no hay sobreajuste.

## Qué lo reduciría

- Menos capacidad en la cabeza: el 62 % de los parámetros está en la atención, y un promedio
  lineal de las características rinde lo mismo (`objetivo3_lineas_de_base.json`).
- Una validación interna más grande o estratificada por clase (la del fold 0 es 66 % `Sereno`).
- Calibrar las probabilidades sobre validación (*temperature scaling*) antes de mostrarlas.
- Más caballos: es la única vía que ataca la causa y no el síntoma.
