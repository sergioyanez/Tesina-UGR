# Cabezas temporales — resultados y estado

Corrida el **28/08/2026** sobre la partición persistida y las características del backbone
congelado. Reproducible con `cabezas_temporales.py`.

## Configuraciones probadas: 18

El diseño pide comparar a lo sumo tres variantes de cabeza y **dejar registrado cuántas se
probaron**, porque con 141 muestras probar veinte y quedarse con la mejor es elegir ruido. Se
probaron **18 en total**, y se informan las dieciocho: tres cabezas × dos regímenes de
aumentación a nivel de clip (6), y la prueba del orden —tres cabezas × orden real/permutado— en
dos regímenes de capacidad sobre ventanas (12).

Todas comparten backbone, proyección 512→64, presupuesto de parámetros, y régimen de
entrenamiento: 400 pasos de gradiente fijos, lote 64, AdamW `lr=1e-3`, `weight_decay=1e-2`,
dropout 0,3, pesos por clase balanceados, `semilla=42`. **Sin early stopping**: elegir la época
mirando el fold de test es fuga, y con 35 clips por fold se pueden ganar varios puntos así.

Se cuentan **pasos** y no épocas a propósito: con aumentación el conjunto es 20 veces más grande,
así que a igual cantidad de épocas la comparación mediría dos cosas mezcladas.

## La distinción que el diseño no hacía

El diseño listaba «promedio de características, GRU, atención» como candidatas equivalentes. No lo
son para la pregunta que importa:

- **`promedio` es invariante al orden.** Barajar los 16 frames no cambia una predicción. Es una
  cabeza a nivel de clip, no un modelo temporal. Si gana, la hipótesis del trabajo **no** queda
  sostenida — sería la conclusión de la línea de base de un frame, disfrazada de modelo secuencial.
- **`gru` y `atencion` sí ven el orden.** La atención lo ve gracias al embedding de posición; sin
  él también sería invariante.

La prueba de la hipótesis es entonces: *¿le gana una cabeza sensible al orden a la invariante,
sobre los mismos folds y con todo lo demás igual?*

## Resultados

| Cabeza | | Entrenables | F1 macro sin aum. | F1 macro con aum. | Δ |
|---|---|---:|---:|---:|---:|
| `promedio` | invariante al orden | 33 027 | 0,584 ± 0,097 | 0,578 ± 0,081 | −0,005 |
| `gru` | sensible al orden | 42 339 | 0,520 ± 0,039 | 0,513 ± 0,038 | −0,006 |
| `atencion` | sensible al orden | 34 115 | **0,599 ± 0,118** | 0,599 ± 0,063 | +0,000 |

Referencias fijas: línea de base de un frame **0,626 ± 0,065**; criterio de aceptación declarado
**0,691**.

## Tres conclusiones, y ninguna es la que se esperaba

### 1. La aumentación no mejora nada

−0,005, −0,006 y +0,000, con desvíos entre folds de 0,04 a 0,12. **No hay efecto.**

Se aplicó la aumentación temporalmente coherente de la rama `hector` —parámetros sorteados una vez
por clip y aplicados idénticos a sus 16 frames— con 20 vistas por clip, llevando el conjunto de
entrenamiento de 106 a 2 120 muestras. Sobre las vistas aumentadas sólo se entrena; el test se
evalúa siempre sobre la vista limpia.

El resultado **confirma empíricamente lo que el diseño ya sostenía**: la aumentación es un
regularizador y no una mejora del dataset. No crea ejemplares nuevos, y con 73 ejemplares el
cuello de botella son los ejemplares. Veinte vistas del mismo clip siguen siendo el mismo caballo,
la misma toma y la misma expresión.

### 2. Ninguna cabeza de clip le gana a la línea de base de un frame

La mejor (`atencion`, 0,599) queda por debajo de 0,626, y muy por debajo del criterio de 0,691.

**Y hay una explicación que no es "el modelo temporal no sirve":** la línea de base entrena sobre
**frames** —106 × 16 = 1 696 muestras— y después vota. Las cabezas de clip entrenan sobre
**clips**: 106 muestras. La línea de base tiene 16 veces más muestras de entrenamiento, y con este
tamaño de dataset eso pesa más que cualquier ventaja arquitectónica.

Es la restricción del diseño apareciendo en su forma más concreta: **con 141 clips, un modelo a
nivel de clip está hambriento de muestras por construcción.**

### 3. Las cabezas memorizan casi de inmediato

Diagnóstico sobre el conjunto de **entrenamiento**, sin tocar el de test: las tres llegan a
exactitud 1,00 y pérdida 0,000 antes del paso 80, con 33–42 K parámetros contra 106 clips.

Es lo que motivó probar la aumentación, y es también por qué la aumentación no alcanzó: el
problema no es falta de variedad fotométrica, es falta de ejemplares.

## La prueba del orden — el experimento que aísla la hipótesis

La comparación anterior tiene un confound: la línea de base entrena sobre 1 696 frames y las
cabezas de clip sobre 106 clips. Para separar *el orden* del *tamaño del conjunto*, se parte cada
clip en sus **9 ventanas contiguas de 8 frames** —954 muestras de entrenamiento, comparables a las
de la línea de base— y se entrena dos veces con todo lo demás idéntico:

- **orden real**: los frames de cada ventana en su secuencia;
- **permutado**: los mismos frames de la misma ventana, barajados.

Verificado que la permutación se aplica de verdad: cambia el orden, conserva el contenido y no
toca el tensor original. En test se promedian las probabilidades de las 9 ventanas de cada clip.

| Cabeza | Muestras ent. | Orden real | Permutado | Δ |
|---|---:|---:|---:|---:|
| `atencion` | 954 | **0,632 ± 0,155** | 0,632 ± 0,155 | **+0,000** |
| `gru` | 954 | 0,585 ± 0,114 | 0,619 ± 0,151 | **−0,034** |

**La atención da predicciones idénticas con los frames barajados.** No es un empate estadístico:
es el mismo número hasta el tercer decimal, y también el mismo recall de `Disgustado`. El modelo
**puede** usar el orden y **no lo usa**.

El mecanismo, medido sobre el modelo entrenado:

- el embedding de posición aprendido tiene norma 1,07 contra 10,74 de las características
  proyectadas: la posición aporta el **3,5 %** de la representación;
- los pesos de atención por posición van de 0,114 a 0,154, con desvío 0,018 sobre un uniforme de
  0,125. **La atención aprendió a promediar.**

Y la GRU, que es sensible al orden por construcción, rinde **peor** con el orden real que con los
frames barajados. Con 954 muestras y un desvío entre folds de 0,11, esa diferencia es ruido — pero
lo que no hay, en ninguna de las dos, es señal a favor del orden.

## Reducción de capacidad

Las cabezas de 33 K parámetros memorizan 106 clips antes del paso 80, así que se repitió todo con
la proyección bajada de 64 a 8 dimensiones: **~4–8 K parámetros**, un orden de magnitud menos.

| Cabeza | Entrenables | F1 macro (orden real) | Permutado | Δ orden |
|---|---:|---:|---:|---:|
| `promedio` 64 | 33 027 | 0,574 ± 0,099 | 0,574 | +0,000 |
| `atencion` 64 | 33 603 | **0,632 ± 0,155** | 0,632 | +0,000 |
| `gru` 64 | 42 339 | 0,585 ± 0,114 | 0,619 | −0,034 |
| `promedio` 8 | 4 131 | 0,587 ± 0,078 | 0,587 | +0,000 |
| `atencion` 8 | 4 203 | 0,580 ± 0,098 | 0,580 | +0,000 |
| `gru` 8 | 8 235 | 0,605 ± 0,072 | 0,613 | −0,008 |

**Bajar la capacidad no mejora la media**, y el efecto es mixto: `promedio` +0,013, `gru` +0,020,
`atencion` −0,052. Lo que sí hace, de forma consistente, es **bajar la varianza entre folds** —de
0,099–0,155 a 0,072–0,098—, que es el efecto de regularización esperado. Un modelo más estable
pero no mejor.

### Una validación del arnés que salió gratis

`promedio` se incluyó en la prueba del orden aunque sea **demostrablemente** invariante: promediar
no puede depender del orden. Que dé exactamente el mismo número en las dos columnas, en los dos
regímenes de capacidad, confirma que el arnés de permutación funciona.

Eso importa porque es lo que permite afirmar que el **+0,000 de la atención es un resultado y no
un bug**: si la permutación no se estuviera aplicando, `promedio` también daría +0,000 —y lo hace—
pero la verificación directa del tensor permutado (contenido conservado, orden cambiado) descarta
esa explicación.

## Estado y qué haría falta

**La hipótesis del trabajo no queda sostenida por esta evidencia.** Se informa así, que es lo
que el diseño exige, y se sostiene sobre tres pruebas independientes que dan lo mismo:

1. **A nivel de clip**, la cabeza invariante al orden (0,584) y las sensibles al orden (0,599 y
   0,520) no se distinguen más allá del desvío entre folds.
2. **Con muestras igualadas**, permutar los frames no cambia una sola predicción de la atención en
   ninguno de los dos regímenes de capacidad, y la GRU rinde peor con el orden real en los dos.
3. **Bajar la capacidad un orden de magnitud no cambia la conclusión.** Baja la varianza, no sube
   la media, y el orden sigue sin aportar.
4. **Ningún modelo temporal le gana a la línea de base.** El mejor (0,632 ± 0,155) apenas supera a
   la línea de base que ignora el orden por completo (0,626 ± 0,065), con un desvío entre folds
   más del doble de grande — o sea que ni siquiera es distinguible de ella. Y queda lejos del
   criterio declarado (0,691).

**El mejor modelo sigue siendo la línea de base de un frame: 0,626 ± 0,065.** Es también el más
simple y el más estable de todos los probados.

**Qué NO dice este resultado.** No dice que la información temporal sea inútil en el problema.
Dice que **con 141 clips de 73 ejemplares no se pudo extraer señal temporal utilizable**, y que
los modelos que podían usar el orden aprendieron a ignorarlo. La respuesta del diseño a esto
estaba escrita de antemano y se confirma: lo que haría falta es **más ejemplares**, no más
arquitectura ni más aumentación.

**Lo que sí quedó demostrado, y es un resultado positivo:** las imágenes tienen señal. De 0,188
(clase mayoritaria) a 0,626–0,632 con partición honesta por ejemplar. El preprocesamiento facial
del objetivo 2 —recorte, alineación por el eje ojo-hocico, encuadre que incluye las orejas—
produce una entrada de la que un clasificador lineal sobre características congeladas extrae
bastante más que las clases a priori.

**Lo que falta para cerrar el objetivo 3:** entrenar el modelo elegido y reportar (3.4, 4.1–4.3),
la ablación por `origen_alineacion` (4.4) y el análisis de errores por ejemplar (4.5). Queda una
variante razonable sin probar —cabezas de ~1 K parámetros, dado que las de 33 K memorizan— y
conviene decidir de antemano si se prueba, porque suma al conteo de configuraciones.
