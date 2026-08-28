# Cabezas temporales — resultados y estado

Corrida el **28/08/2026** sobre la partición persistida y las características del backbone
congelado. Reproducible con `cabezas_temporales.py`.

## Configuraciones probadas: 6

El diseño pide comparar a lo sumo tres variantes de cabeza y **dejar registrado cuántas se
probaron**, porque con 141 muestras probar veinte y quedarse con la mejor es elegir ruido. Se
probaron **tres cabezas × dos regímenes de aumentación = 6**, y se informan las seis.

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

## Estado y qué haría falta

**El resultado provisorio es negativo: la hipótesis del trabajo no queda sostenida por esta
evidencia.** Se informa así, que es lo que el diseño exige.

Antes de darlo por cerrado quedan dos cosas por probar, y las dos son **reducción de capacidad**,
no más arquitectura ni más aumentación:

1. **Cabezas mucho más chicas.** 33 K parámetros memorizan 106 clips. Una cabeza de ~1 K
   —proyección a 8 o 16 dimensiones, o directamente lineal sobre el promedio— es lo que el tamaño
   del dataset admite. La línea de base que gana es, justamente, la más simple de todas.
2. **Entrenar la cabeza temporal a nivel de frame y agregar después**, para que vea las mismas
   1 696 muestras que la línea de base. Así la comparación aislaría el orden y no el tamaño del
   conjunto de entrenamiento.

Cada una suma configuraciones al conteo, y con 141 muestras eso tiene un costo. Se decide antes de
correrlas y se registra el total.
