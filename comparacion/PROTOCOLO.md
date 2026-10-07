# Protocolo de comparación de modelos

Declarado ANTES de comparar. Versión 1, 07/10/2026. Lo implementa `equinos/comparacion.py`; si algo
de este documento cambia después de ver resultados de un candidato, el cambio se registra como
enmienda con fecha al final y los resultados previos se rotulan «anteriores a la enmienda».

## 1. Pregunta

Cuál de los modelos candidatos clasifica mejor la expresión (Sereno / Interesado / Disgustado) en
**caballos nunca vistos**, sin que la ventaja pueda explicarse por rasgos de filmación.

## 2. Datos y particiones

- 359 clips íntegros de 133 caballos (el conjunto vigente). Unidad de agrupamiento: **el caballo**.
- **Reserva (decisión de Sergio, confirmada el 07/10/2026): 15 % de los caballos.** 20 caballos
  (68 clips: Disgustado 17, Interesado 16, Sereno 35), sorteados con semilla fija (20261007)
  estratificando por **clase dominante del caballo** (la de más clips; empate: la clase más rara).
  Archivo versionado: `comparacion/reserva_caballos.json` (con huella del dataset y de su contenido).
  Los 113 caballos restantes (291 clips: Disgustado 86, Interesado 102, Sereno 103) son el
  **desarrollo**.
- **La reserva no se usa en ninguna evaluación de desarrollo**: ni para elegir hiperparámetros, ni
  umbrales, ni épocas, ni arquitectura, ni para mirar curvas. El único acceso es
  `comparacion.evaluar_en_reserva`, que exige `uso_autorizado_por_sergio=True` y un motivo, deja
  cada uso en `comparacion/reserva_uso.log` y se niega a un segundo uso.
- **Particiones de desarrollo**: R = 5 repeticiones × 4 folds por caballo
  (`StratifiedGroupKFold`, `shuffle=True`, semillas 1101–1105), persistidas en
  `comparacion/particiones.json`. Son 20 evaluaciones por candidato; todos los candidatos usan
  exactamente las mismas. Se verifica por aserción que ningún caballo cruza train/test, que la
  reserva no aparece y que cada test contiene las tres clases.
- Todo ajuste del candidato (hiperparámetros, época, regularización) se hace **dentro del train de
  cada evaluación**, con una validación interna por caballo estratificada por clase
  (`particion.dividir_estratificada`, semilla fija). Está prohibido elegir con el test del fold.

## 3. Candidatos

| # | Candidato | Entrada | Simplicidad (1 = más simple) |
|---|---|---|---|
| 1 | Baseline: regresión logística sobre la media de los 16 frames (ResNet-18 congelada) | visual | 1 |
| 2 | RF de pose tipo SimBA (160 features geométricas propias) | pose | 2 |
| 3 | Fusión pose + visual | pose + visual | 3 |
| 4 | MIL multimodal corregido (geometría v2, máscaras ≥ 0,50, validación interna estratificada) | visual + pose | 4 |

Cada candidato implementa `ajustar(ids_train, datos)` y `predecir(ids, datos)` y se evalúa con
`comparacion.evaluar`. Una implementación por worktree; ninguna modifica el evaluador.

## 4. Métricas (las calcula el evaluador, no el candidato)

- **Principal: F1 macro** (`average='macro'`, las tres clases con igual peso), media ± desvío sobre
  las 20 evaluaciones; también por repetición (media ± desvío entre sus 4 folds y F1 agrupado de sus
  predicciones fuera de muestra).
- Accuracy; precisión, recall y F1 por clase; matrices de confusión (suma de las 20).
- **Subconjunto limpio**: clips 1080×1920 y de al menos 5 s, que no delatan la clase por el formato
  de filmación. Se mide el F1 macro de las predicciones fuera de muestra de cada repetición
  restringidas a ese subconjunto (media ± desvío entre las 5 repeticiones).
- **Gap train−test** (F1 macro de entrenamiento menos F1 macro de test) como indicador de
  sobreajuste.
- **IC95 bootstrap pareado por caballo** de la diferencia entre dos candidatos
  (`comparacion.comparar`): 2 000 réplicas que remuestrean caballos de desarrollo con reposición y
  arrastran todos sus clips; en cada réplica se calcula el F1 macro de ambos candidatos (promedio de
  las 5 repeticiones) y la diferencia. Se informa en todo el desarrollo y en el subconjunto limpio.

## 5. Criterio de elección (fijado antes de comparar)

1. **Ranking**: el candidato con mayor **F1 macro medio sobre las 5×4 evaluaciones de desarrollo**.
2. **Condición de no inferioridad en el subconjunto limpio**: el candidato elegido no puede ser
   *inferior* a ningún otro en el subconjunto limpio. «Inferior» = el límite superior del IC95
   bootstrap pareado de `F1_limpio(candidato) − F1_limpio(otro)` es menor que 0 (el candidato es
   significativamente peor en limpio). Si el líder incumple la condición, pasa a evaluarse el
   siguiente del ranking.
3. **Empate**: se considera empate con el líder a todo candidato cuyo IC95 pareado de la diferencia
   en desarrollo contra el líder incluye 0. Entre los empatados gana **el más simple** (columna de la
   tabla del §3).
4. Se registra el resultado completo aunque haya ganador claro; no se descarta ningún candidato por
   razones distintas a las de este párrafo.

## 6. Uso de la reserva

La reserva se usa **UNA sola vez, con el ganador**, después de fijar el ganador y de registrar la
decisión por escrito. El modelo se ajusta con todo el desarrollo con los hiperparámetros ya elegidos
y se mide sobre los 20 caballos. Ese número es la estimación final fuera de muestra; no se vuelve a
tocar el modelo después. Con 20 caballos y 68 clips el intervalo será ancho: se informa con su IC
bootstrap por caballo y sin interpretarlo como comparación entre candidatos. **Confirmado por
Sergio el 07/10/2026** (15 %, una sola vez, con el ganador).

## 7. Límites declarados

- El agrupamiento es por caballo, no por sesión: caballos de una misma sesión pueden quedar de
  lados distintos (ver `sesiones_de_filmacion.csv`, a completar, para un futuro leave-session-out).
- Un solo observador etiquetó los clips; no hay acuerdo interobservador medido.
- Las 5 repeticiones remuestrean la partición, no los datos: el desvío entre ellas no es un error
  estándar. Para la incertidumbre se usa el bootstrap por caballo.
- 22 caballos aparecen en dos clases y 7 en las tres; por eso la estratificación del caballo es por
  clase dominante, y la de los folds, por clip.
