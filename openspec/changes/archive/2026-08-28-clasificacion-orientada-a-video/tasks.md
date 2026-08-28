# Tareas — clasificación orientada a video

Orden deliberado: **las líneas de base van antes que el modelo.** Si la de identidad predice
bien, el problema no es abordable con este material y conviene saberlo antes de invertir en
arquitectura, no después.

## 1. El conjunto de datos, y saber qué hay adentro

- [x] 1.1 Armar el cargador que arma el tensor desde `dataset_frames_alineados.csv`, tomando
      **sólo clips íntegros** y respetando el orden de `orden`. Vive en
      `equinos.caracteristicas._tensor_del_clip`, dentro del extractor: el modelo consume las
      características congeladas, así que el tensor de píxeles no se materializa aparte.
- [x] 1.2 Verificar que el frame `orden=k` del tensor es el archivo que dice el CSV.
      `verificar_orden_temporal.py`: el vector de cada archivo se parece a su propia fila
      (d≈0,006) tres órdenes de magnitud más que a cualquier otra (d≈5–9). **Es load-bearing**:
      con los frames barajados de fábrica, «permutar no cambia nada» sería trivial y la
      conclusión del bloque 3 no valdría.
- [x] 1.3 Informar la composición del conjunto: clips y ejemplares por clase, y cuántos clips
      quedaron afuera por no ser íntegros.
- [x] 1.4 Construir la partición `GroupKFold` por `id_caballo` con 4 folds y **persistirla**,
      para que todas las corridas y líneas de base usen exactamente los mismos cortes.
- [x] 1.5 Verificar que ningún ejemplar aparece en entrenamiento y test del mismo fold, y que
      ninguna clase queda vacía. Informar clips por clase en cada fold.

## 2. Líneas de base

- [x] 2.1 Clase mayoritaria, sobre la partición persistida. **F1 macro 0,188 ± 0,029.**
- [x] 2.2 **Sólo `id_caballo`**, sin mirar la imagen. Bajo el protocolo real da **exactamente lo
      mismo que la mayoritaria (0,188)**, porque con partición por ejemplar el caballo de test
      nunca se vio y cae al voto mayoritario en el 100 % de los clips: la partición neutraliza el
      confounder. Para contestar la pregunta de fondo se midió además con partición ingenua por
      clip, donde llega a **F1 macro 0,526** sin mirar un píxel — 63 de los 73 ejemplares aparecen
      en una sola clase. Resultados en `lineas_de_base.md`.
- [x] 2.3 Un solo frame del clip, sin información temporal, con el mismo backbone congelado que
      usará el modelo. **F1 macro 0,626 ± 0,065**, recall `Disgustado` 0,346. Usa los 16 frames
      pero es invariante a su orden, así que aísla exactamente lo que el modelo secuencial tiene
      que aportar. Las imágenes tienen señal: de 0,188 a 0,626 sin información temporal.
- [x] 2.4 Tabla comparativa de las tres, con F1 macro y balanced accuracy, media y desvío entre
      folds, en `lineas_de_base.md`. **Criterio declarado antes de entrenar: el modelo secuencial
      se considera útil si su F1 macro medio supera 0,691** — la línea de base más alta (0,626)
      más un desvío entre folds (0,065), según la Decisión 7 del diseño.

## 3. El modelo

- [x] 3.1 Extraer las características por frame con el backbone congelado y **cachearlas**.
      `equinos/caracteristicas.py`: ResNet-18 de ImageNet congelada, 141 × 16 × 512, 11 s en la
      RTX 3070. La caché se invalida sola si cambia el backbone o el dataset. **Ojo: con la caché
      no se puede aumentar** — si el entrenamiento final usa aumentación, tiene que re-extraer.
- [x] 3.2 Implementar la cabeza temporal sobre las características cacheadas. Tres variantes en
      `cabezas_temporales.py`, separadas por lo que el diseño no distinguía: `promedio` es
      **invariante al orden** y por lo tanto no es un modelo temporal; `gru` y `atencion` sí lo
      son. La prueba de la hipótesis es esa comparación, no sólo la de contra la línea de base.
- [x] 3.3 Comparar a lo sumo tres variantes de cabeza sobre los mismos folds, y **dejar
      registrado cuántas se probaron**. **Se probaron 18, todas informadas**: tres cabezas × dos
      regímenes de aumentación, y la prueba del orden (real/permutado) sobre las tres cabezas en
      dos regímenes de capacidad (33 K y ~4 K parámetros), y se informan las seis en `cabezas_temporales.md`. Mejor cabeza `atencion`
      0,599 ± 0,118 — **por debajo de la línea de base de un frame (0,626) y del criterio
      (0,691)**. La aumentación no mueve nada (−0,005 / −0,006 / +0,000). **Prueba del orden con
      muestras igualadas: permutar los frames no cambia una sola predicción de la
      atención (+0,000) y la GRU rinde peor con el orden real (−0,034). La hipótesis
      del trabajo no queda sostenida.**
- [x] 3.4 Entrenar la elegida con `random_state=42` y guardar los pesos por fold. **La elegida
      es el clasificador de un frame con voto** — la que se diseñó como línea de base y terminó
      ganándole a las 18 configuraciones. Pesos en `modelos/objetivo3/`.

## 4. Evaluación

- [x] 4.1 F1 macro y balanced accuracy por fold, con media y desvío. Sin accuracy global.
      **F1 macro 0,626 ± 0,065 · balanced accuracy 0,638 ± 0,069 · recall `Disgustado` 0,346.**
- [x] 4.2 Matriz de confusión agregada sobre las predicciones fuera de muestra. **La confusión
      dominante contradice lo previsto**: el eje que el modelo mezcla es `Sereno`/`Disgustado`
      (12 y 9 errores), no `Disgustado`/`Interesado` (5) como anticipaba el caso `178`.
- [x] 4.3 Contrastar contra el criterio declarado en 2.4 e informar el resultado **aunque no lo
      cumpla**. **NO LO CUMPLE: 0,626 contra 0,691, diferencia −0,065.**
- [x] 4.4 Ablación sobre los 82 clips íntegros por detección propia. **0,518 ± 0,060 contra
      0,626, y el recall de `Disgustado` cae a 0,000**: el subconjunto tiene 7 clips de esa clase
      repartidos en 4 folds. No invalida el resultado completo — mide cuánto se apoya en
      posiciones estimadas, y el resultado es un argumento a favor del relleno temporal.
- [x] 4.5 Análisis de errores por ejemplar. **Se concentran: 22 de 73 ejemplares fallan alguna
      vez y los 10 peores acumulan el 68 % de los 41 errores.** Cinco de los diez peores están
      entre los que aparecen en más de una clase, coherente con el techo del caso `178`. `13`,
      `44` y `231` fallan en el 100 % de sus clips: candidatos a revisión de etiqueta.

## 5. Documentación

- [x] 5.1 Notebook `03_clasificacion.ipynb` con la estructura de Bloques y Pasos del proyecto.
      31 celdas, 6 bloques, ejecutado entero con papermill sin errores en 4 s. Consume
      `equinos.clasificacion`, que se creó para que el notebook **no dependa del directorio del
      change**: al archivarse, un import desde `openspec/changes/...` se rompería.
- [x] 5.2 Actualizar `openspec/project.md` con el estado del objetivo 3 y los resultados.
- [x] 5.3 Dejar escrito el techo de exactitud conocido: los frames de frontera entre tomas
      contiguas de un mismo ejemplar con clases opuestas. **El análisis de errores lo confirma**:
      cinco de los diez ejemplares que más fallan aparecen en más de una clase.
- [x] 5.4 Registrar qué haría falta. **Más ejemplares**, con los números al lado, en
      `resultado_final.md`: la aumentación se probó y no movió nada, y ninguna de las tres cabezas
      en dos regímenes de capacidad superó a una regresión logística sobre características
      congeladas.
