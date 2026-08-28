# Tareas — clasificación orientada a video

Orden deliberado: **las líneas de base van antes que el modelo.** Si la de identidad predice
bien, el problema no es abordable con este material y conviene saberlo antes de invertir en
arquitectura, no después.

## 1. El conjunto de datos, y saber qué hay adentro

- [ ] 1.1 Armar el cargador que arma el tensor `(N, 16, 343, 224, 3)` desde
      `dataset_frames_alineados.csv`, tomando **sólo clips íntegros** y respetando el orden de
      `orden` dentro de cada clip.
- [ ] 1.2 Verificar contra un caso conocido que el frame `orden=k` del tensor es el archivo que
      dice el CSV. Un desorden temporal no rompe nada y arruina todo modelo secuencial.
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
- [ ] 3.2 Implementar la cabeza temporal sobre las características cacheadas.
- [ ] 3.3 Comparar a lo sumo tres variantes de cabeza sobre los mismos folds, y **dejar
      registrado cuántas se probaron**. Con 141 muestras, probar veinte y quedarse con la mejor
      es elegir ruido.
- [ ] 3.4 Entrenar la elegida con `random_state=42` y guardar los pesos por fold.

## 4. Evaluación

- [ ] 4.1 F1 macro y balanced accuracy por fold, con media y desvío. Sin accuracy global.
- [ ] 4.2 Matriz de confusión agregada sobre las predicciones fuera de muestra.
- [ ] 4.3 Contrastar contra el criterio declarado en 2.4 e informar el resultado **aunque no lo
      cumpla**.
- [ ] 4.4 Ablación: repetir 3.4 y 4.1 sobre los 82 clips íntegros por detección propia
      (`origen_alineacion = keypoints`) e informar las dos corridas lado a lado.
- [ ] 4.5 Análisis de errores por ejemplar: si los fallos se concentran en pocos caballos,
      decirlo — es información sobre el dataset, no sobre el modelo.

## 5. Documentación

- [ ] 5.1 Notebook `03_clasificacion.ipynb` con la estructura de Bloques y Pasos del proyecto.
- [ ] 5.2 Actualizar `openspec/project.md` con el estado del objetivo 3 y los resultados.
- [ ] 5.3 Dejar escrito el techo de exactitud conocido: los frames de frontera entre tomas
      contiguas de un mismo ejemplar con clases opuestas —el caso `178_20260701-081313`— son
      ruido irreducible, medido en `orientacion-de-video-en-origen`.
- [ ] 5.4 Registrar qué haría falta para mejorar el resultado si no alcanza. La respuesta
      esperada es **más ejemplares**, no más arquitectura, y conviene que quede dicho con los
      números al lado.
