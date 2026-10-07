# Experimento comparativo: geometría y movimiento facial sin red visual profunda

Pregunta: ¿cuánta información discriminativa existe exclusivamente en la geometría y el movimiento
facial/cefálico del caballo, sin una red visual profunda? Es un experimento ADICIONAL: no modifica
`equinos/` ni el MIL vigente. Todo vive en `experimentos/simba/`.

**Reproducir:** `bash experimentos/simba/correr_todo.sh` (lee `datos_equinos/` del repo principal en solo
lectura; ruta alternativa con `DATOS_EQUINOS`). Resultados en `resultados/` (`resumen.json` = las 12
preguntas del §72).

## Nombres: qué es y qué no es SimBA
- DLC = video -> landmarks. SimBA = landmarks -> features -> clasificador. Ninguno hace ambas cosas.
- **Conjunto principal («features tipo SimBA calculadas con código propio», `features_simba.py`) NO es SimBA.**
  Se implementó porque el extractor real de SimBA para un solo animal no produce features faciales.
- SimBA real (`simba-uw-tf-dev 5.6.1`, venv `~/.venvs/simba`) SÍ se instaló y ejecutó sobre los 359 clips
  (`simba_real_intento.py`): `UserDefinedFeatureExtractor` solo entrega movimiento por bodypart, su
  resumen y ventanas móviles (59 columnas por frame), sin distancias, ángulos ni nada facial. Se agregó por clip
  (174 columnas, conjunto B2). Para importarlo hizo falta un stub vacío de `tkinter` (no instalado, sin sudo).
- Compatibilidad DLC->SimBA: `resultados/compatibilidad_dlc_simba.json`. Resumen: pose en formato largo
  (clip, orden, bodypart, x, y, confianza), píxeles del cuadro original, sin normalizar; importación directa NO
  (hay que pasar a formato ancho, crear proyecto, `video_info.csv`; px/mm sin calibrar = 1).

## Datos y protocolo
- Unidad: clip. Mismos 359 clips, 133 caballos y 4 folds por caballo que el MIL vigente
  (`particion_clips.json`). `evaluar.py` verifica con asserts: fold == `particion.construir`, tamaños de test
  (90/90/90/89) y nº de caballos de test (33/33/34/33) iguales a los de `objetivo3_evaluacion_v2.json`, clase y
  caballo de cada clip iguales en las predicciones del MIL, y ningún caballo en train y test.
- Pose: los 16 frames por clip que ve el MIL (`pose_seleccionada_v2.csv`, 39 bodyparts). Keypoint con
  confianza < 0,50 = ausente (NaN). Separación entre frames consecutivos: mediana 0,33 s (0,1-0,5 s).
- Normalización dentro de la muestra (escala = mediana del clip de la distancia ojo-hocico). Imputación por
  mediana ajustada solo con el train del fold (+ 9 features de disponibilidad).
- RF: 500 árboles, `max_features=sqrt`, `min_samples_leaf=2`, `class_weight=balanced`, semilla 0. LogReg
  como el baseline vigente (C fijo a priori). Sin búsqueda de hiperparámetros.
- F1 macro: media ± desvío (n=4 folds) como en el pipeline. En bootstrap se usa F1 agrupado por réplica (otra
  cantidad; ver `evaluacion.json`).
- Subconjunto limpio: 1080x1920 y >= 5 s (236 clips: 28 Disgustado, 80 Interesado, 128 Sereno). Los modelos se
  entrenan con todos los clips del train y se evalúan solo en los limpios (como la auditoría 25/09); además se
  reporta el reentrenado solo con limpios (no para el MIL: no se reentrenó).
- Los controles `K1` (filmación) y «sin dinámica de cabeza» se agregaron DESPUÉS de ver el SHAP (post hoc);
  el resto de las ablaciones estaban planificadas.

## Resultados principales (F1 macro, media ± desvío entre 4 folds; original = 359 clips)
Ver `resumen.json`, `evaluacion.json` y `comparacion_modelos.png`. Resumen:
| Modelo | Original | Limpio (media de folds) | Limpio (agrupado) |
|---|---|---|---|
| Mayoritaria | 0,183 ± 0,033 | 0,231 | 0,234 |
| A. 9 geom. actuales (RF / LR) | 0,694 ± 0,044 / 0,607 ± 0,033 | 0,550 / 0,552 | 0,603 / 0,580 |
| B. Features tipo SimBA propias, RF (principal) | 0,851 ± 0,050 | 0,723 ± 0,033 | 0,750 |
| B. ídem, LogReg | 0,832 ± 0,037 | 0,747 | 0,769 |
| B2. SimBA real (movimiento), LR | 0,720 ± 0,066 | 0,656 | 0,679 |
| C. Baseline estático (LogReg media ResNet) | 0,741 ± 0,020 | 0,630 | 0,671 |
| D. Visual + 9 geom., LR | 0,747 ± 0,025 | 0,640 | 0,673 |
| E. MIL (artefactos, no reentrenado) | 0,740 ± 0,052 | 0,670 | 0,705 |
| F. SimBA propio + visual, RF | 0,844 ± 0,071 | 0,729 | 0,751 |
Los números del MIL (0,740), del baseline (0,741) y del limpio (0,705) coinciden con los de la auditoría.

Lectura cuidadosa:
- En el conjunto original SimBA-RF supera al baseline y al MIL (diferencia de F1 agrupado +0,106 [0,037; 0,179] y
  +0,110 [0,042; 0,181], IC95 por caballo). En el limpio la diferencia se achica (+0,083 [-0,018; 0,202] y
  +0,048 [-0,056; 0,169]): no concluyente.
- Control de filmación (duración, fps, resolución, ventana, escala de cabeza, dt): 0,58-0,62; en el limpio 0,40.
- Las velocidades de cabeza lideran el SHAP, y correlacionan con la duración del clip (Spearman -0,53; el
  jitter de keypoints dividido por un dt corto infla la velocidad). Pero quitar la dinámica de cabeza casi no
  baja el F1 (0,848) y las features estáticas solas dan 0,826: la ventaja no depende de ese artefacto. Aun en el
  limpio Disgustado tiene mayor velocidad de cabeza mediana (0,82 vs 0,17-0,29), lo que puede ser movimiento
  real, movimiento de cámara o punto de vista (no se puede separar con estos datos).
- Reentrenar solo con clips limpios baja a 0,678 (SimBA-RF) y 0,606 (baseline): con ~236 clips y pocos
  Disgustado, la estimación es mucho más ruidosa.
- Sobreajuste (F1 train - test): RF 0,143, LogReg SimBA 0,148, baseline 0,259, MIL 0,240. El RF memoriza el
  train (F1 ~0,99); su gap train-validación interna (0,337, validación de ~7 caballos) no es menor que el del MIL (0,267).

## Limitaciones
- Pose 2D monocular: ángulos y ratios dependen del punto de vista; el punto de vista puede asociarse a la clase.
- Un solo observador de etiquetas; 133 caballos; 3-7 clips de la clase minoritaria por fold en el limpio.
- Sin ojos abiertos/cerrados ni parpadeo posibles (un keypoint por ojo; muestreo 0,1-0,5 s). Ver
  `inventario_anatomico.json` (disponibilidad por bodypart sobre 5744 frames seleccionados: sesgada hacia ojo/hocico).
- `antler_*` no se usó (semántica en caballo no verificable con los archivos disponibles).
- SHAP: contribuciones respecto de un valor base (valor esperado del modelo); asociación, no causalidad;
  requiere validación experta. Un RF distinto por fold: las importancias son un agregado de 4 modelos.
- Parsimonia no medida: tiempo de extracción de pose (DLC, ~1 h para todo el dataset según `equinos/pose.py`) y
  de ResNet por clip; entrenamiento del MIL medido en una sola corrida (fold 0).
- La ventaja de SimBA-RF no es una afirmación definitiva sobre generalización a caballos nuevos fuera de este conjunto.
