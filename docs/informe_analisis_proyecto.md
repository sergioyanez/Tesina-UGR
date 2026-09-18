# Informe de análisis del proyecto de expresiones faciales equinas

Fecha de revisión: 17 de septiembre de 2026.

Análisis de solo lectura del código, notebooks, documentación, resultados locales e historial Git disponible. No se ejecutaron notebooks, entrenamientos, inferencias ni tests. La única escritura autorizada es este informe. Las celdas se numeran desde 1 contando tanto las de texto como las de código, no por el contador de ejecución.

**Hallazgo principal:** existe un sistema desarrollado de preparación, alineación, clasificación e interfaz, pero el estado actual reúne versiones diferentes. El código de la rama `feat/arquitectura-multimodal-mil` utiliza MIL multimodal; el notebook principal muestra un resultado MIL de **0,754 sobre 257 clips**; los artefactos locales de evaluación corresponden a un modelo de promedio de **0,704 sobre 153 clips**. No deben presentarse como una única corrida ni como una comparación controlada.

Para Maruca: encontramos varias recetas, varios platos ya preparados y varias libretas de resultados. Podemos explicar cada uno, pero no atribuirle a un plato la calificación que recibió otro.

## 1. Objetivo general y qué significa el problema

El objetivo es construir un sistema que reciba un video corto de un caballo y clasifique su expresión en **Sereno, Interesado o Disgustado**, con la intención de asistir a personas que evalúan su bienestar.

El plan original, de mayo de 2026, propone cuatro trabajos: reunir videos de 5–7 segundos; preparar sus imágenes y localizar la cara; entrenar y evaluar clasificadores; construir una interfaz.

No se trata de reconocer qué caballo aparece, aunque esa identidad debe registrarse para evaluar correctamente. Tampoco se entrenó un clasificador de todas las emociones posibles ni un sistema de diagnóstico de dolor: las etiquetas efectivas son esas tres.

Para Maruca: queremos que la computadora mire la cara y diga cuál de tres expresiones se parece más a lo que ve. La etiqueta que aprende proviene de ejemplos previamente clasificados por personas; no tiene acceso directo a lo que siente el animal.

**Fuentes:** `plan_de_trabajo.pdf`, páginas impresas 2–3; `openspec/project.md`, “Problema e hipótesis” y “Objetivos”; `equinos/clasificacion.py::CLASES`.

## 2. Recorrido cronológico: el proyecto no tuvo una sola versión

| Momento o línea de trabajo | Qué se hizo y qué quedó registrado |
|---|---|
| Mayo de 2026 | Se formuló el plan: clasificación de expresiones en videos cortos y expectativa de aprovechar su dinámica. |
| Primera preparación, documentada en agosto | Dataset pequeño: 34 videos, 92 clips, 1.472 frames y 22 caballos. Disgustado provenía de un solo ejemplar. |
| Agosto: persistencia y ejecución | Se incorporaron rutas comunes, cachés ZIP, checkpoints y orquestación para sobrevivir a los cortes de Colab. También ejecución local. |
| 24–25 de agosto | Se detectó una corrida sobre una copia parcial de videos. Se retiró ese espejo como fuente y se hizo explícita la selección del catálogo. |
| 25–28 de agosto | Ampliación del material; incorporación de videos cortos; corrección de orientación; conservación de pose completa; interpolación y anotación manual. |
| 28 de agosto | El recorte cuadrado se sustituyó por uno de 224×343 para conservar mejor las orejas. Se evaluaron 141 clips de 73 caballos y 18 configuraciones; se eligió el modelo por frame con voto, F1 0,626. |
| Informes intermedios | Se describen conjuntos de 142 y 147 clips y resultados 0,659 y 0,649. Son documentos distintos, con inconsistencias que se detallan después. |
| 5 de septiembre, rama de ejecución local | Evaluación sobre 153 clips de 83 caballos: promedio 0,704, GRU 0,681 y atención 0,751. Se seleccionó promedio. |
| 6–7 de septiembre, otra rama | Experimentos de clips incompletos con 12 frames y factorial de apariencia/geometría con 8 frames. No sustituyeron al modelo de promedio en esa línea de trabajo. |
| 8 de septiembre | “Vía C”: clasificación desde movimiento de cabeza. Sus resultados y controles no respaldaron la ventaja buscada. |
| 14–15 de septiembre | Una ejecución local guardada reproduce el modelo de promedio: 153 clips y F1 0,704. |
| 15 de septiembre, rama actual | Se incorporaron MIL multimodal y su integración con la interfaz. El notebook principal conserva salidas de 257 clips y F1 0,754, procedentes de un entorno Windows. |
| 15–17 de septiembre | Notebook de figuras y tres HTML explicativos. Sus contenidos no están sincronizados entre sí ni con todos los artefactos locales. |

Esto no constituye necesariamente una sucesión lineal donde cada paso reemplazó al anterior: **hay ramas de desarrollo diferentes**.

Fuentes históricas verificadas sin cambiar de rama: `git log --all`; rama `ejecucion-local-gpu`; rama `nuevo-intento-con-enmascarado-y-geometria`; commits `dd13443`, `8a3bffe` y `4c13c20`. Los archivos históricos citados de esas ramas no necesariamente existen en el árbol de trabajo actual.

## 3. Dataset original: origen, etiquetas, organización y cantidades

La fuente declarada es la carpeta compartida **“Videos de prueba” de Google Drive**. En local se configura mediante `TESINA_EQUINOS_VIDEOS`. El plan menciona bases propias e institucionales; no encontré un inventario completo que identifique procedencia institucional, responsable y protocolo de etiquetado de cada toma.

Hay videos de celular y formatos MP4, AVI, MKV, entre otros admitidos. La documentación describe abundancia de tomas verticales, distintas resoluciones, duraciones y frecuencias de captura.

Las clases se organizan en carpetas, con inclusión explícita de subcarpetas:

| Clase | Material admitido además de los archivos de su carpeta principal |
|---|---|
| Disgustado | `Seleccionar cabeza?` y `Se tapa una parte de la cara` |
| Interesado | `Seleccionar cabeza?` |
| Sereno | `Seleccionar cabeza?` |

`Sereno/Separar cabeza?` queda fuera porque no tiene decisión de etiqueta. `catalogo.clave()` permite reconocer variantes de nombres saneados al copiar entre sistemas.

El nombre suele comenzar por el identificador del caballo. `obtener_id_caballo()`, celda 26 del notebook 01, extrae ese prefijo. El código presupone que esos identificadores representan correctamente a los individuos.

**Inventario local efectivamente leído:**

| Clase | Videos inventariados | IDs de caballo en esos videos | Clips preparados | Frames extraídos |
|---|---:|---:|---:|---:|
| Disgustado | 110 | 63 | 113 | 1.808 |
| Interesado | 71 | 60 | 120 | 1.920 |
| Sereno | 93 | 54 | 160 | 2.560 |
| Total | **274** | **138 distintos** | **393** | **6.288** |

Los caballos por clase no se suman: algunos aparecen en varias clases. En los clips preparados hay **136 IDs distintos**.

La suma de la columna `frames` del inventario original es **78.839 fotogramas de video** y la de duración, aproximadamente **2.299 segundos, 38,3 minutos**. Esos son metadatos del inventario, no una nueva decodificación de todos los videos. No confundirlos con los 6.288 JPEG muestreados.

**Archivos físicos y catálogo no son lo mismo:** `videos/` contiene 196 archivos de video contando dos bajo `_retirado/`. No es una copia completa de los 274 inventariados. `datos_equinos/clips/` contiene 397 MP4, frente a 393 filas del inventario: no se debe contar todo archivo sobrante o auxiliar como una muestra adicional. Las carpetas de trabajo sí contienen 6.288 JPEG crudos y 6.288 JPEG de salida del preprocesamiento.

No hay evidencia de anotaciones EquiFACS exhaustivas, acuerdo entre observadores ni validación independiente de las etiquetas emocionales. Una etiqueta se hereda del video a todos sus clips; eso presupone que la expresión corresponde a toda la toma.

**Fuentes:** `equinos/catalogo.py::SUBCARPETAS_POR_CLASE`; `inventario_videos.csv`, `inventario_clips.csv` y `dataset_frames.csv` en `datos_equinos/resultados/`; limitaciones de `openspec/project.md`.

## 4. Preparación: del video largo a las imágenes

El notebook `01_preparar_dataset_equinos.ipynb` realiza:

1. Localización del material mediante el catálogo.
2. Inventario técnico: FPS, cantidad de frames, duración, ancho y alto.
3. Apertura considerando la orientación del video. Una toma vertical puede almacenar su rotación como metadato.
4. Extracción del identificador del caballo y normalización del nombre del archivo.
5. Segmentación en clips.
6. Extracción uniforme de imágenes.
7. Guardado de CSV, clips y caché de frames.

Parámetros principales: `DURACION_CLIP=6.0`, mínimo ordinario 5 s, máximo 7 s, `SOLAPAMIENTO=0.0`, `PISO_VIDEO_CORTO=1.0`, `NUM_FRAMES=16`.

Un video demasiado corto para las ventanas normales puede entrar entero desde un segundo. La razón registrada es que excluir videos cortos eliminaba muchos caballos de Disgustado, cuya duración era menor.

`calcular_ventanas()` absorbe restos cortos hasta el máximo permitido. Aunque el solapamiento configurado es cero, **el algoritmo puede generar solapamiento al extender una ventana y avanzar después un paso fijo de 6 s**. Por ejemplo, con suficiente material restante, una ventana extendida a 7 s puede coexistir con otra que comienza en el segundo 6. No se ejecutó una nueva segmentación para medir cuántos casos concretos hay.

La celda 35 usa `np.linspace`: reparte los frames por toda la duración. Actualmente rechaza videos con menos de 12 frames y extrae como máximo 16, sin repetir índices por pedir más que los disponibles.

**No es selección automática de “los momentos más expresivos”**. Es muestreo uniforme: como sacar fotos a intervalos regulares.

## 5. Limpieza, selección y qué se descarta

La limpieza comprende varias decisiones diferentes:

- Excluir carpetas sin etiqueta admitida.
- Retirar material con conflicto de etiqueta.
- Corregir orientación.
- Rechazar entradas insuficientes o ilegibles.
- Seleccionar imágenes con puntos faciales utilizables.
- Registrar si la alineación vino de detección, interpolación o intervención humana.

El caso `501_8v1_LY1ucwR9.avi` estaba duplicado en Disgustado e Interesado y fue retirado. Un estudio posterior documentó búsqueda de duplicados sobre 249 videos. Eso no certifica automáticamente todos los archivos añadidos después.

El caso `178` tiene tomas contiguas con etiquetas distintas y fotogramas muy similares cerca de la frontera. El documento lo interpreta como transición de expresión; esa interpretación no reemplaza una evaluación independiente del etiquetado.

`docs/analisis_y_tareas.md` propone `imagededup` y PHash. **No encontré esa deduplicación perceptual integrada en el pipeline actual.** Evitar repetir un índice de video no elimina imágenes distintas que se ven casi iguales.

Hay dos reglas de selección que deben separarse:

| Regla aplicada al CSV local | Clips | Caballos | Disgustado / Interesado / Sereno |
|---|---:|---:|---|
| Los 16 frames alineados | **153** | **83** | 33 / 56 / 64 |
| Entre 12 y 16 alineados | **254** | **111** | 63 / 91 / 100 |

La primera coincide con `particion_clips.json` local. La segunda coincide con la condición actual de `particion.tabla_clips()`, reconstruida leyendo el CSV, sin ejecutar esa función.

El notebook MIL guardado muestra **257 clips y 113 caballos**, con 62 / 93 / 102. No corresponde exactamente a ninguna de esas dos composiciones locales.

**Fuentes:** notebook 01, celdas 26 y 35; `equinos/particion.py::tabla_clips`; `openspec/changes/archive/2026-08-28-orientacion-de-video-en-origen/duplicados.md`.

## 6. Detección del caballo y recorte de su cara

Primero se usa **YOLO11x preentrenado**, restringido a la clase `horse`. El umbral es `CONF_MINIMA_DETECCION=0.50`. Si aparecen varias detecciones, se conserva la de mayor confianza.

**YOLO detecta al caballo; la cara se obtiene después mediante puntos anatómicos.** No es correcto describirlo como un detector facial entrenado específicamente en este proyecto.

Las cajas se suavizan por clip con mediana móvil de ventana 5 y relleno de huecos. No se mezclan clips al suavizar.

Después se usa **DeepLabCut SuperAnimal-Quadruped**, con `hrnet_w32` y detector `fasterrcnn_resnet50_fpn_v2`, para estimar la pose. No encontré entrenamiento local de YOLO o DeepLabCut para estas tres expresiones: se aprovechan modelos preentrenados.

Se guarda la pose completa. El CSV local tiene **245.232 filas = 6.288 frames × 39 puntos corporales**.

Para escoger el animal en la salida de pose, `pose.elegir_individuo()` usa la confianza media. Para alinear, `pose.seleccionar_par()` obtiene un ojo y un punto del hocico. El ojo se elige por confianza porque en perfil puede verse solo uno.

Con un par válido se transforma y recorta la cara. Sin él puede generarse una salida basada en la caja del animal, marcada como **no alineada**; esa salida no equivale a una cara normalizada.

**Fuentes:** notebook 02, celdas 5, 17, 19, 21, 26–33 y 38; `equinos/pose.py`.

## 7. Orejas, región orbital, hocico y otras regiones

**El pipeline principal no produce tres imágenes independientes —orejas, órbita y hocico— que luego clasifica por separado.** Produce un recorte facial completo que intenta incluirlas.

- **Región orbital:** se representa mediante puntos del ojo, utilizados como referencia geométrica. No se observó una máscara independiente de toda la región orbital.
- **Hocico:** se elige entre candidatos como `nose`, `snout`, `muzzle`, `upper_jaw` y `mouth`. En el notebook 02, `BODYPART_HOCICO=None` significa primer candidato disponible, no necesariamente el más confiable.
- **Orejas:** existen puntos como `left_earbase`, `left_earend` y sus equivalentes derechos. Sirven para medir si la oreja queda dentro del recorte.
- **Pose completa:** conserva puntos adicionales para experimentar sin repetir la inferencia costosa.

El cambio de encuadre de agosto midió la visibilidad del **vector base→punta de la oreja**, que pasó de 4,3 % a 77,2 % en el conjunto analizado entonces. No es accuracy de clasificación ni prueba de que todas las orejas actuales estén correctamente detectadas.

En el MIL actual, `MultimodalEquineDataset._mejor_de()` escoge **un único punto de oreja**, el de mayor confianza entre coincidencias de base y punta. Por tanto, su vector de entrada no contiene explícitamente ambas puntas, ambas bases ni un ángulo auricular completo.

Otra rama sí experimentó con seis descriptores: seno y coseno del ángulo de oreja, largo de oreja, apertura de boca, ancho de boca y distancia interocular. Es una representación diferente de los nueve valores del MIL actual.

## 8. Normalización: poner las fotografías en condiciones comparables

Hay tres normalizaciones distintas.

**Geométrica de la imagen.** Con ojo y hocico se calcula una transformación de similitud: gira, traslada y escala uniformemente. Ambos puntos quedan en posiciones canónicas.

Para Maruca: es como acomodar todas las fotos en un álbum para que las caras queden al mismo tamaño y orientadas igual. Así cuesta menos comparar la expresión.

El recorte final mide **224 de ancho × 343 de alto**. Su campo se define usando la distancia ojo–hocico como unidad:

- 1,00 por encima del ojo.
- 0,30 por debajo del hocico.
- 0,75 hacia cada lado.

La transformación principal preserva proporciones; ser rectangular no implica estirar la cabeza. El relleno fuera de imagen usa `BORDER_REPLICATE`, repitiendo el borde. `relleno_fraccion` registra cuánto campo se completó artificialmente. Eso no reconstruye anatomía invisible.

**Normalización de color para ResNet.** Se convierte a RGB, se divide por 255 y se normaliza por canal con medias `(0.485, 0.456, 0.406)` y desvíos `(0.229, 0.224, 0.225)`, correspondientes a los pesos preentrenados utilizados.

**Normalización de coordenadas en MIL.** Ojo, hocico y oreja se expresan respecto del punto medio ojo–hocico y se dividen por su distancia. Son seis coordenadas y tres indicadores de validez.

Esta última normalización **no rota las coordenadas al eje canónico**. Tampoco aplica el umbral 0,50 del recorte: el dataset MIL acepta confianza mayor que cero. Los indicadores distinguen presencia/validez según esa regla, no distinguen automáticamente “manual”, “interpolado” y “detectado”.

**Fuentes:** `similitud_desde_dos_puntos()`, notebook 02 celda 12; celda 38; `equinos/encuadre.py::destino`; `equinos/caracteristicas.py::_tensor_del_clip`; `MultimodalEquineDataset.__getitem__`.

## 9. Interpolación y anotación manual

Cuando falta un par facial entre detecciones válidas, se puede estimar su posición interpolando.

La configuración actual admite **un hueco interior de un frame**: `MAX_HUECO_INTERPOLADO=1`. No prolonga posiciones hacia los extremos: `MAX_EXTREMO_SOSTENIDO=0`.

La justificación guardada comparó estimaciones con posiciones detectadas ocultadas para la prueba. Un hueco de uno dio mediana de error 13,99 píxeles frente a 16,28 de una referencia de desplazamiento; dos dieron 16,33. Es una validación geométrica local, no una medición nueva de este análisis.

También existe anotación manual del par ojo–hocico mediante `equinos/anotacion.py` y un HTML de anotación.

El CSV local de salida registra:

| Origen | Frames |
|---|---:|
| Keypoints detectados | 4.326 |
| Interpolados | 296 |
| Manuales | 26 |
| Caja, sin alinear | 1.640 |
| Total | **6.288** |

Hay **4.648 alineados, 73,9 %** del total. Detectados directamente son **68,8 %**.

**Contradicción de trazabilidad:** el `keypoints_manuales.csv` actualmente presente tiene **14 filas**, mientras que el resultado atribuye 26 frames a anotación manual. No se puede reconstruir toda esa intervención humana únicamente desde ese archivo actual.

## 10. Secuencias temporales: qué recibe realmente el modelo

Un clip es un pequeño video; un frame es una foto de ese video.

La representación histórica de imagen es `(16, 3, 343, 224)`: 16 fotografías, tres canales de color, alto y ancho. ResNet transforma cada foto en 512 números. Un clip queda representado por `(16, 512)`.

La caché local actual fue comprobada leyendo su encabezado: **`(153, 16, 512)`, float32**. No es una caché variable aunque el código actual ya permita guardarlas como arreglos de objetos.

MIL añade por frame nueve valores geométricos. Por lote, su diseño admite:

- Apariencia: `(B, T, 512)`.
- Geometría y validez: `(B, T, 9)`.
- Máscara de frames reales: `(B, T)`.

`collate_fn()` completa secuencias más cortas con ceros y enmascara esas posiciones. Los frames agregados como relleno no deben recibir atención.

Hay una diferencia importante entre **tener varios frames** y **usar su orden**. MIL asigna peso a cada uno y suma sus representaciones; no tiene recurrencia, posiciones temporales ni tiempos como entrada. Si se permutan conjuntamente apariencia, geometría y máscara, el resultado matemático es el mismo, salvo diferencias numéricas de redondeo.

Para Maruca: puede elegir qué fotos del sobre le parecen más útiles, pero no distingue la historia que forman al ordenarlas.

Además, los 16 frames no abarcan siempre los mismos segundos: hay clips cortos y largos. No se impone una frecuencia temporal uniforme entre todos los clips.

**Problema de implementación actual:** `particion.tabla_clips()` admite clips con 12 alineados, pero `caracteristicas.extraer()` lee **todas** las filas del clip, sin filtrar `alineado`. Por tanto, en clips incompletos puede incluir recortes de caja junto con los alineados. La máscara MIL cubre padding, no elimina automáticamente esos recortes.

## 11. Modelos y configuraciones comprobables

Todos los clasificadores buscan una de tres clases. YOLO y DeepLabCut son auxiliares de localización, no competidores directos del clasificador emocional.

| Modelo | Arquitectura, entrada y salida | Entrenamiento principal |
|---|---|---|
| Mayoritaria | Devuelve la clase más frecuente del entrenamiento. Sin imagen. | Conteo; sin pérdida ni optimizador. |
| Identidad | Clase más frecuente del caballo si se vio antes; si no, mayoritaria. | Conteo; control de fuga. |
| Frame + voto | ResNet-18 congelada → 512 valores → StandardScaler → regresión logística por frame → voto por clip. | `LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)`; pérdida logística regularizada. Los parámetros no explícitos quedan a los valores predeterminados de la biblioteca. |
| Promedio | Proyección 512→64, ReLU, promedio sobre frames, dropout y salida 64→3. | Entropía cruzada ponderada; AdamW; 400 pasos; lote 64; lr 0,001; weight decay 0,01; dropout 0,3; semilla 42. 33.027 parámetros entrenables. |
| GRU | Proyección 512→64, GRU de 32 unidades y salida 32→3. | Mismo régimen de las cabezas anteriores. 42.339 parámetros. |
| Atención histórica | Proyección 512→64, embedding de posición, consulta aprendida, suma ponderada y salida de tres clases. | Mismo régimen. 34.115 parámetros con 16 posiciones; cambia con la longitud. |
| Cabezas pequeñas | Proyección reducida a 8; variantes promedio, GRU y atención. | Mismo presupuesto general; aproximadamente 4.000–8.000 parámetros. |
| Dispersión, rama histórica | Cabeza invariante que conserva información de variación entre frames, además de su nivel medio. | Arnés histórico de 400 pasos, entropía cruzada ponderada y AdamW. Usada con entradas de 512, 6 y 518 dimensiones. |
| MIL multimodal actual | ResNet congelada; geometría 9→64→128; concatenación de 640; atención 640→128→1; suma ponderada; clasificador 640→64→3. | Entropía cruzada con pesos por clase; **Adam**, lr 0,001, weight decay 0,0001; batch 32; **50 épocas**; semilla 42. Sin dropout ni early stopping en esta implementación. |

MIL produce tres puntuaciones, llamadas logits. En inferencia se aplica softmax para presentarlas como probabilidades y se escoge la mayor. Estas probabilidades no equivalen automáticamente a confiabilidad calibrada.

Los pesos por clase de MIL son `N / (3 × cantidad_de_la_clase)`, calculados con el entrenamiento de cada fold. Se usa `shuffle=True`; **no hay WeightedRandomSampler en el entrenamiento MIL actual**.

Las métricas comunes de evaluación son F1 macro, balanced accuracy y recall de Disgustado. La matriz y la precisión por clase se obtienen de las predicciones fuera de muestra.

**Fuentes:** `equinos/clasificacion.py`; `equinos/caracteristicas.py`; `openspec/changes/archive/2026-08-28-clasificacion-orientada-a-video/cabezas_temporales.py`; factorial histórico del commit `dd13443`.

**CNN-LSTM documentada, no modelo actual.** `docs/informe_etapa_03_modelado_y_entrenamiento.md` describe ResNet-18 parcialmente descongelada, LSTM de 128 unidades, dropout 0,3 y salida de tres clases, con imágenes 224×224 y 16 frames. Describe GroupShuffleSplit 78/22, aumentación coherente y WeightedRandomSampler. Informa accuracy 79,4 %, F1 macro 0,72 y recall de Disgustado 0,64. No encontré ese modelo seleccionado ni `mejor_cnn_lstm.pt` entre los artefactos actuales revisados. No hay detalle verificable suficiente en ese informe para completar su optimizador, tasa y épocas sin inventarlos.

**Vía C, modelos de movimiento.** El código del commit `4c13c20`, `via_c.py`, usa entradas de cinco variables por frame y 13 frames: posición relativa, escala y seno/coseno del ángulo de cabeza. Incluye regresión logística del promedio, MLP 5→32→3 y red 5→16→3 por frame con voto. Las redes usan Adam, lr 0,001, weight decay 0,01, dropout 0,3, entropía cruzada ponderada y 400 actualizaciones. La variante denominada `cabeza_promedio` en ese script realmente termina agregando por voto: su nombre y su descripción no bastan para conocer su ejecución. También hay regresiones de duración y duración más movimiento.

## 12. Validación por caballo y GroupKFold, explicado para Maruca

Supongamos que Maruca le enseña cien fotos del mismo caballo a alguien y después lo examina con otra foto de ese caballo. Puede haber aprendido sus manchas, no su expresión.

Por eso se mantienen **todos los clips de cada caballo en el mismo grupo**. En cada vuelta se entrena con tres grupos y se evalúa con el cuarto. Se repite cuatro veces.

En la partición local de 153 clips hay grupos de **39, 38, 38 y 38 clips**, con **20, 21, 21 y 21 caballos**. Revisando el JSON, ningún ID aparece en dos folds.

Aunque la documentación lo llama GroupKFold, `particion.construir()` contiene **una implementación propia de reparto agrupado**, inspirada en ese esquema: ordena caballos por cantidad de clips y asigna cada uno al grupo menos cargado. No llama directamente a sklearn ni garantiza los mismos desempates.

**No es StratifiedGroupKFold.** Equilibra carga de clips; no optimiza explícitamente las proporciones de clase. La documentación que lo llama “estratificado” no coincide con ese código.

La partición es por caballo, no por pareja caballo–clase: un animal que aparece Sereno y Disgustado debe seguir completo en un solo grupo.

La evaluación descrita es validación cruzada agrupada. No encontré un test externo independiente para certificar el rendimiento final. Tampoco una validación anidada que separe completamente selección de arquitectura y evaluación final. El entrenamiento histórico usa pasos fijos y MIL épocas fijas, evitando elegir la época por el resultado del fold evaluado.

Agrupar elimina la superposición de IDs; **no demuestra por sí solo que desaparezcan todos los sesgos** de cámara, entorno, duración, etiquetas o selección de imágenes.

## 13. Métricas: qué pregunta contesta cada una

| Métrica | Explicación sencilla |
|---|---|
| Accuracy o exactitud | De todas las predicciones, cuántas acertó. Puede esconder que falla toda una clase pequeña. |
| Precision de una clase | Cuando dijo “Disgustado”, ¿cuántas veces era Disgustado? |
| Recall de una clase | De todos los Disgustado reales, ¿cuántos encontró? |
| F1 de una clase | Combina precision y recall; baja si alguna de las dos es baja. |
| F1 macro | Promedia los tres F1 dando el mismo peso a cada clase. Es la métrica principal. |
| Balanced accuracy | Promedio del recall de las clases presentes. |
| Matriz de confusión | Tabla que cruza clase real y clase predicha; muestra qué confusiones ocurren. |
| Media ± desvío entre folds | Rendimiento medio y cuánto cambia entre grupos de caballos. No es automáticamente un intervalo de confianza. |

El código usa `statistics.stdev` para el desvío muestral. F1 macro se calcula con las tres clases; balanced accuracy omite clases ausentes.

Promediar el F1 de cuatro folds no da necesariamente el mismo número que juntar todas las predicciones y calcular un solo F1. No se debe tratar esa diferencia como un error sin revisar la agregación.

Otras medidas presentes: cobertura de detección/alineación, clips admitidos, visibilidad de orejas, error geométrico, fracción de relleno, tasa de negativa, errores por caballo y pérdida de entrenamiento. Cada una mide algo diferente.

La afirmación de algunos textos “sin predecir Disgustado se alcanza 85 % sin aprender nada” es excesiva: 85 % sería alcanzable si se acertaran todas las otras clases. Predecir siempre Sereno en el ejemplo 45/40/15 da **45 %**, como comprueba `tests/test_clasificacion.py`.

## 14. Resultados históricos comparables: qué modelos se probaron

**Sobre 141 clips y 73 caballos, agosto:**

| Modelo, régimen de clip completo | F1 macro |
|---|---:|
| Mayoritaria | 0,188 ± 0,029 |
| Identidad, agrupada por caballo | 0,188 ± 0,029 |
| Identidad, división ingenua por clip | 0,526 ± 0,116 |
| Frame + voto | **0,626 ± 0,065** |
| Promedio | 0,584 ± 0,097 |
| GRU | 0,520 ± 0,039 |
| Atención con posición | 0,599 ± 0,118 |

El control de identidad mide un problema concreto: sin mirar imágenes, dividir mal permite alcanzar 0,526. No es un modelo válido para demostrar reconocimiento de expresiones.

Hubo aumentación coherente por clip: rotación ±10°, brillo y contraste 0,85–1,15 y espejo con probabilidad 0,5; veinte vistas por clip. Las cabezas dieron aproximadamente 0,578, 0,513 y 0,599, sin mejora observada frente a sus versiones sin aumentación.

En la prueba específica de orden se usaron nueve ventanas de ocho frames por clip. Atención dio **0,632 tanto ordenada como permutada**; GRU pasó de **0,585 a 0,619 al permutar**. Reducir capacidad tampoco produjo evidencia a favor del orden.

**Sobre 153 clips y 83 caballos, septiembre:**

| Modelo | F1 macro |
|---|---:|
| Mayoritaria / identidad agrupada | 0,175 ± 0,028 |
| Identidad con partición ingenua | 0,503 ± 0,076 |
| Frame + voto | 0,660 ± 0,015 |
| Promedio, modelo seleccionado de esa rama | **0,704 ± 0,044** |
| GRU | 0,681 ± 0,063 |
| Atención con posición | **0,751 ± 0,078** |

La mayor media puntual fue atención; **promedio fue el elegido**, no el máximo numérico de esa tabla. La documentación justifica esa decisión por parsimonia y falta de evidencia temporal.

La prueba de orden de esa corrida dio: promedio 0,684/0,684; atención 0,728/0,738; GRU 0,711/0,711, comparando orden real/permutado. No mostró deterioro al destruir el orden.

Las tablas de clip completo y de ventanas responden preguntas diferentes y no deben cruzarse como si fueran una sola clasificación de modelos.

**Fuentes:** `cabezas_temporales.md` del cambio archivado de agosto; `resultado_final.md` del cambio `2026-09-05-rehacer-objetivo-3-con-dataset-ampliado`, leído en `ejecucion-local-gpu`; notebook ejecutado `datos_equinos/ejecuciones/clasificacion_20260915_001121.ipynb`.

## 15. Experimentos alternativos y descartados

**Clips incompletos, rama de septiembre.** Se compararon referencia de 153×16, control de los mismos 153×12 y propuesta de 254×12:

| Cabeza | Referencia 153×16 | Control 153×12 | Propuesta 254×12 |
|---|---:|---:|---:|
| Promedio | 0,704 | 0,697 | 0,736 |
| GRU | 0,681 | 0,656 | 0,752 |
| Atención | 0,751 | 0,723 | 0,767 |
| Criterio propio | 0,6747 | 0,7005 | 0,7847 |

Ninguna alcanzó el criterio de la propuesta, aunque recuperar material mejoró las medias respecto al control. Comparar propuesta con referencia mezcla cambio de longitud y de dataset. El commit `8a3bffe` registra que el candidato no se despliega.

**Factorial de geometría: 177 clips, 93 caballos, 8 frames.** Se compararon exactamente las mismas muestras con apariencia, seis variables geométricas y ambas:

| Cabeza | Apariencia | Geometría | Ambas |
|---|---:|---:|---:|
| Promedio | 0,676 | 0,460 | 0,681 |
| Dispersión | 0,673 | 0,492 | 0,695 |
| GRU | 0,629 | 0,491 | 0,606 |
| Atención | 0,661 | 0,407 | 0,643 |

El criterio era 0,6825. Solo “ambas + dispersión” lo superó, con 0,695 ± 0,107: margen 0,012, pequeño frente a su variación. No se demostró una ventaja robusta de la geometría ni del orden en ese experimento. Esta geometría no es idéntica a la del MIL actual.

**Vía C: movimiento de cabeza, 218 clips y 104 caballos, 13 frames.**

| Variante | F1 macro |
|---|---:|
| Regresión logística sobre promedio | 0,5192 ± 0,1017 |
| MLP | 0,5416 ± 0,0798 |
| Variante llamada cabeza promedio | 0,5709 ± 0,1319 |
| Solo duración | **0,5782 ± 0,0784** |
| Duración más movimiento | 0,5753 ± 0,0715 |

`via_c_evaluacion.json` declara criterio **0,6265** y `mejor_modelo="solo_duracion"`. El commit `4c13c20` registra el cierre negativo. El HTML original todavía dice que esa vía está en curso: quedó desactualizado.

La duración resultó un predictor relevante de las etiquetas. Esto impide atribuir automáticamente toda señal de un video al gesto facial.

Otras alternativas históricas: clasificación binaria cuando Disgustado procedía de un solo caballo; recorte cuadrado; selección fija de un ojo; descarga repetida con gdown; espejo local parcial; CNN-LSTM documentada. Los respaldos `.bak`, carpetas de backups y `_retirado_501_20260825` no deben confundirse con entradas o resultados activos.

## 16. Resultados finales reales: dos conjuntos de evidencia diferentes

**A. Resultado respaldado por los artefactos locales de evaluación.**

`datos_equinos/resultados/objetivo3_evaluacion.json`, `particion_clips.json`, la caché y `promedio_produccion.json` coinciden con la ejecución local guardada del 15 de septiembre:

- Modelo: cabeza **promedio** sobre ResNet congelada.
- 153 clips, 83 caballos.
- F1 macro: **0,703737 ± 0,043548**.
- Balanced accuracy: **0,719433 ± 0,050461**.
- Recall Disgustado: **0,678968 ± 0,171695**.
- Criterio registrado: **0,6747**.
- Superación en media: aproximadamente **0,029**, menor que el desvío entre folds.

Su matriz fuera de muestra es:

| Real / Predicha | Disgustado | Interesado | Sereno | Total |
|---|---:|---:|---:|---:|
| Disgustado | 22 | 5 | 6 | 33 |
| Interesado | 3 | 48 | 5 | 56 |
| Sereno | 15 | 8 | 41 | 64 |
| Total | 40 | 61 | 52 | 153 |

Precision derivada: Disgustado **22/40=55 %**; Interesado **48/61=78,7 %**; Sereno **41/52=78,8 %**. Accuracy derivada de la matriz: **111/153=72,5 %**; se incluye como cálculo descriptivo, no como métrica principal del entrenamiento.

La ablación con 82 clips de detección propia da **F1 0,549 ± 0,059**, recall Disgustado **0**.

**B. Resultado MIL guardado en el notebook principal.**

`03_clasificacion.ipynb`, celdas 15, 22, 24, 26, 28 y 30, y `docs/informe_resultados_03.md` registran:

- Modelo ejecutado por el código de las celdas: **MIL multimodal**, aunque las etiquetas impresas dicen “un frame + voto”.
- 257 clips, 113 caballos.
- F1 macro: **0,754 ± 0,028**.
- Balanced accuracy: **0,761 ± 0,030**.
- Recall Disgustado: **0,753 ± 0,084**.
- Criterio usado: **0,691**.
- Ablación: 117 clips, F1 **0,673 ± 0,059**, recall Disgustado **0,461 ± 0,173**.
- 61 errores, repartidos entre 41 de los 113 caballos.

Su matriz guardada es:

| Real / Predicha | Disgustado | Interesado | Sereno | Total |
|---|---:|---:|---:|---:|
| Disgustado | 46 | 7 | 9 | 62 |
| Interesado | 2 | 77 | 14 | 93 |
| Sereno | 13 | 16 | 73 | 102 |
| Total | 61 | 100 | 96 | 257 |

Precision derivada: **75,4 %, 77,0 % y 76,0 %**, respectivamente. Accuracy derivada: **196/257=76,3 %**.

Este resultado **sí está presente**; no corresponde borrarlo del relato por discrepar con los JSON. Pero no encontré su partición de 257 clips y evaluación detallada sincronizadas en los artefactos locales actuales.

No es válido atribuir la diferencia entre 0,704 y 0,754 exclusivamente a MIL: cambian dataset, selección, arquitectura y régimen de entrenamiento.

## 17. Hipótesis original y conclusiones que permite la evidencia

Conviene separar cinco conceptos:

| Concepto | En este proyecto |
|---|---|
| Objetivo | Construir un sistema que clasifique expresiones desde videos. |
| Hipótesis general original | Es factible lograr clasificación automática con precisión útil para asistir a expertos. |
| Hipótesis temporal desarrollada después | El orden o evolución de los frames aporta ventaja frente a métodos que lo ignoran. |
| Metodología | Preparación, pose, alineación, características, clasificación y evaluación agrupada. |
| Resultados y conclusiones | Números observados y afirmaciones limitadas por lo que realmente se comparó. |

**Qué queda respaldado:**

- Hay una implementación sustancial de los cuatro objetivos, aunque su integración documental y de artefactos está incompleta.
- Se puede predecir las etiquetas mejor que la regla mayoritaria en las evaluaciones guardadas con caballos separados.
- Separar por caballo es necesario: los controles de identidad cuantifican la inflación de métricas al no hacerlo.
- El recorte nuevo mejoró la visibilidad de orejas en la medición correspondiente.
- El modelo promedio supera su criterio local y el resultado MIL guardado supera el umbral que utiliza su notebook.

**Qué no queda respaldado:**

- **Que MIL haya demostrado aprovechar el orden temporal.** El modelo actual es invariante al orden por construcción.
- Que las pruebas históricas demuestren una ventaja del orden: las permutaciones no la muestran.
- Que la geometría por sí sola causó la mejora MIL: falta una comparación emparejada de MIL con y sin geometría sobre los mismos clips y folds.
- Que el sistema esté validado como ayuda clínica a expertos. Un umbral interno de F1 no demuestra utilidad práctica, acuerdo con especialistas o generalización externa.
- Que ya se haya alcanzado un techo definitivo y ninguna arquitectura pueda mejorar. Los experimentos negativos limitan lo observado; no prueban imposibilidad.
- Que toda señal predictiva sea exclusivamente expresión facial: existen problemas de etiquetas y atajos medidos, incluida duración.

**La ablación no prueba todo lo que dicen algunos informes.** Comparar todos los clips con el subconjunto de detección propia cambia cantidad de datos, individuos y composición de clases. No equivale a entrenar sobre las mismas imágenes “con y sin normalización”, ni a eliminar únicamente la máscara geométrica. Por eso la caída no puede atribuirse causalmente a un solo componente.

Para Maruca: sabemos que una receta obtuvo una nota mejor que otra. Si también cambiaron los ingredientes y los comensales, todavía no sabemos qué cambio produjo esa mejora.

## 18. Inconsistencias que afectan la interpretación

| Tema | Versiones encontradas y consecuencia |
|---|---|
| Dataset “actual” | README: 34 videos/92 clips/22 caballos; project.md: mezclas de 234 y 249 videos; CSV: 274/393/136 en clips. Son cortes diferentes, presentados a veces como vigentes. |
| Caballos originales | HTML original afirma 139 catalogados; los prefijos del inventario local dan 138. No se asume cuál conteo histórico pretendía el HTML. |
| Clasificación final | README y textos del notebook: voto 0,626; JSON local: promedio 0,704; código y salidas principales: MIL 0,754. |
| Conjunto MIL | Notebook: 257/113; regla actual aplicada al CSV local: 254/111; partición persistida: 153/83. |
| Etiquetas de las celdas | Las celdas 15 y 22 llaman `entrenar_mil_multimodal` pero imprimen “un frame + voto”. |
| Criterio | Código actual: 0,691; evaluación local: 0,6747; otros experimentos: 0,7005, 0,7847, 0,6825 y 0,6265. Cada uno requiere su conjunto y justificación. |
| Recalcular criterio | Celda 19 del notebook MIL imprime 0,7821, pero conserva una explicación que llama a la diferencia “redondeo” de 0,691. No es redondeo: reemplazó la línea de base por el propio MIL. |
| Hipótesis temporal | HTML v2/v3 e informe MIL la declaran sostenida; `MILFusionModel.forward()` no modela orden. Las pruebas históricas tampoco respaldan esa conclusión. |
| Comparaciones HTML | Mezclan mayoritaria 0,175, promedio 0,704 y MIL 0,754 como comparables; MIL guarda mayoritaria 0,168 y otro conjunto. |
| Optimizador | HTML v2/v3 dicen AdamW; MIL usa Adam. En v3 quedan hiperparámetros históricos incompatibles con batch 32, 50 épocas y weight decay 0,0001 actuales. |
| Frames admitidos | Textos y tests exigen 16; partición e inferencia actuales aceptan 12–16. El notebook 02 sigue muestreando 16 sin adoptar el límite de la etapa 01 para clips muy cortos. |
| Frames realmente consumidos | Se admiten clips con 12 alineados, pero el extractor de características no descarta sus frames no alineados. |
| Alineación visual/geometría en inferencia | `preparar_clip()` elimina recortes inválidos pero devuelve la pose con sus órdenes originales. El dataset MIL consulta órdenes 0..T−1. Si falta un frame intermedio, pueden emparejarse imagen y pose de momentos distintos. Es un problema observable en la lógica; no se cuantificó ejecutando videos. |
| Hocico | El recorte usa el primer candidato disponible por defecto; MIL elige por máxima confianza entre candidatos. No necesariamente usa el mismo punto anatómico. |
| Máscaras | La máscara temporal ignora padding; la geométrica indica validez. Ninguna por sí sola identifica toda interpolación o anotación manual, pese a ciertas descripciones. |
| Confianza mostrada | La app carga `un_frame_produccion.pt` como MIL, pero obtiene precision de `objetivo3_evaluacion.json`, que localmente corresponde a promedio. |
| Procedencia de pesos | Existe el PT actual, pero `un_frame_produccion.json` describe todavía el modelo lineal de 141 clips. No documenta de forma coherente ese PT. |
| Mensajes de interfaz | `app_inferencia.py` y `texto_de_confiabilidad()` mantienen “0,626 contra 0,691, no alcanzó”, aunque otras fuentes informan cumplimiento. |
| Anotaciones | 26 frames manuales en resultado frente a 14 filas en el archivo manual actual. |
| Ablación MIL | La salida dice 20 clips Disgustado en el subconjunto; el texto impreso todavía explica la caída por tener 7. |
| Documentación intermedia | `analisis_resultados.md` dice que Disgustado pasa a 20 clips con “+1” respecto de un conjunto histórico de 21; no cuadra. |
| Precision y recall | `analisis_resultados_clasificacion.md` llama precisión al 82,8 % de Interesado, pero surge de 48/58: es recall. |
| Orquestación | README presenta 00 como entrada a todo el pipeline; `equinos.etapas.PIPELINE` solo registra 01 y 02, no 03 ni 04. |
| Ejecución conservadora | 00 dice que saltea etapas completas, pero su celda 5 fija `ETAPAS_A_FORZAR=["preparar_dataset","roi_normalizacion"]`. Ejecutarlo tal como está fuerza esas etapas. No se ejecutó. |
| Documentos pendientes | README dice que objetivos 3 y 4 no están implementados, pero existen clasificadores e interfaz. project.md dice que equinos no tiene lógica científica, aunque ahora sí la contiene. |
| Afirmaciones de infraestructura | Los informes describen DirectML/AVX-512 y reproducibilidad bit a bit. El clasificador actual selecciona CUDA o CPU; no se verificaron esos relatos como garantías del código actual. |

La existencia de contradicciones no convierte todos los resultados en falsos. Significa que **deben citarse junto con modelo, dataset, protocolo y archivo concreto**, sin unirlos por conveniencia.

## 19. Código, tests, configuración, figuras y estado del pipeline final

**Organización del código.**

- `rutas.py`: raíz persistente, rutas relativas y área de trabajo.
- `catalogo.py`: inclusión de videos y clases.
- `checkpoint.py`, `cache.py`, `etapas.py`: reanudación, ZIP y registro de etapas.
- `pose.py`, `encuadre.py`, `interpolacion.py`, `anotacion.py`: puntos anatómicos, recorte, estimación y correcciones humanas.
- `reporte.py`: cobertura, calidad y selección.
- `particion.py`, `caracteristicas.py`, `clasificacion.py`: datos para entrenamiento, ResNet y modelos.
- `inferencia.py`: video nuevo hasta predicción o negativa.
- `app_inferencia.py`: presentación Streamlit.
- `correr_local.sh`: configuración y comandos locales.

`datos_equinos/` contiene datos derivados, cachés, modelos, ejecuciones y respaldos de varias fechas. No es un único dataset limpio listo para sumar. `.gitignore` excluye videos, pesos y gran parte de los resultados; conserva como excepción las anotaciones manuales. Esto explica por qué Git, los notebooks y los resultados de cada máquina pueden divergir.

`requirements-local.txt` fija, entre otros, PyTorch 2.5.1+cu121, torchvision 0.20.1+cu121, NumPy 1.26.4, DeepLabCut 3.0.1, ultralytics 8.4.100 y scikit-learn 1.7.2. TensorFlow aparece como previsto en README, pero no es el framework de los clasificadores actuales.

**Tests.** Hay once archivos: geometría, persistencia, catálogo, pose, entorno, anotación, interpolación, encuadre, partición, clasificación e inferencia. Cubren cálculos, fallos de entrada, persistencia, agrupamiento y métricas, mediante comprobaciones `check()` y ejemplos sintéticos. No equivalen a validación científica del modelo.

No se ejecutaron. La lectura muestra desactualización en `test_inferencia.py`: usa la firma antigua de `predecir_desde_caracteristicas`, modelos falsos anteriores y exige rechazar 15 frames, contrario a la regla actual. Por tanto, no se puede afirmar que “todos los tests pasan” en este estado. Tampoco observé pruebas específicas que acrediten la consistencia completa de la nueva rama MIL.

**Notebook de figuras.** `04_figuras_tesis.ipynb`, celda 3, genera preprocesamiento, atención, espacio geométrico y matriz:

- La atención usa el modelo que carga la inferencia, actualmente MIL.
- La matriz usa el JSON local del promedio: **pueden mostrarse figuras de dos modelos distintos bajo un título MIL**.
- Para la figura 1, el código toma `frames_crudos[idx_frame]` de la lista completa del video, pero el recorte corresponde al frame muestreado en esa posición. No convierte el índice de muestra al índice del video original; las dos mitades pueden mostrar instantes diferentes.
- La figura geométrica utiliza puntos de una muestra de clips, no necesariamente solo del conjunto evaluado.
- Los pesos de atención muestran cómo pondera frames, pero no prueban causalmente que identificó un gesto anatómico concreto.
- Las salidas guardadas contienen rastros de ejecución Linux y Windows. No conviene tratarlas como una única corrida homogénea.

**Cuál parece ser el pipeline final.** La intención actual es **preprocesamiento facial → ResNet congelada + geometría → MIL → interfaz**. La evidencia local más coherente entre partición, características, evaluación y procedencia de pesos es todavía **preprocesamiento → ResNet → cabeza promedio**, de la otra línea de desarrollo.

Existe un archivo PT de producción y código para entrenarlo, pero el notebook 03 actual termina sin las celdas de persistencia de evaluación y entrenamiento de producción que sí aparecen en la ejecución histórica de promedio. Por eso el flujo completo actual no queda reproducible únicamente con “ejecutar 00” o “ejecutar 03” tal como están.

## Pipeline final del proyecto

Este es el flujo de la **implementación actualmente pretendida**, con sus límites verificados:

1. **Reunir y etiquetar videos.** Se organizan en Sereno, Interesado y Disgustado. El nombre identifica al caballo.
2. **Seleccionar la fuente.** El catálogo determina qué carpetas y subcarpetas entran. La fuente se declara con `TESINA_EQUINOS_VIDEOS`.
3. **Inventariar.** Se registran resolución, duración, FPS y cantidad de frames. El inventario local tiene 274 videos.
4. **Preparar clips.** Se buscan segmentos de aproximadamente seis segundos y se admiten tomas cortas desde un segundo. Hay 393 clips inventariados.
5. **Muestrear imágenes.** Se toman hasta 16 fotos uniformemente distribuidas. La corrida local produjo 6.288.
6. **Detectar al caballo.** YOLO11x localiza el animal y las cajas se estabilizan temporalmente.
7. **Localizar puntos anatómicos.** DeepLabCut produce pose; se conserva la tabla completa y se seleccionan ojo y hocico.
8. **Completar posiciones cuando corresponde.** Se interpola como máximo un hueco interior y se incorporan anotaciones manuales disponibles, dejando registrado el origen.
9. **Alinear y recortar la cara.** Se gira, escala y traslada a 224×343. Se registra si la salida está alineada y cuánto relleno tiene.
10. **Decidir qué clips entran al modelo.** La regla histórica exige 16 alineados; la actual admite 12–16. Localmente dan 153 y 254 clips, respectivamente. Debe conservarse esa distinción frente a los 257 de la salida MIL guardada.
11. **Separar por caballo.** Se asigna cada animal completo a uno de cuatro folds para que la evaluación use animales no vistos en entrenamiento.
12. **Extraer apariencia.** ResNet-18 congelada convierte cada imagen normalizada en 512 valores.
13. **Construir geometría.** El MIL añade coordenadas normalizadas de ojo, hocico y un punto de oreja, más indicadores de validez.
14. **Preparar lotes.** Se completan longitudes con ceros y se enmascara el padding. Sigue pendiente de coherencia el filtrado efectivo de frames no alineados y su correspondencia con la pose.
15. **Entrenar y evaluar MIL.** Se aprende a combinar apariencia y geometría y a ponderar los frames; se evalúa con los cuatro folds. Es agregación de múltiples imágenes, sin modelado explícito de su orden.
16. **Guardar modelo y evaluación correspondiente.** El código permite entrenar producción con todos los clips. Actualmente los pesos MIL, su procedencia y el JSON de evaluación local no están sincronizados.
17. **Clasificar una toma nueva.** La interfaz vuelve a detectar, estimar pose, recortar y extraer características. Acepta 12–16 recortes utilizables; no aplica la misma interpolación y anotación manual del entrenamiento. Si no cumple condiciones, devuelve una negativa explicada.
18. **Mostrar el resultado.** Presenta clase, probabilidades y pesos de atención. La confiabilidad debe leerse con la contradicción actual: se consulta un JSON del modelo promedio mientras se carga MIL.
19. **Generar figuras e informe.** El notebook 04 produce gráficos; deben identificarse por su fuente concreta, porque hoy puede mezclar el modelo MIL con la evaluación del promedio.

Para Maruca: el proyecto ya sabe preparar las fotos, acomodar las caras y producir una clasificación. Lo que aún no está unido de manera coherente es la receta actual, el modelo guardado y la libreta con la nota que le corresponde. Los resultados existentes permiten afirmar que hay capacidad predictiva sobre las etiquetas del dataset; no permiten afirmar que MIL haya demostrado comprender el orden del gesto ni que todas las cifras actuales describan el mismo sistema.

