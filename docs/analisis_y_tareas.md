# Análisis del Proyecto: Detección de Expresiones en Equinos

He analizado el contenido del directorio, incluyendo el **Plan de Trabajo (`plan_de_trabajo.pdf`)**, la existencia de las **Filminas de presentación (`presentacion_filminas.pptx`)** y los **Notebooks del Pipeline** (`00`, `01`, y `02`).

## 1. Análisis del Estado Actual

En líneas generales, **el proyecto está muy bien planteado y estructurado**. Destacan los siguientes aciertos:
*   **Orquestación robusta:** El uso de `papermill` en `00_pipeline.ipynb` para ejecutar y persistir el estado de los notebooks de manera modular es una excelente práctica para lidiar con las limitaciones de Google Colab.
*   **Normalización geométrica:** El notebook `02_roi_y_normalizacion.ipynb` aborda de manera muy inteligente un problema crítico: evitar que el modelo aprenda del fondo (sesgo de escenario) alineando el rostro del equino mediante una transformación afín basada en el eje ojo-hocico.
*   **Plan de Trabajo:** El PDF define objetivos claros, una hipótesis contrastable y un alcance temporal realista (videos de 5 a 7 segundos, particionados por caballo para evitar fuga de datos).

## 2. Mejoras a Incorporar (Hallazgos)

Además del punto que bien observaste sobre `imagededup`, identifiqué un par de áreas críticas para el éxito del modelo:

> [!TIP]
> **Falta de Filtrado por Similitud (Hashing de frames)**
> Actualmente, la función `indices_uniformes` en el paso 6 del notebook 02 extrae exactamente 16 frames por clip equitativamente espaciados. Si el caballo se queda quieto durante esos 5-7 segundos, se extraerán múltiples imágenes idénticas. Esto causa **overfitting** (el modelo memoriza la imagen en lugar de la expresión) y desperdicia capacidad de cómputo. Incorporar `imagededup` mediante hashing perceptual (PHash) es el enfoque correcto.

> [!WARNING]
> **Desbalanceo Extremo de Clases**
> En el inventario del notebook 01, la distribución de videos es: **Sereno (23), Interesado (8), Disgustado (3)**. Esto es un desbalance severo. El modelo tenderá a predecir "Sereno" siempre para minimizar el error.

> [!NOTE]
> **Pérdida de Contexto en las Orejas**
> En el notebook 02, se modificaron los parámetros `FRAC_OJO=0.20` y `FRAC_HOCICO=0.80`. Esto acentúa el recorte sobre el hocico. En los caballos, **la posición de las orejas (hacia atrás o planas) es el indicador número uno de disgusto o estrés**. Hay que asegurar que el recorte de `LADO_RECORTE = 224` no esté dejando las orejas fuera del frame.

---

## 3. Informe de Tareas a Realizar

Basado en el análisis, aquí está la lista de tareas priorizadas para continuar con el desarrollo de la Tesina:

### Tarea 1: Implementar `imagededup` para limpieza del dataset
- [ ] Integrar la librería `imagededup` en el notebook `01` o como un paso intermedio `01b`.
- [ ] Generar hashes perceptuales (PHash) o usar un CNN encoding ligero para cada frame extraído.
- [ ] Definir un umbral de distancia de Hamming para descartar frames que sean > 95% idénticos al frame anterior en el mismo clip.

### Tarea 2: Validar el encuadre de las orejas (ROI)
- [ ] En el notebook `02`, tomar una muestra de los recortes generados (especialmente de la clase "Disgustado").
- [ ] Verificar visualmente si las orejas del equino están dentro de la imagen de `224x224`. 
- [ ] Si las orejas quedan cortadas, ajustar `MARGEN_ROI` o subir `FRAC_OJO` levemente (ej. `0.25`) para dar más aire en la parte superior de la cabeza.

### Tarea 3: Estrategia de Balanceo y Data Augmentation
- [ ] Desarrollar una estrategia para manejar la escasez de videos de la clase "Disgustado".
- [ ] Implementar aumento de datos (Data Augmentation) focalizado: ligeras rotaciones, espejado horizontal (flip), y variaciones de brillo.
- [ ] Considerar el uso de pesos por clase (Class Weights) al momento de definir la función de pérdida (Loss Function) del modelo.

### Tarea 4: Desarrollo del Modelo Base (Notebook 03)
- [ ] Crear el notebook `03_entrenamiento.ipynb`.
- [ ] Implementar una arquitectura convolucional temporal (ej. CNN + LSTM) o un modelo preentrenado (ResNet3D / TimeSformer) capaz de analizar la secuencia espacio-temporal de los clips ya normalizados.
