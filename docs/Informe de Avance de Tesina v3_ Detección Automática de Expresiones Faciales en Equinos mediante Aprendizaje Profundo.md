# **Informe de Avance de Tesina v3: Detección Automática de Expresiones Faciales en Equinos mediante Aprendizaje Profundo**

El presente documento constituye la tercera entrega evolutiva del informe de avance de tesina en el marco de la Licenciatura en Ciencia de Datos de la Universidad del Gran Rosario (UGR). Esta versión consolida una transición crítica: el paso de un modelo de clasificación estática hacia un marco de validación biométrica animal robusto, diseñado para mitigar los sesgos de identidad y optimizar la observabilidad anatómica mediante la integración de estimación de pose.

**Institución:** Universidad del Gran Rosario (UGR)&nbsp;

**Carrera:** Licenciatura en Ciencias de Datos&nbsp;

**Asignatura:** Seminario de Trabajo Final (Tesina)&nbsp;

**Tema:** Detección Automática de Expresiones Faciales en Equinos mediante Aprendizaje Profundo en Secuencias de Video&nbsp;

**Autores:** Héctor R. Liceaga, Sergio M. Yañez&nbsp;

**Equipo Evaluador:** Alejandro Hernández, Brian Gauna, Jorge Fernández&nbsp;

**Fecha de Presentación:** Septiembre de 2026

&nbsp;

La viabilidad de este proyecto no reside únicamente en la precisión algorítmica, sino en la necesidad de fundamentar la investigación en una problemática ética y metodológica sólida que responda a las demandas de bienestar animal contemporáneas.

## 

## **2\. PROBLEMÁTICA, HIPÓTESIS Y MARCO METODOLÓGICO**

La intersección entre la etología computacional y el paradigma de "Un Bienestar" (*One Welfare*) establece que el estado emocional del equino es un componente observable y cuantificable de su salud sistémica. En una industria donde la evaluación del bienestar ha dependido históricamente de la subjetividad humana, la automatización mediante visión computacional no es solo una optimización de procesos, sino un imperativo ético para garantizar diagnósticos objetivos y no invasivos.

Bajo la Taxonomía de Bloom, este informe eleva su rigor científico al transicionar de verbos descriptivos a objetivos de alto nivel como **"Desarrollar y evaluar"** y **"Recopilar y conformar"**. Este cambio refleja una madurez metodológica: ya no se busca simplemente entrenar un clasificador, sino validar un sistema de biometría animal mediante protocolos de validación cruzada *GroupKFold*, transformando el proyecto en un marco científico reproducible.

Un hallazgo crítico en esta evolución es el cambio de foco anatómico. Mientras que en versiones anteriores (v2), siguiendo los Mapas de Activación de Clase (CAM) tradicionales, el ojo se identificaba como la región de mayor peso, en esta v3 hemos pivotado hacia la **movilidad auricular**. Esta transición fue posible gracias a la integración de *DeepLabCut*, que permite capturar puntos clave dinámicos que los modelos estáticos ignoraban, reconociendo que la posición de las orejas es el indicador más precoz y dinámico de los estados de estrés y atención.

La **hipótesis principal** se centra en el modelado secuencial temporal. Si bien la premisa inicial sugería que el flujo temporal aportaría información crítica, esta v3 somete dicha idea a una validación empírica estricta, contrastando la complejidad de las arquitecturas recurrentes frente a la robustez del voto mayoritario en imágenes independientes.

## **3\. DESARROLLO EXPERIMENTAL Y ANÁLISIS DE RESULTADOS**

En biometría animal, la identidad del ejemplar es la variable de confusión más peligrosa. Sin una curación estratégica, los modelos tienden a memorizar rasgos individuales (pelaje, marcas faciales) en lugar de expresiones, invalidando la capacidad de generalización.

### **3.1 Diseño Experimental del Corpus y Curación de Datos**

El dataset actual se compone de **147 clips íntegros** que representan a **79 ejemplares únicos**. La distribución de clases, aunque presenta el desafío intrínseco del desbalanceo en campo, es la siguiente:

&nbsp;

| Clase | Clips | Ejemplares Únicos |
| :---- | :---- | :---- |
| **Disgustado** | 25 | 20 |
| **Interesado** | 58 | 36 |
| **Sereno** | 64 | 36 |

Para neutralizar el *data leakage* por identidad, se implementó un protocolo **GroupKFold (4 folds)**. A diferencia del *Train-Test Split* ingenuo de versiones previas, este método garantiza la independencia biológica absoluta entre conjuntos. Esta rigurosidad técnica explica por qué, aunque las métricas parecen inferiores a los resultados del 85% reportados en investigaciones base (Mateo Marcos), nuestra v3 es científicamente más honesta y posee una capacidad de generalización real, logrando una ganancia de **\+0.239 en F1 Macro** frente a particiones aleatorias que sufren de memorización de sujetos.

### **3.2 Normalización Geométrica y Aislamiento de la Región de Interés (ROI)**

El pipeline utiliza **YOLO11x** para la detección del sujeto y **DeepLabCut** para la estimación de pose. Se definió un recorte de **224x343** píxeles, justificado anatómicamente para incluir el arco supraciliar y el eje auricular. La resolución del "efecto orejas" es un hito de esta entrega: mediante el cálculo de polígonos dinámicos de puntos clave, elevamos la visibilidad útil de las orejas del **4.3% al 77.2%**, capturando información vital sobre la tensión muscular en la base del pabellón auricular.

### **3.3 Arquitectura del Modelo y Evaluación de Rendimiento**

Se implementó una arquitectura **ResNet-18 (512-d)** con una estrategia de **voto mayoritario**. Los resultados obtenidos son:

* **F1 Macro:** 0.649 ± 0.092  
* **Balanced Accuracy:** 0.664 ± 0.070

El **umbral de aceptación (0.691)** se deriva de la suma del modelo base (0.626) más una desviación estándar (0.065) observada en los 4 folds. Este umbral no es arbitrario; actúa como una salvaguarda contra el *p-hacking*, forzando al modelo a superar el "ruido de rendimiento" inherente a la arquitectura para ser considerado exitoso.

&nbsp;

### **3.4 La Navaja de Ockham en Ciencia de Datos**

El rechazo de la hipótesis temporal (resultados entre **0.580 y 0.632**) en favor del voto mayoritario es un hallazgo arquitectónico valioso. Siguiendo el principio de parsimonia, se optó por la simplicidad estática. Esta decisión no solo es estadística, sino de ingeniería: al reducir la complejidad del modelo, disminuimos drásticamente la latencia de inferencia, garantizando la viabilidad de la ejecución en tiempo real dentro de la `app_inferencia.py`.

## **4\. LIMITACIONES DE INFRAESTRUCTURA, PORTABILIDAD E INGENIERÍA DE SOFTWARE**

La resiliencia en ingeniería de software es crítica cuando los modelos de *Deep Learning* se enfrentan a hardware heterogéneo. El desarrollo se vio condicionado por un severo "lock-in" tecnológico que exigió soluciones creativas.

**Desafíos Técnicos Identificados:**

* **Incompatibilidad CUDA vs. AMD:** El uso de una **AMD Radeon RX 7600 (RDNA 3\)** imposibilitó el uso directo de librerías optimizadas para NVIDIA. La solución resiliente involucró el uso de capas de abstracción como **PyTorch sobre DirectML** y entornos virtuales (venvs) desacoplados para evitar conflictos de drivers.  
* **Conflictos de Dependencias:** La incompatibilidad de **Numpy \< 2.0** con las versiones actuales de DeepLabCut no fue un simple error de versión, sino una limitación en la estructura de tensores que amenazaba la portabilidad entre Google Colab y el hardware local.  
* **Latencia de Preprocesamiento:** Ante la ausencia de núcleos Tensor operativos en la GPU AMD para ciertas tareas, el pipeline dependió de la computación vectorizada **AVX-512** en CPU para mantener la cadencia de frames necesaria.

**Soluciones de Ingeniería Implementadas:**

* **Desacoplamiento de pose:** Persistencia de puntos clave en `pose_completa.csv`, separando la carga computacional de la estimación de la clasificación.  
* **Control transaccional:** Uso de `equinos.checkpoint` para garantizar la integridad del entrenamiento ante interrupciones del hardware.  
* **Entornos de Abstracción:** Implementación de contenedores que emulan las dependencias necesarias para la portabilidad del modelo.

  ## **5\. CONCLUSIONES Y TRABAJO FUTURO**

La investigación ha alcanzado una madurez donde la honestidad clínica prima sobre las métricas infladas. Hemos demostrado que el cuello de botella actual no es algorítmico, sino de **diversidad de datos**, específicamente en la clase "Disgustado".

La hoja de ruta para la consolidación del proyecto incluye:

1. **Captura de Alta Velocidad:** Implementar grabaciones a **120 fps** para mitigar el *motion blur* en las orejas durante movimientos bruscos de defensa, lo que mejorará la precisión de DeepLabCut.  
2. **Expansión del Corpus:** Recolección dirigida de ejemplares en situaciones de malestar real (clínicas veterinarias) para equilibrar la clase minoritaria.  
3. **Despliegue en Streamlit:** Optimizar la `app_inferencia.py` para su uso en dispositivos móviles mediante cuantización de modelos (INT8).

Este trabajo establece un precedente en la biometría animal automatizada, demostrando que la solidez de la ingeniería de datos y el rigor metodológico son los únicos caminos hacia un sistema de bienestar animal transparente y auditable.

&nbsp;

&nbsp;