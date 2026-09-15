Hola Héctor. Es un excelente ejercicio someter este informe de avance a una revisión crítica antes de la presentación final.

A continuación, estructuro mi respuesta en dos partes: primero, una explicación detallada de lo que comunica actualmente tu documento (para que confirmes si el mensaje que intentas dar es el que realmente se lee) y, segundo, asumo el rol de un docente evaluador (Doctor en Ciencia de Datos) para darte una devolución metodológica y recomendaciones de mejora.

---

### Parte 1: Explicación Detallada de tu Informe de Avance

El documento que presentas es un reporte estructurado que comunica el estado actual de tu tesis, justificando metodológicamente las decisiones tomadas y exponiendo los resultados preliminares. Aquí tienes el desglose de lo que transmite cada sección:

**1. Carátula y Contexto Académico / Propósito**

* **Qué comunica:** Establece el marco formal del proyecto en la UGR y define que el documento sirve como una "auditoría de rigor metodológico". Deja claro que el proyecto ya pasó la fase de preprocesamiento y está en la etapa de modelado, enfocándose en evaluar si el sistema cumple su objetivo.



**2. Problemática, Hipótesis y Marco Metodológico**

* **Qué comunica:** Planteas que la evaluación humana del bienestar animal es subjetiva y sesgada, y propones la visión por computadora como una solución objetiva y preventiva.


* **El núcleo científico:** Defines una hipótesis clara (el deep learning supera la observación azarosa) y, lo más importante, estableces una métrica de éxito concreta: un F1 Macro de 0.691.


* **Madurez académica:** Justificas el cambio de verbos en tus objetivos (usando la Taxonomía de Bloom) para demostrar que el valor de la tesis no es solo hacer código, sino investigar, validar empíricamente y analizar errores.



**3. Objetivos Oficiales**

* **Qué comunica:** Enumera el objetivo general (desarrollar y evaluar el sistema en secuencias de video) y cuatro específicos que marcan el ciclo de vida del dato: recolección de videos cortos, preprocesamiento (YOLO/DeepLabCut), modelado (Machine Learning orientado a video) y despliegue (interfaz de usuario).



**4. Estado de Avance y Resultados Experimentales**

* **Dataset (4.1):** Muestras tu volumen de datos (147 clips de 79 caballos) y expones una decisión crítica: usar *GroupKFold* por ID de caballo. Explicas que sin esto, el modelo hace "trampa" (aprende a reconocer al caballo, no la expresión), inflando el F1 Macro en +0.239 de forma artificial.


* **Preprocesamiento (4.2):** Explicas el uso de YOLO11x y DeepLabCut. Demuestras con un estudio de ablación que normalizar geométricamente los rostros (alineación ojo-hocico) mejora drásticamente la legibilidad de las orejas (del 4.3% al 77.2%) y evita una caída de -0.122 en el F1 Macro.


* **Clasificación (4.3):** Reportas los resultados de una ResNet-18. El F1 Macro actual es de 0.649 ± 0.092, lo cual significa que **rechazas temporalmente tu hipótesis**, ya que no alcanzas el 0.691. Además, haces un descubrimiento clave: barajar el orden de los fotogramas no empeora el modelo, lo que significa que en clips de 5-7 segundos no hay información temporal útil que modelos complejos (como GRU) puedan aprovechar por sobre un enfoque estático.


* **Prototipo (4.4):** Describes una app en Streamlit funcional, pero con un enfoque ético: muestras la confiabilidad de la predicción para advertir que el modelo aún falla mucho en la clase "Disgustado" (33% de precisión), evitando que se use como una verdad absoluta.



**5. Conclusiones y Futuro**

* **Qué comunica:** Concluyes que la señal visual existe, pero el modelo tocó un "techo" debido a la falta de datos representativos, no por un problema del algoritmo. Tu plan de acción es claro: conseguir más caballos diferentes, específicamente para la clase "Disgustado".



---

### Parte 2: Revisión y Recomendaciones (Rol: Docente Doctor en Ciencia de Datos)

*Estimados tesistas (Héctor y Sergio):*
*He revisado su informe de avance. El trabajo presenta un nivel de madurez metodológica excepcional para el nivel de grado. La honestidad intelectual al reportar resultados negativos y la rigurosidad en la validación son destacables. No obstante, para llevar esta tesis a la excelencia en su defensa final, les dejo las siguientes observaciones y recomendaciones:*

**Puntos Fuertes (Lo que deben resaltar en su defensa):**

1. **Manejo del Data Leakage:** La implementación de *GroupKFold* por ID de individuo es brillante. Es un error clásico en biometría y etología computacional que el modelo aprenda la identidad (el individuo) y no la característica (la expresión). Haberlo detectado y cuantificado (+0.239 de mejora artificial) demuestra un profundo entendimiento de la evaluación de modelos.


2. **Estudios de Ablación:** Justificar las decisiones de preprocesamiento (alineamiento ojo-hocico) midiendo su impacto directo en el F1 Macro (-0.122 si se omite) es ciencia de datos rigurosa.


3. **Transparencia Clínica:** Limitar el alcance del entregable (Streamlit) exponiendo empíricamente la baja confianza en la clase minoritaria ("Disgustado") denota ética profesional y madurez investigativa.



**Recomendaciones Críticas de Mejora (Para trabajar antes de la entrega final):**

1. **Justificación del Umbral de Hipótesis:**
* *Observación:* Mencionan que el umbral de 0.691 representa "el rendimiento del modelo base más una desviación estándar".


* *Acción requerida:* En el documento final, deben detallar **cuál es exactamente ese modelo base**. ¿Es un clasificador *Dummy*? ¿Es un modelo de Machine Learning tradicional (ej. SVM) sobre features extraídos manualmente? ¿De dónde provino ese cálculo inicial? Sin esta aclaración, el número 0.691 parece arbitrario.


2. **El "Colapso de Varianza Temporal":**
* *Observación:* El hallazgo de que el test de permutación temporal arroja una variación de +0.000 es fascinante y problemático a la vez. Si el orden de los frames no importa, la premisa de usar "secuencias de video" y arquitecturas temporales (GRU/Atención) pierde fuerza.


* *Acción requerida:* Deben discutir esto profundamente en las conclusiones. Si una ResNet-18 con voto mayoritario estático iguala a un modelo temporal, ¿es realmente necesario analizar el video como secuencia temporal, o basta con muestrear *frames* aislados de un clip? Esta discusión elevará muchísimo el nivel académico de la tesis.


3. **Estrategias para el Desbalance de Clases:**
* *Observación:* Su conclusión identifica correctamente que necesitan más datos de la clase "Disgustado" (actualmente 25 clips de 20 ejemplares frente a los 64 clips de "Sereno").


* *Acción requerida:* Como plan de contingencia por si la recolección en campo se complica, sugiero que agreguen al informe qué técnicas analíticas intentarán. ¿Han probado *Data Augmentation* agresivo específico para esa clase (rotaciones, cambios de iluminación, *mixup*)? ¿Han considerado aplicar *Focal Loss* o pesos por clase (*class weights*) durante el entrenamiento de la ResNet-18 para penalizar más los errores en la clase "Disgustado"?


4. **Análisis de Errores (Error Analysis):**
* *Observación:* Mencionan que el Caballo 906 concentra 5 errores sobre 9 clips.


* *Acción requerida:* ¿Por qué falla tanto este individuo en particular? ¿Tiene un pelaje distinto? ¿Una morfología facial atípica? ¿Las condiciones de iluminación de sus videos son anómalas? Realizar un análisis cualitativo del "peor caso" demostrará una capacidad analítica superior.



El informe es metodológicamente muy sólido. Si logran ajustar la métrica de la clase minoritaria y profundizan en por qué el análisis temporal no está aportando valor, tendrán una tesis sobresaliente. ¡Adelante con la fase final!
