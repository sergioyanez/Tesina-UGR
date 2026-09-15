Informe de Avance: Detección Automática de Expresiones Faciales en Equinos mediante Aprendizaje Profundo

1. Carátula y Contexto Académico

Institución: Universidad del Gran Rosario (UGR)
Carrera: Licenciatura en Ciencias de Datos
Asignatura: Seminario de Trabajo Final
Título del Proyecto: Detección Automática de Expresiones Faciales en Equinos mediante Aprendizaje Profundo en Secuencias de Video
Autores: Héctor R. Liceaga, Sergio M. Yañez
Cuerpo Docente: Alejandro Hernández, Brian Gauna, Jorge Fernández
Fecha: Septiembre de 2026

Propósito del Informe
El presente documento sistematiza el progreso técnico y los hallazgos experimentales del proyecto, actuando como una auditoría de rigor metodológico. Se expone la transición de la fase de preprocesamiento a la de modelado, evaluando el cumplimiento de los objetivos operativos y la validez de la hipótesis central frente a métricas de rendimiento estandarizadas en el dominio de la etología computacional.

2. Problemática, Hipótesis y Marco Metodológico

La evaluación tradicional del bienestar animal ha estado limitada por la subjetividad inherente a la observación humana, vulnerable a sesgos cognitivos y fatiga operativa. Este proyecto propone un cambio de paradigma estratégico: el tránsito hacia métricas objetivas mediante sistemas de visión por computadora. La capacidad de capturar microexpresiones faciales y traducirlas en datos computables permite una vigilancia preventiva y no invasiva, fundamental para la industria equina y la medicina veterinaria moderna.

Hipótesis y Criterio de Rigor Científico
Se postula que un sistema de aprendizaje profundo puede discriminar estados emocionales con una precisión superior a la observación azarosa. Para validar la relevancia clínica y técnica del modelo, se ha fijado un umbral de 0.691 de F1 Macro. Este valor representa el rendimiento del modelo base más una desviación estándar, constituyendo el eje rector que determina si el sistema posee una utilidad práctica real para la asistencia de expertos.

Justificación de la Actualización Metodológica
La redefinición de los verbos en el plan de trabajo responde a una exigencia de mayor profundidad académica:

* De "Construir" a "Desarrollar y evaluar": Esta modificación enfatiza que el valor científico no reside únicamente en la arquitectura del sistema, sino en su validación empírica y análisis crítico de errores.
* Taxonomía de Bloom: La inclusión de verbos como "Recopilar" y "Aplicar" reconoce el desafío investigativo de conformar un dataset de campo inédito y la complejidad de implementar técnicas de vanguardia en un dominio biológico específico.

Esta base metodológica fundamenta la hoja de ruta operativa presentada a continuación.

3. Objetivos Oficiales (Plan de Trabajo Corregido)

Los siguientes objetivos dirigen el desarrollo del sistema y actúan como métricas de éxito para la validación de la hipótesis de investigación:

Objetivo General
Desarrollar y evaluar un sistema informático para la detección automatizada de expresiones faciales en caballos mediante el procesamiento de secuencias espacio-temporales.

Objetivos Específicos

1. Recopilar y conformar una base de datos de videos cortos (5 a 7 segundos) de rostros de equinos.
2. Desarrollar un pipeline que permita el preprocesamiento de los videos, incluyendo la extracción de fotogramas clave y detección de la región de interés (ROI).
3. Aplicar técnicas de clasificación de Aprendizaje Automático orientadas a video para identificar microclimas emocionales.
4. Desarrollar una aplicación o interfaz para el uso del modelo resultante.

4. Estado de Avance y Resultados Experimentales

El proyecto ha alcanzado una fase de madurez técnica donde el pipeline se encuentra totalmente orquestado. La solidez de los resultados presentados es consecuencia directa de un diseño que prioriza la neutralización de sesgos biológicos.

4.1. Objetivo 1: Consolidación del Dataset y Neutralización del Sesgo Identitario

Se ha consolidado una base de datos propia, robustecida para enfrentar el desbalance de clases y asegurar la diversidad de ejemplares.

Clase	Clips Íntegros	Ejemplares Únicos
Disgustado	25	20
Interesado	58	36
Sereno	64	36
Total	147	79

Análisis de Integridad: Se detectó que 10 ejemplares (como los individuos 906, 907 y 910) presentan clips en múltiples clases. Este hallazgo hace obligatorio el uso de GroupKFold por ID de caballo. Sin esta partición estricta, el modelo actúa como un reconocedor de identidad y no de expresiones; una partición ingenua arroja una mejora artificial de +0.239 en F1 Macro, detectando al individuo en lugar de su estado etológico.

4.2. Objetivo 2: Pipeline de Preprocesamiento y Validación Biológica

Mediante el orquestador 00_pipeline.ipynb, se implementó una cadena que utiliza YOLO11x para detección y DeepLabCut para la extracción de keypoints faciales.

* Geometría e Isotropía: Se definió un recorte de 224x343 basado en la isotropía de la transformada de similitud. Este ajuste no es solo estético: permitió que la legibilidad de las orejas —principal señal etológica de la clase "Disgustado"— aumentara del 4.3% al 77.2%.
* Impacto Predictivo: El alineamiento ojo-hocico es crítico; un estudio de ablación demostró que sin este preprocesamiento, el rendimiento cae en -0.122 en F1 Macro, confirmando que la normalización geométrica es la que habilita al modelo a extraer rasgos biológicos genuinos.

4.3. Objetivo 3: Clasificación y Contraste de Hipótesis

El clasificador definitivo utiliza una ResNet-18 (congelada en ImageNet) con agregación por voto mayoritario.

Métricas Experimentales:

* F1 Macro: 0.649 ± 0.092
* Balanced Accuracy: 0.664 ± 0.070
* Análisis de Confusión: La clase "Interesado" es la más robusta (82.8% de recall). La clase "Sereno" muestra alta dispersión, sugiriendo el solapamiento de microexpresiones intermedias. El análisis forense revela que solo 10 ejemplares concentran el 57% de los fallos, destacando el Caballo 906 con 5 errores sobre 9 clips.

Rechazo de la Hipótesis Temporal: El valor de 0.649 es inferior al umbral exigido de 0.691. Un hallazgo crítico fue el test de permutación temporal: barajar aleatoriamente los frames no alteró la predicción (+0.000 de variación). Esto demuestra un colapso de varianza temporal donde las arquitecturas complejas (GRU/Atención) no logran superar al voto mayoritario estático en secuencias breves.

4.4. Objetivo 4: Prototipo de Inferencia (Streamlit)

La aplicación app_inferencia.py ofrece una demostración funcional con un tiempo de procesamiento de ~35 segundos.

* Transparencia Clínica y Límite del Entregable: Siguiendo una política de integridad diagnóstica, la interfaz muestra la etiqueta junto a su confiabilidad empírica. Es vital reportar que, aunque el sistema es superior al azar, posee una precisión del 33% en la clase "Disgustado" (solo 7 aciertos en 21 predicciones). Exponer esta métrica previene que el usuario asuma la predicción como una verdad diagnóstica absoluta.

5. Conclusiones y Plan de Trabajo Futuro

El proyecto ha demostrado que existe señal facial inequívoca para la clasificación emocional en equinos. Sin embargo, se ha alcanzado un "techo predictivo" que no es de naturaleza algorítmica, sino de representatividad estadística. La metodología es robusta, pero la diversidad de los ejemplares actuales limita la generalización requerida para superar el umbral de la hipótesis.

Prioridades Estratégicas Finales:

* Diversificación Identitaria: Es imperativo capturar nuevos ejemplares únicos para diluir el sesgo de individuos específicos.
* Foco en Clase Minoritaria: La recolección debe centrarse exclusivamente en la clase "Disgustado" para nutrir la variabilidad estadística del modelo.

El equipo se declara preparado para la defensa final, fundamentando sus conclusiones en la transparencia de los resultados y el rigor de los protocolos de validación empleados.
