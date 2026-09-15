# Informe de Infraestructura Computacional, Desafíos de Hardware y Estrategias de Reproducibilidad

**Proyecto:** Detección Automática de Expresiones Faciales en Equinos mediante Aprendizaje Profundo  
**Ámbito:** Licenciatura en Ciencias de Datos — Universidad del Gran Rosario (UGR)  
**Documento para:** Sección metodológica de la Memoria de Tesina y presentación ante el comité evaluador / docentes.

---

## 1. Resumen Ejecutivo

Durante el desarrollo y ejecución del pipeline de visión por computadora para secuencias temporales de equinos, el equipo enfrentó restricciones técnicas derivadas de la heterogeneidad de hardware, el acoplamiento a librerías propietarias y la evolución de arquitecturas de GPUs. 

En lugar de depender exclusivamente de entornos efímeros en la nube (Google Colab), se diseñó una **arquitectura de software resiliente y desacoplada** en el paquete local `equinos`. Esto garantizó la reproducibilidad determinista del dataset procesado (236 videos catalogados, 282 clips, más de 4.500 frames), independizando el proyecto de fallos de hardware o cuellos de botella de proveedores específicos.

---

## 2. Diagnóstico de Desafíos de Hardware y Ecosistema

```mermaid
graph TD
    A["Requerimiento de Modelos<br/>(DeepLabCut / YOLO / PyTorch)"] --> B["Ecosistema CUDA Lock-in"]
    B --> C["Intento 1: GPU NVIDIA Nueva Generación<br/>(Blackwell / Ada Lovelace)"]
    B --> D["Intento 2: GPU AMD Radeon RX 7600<br/>(Arquitectura RDNA 3 en Windows)"]
    B --> E["Entorno en la Nube<br/>(Google Colab T4)"]
    
    C -->|Incompatibilidad binaria<br/>PyTorch legacy vs sm_arch| F["Fallo de Ejecución / Kernel faltante"]
    D -->|Llamadas hardcodeadas a .cuda()<br/>Falta de soporte DirectML en DLC| G["Fallback automático a CPU"]
    E -->|Límites de tiempo y<br/>latencia de Drive| H["Cortes de sesión frecuentes"]
    
    G --> I["Solución de Ingeniería: Pipeline Desacoplado<br/>AVX-512 + Checkpointing + Pose Cruda"]
    H --> I
    F --> I
```

### 2.1. El monopolio de CUDA y la portabilidad en GPUs no-NVIDIA (AMD Radeon)
* **Problemática:** La mayoría de las librerías de estimación de pose animal (e.g., *DeepLabCut / SuperAnimal*) asumen como estándar de facto la plataforma NVIDIA CUDA. Aunque existen capas de abstracción en Windows como **DirectML** (`torch-directml`) o en Linux como **ROCm**, el código base de modelos preentrenados posee invocaciones explícitas a `torch.cuda` o primitivas que impiden la aceleración nativa en tarjetas AMD (como la Radeon RX 7600) sin reescribir dependencias internas de terceros.
* **Impacto en el proyecto:** Al ejecutar en Windows nativo con GPU AMD, el framework deriva la ejecución al procesador central (CPU). Aunque esto incrementa el tiempo de cómputo por clip, el cómputo matemático subyacente es idéntico bit a bit.

### 2.2. Incompatibilidad de binarios en GPUs NVIDIA de última generación (Blackwell / Ada Lovelace)
* **Problemática:** Al intentar ejecutar el stack en GPUs NVIDIA de arquitecturas muy recientes (como la serie Blackwell o núcleos con capacidades de cómputo *sm_90* / *sm_100*), surge una colisión entre versiones de software:
  1. *DeepLabCut* y ciertas versiones de detección fijan restricciones estrictas de dependencias históricas (por ejemplo, `numpy < 2.0` o ramas de `torchvision` anteriores).
  2. Las distribuciones binarias de PyTorch compatibles con ese stack legacy no traen compilados los kernels de CUDA específicos para la arquitectura de GPU más moderna.
  3. Esto genera errores en tiempo de ejecución del tipo `CUDA error: no kernel image is available for execution on the device`.
* **Impacto en el proyecto:** Se evidenció que contar con hardware más potente no garantiza compatibilidad si el ecosistema de investigación de la librería tiene dependencias congeladas.

### 2.3. Límites de sesión y latencia en entornos Cloud (Google Colab)
* **Problemática:** Los entornos gratuitos o estándar de Google Colab imponen desconexiones por inactividad y límites de cuota (GPU T4 con 2 vCPUs). Además, la latencia de red al leer y escribir miles de archivos de video sobre Google Drive montado multiplica el tiempo total de ejecución.

---

## 3. Decisiones de Ingeniería de Software Aplicadas

Para transformar estas limitaciones en una ventaja metodológica, se implementaron cuatro patrones de arquitectura en el código:

### 3.1. Desacoplamiento de Inferencia Pesada vs. Análisis (`pose_completa.csv`)
* **Decisión:** La inferencia neuronal de keypoints es la operación más costosa (~30–50 s por clip). En lugar de descartar la salida cruda o calcular directamente los puntos finales, el pipeline almacena `pose_completa.csv` en formato largo con todos los 39 bodyparts y confianzas.
* **Beneficio:** La selección del par de referencia (ojo/hocico) y el barrido analítico de candidatos (`equinos.pose.barrido`) se realizan en segundos sobre el archivo persistido, sin tener que volver a pagar horas de inferencia.

### 3.2. Checkpointing Transaccional Tolerante a Fallos
* **Decisión:** Cada etapa procesa unidades atómicas (por clip / por video) y registra un checkpoint transaccional (`equinos.checkpoint`).
* **Beneficio:** Si la máquina se apaga, Colab corta la sesión o se interrumpe el proceso, la ejecución se reanuda exactamente desde el último clip pendiente sin reprocesar ni duplicar registros.

### 3.3. Computación Vectorizada en CPU de Alto Rendimiento (AVX-512)
* **Decisión:** Se adaptó la ejecución para aprovechar procesadores modernos x86-64 (AMD Ryzen 7 8700F con 8 núcleos, 16 hilos y soporte completo para instrucciones vectoriales AVX-512).
* **Beneficio:** Permite que cualquier investigador sin una GPU NVIDIA dedicada pueda reproducir la totalidad de los experimentos en su estación de trabajo con estabilidad absoluta.

### 3.4. Abstracción de Rutas y Compatibilidad de Sistema de Archivos
* **Decisión:** Se creó el módulo `equinos.rutas` y `equinos.catalogo` con normalización canónica de caracteres (`clave()`).
* **Beneficio:** Resuelve discrepancias entre nombres de carpetas en Google Drive (que admiten caracteres especiales como `?`) y sistemas de archivos locales NTFS de Windows (que los prohíben con `WinError 123`), además de traducir automáticamente rutas absolutas heredadas de Colab (`/content/...`).

---

## 4. Estructura de Redacción Sugerida para la Memoria de Tesina

A continuación, se propone un extracto formal listo para ser incorporado en el cuerpo principal de la memoria académica:

> ### *Capítulo: Consideraciones de Infraestructura y Reproducibilidad Experimental*
> 
> *El entrenamiento y la inferencia de modelos profundos de visión en secuencias espacio-temporales imponen desafíos significativos de cómputo. Durante el desarrollo de este trabajo, se evaluaron diversos entornos de ejecución, incluyendo instancias en la nube aceleradas por GPU (Google Colab) y estaciones de trabajo locales bajo arquitecturas heterogéneas (x86_64 y GPUs dedicadas).*
> 
> *Se constató la existencia de un fuerte acoplamiento tecnológico hacia el ecosistema propietario NVIDIA CUDA en librerías especializadas de estimación de postura (*DeepLabCut*), lo que dificultó la aceleración directa en arquitecturas alternativas (AMD RDNA 3) e introdujo incompatibilidades de compilación de kernels en arquitecturas NVIDIA de reciente lanzamiento debido a dependencias heredadas.*
> 
> *Frente a este escenario, se priorizó la reproducibilidad científica mediante el diseño de una infraestructura propia de persistencia desacoplada y tolerancia a fallos (`equinos`). Mediante la vectorización multinúcleo (AVX-512), el almacenamiento de pose cruda en formato largo y el control transaccional de estados por clip, se garantizó que el procesamiento de los 282 clips del dataset fuera 100 % determinista, independiente del proveedor de infraestructura y replicable en hardware estándar de propósito general.*

---

## 5. Puntos Clave para la Reunión con los Docentes

1. **Estado del Pipeline:** Las etapas de catalogación (`01`) y normalización de pose (`02`) están completamente ejecutadas y auditadas sobre el dataset ampliado (**236 videos / 282 clips**).
2. **Robustez Demostrada:** El sistema no es un script frágil de Colab: cuenta con suite de pruebas unitarias (`tests/`), resolución multiplataforma y reanudación automática.
3. **Justificación Teórica:** La separación entre pose cruda y normalización geométrica justifica con datos empíricos la elección del mejor keypoint para el Objetivo 3 (clasificación temporal de emociones).
