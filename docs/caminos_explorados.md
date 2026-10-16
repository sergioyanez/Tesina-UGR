# Caminos explorados y decisiones de la tesina

Documento de trazabilidad metodológica. Estado del repositorio al 08/10/2026 (`main` @ `1b63be6`, más la rama `feat/validacion-por-sesion`).
Reconstruido a partir del historial de Git (todas las ramas y el reflog local), los changes de
OpenSpec, los informes de `docs/`, la auditoría v2, el experimento SimBA, el protocolo de comparación
y las notas de trabajo del asistente.

## 1. Propósito y cómo leer este documento

La tesina se defiende ante un tribunal y la trazabilidad de las decisiones forma parte del
entregable. Este documento registra **cada camino que se siguió, qué se midió y por qué se adoptó,
se descartó, se abandonó o quedó pendiente**, para volcarlo después al informe final. No reemplaza a
`openspec/project.md` (estado vigente) ni a `docs/registro_de_experimentos.md` (qué archivo de
resultados sale de qué código y rama).

Cada camino (`C-01` a `C-47`) tiene siempre el mismo formato: **Qué se intentó**, **Por qué**,
**Resultado**, **Decisión** y **Evidencia**.

### Vocabulario de la decisión

| Término | Significado |
|---|---|
| **Adoptado** | Forma parte del pipeline o del modelo vigente. |
| **Reemplazado** | Estuvo vigente y otro camino lo sustituyó; se indica si fue por una medición. |
| **Descartado: peor** | Se midió y rindió peor que la alternativa, con diferencia que supera su incertidumbre. |
| **Descartado: sin diferencia** | Se midió y no se distingue de la alternativa; se descartó por costo, parsimonia o riesgo. |
| **Descartado: otra razón** | No se descartó por rendimiento (imposibilidad técnica, riesgo metodológico, incompatibilidad). |
| **Descartado: sin medición** | Se descartó por costo, tiempo o argumento, sin medirlo. |
| **Abandonado** | La línea de trabajo se dejó sin cierre formal; se aclara si el motivo está documentado. |
| **Pendiente** | Identificado o planificado, no ejecutado. |

Cuando no hay evidencia del porqué, el texto dice **«motivo no documentado»**. Cuando una cifra sale
de las notas de trabajo del asistente y no pudo confirmarse contra el repositorio, dice **«según
notas de trabajo, no verificado en el repo»**.

### Advertencias de lectura

1. **Las cifras de conjuntos y protocolos distintos no se comparan entre sí.** Cada cifra lleva su
   composición (clips × fotogramas, caballos) y su protocolo. Los F1 macro de 141, 153, 177, 218, 254,
   257 y 359 clips, y los de 4 folds o 5×4 repeticiones, son magnitudes diferentes.
2. **Hay dos líneas de desarrollo paralelas, con ramas no fusionadas.** Entre el 28/08 y el 08/09 se
   trabajó en `ejecucion-local-gpu` y `nuevo-intento-con-enmascarado-y-geometria` (conjuntos de 153,
   177, 218 y 254 clips); `main` no contiene esa línea. Desde el 15/09 `main` sigue la línea del MIL
   multimodal sobre 359 clips (ver sección 3).
3. **Evidencia solo recuperable por el reflog local.** Los commits `4c13c20` (cierre de la vía C),
   `1a3859b` (registro «caminos e impedimentos») y `5ad6b64` (plan de filmación) **no pertenecen a
   ninguna rama**: solo son alcanzables desde el reflog local (`feat2/otro-camino@{8}`, `@{9}`, `@{7}`).
   Se preservaron el 08/10/2026 en la rama `rescate/otro-camino-2026-09-08` (apunta a `5ad6b64`, que
   contiene a los otros dos).
4. **Archivos de resultados fuera del control de versiones.** `datos_equinos/resultados/*` está
   ignorado por Git. Las cifras citadas de esos JSON se verificaron en el disco local el 08/10/2026
   (por ejemplo `via_c_evaluacion.json`, `objetivo3_finetuning_evaluacion.json`).

## 2. Línea de tiempo resumida

| Fecha (aprox.) | Hito |
|---|---|
| 05/2026 | Plan de trabajo: clasificar tres expresiones en videos de 5–7 s; hipótesis de que la dinámica temporal aporta. |
| 13/08 | Commit inicial: 34 videos, 92 clips, 22 caballos. Detección YOLO y pose DeepLabCut; alineación por ojo-hocico. |
| 16–17/08 | Persistencia y orquestación en Colab (papermill). Primera corrida completa: 59,5 % de frames alineados. |
| 18–19/08 | Diagnóstico de orientación de video, bodypart del hocico (`nose` vs `upper_jaw`), encuadre canónico imposible. |
| 20/08 | Catálogo por clase. Rama `hector`: CNN-LSTM, `imagededup`. |
| 21–22/08 | Colab pierde el acceso a GPU; el cómputo pasa a la máquina local con GPU. |
| 24–26/08 | Corrida sobre un espejo parcial de videos; fuente única; piso de video corto (2 s). 234 videos, 342 clips. |
| 27/08 | Rama `hector`: aumentación coherente por clip e informe CNN-LSTM (F1 0,72 en una partición). |
| 28/08 | Interpolación de keypoints, piso 1,0 s, anotación manual (141 clips íntegros), encuadre con orejas (224×343). Objetivo 3: F1 0,626, criterio 0,691 no cumplido. Interfaz Streamlit. |
| 05/09 | 25 videos nuevos de `Disgustado`; objetivo 3 rehecho sobre 153 clips: promedio 0,704 contra criterio 0,6747 (cumplido). |
| 06–07/09 | Enmascarado (254×12) y geometría facial (177×8): no se despliegan. Diagnóstico 0 del backbone. Prueba con una persona ajena (n = 1). |
| 08/09 | Vía C (movimiento de cabeza): cerrada con resultado negativo. |
| 15/09 | Integración del MIL multimodal (Héctor): F1 0,754 sobre 257 clips (otra máquina, sin artefacto local). |
| 17–18/09 | Auditoría del pipeline; selección de 16 fotogramas por ventana; 359 clips; validación interna y early stopping: F1 0,740 ± 0,052. |
| 19–21/09 | Documentación y «limpieza de código» (sube `CONF_MINIMA_KEYPOINT` de 0,50 a 0,55; se detecta y revierte el 07/10). |
| 22–23/09 | Ramas laterales: canal geométrico enriquecido, microexpresiones, fine-tuning, videos largos, validación cruzada anidada. Ninguna mejora al modelo. |
| 25/09 | Auditoría del plan de trabajo; líneas de base (el promedio estático iguala al MIL); se conserva el MIL; fine-tuning descartado. |
| 28/09–07/10 | Formulario de la etiquetadora; único observador; se restituye el umbral 0,50. |
| 07/10 | Auditoría v2 y experimento SimBA (RF de pose 0,841). Base común y protocolo de comparación (reserva del 15 %). |
| 08/10 | Cuatro candidatos comparados; gana la fusión pose + visual; un único uso de la reserva (0,834). La app sirve la fusión. Validación leave-session-out: la fusión mantiene 0,864 (C-42). |
| 29/10 | Entrega prevista del informe final. |

## 3. Mapa de ramas

| Rama | Contenido | Estado respecto de `main` |
|---|---|---|
| `main` | Línea vigente: MIL multimodal sobre 359 clips, auditoría, comparación y fusión | — |
| `sergio` | Estado del 20/08 | Ancestro de `main`, sin aportes propios |
| `hector` | CNN-LSTM, `imagededup`, aumentación coherente (08/20–08/27) | 8 commits **no fusionados** (el CNN-LSTM no se fusionó: motivo confirmado por Sergio el 08/10/2026, C-29) |
| `ejecucion-local-gpu` | Migración a GPU local, objetivo 3 y 4 del 28/08, ampliación y objetivo 3 rehecho del 05/09 | 23 commits **no fusionados** (el tramo hasta el 28/08 sí está en `main`). Se retomó desde la línea de Héctor y se dejó esta rama (Sergio, 08/10/2026) |
| `nuevo-intento-con-enmascarado-y-geometria` | Enmascarado, geometría facial, diagnóstico 0, vía C, bitácora de decisiones, prueba con persona ajena | 82 commits **no fusionados**. Se dejó de lado; enmascarado y geometría facial no se retomaron (Sergio, 08/10/2026) |
| `feat/arquitectura-multimodal-mil` | MIL multimodal (15/09) | Fusionada (en `origin` tiene 1 commit de informe más) |
| `feat2/otro-camino` | Revisión v4 del pipeline (18/09) | Fusionada; su historial anterior se perdió en un `reset` (ver advertencia 3) |
| `chore/limpieza-codigo` | Limpieza del 21/09 (introdujo el umbral 0,55) | Fusionada |
| `feat/enriquecer-canal-geometrico` | Canal de 21 dimensiones y control de ruido | **No fusionada** (por decisión: no se adopta) |
| `feat/validacion-cruzada-anidada-NestedCV` | CV anidada | **No fusionada** (no se adopta) |
| `feat/finetuning-visual` | Fine-tuning de `layer4` de la ResNet-18 | **No fusionada** (no se adopta) |
| `origin/feat/microexpresiones-ollares` | Comisuras y ollares en el canal geométrico | Solo en el remoto; **no fusionada** (motivo confirmado por Sergio el 08/10/2026, C-22) |
| `origin/feat/recortes-largos` | Videos largos en la app, en recortes de 5 s | Solo en el remoto; **no fusionada** (respuesta de Sergio el 08/10/2026, C-46) |
| `feat/mejoras-prioridad-alta`, `chore/deuda-menor-y-observador-unico` | Mejoras de la auditoría del 25/09 | Fusionadas |
| `feat/auditoria-y-experimento-simba`, `feat/base-comun-comparacion`, `feat/candidato-baseline`, `feat/candidato-fusion`, `feat/app-modelo-fusion` | Auditoría v2, SimBA, base común, candidatos y app | Fusionadas |
| `feat/candidato-mil`, `feat/candidato-rf-pose` | Resultados de esos dos candidatos (`comparacion/resultados/`) | Su código y sus JSON se trajeron a `main` el 08/10/2026 (`git checkout <rama> -- <archivos>`, sin merge) |

---

## 4. Caminos explorados

### A. Datos y etiquetado

#### C-01. Alcance de clases: plan B de dos clases

- **Qué se intentó:** considerar un problema binario (sin `Disgustado`) como plan alternativo.
- **Por qué:** al 17/08/2026 `Disgustado` provenía de un único caballo (`037`), lo que impedía
  particionar por ejemplar con la clase presente en todos los folds.
- **Resultado:** el motivo desapareció con datos: `Disgustado` pasó a 11 y luego a 32 de 42 caballos
  (25/08) al bajar el piso de duración (C-04); `openspec/project.md` declara que el plan B «queda sin
  motivo». No hay medición de un modelo binario.
- **Decisión:** **Descartado: otra razón** (dejó de ser necesario). Se mantienen las tres clases.
- **Evidencia:** `openspec/project.md` (sección Dataset); `docs/informe_analisis_proyecto.md` §15.

#### C-02. Catálogo de videos declarado por clase

- **Qué se intentó:** declarar en un solo lugar (`equinos/catalogo.py`, `SUBCARPETAS_POR_CLASE`) qué
  subcarpetas suman a cada clase, en lugar de deducir la etiqueta de la carpeta contenedora.
- **Por qué:** el recorrido recursivo ignoraba parte del material etiquetado y aceptaba, sin aviso,
  carpetas anidadas con nombre de clase (una descarga parcial de Drive metía 13 videos como `Sereno`).
- **Resultado:** el catálogo pasó de 122 a 133 videos (`Disgustado` 31 a 41, +32 %).
- **Decisión:** **Adoptado.** Alternativas **descartadas: otra razón**: lista única de subcarpetas para
  todas las clases (la lista no es igual en todas), archivo de configuración externo y recorrido
  recursivo sin declaración (el defecto que se corregía). Copiar los videos a la raíz de proyecto
  también se descartó por duplicar ~2 GB.
- **Evidencia:** change `catalogo-de-videos-por-clase` (proposal y design); commit `f6c9782`.

#### C-03. Fuente única de videos

- **Qué se intentó:** retirar el espejo `datos_equinos/videos` y dejar `~/videos_de_prueba` como única
  fuente, con las rutas en el `kernel.json`.
- **Por qué:** una corrida del 24/08 se hizo sobre 223 de 234 videos (faltaban 12 de `Sereno`, que
  alinean al 55,8 %) y publicó 72,0 % de cobertura, por encima del umbral; el valor sobre el material
  completo era 70,2 %.
- **Resultado:** cobertura corregida a 70,2 %; el pipeline ahora corta ruidosamente si falta la fuente.
- **Decisión:** **Adoptado.** Descartadas por **otra razón**: sincronizar el espejo (dos copias que
  pueden divergir), enlace simbólico (imposible: el disco externo es exFAT), ruta fija en
  `catalogo.py` (ata el código a una máquina).
- **Evidencia:** change `fuente-unica-de-videos` (design); `openspec/project.md`; memoria de trabajo
  «dónde agregar videos nuevos» (el espejo `<repo>/videos` volvió a recibir material nuevo; la trampa
  sigue abierta).

#### C-04. Duración de los clips y piso de video corto

- **Qué se intentó:** sostener el rango de 5–7 s del plan o admitir entero el video más corto que un
  clip (`PISO_VIDEO_CORTO`).
- **Por qué:** el piso de 5 s dejaba entrar solo 11 de 42 caballos de `Disgustado` (video mediano de
  la clase: 2,9 s).
- **Resultado:** con piso 5 s, 280 clips y `Disgustado` 11 caballos; admitiendo entero el video de
  al menos 2 s, 342 clips y 32 caballos (el 26/08). El 28/08 se bajó a 1,0 s: entraron 7 de los 11
  caballos que se perdían, con alineación de 68,0 % (mejor que el 63,1 % del resto de la clase).
- **Decisión:** **Adoptado** (2,0 s y luego 1,0 s), con la desviación del plan declarada: los clips
  de `Disgustado` cubren una ventana temporal más corta. Descartadas: bajar `DURACION_MINIMA` a 2 s
  (diluye a la clase con colas cortas; **otra razón**), bajar `DURACION_CLIP` a 3 s (610 clips que
  son cortes del mismo material; infla el tamaño aparente; **otra razón**), solapamiento (no aporta
  ejemplares de `Disgustado` y encarece la etapa 02: 4,2 h contra 2,7 h; **otra razón**), sostener
  5–7 s (**otra razón**: conserva el problema).
- **Evidencia:** changes `piso-de-video-corto` (design) y `interpolacion-temporal-de-keypoints`;
  commits `d6ae693`, `ea924f2`.
- **Consecuencia que sigue vigente:** la duración del clip separa casi sola a `Disgustado`
  (mediana 3,15 s contra 6,0 s en el conjunto de la vía C); ver C-25 y la lección L-3.

#### C-05. Ampliación del dataset y plan de filmación

- **Qué se intentó:** agregar videos en tandas y, al final, filmar con duraciones comparables entre
  clases.
- **Por qué:** el tamaño muestral efectivo son los caballos, y `Disgustado` limitaba cualquier
  partición por ejemplar.
- **Resultado:** 34 videos (13/08), 146 (22/08), 234 (25/08), 249 (28/08), 274 (05/09). Caballos de
  `Disgustado` con clip íntegro: 5 (25/08), 15 (28/08), 25 (05/09). Ninguna palanca de software
  bastó sola: 15 videos nuevos filmados no movieron la métrica, el relleno temporal llevó a 8
  caballos, el piso de 1,0 s a 10 y 14 frames anotados a mano a 15. El plan de filmación del 08/09
  (+20 ejemplares de `Disgustado`, clips de 5–7 s) **no tiene evidencia de ejecución**: el dataset
  sigue en 274 videos al 08/10.
- **Decisión:** ampliación **adoptada**. Plan de filmación del 08/09: **no ejecutado**, cerrado sin ejecutar
  (ver la respuesta de Sergio del 08/10/2026 más abajo; motivo específico no documentado).
- **Respuesta de Sergio (08/10/2026):** «se fueron probando nuevos modelos pero no daban mejoras o eran peores». *Nota del repositorio:* el plan de filmación no es un
  modelo y la respuesta no explica por qué no se ejecutó; lo que consta es que el dataset quedó en 274
  videos (sin evidencia de filmación nueva desde el 05/09) y que el plan solo existe en el commit
  huérfano `5ad6b64`. Se registra la respuesta tal cual; el motivo específico de no filmar sigue sin
  documentarse.
- **Evidencia:** `openspec/project.md` (tablas de «Dónde quedó el dataset»); change
  `ampliacion-dataset-disgustado-25-videos` y commits `ecff932`, `12d4b5e` (rama
  `nuevo-intento-con-enmascarado-y-geometria`); plan de filmación en el commit huérfano `5ad6b64`
  (`openspec/changes/ampliar-dataset/plan_filmacion.md`, solo en el reflog).

#### C-06. Conflictos y duplicados de etiqueta

- **Qué se intentó:** buscar videos duplicados o con etiquetas contradictorias.
- **Por qué:** `501_8v1_LY1ucwR9.avi` figuraba con el mismo nombre en `Disgustado` e `Interesado`.
- **Resultado:** tres pasadas (byte a byte con md5, mismo ejemplar con igual duración y resolución,
  marcas de tiempo repetidas verificadas con contenido visual): cero duplicados. El caso `178` son dos
  tomas **contiguas** (el ejemplar cambia de expresión), no un duplicado: ruido irreducible en la
  frontera de un evento continuo, que acota el techo de exactitud.
- **Decisión:** el `501` se **excluye** de todos los artefactos; el resto se **mantiene**. Queda
  declarado el límite del `178` para el informe.
- **Evidencia:** change `orientacion-de-video-en-origen` (`duplicados.md`); change
  `fuente-unica-de-videos` (purga del `501`). Deuda abierta: `catalogo.verificar()` solo detecta
  homónimos, no copias con nombres distintos.

#### C-07. Etiquetado: herencia video a clip, protocolo, observador único

- **Qué se intentó:** (a) heredar la etiqueta del video a todos sus clips; (b) documentar un
  protocolo de etiquetado con criterio EquiFACS; (c) medir el acuerdo entre observadores.
- **Por qué:** al 08/09 no había protocolo documentado ni se sabía quién etiquetó; heredar la etiqueta
  asume que la expresión es constante durante todo el video.
- **Resultado:** la etiquetadora (veterinaria, codificadora certificada de EquiFACS) devolvió el
  formulario el 28/09. El acuerdo se midió solo en un piloto sobre imágenes (kappa de Fleiss 0,99);
  sobre el dataset hay un único observador, que además filmó, y el criterio usa contexto y postura.
  El formulario confirma un confusor parcial de establecimiento (box de hipódromo a `Interesado`,
  manada a `Disgustado`).
- **Decisión:** protocolo **adoptado**. Medir el acuerdo interobservador sobre el dataset:
  **descartado: sin medición** (decisión de los autores del 07/10: único observador; queda como
  limitación y trabajo futuro). Dos alternativas anotadas el 08/09 como «sin descartar»: **A**,
  etiqueta débil con la bolsa en el video y no en el clip, y **B**, cambiar la etiqueta a unidades de
  acción (EquiFACS / Horse Grimace Scale). Ambas **descartadas** por decisión de Sergio (08/10/2026):
  no forman parte del modelo final.
- **Evidencia:** `docs/protocolo_de_etiquetado.md`; `docs/formulario_etiquetado_2026-09-28.docx`;
  commit `b75afd6`; `openspec/project.md` (Riesgos 8 y estado del 07/10); `openspec/caminos-e-impedimentos.md`
  del commit huérfano `4c13c20` (I-3 y caminos A y B).

### B. Preprocesamiento

#### C-08. Orientación del video

- **Qué se intentó:** aplicar `CAP_PROP_ORIENTATION_AUTO` en un único punto de lectura
  (`abrir_video()`).
- **Por qué:** 42 de 146 videos guardan la rotación como metadato; OpenCV entregaba el frame acostado
  y el detector no reconocía un caballo girado 90°. Perdían el 55,6 % de sus frames contra 0,0 % de
  los derechos (720 frames en juego, contra 24 que movía la hipótesis del hocico, C-10).
- **Resultado:** corregido y verificado el 19/08. En la inferencia faltaba la misma corrección:
  la auditoría del 07/10 halló que 69 de 274 videos declaran rotación; al normalizar el video como
  el notebook 01 (orientación y `mp4v`), la prueba de punta a punta pasó de 8/16 a 16/16 aciertos
  sobre videos del propio dataset (con el modelo ya entrenado sobre ellos).
- **Decisión:** **Adoptado.** Descartadas por **otra razón**: activar la bandera solo en los videos
  de origen (regla tácita), rotar a mano con `cv2.rotate` (riesgo de rotar dos veces), rotar solo los
  recortes en la inferencia (la pose ya estaría mal), códec sin pérdida en la inferencia (sería
  distinto de lo que vio el modelo).
- **Evidencia:** changes `orientacion-de-video-en-origen` y `base-comun-comparacion-de-modelos`
  (D1, H1–H3); commits `ffc42b6`, `e95d431`, `1f69797`.

#### C-09. Detección y estimación de pose

- **Qué se intentó:** YOLO preentrenado en COCO (clase `horse`) más DeepLabCut `superanimal_quadruped`
  (HRNet-w32) sin ajuste, con cero anotación propia.
- **Por qué:** con 34 videos y 22 caballos, entrenar un detector propio exigiría anotar cientos de
  cuadros y se entrenaría sobre los mismos ejemplares que se quiere generalizar.
- **Resultado:** la pose se persiste para los 39 bodyparts de todos los fotogramas. YOLO dejó de
  participar en la inferencia v4 (el 18/09 se eliminó el gateo de la pose detrás de YOLO, que
  rechazaba videos que DeepLabCut, con su propio detector, sí habría resuelto). La devolución docente
  preguntó por DeepLabCut con Horse-30: **no se usó**; el modelo es generalista y no se ajustó con
  caballos.
- **Decisión:** **Adoptado.** Descartadas: derivar la cabeza como fracción de la caja del cuerpo
  (**otra razón**: no hay relación geométrica estable entre caja y cara; sin medición), detector
  propio entrenado (**sin medición**, por costo), modelo ajustado con Horse-30 (**descartado: no forma parte del modelo final; decisión de Sergio del 08/10/2026, sin medición**; no hay
  referencias en el repo).
- **Evidencia:** change `roi-normalizacion-pose` (design); commit `a6b7bf9`; `auditoria/secciones/12_preprocesamiento_inferencia.md`
  §12.1 y §12.3; `auditoria/secciones/01_plan_de_trabajo.md` §1.5.

#### C-10. Puntos de anclaje de la alineación y bodypart del hocico

- **Qué se intentó:** alinear con una transformada de similitud de dos puntos (ojo y hocico); probar
  `upper_jaw` y una estrategia de «máxima confianza por frame» en lugar de `nose`.
- **Por qué:** la baja cobertura (59,5 %) se atribuyó primero a que el hocico era el eslabón débil.
- **Resultado:** barrido del 19/08 sobre 1680 frames: `nose` 62,4 % (1049), `upper_jaw` 54,4 % (914),
  máxima confianza 63,9 % (1073). La hipótesis queda refutada. La estrategia de máxima confianza gana
  24 frames a cambio de 9,0 % de dispersión en la distancia ojo-hocico.
- **Decisión:** `nose` **adoptado**. `upper_jaw`: **descartado: peor**. Máxima confianza: **descartado:
  sin diferencia** (24 frames no justifican variación sistemática de escala en la señal temporal).
  Anclar en oreja, tres puntos o Procrustes: **descartado: otra razón** (reduce frames alineables, o
  normaliza justamente el rasgo que se quiere clasificar; sin medición).
- **Evidencia:** change `pose-completa-y-bodypart-hocico` (tasks §7, design); change
  `encuadre-que-incluye-las-orejas` (design, decisión 2).

#### C-11. Encuadre del recorte

- **Qué se intentó:** tres encuadres sucesivos: cuadrado de 2,5 `d` de lado (inicial), separación 0,60
  con relleno por replicación (19/08) y recorte rectangular de 224×343 con 1,00 `d` de aire sobre el
  ojo (28/08), siendo `d` la distancia ojo-hocico.
- **Por qué:** el primero pedía 1454 px de lado en cuadros de 1080 px: solo el 2,5 % de los recortes
  salía sin relleno. El segundo cortaba las orejas (observación de Héctor R. Liceaga: son el indicador
  principal de disgusto): el vector base-punta era legible en el 4,3 % de los frames.
- **Resultado:** recortes mayormente vacíos de 97,5 % a 0,2 % (19/08); vector de oreja legible de
  4,3 % a 77,2 % con el relleno mediano igualado (14,1 %). El barrido (`equinos.encuadre.barrido()`)
  predijo el relleno de la corrida real (14,06 % contra 14,1 %).
- **Decisión:** encuadre D (224×343) **adoptado**. Descartadas, todas con medición: mover `FRAC_OJO`
  de 0,20 a 0,25 (propuesta de la rama `hector`: 13,7 % de vector de oreja y más relleno;
  **descartado: peor**), candidato C (50,6 % de vector contra 77,2 %; **peor**), A (+1,1 puntos de
  oreja por +4,7 de relleno; **peor** contra el criterio declarado), E/F/G (superan la línea de base de
  relleno; F queda como segundo escalón), relleno por reflexión, descartar frames con relleno alto y
  dos recortes por frame (**otra razón**), cuadrado más grande (**otra razón**: es el encuadre
  imposible ya descartado). Filtrar recortes con relleno excesivo: **descartado por ahora, sin
  medición**.
- **Evidencia:** changes `encuadre-canonico-y-relleno` y `encuadre-que-incluye-las-orejas` (`medicion.md`);
  commits `de0af1c`, `b8266eb`, `b274309`.

#### C-12. Relleno temporal de keypoints (interpolación)

- **Qué se intentó:** rellenar huecos interiores de pose interpolando las coordenadas de ojo y hocico.
- **Por qué:** el cuello de botella pasó a ser cuántos clips quedan íntegros (16/16 alineados).
- **Resultado:** el umbral declarado antes de medir (mediana ≤ 3,0 px) fue imposible: ninguna
  estrategia lo cumplió. Faltaba una línea de base: el recorte se mueve 16,28 px (mediana) entre dos
  detecciones propias consecutivas; la interpolación de un hueco de 1 da 13,99 px, más precisa que
  usar el vecino detectado. Solo el hueco de largo 1 queda por debajo de la línea de base. Con
  `MAX_HUECO_INTERPOLADO = 1` los clips íntegros pasaron de 81 a 130 y `Disgustado` de 5 a 8
  caballos, no a los 19 proyectados.
- **Decisión:** **Adoptado** con tope 1 (extremos: 0), 28/08. Después **reemplazado** por la
  selección de ventana (C-14): `MAX_HUECO_INTERPOLADO = 1` quedó sin uso en la v4 y los 359 clips son
  todos de detección propia. Descartadas: interpolar la matriz de similitud (**otra razón**: un solo
  camino de construcción y error en píxeles) y rellenar cualquier hueco (**otra razón**: indefendible).
  El error validado es optimista (el error crece con el movimiento, correlación +0,286).
- **Evidencia:** change `interpolacion-temporal-de-keypoints` (design, `validacion.md`); commits
  `95f746d`, `82d5fd9`; `docs/informe_mejora_pipeline_mil.md` §2.2.

#### C-13. Anotación manual de keypoints

- **Qué se intentó:** marcar a mano el par ojo-hocico de los frames que completan un clip, en el
  navegador.
- **Por qué:** cerrar los últimos clips íntegros de `Disgustado` donde ninguna palanca automática
  llegaba.
- **Resultado:** 14 anotaciones (28/08) llevaron `Disgustado` de 10 a 15 caballos; 12 más (05/09)
  sumaron 5 caballos y 7 clips. 9 de 23 candidatos se saltearon (confianza 0 en ambos puntos: cara
  fuera de cuadro o tapada). En la ablación por origen de alineación, con solo detección propia el
  F1 macro cae a 0,518 y el recall de `Disgustado` a 0,000 (7 clips): defiende al preprocesamiento.
  Incidente: la descarga del navegador pisó el almacén de anotaciones (`keypoints_manuales.csv`);
  se reconstruyó verificando las 14 filas contra la corrida.
- **Decisión:** **Adoptado**, luego **reemplazado**: la v4 (359 clips) usa 0 anotaciones manuales
  (C-14). La deuda de que la descarga pueda pisar el almacén **no se arregló** (candidato a change).
- **Evidencia:** commits `75b54c1`, `fa1d217`, `12d4b5e`; change `clasificacion-orientada-a-video`
  (resultado final); memoria de trabajo «la descarga pisa el almacén».

#### C-14. Selección de los 16 fotogramas de cada clip

- **Qué se intentó:** (a) 16 posiciones fijas con `np.linspace` decididas antes de saber qué
  fotogramas tienen keypoints; (b) muestreo estricto de 12 a 16 fotogramas con relleno (Héctor,
  15/09); (c) búsqueda de la ventana contigua (hasta 7 s) con al menos 16 mediciones y sin tramos de
  más de 0,5 s sin medir, con 16 posiciones ancladas al fotograma mejor medido (18/09).
- **Por qué:** con (a), solo 82 de 393 clips tenían los 16 fotogramas válidos; el modelo usaba 153
  (38,9 %) y descartaba el resto. Además, el código aceptaba clips con entre 12 y 16 fotogramas alineados, de modo que hasta 4 de
  los 16 fotogramas de un clip podían ser recortes de caja sin alinear (dos geometrías en una misma
  secuencia), contra lo que declaraban el docstring y `tests/test_particion.py`.
- **Resultado:** con (c), 359 de 393 clips (91,3 %), 133 caballos, `Disgustado` 103 clips en 57
  caballos, ventana mediana 5,77 s. La variante «racha de válidos consecutivos» aceptaba 87 % pero
  con ventana mediana de 2,33 s (44 % por debajo de 2 s): **descartada: otra razón**. Rechazos:
  31 por ventana corta y 3 sin keypoints. (El informe da 93 % de aceptación para el criterio de
  ventana y 91,3 % para el conjunto final; la diferencia no se explica en él.)
- **Decisión:** (c) **adoptado**. (a) **reemplazado**. (b) **reemplazado** (admitía mezclas; motivo
  del abandono posterior no documentado más allá del criterio estricto de 16 alineados restituido).
- **Evidencia:** commit `a6b7bf9`; `docs/informe_mejora_pipeline_mil.md` §1–3; `docs/implementation_plan_2.md`
  y `docs/walkthrough_2.md` (variante b); commit `d872458`.

#### C-15. Umbral de confianza de keypoints

- **Qué se intentó:** `CONF_MINIMA_KEYPOINT` en 0,50 (original), luego en 0,55 durante la «limpieza de
  código» del 21/09.
- **Por qué:** el cambio consta como «resolución punto 4.4» de la limpieza. Motivo del aumento:
  no documentado en el repositorio; ver la respuesta de Sergio del 08/10/2026 más abajo.
- **Resultado:** la interfaz pasó a preprocesar distinto de como se construyó el dataset y
  `tests/test_equivalencia.py` falló; los informes de las ramas laterales del 22–23/09 citan 0,55
  como umbral del proyecto. El valor con que se entrenó el modelo vigente es 0,50.
- **Decisión:** 0,55 **revertido** a 0,50 (07/10). Subir el umbral para reducir el 3,4 % de fotogramas
  pegados al umbral: **no se hará** (propuesto en `informe_mejora_pipeline_mil.md` §9, nunca medido; ver la respuesta de Sergio).
- **Respuesta de Sergio (08/10/2026):** «se fueron probando nuevos modelos pero no daban mejoras o eran peores». *Nota del repositorio:* 0,55 no es un modelo sino un
  parámetro de preprocesamiento; el repo muestra que se introdujo en una limpieza de código, que no
  se midió su efecto sobre el modelo y que se revirtió el 07/10 por romper la equivalencia entre la
  app y el entrenamiento. Se registra la respuesta tal cual.
- **Evidencia:** commits `7dc16c7`, `bff0b90`; `openspec/project.md` (estado del 07/10).

#### C-16. Enmascarado: admitir clips incompletos (254 × 12)

- **Qué se intentó:** en lugar de exigir 16 fotogramas alineados, que todo clip aporte exactamente 12,
  elegidos por una regla fija de 12 posiciones equiespaciadas.
- **Por qué:** el modelo usaba 153 de 393 clips (61 % descartado); `Disgustado` podía pasar de 25 a
  40 caballos sin filmar nada.
- **Resultado:** con umbral 12, un predictor que ve solo la cantidad de fotogramas alineados gana
  +0,139 de F1 macro sobre la clase mayoritaria (con umbral 16, +0,000); por eso se fijó el conteo en 12
  y el atajo quedó en +0,000 exacto. Con tres condiciones (Referencia 153×16, Control 153×12,
  Propuesta 254×12), `promedio` rindió 0,704 ± 0,044, 0,697 ± 0,043 y 0,736 ± 0,016; el material
  nuevo aporta +0,040 contra el Control. Ninguna cabeza cumplió su criterio (0,7847; faltaron 0,048,
  0,033 y 0,018). Precisión de `Disgustado` 55 % a 60 %, dentro del ruido (±8 a ±10 puntos sobre 40 y 68
  predicciones).
- **Decisión:** **Descartado: sin diferencia** para despliegue (no alcanza su propio criterio, la
  mejora está en el ruido y los dos modelos se miden sobre conjuntos distintos). Lo que sí se
  reconoce: desvío ±0,016 contra ±0,044 y 40 contra 25 caballos de `Disgustado`. **Abandonado** al
  adoptarse la selección por ventana (C-14), que persigue el mismo objetivo (recuperar los clips
  descartados) con otro método y mejor aprovechamiento (91,3 %); el cambio de línea no tiene un
  registro de decisión en el repositorio; su motivo lo confirmó Sergio el 08/10/2026 (abajo).
- **Motivo confirmado por Sergio (08/10/2026):** no se retomó porque
  «se fueron probando nuevos modelos pero no daban mejoras o eran peores».
- **Evidencia:** change `admitir-clips-incompletos-con-enmascarado` (rama `nuevo-intento-con-enmascarado-y-geometria`);
  `openspec/bitacora/2026-09-05-y-06.md` (D-11 a D-18, D-29, Parte V); commits `336c2f5`, `8a3bffe`.

#### C-17. Exigir oreja medible en N fotogramas (177 × 8)

- **Qué se intentó:** construir un conjunto en el que todos los fotogramas admitidos tienen oreja
  medible, para que la disponibilidad de la oreja quede constante por construcción (N = 8).
- **Por qué:** la disponibilidad de la oreja predice la clase: un predictor que ve solo cuántos de
  12 fotogramas la tienen gana +0,260 de F1 macro sobre la mayoritaria.
- **Resultado:** 177 clips, 93 caballos, `Disgustado` 35 clips y 25 caballos; reparto en test de
  `Disgustado` {10, 4, 11, 10}. Atajo de disponibilidad de oreja +0,000 exacto, pero el atajo
  posicional queda en +0,143 y el de disponibilidad de la boca en +0,085: igualar «cuántos» mueve la
  fuga a «cuáles». Diseños descartados con medición: K = 6 de 12 (patrón de 12 bits: +0,152;
  **peor/sin ganancia**), N = 12 (un fold con un solo clip de `Disgustado` en test; **otra razón**),
  exigir las dos orejas (5 caballos de `Disgustado`; **otra razón**), misma oreja en los N (21
  caballos; **peor**), los 12 primeros alineados (pierde el último cuarto del evento; **no se
  probó** para no seleccionar sobre el test), imputar la oreja (firma propia del interpolador;
  **sin medición**).
- **Decisión:** el conjunto 177×8 se **construyó** y se usó solo para el factorial (C-24); no se
  desplegó. Ver C-24 para el resultado del modelo.
- **Motivo confirmado por Sergio (08/10/2026):** no se retomó porque
  «se fueron probando nuevos modelos pero no daban mejoras o eran peores».
- **Evidencia:** change `clasificar-desde-la-geometria-facial` (README, `atajo_de_seleccion.md`,
  `costo_del_diseno.md`); `openspec/bitacora/2026-09-05-y-06.md` (D-19 a D-22, Parte III); commits
  `6b301d4`, `dd13443`.

### C. Representación

#### C-18. Backbone: congelado, fine-tuning o red de video

- **Qué se intentó:** (a) ResNet-18 de ImageNet congelada (512 por fotograma) con cabeza chica; (b)
  fine-tuning de `layer4` dentro del `MILFusionModel` (22/09, `feat/finetuning-visual`); (c) redes de
  video de punta a punta (I3D, SlowFast, VideoMAE).
- **Por qué:** con backbone congelado los parámetros entrenables quedan en el orden de 10^4–10^5
  contra pocas muestras; el fine-tuning buscaba que la red aprendiera texturas propias de la tensión
  muscular.
- **Resultado:** el informe de la rama de fine-tuning presenta F1 0,7737 y lo llama «la mejora
  arquitectónica más importante»; ese número es de **una sola partición** de validación (15 %
  hold-out). Con el protocolo de 4 folds por caballo el fine-tuning da **0,747 ± 0,057** contra
  0,740 ± 0,052 del vigente (diferencia indistinguible; confirmado en el JSON local). El commit
  `544d323` concluye «NO LA HAY».
- **Decisión:** (a) **adoptado**. (b) **Descartado: sin diferencia**, y por costo (multiplica los
  parámetros entrenables); decisión de los autores del 25/09. (c) **Descartado: sin sustento de
  datos, sin medición** (memoriza con pocas muestras). Fine-tuning parcial con más capas o con un
  backbone mejor: **descartado: no forma parte del modelo final; decisión de Sergio del 08/10/2026, sin medición** (vía 4 de la bitácora, no ejecutada).
- **Evidencia:** change `clasificacion-orientada-a-video` (design, decisión 1); rama `feat/finetuning-visual`
  (`c6d6f80`, `544d323`, `docs/informe_finetuning.md`); `docs/registro_de_experimentos.md`; change
  `mejoras-prioridad-alta` (D7).

#### C-19. Cambiar el extractor visual («el backbone es viejo»)

- **Qué se intentó:** una pregunta previa: ¿la representación visual conserva la dinámica temporal?
  (diagnóstico 0, 07/09, un minuto de cómputo, sin GPU).
- **Por qué:** Sergio sospechó que la hipótesis temporal fallaba porque la ResNet-18 es antigua.
- **Resultado:** la conserva: dos fotogramas consecutivos están 30 % más cerca que dos al azar
  (salto 1 = 0,700 del azar, correlación salto-distancia +0,969, z = −44); promediar tira 29 % de la
  varianza y `promedio` gana igual. La cantidad de movimiento distingue clases pero no se traduce en
  F1. Conclusión: el problema es la pregunta, no el modelo.
- **Decisión:** como respuesta a la hipótesis temporal, cambiar el extractor quedó **descartado:
  otra razón** (no contesta la pregunta). Como mejora del acierto crudo con un extractor moderno
  (auto-supervisado): **descartado: no forma parte del modelo final; decisión de Sergio del 08/10/2026, sin medición**; la bitácora lo calificaba como el
  experimento más barato.
- **Evidencia:** commit `f0de34d` (`diagnosticos/`); `openspec/bitacora/2026-09-05-y-06.md` (Parte VI,
  vía 1).

#### C-20. Canal geométrico v1 (9 variables): conservarlo o retirarlo

- **Qué se intentó:** retirar el canal geométrico del MIL (MIL visual puro).
- **Por qué:** una ablación accidental (el canal leía los fotogramas 0–15 en lugar de los elegidos por
  la ventana) dio 0,730 contra 0,734 corregida; se interpretó que no aportaba y se recomendó retirarlo.
- **Resultado:** retirarlo de verdad bajó el F1 de 0,740 a 0,686 (precisión de `Disgustado` de 74 % a
  61 %). Un control de ruido de la misma dimensión (9 variables) da 0,680 ± 0,102: el canal real le
  gana por +0,059, por encima del desvío entre folds del protocolo (±0,052). Una ablación debe
  quitar la parte, no corromperle la entrada.
- **Decisión:** retiro **descartado: peor**; canal **conservado** dentro del MIL. La auditoría v2 halló
  después que las 9 variables llevan un solo grado de libertad (la inclinación de la cabeza) y que
  sus máscaras valen siempre 1 (umbral `> 0` y no `≥ 0,50`); ver C-23.
- **Evidencia:** `docs/informe_mejora_pipeline_mil.md` §7; rama `feat/enriquecer-canal-geometrico`
  (`docs/informe_canal_geometrico.md`, commits `4c4f658`, `b0b1379`); `auditoria/secciones/17_experimento_simba.md` §17.4.

#### C-21. Canal geométrico enriquecido (7 keypoints, 21 dimensiones)

- **Qué se intentó:** sumar los siete keypoints medidos (ojos, nariz, mandíbulas, bases de oreja) por
  separado, con control de ruido de la misma dimensión.
- **Por qué:** recomendación 4.1 del informe comparativo: atacar la confusión `Sereno` a
  `Disgustado` con asimetría bilateral y apertura mandibular.
- **Resultado:** F1 0,698 ± 0,103 (contra 0,740 ± 0,052); el margen sobre su control de ruido de 21
  dimensiones (0,680 ± 0,072) es +0,018, dentro del desvío. El desvío se duplica, la época de corte
  baja de 28,5 a 15,2 y `Sereno` a `Disgustado` sube de 20 a 26 errores. Las comisuras (confianza media
  0,39–0,43) y la garganta están bajo el umbral del proyecto, y no existe un keypoint de ollar.
  No se afirma que sea peor (la diferencia de −0,042 cae dentro del desvío).
- **Decisión:** **Descartado: sin diferencia / no mejora.** Hipótesis abierta no ejecutada: reemplazar
  la máscara binaria por la confianza como peso (**descartado: no forma parte del modelo final; decisión de Sergio del 08/10/2026, sin medición**; «dos minutos de
  corrida»).
- **Evidencia:** rama `feat/enriquecer-canal-geometrico`; `docs/registro_de_experimentos.md`;
  `objetivo3_canal_*_evaluacion.json` (verificados localmente).

#### C-22. Comisuras labiales y ollares como entrada separada (15 dimensiones)

- **Qué se intentó:** separar comisuras y ollares del hocico en el canal geométrico, anclando en el eje
  ojo-mandíbula.
- **Por qué:** etológicamente, el disgusto contrae músculos de la boca y dilata los ollares.
- **Resultado:** F1 macro 0,669 ± 0,077 (accuracy 0,681, convergencia en 17,5 épocas) contra una
  línea base citada de ~0,740. El informe de la rama concluye que el ruido de DeepLabCut sobre
  comisuras y ollares supera la señal que aportan.
- **Decisión:** **Descartado: peor** (según su propio informe; **sin fusionar**). Inconsistencias que
  conviene aclarar antes de citarlo: el informe suma 355 clips (101+119+135) y no los 359 del
  conjunto vigente, y afirma haber separado «ollares» cuando el modelo de pose no expone ese punto
  (según `informe_canal_geometrico.md`); no se verificó qué punto se usó. El motivo de no fusionarlo
  lo confirmó Sergio el 08/10/2026 (ver abajo).
- **Motivo confirmado por Sergio (08/10/2026):** no se fusionó porque
  «se fueron probando nuevos modelos pero no daban mejoras o eran peores».
- **Evidencia:** `origin/feat/microexpresiones-ollares` (`bd71559`, `562cd23`,
  `docs/informe_microexpresiones.md`).

#### C-23. Geometría v2 (17 columnas, máscaras ≥ 0,50)

- **Qué se intentó:** reemplazar el canal por vectores base-punta de ambas orejas, vector hocico-mandíbula
  inferior, ancho de boca, distancia interocular, escorzo y orientación, en el marco de la cabeza, con
  máscaras reales.
- **Por qué:** la auditoría del 07/10 halló que las 9 variables vigentes aportaban solo la inclinación
  de la cabeza y un punto de oreja.
- **Resultado:** dimensiones efectivas (95 % de la varianza): 3 en v1 contra 9 en v2. El MIL corregido
  con v2, máscaras 0,50 y validación interna estratificada da 0,722 ± 0,102 (5×4, 113 caballos), tercero
  de cuatro; el vigente es 0,740 ± 0,052 (4 folds, 133 caballos): no son estrictamente comparables.
  La columna de ancho de boca es casi vacía (máscara 0,09).
- **Decisión:** v2 **adoptada** como módulo propio (la v1 queda intacta); el MIL corregido
  **descartado: peor** frente a la fusión (C-35).
- **Evidencia:** commit `fefc67e`; change `base-comun-comparacion-de-modelos` (D2, H4); rama
  `feat/candidato-mil` (`e0f7e15`, `3db7e8e`).

#### C-24. Geometría facial pura (seis derivadas interpretables, 177 × 8)

- **Qué se intentó:** un factorial de 12 celdas: tres representaciones (Apariencia 512, Geometría 6,
  Ambas 518) por cuatro cabezas (`promedio`, `dispersion`, `gru`, `atencion`), con el criterio
  declarado antes (0,6825 = piso 0,6174 + margen 0,0651).
- **Por qué:** los once puntos faciales se usaban para alinear y se descartaban; la oreja es el
  indicador principal de disgusto. La premisa se dividió en (a) importa cuánto cambia la cara y (b)
  importa en qué secuencia.
- **Resultado:** Geometría sola 0,407 a 0,492, **0,184 por debajo** de Apariencia. Solo 1 de 12 celdas
  cumple el criterio (Ambas con `dispersion`, 0,695 ± 0,107, margen +0,012, nueve veces menor que el
  desvío). Los seis testigos invariantes al orden dieron +0,000 exacto. (a) y (b) **no sostenidas**.
  La prueba es menos sensible que la del 05/09 (margen 0,0651 contra 0,0149).
- **Decisión:** **Descartado: peor** (geometría sola) y no sostenida la hipótesis. No se corrieron las
  continuaciones declaradas (geometría sin boca, ablación de `dispersion`, imputar la oreja) porque
  su condición de ejecución (que Geometría ganara) no se dio.
- **Evidencia:** change `clasificar-desde-la-geometria-facial` (`factorial_representacion_cabeza.md`);
  commit `dd13443`; bitácora D-23 a D-28.

#### C-25. Movimiento de cabeza en el cuadro original (vía C)

- **Qué se intentó:** clasificar solo con la trayectoria de posición, escala y ángulo de la cabeza en
  el cuadro original (5 variables por fotograma, 13 fotogramas).
- **Por qué:** la alineación canónica quita traslación, rotación y escala de la cabeza fotograma a
  fotograma: si la señal es que el caballo cabecea, el objetivo 2 la borra a propósito.
- **Resultado:** sobre 218 clips × 13 y 104 caballos (47 clips de `Disgustado`), piso estático
  0,571 ± 0,055 y criterio 0,6265, declarados antes de entrenar. Regresión logística sobre el
  promedio 0,519 ± 0,102; MLP 5→32→3 0,542 ± 0,080; red por fotograma con voto 0,571 ± 0,132; solo
  duración del clip 0,578 ± 0,078; duración más movimiento 0,575 ± 0,071. Ninguno cumple el criterio.
  El «atajo de encuadre» (duración, fps, ancho, alto y escala de la cabeza) llega a 0,614 ± 0,164, el
  confusor más grande del proyecto (+0,435 sobre la mayoritaria).
- **Decisión:** **Descartado: peor / sin señal utilizable** (cierre del 08/09). Nota: el commit de
  cierre y el registro «caminos e impedimentos» llaman a esa cifra de 0,614 «atajo de cámara»,
  mientras que `lineas_de_base.md` de la rama atribuye 0,614 al atajo de encuadre y 0,508 al de cámara.
  Se cita aquí la atribución de `lineas_de_base.md`.
- **Evidencia:** change `clasificar-desde-el-movimiento-de-cabeza` (rama `nuevo-intento-con-enmascarado-y-geometria`,
  `lineas_de_base.md`, `atajos_de_toma.md`; commits `0d77921`, `b9ba1fb`); cierre en el commit huérfano
  `4c13c20` (`via_c.py`, `openspec/caminos-e-impedimentos.md`); `datos_equinos/resultados/via_c_evaluacion.json`
  (local, sin versionar).

#### C-26. Features geométricas tipo SimBA calculadas con código propio

- **Qué se intentó:** 160 features por clip (130 estáticas, 21 dinámicas, 9 de disponibilidad) a partir
  de la pose de los mismos 16 fotogramas, con Random Forest.
- **Por qué:** medir cuánta información hay en la geometría y el movimiento sin red visual profunda.
- **Resultado:** F1 macro 0,851 ± 0,050 (semilla 0) y 0,841 ± 0,003 (media de cinco semillas) contra
  0,740 del MIL (+0,111; IC95 por caballo [0,045; 0,186]); en el subconjunto limpio 0,750 contra 0,705
  (no concluyente). Las features predicen la duración del clip (R² 0,53): deconfundido por filmación
  0,777 (piso). La verificación adversarial confirmó que no hay fuga (partición idéntica, control de
  etiquetas permutadas 0,29–0,32), pero marcó alerta de filmación.
- **Decisión:** **Adoptado** como base del candidato RF de pose y de la rama de pose de la fusión
  (C-33, C-35). La cifra titular es 0,841, no 0,851 (semilla más alta).
- **Evidencia:** `experimentos/simba/README.md` y `verificacion/README.md`; `auditoria/secciones/17_experimento_simba.md`;
  commit `b0f1b26`; change `experimento-comparativo-simba`.

### D. Modelos

#### C-27. Un fotograma con voto y cabezas temporales (objetivo 3, 28/08)

- **Qué se intentó:** 18 configuraciones: tres cabezas (promedio, GRU, atención) por dos regímenes de
  aumentación por dos de capacidad, con muestras igualadas (ventanas de 8 fotogramas) y un clasificador
  de un solo fotograma como línea de base temporal.
- **Por qué:** poner a prueba la hipótesis de que la expresión es un evento dinámico.
- **Resultado:** modelo final: un fotograma con voto, F1 0,626 ± 0,065 contra criterio declarado
  0,691: **no cumple**. Líneas de base: mayoritaria 0,188 ± 0,029; solo identidad con partición ingenua
  0,526 ± 0,116 (+0,338 de fuga); con partición por ejemplar, 0,188. El orden no aportó en ninguna
  configuración: la atención da predicciones idénticas con los fotogramas barajados, la GRU rinde peor
  con el orden real, el embedding de posición aporta 3,5 %. Se verificó que el tensor está en el orden
  declarado. Se repitió el 05/09 sobre 153 clips y 83 caballos: promedio 0,704 ± 0,044 contra criterio
  0,6747 (piso 0,6598 + margen 0,0149), **cumplido**; la ganancia sobre el piso (+0,044) es del tamaño
  de su desvío. GRU 0,681 y atención 0,751 no se informaron como modelo (evitar elegir el máximo de
  tres sobre el mismo test).
- **Decisión:** la hipótesis temporal **no queda sostenida** (se midió dos veces más: C-16 y C-24).
  Cabeza `promedio` **adoptada** (05/09), luego **reemplazada** por el MIL (C-30). Las cifras del 28/08
  (0,626, 0,691) se declararon invalidadas para el conjunto de 153 clips el 05/09.
- **Evidencia:** change `clasificacion-orientada-a-video` (`resultado_final.md`, `cabezas_temporales.md`,
  `lineas_de_base.md`); change `rehacer-objetivo-3-con-dataset-ampliado` y commits `f10b02b`,
  `84e130b`, `30fc1e2` (rama `ejecucion-local-gpu`); `openspec/bitacora/2026-09-05-y-06.md` (D-04 a D-06).

#### C-28. Aumentación temporalmente coherente y capacidad de la cabeza

- **Qué se intentó:** 20 vistas aumentadas por clip (de 106 a 2120 muestras) y capacidad reducida un
  orden de magnitud.
- **Por qué:** descartar que la falta de señal temporal fuera sobreajuste o exceso de capacidad.
- **Resultado:** aumentación: −0,005, −0,006 y +0,000 sobre las tres cabezas (veinte vistas del mismo
  clip siguen siendo el mismo caballo). Capacidad: la media se mueve entre −0,052 y +0,020 sin cambiar
  ninguna conclusión.
- **Decisión:** **Descartado: sin diferencia.** No se reabrió en las sesiones siguientes. De la rama
  `hector` se propuso portar la aumentación coherente y el `WeightedRandomSampler`; ninguno de los dos
  se portó (motivo no documentado más allá de la medición de la aumentación).
- **Evidencia:** change `clasificacion-orientada-a-video` (`cabezas_temporales.md`); bitácora, Parte III;
  commit `38e9ae1` (rama `hector`).

#### C-29. CNN-LSTM (rama `hector`)

- **Qué se intentó:** ResNet-18 parcialmente descongelada con LSTM de 128 unidades, aumentación
  coherente por clip y `WeightedRandomSampler`, sobre 281 clips.
- **Por qué:** arquitectura espacio-temporal del plan original.
- **Resultado:** el informe de la rama declara accuracy 79,4 %, F1 macro 0,72 y recall de `Disgustado`
  0,64 sobre una partición única por sujeto (`GroupShuffleSplit` 78/22, 68 clips de validación). La
  validación de las notas de trabajo y del análisis del 17/09 es que esos números **no tienen corrida
  que los respalde** (ninguna celda ejecutada, ningún `mejor_cnn_lstm.pt` versionado). Además es un
  protocolo sin validación cruzada y no comparable con los demás.
- **Decisión:** **Abandonado.** Motivo de no fusionarlo: confirmado por Sergio el 08/10/2026 (ver arriba); el cierre del
  08/09 lo cita como «no se citan sus resultados».
- **Motivo confirmado por Sergio (08/10/2026):** no se fusionó porque
  «se fueron probando nuevos modelos pero no daban mejoras o eran peores».
- **Evidencia:** rama `hector` (`87459ca`, `9e582d9`, `38e9ae1`, `174dbc8`);
  `docs/informe_etapa_03_modelado_y_entrenamiento.md`; `docs/informe_analisis_proyecto.md` §8.

#### C-30. MIL multimodal (atención sobre fotogramas más canal geométrico)

- **Qué se intentó:** una red de atención sobre los 16 fotogramas con la fusión de la característica
  visual (512) y un MLP geométrico (128), con máscaras de padding.
- **Por qué:** la geometría (orejas) debía darle al clasificador la señal que la apariencia no daba.
- **Resultado:** 15/09 (Héctor): F1 0,754 ± 0,028 sobre 257 clips, otra máquina, 50 épocas fijas,
  **sin artefacto local** («no citar»). Rehecho el 18/09 con selección por ventana, validación
  interna y early stopping: **0,740 ± 0,052** sobre 359 clips y 133 caballos (accuracy 0,744); la
  misma arquitectura sobre los datos viejos (153 clips) bajo ese protocolo da 0,618. Pero: la
  regresión logística sobre el promedio de las 16 características da 0,741 ± 0,020
  (MIL − estática = −0,004; IC95 por caballo [−0,066; +0,058]); los metadatos de filmación solos dan
  0,618; el MIL conserva 0,705 sobre el subconjunto limpio. Verificado numéricamente (0 de 359
  predicciones cambian al barajar o invertir los fotogramas): el MIL **no modela la dinámica
  temporal**.
- **Decisión:** **Adoptado** el 15/09; **conservado** el 25/09 por decisión de los autores (D7,
  revisable); **reemplazado** por la fusión el 08/10 (queda tercero, C-35). Adoptar el promedio estático
  por parsimonia: **descartado: sin diferencia** (era defendible; queda medido).
- **Evidencia:** commits `d872458`, `aff78ee`, `a6b7bf9`, `b16e45b`; `docs/informe_resultados_03.md`;
  `docs/informe_mejora_pipeline_mil.md`; change `mejoras-prioridad-alta` (D7);
  `objetivo3_evaluacion_v2.json`, `objetivo3_lineas_de_base.json`.

#### C-31. Pesos de clase y regularización del MIL

- **Qué se intentó:** dropout 0,0 / 0,2 / 0,3 sobre la representación de la bolsa y pesos de clase
  por frecuencia inversa o por su raíz, decididos con validación interna.
- **Por qué:** `Sereno` a `Disgustado` era el error dominante y la clase `Disgustado` se sobrepredecía
  (113 veces contra 103).
- **Resultado:** F1 de validación interna 0,687 / 0,690 / 0,707 (dropout) y 0,7069 contra 0,7124 (pesos):
  estadísticamente indistinguibles (diferencias de 0,005 a 0,02 contra desvíos de 0,04 a 0,06). En test,
  la raíz da 0,740 contra 0,734.
- **Decisión:** dropout 0,3 y raíz de la inversa **adoptados** «porque es la regla declarada, no porque
  esté demostrada»; el `WeightedRandomSampler` no se probó en el MIL (**sin medición**).
- **Evidencia:** `docs/informe_mejora_pipeline_mil.md` §2.6 y §5.

#### C-32. Líneas de base como candidatos

- **Qué se intentó:** una regresión logística sobre la media de las 512 características de los 16
  fotogramas (sin canal geométrico ni atención).
- **Por qué:** el criterio de que un modelo se mide contra el atajo más fuerte, no contra la mayoritaria.
- **Resultado:** 0,741 ± 0,020 en el protocolo de 4 folds (iguala al MIL); 0,694 ± 0,062 en el
  protocolo 5×4 sobre 113 caballos (limpio 0,597), último de los cuatro candidatos.
- **Decisión:** **Reemplazado** como candidato por la fusión (C-35). Hoy es la línea de base de
  humo del protocolo.
- **Evidencia:** `objetivo3_lineas_de_base.json`; `comparacion/resultados/baseline_logreg_media_frames.json`;
  commit `1dd7db6`.

#### C-33. Random Forest de pose tipo SimBA como candidato

- **Qué se intentó:** el RF sobre las 160 features propias (C-26), evaluado en el protocolo común.
- **Por qué:** la auditoría v2 lo había mostrado superior al MIL.
- **Resultado:** 0,844 ± 0,059 en desarrollo (limpio 0,718 ± 0,020; gap train−test +0,15); segundo,
  detrás de la fusión por +0,026 [+0,002; +0,051] (en limpio, +0,022 [−0,017; +0,056], sin
  distinción).
- **Decisión:** **Descartado: sin diferencia / segundo lugar.** La regla de simplicidad no se aplicó
  porque el IC de la diferencia en desarrollo excluye el 0 (por 0,002).
- **Evidencia:** `comparacion/DECISION.md`; rama `feat/candidato-rf-pose` (`097e98e`, `4fabdf5`);
  `comparacion/resultados/comparaciones_pareadas.json`.

#### C-34. SimBA real

- **Qué se intentó:** instalar y ejecutar SimBA 5.6.1 (`UserDefinedFeatureExtractor`) sobre los 359 clips.
- **Por qué:** la herramienta de nombre homónimo es la referencia del campo.
- **Resultado:** para un solo animal solo entrega movimiento por bodypart (59 columnas por fotograma,
  174 por clip), sin distancias ni ángulos faciales; con regresión logística da 0,720 ± 0,066. La
  importación directa de la pose del proyecto no es posible (formato largo, sin normalizar). Hizo falta
  un stub de `tkinter`.
- **Decisión:** **Descartado: otra razón** (la herramienta no produce features faciales para un
  animal; el conjunto principal es código propio y no se llama SimBA).
- **Evidencia:** `experimentos/simba/README.md`; `simba_real_intento.py`; `compatibilidad_dlc_simba.json`.

#### C-35. Fusión tardía pose más visual

- **Qué se intentó:** combinar las probabilidades de la rama de pose (RF) y de la rama visual
  (regresión logística sobre el promedio ResNet), con el peso `w` elegido por validación interna.
- **Por qué:** unir las dos fuentes sin los costos del MIL.
- **Resultado:** en desarrollo (5×4, 113 caballos, 291 clips) 0,868 ± 0,046 (limpio 0,740 ± 0,028)
  contra RF 0,844, MIL corregido 0,722 y baseline 0,694. Fusión − MIL +0,147 [+0,100; +0,193];
  en limpio +0,122 [+0,020; +0,183]. En la reserva de 20 caballos y 68 clips (un único uso, 08/10):
  F1 macro **0,834**, IC95 por caballo [0,698; 0,925], limpio 0,767. El peso de pose fue 0,75 en 19
  de 20 folds. Salvedades declaradas: la ventaja sobre el RF es mínima y no significativa en limpio;
  la fusión está dominada por la pose; todos los candidatos caen 10–13 puntos en el subconjunto limpio y
  las features de pose predicen la duración (R² 0,53).
- **Decisión:** **Adoptado** como modelo de la app (08/10), con el MIL como opción explícita.
  Descartados para la app: mantener el MIL (peor), servir solo el RF (la regla de simplicidad no
  aplicó), reutilizar `w = 0,75` fijo y copiar `vector_del_clip` (**otra razón**).
- **Evidencia:** `comparacion/DECISION.md`; `comparacion/resultados/fusion_tardia_pose_visual*.json`;
  commits `ea1cf53`, `b5b3a26`, `69b228b`, `30acc23`, `9977bd5`; change `app-modelo-fusion` (design).

### E. Validación

#### C-36. Estrategia de partición

- **Qué se intentó:** partición por clip o por fotograma (inicial), `GroupKFold` por caballo con 3, 4 o 5
  folds, hold-out simple.
- **Por qué:** los 16 fotogramas de un clip son casi idénticos, los clips de un video son contiguos
  y varios videos comparten caballo.
- **Resultado:** solo la identidad con partición ingenua da 0,526 (28/08) o 0,503 (05/09), contra 0,188
  honesto: +0,338 o +0,328 de fuga. Con 5 folds dos folds tenían solo 2 clips de `Disgustado` en test
  (3 con 4 folds; 5 con 3).
- **Decisión:** **partición por caballo, 4 folds, adoptada.** Partición por clip: **descartado: peor**
  (fuga). Hold-out simple: **descartado: otra razón** (la métrica queda a merced de qué caballos caen en
  test). Cinco folds: **descartado: otra razón**. En la auditoría v2: 0 cruces de caballo, video, clip
  o fotograma; riesgo residual de sesión y lugar (C-42).
- **Evidencia:** change `clasificacion-orientada-a-video` (design, decisiones 3 y 4);
  `auditoria/secciones/06_leakage.md`.

#### C-37. Criterio de aceptación declarado antes de medir

- **Qué se intentó:** declarar el umbral antes de entrenar y anclarlo en una línea de base de la misma
  composición.
- **Por qué:** evitar elegir el número que conviene; el criterio de relleno de keypoints se había
  declarado pero sin referencia física y resultó imposible (C-12).
- **Resultado:** 0,691 (28/08, no cumplido); 0,6747 (05/09, cumplido); 0,6825 (177×8), 0,7847
  (254×12) y 0,6265 (vía C), cada uno en su composición. El 25/09, tras medir el promedio estático, el
  criterio del conjunto vigente se re-declaró en 0,761 (el MIL cumple el declarado, 0,691, y no el
  re-declarado): «se re-declaró después de medir, y así consta». Un defecto de implementación estampó
  el criterio de otro conjunto (0,6825) en los artefactos del modelo desplegado: ahora es
  obligatorio.
- **Decisión:** declaración previa anclada **adoptada** como regla. En la comparación final se
  **reemplaza** por el criterio de elección del protocolo (ranking, no inferioridad en limpio, regla de
  empate), también declarado antes de comparar.
- **Evidencia:** commits `f10b02b`, `accb58f`, `31dc36c`, `7b999d1`; `openspec/project.md`;
  `comparacion/PROTOCOLO.md` §5.

#### C-38. Validación interna y early stopping

- **Qué se intentó:** (a) 50 épocas fijas sin validación; (b) validación interna por caballo (los 7
  caballos con más clips) con early stopping; (c) validación interna estratificada por caballo y clase.
- **Por qué:** con 50 épocas no se podía saber si el modelo sobreajustaba; la auditoría halló que (b)
  usaba los 7 caballos con más clips (~10 clips c/u contra 2,1 en el train interno).
- **Resultado:** (b) mejor época 28,5 ± 8,4, ninguna sobre 40; sobreajuste moderado (F1 train 0,979,
  validación 0,712, test 0,740). La caída de 0,704 a 0,618 sobre los mismos 153 clips es el costo de
  apartar ~70 clips para validación, no una degradación. (c) sube la validación a 22–28 caballos.
- **Decisión:** (a) **reemplazado**; (b) **adoptado** en la v4; (c) **adoptado** como opción nueva para
  el protocolo común (`mas_clips` sigue por omisión para reproducir lo medido).
- **Evidencia:** commit `b16e45b`, `4cf9d3e`; `docs/analisis_sobreajuste.md`;
  `auditoria/secciones/07_validacion_early_stopping.md`; change `base-comun-comparacion-de-modelos` (D3, H5).

#### C-39. Validación cruzada anidada

- **Qué se intentó:** elegir la época promediando k curvas de validación interna (modo `anidado`),
  con un modo intermedio `reentrenado` para aislar el efecto.
- **Por qué:** la época elegida con 7 caballos era ruidosa (desvío 8,4 sobre media 28,5); un canal de
  ruido puro elegía épocas entre 12 y 46.
- **Resultado:** sobre el canal real: simple 0,740 ± 0,052; reentrenado 0,755 ± 0,022; anidado
  0,784 ± 0,036, pero el desvío de la época sube de 8,4 a 23,7 (la curva de validación es una meseta;
  épocas elegidas 21, 69, 57, 25). Sobre ruido sí reduce la varianza de la época 87 %. El +0,045 del
  anidado cae dentro del ±0,052 del protocolo; el 0,784 fue una de varias variantes medidas contra el
  mismo test, así que sería optimista. Observación: el modelo de producción se entrena sobre el 100 %
  de los clips, como el modo `reentrenado`, por lo que el 0,740 probablemente subestima al desplegado.
- **Decisión:** **Descartado: peor** (el anidado empeora la selección de época). No se probó la regla
  de un error estándar (**descartado: no forma parte del modelo final; decisión de Sergio del 08/10/2026, sin medición**; «la mejora no justificaba seguir»). El 0,755 de `reentrenado`
  no se adoptó como cifra de referencia (decisión de alcance mayor).
- **Evidencia:** rama `feat/validacion-cruzada-anidada-NestedCV` (`23d4b25`,
  `docs/informe_validacion_cruzada_anidada.md`); `objetivo3_nested_*_evaluacion.json` (local).

#### C-40. Protocolo común 5×4 con reserva del 15 % de los caballos

- **Qué se intentó:** reserva de 20 caballos (68 clips) sorteada con semilla fija, estratificada por
  clase dominante del caballo; desarrollo de 113 caballos y 291 clips con 5 repeticiones por 4 folds;
  criterio de elección y bootstrap pareado por caballo (2000 réplicas), todo declarado antes de comparar.
- **Por qué:** elegir el mejor clasificador con incertidumbre acotada y una estimación final fuera de
  muestra; la auditoría del 07/10 recomendó repetir la validación cruzada (al menos 5 particiones).
- **Resultado:** 20 evaluaciones por candidato, las mismas particiones (huella `089b347d4d…`).
  La reserva se usó una sola vez, con autorización explícita, y se registró en `reserva_uso.log`.
- **Decisión:** **Adoptado.** Quedan declarados como límites: el agrupamiento es por caballo y no por
  sesión; hay un solo observador; las repeticiones remuestrean la partición, no los datos.
- **Evidencia:** `comparacion/PROTOCOLO.md`; `comparacion/DECISION.md`; commits `3eaa38e`, `b5b3a26`,
  `69b228b`.

#### C-41. Subconjunto limpio y control de confusores de filmación

- **Qué se intentó:** reportar el rendimiento sobre clips 1080×1920 y de al menos 5 s, que no delatan la
  clase por el formato.
- **Por qué:** los metadatos de filmación solos dan 0,618 (duración sola 0,544; fecha de sesión
  0,535).
- **Resultado:** 188 clips en desarrollo (23 de `Disgustado`); todos los candidatos caen entre 10 y 13
  puntos (fusión 0,740, RF 0,718, MIL 0,618, baseline 0,597). No se re-entrenó sobre el subconjunto
  (cambiaría el material); reentrenar solo con limpios baja a 0,678 (RF) y 0,606 (baseline).
- **Decisión:** **Adoptado** como control obligatorio. **Recortar todos los clips a la misma duración**
  (corrección de fondo): **descartado: sin medición**, por costo (cambia el dataset y obliga a
  re-ejecutar la etapa 02); queda como recomendación.
- **Evidencia:** change `mejoras-prioridad-alta` (D4); `auditoria/secciones/` (caps. 10 y 13);
  `experimentos/simba/README.md`.

#### C-42. Leave-session-out

- **Qué se intentó:** una validación que deje afuera sesiones de filmación completas, no solo caballos.
- **Por qué:** caballos de una misma sesión pueden quedar a ambos lados de la partición, y el contexto
  de captura contiene señal.
- **Resultado:** la tabla de sesiones se completó el 08/10 (todos los videos se filmaron en Tandil, 22
  sesiones) y ese mismo día se ejecutó el análisis (change `validacion-por-sesion`). Partición: test =
  una sesión; train = las otras sesiones sin los caballos presentes en la de test; 20 sesiones con
  clips de desarrollo, ninguna saltada, 40 clips de train perdidos por la exclusión; 291 predicciones
  fuera de muestra, sin tocar la reserva. F1 macro agrupado [IC95 bootstrap por caballo]: fusión 0,864
  [0,793; 0,921], RF de pose 0,831 [0,774; 0,885], MIL corregido 0,690 [0,606; 0,763], baseline 0,628
  [0,555; 0,696]. Respecto del 5×4 (mismas métricas agrupadas) la fusión y el RF no caen (−0,002 y
  +0,005), el MIL cae 0,025 y el baseline 0,068. Fusión − RF: +0,032 [−0,006; +0,072], ya no se
  distingue de cero; fusión − MIL: +0,174 [+0,105; +0,245]; fusión − baseline: +0,236 [+0,177; +0,301].
  En el subconjunto limpio (188 clips) la fusión da 0,744 y el RF 0,715, unos 12 puntos por debajo
  del total, como en el 5×4.
- **Decisión:** **Adoptado como análisis complementario** (ya no pendiente). El ranking y el nivel de
  los candidatos con pose se sostienen ante sesiones no vistas, así que la ventaja no se explica por
  rasgos de cada sesión; el confusor de duración y formato no se descarta (limpio sigue unos 12
  puntos abajo). La ventaja de la fusión sobre el RF de pose no se distingue de cero: se declara como
  limitación; el ganador no cambia, porque el criterio estaba fijado antes (`PROTOCOLO.md` §5).
- **Evidencia:** `comparacion/validacion_por_sesion.py`, `comparacion/resultados/validacion_por_sesion.json`;
  `comparacion/DECISION.md` (sección «Validación por sesión de filmación»); `comparacion/PROTOCOLO.md`
  (enmienda del 08/10/2026); `comparacion/sesiones_de_filmacion.csv`; commit `59f5144`.

### F. Entorno de ejecución

#### C-43. Colab con GPU o máquina local con GPU

- **Qué se intentó:** migrar el cómputo a la máquina local (Ubuntu 22.04, RTX 3070 Laptop 8 GB,
  `torch 2.5.1+cu121`) en un entorno aislado.
- **Por qué:** Colab cortó el acceso a GPU por límites de la suscripción (21/08).
- **Resultado:** la creencia previa de que «no hay GPU utilizable» venía de una causa mal atribuida
  (un `torch` compilado para CUDA 13 contra un driver 535 con CUDA 12.2, no la GPU); con una rueda
  `cu121` la GPU responde (YOLO11x 153 ms por frame, DeepLabCut 394 ms por frame). La etapa 02 corrió a
  29 s por clip. Una re-ejecución desde cero (14–15/09) habría reproducido exacto el objetivo 3 de
  entonces (según notas de trabajo, no verificado en el repo). Es un portátil de 80 W (throttling
  térmico: afecta tiempos, no resultados).
- **Decisión:** **Adoptado**, conservando Colab como ruta válida. Descartadas, por **otra razón**:
  subir el driver NVIDIA a la serie 580 (puede dejar un portátil de gráficos híbridos sin entorno
  gráfico), instalar sobre el Python del sistema (numpy 2 contra el `numpy<2` de DeepLabCut), abandonar
  Colab (deja sin ruta al otro integrante), re-correr todo desde cero como camino de entrada
  (~4 h de GPU; se hizo después como validación). La tarea de verificar las celdas de arranque en
  Colab sigue abierta.
- **Evidencia:** change `ejecucion-local-con-gpu` (proposal, design); `docs/informe_infraestructura_y_hardware.md`;
  memoria de trabajo «entorno local».

#### C-44. Persistencia y orquestación en Colab

- **Qué se intentó:** raíz persistente única, checkpoints por unidad de trabajo con invalidación por
  configuración, caché comprimida y orquestador `00_pipeline` con `papermill`.
- **Por qué:** disco efímero y cortes de sesión; reanudar con otra configuración mezcla dos corridas
  sin avisar.
- **Resultado:** todas las rutas de los CSV son relativas a la raíz; el pipeline es portable entre Colab y
  local. Costos: `etapas.estado()` informa «completa» sin comparar contra el catálogo (con material
  nuevo hay que llamar `etapa01` y `etapa02` a mano); la purga de unidades retiradas es manual.
- **Decisión:** **Adoptado.** Descartadas: `%run` y `nbconvert --execute` (no ejecutan `.ipynb` o no
  inyectan parámetros), repositorio público para simplificar el clonado (**otra razón**: decisión del
  equipo, no de infraestructura).
- **Evidencia:** change `persistencia-y-orquestacion-colab` (design); `openspec/project.md` (Deudas).

### G. Aplicación

#### C-45. Qué modelo sirve la interfaz

- **Qué se intentó:** interfaz Streamlit que ejecuta el mismo pipeline que el entrenamiento, con
  cuatro modelos sucesivos: un fotograma con voto (28/08), `promedio` (05/09, solo en las ramas no
  fusionadas), MIL multimodal (15/09) y fusión pose más visual (08/10).
- **Por qué:** objetivo 4; mostrar un resultado con la confiabilidad medida, derivada del archivo de
  evaluación (ningún número a mano).
- **Resultado:** la tasa de respuesta era 42 % y pasó a 67 % al buscar la ventana dentro del video
  (28/08). La cifra de 58 % del 05/09 comparaba otra muestra: el generador aleatorio se compartía entre
  clases; hoy la muestra se persiste. Con la fusión: prueba de punta a punta 16/16 y equivalencia
  entrenamiento-inferencia verificada. La tasa de negativa de 42 % (anterior a la selección por ventana)
  **no se volvió a medir** tras la v4.
- **Decisión:** la fusión **adoptada** para la app; el MIL queda como opción `modelo="mil"`. Los tres
  anteriores **reemplazados**, cada uno por una medición de la sección correspondiente.
- **Evidencia:** commits `55c6cd3`, `4baac03`, `aff78ee`, `9977bd5`, `a19b90a`; changes
  `interfaz-de-inferencia`, `servir-el-modelo-del-objetivo-3` (rama `ejecucion-local-gpu` y
  `nuevo-intento-con-enmascarado-y-geometria`) y `app-modelo-fusion`; bitácora E-2.

#### C-46. Videos largos en la app (recortes de 5 s)

- **Qué se intentó:** subir videos de hasta 30 s y dividirlos al vuelo en recortes consecutivos de 5 s,
  con una pestaña de análisis por recorte.
- **Por qué:** una función de uso, no una mejora del modelo.
- **Resultado:** sin medición; la app vigente declara que «con videos más largos no funciona bien».
- **Decisión:** **Abandonado** (rama `origin/feat/recortes-largos` sin fusionar); motivo en el repositorio:
  no documentado (ver la respuesta de Sergio del 08/10/2026).
- **Respuesta de Sergio (08/10/2026):** «se fueron probando nuevos modelos pero no daban mejoras o eran peores». *Nota del repositorio:* los videos largos son una
  función de la app y no un modelo; la respuesta se registra tal cual y no cambia lo que consta: sin
  medición, rama sin fusionar y la app declara su limitación.
- **Resolución parcial de la observación D4 del docente (08/10/2026):** sin dividir en recortes, la app
  ahora **informa el tramo analizado** («Se analizó el tramo del segundo X al segundo Y del video (de
  Z s en total)») y, si el video dura más que la ventana de 7 s, **recomienda subir un video recortado**
  al momento de interés (de 5 a 7 segundos). La selección de la ventana y el modelo no cambiaron.
  Estado de D4: Resuelta. Prueba: `tests/test_tramo_analizado.py`.
- **Evidencia:** commit `7803e90`; `app_inferencia.py` (texto de limitaciones).

#### C-47. Textos de confiabilidad y prueba con una persona ajena

- **Qué se intentó:** expresar la confiabilidad como precisión por clase y como frecuencia natural
  («se equivoca en 4 o 5 de cada 10»), con aviso de alcance junto al resultado; probar los textos con
  una persona ajena al trabajo.
- **Por qué:** la persona que probó la interfaz llegó a la conclusión correcta por el camino equivocado
  (desconfió por creer el sistema provisional), entendió el 79 % pero no lo aplicó, y no le dio
  importancia al cartel rojo.
- **Resultado:** n = 1: hipótesis de diseño, no medición. En `main` los textos de la fusión se
  reescribieron en lenguaje llano y la confiabilidad se da como rango entre desarrollo y reserva.
- **Decisión:** **Adoptado.** Repetir la prueba con otra persona: **no se realiza** (decisión de Sergio del 08/10/2026 de cerrar el
  trabajo; los textos se reescribieron después con su revisión directa).
- **Evidencia:** commits `d6fb240`, `c15aac7`, `f4071ac` (rama `nuevo-intento-con-enmascarado-y-geometria`,
  `prueba_con_persona_ajena.md`); commits de `main` `935753d`, `0aab1eb`, `6090154`, `3a39454`, `36db174`, `d471e66` y `fb5fa21`;
  `openspec/bitacora/2026-09-06-y-07.md`.

---

## 5. Tabla resumen

| Id | Camino | Etapa | Decisión | Motivo en una línea | Evidencia |
|---|---|---|---|---|---|
| C-01 | Plan B de dos clases | Datos | Descartado: otra razón | Dejó de ser necesario al llegar a 32 caballos de `Disgustado` | `project.md` |
| C-02 | Catálogo declarado por clase | Datos | Adoptado | Etiqueta del catálogo, no de la carpeta; +11 videos | `catalogo-de-videos-por-clase` |
| C-03 | Fuente única de videos | Datos | Adoptado | Una corrida usó un espejo parcial y publicó 72,0 % falso | `fuente-unica-de-videos` |
| C-04 | Piso de video corto (2 s, luego 1,0 s) | Datos | Adoptado | Recupera a `Disgustado` (11 a 32 caballos) con desviación declarada | `piso-de-video-corto`, `ea924f2` |
| C-05 | Ampliación del dataset / plan de filmación | Datos | Adoptado / Plan de filmación no ejecutado | Tamaño muestral = caballos; el plan del 08/09 no se ejecutó (respuesta de Sergio, 08/10/2026: «se fueron probando nuevos modelos pero no daban mejoras o eran peores»; motivo específico no documentado) | `project.md`, `5ad6b64` |
| C-06 | Búsqueda de duplicados y conflictos | Datos | Adoptado (501 excluido) | 0 duplicados; `178` es contiguo, no duplicado | `orientacion-de-video-en-origen` |
| C-07 | Protocolo de etiquetado / observador único / etiqueta débil / EquiFACS | Datos | Adoptado / Sin medición / A y B descartados | Un solo observador; A y B no forman parte del modelo final (decisión de Sergio, 08/10/2026) | `protocolo_de_etiquetado.md`, `b75afd6` |
| C-08 | Orientación de video | Preproc. | Adoptado | 42/146 videos rotados perdían 55,6 % de frames; 8/16 a 16/16 en la app | `orientacion-de-video-en-origen`, `base-comun` |
| C-09 | YOLO + DLC SuperAnimal, sin ajuste | Preproc. | Adoptado | Cero anotación propia; detector propio y Horse-30 sin evaluar | `roi-normalizacion-pose` |
| C-10 | Hocico `upper_jaw` / máxima confianza | Preproc. | Descartado: peor / sin diferencia | 54,4 % contra 62,4 % de `nose` | `pose-completa-y-bodypart-hocico` |
| C-11 | Encuadre 224×343 con orejas | Preproc. | Adoptado | Vector de oreja 4,3 % a 77,2 %; alternativas medidas peores | `encuadre-que-incluye-las-orejas` |
| C-12 | Relleno temporal de keypoints | Preproc. | Adoptado, luego Reemplazado | Solo el hueco de 1 supera la línea de base; la ventana lo dejó sin uso | `interpolacion-temporal-de-keypoints` |
| C-13 | Anotación manual de keypoints | Preproc. | Adoptado, luego Reemplazado | +5 caballos; la v4 usa 0 anotaciones | `12d4b5e` |
| C-14 | 16 fotogramas por ventana temporal | Preproc. | Adoptado | 153 a 359 clips (91,3 %); 16 fijos y muestreo variable reemplazados | `a6b7bf9` |
| C-15 | Umbral de confianza 0,55 | Preproc. | Revertido a 0,50 | Cambió la preparación de la app sin avisar; respuesta de Sergio (08/10/2026): «se fueron probando nuevos modelos pero no daban mejoras o eran peores» | `7dc16c7`, `bff0b90` |
| C-16 | Enmascarado 254×12 | Datos/Modelo | Descartado: sin diferencia; Abandonado | No cumple su criterio (0,736 contra 0,7847); mismo objetivo logrado por C-14; no se retomó (Sergio, 08/10/2026: «se fueron probando nuevos modelos pero no daban mejoras o eran peores») | bitácora D-29 |
| C-17 | Oreja medible en N fotogramas (177×8) | Datos | Descartado (diseños peores); construido | La fuga pasa de «cuántos» a «cuáles» (+0,143); no se retomó (Sergio, 08/10/2026: «se fueron probando nuevos modelos pero no daban mejoras o eran peores») | `clasificar-desde-la-geometria-facial` |
| C-18 | Backbone: congelado / fine-tuning / red de video | Representación | Congelado adoptado; fine-tuning descartado: sin diferencia | 0,747 ± 0,057 contra 0,740 ± 0,052 | `feat/finetuning-visual` |
| C-19 | Cambiar el extractor visual | Representación | Descartado: no forma parte del modelo final (sin medición) | El diagnóstico 0 lo descartó como respuesta a la hipótesis temporal | `f0de34d` |
| C-20 | Retirar el canal geométrico v1 | Representación | Descartado: peor | 0,686 sin canal; el canal le gana al ruido por +0,059 | `informe_mejora_pipeline_mil.md` §7 |
| C-21 | Canal enriquecido de 21 dimensiones | Representación | Descartado: sin diferencia | 0,698 ± 0,103; margen sobre ruido +0,018 | `feat/enriquecer-canal-geometrico` |
| C-22 | Comisuras y ollares (15 dimensiones) | Representación | Descartado: peor (sin fusionar) | 0,669 ± 0,077; ruido de pose mayor que la señal; no fusionado (Sergio, 08/10/2026: «se fueron probando nuevos modelos pero no daban mejoras o eran peores») | `origin/feat/microexpresiones-ollares` |
| C-23 | Geometría v2 (17 columnas) | Representación | Adoptado (módulo); MIL corregido descartado | Descripción más rica; el MIL con v2 no supera al vigente | `fefc67e` |
| C-24 | Geometría pura (6 derivadas), factorial 12 celdas | Representación | Descartado: peor | 0,184 por debajo de Apariencia; 1/12 celdas cumple | `clasificar-desde-la-geometria-facial` |
| C-25 | Movimiento de cabeza (vía C) | Representación | Descartado: peor / sin señal | Ningún modelo supera el piso 0,571; la duración sola da 0,578 | `4c13c20` (reflog), `b9ba1fb` |
| C-26 | Features tipo SimBA propias (160) | Representación | Adoptado | RF 0,841 contra MIL 0,740, con alerta de filmación | `experimentos/simba` |
| C-27 | Un fotograma con voto / 18 cabezas temporales | Modelo | Hipótesis temporal no sostenida; `promedio` adoptado, luego Reemplazado | El orden no aportó en ninguna de las 18 configuraciones | `clasificacion-orientada-a-video` |
| C-28 | Aumentación coherente y capacidad | Modelo | Descartado: sin diferencia | −0,005 a +0,000 con 20 vistas por clip | `cabezas_temporales.md` |
| C-29 | CNN-LSTM (`hector`) | Modelo | Abandonado | Partición única, sin corrida que lo respalde; no se fusionó (Sergio, 08/10/2026: «se fueron probando nuevos modelos pero no daban mejoras o eran peores») | `docs/informe_etapa_03_...` |
| C-30 | MIL multimodal | Modelo | Adoptado, luego Reemplazado | 0,740 ± 0,052 = promedio estático 0,741; tercero en la comparación | `objetivo3_evaluacion_v2.json` |
| C-31 | Dropout y pesos de clase | Modelo | Adoptado por regla declarada | Diferencias indistinguibles del ruido | `informe_mejora_pipeline_mil.md` |
| C-32 | Regresión logística sobre la media de frames | Modelo | Reemplazado como candidato | Cuarto: 0,694 en 5×4 | `baseline_logreg_media_frames.json` |
| C-33 | RF de pose tipo SimBA como candidato | Modelo | Descartado: sin diferencia | Segundo: 0,844; −0,026 [−0,051; −0,002] frente a la fusión | `DECISION.md` |
| C-34 | SimBA real | Modelo | Descartado: otra razón | Para un animal solo da movimiento (0,720) | `experimentos/simba` |
| C-35 | Fusión pose más visual | Modelo | Adoptado | 0,868 en desarrollo y 0,834 en reserva (único uso) | `DECISION.md`, `69b228b` |
| C-36 | Partición por clip, hold-out, 5 folds | Validación | Descartado: peor / otra razón | La identidad sola da +0,338 con partición ingenua | `clasificacion-orientada-a-video` |
| C-37 | Criterio declarado y anclado | Validación | Adoptado (reemplazado por criterio de elección) | Evita elegir el número que conviene; re-declaración del 25/09 consta como tal | `f10b02b`, `PROTOCOLO.md` §5 |
| C-38 | Validación interna y early stopping | Validación | Adoptado | 50 épocas fijas no permitían detectar sobreajuste | `b16e45b`, `4cf9d3e` |
| C-39 | Validación cruzada anidada | Validación | Descartado: peor | Triplica el desvío de la época sobre el canal real | `feat/validacion-cruzada-anidada-NestedCV` |
| C-40 | Protocolo 5×4 con reserva del 15 % | Validación | Adoptado | Incertidumbre acotada y una estimación final fuera de muestra | `PROTOCOLO.md`, `3eaa38e` |
| C-41 | Subconjunto limpio; recortar clips a igual duración | Validación | Adoptado / Sin medición | Los candidatos pierden 10–13 puntos; igualar duración cambia el dataset | `mejoras-prioridad-alta` (D4) |
| C-42 | Leave-session-out | Validación | Adoptado (análisis complementario) | El ranking y el nivel de la fusión se sostienen con sesiones no vistas (0,864); fusión − RF ya no se distingue de cero | `validacion_por_sesion.json`, `DECISION.md` |
| C-43 | Colab a máquina local con GPU | Entorno | Adoptado | Colab cortó la GPU; driver 580, Python del sistema y abandonar Colab descartados | `ejecucion-local-con-gpu` |
| C-44 | Persistencia y orquestación (papermill) | Entorno | Adoptado | Cortes de sesión; reanudar con otra configuración mezcla corridas | `persistencia-y-orquestacion-colab` |
| C-45 | Modelo servido por la app | App | Fusión adoptada; anteriores Reemplazados | Cada reemplazo se apoya en una medición | `app-modelo-fusion` |
| C-46 | Videos largos en la app (recortes de 5 s) | App | Abandonado | Rama sin fusionar; respuesta de Sergio (08/10/2026): «se fueron probando nuevos modelos pero no daban mejoras o eran peores» (la función no es un modelo) | `7803e90` |
| C-47 | Textos de confiabilidad / prueba con persona ajena | App | Adoptado; repetición no realizada | n = 1 | `f4071ac` |

## 6. Lecciones de método

Las siguientes surgen de la evidencia citada, no de una síntesis posterior. Varias figuran con el
nombre de la nota de trabajo en que se registraron.

- **L-1. Medir contra el denominador elegido a propósito.** La cobertura de 68,1 % era 51,9 % contra los
  frames esperados (400 frames de videos rotados que el detector no veía); una cobertura de 72,0 %
  pasaba el umbral porque faltaban los 12 videos que peor alinean; una cobertura global que sube
  73,5 % a 73,9 % no reflejaba mejora alguna; una tasa de respuesta de la interfaz superó el techo del
  pipeline; dos tasas (67 % y 58 %) se compararon siendo muestras distintas. Tabla de supervivencia de
  la oreja presentada sobre el conjunto equivocado (bitácora E-4). Ver C-03, C-13, C-45.
- **L-2. Un umbral declarado antes de medir debe estar anclado en una línea de base física.** El umbral
  de 3,0 px para el relleno exigía cinco veces más precisión que el movimiento natural del caballo
  entre fotogramas (16,28 px). Declararlo antes protege contra elegir el número que conviene; anclarlo
  distingue «el método no sirve» de «el criterio era imposible». Ver C-12, C-37.
- **L-3. La vara es el atajo más fuerte, no la clase mayoritaria.** Sin mirar un píxel: identidad del
  caballo con partición ingenua (+0,328 a +0,460), disponibilidad de la oreja (+0,260), cantidad de
  fotogramas alineados (+0,139), posiciones elegidas (+0,143), disponibilidad de la boca (+0,085),
  encuadre y duración del clip (+0,435). Ver C-16, C-17, C-25, C-41.
- **L-4. Dónde falla la detección predice la etiqueta.** Un caballo disgustado se mueve distinto y el
  detector falla distinto; cualquier pipeline que filtre por calidad de detección hereda una fuga
  correlacionada con la etiqueta. La técnica que funcionó es hacer constante por construcción la
  variable que filtra, y verificarlo sobre el tensor entregado. Igualar «cuántos» mueve la fuga a
  «cuáles». Ver C-16, C-17.
- **L-5. Dos decisiones cerradas por separado pueden ser incompatibles.** `N = 15` se fijó con el
  predicado del centroide; con el predicado que la fórmula necesita (eje utilizable) dejaba un fold de
  test sin un solo clip de `Disgustado` (cobertura efectiva 72,7 %, no 89,8 %). El predicado que admite
  el dato no es el que la fórmula necesita; el error se corrigió sin borrar el cierre superado. Ver
  C-25.
- **L-6. Una ablación debe quitar la parte, no corromperle la entrada.** La ablación accidental del
  canal geométrico sugirió retirarlo; retirarlo de verdad lo refutó. Para separar información de
  capacidad hace falta un control de ruido de la misma dimensión. Ver C-20, C-21.
- **L-7. No comparar cifras de composiciones o protocolos distintos.** 0,618, 0,704 y 0,754; el 0,7737
  del informe de fine-tuning (una partición) contra 0,747 ± 0,057 (4 folds); el 0,784 de la validación
  anidada (una variante entre varias, sobre el mismo test); los márgenes sí se comparan, las medias no.
  Ver C-18, C-30, C-39.
- **L-8. No elegir sobre el conjunto de evaluación.** Se informó `promedio` y no `atencion` (0,751)
  para no tomar el máximo de tres sobre el mismo test; las dos reglas de selección de fotogramas no se
  compararon por su métrica; la reserva se usó una sola vez, con el ganador ya registrado por escrito.
  Ver C-27, C-40.
- **L-9. Una advertencia escrita no es un control.** El criterio de aceptación tomado por omisión de
  una constante global estampó la vara de otro conjunto en los artefactos del modelo desplegado, con el
  archivo bien formado y sin que nada fallara; hoy es obligatorio y probado en ambas direcciones.
  También: `etapas.estado()` informa «completa» sin comparar contra el catálogo; la descarga del
  navegador pisó el almacén de anotaciones; el umbral 0,55 cambió el preprocesamiento de la app sin
  aviso. Ver C-13, C-15, C-37, C-44.
- **L-10. Entrenamiento e inferencia deben compartir implementación.** Dos implementaciones habían
  divergido (la interfaz era más estricta que el pipeline que fabricó el entrenamiento y mostraba
  métricas de otro modelo); la corrección de la orientación llevó la prueba de punta a punta de 8/16 a
  16/16. Ver C-08, C-14, C-35.
- **L-11. Un resultado negativo bien medido es un resultado.** El objetivo 3 se informó con el criterio
  no cumplido (28/08), la hipótesis temporal quedó sin sostener cuatro veces (18 configuraciones,
  factorial de geometría, enmascarado, vía C) y las tres recomendaciones del informe comparativo
  (canal enriquecido, fine-tuning, validación anidada) se midieron y no se adoptaron, cada una con la
  explicación medida de su fallo. Ver C-18, C-21, C-24, C-25, C-27, C-39.
- **L-12. Persistir la muestra, no la semilla.** Una semilla fija no vuelve reproducible un muestreo si
  el generador se comparte entre estratos y la población cambia. Ver C-45.
- **L-13. Una conclusión sobre el etiquetado exige mirar lo que vio quien etiquetó.** El caso del
  ejemplar `13` se leyó primero sobre doce recortes estáticos como límite subjetivo de etiquetado; el
  video completo mostró que la etiqueta era correcta y los errores del modelo. Ver C-07.
- **L-14. La ventaja de un modelo debe separarse del confusor de filmación.** El RF de pose y la
  fusión pierden 10–13 puntos en el subconjunto limpio y sus features predicen la duración (R² 0,53); el
  MIL pierde menos, pero no supera a un promedio estático. La reserva de 20 caballos (IC95 [0,698;
  0,925]) no sirve para comparar candidatos. Ver C-33, C-35, C-41.

## 7. Puntos sin motivo documentado o a confirmar

Estado al 08/10/2026, después de las respuestas de Sergio. Los puntos 1 a 14 de la versión anterior
se resolvieron como se indica; quedan tres notas menores, explícitas, que no bloquean el informe.

### Resueltos con la respuesta de Sergio (08/10/2026)

Para los motivos de los puntos 1, 3, 4, 5, 6 y 7 de la versión anterior (retomar desde la rama de
Héctor y dejar `ejecucion-local-gpu` y `nuevo-intento-con-enmascarado-y-geometria`; no retomar
enmascarado ni geometría facial; plan de filmación del 08/09; no fusionar el CNN-LSTM de `hector`,
`feat/recortes-largos` ni microexpresiones; subir `CONF_MINIMA_KEYPOINT` a 0,55), la respuesta de
Sergio fue: «se fueron probando nuevos modelos pero no daban mejoras o eran peores». Se registró como **motivo confirmado por Sergio** en cada camino afectado (C-05,
C-15, C-16, C-17, C-22, C-29, C-46 y el mapa de ramas). Dos puntos no son modelos y la respuesta no
los explica del todo, así que se dejó la nota correspondiente:

- **Plan de filmación del 08/09 (C-05):** no es un modelo. Lo que consta en el repositorio: el plan solo
  existe en el commit huérfano `5ad6b64` y el dataset sigue en 274 videos.
- **`CONF_MINIMA_KEYPOINT` en 0,55 (C-15) y videos largos de la app (C-46):** son un parámetro de
  preprocesamiento y una función de la app, no modelos. Se registra la respuesta tal cual.

### Cerrados por decisión de Sergio (08/10/2026)

- **Caminos A (etiqueta débil) y B (unidades de acción EquiFACS) (C-07):** **descartados**, porque no
  forman parte del modelo final.
- **Leave-session-out (C-42):** ya no está pendiente: se ejecutó el 08/10/2026 (change
  `validacion-por-sesion`; resultados en `comparacion/resultados/validacion_por_sesion.json` y en la
  sección «Validación por sesión de filmación» de `comparacion/DECISION.md`).

### Resueltos con evidencia del repositorio

- **Comparabilidad del MIL corregido (0,722) con el vigente (0,740):** `DECISION.md` ya no los presenta
  como una caída (0,740 → 0,722); aclara que son protocolos y conjuntos distintos (C-23, C-30).
- **Atribución de la cifra 0,614 (C-25):** corresponde al **atajo de encuadre** (duración, fps, ancho,
  alto y escala de la cabeza). `lineas_de_base.md` de la rama
  `nuevo-intento-con-enmascarado-y-geometria` lo define así en su tabla (0,614 ± 0,164 en 218 × 13) y
  reserva el nombre «atajo de cámara» para otra medición (desplazamiento de lo no cefálico, 0,508 ±
  0,060). El commit de cierre `4c13c20` y el registro «caminos e impedimentos» usaron «atajo de cámara»
  para designar el mismo conjunto de cinco variables de encuadre: es un rótulo impreciso, no una cifra distinta.

### Notas menores

- **Caballos 123 y 169 (observación D3 del docente):** Sergio confirmó el 08/10/2026 que son el mismo
  caballo. El efecto medido sobre la comparación es despreciable (≤ 0,004 de F1, mismo ranking); detalle
  en `comparacion/DECISION.md`.

1. **Tres commits que no estaban en ninguna rama** (`4c13c20`, `1a3859b`, `5ad6b64`; advertencia 3 de
   la sección 1): **resuelto el 08/10/2026**, se preservaron en la rama `rescate/otro-camino-2026-09-08`.
2. **Diferencia de 93 % contra 91,3 % (C-14):** el 93 % es el aprovechamiento medido para el criterio de
   ventana al diseñarlo (tabla del §2 de `informe_mejora_pipeline_mil.md`) y el 91,3 % es el del conjunto
   final (359 de 393). El informe no explica la diferencia (los ajustes posteriores, como el anclaje
   por confianza, son una hipótesis no verificada). La cifra que corresponde citar es 91,3 %.
3. **Extractor visual, fine-tuning de más capas y cifras del informe de fine-tuning (C-18, C-19):**
   **descartados**: no forman parte del modelo final (decisión de Sergio del 08/10/2026); la contradicción 0,7737 (una partición) contra
   0,747 (cuatro folds) ya está advertida en `docs/registro_de_experimentos.md`.
