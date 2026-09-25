# Revisión técnica y reconstrucción del pipeline MIL

**Fecha:** 18 de septiembre de 2026
**Rama:** `feat2/otro-camino`
**Arquitectura:** Multimodal MIL (`MILFusionModel`), sin cambios estructurales

---

## Resumen

El pipeline aprovechaba el **38,9 %** de sus clips y la interfaz mostraba métricas que no
correspondían al modelo que estaba ejecutando. Después de esta revisión aprovecha el **91,3 %**,
entrenamiento e inferencia comparten una única implementación de la preparación, y todo número que
ve el usuario se lee del archivo de la evaluación vigente.

| | Antes | Después |
|---|---:|---:|
| Clips aprovechados | 153 / 393 (38,9 %) | 359 / 393 (91,3 %) |
| Caballos | 83 | 133 |
| Clips de `Disgustado` | 33 | 103 |
| Épocas | 50 fijas, sin validación | early stopping, mejor época 28,5 ± 8,4 |
| Elección de época / dropout | no existía | validación interna por ejemplar |
| F1 macro (protocolo riguroso) | 0,618 | **0,740 ± 0,052** |

---

## 1. Estado inicial

### El pipeline original

```
video → clips de 5-7 s → np.linspace(0, N-1, 16) → DeepLabCut → ¿los 16 tienen keypoints?
                                                                      ↓ no
                                                                 clip descartado
```

Las 16 posiciones se fijaban **antes** de saber qué fotogramas tenían keypoints. DeepLabCut corría
sobre el clip entero pero sólo se conservaban las filas de esas 16 posiciones; el resto de la
inferencia —el 90 % de los fotogramas— se descartaba sin escribirse. Un parpadeo, un giro de
cabeza o un desenfoque que cayera en una de las 16 posiciones costaba el fotograma, y cinco de
esos costaban el clip completo, habiendo en un clip de 5 s a 30 fps otros 134 fotogramas ya
procesados que nadie miraba.

Medido: de 393 clips, sólo **82 tenían los 16 fotogramas válidos** y 220 tenían al menos 12.

### Problemas detectados en la auditoría

**1. Entrenamiento e inferencia eran dos implementaciones distintas.** La etapa 02 interpolaba
huecos de 1 fotograma, inyectaba anotaciones manuales y, cuando no había par ojo-hocico, recortaba
por caja de YOLO. `equinos/inferencia.py` no hacía nada de eso: descartaba el fotograma. El efecto
era **una interfaz más estricta que el pipeline que fabricó su conjunto de entrenamiento**,
rechazando videos que el dataset habría aceptado mientras informaba una confiabilidad medida sobre
otra distribución. Nada fallaba.

**2. El dataset contenía muestras contaminadas.** `particion.tabla_clips` aceptaba clips con
`12 <= alineados <= 16`, y `caracteristicas.extraer` lee todas las filas del clip sin filtrar por
`alineado`. Hasta 4 de los 16 fotogramas que entraban al backbone podían ser recortes de caja sin
alinear: dos geometrías dentro de la misma secuencia, sin ninguna señal que lo distinguiera. El
docstring del módulo y `tests/test_particion.py` declaraban el criterio estricto de 16; el código
se había desviado y el test fallaba.

**3. Sin validación interna ni selección de época.** 50 épocas fijas, sin evaluar nada durante el
entrenamiento. No es que se eligiera mal la época: no se elegía, y nadie podía saber si el modelo
sobreajustaba o si le faltaba entrenar.

**4. Las métricas mostradas no eran del modelo desplegado.** `objetivo3_evaluacion.json` lo escribe
un script archivado (`openspec/changes/archive/2026-08-28-.../evaluacion_final.py`) que corre el
modelo **de promedio** sobre 141 clips. De ahí salían las precisiones por clase que veía el
usuario. Al lado, el texto citaba `F1 macro 0,754 contra 0,691 exigido`, escrito a mano en
`clasificacion.py` y en `app_inferencia.py`, que es el número de **otro** modelo medido en **otra**
máquina sobre **otro** conjunto (257 clips, entorno Windows). Ninguno de los dos describía al MIL
que estaba infiriendo.

**5. Artefactos desactualizados.** `particion_clips.json` tenía 153 clips mientras
`tabla_clips()` sobre el mismo CSV devolvía 254: la partición persistida correspondía a una regla
anterior. 101 clips que el código aceptaba nunca habían entrado a ninguna corrida.

**6. El notebook 03 se contradice.** Su texto narra 141 clips, F1 0,626 y "la hipótesis no queda
sostenida"; sus outputs guardados dicen 257 clips, F1 0,754 y "CUMPLE". La celda rotulada
"Un frame + voto, sin información temporal" invoca en realidad `entrenar_mil_multimodal`.

**7. Un video etiquetado de DeepLabCut dentro de `datos_equinos/clips/Disgustado/`**, que un glob
por `*.mp4` levanta como si fuera un clip del dataset: mismo caballo, misma clase, pero con los
keypoints dibujados sobre la imagen que ve el backbone.

---

## 2. Cambios implementados

### 2.1 Selección de fotogramas (`equinos/secuencia.py`)

Se reemplazó el muestreo por posiciones fijas por una búsqueda de **ventana temporal**:

```
pose de TODOS los fotogramas
        ↓
fotogramas con par ojo-hocico válido (ambos ≥ CONF_MINIMA_KEYPOINT)
        ↓
ventana contigua MÁS LARGA (≤ 7 s) con ≥16 válidos y sin tramos > 0,5 s sin medición
        ↓
16 posiciones repartidas parejo, cada una anclada a un fotograma medido
```

**Por qué la ventana y no la racha de válidos consecutivos.** La primera implementación siguió la
especificación literal —la racha contigua más larga de fotogramas válidos, con huecos de 1—. Medido
sobre el material, aceptaba el 87 % de los clips pero con una ventana mediana de **2,33 s** contra
los 6,00 s que abarcaba el muestreo anterior: 16 fotogramas casi simultáneos. Como la hipótesis del
trabajo es que la evolución temporal permite clasificar, una secuencia de medio segundo no es una
muestra más corta sino una muestra de otra cosa.

| Criterio | Aceptados | Ventana mediana | < 2 s |
|---|---:|---:|---:|
| Racha contigua de válidos | 87 % | 2,33 s | 44 % |
| **Ventana contigua con ≥16 válidos** | **93 %** | **5,74 s** | **11 %** |
| *(referencia: muestreo anterior)* | | *6,00 s* | |

La contigüidad se define sobre la **ventana** y no sobre los válidos, lo que conserva la propiedad
que importaba —una sola ventana, sin fragmentos temporalmente desconectados, porque ningún tramo
sin medición puede pasar de 0,5 s— y recupera la duración del entrenamiento.

El tope de tramo va **en segundos y no en fotogramas** porque el material mezcla 30 y 60 fps: "diez
fotogramas" significa dos cosas distintas según el clip, medio segundo de cara no medida significa
lo mismo en los dos.

**Anclaje por confianza.** El umbral de 0,50 es un piso, no una garantía: DeepLabCut devuelve el
par por encima de 0,50 en fotogramas con desenfoque de movimiento donde la cara no se distingue.
Verificado visualmente sobre recortes reales y cuantificado: el 6,6 % de los fotogramas elegidos
quedaba pegado al umbral. Cada posición se ancla ahora al **mejor medido** dentro de una tolerancia
de 0,10 s, recortada además a un tercio del paso entre posiciones para que dos posiciones contiguas
no puedan cruzarse.

| | p05 | mediana | < 0,55 | < 0,60 |
|---|---:|---:|---:|---:|
| Sin preferencia | 0,53 | 0,76 | 6,6 % | 14,8 % |
| Con preferencia | 0,56 | 0,80 | **3,4 %** | **9,0 %** |

Desplazamiento temporal: media 25 ms, máximo 167 ms sobre ventanas de ~5,8 s. La ventana no se
mueve; se corrige cuál fotograma, nunca dónde.

### 2.2 Interpolación

`MAX_HUECO_INTERPOLADO = 1` queda **sin uso**, y conviene ser explícito: ese parámetro autorizaba a
rellenar una coordenada faltante, y con la selección nueva no se rellena ninguna. Los 16 fotogramas
salen de mediciones reales, y anclar al medido más cercano da mejor dato que inventar una
coordenada teniendo una medida al lado. Su lugar lo toma `secuencia.HUECO_MAXIMO_SEG`, que no
autoriza nada sino que decide hasta dónde se considera que la ventana sigue siendo el mismo
momento. Son criterios distintos —uno se validó contra el error de una coordenada estimada, el otro
contra la duración de un gesto— y por eso no comparten número.

La lógica de interpolación sigue centralizada en `equinos/interpolacion.py`, disponible para quien
la necesite, con su tope derivado de una curva medida contra línea de base.

### 2.3 Unificación entrenamiento/inferencia (`equinos/alineado.py`)

`preparar_secuencia()` es el único lugar donde se decide qué fotogramas entran y cómo se alinean.
La llaman `equinos/dataset.py` (generación del conjunto) y `equinos/inferencia.py` (interfaz), con
la misma pose y los mismos parámetros de la etapa 02. Por construcción no pueden divergir.

Se eliminaron de `inferencia.py` las funciones `_detectar` y `_keypoints`, que quedaron muertas, y
con ellas **el gateo de la pose detrás de YOLO**: si el detector fallaba en los 16 fotogramas
muestreados, el video se rechazaba sin darle a DeepLabCut —que trae su propio detector— la
oportunidad de encontrar la cara. La etapa 02 nunca hizo ese gateo.

`tests/test_equivalencia.py` verifica sobre clips reales que las dos ramas eligen los mismos
fotogramas, producen las mismas transformadas y generan **recortes idénticos píxel a píxel**.

### 2.4 Criterio de cobertura

Se restauró el criterio declarado: **exactamente 16 fotogramas alineados por keypoints**. Con la
selección por ventana desapareció la disyuntiva entre recuperar clips y no contaminar la muestra;
el rescate viene de mirar mejor el video, no de admitir muestras mezcladas. `tests/test_particion.py`
vuelve a pasar, y se le agregó el caso de 12 de 16.

Los 359 clips son **todos de detección propia**: cero interpolados, cero anotados a mano. El
conjunto de la ablación y el conjunto completo pasaron a ser el mismo.

### 2.5 Validación interna y early stopping

```
GroupKFold externo por caballo (4 folds)
   ├── test          ← se evalúa UNA vez, con el mejor checkpoint ya cargado
   └── train_val
        ├── train interno      ← ~199 clips / ~93 caballos
        └── validación interna ← ~70 clips / 7 caballos
```

`particion.dividir_por_ejemplar()` hace el corte interno reusando el criterio determinista de
`construir()`, con `assert` de que ningún caballo cruce. `entrenar_mil_multimodal()` verifica
además que ni el train interno ni la validación compartan caballo con el test.

Early stopping sobre F1 macro de validación interna, paciencia 10, máximo 100 épocas. El mejor
checkpoint se guarda en memoria y se recarga antes de tocar el test.

**El test no interviene en ninguna decisión**: ni época, ni dropout, ni pesos de clase.

### 2.6 Dropout

Se agregó `MILFusionModel(dropout=...)` sobre la representación de la bolsa, no sobre las
instancias: apagar fotogramas sueltos cambiaría lo que la cabeza de atención puede aprender a
ponderar, que es el objeto del modelo. Con 0.0 la red es exactamente la anterior.

| dropout | F1 validación interna | por fold |
|---|---:|---|
| 0,0 | 0,687 ± 0,044 | 0,685 0,660 0,653 0,749 |
| 0,2 | 0,690 ± 0,043 | 0,700 0,737 0,634 0,688 |
| **0,3** | **0,707 ± 0,061** | 0,710 0,767 0,623 0,728 |

Elegido 0,3 por ser el máximo en validación interna, que es la regla declarada. **Los tres son
estadísticamente indistinguibles** con este tamaño de muestra: la diferencia (0,02) está muy dentro
del desvío entre folds (0,04–0,06). No debe presentarse como "0,3 es mejor".

### 2.7 Un bug introducido durante esta revisión, y su guarda

`MultimodalEquineDataset` busca la geometría con `pose_por_clip[clip][orden]` recorriendo
`range(16)`: interpreta `orden` como **posición en la secuencia**. El archivo de la pasada de GPU
usa `orden` = índice de fotograma del video. Pasárselo directo hacía que el canal geométrico leyera
los fotogramas 0–15 mientras el visual usaba los que la ventana había elegido:

```
secuencia visual usa los fotogramas : 178, 192, 203, ..., 340
canal geométrico leyó los fotogramas:   0,   1,   2, ...,  15
```

No fallaba: el lookup encontraba filas válidas, sólo que de otros fotogramas. Se corrigió emitiendo
`pose_seleccionada_v2.csv` con `orden` renumerado, y se agregó una guarda en
`MultimodalEquineDataset` que levanta `ValueError` si recibe una pose cuyo `orden` máximo supera
las 16 posiciones, nombrando cuál archivo corresponde.

### 2.8 Métricas de la aplicación

`clasificacion.metricas_globales()` lee las métricas del mismo archivo del que salen las
precisiones por clase. Se eliminó el `F1 macro 0,754` escrito a mano de `texto_de_confiabilidad()`
y de `app_inferencia.py`. Los pesos de producción pasaron de `un_frame_produccion.pt` —nombre
heredado del modelo de un fotograma con voto— a `mil_multimodal_produccion.pt`.

---

## 3. Dataset resultante

```
clips candidatos : 393
clips aceptados  : 359
clips rechazados :  34
aprovechamiento  : 91,3 %

motivos de rechazo:
  VENTANA_CORTA   31   no hay ventana con 16 mediciones sin tramos > 0,5 s
  SIN_KEYPOINTS    3   DeepLabCut no midió el par en ningún fotograma
```

| | Antes | Después |
|---|---:|---:|
| Clips | 153 | 359 |
| Caballos | 83 | 133 |
| `Disgustado` | 33 clips / 25 caballos | 103 clips / 57 caballos |
| `Interesado` | 56 clips / 35 caballos | 118 clips / 60 caballos |
| `Sereno` | 64 clips / 36 caballos | 138 clips / 52 caballos |
| Desbalance may:min | 1,9 : 1 | 1,3 : 1 |
| Ventana temporal | *(el clip entero)* | mediana 5,77 s (p25 3,27 — p75 6,07) |

29 caballos tienen clips íntegros en más de una clase, por lo que el grupo de la partición es el
ejemplar y no el par ejemplar-clase.

---

## 4. Entrenamiento

Configuración final: `lr=1e-3`, `weight_decay=1e-4`, `batch_size=32`, `geom_embed_dim=128`,
`dropout=0,3`, `max_epocas=100`, `paciencia=10`, pesos de clase por **raíz de la frecuencia
inversa**, `SEMILLA=42`. Los dos únicos parámetros elegidos —dropout y estrategia de pesos— se
decidieron con validación interna; ningún otro fue barrido.

| fold | train | validación | test | mejor época | F1 val | F1 test |
|---:|---|---|---|---:|---:|---:|
| 0 | 199 c / 93 cab | 70 c / 7 cab | 90 c / 33 cab | 28 | 0,700 | 0,718 |
| 1 | 199 c / 93 cab | 70 c / 7 cab | 90 c / 33 cab | 26 | 0,766 | 0,769 |
| 2 | 199 c / 92 cab | 70 c / 7 cab | 90 c / 34 cab | 20 | 0,641 | 0,677 |
| 3 | 197 c / 93 cab | 73 c / 7 cab | 89 c / 33 cab | 40 | 0,742 | 0,795 |

**Mejor época: media 28,5 ± 8,4.** Ningún fold pasó de la 40.

---

## 5. Resultados globales

```
F1 macro           0,740 ± 0,052
Balanced accuracy  0,740
Accuracy           0,744
Precision macro    0,744
Recall macro       0,742
```

### Métricas por clase

| clase | precision | recall | F1 | support |
|---|---:|---:|---:|---:|
| `Disgustado` | 0,740 | 0,689 | 0,714 | 103 |
| `Interesado` | 0,764 | 0,797 | 0,780 | 118 |
| `Sereno` | 0,729 | 0,739 | 0,734 | 138 |

La precisión de `Disgustado` —el número que la interfaz le muestra al usuario— pasó de **55 % a
74 %**, y su recall de 67 % a 69 %.

### Matriz de confusión (fuera de muestra, agregada sobre los folds)

```
real / predicha   Disgustado  Interesado   Sereno   total
Disgustado                71          13       19     103
Interesado                 5          94       19     118
Sereno                    20          16      102     138
total                     96         123      140     359
```

### `Sereno` vs `Disgustado`

| | Antes | Después |
|---|---:|---:|
| `Sereno` clasificado como `Disgustado` | 15 / 64 = 23,4 % | 20 / 138 = **14,5 %** |

Bajó de 23,4 % a 14,5 %, pero **sigue siendo el error dominante**: 20 de los 25 falsos
`Disgustado` son `Sereno`. Más datos, mejor selección y pesos atenuados lo redujeron sin
resolverlo, lo que sugiere que la confusión no es un artefacto del muestreo sino una limitación de
la señal disponible.

### Experimento de pesos de clase

Primero se verificó si seguía habiendo sobrepredicción, que era la condición para intervenir:

| clase | veces que lo dijo | veces que aparece | ratio |
|---|---:|---:|---:|
| `Disgustado` | 113 | 103 | 1,10x *(era 1,21x)* |
| `Interesado` | 109 | 118 | 0,92x |
| `Sereno` | 137 | 138 | 0,99x |

La sobrepredicción ya era leve, pero como `Sereno → Disgustado` seguía siendo el error dominante
se corrió igual la comparación, **decidida con validación interna**:

| estrategia | F1 validación interna | por fold |
|---|---:|---|
| Frecuencia inversa | 0,7069 ± 0,0611 | 0,710 0,767 0,623 0,728 |
| **Raíz de la inversa** | **0,7124 ± 0,0550** | 0,700 0,766 0,641 0,742 |

Recién después de decidir se miró el test, sólo para reportar:

| estrategia | F1 test | precision `Disgustado` | `Sereno → Disgustado` |
|---|---:|---:|---:|
| Frecuencia inversa | 0,734 | 0,699 (79/113) | 26 |
| Raíz de la inversa | 0,740 | 0,740 (71/96) | 20 |

Mejora en los tres números y la dirección tiene sentido: atenuar el peso de `Disgustado` reduce
cuántas veces lo dice, y los que deja de decir son mayoritariamente `Sereno` mal clasificados.
**Igual que con dropout, la diferencia en validación interna (0,005) está muy dentro del desvío
entre folds (±0,055).** Se adopta porque es la regla declarada, no porque esté demostrada; que el
test la confirme es tranquilizador pero no es evidencia, porque el test no participó de la
decisión.

---

## 6. Comparación con versiones anteriores

| Modelo | Clips | Caballos | Evaluación | F1 macro |
|---|---:|---:|---|---:|
| Promedio (histórico) | 153 | 83 | GroupKFold por caballo, 50 épocas fijas | 0,704 |
| MIL (histórico, otra máquina) | 257 | 113 | GroupKFold por caballo, 50 épocas fijas | 0,754 |
| MIL sobre datos v1 | 153 | 83 | GroupKFold + validación interna + early stopping | 0,618 |
| **MIL v2** | **359** | **133** | **GroupKFold + validación interna + early stopping** | **0,740** |

**Las dos primeras filas no son comparables con las dos últimas.** Eligieron cuántas épocas
entrenar sin ninguna validación, así que su número no es una estimación limpia fuera de muestra: el
modelo que reportan es el de la época 50, elegida a priori, y no hay forma de saber si ese punto
estaba sobreajustado. Además la segunda corrió en otro entorno sobre otro conjunto.

**Las dos últimas sí son comparables entre sí**: mismo protocolo, misma implementación, misma
semilla. La diferencia 0,618 → 0,740 aísla el efecto de los 206 clips recuperados.

La caída aparente de 0,704 a 0,618 sobre el mismo material **no es una degradación del modelo**: es
el costo metodológico de apartar ~70 clips para validación interna en un conjunto de 153. El modelo
pasa de entrenar con 114 clips a hacerlo con 84.

---

## 7. Ablación del canal geométrico: dos experimentos que parecen contradecirse

### Primer experimento: la ablación accidental

El bug descrito en 2.7 funcionó como una ablación no planificada. Con el canal geométrico leyendo
fotogramas equivocados el F1 fue **0,730**; corregido y con la misma configuración, **0,734**.
Cuatro milésimas, muy dentro del ±0,07 entre folds.

La lectura inmediata fue que el componente multimodal no aportaba señal medible, y sobre esa
lectura se recomendó retirarlo.

### Segundo experimento: retirarlo de verdad

Se implementó el MIL visual puro —sin `geom_mlp`, con la atención y el clasificador operando
directamente sobre las 512 dimensiones del backbone— y se volvieron a elegir dropout y estrategia
de pesos con validación interna, porque la red había cambiado y la elección anterior no aplicaba.

| Modelo | F1 macro | Accuracy | Precision `Disgustado` | `Sereno` → `Disgustado` |
|---|---:|---:|---:|---:|
| **Multimodal** | **0,740** | **0,744** | **0,740** | **20 (14,5 %)** |
| Visual puro | 0,686 | 0,702 | 0,612 | 32 (23,2 %) |

En validación interna, que es la métrica con la que se decide: 0,686 contra 0,712 del multimodal.

### Por qué no se contradicen

**Los dos experimentos miden preguntas distintas.**

El primero mide *¿importa que la geometría sea la correcta?* Con el bug, `geom_mlp` **seguía
existiendo** y aportando sus 128 dimensiones a la fusión sobre la que opera la atención; lo que
cambiaba era de qué momento del clip venían esas coordenadas. Respuesta: no demasiado.

El segundo mide *¿importa el canal?* Respuesta: **sí**. Retirarlo cuesta 0,054 de F1, derrumba la
precisión de `Disgustado` de 74 % a 61 % y devuelve la confusión `Sereno → Disgustado` al 23,2 %,
exactamente donde estaba antes de esta revisión.

Confundir las dos preguntas llevó a recomendar el retiro del canal. **La medición lo refutó y el
canal se conservó.** Queda documentado como ejemplo de por qué una ablación tiene que *quitar* la
parte y no corromperle la entrada: una entrada corrupta sigue ocupando capacidad y sigue
participando del aprendizaje.

### Qué sí se puede afirmar

Que el canal geométrico aporta a la discriminación `Sereno` / `Disgustado`, que es justamente
donde el modelo más falla. Lo que **no** se puede afirmar con estos experimentos es *cuánto* de
esa contribución es información geométrica y cuánto es simplemente capacidad adicional: la
diferencia entre las dos hipótesis requeriría comparar contra un canal de ruido de la misma
dimensión, que no se corrió.

> **Actualización del 25/09/2026.** El control de ruido se corrió después, en la rama
> `feat/enriquecer-canal-geometrico` (`docs/registro_de_experimentos.md`): con 9 dimensiones de ruido
> el F1 es 0,680 ± 0,102, así que el canal real le gana por +0,059. Pero la línea de base visual
> estática —el promedio de las 512 características de los 16 frames con una regresión logística,
> **sin** canal geométrico ni atención— da 0,741 ± 0,020 (`objetivo3_lineas_de_base.json`). El canal
> ayuda **dentro** del MIL; no está demostrado que el MIL con canal supere a un promedio simple.

---

## 8. Reproducibilidad

Semillas fijadas en `random`, `numpy` y `torch` (`SEMILLA = 42`, `torch.cuda.manual_seed_all`,
generador propio del `DataLoader`). `GroupKFold` no lleva `random_state` porque la partición es
determinista por construcción.

Entorno registrado en `objetivo3_evaluacion_v2.json`: Python 3.10.12, torch 2.5.1+cu121,
numpy 1.26.4. GPU: NVIDIA RTX 3070 Laptop (8 GB).

### Reproducir desde cero

```bash
cd "<raíz del proyecto>"
export PYTHONPATH="$PWD"
V=~/.venvs/tesina-equinos/bin/python

# 1. Pose de TODOS los fotogramas de cada clip  (~3 h en GPU; reanudable)
$V -m equinos.reproceso

# 2. Selección de ventana, alineado y recortes  (~10 min)
$V -m equinos.dataset

# 3. Partición por ejemplar, con verificación de fuga
$V -c "from equinos import rutas, particion; rutas.inicializar(verbose=False); particion.informe()"

# 4. Características con el backbone congelado  (~2 min)
$V -c "from equinos import rutas, particion, caracteristicas as c; rutas.inicializar(verbose=False); c.extraer(particion.cargar())"

# 5. Evaluación + modelo de producción  (~5 min)
$V scripts/evaluar_v2.py

# 6. Pruebas
for t in tests/test_*.py; do $V "$t"; done

# 7. Interfaz
./correr_local.sh interfaz
```

El paso 1 es el único costoso y **sólo hace falta una vez**: persiste la pose de todos los
fotogramas, así que cambiar el criterio de selección cuesta minutos y no horas.

---

## 9. Conclusiones

**1. ¿Cuántos clips adicionales se recuperaron?** 206 (153 → 359), y 50 caballos nuevos (83 → 133).
`Disgustado`, la clase que limitaba el trabajo, se triplicó.

**2. ¿Mejoró la nueva selección el aprovechamiento?** De 38,9 % a 91,3 %. Ningún clip se pierde ya
porque las 16 posiciones fijas hayan caído mal. Los 34 restantes son material que genuinamente no
tiene una cara medible sostenida.

**3. ¿Qué época elige el early stopping?** 28,5 ± 8,4, ninguna por encima de 40.

**4. ¿Había sobreajuste con 50 épocas fijas?** Sí. El mejor checkpoint aparece consistentemente por
debajo de 50, tanto con 153 clips (media 15,0) como con 359 (media 28,5 con dropout 0,3 y pesos
atenuados; 14,5 sin dropout). Es estructural, no ruido.

**5. ¿Mejoró el F1 macro?** Bajo protocolo comparable, 0,618 → 0,740. Contra las cifras históricas
la comparación no es válida y no debe presentarse como tal.

**6. ¿Se redujo la confusión `Sereno` → `Disgustado`?** De 23,4 % a 14,5 %. Sigue siendo el error
dominante: 20 de los 25 falsos `Disgustado`.

**7. ¿Cuál es el cuello de botella ahora?** La confusión `Sereno` → `Disgustado`, que concentra 20
de los 25 falsos `Disgustado`. Y por detrás, el tamaño de la muestra: 133 caballos. El canal
geométrico sí aporta (sección 7), así que la vía más prometedora es enriquecerlo antes que tocar
el backbone.

**8. ¿Qué haría después?**

- **Enriquecer el canal geométrico antes que nada.** La ablación de la sección 7 mostró que aporta,
  y hoy usa tres keypoints —ojo, hocico, oreja— normalizados contra la escala ojo-hocico, de los 39
  que expone `superanimal_quadruped`. Es la palanca con mejor relación entre evidencia y costo.
- **Descongelar el backbone,** o hacer fine-tuning de sus últimas capas. Con 359 clips es
  defendible; con 141 no lo era.
- **Validación cruzada anidada en vez de corte interno simple.** Con 7 caballos en validación
  interna, el fold 2 dio F1 val 0,623 contra 0,767 del fold 1. Esa varianza hace que la elección de
  época sea ruidosa.
- **Subir `CONF_MINIMA_KEYPOINT`** y medir el costo en clips: queda un 3,4 % de fotogramas pegados
  al umbral que el anclaje por confianza no pudo evitar.

---

## Anexo: pendientes que no forman parte de este trabajo

- **`tests/test_entorno.py` falla**, y falla igual sin estos cambios. Verifica que importar
  `equinos.entorno` no cargue numpy ni torch. Es preexistente y ajeno a esta revisión.
- **El notebook 03 sigue siendo internamente inconsistente** (sección 1, problema 6). Su texto y
  sus outputs describen experimentos distintos. No se tocó.
- **`datos_equinos/clips/Disgustado/022-copia-de-chocolate-6_..._labeled_before_adapt.mp4`** es una
  salida de DeepLabCut dentro de la carpeta de clips. Se la filtra por nombre
  (`reproceso.es_artefacto`), pero conviene eliminarla.
- **Los artefactos v1 se conservan** (`dataset_frames_alineados.csv`, `pose_completa.csv`,
  `particion_clips_v1.json`, `objetivo3_evaluacion.json`) para poder reproducir los números
  históricos. Ninguno lo lee el código vigente.
