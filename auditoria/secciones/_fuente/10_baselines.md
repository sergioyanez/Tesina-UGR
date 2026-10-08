# 10. Baselines y valor agregado del modelo

## 10.1 Baselines que existen en el repositorio

| Baseline | Dónde | Estado |
|---|---|---|
| Clase mayoritaria, azar estratificado | `equinos/lineas_de_base.py`, `scripts/evaluar_lineas_de_base.py` | Vigente; mismos folds que el modelo |
| Solo `id_caballo` (identidad) | `clasificacion.predecir_por_identidad` | Vigente (mide el efecto del confusor de individuo; §6.1) |
| Duración, metadatos de filmación, geometría sola, un frame, media de 16 frames | `lineas_de_base.py` | Vigente, `objetivo3_lineas_de_base.json` |
| Un frame con voto (0,626 en 141 clips), cabeza promedio (0,704 en 153), GRU, atención, ≥ 18 configuraciones | `openspec/changes/archive/2026-08-28-clasificacion-orientada-a-video/` | Histórico (otro conjunto de clips) |
| Variantes del MIL (canal geométrico, CV anidada, *fine-tuning*) | ramas `feat/enriquecer-canal-geometrico`, `feat/validacion-cruzada-anidada-NestedCV`, `feat/finetuning-visual` | No adoptadas; documentadas en `docs/registro_de_experimentos.md` |

## 10.2 El modelo frente a cada línea de base (mismos 4 folds)

{{TABLA:baselines}}

{{FIG:fig_baselines.png|Figura 10.1. Matriz de confusión del fold 3 (izquierda) y F1 macro de las líneas de base y del modelo vigente con su desvío entre folds (derecha).}}

**IC 95 % de las diferencias (bootstrap por caballo, 2 000 réplicas, diferencias pareadas por réplica).** Recalculado de forma independiente por esta auditoría; coincide con `objetivo3_lineas_de_base.json` hasta la última cifra:

{{TABLA:bootstrap}}

{{FIG:fig_estabilidad_y_bootstrap.png|Figura 10.2. F1 macro por fold del MIL y de la media de 16 frames (izquierda) y diferencias MIL − baseline con su IC 95 % (derecha).}}

**Lectura.**

1. El modelo supera con holgura al azar, a la mayoritaria, a los metadatos de filmación (+0,121; IC [+0,031; +0,211]), a la geometría sola (+0,128; IC [+0,053; +0,199]) y a **un frame central** (+0,059; IC [+0,001; +0,116], el límite inferior roza el cero).
2. **No supera a la regresión logística sobre la media de los 16 frames**: diferencia −0,005, IC 95 % [−0,068; +0,058]. En los 4 folds el signo alterna (−0,022; +0,029; −0,039; +0,029). Con la partición vigente, **la arquitectura (atención, canal geométrico, MIL) no demuestra valor agregado sobre un promedio lineal**. El baseline, además, es mucho más estable entre folds (DE 0,020 contra 0,052).
3. **Evidencia contraria (sensibilidad a la partición).** En 4 asignaciones alternativas de caballos a folds (mismo código de entrenamiento, `scripts/a15_particiones_alternativas.py`) el MIL supera a la media estática por +0,003 a +0,021 (media +0,015) y 4 de 4 veces es mayor:

{{TABLA:particiones_alt}}

   La ventaja media (~+0,015) es un orden de magnitud menor que el IC de una sola partición (±0,06) y proviene de los mismos 133 caballos, de modo que **no hay evidencia suficiente para afirmar ni para descartar una ventaja pequeña del MIL**; sí la hay para afirmar que no es grande.

## 10.3 Confusores de filmación: lo que se puede predecir sin mirar la cara

{{TABLA:confusores}}

{{TABLA:rotacion}}

- **Con metadatos (duración, fps, resolución, ventana) el F1 es 0,618**; con otros metadatos técnicos (códec, contenedor, tasa de bits, rotación) **0,556**, y con la **fecha de sesión** (del nombre de archivo) **0,535**; combinando técnicos y fecha, **0,605**. Ninguno alcanza 0,740, pero **todos están muy por encima del azar (0,327)**: el contexto de captura contiene una parte sustancial de la señal que el modelo podría usar si la lee del fondo, del encuadre o de la propia ventana temporal.
- **La rotación declarada está asociada a la clase**: 1 de 110 videos de `Disgustado` frente a 21 de 71 de `Interesado` y 47 de 93 de `Sereno`; el 35,4 % de los clips válidos proviene de videos rotados. No es un predictor fuerte por sí sola (F1 0,384), pero es un indicio de que el dispositivo o la forma de grabar difiere por clase.
- **Control del propio proyecto** (`objetivo3_lineas_de_base.json`, predicciones ya existentes, sin re-entrenar): en el subconjunto 1080×1920 y ≥ 5 s (236 clips: 28 `Disgustado`, 80 `Interesado`, 128 `Sereno`) el MIL obtiene **F1 0,705** (accuracy 0,746) y la media estática 0,671. Es la mejor estimación del rendimiento donde la filmación no delata la clase, aunque con muy pocos `Disgustado`.
- **Lectura prudente.** La caída de 0,740 a 0,705 (−0,035) es menor que el desvío entre folds; sugiere que el modelo **no depende principalmente** de la duración y la resolución, pero no puede descartar que dependa de otros rasgos de contexto (fondo, sesión, iluminación) que las ventanas del recorte facial conservan. Es una limitación que debe declararse, no una refutación.
