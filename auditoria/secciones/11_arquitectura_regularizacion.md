# 11. Arquitectura, dimensionalidad, regularización y *data augmentation*

## 11.1 Arquitectura vigente (instanciada y contada por la auditoría)

`clasificacion.MILFusionModel` (aprendizaje de múltiples instancias, con cada frame como instancia y el clip como bolsa):

```
frame_i:  v_vis(512, ResNet-18 congelada) ──────────────┐
          v_geom(9: 3 puntos × (x, y, válido)) → MLP 9→64→128 ┘ concat (640)
atención: Linear(640→128) · tanh · Linear(128→1)  →  softmax sobre los frames (con máscara)
bolsa:    Σ_i  a_i · [v_vis ‖ geom_embed]_i   (640)
clasificador: Dropout(0,3) · Linear(640→64) · ReLU · Dropout(0,3) · Linear(64→3)
```

| Elemento | Valor |
|---|---|
| MIL multimodal: parámetros totales | 132 355 |
| MIL multimodal: parámetros entrenables | 132 355 |
|    · geom_mlp (9 → 64 → 128) | 8 960 |
|    · atención (640 → 128 → 1) | 82 176 (62,1 %) |
|    · clasificador (640 → 64 → 3) | 41 219 |
| ResNet-18 sin capa final: parámetros | 11 176 512 (todos congelados) |
| Entrenables tras congelar la ResNet-18 | 0 |
| Pipeline completo: totales / entrenables | 11 308 867 / 132 355 |
| ¿Coincide con la guía (132 355 entrenables)? | sí |

- **Verificado:** el modelo tiene **132 355 parámetros entrenables**, la cifra de la guía del proyecto. El 62 % están en la cabeza de atención.
- **ResNet-18 congelada**: `models.resnet18(weights=IMAGENET1K_V1)` sin la capa final, con `requires_grad = False`, en modo `eval`; se aplica una única vez a cada frame y las características (512 por frame) se cachean (`caracteristicas_backbone_v2.npz`). Las 11 176 512 parámetros de la red no se entrenan.
- **Es un MLP con atención, no un modelo temporal**: no hay codificación de posición ni recurrencia; la salida es una suma ponderada por softmax de instancias independientes. Permutar los frames no cambia la salida (0/359 predicciones, diferencia máxima de logits 3,8·10⁻⁶; barajado e invertido).
- El **canal geométrico** (puntos de ojo, hocico y oreja normalizados por frame con la distancia ojo–hocico, más tres indicadores de validez) aporta dentro del MIL (el MIL visual puro rinde 0,686 y un canal de ruido de 9 dimensiones 0,680; `docs/registro_de_experimentos.md`), pero **un promedio lineal sin canal geométrico ni atención rinde lo mismo que el MIL completo** (capítulo 10).

## 11.2 Dimensionalidad frente a muestras

| Medida | Valor |
|---|---|
| Entrada por clip | 16 frames × (512 visuales + 9 geométricas) = 8 336 valores |
| Clips de entrenamiento por fold (train interno / train+val) | ≈ 199 / ≈ 269 |
| Caballos de entrenamiento por fold (train interno) | ≈ 93 |
| Parámetros entrenables por clip de train interno | 665 |
| Parámetros entrenables por caballo de train interno | 1423 |
| Valores de entrada por clip / clips de train interno | 41,9 |

La entrada por clip es **8 336 valores** (16 × 521) para entrenar ≈ 199 clips de ≈ 93 caballos por fold con 132 355 parámetros entrenables (≈ 665 por clip, ≈ 1 423 por caballo). Es un régimen de alta dimensión con pocas muestras independientes: la memorización del train (F1 0,98; capítulo 8) es esperable y la elección de congelar el backbone es la correcta (el propio código documenta que descongelarlo llevaría los entrenables al orden de 10⁷). Con **133 caballos** como tamaño muestral efectivo (capítulo 13), ni siquiera un modelo de 10⁵ parámetros puede estimarse con precisión; el único modelo con capacidad claramente proporcional a los datos es el lineal sobre una media.

## 11.3 Regularización implementada de verdad

| Técnica | ¿Está? | Detalle y evidencia |
|---|---|---|
| Backbone preentrenado y congelado | **Sí** | `caracteristicas._cargar_backbone` |
| Dropout | **Sí, 0,3** | Dos capas `Dropout(0,3)` en el clasificador de la bolsa; **no** sobre las instancias (decisión de diseño documentada). Elegido por F1 de validación interna entre 0,0 / 0,2 / 0,3 (indistinguibles: 0,687 / 0,690 / 0,707) |
| *Weight decay* | **Sí, 1e-4** | `Adam(weight_decay=1e-4)` (L2 acoplado, no AdamW) |
| *Early stopping* | **Sí** | Paciencia 10 sobre F1 de validación interna (capítulo 7) |
| Pesos de clase | **Sí** | Raíz de la frecuencia inversa (no es regularización en sentido estricto) |
| *Label smoothing*, *batch norm*, *scheduler*, recorte de gradiente | No | No hay en el código |
| Reducción de capacidad de la cabeza | No | Sugerida en `docs/analisis_sobreajuste.md`; no probada |
| Calibración posterior | No | La interfaz muestra la probabilidad softmax cruda (sobreconfiada, §11.4) |

## 11.4 Calibración de lo que se muestra al usuario

Con los modelos de fold sobre su test (359 predicciones fuera de muestra; coinciden con la evaluación vigente en 359/359):

| Rango de confianza | Clips | Confianza media | Acierto observado |
|---|---|---|---|
| (0.4,0.5] | 21 | 0,471 | 0,381 |
| (0.5,0.6] | 32 | 0,557 | 0,562 |
| (0.6,0.7] | 30 | 0,651 | 0,400 |
| (0.7,0.8] | 49 | 0,748 | 0,653 |
| (0.8,0.9] | 67 | 0,848 | 0,806 |
| (0.9,1.0] | 160 | 0,969 | 0,894 |

Confianza media 0,824, acierto 0,744, ECE 0,081, NLL 0,776 (frente a 1,099 del azar uniforme), Brier 0,373. La interfaz acompaña cada predicción con la **precisión medida de esa clase** (p. ej., «cuando el modelo predijo `Disgustado`, acertó 71 de 96 veces»), que es una corrección apropiada, pero la barra de probabilidad por clase sigue siendo optimista en el rango 0,6–0,8.

## 11.5 *Data augmentation*

- **En el modelo vigente: ninguna.** El entrenamiento consume las características del backbone **precalculadas** de los frames originales (`caracteristicas_backbone_v2.npz`), y no hay transformaciones de imagen ni de característica en `entrenar_mil_multimodal`. Por lo tanto **tampoco hay riesgo de que se aplique al test**.
- **Existe pero no se usa en el vigente:** `caracteristicas.extraer_aumentadas` genera vistas con espejo horizontal (p = 0,5), rotación ±10°, brillo y contraste ±15 %, con sorteo **por clip** (coherente entre sus 16 frames), pre-calculadas y cacheadas. Se probó en la etapa anterior (20 vistas por clip) y **no movió el F1** (−0,005 a 0,000; `openspec/project.md`).
- **Advertencia técnica:** el espejo horizontal es una transformación dudosa para expresiones asimétricas (un oído «hacia atrás» no cambia de significado, pero el canal geométrico incluye coordenadas x con signo); si se reutilizara debería aplicarse también al canal geométrico, cosa que el código actual no hace (sólo transforma píxeles).
