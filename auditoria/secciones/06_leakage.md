# 6. Fuga de información (*data leakage*)

## 6.1 Qué variable agrupa y por qué

- **Variable de grupo: `id_caballo`**, el prefijo numérico del nombre de archivo (`equinos.catalogo.obtener_id_caballo`), usada en `particion.construir` (folds externos), `particion.dividir_por_ejemplar` (corte interno) y `clasificacion.folds_por_ejemplar`.
- **Por qué evita memorizar individuos.** Los clips de un caballo son visualmente casi idénticos entre sí (misma cabeza, mismo fondo, misma luz), y **el caballo predice la etiqueta**: 104 de 133 caballos (78,2 %) aparecen en una sola clase. Un modelo que reconoce al individuo «acierta» sin mirar la expresión. Medido por esta auditoría con el conjunto vigente (`scripts/a19_identidad.py`): un predictor que usa **únicamente el id del caballo** obtiene **F1 macro 0,613** con una partición ingenua por clip (≈ 80 % de los clips de test pertenecen a caballos vistos en entrenamiento) y **0,183** con la partición por caballo (= mayoritaria): la fuga que el protocolo elimina vale ≈ +0,43 de F1 macro espurio.
- **Caballos en varias clases:** 29 (22 en dos clases y 7 en las tres; 145 clips). Se agrupa por caballo y **no** por el par caballo-clase: con el par, un mismo animal quedaría de ambos lados del corte.

## 6.2 Verificaciones ejecutadas (con aserciones)

`scripts/a03_balance_folds_leakage.py` falla si alguna de estas condiciones no se cumple:

| Comprobación | Resultado |
|---|---|
| Caballos en más de un fold | **0** de 133 |
| Videos de origen en más de un fold | **0** de 244 |
| Clips en más de un fold | **0** de 359 |
| Intersección caballo / video / clip entre `train_val` y test, en cada uno de los 4 folds | **0** (12 aserciones) |
| Intersección de caballos entre train interno, validación interna y test, en cada fold | **0** |
| `train_interno + validación + test` = 359 clips, sin repetidos | Sí |
| Las 359 predicciones vigentes corresponden exactamente a los folds persistidos | Sí (`particion_clips.json` se reescribió a las 15:50, 7 min después de la evaluación de las 15:43; el contenido es consistente) |

El código tiene, además, aserciones propias que abortan la ejecución: `particion.verificar`, `dividir_por_ejemplar` y `entrenar_mil_multimodal` (`assert not cruce` para train interno–test y validación–test).

## 6.3 Estadísticas de normalización, selección de épocas e hiperparámetros

| Punto | Estado | Evidencia |
|---|---|---|
| ¿Se ajustan estadísticas de normalización con el test? | **No** | El canal visual no se estandariza (características crudas del backbone); el canal geométrico se normaliza **dentro de cada frame** con la distancia ojo–hocico del mismo frame (sin estadísticas globales); la ResNet-18 usa las constantes fijas de ImageNet. En los baselines lineales el `StandardScaler` va dentro de un `Pipeline` y se ajusta con `fit(X[entrena])` de cada fold |
| ¿Se calculan los pesos de clase con datos de test? | No | `pesos_de_clase` sobre el train interno |
| ¿La época se elige con el test? | **No** | Prueba empírica: se re-entrenaron los 4 folds **con las etiquetas del test permutadas al azar**; la mejor época, las curvas de validación y **los pesos finales resultaron idénticos** a los originales en los 4 folds (`datos/early_stopping_sobreajuste.json`). El test sólo se predice después de cargar el mejor estado |
| ¿Los hiperparámetros se eligen con el test? | **No directamente** | Dropout 0,3 y pesos `raiz` se eligieron con el F1 de validación interna (`docs/informe_mejora_pipeline_mil.md` §2.6) |
| Duplicación de frames | No | §3.5: 0 recortes idénticos entre clips; 0 frames repetidos dentro de un clip |
| Archivos derivados | Sin etiquetas | `pose_seleccionada_v2.csv` y `caracteristicas_backbone_v2.npz` se derivan de los frames; la huella de las características coincide con la registrada en el modelo de producción (`806a6f478408ab0b`) |

## 6.4 Riesgos residuales (no detectados como fuga, pero a declarar)

1. **Selección de hiperparámetros y de arquitectura no anidada.** Las variantes (≥ 12 evaluaciones sobre los mismos 359 clips y los mismos folds; `docs/registro_de_experimentos.md`) se compararon contra los mismos tests. La validación interna de un fold está formada por caballos que son *test* en otros folds, de modo que la elección de dropout/pesos con la media de las validaciones internas incorpora, de forma indirecta, información de los tests de los otros folds. Con tres valores de dropout indistinguibles (0,687 / 0,690 / 0,707 de F1 de validación) el efecto es despreciable; la elección de *arquitectura* (MIL en lugar de un frame con voto o GRU) sí se hizo mirando el F1 de CV. El 0,740 puede tener un sesgo optimista pequeño que no se puede cuantificar con estos datos. **Contrapeso:** el modelo vigente no es el de mayor F1 entre los medidos (la CV anidada dio 0,784 y no se adoptó).
2. **Sesión y lugar.** El grupo es el caballo, no la sesión. Caballos distintos de la misma sesión (misma fecha, mismo box) pueden caer en folds distintos y compartir fondo, luz y dispositivo. Sin mirar la cara, la **fecha de sesión inferida del nombre** logra F1 0,535, los metadatos técnicos 0,556 y ambos 0,605 (§10.3). La partición actual no controla este efecto.
3. **Identidad por nombre de archivo.** Un mismo caballo con dos identificadores atraviesa el control (caso `123`/`169`, §3.5; no verificable).
4. **Videos con varios animales** (`Seleccionar cabeza?`, 33 videos): la identidad del animal enfocado puede no coincidir con el id del archivo.

## 6.5 Veredicto

**No se observan evidencias de fuga de información por individuo, video, clip ni frame.** El protocolo de separación es sólido y está verificado por código. Persiste riesgo de **confusión de contexto** (sesión, lugar, dispositivo), que no es fuga de identidad pero actúa de forma análoga, y un sesgo optimista pequeño por selección no anidada de arquitectura.
