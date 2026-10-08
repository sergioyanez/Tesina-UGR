# 7. Validación cruzada, validación interna y *early stopping*

## 7.1 Protocolo

| Elemento | Valor | Fuente |
|---|---|---|
| Validación cruzada externa | 4 folds agrupados por caballo (≈ 33 caballos de test por fold) | `particion.FOLDS = 4`, `particion_clips.json` |
| Por qué 4 y no 5 | Con 5 folds hay folds con ~2 clips de `Disgustado` en test (comentario de `particion.py`; no se re-verificó) | `equinos/particion.py` |
| Validación interna | 25 % de los clips del `train_val`, por caballo (7 caballos), determinista | `particion.dividir_por_ejemplar` |
| Métrica de selección | **F1 macro de la validación interna** (`clasificacion.f1_macro`) | `entrenar_mil_multimodal` |
| Criterio de mejora | estricto (`f1_val > mejor`); un empate no reemplaza al checkpoint | ídem |
| Paciencia / máximo de épocas | **10 / 100** (valores por defecto: `evaluar_v2.py` no los pasa) | ídem |
| Checkpoint | copia del `state_dict` de la mejor época en memoria; se restaura antes de predecir el test | ídem |
| Optimizador | Adam, lr 1e-3, `weight_decay` 1e-4, batch 32; sin *scheduler* | ídem |
| Pérdida | entropía cruzada con pesos de clase `raiz` (raíz de la frecuencia inversa, calculada sobre el train interno) | ídem |
| Test | se predice **una sola vez**, tras cargar el mejor estado | ídem (verificado por inspección del código) |
| Épocas del modelo de producción | `ceil(media de la mejor época)` = 29, sin validación (usa los 359 clips) | `evaluar_v2.py`, `mil_multimodal_produccion.json` |

## 7.2 Resultado por fold

{{TABLA:early_stopping}}

- El máximo de 100 épocas **nunca se alcanzó**: la ejecución se detiene exactamente 10 épocas después de la mejor (paciencia agotada).
- **La época se elige con la validación y no con el test**, y esto está verificado de tres maneras: (1) inspección del código (el único `predecir(dl_pru)` ocurre después de `load_state_dict(mejor)`); (2) la prueba del test alterado de §6.3 (pesos idénticos aun con las etiquetas de test permutadas); (3) la re-ejecución con la semilla 42 reproduce las curvas de validación.
- **La mejor época varía de 20 a 40 entre folds** (media 28,5, DE 8,4) con la misma semilla; entre semillas el rango sube a 13–50 (§14). El F1 de validación de la mejor época es un *máximo sobre una serie ruidosa* (7 caballos): el F1 de validación medio en la ventana de ±3 épocas alrededor de la mejor es 0,03–0,04 más bajo que el máximo en los 4 folds (`datos/early_stopping_tabla.csv`), de modo que **la métrica de validación de la mejor época es optimista por selección**; la métrica de test, no.
- El test es **mayor o igual que la validación en los 4 folds** (diferencia media de +0,027). Es la comprobación empírica de que la estimación de test no está inflada por la selección; la dirección opuesta (test > validación) es coherente con una validación interna más difícil o más ruidosa que el test (§5.3), no con una fuga.

## 7.3 Qué no se puede afirmar

- Que la época 28,5 sea «la» correcta: la curva de F1 de validación es una meseta ruidosa y las alternativas probadas (CV anidada con curva promediada) no la estabilizaron (`docs/registro_de_experimentos.md`).
- Que el *early stopping* reduzca la memorización: la detiene antes de que *empeore* el F1 de validación, pero el F1 de train ya es 0,97–0,99 en la época elegida (capítulo 8).
