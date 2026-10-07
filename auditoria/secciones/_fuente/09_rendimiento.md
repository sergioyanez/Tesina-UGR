# 9. Rendimiento: métricas, estabilidad entre folds, confusiones y cifras históricas

## 9.1 El valor vigente (único)

**Archivo que constituye la evaluación vigente: `datos_equinos/resultados/objetivo3_evaluacion_v2.json`** (generado por `scripts/evaluar_v2.py`; re-ejecutado el 25/09/2026 con resultado idéntico). Es también el archivo del que lee la interfaz (`clasificacion.ARCHIVO_EVALUACION`).

| Métrica (fuera de muestra, 359 clips, 133 caballos) | Valor |
|---|---|
| **F1 macro (media de los 4 folds ± DE muestral)** | **{{VAL:metricas.json:estabilidad.f1_macro_media}} ± {{VAL:metricas.json:estabilidad.f1_macro_desvio_ddof1}}** |
| IC 95 % del F1 macro (bootstrap por caballo, 2 000 réplicas, F1 agrupado) | [{{VAL:metricas.json:bootstrap.modelos.MIL multimodal (vigente).f1_ic95.0}}; {{VAL:metricas.json:bootstrap.modelos.MIL multimodal (vigente).f1_ic95.1}}] |
| F1 macro agrupado (las 359 predicciones juntas) | {{VAL:metricas.json:global.f1_macro_agrupado_sklearn}} |
| Accuracy | {{VAL:metricas.json:global.accuracy}} (267/359) |
| Balanced accuracy | {{VAL:metricas.json:global.balanced_accuracy}} |
| F1 ponderado agrupado | {{VAL:metricas.json:global.f1_ponderado_agrupado}} |

**Verificación de la métrica.** `clasificacion.f1_macro` es una implementación propia de la media **no ponderada** del F1 por clase (cada clase pesa igual). Se comprobó contra `sklearn.metrics.f1_score(average="macro", labels=CLASES)` en los 4 folds con tolerancia 10⁻¹²: coinciden. Hay **dos formas de «F1 macro»** que no deben mezclarse: la **media de los F1 macro por fold (0,740)**, que es la que informa el proyecto, y el F1 macro de las predicciones agrupadas (0,742). La desviación estándar usa *n − 1* (0,0524); con *n* sería 0,0454. Con sólo 4 folds, es una estimación gruesa de la variabilidad.

## 9.2 Estabilidad entre folds

{{TABLA:metricas_fold}}

{{TABLA:estabilidad}}

El fold 2 es el peor (0,677) y el fold 3 el mejor (0,795): una diferencia de 0,118, dentro de lo que cabe esperar con ≈ 33 caballos por fold y composiciones de clase distintas (capítulo 4). **No hay folds colapsados**, pero la dispersión es 2,6 veces la del baseline lineal (DE 0,020).

## 9.3 Precision, recall y F1 por clase

{{TABLA:por_clase}}

{{TABLA:prf_folds}}

## 9.4 Matrices de confusión

**Global (agregada sobre los 4 folds; filas = clase real):**

{{TABLA:confusion_global}}

{{FIG:fig_confusion_global_y_folds.png|Figura 9.1. Matriz de confusión global (izquierda) y de los folds 0 a 2; la del fold 3 está en la tabla de abajo. Porcentajes por fila.}}

**Por fold:**

{{TABLA:confusion_folds}}

- Los errores se reparten sin una dirección dominante clara: `Sereno → Disgustado` (20), `Disgustado → Sereno` (19), `Interesado → Sereno` (19) y `Sereno → Interesado` (16). La confusión entre extremos conceptuales (`Interesado` ↔ `Disgustado`: 13 + 5) es la menor.
- **Los errores se concentran en pocos caballos**: 55 de 133 caballos tienen al menos un error y los 10 peores acumulan el 40,2 % de los 92 errores (`906`: 6/13; `013`: 5/5; `037`: 5/14; `053`: 4/4; `044`, `251`, `916`: 3/3). 9 caballos con ≥ 2 clips fallan en el 100 % de sus clips. Es un hallazgo del conjunto de datos (candidatos a revisión de etiqueta o de identidad), no sólo del modelo.

## 9.5 Cifras históricas y su estatus

Ninguna de estas cifras es comparable con 0,740: cambian el conjunto de clips y, en las primeras, el protocolo de selección de época.

{{TABLA:historico}}

**Regla de uso:** citar sólo `objetivo3_evaluacion_v2.json` como resultado del modelo vigente, siempre acompañado de `objetivo3_lineas_de_base.json`. La cifra **0,754** no tiene artefacto que la respalde (apareció escrita a mano en `clasificacion.py` el 15/09, en el commit `6b66044`, y se retiró el 18/09) y no debe citarse.
