# 15. Buenas prácticas y condiciones de un buen modelo de clasificación

## 15.1 Checklist de buenas prácticas para datasets pequeños

| Práctica | Estado | Evidencia |
|---|---|---|
| Separar por la unidad de dependencia (individuo), no por muestra | **Cumple** | 133 caballos; 0 cruces; aserciones en `particion.py` y `clasificacion.py` (§6) |
| Validación cruzada con predicciones fuera de muestra para todo el conjunto | **Cumple** | 359/359 predicciones fuera de muestra |
| Validación interna independiente para decidir época e hiperparámetros | **Cumple, con debilidad** | Por caballo y sin test; pero es 7 caballos prolíficos, desbalanceada (§5.3) |
| Test usado una sola vez | **Cumple** | Verificado por inspección y por la prueba del test alterado (§6.3) |
| Métrica adecuada al desbalance (F1 macro, no sólo accuracy) | **Cumple** | `clasificacion.f1_macro`; equivalente a sklearn |
| Baselines triviales y baselines fuertes sobre los mismos folds | **Cumple** (desde el 25/09) | `objetivo3_lineas_de_base.json` |
| Intervalos de confianza que respeten la dependencia | **Cumple** | Bootstrap por caballo, 2 000 réplicas |
| Reportar dispersión entre folds | **Cumple** | 0,740 ± 0,052 |
| Reportar y controlar confusores de captura | **Parcial** | Subconjunto 1080×1920 y ≥ 5 s (0,705); no se controló sesión, rotación ni contenedor (§10.3) |
| Congelar el extractor con pocos datos | **Cumple** | ResNet-18 congelada; 132 355 entrenables |
| Registrar curvas de train y validación y el gap | **Cumple** (desde el 25/09) | `curva` por época; `docs/analisis_sobreajuste.md` |
| Declarar el criterio de éxito antes de medir | **Cumple, con una reserva** | Criterio 0,691 declarado antes (28/08); re-declarado después de medir como 0,761 y así consta |
| Evitar selección de modelos sobre el test (CV anidada) | **Parcial** | Selección de variantes con los mismos folds; la CV anidada se probó y no se adoptó |
| Comparar contra un baseline simple y adoptarlo si empata (parsimonia) | **Parcial** | Se detectó el empate (0,740 frente a 0,741); se conservó el MIL por decisión de los autores del 25/09 («revisable») |
| Repetir la CV con distintas particiones/semillas | **No** (hasta esta auditoría) | Una sola partición y una semilla en el proyecto; §14.2 |
| Validación externa (otro centro/cámara/observador) | **No** | Un único observador, un único establecimiento regional (§13.3) |
| Segundo observador / acuerdo sobre el dataset | **No** | Decisión del 07/10 de trabajar con un observador |
| Calibración de las probabilidades mostradas | **No** | Probabilidades softmax crudas, sobreconfiadas |
| Control de versiones y trazabilidad de resultados | **Parcial** | Registro de experimentos; ramas laterales sin fusionar |
| Entorno fijado y semillas | **Cumple** | `requirements-local.txt`, `SEMILLA` |
| Pruebas automáticas | **Cumple** | 15/15 archivos de prueba pasan |
| Documentación coherente con el código | **Parcial** | Cifras viejas en 3 archivos (§12.4) |

Resumen: de 22 prácticas, **11 cumplen**, **7 cumplen parcialmente o con reserva** y **4 no se cumplen** (repetición de CV, validación externa, segundo observador, calibración).

## 15.2 Condiciones de un buen modelo de clasificación

| Condición | Estado | Evidencia |
|---|---|---|
| Datos suficientes para la complejidad del modelo | **Parcial** | 359 clips / 133 caballos frente a 132 355 parámetros entrenables; un modelo lineal rinde igual (§11.2) |
| Clases bien definidas y etiquetas fiables | **Parcial** | Criterio EquiFACS y etogramas; un observador no ciego; sin acuerdo medido sobre el dataset |
| Balance de clases razonable | **Cumple** | IR 1,34 (clips) / 1,15 (caballos); folds variables (§4) |
| Sin fuga de información | **Cumple** | 0 cruces; test no interviene (§6) |
| Muestras independientes o con dependencia tratada | **Cumple** | Agrupación por caballo; bootstrap por caballo; el tamaño efectivo es ≈ 133–236 (§13.1) |
| Preprocesamiento coherente entrenamiento–inferencia | **Parcial** | Misma función; difieren origen de píxeles y orientación (§12.3) |
| Superar baselines relevantes | **Parcial** | Supera azar, mayoritaria, metadatos y un frame; **no** la media lineal |
| Rendimiento estable entre particiones | **Parcial** | DE 0,052 entre folds (2,6 × la del baseline); ±0,015 entre semillas |
| Generalización a individuos nuevos | **Cumple** | Evaluación por caballo |
| Generalización externa | **No evaluada** | §13.3 |
| Sobreajuste controlado | **Parcial** | Parada temprana contiene el efecto sobre el test; memoriza el train; probabilidades sobreconfiadas |
| Métricas completas (global, por clase, matriz, dispersión, IC) | **Cumple** | Capítulo 9 |
| Reproducible | **Cumple** | 359/359 bit a bit con semilla 42; ±0,015 entre semillas |
| Trazable | **Parcial** | Registro de experimentos; ramas sin fusionar; fragilidad de la huella |
| Interpretable / explicable | **Parcial** | Atención por frame en la interfaz; sin análisis de qué rasgos usa (la señal del contexto no se descarta) |
| Utilidad real validada con el usuario | **No evaluada** | Sin validación con veterinarios |

Resumen: de 16 condiciones, **6 cumplen**, **8 son parciales** y **2 no están evaluadas** (generalización externa y utilidad con el usuario).
