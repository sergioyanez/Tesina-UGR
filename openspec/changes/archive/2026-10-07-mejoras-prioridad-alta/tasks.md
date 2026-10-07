## 1. Líneas de base y control de confusores

- [x] 1.1 Crear `equinos/lineas_de_base.py`: mayoritaria, aleatorio estratificado, visual estática (1 frame y media de 16), geometría sola, metadatos de filmación, duración sola, bootstrap por caballo y evaluación por subconjunto
- [x] 1.2 Crear `tests/test_lineas_de_base.py` (bootstrap pareado, agrupamiento por caballo, que no haya fuga de estandarización)
- [x] 1.3 Crear `scripts/evaluar_lineas_de_base.py` y persistir `objetivo3_lineas_de_base.json`
- [x] 1.4 Correrlo y verificar que reproduce las cifras de la auditoría

## 2. Registro del sobreajuste

- [x] 2.1 Agregar a la curva de `entrenar_mil_multimodal` pérdida, F1 y accuracy del train interno sin dropout
- [x] 2.2 Agregar al resumen de `scripts/evaluar_v2.py` el F1 de train por fold y los gaps
- [x] 2.3 Prueba: registrar el train no cambia predicciones

## 3. Trazabilidad del modelo desplegado

- [x] 3.1 `_huella_caracteristicas` lee `caracteristicas.ARCHIVO_HUELLA`; prueba asociada
- [x] 3.2 Re-ejecutar `scripts/evaluar_v2.py`; verificar predicciones idénticas y comparar el hash de los pesos de producción

## 4. Figuras y notebook 03

- [x] 4.1 `04_figuras_tesis.ipynb`: figura 4 desde la evaluación vigente y figura 5 (curvas y gap)
- [x] 4.2 Reescribir `03_clasificacion.ipynb` para el protocolo vigente, con las líneas de base nuevas
- [x] 4.3 Re-ejecutar los notebooks 03 y 04 y revisar que texto y salidas coincidan

## 5. Documentación

- [x] 5.1 `docs/registro_de_experimentos.md`
- [x] 5.2 `docs/protocolo_de_etiquetado.md`
- [x] 5.3 `docs/analisis_sobreajuste.md`
- [x] 5.4 Actualizar `README.md` y `openspec/project.md`
- [x] 5.5 `openspec validate mejoras-prioridad-alta`
