## 1. Inferencia coherente con el entrenamiento
- [x] 1.1 `equinos/video.py`: `abrir_video`, `rotacion_declarada`, `normalizar_video` (ORIENTATION_AUTO + mp4v)
- [x] 1.2 `inferencia.preparar_clip(normalizar=True, info=...)` y `clasificar_video(normalizar=True)`; `normalizar=False` conserva lo anterior
- [x] 1.3 Prueba e2e sobre los 16 videos antes/después y comparación contra la caché (`comparacion/verificacion_inferencia/`)

## 2. Geometría
- [x] 2.1 `equinos/geometria.py` con v2 (17 columnas), huella y caché propias
- [x] 2.2 `MultimodalEquineDataset(geometria, umbral_mascara)`; `entrenar_mil_multimodal(geometria=...)` (modelo desplegado intacto)

## 3. Validación interna
- [x] 3.1 `particion.dividir_estratificada` y `dividir_validacion(modo)`; `entrenar_mil_multimodal(validacion_interna=...)`

## 4. Protocolo de comparación
- [x] 4.1 Reserva al 15 % persistida con huella (`reserva_caballos.json`) y puerta `evaluar_en_reserva`
- [x] 4.2 Particiones 5×4 (`particiones.json`)
- [x] 4.3 Interfaz `Candidato` y evaluador con esquema `comparacion-v1`; `comparar` (bootstrap pareado por caballo)
- [x] 4.4 `comparacion/PROTOCOLO.md` con la decisión del 07/10/2026
- [x] 4.5 Baseline LogReg como prueba de humo (`resultados/baseline_logreg_media_frames.json`)

## 5. Sesiones
- [x] 5.1 `comparacion/sesiones_de_filmacion.csv` (lugar y sesion_confirmada vacías)

## 6. Tests y documentación
- [x] 6.1 `test_rotacion_inferencia.py`, `test_geometria_v2.py`, `test_comparacion.py`
- [x] 6.2 Suite completa en verde
- [x] 6.3 `openspec/project.md` actualizado

## 7. Pendientes (no son de este change)
- [ ] 7.1 Sergio completa `lugar` y `sesion_confirmada` en la tabla de sesiones
- [ ] 7.2 Implementar los candidatos 2, 3 y 4 en sus worktrees con `comparacion.evaluar`
- [ ] 7.3 Uso único de la reserva con el ganador (el criterio ya está declarado; sólo se ejecuta tras fijar el ganador)
