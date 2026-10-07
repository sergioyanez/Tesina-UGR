## Context

Dataset: 359 clips íntegros de 133 caballos (Disgustado 103, Interesado 118, Sereno 138 clips; ver `comparacion/reserva_caballos.json` para la composición por conjunto). Con tan pocos caballos, el cuello de botella son los individuos: toda comparación debe agrupar por caballo, repetir las particiones y acotar la incertidumbre remuestreando caballos. La auditoría (`auditoria/secciones/`, caps. 6, 7, 11.6, 12, 17) dejó cuatro defectos de base que este change corrige o aísla antes de medir candidatos.

## Decisiones

### D1. Inferencia: normalizar el video como los clips (no sólo activar la bandera)

DeepLabCut abre el archivo por su cuenta, así que activar `CAP_PROP_ORIENTATION_AUTO` en el lector de OpenCV de la interfaz no alcanza: la pose se seguiría estimando sobre el frame sin rotar. Se recodifica el video completo con la misma rutina que el notebook 01 (`equinos/video.normalizar_video`: orientación + `mp4v`, escritor con la forma del primer frame) y se corre pose y recortes sobre esa copia temporal (fuera de la raíz de datos). Esto iguala de una vez orientación, recodificación `mp4v` y lectura por búsqueda de fotograma (el clip del dataset también se leyó con `set(POS_FRAMES)` sobre un `.mp4` `mp4v`). El viaje por JPEG ya era igual (`_caracteristicas_de`).
Descartado: rotar sólo los recortes (la pose ya estaría mal); codec sin pérdida (sería *distinto* de lo que vio el modelo).
Alternativa conservada: `normalizar=False` reproduce la lectura anterior, sólo para medir.

### D2. Geometría v2 como módulo aparte, v1 intacta

`equinos/geometria.py` define 17 columnas en el marco de la cabeza (eje ojo→hocico, escala = distancia ojo-hocico del frame, sin estadística global): vectores base→punta de ambas orejas (4), vector hocico→`lower_jaw` (2), ancho de boca, distancia interocular, escorzo (escala del frame / mediana del clip), seno y coseno de la inclinación (3 más 2) y 6 máscaras con umbral 0,50. Ojo y hocico definen el marco y no se repiten. La v1 se conserva con un parámetro `umbral_mascara` (`None` = histórico `> 0`, que es lo que vio el modelo desplegado; 0,50 = corregido): cambiar la máscara por omisión alteraría en silencio la entrada del modelo desplegado. La huella (`geometria.huella`) incluye versión, revisión, umbral, nombres de columnas; la caché de v2 lleva la huella en el nombre.

### D3. Validación interna estratificada

`StratifiedGroupKFold(n_splits=round(1/fracción), shuffle=True, random_state=42)` y primer pliegue como validación. El modo histórico (`mas_clips`) sigue siendo el valor por omisión.

### D4. Protocolo

- Reserva 15 % (decisión de Sergio, 07/10/2026) por clase dominante del caballo, sorteada con semilla fija; huella sobre dataset y contenido; una sola puerta (`evaluar_en_reserva`) con flag y motivo explícitos, log y negativa a un segundo uso.
- 5 particiones × 4 folds por caballo con `StratifiedGroupKFold` (semillas 1101–1105) sobre desarrollo; se persiste caballo→fold.
- Candidato = `ajustar(ids_train, datos)` / `predecir(ids, datos)`; `Datos` da acceso por id de clip a etiqueta, caballo, características visuales, pose y metadatos.
- Métricas con scikit-learn (`average='macro'`, `labels` fijos, `zero_division=0`). Subconjunto limpio: F1 de las predicciones fuera de muestra por repetición (las 4 predicciones de cada clip salen de folds distintos; un F1 por fold en limpio dejaría clases con 0–3 clips). Gap = F1 de train − F1 de test.
- IC95: bootstrap pareado por caballo (2 000 réplicas), misma réplica para ambos candidatos.
- Criterio de elección y regla de empate: `comparacion/PROTOCOLO.md` (declarado antes de comparar).

## Hallazgos de la implementación (a conservar)

### H1. Prueba extremo a extremo de `clasificar_video` sobre los 16 videos de la auditoría
Antes (lectura anterior, `normalizar=False`): **8/16** coinciden con la etiqueta (coincide el resultado de la auditoría). Después (`normalizar=True`): **16/16**. El modelo desplegado, aplicado a las características de la caché de entrenamiento de esos mismos 16 clips, acierta 16/16 (los entrenó). Con la lectura anterior la predicción de inferencia igualaba la de la caché en 8/16; ahora en 16/16. Resultados: `comparacion/verificacion_inferencia/e2e_legacy.json` y `e2e_normalizado.json`.

### H2. Aislamiento del efecto de la lectura (sin DeepLabCut; `fidelidad_pixeles.json`)
Con pose y frames fijados por la caché, reconstruyendo los recortes desde cuatro fuentes y comparando características contra la caché (coseno medio, 16 videos): clip del dataset 1,000 (control); original normalizado como hace ahora la inferencia **1,000, diferencia relativa 0** (idéntico bit a bit en la práctica); original orientado sin recodificar 0,995 (diferencia relativa 8,8 %: es el efecto de `mp4v`); original crudo como antes 0,905 (con rotación 0,817; sin rotación 0,994). Predicción del modelo con la lectura anterior: cambia en 3 de 16 (los tres con rotación).

### H3. Discrepancias residuales tras el arreglo (cada una con su causa)
Sobre los frames en común entre la ventana de inferencia y la de la caché: características con coseno 1,000 y diferencia de pose 0,0 px en 15 de 16 videos (DeepLabCut sobre el video entero da la misma pose que sobre el clip; el detector no introduce diferencia medible). Lo que queda es **sólo selección de ventana**: (a) el clip del dataset cubre los primeros ≈7 s (210 frames) y el video es más largo (p. ej. 288, 302, 307 frames), la inferencia elige la mejor ventana ≤ 7 s del video completo y puede extenderse más allá del clip (videos `207_…151913`, `230_…120401`, `217_…144422`: 10–15 frames en común); (b) `912_20260702_142421` (263 frames): la ventana elegida (frames 188–261) cae casi toda fuera del clip (0–209), sin frames en común, y aun así la predicción coincide; (c) `914_…141211`: 15 de 16 frames en común, con las mismas ventanas; no se investigó qué frame ancla difiere (probablemente el desempate de la elección entre candidatos casi iguales), así que es una causa no verificada. Estas diferencias son de diseño (unidad de entrada: clip de 5–7 s frente a video de hasta 15 s) y NO se eliminan.

### H4. Geometría
Sobre los 359 clips: con la máscara histórica las tres máscaras valen 1,00; con umbral 0,50 en v1 sólo cambia la de oreja (0,81); ojo y hocico valen 1,00 porque la selección de frames ya exige el par ojo-hocico ≥ 0,50. En v2: oreja izquierda 0,55, oreja derecha 0,51, mandíbula 0,79, **ancho de boca 0,09** (las comisuras casi nunca superan 0,50: la columna es casi vacía y se conserva con su máscara), interocular 0,91, eje 1,00. Dimensiones efectivas (95 % de la varianza de las columnas continuas): v1 3, v2 9.

### H5. Validación interna
Sobre los 4 folds externos vigentes, el modo histórico deja 7 caballos en validación (70–73 clips) con composición D/I/S de 11/13/46 a 26/19/28; el estratificado deja 22–28 caballos (58–76 clips; p. ej. 12/26/34, 15/24/19, 10/18/38, 20/28/28).

### H6. Sesiones
274 videos: fecha del nombre en 239, de `creation_time` (ffprobe) en 7 y **28 sin fecha deducible**; 24 sesiones propuestas (corte de 120 min). La fecha de ffprobe puede ser la de re-exportación; la columna `fuente_fecha` lo distingue. `lugar` y `sesion_confirmada` quedan vacías.

## Riesgos

- Las predicciones de la interfaz cambian para videos con rotación declarada (para bien): conviene volver a medir la tasa de negativa (42 % anterior a la selección por ventana).
- Normalizar recodifica el video completo (≤ 15 s): +unos segundos por video.
- La v2 no se probó con modelos entrenados en este change.
