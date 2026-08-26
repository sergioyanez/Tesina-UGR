# Tareas — fuente única de videos

## 1. Diagnóstico

- [x] 1.1 Reconstruir por qué la corrida del 24/08 midió sobre 223 videos: la columna `ruta` del
      inventario y los tiempos de copia del espejo lo delatan.
- [x] 1.2 Encontrar la corrida abortada (`ejecuciones/preparar_dataset_20260824_120931_FALLIDA`)
      y su causa: el `501` duplicado entre dos clases.
- [x] 1.3 Medir el efecto sobre la métrica: 72,0 % contra 70,2 % sobre el material completo, y de
      dónde salían los 2,0 pp de diferencia (los 12 videos ausentes alinean al 55,8 %).

## 2. Unificación de la fuente

- [x] 2.1 Verificar que el espejo es subconjunto estricto de la fuente antes de tocarlo: 223
      archivos, todos con el mismo tamaño, y un único archivo presente sólo en el espejo (el `501`).
- [x] 2.2 Retirar el espejo a `datos_equinos/videos_espejo_retirado_20260825`.
- [x] 2.3 Comprobar que sin `TESINA_EQUINOS_VIDEOS` la resolución muere con el mensaje de
      `catalogo.raiz_videos` en vez de continuar.
- [x] 2.4 Comprobar que con las variables puestas el catálogo informa 234 videos (70/71/93) y las
      tres subcarpetas declaradas, incluida la que tiene un espacio final en el nombre.

## 3. Rutas del entorno de ejecución

- [x] 3.1 Agregar `TESINA_EQUINOS_RAIZ` y `TESINA_EQUINOS_VIDEOS` al `kernel.json` del kernel
      `tesina-equinos`, con respaldo del original.
- [x] 3.2 Cambiar la raíz por defecto de `correr_local.sh` a `$REPO/datos_equinos` y explicar en el
      encabezado las tres defensas que hoy existen.
- [x] 3.3 Agregar el subcomando `etapa01`, con el motivo escrito: `etapas.estado()` no compara los
      artefactos contra el catálogo.

## 4. Retiro del video con etiqueta contradictoria

- [x] 4.1 Respaldar `checkpoints/` antes de tocar nada.
- [x] 4.2 Sacar el `501` de `preparar_dataset.json` (223 -> 222 unidades) y de su CSV parcial
      (251 -> 250 filas).
- [x] 4.3 Sacarlo de `roi_normalizacion.json` (251 -> 250) y de su CSV parcial (156624 -> 156000).
- [x] 4.4 Apartar su clip, sus frames y su JSON de inferencia a `_retirado_501_20260825`.

## 5. Re-medición sobre el material completo

- [x] 5.1 Correr la etapa 01 con el checkpoint puesto: procesa los 12 videos pendientes y ninguno
      más.
- [x] 5.2 Correr la etapa 02 sobre los 30 clips nuevos.
- [x] 5.3 Verificar que el `501` no aparece en ningún artefacto derivado.
- [x] 5.4 Reportar la cobertura desglosada, no sólo la global: 68,0 % sobre los 146 videos
      originales, 77,1 % sobre el material nuevo del catálogo, 55,8 % sobre los 12 recuperados,
      70,2 % en total.
