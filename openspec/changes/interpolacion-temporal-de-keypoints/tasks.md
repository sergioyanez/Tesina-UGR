# Tareas — interpolación temporal de keypoints

## 1. Validación primero, implementación después

- [x] 1.1 Escribir el experimento de dejar uno afuera sobre los 3824 frames alineados: ocultar la
      detección de un frame, interpolar desde los vecinos, comparar contra la real.
- [x] 1.2 Medir el error en las tres unidades: píxeles en la imagen original, fracción de la
      distancia ojo-hocico, y píxeles dentro del recorte de 224.
- [x] 1.3 Estratificar por largo de hueco simulado (1, 2, 3, 4, ≥5 frames) e informar mediana, p90
      y máximo de cada estrato.
- [x] 1.4 Repetir el protocolo para los extremos, ocultando el primer o el último frame alineado y
      sosteniendo el valor del vecino.
- [x] 1.5 Fijar `MAX_HUECO_INTERPOLADO` y `MAX_EXTREMO_SOSTENIDO` con la curva medida, no con
      criterio, y dejar registrado el umbral declarado y si se cumplió.
      **`MAX_HUECO_INTERPOLADO = 1`, `MAX_EXTREMO_SOSTENIDO = 0`.** El umbral original no lo
      cumplió ningún largo y se re-declaró anclándolo en la línea de base, con el motivo escrito
      en la spec y en `validacion.md`. El informe imprime el resultado contra los dos.
- [x] 1.6 Comprobar si el error crece con la velocidad aparente del caballo entre los frames
      vecinos, que es el sesgo conocido de esta validación.

## 2. Implementación en el notebook 02

- [x] 2.1 Agregar la función de relleno temporal de keypoints junto a `rellenar_faltantes()`, con
      el tope de hueco como parámetro.
- [x] 2.2 Agregar `MAX_HUECO_INTERPOLADO` y `MAX_EXTREMO_SOSTENIDO` a la celda de parámetros.
- [x] 2.3 Construir la transformada de los frames rescatados con `similitud_desde_dos_puntos()`,
      sin un segundo camino de construcción.
- [x] 2.4 Registrar `origen_alineacion` por frame y propagarlo a `dataset_frames_alineados.csv`.
- [x] 2.5 Verificar la geometría del recorte rescatado contra un caso conocido, como se hizo con
      `calidad_recortes()` — su primera versión tenía la rotación traspuesta y erraba 240 px sin
      romper nada.

## 3. Reportes

- [x] 3.1 Que `reporte.cobertura()` desglose por `origen_alineacion` y no sume todo bajo
      "alineado".
- [x] 3.2 Agregar al reporte la métrica que hoy falta y que es la que decide: **clips con los 16
      frames alineados**, por clase y por ejemplar.

## 4. Re-derivación y medición

- [x] 4.1 Re-derivar los recortes sin reinferir DeepLabCut.
- [x] 4.2 Contrastar contra lo esperado: 80 clips íntegros hoy, 180 rellenando huecos interiores,
      330 si además se sostienen los extremos.
      **Corregido con la medición real (28/08): la proyección no se sostiene.** Con el tope
      autorizado (1) son 130 clips, no 180; con tope 2 serían 153 y con tope 3, 166. Los 330 de
      los extremos quedan descartados: `MAX_EXTREMO_SOSTENIDO = 0`.
- [x] 4.3 Reportar cuántos ejemplares de `Disgustado` quedan con al menos un clip íntegro (4 hoy;
      19 esperados con relleno interior) y si el dataset admite `GroupKFold` por caballo.
      **Medido antes de re-derivar: 5 hoy, 8 con el tope autorizado.** Los 19 esperados exigían
      un tope de 3, que la validación no respalda. El dataset sigue sin admitir `GroupKFold`
      por caballo en `Disgustado`.
- [x] 4.4 Actualizar `openspec/project.md` con el estado y con la métrica nueva.


## Estado al 28/08/2026

**Las 17 tareas están cerradas.** Re-derivado el 28/08/2026 sin reinferir DeepLabCut: la etapa 02
reusó la pose persistida y sólo rehizo los recortes.

Resultado medido, que coincide exactamente con lo proyectado antes de correr —261 frames
interpolados son los 261 huecos de largo 1 del catálogo, lo que da una verificación cruzada de que
el relleno hace lo que dice:

| | antes | después |
|---|---:|---:|
| cobertura global | 68,9 % | **73,5 %** (68,9 % contando sólo detectadas) |
| clips íntegros | 81 | **130** |
| `Disgustado` clips/caballos | 6 / 5 | **10 / 8** |

**El dataset sigue sin admitir `GroupKFold` por caballo en `Disgustado`.** Lo que falta ya no es
software.

Artefactos nuevos:

- `equinos/interpolacion.py` — el relleno (`rellenar_clip`) y su validación completa
  (`informe_validacion`). La geometría se extrae del notebook 02, no se duplica.
- `tests/test_interpolacion.py` — 34 pruebas. Las que importan son las que verifican que la
  medición **no puede dar cero sola**: sobre movimiento lineal el error es 0, sobre movimiento
  acelerado es mayor que 0, y una estimación corrida 30 px produce error distinto de cero.
- `equinos/reporte.py::clips_integros()` — la métrica que decide, por clase y por ejemplar.
- `validacion.md` — el resultado medido, con los dos umbrales y el sesgo acotado.
- `02_roi_y_normalizacion.ipynb` — `MAX_HUECO_INTERPOLADO` y `MAX_EXTREMO_SOSTENIDO` en la celda
  de parámetros, el relleno aplicado antes del recorte usando la MISMA función que se validó
  (`interpolacion.rescates_por_clip`, sin un segundo camino de construcción), y
  `origen_alineacion` registrado por frame.
- `equinos/reporte.py::cobertura()` — desglose por origen, con la cobertura recalculada contando
  sólo las posiciones detectadas, que es la comparable con las corridas anteriores.

Lo que cambió respecto de lo planificado: **la proyección de 180 clips íntegros y 19 ejemplares de
`Disgustado` no se sostiene.** Con el tope que la evidencia respalda son 130 clips y 8 ejemplares.
