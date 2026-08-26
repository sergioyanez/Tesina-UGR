# Tareas — interpolación temporal de keypoints

## 1. Validación primero, implementación después

- [ ] 1.1 Escribir el experimento de dejar uno afuera sobre los 3824 frames alineados: ocultar la
      detección de un frame, interpolar desde los vecinos, comparar contra la real.
- [ ] 1.2 Medir el error en las tres unidades: píxeles en la imagen original, fracción de la
      distancia ojo-hocico, y píxeles dentro del recorte de 224.
- [ ] 1.3 Estratificar por largo de hueco simulado (1, 2, 3, 4, ≥5 frames) e informar mediana, p90
      y máximo de cada estrato.
- [ ] 1.4 Repetir el protocolo para los extremos, ocultando el primer o el último frame alineado y
      sosteniendo el valor del vecino.
- [ ] 1.5 Fijar `MAX_HUECO_INTERPOLADO` y `MAX_EXTREMO_SOSTENIDO` con la curva medida, no con
      criterio, y dejar registrado el umbral declarado y si se cumplió.
- [ ] 1.6 Comprobar si el error crece con la velocidad aparente del caballo entre los frames
      vecinos, que es el sesgo conocido de esta validación.

## 2. Implementación en el notebook 02

- [ ] 2.1 Agregar la función de relleno temporal de keypoints junto a `rellenar_faltantes()`, con
      el tope de hueco como parámetro.
- [ ] 2.2 Agregar `MAX_HUECO_INTERPOLADO` y `MAX_EXTREMO_SOSTENIDO` a la celda de parámetros.
- [ ] 2.3 Construir la transformada de los frames rescatados con `similitud_desde_dos_puntos()`,
      sin un segundo camino de construcción.
- [ ] 2.4 Registrar `origen_alineacion` por frame y propagarlo a `dataset_frames_alineados.csv`.
- [ ] 2.5 Verificar la geometría del recorte rescatado contra un caso conocido, como se hizo con
      `calidad_recortes()` — su primera versión tenía la rotación traspuesta y erraba 240 px sin
      romper nada.

## 3. Reportes

- [ ] 3.1 Que `reporte.cobertura()` desglose por `origen_alineacion` y no sume todo bajo
      "alineado".
- [ ] 3.2 Agregar al reporte la métrica que hoy falta y que es la que decide: **clips con los 16
      frames alineados**, por clase y por ejemplar.

## 4. Re-derivación y medición

- [ ] 4.1 Re-derivar los recortes sin reinferir DeepLabCut.
- [ ] 4.2 Contrastar contra lo esperado: 80 clips íntegros hoy, 180 rellenando huecos interiores,
      330 si además se sostienen los extremos.
- [ ] 4.3 Reportar cuántos ejemplares de `Disgustado` quedan con al menos un clip íntegro (4 hoy;
      19 esperados con relleno interior) y si el dataset admite `GroupKFold` por caballo.
- [ ] 4.4 Actualizar `openspec/project.md` con el estado y con la métrica nueva.
