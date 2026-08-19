## 1. Diagnóstico (cerrado)

- [x] 1.1 Medir la pérdida contra frames esperados y no contra generados, con
      `reporte.frames_perdidos()`. Resultado: 400 de 1680 (23,8 %), de los cuales 176 tenían par
      ojo/hocico válido.
- [x] 1.2 Descartar la hipótesis de los `.avi` viejos. Resultado: pierden 0 %; toda la pérdida
      está en `.mp4` de 1920x1080.
- [x] 1.3 Cruzar la pérdida con la rotación declarada, con `reporte.orientacion_videos()`.
      Resultado: 0° → 0,0 % sobre 49 videos; 90° → 55,6 % sobre 13 videos.
- [x] 1.4 Medir qué hace OpenCV en el entorno real antes de corregir, con
      `reporte.prueba_orientacion()`. Resultado en Colab 4.11: la bandera es efectiva. En 4.10
      local la rotación ya se aplica sola, lo que confirma que no se podía asumir.

## 2. Corrección (implementada)

- [x] 2.1 `abrir_video()` como único punto de lectura, con `CAP_PROP_ORIENTATION_AUTO` activo.
- [x] 2.2 Usarla en el inventario, la vista previa, la segmentación y la extracción de frames.
- [x] 2.3 Dimensionar el `VideoWriter` con la forma del primer frame real e informar si difiere
      de la del inventario.
- [x] 2.4 Verificar de punta a punta con un video rotado sintético: inventario 1080x1920 y clip
      resultante vertical con todos sus frames.

## 3. Verificación sobre datos reales

- [ ] 3.1 Re-ejecutar `preparar_dataset` y `roi_normalizacion` forzando ambas.
- [ ] 3.2 Confirmar con `reporte.frames_perdidos()` que la pérdida en videos rotados cayó a
      aproximadamente cero.
- [ ] 3.3 Confirmar con `reporte.orientacion_videos()` que ya no hay diferencia de tasa entre
      videos rotados y derechos.
- [ ] 3.4 Medir la cobertura de alineación contra el umbral de 70 % y contra el 51,9 % de partida
      —sobre frames esperados, que es la cifra comparable—.
- [ ] 3.5 Verificar si los ejemplares `031`, `053`, `054`, `055`, `155`, `920` y `037` salieron
      del fondo de la tabla. Es la predicción que confirma o refuta el diagnóstico.

## 4. Documentación

- [ ] 4.1 Registrar en `openspec/project.md` la cobertura medida y la composición del dataset
      nuevo, incluidos los ejemplares por clase.
- [ ] 4.2 Dejar anotado el error de etiquetado de `501_8v1_LY1ucwR9.avi` —archivo idéntico en dos
      clases, resuelto como Disgustado— y evaluar si el criterio de armado de las carpetas dejó
      casos análogos que no comparten nombre.
