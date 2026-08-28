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

- [x] 3.1 Re-ejecutar `preparar_dataset` y `roi_normalizacion` forzando ambas.
- [x] 3.2 Confirmar con `reporte.frames_perdidos()` que la pérdida en videos rotados cayó a
      aproximadamente cero.
- [x] 3.3 Confirmar con `reporte.orientacion_videos()` que ya no hay diferencia de tasa entre
      videos rotados y derechos.
- [x] 3.4 Medir la cobertura de alineación contra el umbral de 70 % y contra el 51,9 % de partida
      —sobre frames esperados, que es la cifra comparable—.
- [x] 3.5 Verificar si los ejemplares `031`, `053`, `054`, `055`, `155`, `920` y `037` salieron
      del fondo de la tabla. Es la predicción que confirma o refuta el diagnóstico.

## 4. Documentación

- [x] 4.1 Registrar en `openspec/project.md` la cobertura medida y la composición del dataset
      nuevo, incluidos los ejemplares por clase.
- [x] 4.2 Dejar anotado el error de etiquetado de `501_8v1_LY1ucwR9.avi` —archivo idéntico en dos
      clases, resuelto como Disgustado— y evaluar si el criterio de armado de las carpetas dejó
      casos análogos que no comparten nombre.

## 5. Resultado de la verificación (19/08/2026)

Corrida de 137,5 min sobre 146 videos → 210 clips → 3360 frames esperados.

**La corrección quedó verificada sin ambigüedad:**

| | antes | después |
|---|---|---|
| Frames generados | 1280 / 1680 (76,2 %) | **3360 / 3360 (100,0 %)** |
| Pérdida en videos rotados | 55,6 % | **0,0 %** |
| Pérdida en videos derechos | 0,0 % | 0,0 % |

El inventario también quedó corregido: `037_Copia de Box 7-3.mp4` registra ahora `1080x1920`
donde antes decía `1920x1080`.

Con el dataset completo aparecen **42 videos rotados de 146**, casi el 30 % del corpus. El
problema era bastante mayor de lo que mostraban los 13 detectados sobre el dataset parcial.

**Cobertura de alineación: 51,9 % → 68,0 %** sobre frames esperados, que es la cifra comparable.
No es un A/B limpio —el dataset pasó de 62 a 146 videos en el medio—, así que parte del cambio es
composición. Sigue **66 frames por debajo** del umbral de 70 % (2286 de los 2352 necesarios).

**La predicción sobre los ejemplares se cumplió a medias.** `031` pasó de 6,2 % a 31,2 % y `053`
de 37,5 % a 46,9 %, pero `920` **empeoró** de 35 % a 6,9 % y `057` quedó igual en 42,2 %.

La explicación probable de `920`: antes perdía el 80 % de sus frames y sólo sobrevivían los
fáciles; ahora entran los 144, incluidos los difíciles, y el promedio cae. **Recuperar un frame no
es alinearlo.** Eso lo vuelve la palanca más grande que queda: `920` tiene ~134 frames sin alinear
y faltan 66 para el umbral.

**Pendiente antes de cerrar:** verificar visualmente que los recortes provenientes de videos
rotados salen derechos. 68 % es un buen número sólo si la orientación quedó bien; una cara
invertida no la delata ninguna métrica.


## Cierre — 28/08/2026

- **4.1** cumplida: `openspec/project.md` tiene la cobertura medida y la composición del dataset
  por clase y por ejemplar, actualizada tras la corrida de 368 clips.
- **4.2** cumplida: ver `duplicados.md`. **No hay ningún caso análogo al `501`.** Se buscó en
  tres pasadas —duplicados byte a byte, mismo ejemplar con misma duración y resolución, y marcas
  de tiempo repetidas— y el único par que comparte contenido (`178_20260701-081313`) resultó ser
  una secuencia **contigua**, no duplicada: la toma de `Interesado` termina donde empieza la de
  `Disgustado`.
