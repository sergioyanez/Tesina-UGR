# Tareas — piso de video corto

## 1. Medición del problema

- [x] 1.1 Contar los ejemplares por clase en el catálogo completo y compararlos con los que llegan
      al dataset: `Disgustado` 42 contra 11.
- [x] 1.2 Medir la distribución de duraciones de `Disgustado`: mediana 2,9 s, 55 de 70 videos por
      debajo del piso de 5 s.
- [x] 1.3 Escribir un simulador que replique `calcular_ventanas` y barrer configuraciones sobre el
      inventario real, sin re-ejecutar nada.
- [x] 1.4 Modelar el costo de la etapa 02 por segundos de material y no por cantidad de clips:
      2,5 h contra 2,7 h; sólo el solapamiento la lleva a 4,2 h.

## 2. Implementación

- [x] 2.1 Agregar `PISO_VIDEO_CORTO` a la celda de parámetros del notebook 01, con el argumento
      escrito ahí mismo.
- [x] 2.2 Extender `calcular_ventanas`: si no salió ninguna ventana y el video llega al piso, entra
      entero como un clip único.
- [x] 2.3 Agregar `PISO_VIDEO_CORTO` a `CLAVES_SEGMENTACION` para que el checkpoint se invalide.
- [x] 2.4 Ejecutar la función parchada contra el inventario antes de correr nada: 342 clips,
      ninguno con menos de 16 frames, el más corto de 2,00 s.

## 3. Re-ejecución

- [x] 3.1 Correr la etapa 01 completa; el checkpoint se invalida solo por la clave nueva.
- [x] 3.2 Verificar que los 280 clips previos se regeneraron con rango de frames y clase idénticos,
      y que ninguno desapareció — es lo que habilita reusar el checkpoint de la etapa 02.
- [x] 3.3 Correr la etapa 02 sobre los 62 clips nuevos (35 `Disgustado`, 15 `Interesado`,
      12 `Sereno`).

## 4. Medición del resultado

- [x] 4.1 Cobertura por clase: `Disgustado` 57,6 % -> 62,1 %, `Interesado` 80,2 % -> 80,3 %,
      `Sereno` 65,4 % -> 65,1 %.
- [x] 4.2 Contrastar clips cortos contra largos: 68,5 % contra 70,2 %.
- [x] 4.3 Contar ejemplares con material alineado: `Disgustado` 11 -> 32, y con al menos un clip
      completo alineado, 4 -> 12.
- [x] 4.4 Dejar explicado por qué la cobertura global baja 0,3 pp sin que ninguna clase empeore.
