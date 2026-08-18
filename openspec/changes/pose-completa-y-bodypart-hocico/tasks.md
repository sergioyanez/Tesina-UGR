## 1. Verificar el terreno antes de escribir código

- [ ] 1.1 Confirmar en Colab qué bodyparts expone realmente el modelo de pose (el notebook 02 ya los
      imprime en el paso de descubrimiento del esquema) y cuáles de `nose`, `snout`, `muzzle`,
      `upper_jaw` y `mouth` existen. **Si `nose` es el único candidato de hocico, este change no
      tiene camino y hay que replantearlo antes de seguir.**
- [ ] 1.2 Inventariar los videos nuevos de Disgustado: cuántos son, qué duración tienen y qué
      ejemplares aportan. Registrar si alguno es distinto de `037`, porque de eso depende que la
      clase sea evaluable bajo partición por caballo.

## 2. Persistir la pose completa (notebook 02)

- [x] 2.1 En la celda que hoy borra los intermedios de DeepLabCut, extraer del `.h5` la tabla larga
      `(clip, frame, bodypart, x, y, confianza)` antes de eliminar el archivo.
- [x] 2.2 Acumular esa tabla por clip a través del checkpoint de la etapa, igual que se acumula el
      CSV de keypoints, para que una interrupción no la pierda.
- [x] 2.3 Escribir `resultados/pose_completa.csv` al cierre de la etapa y verificar en esa misma
      corrida que el tamaño y el número de filas quedan en el orden esperado.
- [x] 2.4 Registrar `resultados/pose_completa.csv` entre los `artefactos` de `roi_normalizacion` en
      `equinos/etapas.py`.
- [x] 2.5 Verificar la trazabilidad: que cada fila de `keypoints_faciales.csv` sea reconstruible
      desde `pose_completa.csv` aplicando el criterio de selección vigente.

## 3. Barrido de candidatos (paquete `equinos`)

- [x] 3.1 Módulo nuevo que lea `pose_completa.csv` y calcule, para cada candidato a hocico, la
      proporción de frames que superan el umbral junto al keypoint ocular, global y desglosada por
      clase y por caballo.
- [x] 3.2 Agregar al barrido la estrategia de máxima confianza por frame, como una opción más de la
      comparación.
- [x] 3.3 Reportar la variación de la distancia ojo-hocico entre candidatos, que es el dato que
      decide si esa estrategia es geométricamente admisible (decisión 4 del diseño).
- [x] 3.4 Manejar el candidato ausente informándolo y siguiendo, en lugar de fallar.
- [x] 3.5 Tests con una tabla de pose sintética: candidato ausente, empate de confianzas, un frame
      sin ningún candidato válido. Corren localmente, sin GPU ni pesos.
- [x] 3.6 Exponerlo como una llamada de dos líneas cortas, al estilo de `reporte.confianzas()`.

## 4. Parametrizar el bodypart del hocico (notebook 02)

- [x] 4.1 Declarar `BODYPART_HOCICO` en la celda `parameters`, con `None` por defecto para conservar
      el comportamiento actual.
- [x] 4.2 Usar el parámetro cuando esté fijado y caer en la lista de candidatos cuando no lo esté.
- [x] 4.3 Dejar registrado en la salida de la corrida qué bodypart se usó.
- [x] 4.4 Exponerlo en `PARAMETROS["roi_normalizacion"]` de `00_pipeline.ipynb`.

## 5. Reinferencia con los videos nuevos

- [ ] 5.1 Sincronizar el código al Drive (`SINCRONIZAR_CODIGO = True`) y subir a mano
      `00_pipeline.ipynb`, que queda excluido del sync.
- [ ] 5.2 Poner `ETAPAS_A_FORZAR = ["preparar_dataset", "roi_normalizacion"]` y lanzar la corrida.
- [ ] 5.3 Mirar el primer latido para confirmar que el checkpoint avanza — la reanudación parcial
      nunca se ejercitó en una corrida real.
- [ ] 5.4 Anotar el nuevo total de frames y clips: los números de referencia del análisis
      (59,5 %, 359 frames, 83,9 %) son sobre 1472 y dejan de ser comparables.

## 6. Decidir el bodypart y cerrar

- [ ] 6.1 Correr el barrido sobre la pose recién persistida.
- [ ] 6.2 Decidir el bodypart con el criterio de la decisión 4 del diseño, y **documentar la decisión
      junto con los candidatos descartados y su cobertura**.
- [ ] 6.3 Re-derivar los recortes con el bodypart elegido. No requiere reinferir.
- [ ] 6.4 Verificar la cobertura contra el umbral de 70 % y contra el 59,5 % de partida. Si no
      alcanza, dejar documentado el resultado y el camino que queda (anotación manual acotada a
      `031`, `053`, `054`, `055` y `920`).
- [ ] 6.5 Actualizar `openspec/project.md` —cobertura medida y punto de partida de la próxima
      sesión— y el `README.md` si quedó desactualizado.
