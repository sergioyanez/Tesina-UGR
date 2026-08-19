## 1. Diagnóstico (cerrado)

- [x] 1.1 Medir relleno y escala de los 2286 recortes alineados con `reporte.calidad_recortes()`.
      Resultado: 2,5 % sin relleno, 61,1 % con más del 30 %, 20,2 % mayormente vacíos.
- [x] 1.2 Descartar que los keypoints estuvieran sobre anatomía equivocada. La distancia
      ojo-hocico va de 0,170 a 0,374 de la diagonal entre p5 y p95: rango sano.
- [x] 1.3 Verificar que el relleno correlaciona con el ejemplar y no es ruido. Confirmado: ocho
      ejemplares con 100 % de sospechosos contra nueve con 0 %.
- [x] 1.4 Verificar la transformada inversa del reporte punto por punto contra el par conocido.
      La primera versión tenía la rotación traspuesta y erraba 240 px.

## 2. Corrección (implementada)

- [x] 2.1 Separación canónica de 0,40 a 0,60 (`FRAC_OJO = 0.20`, `FRAC_HOCICO = 0.80`), con la
      aritmética que la justifica escrita junto al parámetro.
- [x] 2.2 Valores por defecto de `destino_canonico` acompañando, para que los tests de geometría
      —que extraen la función del propio notebook— prueben lo que se ejecuta.
- [x] 2.3 Relleno por `BORDER_REPLICATE` en el recorte alineado.
- [x] 2.4 Alinear los valores por defecto de `reporte.calidad_recortes()` con los del notebook.

## 3. Verificación sobre datos reales

- [ ] 3.1 Sincronizar y ejecutar `02_roi_y_normalizacion.ipynb` **suelto**, con
      `LIMPIAR_ETAPA = False`. Confirmar que el bucle informe `Clips pendientes: 0 de 210`, que es
      la señal de que la pose persistida evita la reinferencia.
- [ ] 3.2 Volver a medir con `calidad_recortes()`. Predicción: los recortes sin relleno pasan del
      2,5 % a entre 40 % y 55 %, y los mayormente vacíos desaparecen casi del todo.
- [ ] 3.3 Confirmar con `cobertura()` que la tasa de alineación **no** se movió. El cambio afecta
      cómo se recorta, no qué frames se alinean; un movimiento grande indicaría algo mal
      entendido.
- [ ] 3.4 Mirar la grilla del Paso 15 y contrastarla con lo observado antes del cambio.

## 4. Pendiente de decidir

- [ ] 4.1 Barrer la separación canónica (0,50 / 0,60 / 0,70) ahora que cuesta 30-45 min y no dos
      horas, y elegir con el dato en mano en vez de con el compromiso razonado.
- [ ] 4.2 Contrastar cuánto contexto facial hace falta contra EquiFACS, que es el marco que define
      qué unidades de acción hay que poder ver.
