# Tareas — encuadre que incluye las orejas

## 1. Barrido y elección del encuadre (sin escribir imágenes)

- [x] 1.1 Llevar `medicion_encuadre.py` a `equinos/encuadre.py` como función reutilizable:
      proyecta los keypoints de oreja de `pose_completa.csv` por la transformada de similitud de un
      campo de visión dado y devuelve inclusión de orejas y distribución de relleno.
- [x] 1.2 Escribir las pruebas de `equinos/encuadre.py` en `tests/test_encuadre.py`: proyección de
      un punto conocido, campo de visión que contiene y que no contiene un keypoint, y fracción de
      relleno de un rectángulo enteramente dentro y enteramente fuera del cuadro.
- [x] 1.3 Correr el barrido sobre los candidatos del design más el encuadre vigente como línea de
      base, y dejar la tabla en `medicion.md`. Se agregaron D, E, F y G al ver que C no alcanzaba
      en la métrica del vector de oreja.
- [x] 1.4 Inspeccionar visualmente ~20 recortes de `Disgustado` bajo C y bajo A, generados en
      memoria sin sobrescribir el dataset, y registrar la observación en `medicion.md`.
- [x] 1.5 **Decidir el encuadre** contra el criterio declarado en la spec y dejar escrito cuál se
      adopta y por qué. **Adoptado D (1,00 / 0,30 / ±0,75, destino 224×343)**: vector de oreja
      4,3 % → 77,2 % empatando exacto el relleno mediano de la línea de base. El criterio se
      re-ancló en el vector base→punta y su historia quedó registrada en la spec.

## 2. Parametrización del notebook 02

- [x] 2.1 Reemplazar `FRAC_OJO` / `FRAC_HOCICO` en la celda de parámetros por el campo de visión
      declarado —`AIRE_SOBRE_OJO`, `AIRE_BAJO_HOCICO`, `SEMIANCHO_LATERAL`, en unidades de `d`— más
      `ANCHO_RECORTE`, con el comentario que explica la asimetría medida del déficit y por qué la
      parametrización anterior no la podía expresar. **Se declara sólo el ancho**: el alto se
      deriva del aspecto, porque declarar los dos permitiría deformar la cara sin querer.
- [x] 2.2 Reescribir `destino_canonico()`. **Se movió a `equinos.encuadre.destino()` en vez de
      reescribirla en el notebook**: la usan el notebook, el barrido y la validación de la
      interpolación, y había tres copias que podían divergir. `equinos/interpolacion.py` dejó de
      extraerla del notebook y toma la geometría del módulo.
- [x] 2.3 Actualizar el paso de recorte y alineación para usar el destino rectangular en
      `cv2.warpAffine`, conservando `BORDER_REPLICATE`.
- [x] 2.4 Calcular la fracción de relleno de cada frame en el mismo paso y agregarla al registro
      como `relleno_fraccion`.
- [x] 2.5 Verificar que el camino de degradación por caja siga produciendo recortes de la misma
      forma que el camino alineado, para que el dataset no mezcle dos geometrías. `roi_desde_keypoints`
      acepta `aspecto` (1.0 por defecto, que es el comportamiento histórico) y el camino de caja le
      pasa el del recorte canónico. Verificado sobre un frame real: los dos caminos dan 224×343.

## 3. Pruebas de geometría

- [x] 3.1 Actualizar en `tests/test_geometria.py` las pruebas que fijan el recorte cuadrado: pasan
      a fijar la **isotropía de la escala** de la transformada y la relación de aspecto declarada.
- [x] 3.2 Agregar una prueba de que un destino rectangular no deforma: la razón entre dos
      distancias del frame de origen se conserva en el recorte.
- [x] 3.3 Agregar una prueba de que la fracción de relleno es 0 cuando el rectángulo canónico entra
      entero en el cuadro, y positiva cuando se sale. Vive en `tests/test_encuadre.py`, que es donde
      quedó la función.
- [x] 3.4 Correr la suite completa sin GPU ni internet, y verificar que las 21 pruebas de geometría
      preexistentes siguen pasando o quedan explícitamente reemplazadas. **264 pruebas en verde.**
      Las de `destino_canonico` se reemplazaron por las de `test_encuadre`; la de «la ROI es
      cuadrada» pasó a fijar el aspecto declarado y la isotropía de la escala.

## 4. Re-ejecución de la etapa 02

- [x] 4.1 Verificar antes de correr que `pose_completa.csv`, `keypoints_faciales.csv` y
      `keypoints_manuales.csv` están en su lugar, y que el paso de inferencia queda salteado por
      checkpoint. **No se reinfiere DeepLabCut.**
- [x] 4.2 Re-ejecutar sólo el warp y la escritura de los frames alineados, regenerando los 5.888
      JPEG y `dataset_frames_alineados.csv` con la columna nueva.
- [x] 4.3 Verificar que la cantidad de frames alineados, los 141 clips íntegros, los 73 ejemplares
      y el reparto 21/56/64 **no cambiaron**. Si cambiaron, parar: significa que el encuadre afectó
      algo que no debía.
- [x] 4.4 Regenerar la caché `frames_alineados.zip` para que la corrida en Colab consuma la
      geometría nueva y no la vieja.

## 5. Reportes y validación

- [x] 5.1 Extender `equinos/reporte.py` con la inclusión de orejas y la distribución de
      `relleno_fraccion`, y actualizar `tests/test_persistencia.py` en consecuencia.
- [x] 5.2 Regenerar `reporte_cobertura.csv` e informar cobertura de alineación, inclusión de orejas
      y relleno juntos, cada uno con su denominador y sus sesgos declarados.
- [x] 5.3 Re-correr `equinos.interpolacion.informe_validacion()`. `MAX_EXTREMO_SOSTENIDO = 0` se
      confirma. `MAX_HUECO_INTERPOLADO` **no se toca**: el informe ahora recomienda 2, pero la
      comparación mezcla el cambio de encuadre con el crecimiento del dataset (3.900 → 4.046 frames
      detectados), el margen es de 0,30 px sobre una mediana que antes fallaba por 0,05, y subirlo
      cambiaría la composición del dataset, que está fuera del alcance. Registrado en `medicion.md`
      §8 como change aparte.

## 6. Documentación

- [x] 6.1 Actualizar `openspec/project.md`: composición del dataset con la geometría nueva, el
      encuadre adoptado y su justificación, y la inclusión de orejas como criterio de la etapa 02.
- [x] 6.2 Actualizar `README.md` con la forma del recorte y la columna nueva del dataset.
- [x] 6.3 Actualizar el `design.md` de `clasificacion-orientada-a-video`, que asume entradas de
      224×224, a la forma resultante.
- [x] 6.4 Dejar en `medicion.md` la evidencia completa —tabla del barrido, sesgos, inspección
      visual y decisión— como respaldo citable en la memoria de la tesina.
- [x] 6.5 Registrar en `openspec/project.md` el crédito de la observación a Héctor R. Liceaga y la
      medición que la confirmó, incluido que la corrección propuesta en la rama `hector`
      (`FRAC_OJO = 0,25`) resultó insuficiente y por qué.

## 7. Cierre

- [x] 7.1 `openspec validate encuadre-que-incluye-las-orejas` sin errores.
- [x] 7.2 Commits separados por unidad lógica: barrido y decisión; parametrización del notebook;
      pruebas; re-ejecución y artefactos; reportes; documentación.
- [x] 7.3 Sincronizar las specs delta a `openspec/specs/preprocesamiento-facial/spec.md` y archivar
      el change.
