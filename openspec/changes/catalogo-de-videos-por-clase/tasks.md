# Tareas — catálogo de videos por clase

## 1. Módulo de catálogo en el paquete `equinos`

- [x] 1.1 Crear `equinos/catalogo.py` con `SUBCARPETAS_POR_CLASE`, la declaración por categoría
      de qué subcarpetas suman a cada clase.
- [x] 1.2 Implementar `clave()`, la forma canónica de un nombre de carpeta, para que la misma
      carpeta se reconozca con la grafía de Drive y con la saneada de una copia ZIP.
- [x] 1.3 Implementar `raiz_videos()`, que resuelve en cascada dónde está la carpeta que contiene
      las tres clases y exige que la candidata tenga carpetas de categoría adentro.
- [x] 1.4 Implementar `inventariar()`, que devuelve el catálogo con su detalle de procedencia:
      aportes por carpeta, subcarpetas declaradas ausentes, videos excluidos y clases sin carpeta.
- [x] 1.5 Implementar `catalogar()`, el diccionario `{clase: [rutas]}` que consumen las etapas.
- [x] 1.6 Implementar `informe()` y `verificar()`, separando describir de cortar.
- [x] 1.7 Registrar el módulo en `equinos/__init__.py`.

## 2. Pruebas

- [x] 2.1 Escribir `tests/test_catalogo.py` sobre un árbol de archivos falso, sin GPU ni internet.
- [x] 2.2 Cubrir: grafías equivalentes, subcarpetas declaradas por clase, subcarpeta declarada en
      una clase y no en otra, exclusión informada, clase sin carpeta, carpeta anidada homónima de
      una clase, extensiones, y los dos cortes de `verificar()`.
- [x] 2.3 Ejecutar `python3 tests/test_catalogo.py` — 22 comprobaciones, todas pasan.
- [x] 2.4 Ejecutar el catálogo contra el espejo local del dataset y contrastar contra la regla
      anterior: 122 -> 133 videos, `Disgustado` 31 -> 41.

## 3. Notebook 01 — inventario

- [x] 3.1 Agregar `CARPETA_VIDEOS` a la celda `parameters` y `catalogo` al import del arranque.
- [x] 3.2 Paso 2: localizar la carpeta con `catalogo.raiz_videos(CARPETA_VIDEOS)`.
- [x] 3.3 Paso 3: mostrar la estructura sólo de las carpetas de clase, no de la base entera.
- [x] 3.4 Paso 4: reemplazar el filtrado por nombre de carpeta contenedora por el catálogo,
      su informe y su verificación; exponer `videos`, `videos_por_clase` y `clase_de`.
- [x] 3.5 Paso 5: `df_videos` gana la columna `clase` y `carpeta` pasa a ser la carpeta real.
- [x] 3.6 Paso 6: conteo por clase y desglose por carpeta de procedencia.
- [x] 3.7 Actualizar los consumidores de la etiqueta: la segmentación en clips y la comparación
      de videos contra clips.

## 4. Notebook 00 — orquestador

- [x] 4.1 Eliminar la búsqueda de la carpeta compartida hecha a mano, incluido el `rglob` sobre
      Mi unidad entera.
- [x] 4.2 Localizar la carpeta con `catalogo.raiz_videos()` después de importar el paquete, e
      imprimir el catálogo como parte del arranque.
- [x] 4.3 Inyectar `CARPETA_VIDEOS` en los parámetros de la etapa `preparar_dataset`.

## 5. Documentación

- [x] 5.1 Actualizar `openspec/project.md`: composición del dataset, carpeta de origen y la
      convención de la columna `clase`.
- [x] 5.2 Actualizar `README.md` con la estructura de carpetas que el pipeline espera.

## 6. Verificación en Colab (pendiente de una sesión con Drive montado)

- [ ] 6.1 Correr el Paso 0 de `00_pipeline.ipynb` y comprobar que localiza `Videos de prueba` y
      que el catálogo incluye `Se tapa una parte de la cara` bajo `Disgustado`.
- [ ] 6.2 Comprobar que el informe no reporta videos afuera que debieran entrar.
- [ ] 6.3 Re-ejecutar la etapa `preparar_dataset` con `LIMPIAR_ETAPA = False` y verificar que el
      checkpoint procesa sólo los videos nuevos.
- [ ] 6.4 Registrar el conteo real por clase, que es el que va al informe final.
