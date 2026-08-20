## Why

Contribuye al **objetivo específico 1** (construcción del dataset), y de rebote al 3 y al 4: la
clase que gana material es la minoritaria, que es la que limita todo lo que venga después.

Se incorporaron videos nuevos a la carpeta compartida `Videos de prueba`, y una parte quedó dentro
de subcarpetas de trabajo en vez de sueltos en la carpeta de su clase. El inventario no los veía.

La regla anterior era **estructural**: la clase de un video era el nombre de su carpeta
contenedora, sobre un recorrido recursivo de toda la carpeta de videos. Se eligió así en el
18/08/2026 para que `Seleccionar cabeza?` y `Separar cabeza?` no se convirtieran en dos clases
fantasma, y para eso funcionó. Pero decide dos cosas a la vez —qué archivos entran y con qué
etiqueta— y las decide con el mismo dato, así que no permite que un video esté en una subcarpeta
**y** tenga etiqueta. Medido sobre el espejo local del dataset (20/08/2026):

| clase | regla anterior | con las subcarpetas declaradas | |
|---|---:|---:|---|
| Disgustado | 31 | **41** | +32 % |
| Interesado | 57 | 58 | +1 |
| Sereno | 34 | 34 | = |
| **total en carpetas de clase** | **122** | **133** | +11 |

Los 10 videos que ganaba `Disgustado` son la clase de la que menos hay, y su escasez es el motivo
por el que se llegó a considerar un plan B de 2 clases.

El mismo recorrido recursivo tenía un segundo agujero, en la dirección contraria. Cualquier
carpeta anidada llamada como una clase entraba entera, y una descarga parcial de Drive se llama
`Sereno-20260819T000125Z-1-001/Sereno`: sus 13 videos entraban al inventario como `Sereno`, sin
aviso y sin colisión de nombres que lo delatara. La regla no distinguía una carpeta de clase de
una carpeta que se llama igual que una clase.

## What Changes

- La composición del dataset se declara en **un solo lugar** del paquete `equinos`: por cada
  clase, qué subcarpetas suman además de los videos sueltos en su raíz. **La lista no es la misma
  en todas las clases** —`Disgustado` suma dos, `Interesado` y `Sereno` una— y la estructura del
  código no puede suponer que lo sea.
- La etiqueta de un video deja de deducirse del nombre de su carpeta contenedora. El inventario
  gana una columna `clase` que sale del catálogo, y `carpeta` pasa a significar lo que dice: la
  carpeta real de la que salió, que ahora puede ser una subcarpeta.
- Lo declarado entra; lo no declarado queda afuera **pero se informa** con nombre y conteo. Se
  informa también al revés: una subcarpeta declarada que no existe en el disco.
- Los nombres de carpeta se comparan por su forma canónica, sin mayúsculas, acentos ni signos.
  En Drive la carpeta es literalmente `Seleccionar cabeza?`; en una copia bajada como ZIP el signo
  queda saneado a `Seleccionar cabeza_`. Comparar literalmente hace andar el código en un entorno
  y fallar callado en el otro.
- La carpeta que contiene las tres clases se **resuelve y se verifica**, en lugar de suponerse.
  El orquestador ya la localizaba pero no se la pasaba a la etapa, así que informaba una carpeta
  y la etapa inventariaba otra. Ahora la localiza el paquete —una sola implementación— y la
  inyecta como parámetro.
- La detección de nombres de archivo repetidos se extiende al caso nuevo y ya no hipotético: la
  misma toma suelta en la raíz de la clase y también dentro de una subcarpeta declarada.

**No incluido:** decidir si `Separar cabeza?` (en `Sereno`) entra al dataset. Es material de
trabajo sin decisión de etiqueta tomada; queda documentado como excluido y basta agregar una
línea para incorporarlo.

## Capabilities

### New Capabilities

Ninguna.

### Modified Capabilities

- `persistencia-artefactos`: el requisito "Lectura de los videos sin descarga repetida" supone
  hoy que la carpeta de videos es la de la raíz de proyecto y que los archivos esperados están
  en ella. Se extiende a resolver dónde está esa carpeta, y se agrega el requisito de que la
  pertenencia de cada video a una clase sea una **declaración explícita por categoría** y no una
  consecuencia de dónde quedó guardado el archivo.

## Impact

- **Paquete `equinos`:** módulo nuevo `catalogo.py` y su registro en `__init__.py`.
- **Pruebas:** `tests/test_catalogo.py`, que corre sin GPU, sin internet y sin pesos.
- **Notebooks:** `01_preparar_dataset_equinos.ipynb` (Pasos 2, 3, 4, 5, 6 y la segmentación) y
  `00_pipeline.ipynb` (Paso 0, que pierde ~60 líneas de búsqueda hecha a mano, y Paso 1, que pasa
  la carpeta a la etapa).
- **Cambia `inventario_videos.csv`:** columna `clase` nueva; `carpeta` cambia de significado. Los
  CSV anteriores siguen siendo legibles —ninguna función de `reporte.py` lee esas columnas— pero
  no son comparables.
- **Requiere re-ejecutar el pipeline** para que los videos nuevos lleguen a clips y frames. No
  invalida lo ya generado: lo ya procesado sigue siendo válido y el checkpoint por video hace que
  sólo se procesen los que faltan.
- **Cambian los denominadores de todos los reportes.** Las tasas de cobertura anteriores se
  midieron sobre 146 videos; el catálogo nuevo suma unos y saca otros, así que los porcentajes
  históricos dejan de ser comparables aunque el número se parezca.
