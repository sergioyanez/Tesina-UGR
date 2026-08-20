# Diseño — catálogo de videos por clase

## Contexto

La restricción real sigue siendo el tamaño del dataset. Con 41 videos de `Disgustado` contra 31,
los 10 que estaban invisibles no son un detalle de higiene: son un tercio más de la clase que
limita todo lo que venga después. Eso fija la prioridad del diseño — **que no se pierda material
en silencio, y que tampoco entre material que nadie etiquetó**.

## Decisión 1 — Declaración por categoría, no estructura común

`SUBCARPETAS_POR_CLASE` es un diccionario de categoría a tupla de subcarpetas, en
`equinos/catalogo.py`. `Disgustado` declara dos, `Interesado` y `Sereno` una.

**Alternativa descartada — una lista única de subcarpetas para todas las clases.** Es más corta y
es lo que la estructura del disco sugiere hoy, pero convierte una coincidencia en una regla. `Se
tapa una parte de la cara` existe hoy sólo bajo `Disgustado`; el día que aparezca bajo `Sereno`
—una carpeta con ese nombre puede aparecer en cualquier clase— entraría sola, sin que nadie haya
decidido que ese material es `Sereno`. Una etiqueta inventada por la forma del árbol de carpetas
es exactamente el problema que este change viene a sacar, no a mover de lugar.

**Alternativa descartada — un archivo de configuración (YAML/CSV) fuera del código.** Suena más
"configurable", pero agrega un artefacto que hay que sincronizar a Drive y que puede quedar
desfasado del código sin que nada lo indique. La composición del dataset cambia pocas veces y
cada cambio merece un commit citable en el informe: el control de versiones ya es el mecanismo.

**Alternativa descartada — recorrer todo recursivamente y listo.** Es lo que hacía la regla
anterior, y su costo está medido: 13 videos de una carpeta de descarga entrando como `Sereno` sin
aviso.

## Decisión 2 — El nombre de carpeta se compara por forma canónica

`clave()` normaliza a NFKD, descarta las marcas de acento, pasa a minúsculas y reemplaza todo lo
que no sea alfanumérico por espacio. Así `Seleccionar cabeza?` y `Seleccionar cabeza_` colapsan al
mismo valor.

No es cosmética. En Drive la carpeta lleva el signo de pregunta; una copia bajada como ZIP lo
sanea a guión bajo porque no es un carácter válido en todos los sistemas de archivos. Con
comparación literal, el mismo código incorpora 10 videos en Colab y 0 en la máquina de desarrollo
—o al revés— sin lanzar ningún error. Es la misma clase de divergencia silenciosa entre entornos
que ya costó una sesión con la orientación de video.

**Alternativa descartada — declarar las dos grafías por clase.** Duplica cada entrada, y falla el
día que aparezca una tercera variante.

El riesgo de la normalización es el opuesto: dos carpetas distintas que colapsen al mismo valor.
En este dataset la vecina de `Seleccionar cabeza?` es `Separar cabeza?`, que no colapsa. Queda
cubierto por una prueba.

## Decisión 3 — La clase es una columna del inventario, no el padre de la ruta

`df_videos` gana `clase` (la etiqueta, del catálogo) y conserva `carpeta` con su significado
literal (la carpeta real de la que salió, que ahora puede ser una subcarpeta).

Conservar `carpeta` en vez de reemplazarla mantiene la trazabilidad de la procedencia de cada
muestra sin volver a recorrer Drive, y es información que el informe necesita: "10 de los 41
videos de `Disgustado` vienen de una subcarpeta de trabajo" es una característica del dataset que
hay que poder afirmar.

Los consumidores del CSV en `equinos/reporte.py` no leen ninguna de las dos columnas, así que los
inventarios ya generados siguen siendo legibles.

## Decisión 4 — Una sola implementación de "dónde están los videos"

`raiz_videos()` resuelve en cascada —parámetro explícito, variable de entorno, carpeta compartida
en Colab, carpeta de la raíz de proyecto— y en cada paso **exige que la candidata contenga
carpetas de categoría**. El orquestador la llama, informa lo que encontró, y se la inyecta a la
etapa 01 por papermill.

Antes había dos búsquedas: una en `00_pipeline.ipynb`, escrita a mano, que localizaba la carpeta
compartida y sólo la imprimía; y la de la etapa 01, que usaba la carpeta de la raíz de proyecto.
El orquestador informaba una carpeta y la etapa inventariaba otra. Eso no falla: produce un
inventario correcto de la carpeta equivocada.

La búsqueda amplia bajo Mi unidad quedó acotada a tres niveles y se evalúa **última y de forma
perezosa**. La versión anterior hacía `rglob("*")` sobre Drive montado, que recorre la cuenta
entera archivo por archivo, y cada entrada es una llamada de red.

**Alternativa descartada — copiar los videos a la raíz de proyecto.** Duplicaría ~2 GB en la
cuota de Drive y crearía una segunda copia que puede quedar desfasada de la compartida, que es
donde el equipo agrega material.

## Decisión 5 — Verificar es una función aparte

`inventariar()` describe lo que hay; `verificar()` decide si es utilizable y corta con `SystemExit`
si no. Separarlas permite inspeccionar el catálogo sin que el intento aborte la sesión, y hace que
las dos condiciones de corte —catálogo vacío, nombres repetidos— se prueben sin montar un pipeline.

La detección de nombres repetidos existía y era hipotética. Con las subcarpetas sumando dejó de
serlo: la misma toma puede estar suelta en la raíz de la clase y también dentro de `Seleccionar
cabeza?`. El mensaje ahora muestra las rutas completas, no las clases, porque el caso frecuente ya
no es "dos clases" sino "dos carpetas de la misma clase".

## Riesgos

- **Un video etiquetado dos veces.** Si la misma toma está en dos clases, `verificar()` corta.
  Bien: la decisión de cuál es la etiqueta correcta es humana.
- **Duplicados con nombres distintos.** El mismo video subido dos veces con nombre diferente pasa
  el catálogo y entra dos veces. No lo resuelve este change; requeriría comparar contenido.
- **Las tasas históricas dejan de ser comparables.** El catálogo suma videos y saca otros, así que
  ningún porcentaje de cobertura anterior se puede contrastar directamente con los nuevos.
