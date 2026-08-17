# Diseño técnico: persistencia de artefactos y orquestación en Colab

## Contexto y restricción dominante

La restricción que manda no es el dataset (34 videos) sino **el modelo de ejecución de Colab**: disco
efímero, sesión con límite de tiempo y desconexión por inactividad. Cualquier etapa que supere la
media hora tiene probabilidad no despreciable de no llegar al final, y el entrenamiento del objetivo
3 será más larga que el preprocesamiento actual.

De ahí el criterio que ordena todo el diseño: **ninguna etapa debe poder perder más que una fracción
menor de su trabajo ante un corte**. La reanudación no es una comodidad, es lo que hace que el
pipeline sea ejecutable.

## Disposición del almacenamiento

Raíz única en Drive, con una carpeta por tipo de artefacto:

```
MyDrive/tesina_equinos/            <- RAIZ (configurable en un solo lugar)
├── videos/<clase>/*.mp4           datos crudos, ya existentes
├── clips/<clase>/*.mp4            92 clips, ~2.7 GB   [persistente]
├── resultados/*.csv               inventarios y datasets  [persistente, liviano]
├── cache/<etapa>.zip              frames empaquetados  [persistente]
├── cache/<etapa>.json             manifiesto de la caché
├── checkpoints/<etapa>.json       estado de reanudación
├── ejecuciones/<etapa>_<ts>.ipynb copias ejecutadas por papermill
├── modelos/                       pesos entrenados (objetivo 3)
└── codigo/equinos/                copia del paquete, para el bootstrap
```

**Qué va a Drive y qué se queda en `/content`.** El criterio es el tamaño *unitario*, no el total.
Drive montado (FUSE) tiene una latencia por archivo que domina cuando hay miles de archivos chicos:
escribir 1472 JPEG sueltos tarda más que escribir los 2.7 GB de clips. Entonces:

| Artefacto | Cantidad | Dónde vive | Por qué |
|---|---:|---|---|
| CSV de inventario | 3–5 archivos | Drive directo | Chicos y pocos; el acceso directo simplifica |
| Clips `.mp4` | 92 | Drive directo | Pocos y grandes: FUSE rinde bien |
| Frames extraídos | ~1472 | `/content` + zip en Drive | Muchos y chicos: FUSE colapsa |
| Frames alineados | ~1472 | `/content` + zip en Drive | Ídem |
| Checkpoints | 1 por etapa | Drive directo | Deben sobrevivir al corte, por definición |

## Paquete `equinos/`

Un paquete Python en el repo, versionado, que concentra lo que hoy está duplicado en las celdas de
configuración de cada notebook.

```
equinos/
├── __init__.py
├── rutas.py        resolución de la raíz, montaje de Drive, subcarpetas
├── checkpoint.py   estado reanudable con validación por configuración
├── cache.py        empaquetado y expansión de artefactos numerosos
└── etapas.py       registro de etapas y su estado (pendiente/parcial/completa)
```

**`rutas.py` — detección de entorno.** Decide sin que el notebook tenga que saber dónde corre:
intenta importar `google.colab`; si existe, monta Drive y resuelve contra `MyDrive/tesina_equinos`;
si no, resuelve contra un directorio local. Esto es lo que permite probar la geometría en la máquina
de Sergio —sin GPU ni pesos— y ejecutar el pipeline real en Colab, con el mismo código.

Expone además `resolver(ruta_relativa)`, que reconstruye rutas absolutas a partir de las **rutas
relativas** guardadas en los CSV.

**Decisión: los CSV guardan rutas relativas a la raíz, no absolutas.** Hoy `inventario_clips.csv`
guarda `/content/clips_equinos/...`, que en otra sesión no apunta a nada — es la causa directa del
"no los encontraba". Guardando `clips/Sereno/xxx_clip_01.mp4` y resolviendo contra la raíz vigente,
el mismo CSV sirve en Colab, en local y en cualquier cuenta de Drive. El costo es que las etapas
consumidoras deben llamar a `resolver()` en lugar de usar la columna cruda; se acepta.

## Checkpointing

El checkpoint es un JSON por etapa:

```json
{
  "etapa": "roi_normalizacion",
  "hash_config": "a3f1c8…",
  "config": { "CONF_MINIMA_DETECCION": 0.5, "MARGEN_ROI": 0.45, "…": "…" },
  "unidades_completas": ["Sereno/013_..._clip_01", "…"],
  "actualizado": "2026-08-16T14:22:31"
}
```

Y junto a él, un CSV parcial al que la etapa **anexa** las filas de cada clip terminado.

**Granularidad: el clip.** Es la unidad natural (el 02 ya itera por clip, ~45 s cada uno) y acota la
pérdida ante un corte a menos de un minuto de trabajo. Bajar a frame multiplicaría por 16 las
escrituras a Drive sin beneficio real.

**Escritura segura.** El JSON se escribe a un temporal y se renombra sobre el definitivo. Un corte
en mitad de la escritura deja el checkpoint anterior intacto en lugar de un archivo truncado, que
sería peor que no tener checkpoint.

**Validación por `hash_config`.** Es el mecanismo que responde al riesgo principal de la propuesta.
El hash se calcula sobre el diccionario de parámetros **que afectan el resultado de esa etapa**, no
sobre toda la configuración: así cambiar el tamaño de una figura de inspección visual no invalida
una hora de cómputo. Cada etapa declara explícitamente qué claves la afectan — es una lista corta y
revisable, y el error de omitir una se paga caro, así que queda documentada en el código junto a los
parámetros.

Si el hash difiere, el checkpoint se descarta **entero**, se informa qué claves cambiaron y la etapa
reprocesa desde cero. No se intenta reutilizar parcialmente: mezclar configuraciones es exactamente
el fallo silencioso que queremos impedir.

## Caché comprimida

`cache.empaquetar(dir_local, nombre_etapa)` genera `cache/<etapa>.zip` más un manifiesto con la
cantidad de archivos y el `hash_config`. `cache.expandir(nombre_etapa, dir_local)` lo expande y
**verifica que la cantidad coincida**; si no, trata la caché como inválida.

Sin compresión (`ZIP_STORED`): los JPEG ya están comprimidos, así que deflate gastaría CPU para no
ganar espacio.

La verificación es por conteo, no por hash de contenido: hashear 1472 archivos en cada arranque
costaría más que el beneficio. El conteo detecta el caso real que importa —un zip truncado por un
corte durante la subida—, que es el único modo de corrupción plausible acá.

## Orquestador `00_pipeline.ipynb`

Ejecuta con **papermill**, un notebook por etapa, cada uno en su propio kernel.

Por qué papermill y no las alternativas: `%run` no ejecuta `.ipynb`; `nbconvert --execute` no
inyecta parámetros; compartir kernel entre etapas reintroduce el acoplamiento por estado en memoria
que este cambio busca eliminar. Papermill resuelve los tres puntos y además deja la copia ejecutada,
que es requisito metodológico.

**Inyección de parámetros.** Cada notebook tiene una celda etiquetada `parameters` con los valores
por defecto. Papermill inserta debajo una celda con los valores del orquestador. Así el notebook
sigue siendo ejecutable a mano —toma sus defaults— y el orquestador puede barrer umbrales sin editar
celdas, que es lo que pide la spec.

**Salteo de etapas.** Cada etapa declara en `etapas.py` un predicado sobre sus artefactos de salida.
El orquestador lo consulta antes de ejecutar. El mismo predicado alimenta el diagnóstico de estado.

**Fallo.** Se corta la cadena. La copia ejecutada de una etapa fallida se guarda con sufijo
`_FALLIDA` para no pisar la última exitosa.

## Bootstrap del código en Colab

Problema: el repo `sergioyanez/Tesina-UGR` es **privado**, así que un `git clone` sin autenticar
falla. Y el paquete `equinos/` tiene que estar disponible antes de poder importar nada.

La celda de arranque es por lo tanto **autocontenida** —no importa `equinos`— y resuelve en cascada:

1. Monta Drive.
2. Si `RAIZ/codigo/equinos` existe, lo agrega a `sys.path`. Listo.
3. Si no, y hay un token de GitHub en los *secrets* de Colab (`userdata.get("GITHUB_TOKEN")`),
   clona el repo a `RAIZ/codigo`.
4. Si no, falla con instrucciones concretas para copiar la carpeta a Drive.

Los secrets de Colab son el lugar correcto para el token: quedan fuera del notebook, así que no se
filtran al versionarlo ni al compartirlo con el tribunal.

Se descartó hacer público el repo para simplificar el clonado: es una decisión del equipo sobre un
trabajo todavía inédito, no una decisión de infraestructura.

## Qué NO cambia

Ninguna decisión metodológica. Segmentación en clips de 5–7 s, 16 frames uniformes por clip,
geometría de la ROI, umbrales del preprocesamiento facial, columnas de los CSV y `random_state=42`
quedan idénticos. Este cambio mueve archivos de lugar y agrega reanudación; si alterara un resultado
numérico, sería un defecto, no una mejora.

## Verificación

Localmente se puede validar **todo lo que no requiere GPU ni pesos**: resolución de rutas, detección
de entorno, ciclo completo de checkpoint (incluida la invalidación por cambio de configuración),
empaquetado y expansión de la caché con verificación de conteo, y el predicado de estado de cada
etapa. Son precisamente los lugares donde se esconde el fallo silencioso, así que van con pruebas en
`tests/`, junto a las 21 de geometría ya existentes.

Lo que sólo se puede verificar en Colab: el montaje de Drive y la corrida real de papermill.
