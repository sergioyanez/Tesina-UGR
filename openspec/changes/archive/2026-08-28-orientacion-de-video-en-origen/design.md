## Context

El pipeline lee video en cuatro lugares del notebook 01 —inventario, vista previa, segmentación en
clips y extracción de frames— y en dos del notebook 02, sobre los clips ya generados. Todos usaban
`cv2.VideoCapture` directo.

Medido en Colab con OpenCV 4.11, sobre un video con rotación declarada de 90°:

```
auto OFF   propiedades 1920x1080   frame entregado 1920x1080   (acostado)
auto ON    propiedades 1080x1920   frame entregado 1080x1920   (derecho)
```

El comportamiento **no es uniforme entre versiones**: en OpenCV 4.10 local la rotación se aplica
sola, con la bandera apagada o encendida. Esa diferencia es la razón por la que la corrección se
midió antes de escribirse.

## Goals / Non-Goals

**Goals:**

- Que ningún frame llegue acostado al detector ni al modelo de pose.
- Que un problema de orientación se manifieste al principio del pipeline y no como un faltante
  al final.
- Que la corrección sea independiente de la versión de OpenCV.

**Non-Goals:**

- **No** se rota nada a mano. La medición mostró que la bandera alcanza en el entorno real.
- **No** se tocan los notebooks 02 en su lectura de clips: los clips los escribe OpenCV y no
  llevan metadatos de rotación, así que la bandera sería un no-op.
- **No** se intenta recuperar los clips ya generados. Tienen los píxeles acostados sin ninguna
  marca que lo indique; sólo se arreglan regenerándolos.

## Decisions

### 1. Un único punto de lectura, `abrir_video()`

Todas las aperturas de video pasan por una función que activa `CAP_PROP_ORIENTATION_AUTO`. El
valor no está en ahorrar una línea sino en que la regla sea verificable: se puede comprobar que no
queda ningún `cv2.VideoCapture` suelto.

*Alternativa descartada:* activar la bandera sólo donde se leen videos de origen. Es lo mínimo
suficiente hoy, pero deja la regla como conocimiento tácito —"acá sí, allá no"— y el próximo
`VideoCapture` que alguien escriba no la tendrá.

### 2. Rotar con la bandera y no a mano

`cv2.rotate()` según el metadato funcionaría en cualquier versión, pero **rotar un frame que
OpenCV ya rotó deja el dataset entero al revés**, sin ningún síntoma hasta que los resultados no
cierran. La bandera es idempotente por construcción: pide el frame en su orientación de
visualización, sin importar qué haga la versión por debajo.

Por eso `prueba_orientacion()` se escribió **antes** que la corrección, y por eso queda en el
paquete: si el comportamiento cambia con una versión futura de OpenCV, el veredicto lo detecta.

### 3. El `VideoWriter` se dimensiona con el primer frame real

Antes tomaba `(ancho, alto)` del inventario. Con la orientación aplicada esas dimensiones cambian,
y cualquier desajuste entre el tamaño declarado y el del frame produce un clip vacío o corrupto.
Usar la forma del frame que efectivamente se va a escribir elimina la clase entera de fallos, y de
paso desacopla la segmentación del inventario. Si difieren, se informa.

### 4. La vista previa usa el mismo punto de lectura

Es la única señal visual del pipeline y estaba mirando por otra ventana. Con `abrir_video()`, un
frame acostado se ve en el primer minuto del notebook 01.

## Risks / Trade-offs

- **Los ejemplares afectados cambian de peso en el dataset** → `031`, `053`, `054`, `055`, `155`,
  `920` y `037` aportaban muy pocos frames y van a aportar muchos más. Es lo correcto, pero
  significa que la composición cambia y las métricas previas no son comparables. Hay que
  reportarlo como tal, no como una mejora del modelo.
- **La corrección se validó sobre un video sintético**, no sobre uno de los 13 reales. La
  verificación real es la corrida siguiente: la pérdida en videos rotados tiene que caer a
  aproximadamente cero.
- **Podría haber videos con rotación 180°** o con metadatos mal escritos, que la bandera trate de
  otro modo. `orientacion_videos()` los expone si aparecen.

## Migration Plan

1. Re-ejecutar `preparar_dataset` y `roi_normalizacion` forzando ambas. No hay camino incremental:
   los clips viejos están acostados.
2. Verificar con `reporte.frames_perdidos()` que la pérdida en videos rotados haya desaparecido.
3. Verificar con `reporte.cobertura()` la tasa de alineación contra el umbral de 70 %.

Rollback: revertir el commit y re-ejecutar. No hay estado migrado que deshacer.

## Open Questions

- **¿Cuánto sube realmente la cobertura?** Los 400 frames perdidos son el piso; el efecto sobre los
  320 frames rotados que ya pasaban es desconocido y podría ser mayor.
- **¿Quedan errores de etiquetado como el de `501_8v1`?** Ese se detectó porque dos clases tenían
  el mismo nombre de archivo. Dos copias con nombres distintos no serían detectadas por esa vía.
