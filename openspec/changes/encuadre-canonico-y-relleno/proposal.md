## Why

Contribuye al **objetivo específico 2**. Surgió al verificar la corrección de orientación: los
recortes alineados eran geométricamente exactos y aun así mayormente inservibles.

`reporte.calidad_recortes()` sobre los 2286 recortes alineados (19/08/2026):

| relleno del recorte | recortes |
|---|---|
| ninguno | 58 (**2,5 %**) |
| más del 30 % | 1397 (61,1 %) |
| más del 50 % | 462 (20,2 %) |

La causa es aritmética, no un error de implementación. El recorte abarca
`1 / (FRAC_HOCICO - FRAC_OJO)` veces la distancia ojo-hocico. Con 0,34 y 0,74 esa separación era
0,40, o sea **2,5 veces** la distancia. Medida sobre el dataset, la distancia ojo-hocico es el
26 % de la diagonal en la mediana: sobre un cuadro de 1080x1920 son ~582 px, y el cuadrado
canónico pedía **1454 px de lado en un cuadro de 1080 de ancho**. Era imposible de satisfacer.

Concuerda con lo ya observado el 13/08: el encuadre de los videos es cerrado y la cabeza ocupa
buena parte del cuadro. El contexto que los parámetros pedían alrededor de la cara sencillamente
no existe en el original.

**Y el relleno no es ruido: es un confounder.** Su cantidad depende de a qué distancia se filmó
cada caballo, así que correlaciona con el ejemplar, que a su vez predice la clase. Medido: `031`,
`055`, `121`, `168`, `175`, `057`, `907` y `919` con 100 % de recortes sospechosos, contra 0 % de
`123`, `167`, `169`, `508`, `511`, `515`, `108`, `170` y `913`. Un borde negro de alto contraste
es precisamente el atajo que una CNN aprende antes que la expresión — el mismo problema del fondo,
entrando por una tercera puerta.

## What Changes

- La separación canónica pasa de 0,40 a **0,60** (`FRAC_OJO = 0.20`, `FRAC_HOCICO = 0.80`). El
  recorte abarca 1,67 veces la distancia ojo-hocico en lugar de 2,5.
- El relleno pasa de negro a **`BORDER_REPLICATE`**.
- `reporte.calidad_recortes()` mide relleno y escala sin abrir una sola imagen, para que la
  propiedad quede vigilada y no haya que volver a descubrirla mirando una grilla.

**Descartado por ahora:** filtrar los recortes con relleno excesivo. Ataca el síntoma y vuelve a
recortar un dataset cuyo límite real es el tamaño muestral. Queda como último recurso si la
corrección no alcanza.

## Capabilities

### New Capabilities

Ninguna.

### Modified Capabilities

- `preprocesamiento-facial`: el requisito «Región de interés cuadrada» no dice nada sobre qué debe
  contener el recorte ni sobre qué hacer cuando el cuadrado canónico excede el frame. Se agrega
  que el encuadre debe ser satisfacible con la geometría real del dataset y que el relleno no debe
  introducir un artefacto de alto contraste.

## Impact

- **Notebooks:** `02_roi_y_normalizacion.ipynb` (parámetros, `destino_canonico`, recorte).
- **Paquete `equinos`:** `reporte.calidad_recortes()`.
- **Requiere re-derivar los recortes, no reinferir.** La pose está persistida, así que alcanza con
  ejecutar el notebook 02 suelto con el checkpoint intacto: ~30-45 min contra ~2 h.
- **No debería mover la cobertura de alineación:** cambia cómo se recorta, no qué frames se
  alinean. Si se mueve mucho, hay algo mal entendido.
