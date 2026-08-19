## Context

La transformada de similitud lleva el ojo a `(0,5 · lado, FRAC_OJO · lado)` y el hocico a
`(0,5 · lado, FRAC_HOCICO · lado)`. Dos puntos determinan escala, rotación y traslación, así que
la separación entre esas dos fracciones fija por completo **cuánto del frame original entra en el
recorte**: exactamente `1 / (FRAC_HOCICO − FRAC_OJO)` veces la distancia ojo-hocico.

Ese acoplamiento no era evidente al elegir los valores. 0,34 y 0,74 parecen posiciones razonables
—ojo en el tercio superior, hocico en el inferior— y sólo al medir se ve que implican un recorte
de 2,5 veces la distancia, imposible sobre un encuadre cerrado.

## Goals / Non-Goals

**Goals:**

- Que el cuadrado canónico sea satisfacible con la geometría real del dataset.
- Que el relleno remanente no sea un rasgo aprendible.
- Que la propiedad quede medida y vigilada, no verificada a ojo.

**Non-Goals:**

- **No** se filtran recortes por calidad. Sería descartar datos para tapar un problema de
  parámetros.
- **No** se cambia `LADO_RECORTE = 224`: es la entrada estándar de los backbones preentrenados y
  no tiene nada que ver con el problema, que es de proporción y no de resolución.
- **No** se reinfiere.

## Decisions

### 1. Separación 0,60 y no un valor mayor

Para que el recorte entre completo en el percentil 95 haría falta una separación de 0,76 —ojo en
0,12, hocico en 0,88—, lo que dejaría la cara pegada a los bordes y sin ningún contexto. 0,60 hace
entrar hasta pasada la mediana y conserva un 20 % de margen arriba y abajo:

| pct | distancia | lado con 0,40 | lado con 0,60 | ¿entra en 1080? |
|---|---|---|---|---|
| p5 | 374 px | 936 | 624 | sí |
| p25 | 493 px | 1234 | 822 | sí |
| p50 | 582 px | 1454 | 969 | sí |
| p75 | 681 px | 1702 | 1134 | no, por poco |
| p95 | 824 px | 2060 | 1373 | no |

Es un compromiso explícito entre contexto y relleno, no un óptimo. Con la pose persistida, barrer
otros valores cuesta una re-derivación de 30-45 min, así que puede revisarse con el dato en mano.

### 2. `BORDER_REPLICATE` y no negro, ni recorte adaptativo

El negro es el peor relleno posible acá: alto contraste, borde nítido, y **cantidad correlacionada
con el ejemplar**. Replicar el borde inventa píxeles —lo cual es una objeción legítima— pero los
inventa sin filo y sin patrón que valga la pena aprender.

*Alternativas descartadas:* rellenar con el color medio del recorte, que sigue dejando un borde
recto y además introduce una estadística dependiente de la imagen; y ajustar la escala por frame
para que el recorte siempre entre, que rompería la normalización —el punto entero es que todos los
recortes tengan la misma escala—.

### 3. La calidad se mide desde la geometría, sin abrir imágenes

`calidad_recortes()` reconstruye la transformada inversa desde el par ojo/hocico y las dimensiones
del video, y muestrea el cuadrado canónico para contar qué fracción cae fuera del frame. No
necesita las imágenes, así que corre en segundos sobre el dataset entero y no depende de que los
recortes estén expandidos en disco.

Es deliberado que la métrica sea independiente del Paso 16. Aquel verifica que el par detectado
caiga en su posición canónica, y da error cero aunque el hocico esté sobre una oreja: mide la
transformada contra los mismos puntos con que se construyó. Ésta mide algo que la transformada no
puede garantizar.

## Risks / Trade-offs

- **Se pierde contexto facial** → menos ojo-a-oreja y menos mandíbula. Si EquiFACS exige unidades
  de acción en esas zonas, el recorte podría estar cortando señal relevante. Vale contrastarlo con
  el marco de referencia antes de fijar el valor.
- **`BORDER_REPLICATE` inventa píxeles** → en un recorte con 30 % de relleno, casi un tercio de la
  imagen es extrapolación. Es preferible al negro, pero no es información.
- **La estimación de mejora es indirecta** → sale de los percentiles medidos, no de una
  re-derivación. La comprobación es volver a correr `calidad_recortes()`.

## Migration Plan

1. Sincronizar el código y ejecutar `02_roi_y_normalizacion.ipynb` **suelto**, con
   `LIMPIAR_ETAPA = False`. Por el orquestador no se puede: vería la etapa completa y la
   saltearía, y forzarla borra el checkpoint.
2. Verificar con `calidad_recortes()` y con `cobertura()`.

Rollback: los dos parámetros vuelven a 0,34 y 0,74 y se re-deriva.

## Open Questions

- **¿Cuál es la separación óptima?** 0,60 es un compromiso razonado, no un resultado. Ahora que
  barrerla es barato, conviene medir 0,50, 0,60 y 0,70 y comparar.
- **¿Cuánto contexto necesita el modelo?** Depende de qué unidades faciales se quieran capturar, y
  eso se responde con EquiFACS, no con geometría.
