# Búsqueda de conflictos de etiqueta análogos al `501` — 28/08/2026

Cierra la tarea 4.2. El `501_8v1_LY1ucwR9.avi` era **el mismo archivo, con el mismo nombre**, en
`Disgustado` y en `Interesado`. La pregunta que quedaba abierta era si el criterio con que se
armaron las carpetas dejó casos equivalentes que **no comparten nombre** y por eso nadie vio.

## Qué se buscó, en tres pasadas

**1. Duplicados byte a byte.** Tamaño exacto sobre los 249 videos, y md5 de los que compartieran
tamaño. **Cero.** Ni siquiera hay dos archivos del mismo tamaño.

**2. Mismo ejemplar con la misma duración y resolución en clases distintas.** Ninguno. Las 19
coincidencias de duración exacta que aparecen son entre ejemplares distintos y en duraciones
redondas (2300 ms, 4600 ms): casualidad, no duplicación.

**3. Marcas de tiempo repetidas en el nombre.** Tres casos, verificados comparando el contenido
visual —12 frames por video reducidos a 16×16 en gris, diferencia absoluta media—:

| caso | duraciones | diferencia mínima | veredicto |
|---|---|---:|---|
| `201_20260818-153638` en ambas clases | 3,00 s / 3,13 s | 51,9 | escenas distintas |
| `906` vs `916`, `20260702-145433` | 1,80 s / 6,03 s | 40,5 | escenas distintas |
| `178_20260701-081313` en ambas clases | 2,53 s / 9,87 s | **6,6** | **comparten momento** |

En los dos primeros la marca de tiempo es de exportación, no de grabación: el contenido no tiene
nada que ver. El tercero sí comparte contenido y se investigó a fondo.

## El caso `178`: contiguo, no duplicado

Comparando los 26 frames del video corto contra los 99 del largo, los tres más parecidos caen
todos en la **posición 294 de 297** del largo, es decir su último medio segundo. Sólo 1 de los 26
frames tiene un gemelo con diferencia menor a 10, y la mediana del mejor calce es 24,1.

Los dos videos son **contiguos**: la toma etiquetada `Interesado` termina donde empieza la
etiquetada `Disgustado`. No es un error de etiquetado — es el ejemplar cambiando de expresión y
el etiquetado siguiéndolo, que es justamente lo que la hipótesis del trabajo afirma que ocurre.

**Lo que sí conviene declarar en el informe:** los frames de esa frontera son casi idénticos y
quedan en clases opuestas. Es ruido irreducible en el borde de un evento continuo, no un defecto
del dataset, pero acota el techo de exactitud alcanzable y hay que decirlo antes de que un
tribunal lo pregunte.

## Conclusión

**No hay ningún caso análogo al `501`.** Ese fue un duplicado real —mismo archivo, dos carpetas—
y quedó resuelto. Lo que hay es una secuencia contigua correctamente partida.

Reproducible con las tres pasadas descritas; ninguna necesita reinferir nada.
