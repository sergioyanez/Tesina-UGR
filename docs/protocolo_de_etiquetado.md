# Protocolo de etiquetado

Cómo se asignó la clase —`Disgustado`, `Interesado` o `Sereno`— a cada video del dataset.

Este documento separa **lo que el código y los datos permiten afirmar** de **lo que sólo los
autores pueden declarar**.

> **Estado al 25/09/2026.** Según los autores, las etiquetas las asignó **un profesional
> veterinario**. Se le envió la tabla de la sección 3 para que documente su criterio; **la respuesta
> está pendiente**. Hasta recibirla, la sección 3 sigue sin completar y la sección 4 vale tal como
> está, salvo por la autoría, que ya puede declararse. La segunda parte está sin completar a propósito: no hay en el
repositorio ningún registro de quién etiquetó ni con qué criterio, y escribirlo sin esa fuente
sería inventarlo. Hasta que se complete, la ausencia de protocolo es una **limitación declarada**
del trabajo (sección 4), y así debe figurar en la tesis.

Motivo: la auditoría del 25/09/2026 (`condiciones_cumplidas.pdf`, R15) marcó como no cumplido el
anclaje del etiquetado en EquiFACS que el plan de trabajo anticipa, y `openspec/project.md` ya lo
registraba: «No hay registro de quién etiquetó, con qué criterio ni acuerdo inter-observador».

---

## 1. Lo que establece el código

- **La etiqueta es la carpeta de Drive donde está el video**, más las subcarpetas que cada clase
  declara en `equinos/catalogo.py` (`SUBCARPETAS_POR_CLASE`):

  | Clase | Suman además de la raíz de la clase | Queda afuera |
  |---|---|---|
  | `Disgustado` | `Seleccionar cabeza?`, `Se tapa una parte de la cara` | — |
  | `Interesado` | `Seleccionar cabeza?` | — |
  | `Sereno` | `Seleccionar cabeza?` | `Separar cabeza?` (sin decisión de etiqueta) |

- **La etiqueta se asigna al video entero y se hereda a todos sus clips** (`inventario_clips.csv`,
  columna `clase`). Supone que la expresión se mantiene durante toda la toma.
- **No hay etiqueta por fotograma ni por unidad de acción.** El modelo aprende de una etiqueta por
  clip.
- Las clases se escriben igual en todos los artefactos (`clasificacion.CLASES`, los CSV y los JSON).

## 2. Lo que muestran los datos (conjunto vigente: 359 clips, 133 caballos)

**Procedencia de los clips por carpeta:**

| Clase | Raíz de la clase | `Seleccionar cabeza?` | `Se tapa una parte de la cara` |
|---|---:|---:|---:|
| `Disgustado` | 74 | 23 | 6 |
| `Interesado` | 117 | 1 | — |
| `Sereno` | 135 | 3 | — |

Dos observaciones que conviene que los autores expliquen:

- `Seleccionar cabeza?` aporta casi exclusivamente `Disgustado` (23 de 27 clips), y el modelo acierta
  el 85 % de esos clips contra el 66 % de los `Disgustado` de la raíz. Si esa subcarpeta agrupa
  tomas con un encuadre o una procedencia particular, es otro rasgo que puede delatar la clase.
- `Se tapa una parte de la cara` existe sólo en `Disgustado`; el modelo acierta 2 de sus 6 clips.

**El mismo caballo aparece en más de una clase**: 29 de 133 (7 en las tres). La etiqueta describe
un momento, no al individuo, y la partición por caballo lo tiene en cuenta.

**Caballos que el modelo falla siempre** (con al menos 2 clips, predicciones fuera de muestra):
`013` (5/5, Sereno), `053` (4/4, Sereno), `044` (3/3, Interesado y Sereno), `251` (3/3,
Disgustado), `916` (3/3, Interesado y Sereno), `224`, `920`, `215`, `905` (2/2). Son los primeros
candidatos a una revisión de etiqueta: un error sistemático sobre todos los clips de un animal es
más compatible con una etiqueta discutible que con ruido del modelo.

**La confusión dominante es `Sereno` ↔ `Disgustado`** (20 y 19 clips), no la que parecería más
cercana semánticamente. Sin acuerdo entre anotadores no se puede saber cuánto de eso es error del
modelo y cuánto es ambigüedad de la etiqueta.

## 3. Lo que falta declarar (a completar por los autores)

| Pregunta | Respuesta |
|---|---|
| ¿Quién asignó las etiquetas? (nombre y rol; formación en etología o veterinaria) | Un profesional veterinario (según los autores). **[COMPLETAR nombre y formación]** |
| ¿Hubo un criterio escrito? ¿Qué rasgos definen cada clase? (orejas, ojos, ollares, boca, postura) | **[COMPLETAR]** |
| ¿Se usaron unidades de acción de EquiFACS (Wathan et al., 2015) u otra referencia? | **[COMPLETAR]** |
| ¿Se etiquetó mirando el video completo, un tramo, o el contexto de filmación? | **[COMPLETAR]** |
| ¿Cuántas personas etiquetaron cada video? ¿Se midió el acuerdo? | **[COMPLETAR]** |
| ¿Qué significan `Seleccionar cabeza?`, `Se tapa una parte de la cara` y `Separar cabeza?`, y por qué la segunda sólo existe en `Disgustado`? | **[COMPLETAR]** |
| ¿De dónde provienen los videos (establecimientos, fechas, quién filmó) y el contexto de cada clase (p. ej., qué provocaba el disgusto)? | **[COMPLETAR]** |

## 4. Texto de limitación para la tesis

Mientras la sección 3 no esté completa, la tesis debería decir, en sustancia:

> Las etiquetas fueron asignadas por video por un profesional veterinario, sin un protocolo escrito
> basado en unidades de acción faciales y sin medir el acuerdo entre anotadores. En consecuencia, el modelo aprende a reproducir las etiquetas de este conjunto, y no
> puede afirmarse que mida estados emocionales en un sentido etológico; tampoco se conoce el techo
> de rendimiento que impone la ambigüedad de las etiquetas.

## 5. Verificación propuesta (no realizada)

Una forma barata de acotar el problema, sin re-etiquetar todo:

1. Tomar una muestra estratificada por clase de unos 60 clips, incluidos los de los caballos que el
   modelo falla siempre.
2. Que dos personas los etiqueten **a ciegas** (sin ver la carpeta ni la predicción), con un
   criterio escrito basado en EquiFACS.
3. Informar el acuerdo (kappa de Cohen) entre ellas y contra la etiqueta original.

Ese kappa dice cuánto del 0,74 es el techo de las etiquetas y cuánto es margen del modelo.
