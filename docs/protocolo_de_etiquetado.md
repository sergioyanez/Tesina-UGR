# Protocolo de etiquetado

Cómo se asignó la clase —`Disgustado`, `Interesado` o `Sereno`— a cada video del dataset.

Este documento separa **lo que el código y los datos permiten afirmar** (secciones 1 y 2) de **lo
que declara quien etiquetó** (sección 3), y saca las consecuencias para el trabajo (sección 4).

> **Estado al 07/10/2026.** La sección 3 está completa con el formulario que devolvió la
> veterinaria que etiquetó, Bárbara Zimmermann, fechado el 28/09/2026
> (`docs/formulario_etiquetado_2026-09-28.docx`, la fuente primaria). Lo que sigue en este documento
> la resume; ante cualquier diferencia, vale el formulario.

Motivo: la auditoría del 25/09/2026 (`condiciones_cumplidas.pdf`, R15) marcó como no cumplido el
anclaje del etiquetado en EquiFACS que el plan de trabajo anticipa, y `openspec/project.md` lo
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

## 3. Lo que declara quien etiquetó (formulario del 28/09/2026)

**Quién.** Bárbara Zimmermann, veterinaria con especialización en producción equina y experiencia
en haras de Sangre Pura de Carrera; curso de posgrado «Comportamiento y Bienestar animal» (UNT);
**codificadora certificada de EquiFACS** (Animal FACS, Reino Unido). Hizo el muestreo, la filmación,
la selección de tramos y la clasificación.

**Criterio por clase.** Se consideraron las condiciones ambientales, el comportamiento y la postura
corporal, además de la cara. Entre paréntesis, los códigos EquiFACS tal como figuran en el
formulario:

| Clase | Rasgos |
|---|---|
| `Sereno` | Orejas laterales o hacia caudal en línea con la nariz; ollar neutro; puede haber labio mandibular descendido (AD 160). Cabeza por debajo de la cruz o apoyada en la puerta del box. Puede incluir animales echados. |
| `Disgustado` | Pueden estar presentes: orejas hacia caudal (EAD103), ángulo caudal del ollar elevado (AUH13), esclerótica visible (AD1). Puede incluir movimientos amenazantes de cabeza (rostrales o dorsoventrales), boca abierta con dientes expuestos (AU10, AU16, AU17, AU18) y tensión lateral de la cara —regiones infraorbitaria, maxilar, bucal y masetérica— (AUH21). |
| `Interesado` | Orejas hacia craneal (EAD101), posiblemente aducidas (EAD102). Cabeza y cuello neutros o elevados. Puede haber ollares dilatados (AD38) o esclerótica visible (AD1), sobre todo en animales atados. |

**Referencias.** EquiFACS (Wathan et al., 2015) para codificar los movimientos faciales; como
EquiFACS no asigna emociones, el vínculo gesto → emoción se tomó de etogramas: Lewis et al. (2025),
Torres Borda et al. (2024), Mullard et al. (2016), McDonnell y Poulin (2002) y Waring (1983). Se
marcaron las tres opciones: EquiFACS, otra referencia y criterio propio.

**Qué se miró.** Una combinación de criterios. Los videos se filmaron en el hábitat, con expresión
espontánea —muchas veces como parte de la comunicación entre animales—. Después de cada muestreo se
revisaron todos los videos y **se recortaron los tramos que correspondían a cada expresión**; esos
tramos son los videos del dataset.

**Cuántas personas y acuerdo.** Hubo un **estudio piloto sobre imágenes** con investigadores en
comportamiento animal del DIVAS (Universidad de Milán): kappa de Fleiss = 0,9877. **Los videos del
dataset los clasificó una sola persona**, la misma que los filmó y observó el contexto. Las clases
sin coincidencia entre observadores en el piloto se dejaron fuera del dataset: la clase Miedo, las
expresiones de intensidad elevada, las conductas repetitivas y los caballos atados.

**Subcarpetas.**

- `Seleccionar cabeza?` y `Separar cabeza?` **significan lo mismo**: en el video aparecen varios
  animales. Se apartaron para, si daban errores, procesarlos aislando la cabeza de interés o
  descartarlos.
- `Se tapa una parte de la cara`: otro animal tapa parte de la cabeza del que expresa. Ocurre en
  `Disgustado` porque esa expresión surge de la competencia por alimento, agua, espacio o jerarquía,
  en espacios reducidos y con movimientos bruscos y breves. Se conservaron por lo difícil que es
  conseguir esa clase, aparte para poder descartarlos rápido si fallaban.

**Procedencia.** Todo lo filmó la misma persona en: Hipódromo de Tandil (03/03/2023, 03/07/2023,
10/10/2024), Haras Maryland (14/03/2023), Haras General Lavalle (20/04/2023, 16/04/2026,
18/08/2026, 03/09/2026), Hipódromo de Azul (28/04/2023, 04/10/2025), quintas de particulares cerca de
la UNCPBA (30/06/2023, 08/2026), campus de la UNCPBA (07 a 09/2026) y Estancia «Los Molles», San
Cayetano (02/07/2026). Contexto por clase:

- `Interesado`: la clase más frecuente, presente en todos los establecimientos. En los hipódromos,
  los caballos estaban en box y se filmó al abrir la puerta superior, por eso casi todo lo de
  hipódromo es `Interesado`.
- `Sereno`: presente en todos los establecimientos. En hipódromos, en el horario de descanso del
  personal, con los animales alimentados; en cría extensiva, en los lotes o en corrales tras un
  encierro.
- `Disgustado`: sobre todo en cría extensiva, entre animales de la manada (los dominantes desplazan
  a otros de la comida o la bebida), también en encierros a corral. Pocos casos en box.

## 4. Consecuencias para el trabajo

**Lo que queda resuelto.** Hay autoría identificada, un criterio escrito por clase referido a
unidades de acción de EquiFACS y a etogramas publicados, y criterios de exclusión. La observación de
R15 queda atendida en lo documental.

**Lo que no cambia.**

1. **El acuerdo medido no es el del dataset.** El kappa de 0,99 es del piloto, sobre imágenes y con
   otros observadores; los videos del dataset tienen un solo etiquetador, no ciego al contexto. El
   techo de rendimiento que impone la ambigüedad de las etiquetas sigue sin conocerse (sección 5).
2. **La etiqueta no es sólo facial.** El criterio usa ambiente, comportamiento, postura y movimientos
   de cabeza —amenazas en `Disgustado`, cabeza baja o apoyada en `Sereno`, elevada en
   `Interesado`—. Que la cinemática de cabeza informe la clase es coherente con el criterio, no un
   atajo espurio; pero el modelo, que ve un recorte de la cara, no tiene acceso a todo lo que
   decidió la etiqueta.
3. **Las unidades de acción son indicativas, no obligatorias** («pueden estar presentes»). No hay
   codificación EquiFACS por video, así que no puede verificarse qué rasgos tenía cada uno.

**Lo que el formulario confirma de la sección 2.**

- **Contexto ligado a la clase.** `Disgustado` sale de interacciones en manada en cría extensiva;
  lo de box de hipódromo es casi todo `Interesado`. Un recuento aproximado sobre los archivos de
  `~/videos_de_prueba`, fechando cada uno por su nombre y asignándole el establecimiento del
  formulario, da para los hipódromos 12 `Interesado`, 2 `Disgustado` y 0 `Sereno`. El resto del
  material está repartido entre los mismos establecimientos de cría, así que el confusor existe pero
  es parcial; es coherente con que los metadatos de filmación solos den 0,618.
- **`Seleccionar cabeza?` marca «varios animales en cuadro»**, y en `Disgustado` eso es parte de la
  situación que produce la expresión. Explica por qué esa subcarpeta es casi toda `Disgustado` y por
  qué el modelo la acierta más (85 % contra 66 %): la presencia de otros caballos es una pista de
  contexto.
- **El recorte por tramos sostiene el supuesto** de que la expresión dura toda la toma y puede
  heredarse a los clips.

**Pendientes que abre el formulario.**

- **Caballos atados.** Se declaran excluidos, pero el criterio de `Interesado` menciona la esclerótica
  visible «especialmente en animales atados». Conviene preguntarle si hay videos de animales atados
  en el dataset.

`Separar cabeza?` ya no existe ni en Drive ni en la copia local (verificado el 07/10/2026): el
catálogo no excluye nada, así que no queda material fuera por esa carpeta.

**Texto de limitación para la tesis**, en sustancia:

> Las etiquetas las asignó por video una veterinaria codificadora certificada de EquiFACS, con un
> criterio escrito por clase referido a unidades de acción de EquiFACS y a etogramas publicados, y a
> partir de tramos recortados donde la expresión estaba presente. El criterio incluye el contexto,
> el comportamiento y la postura, no sólo la cara. Un estudio piloto sobre imágenes con
> observadores externos dio un acuerdo alto (kappa de Fleiss 0,99), pero los videos del dataset los
> etiquetó una sola persona, que además los filmó; no se midió el acuerdo sobre este conjunto. En
> consecuencia, el modelo aprende a reproducir el criterio de esta etiquetadora, y no se conoce el
> techo de rendimiento que impone la ambigüedad de las etiquetas. Parte de las clases se asocia a
> contextos de filmación distintos (box de hipódromo para `Interesado`, interacción en manada para
> `Disgustado`), lo que el modelo puede aprovechar como pista.

## 5. Verificación propuesta (no realizada)

El piloto mide el acuerdo sobre imágenes; falta medirlo sobre este dataset, sin re-etiquetar todo:

1. Tomar una muestra estratificada por clase de unos 60 clips, incluidos los de los caballos que el
   modelo falla siempre.
2. Que una segunda persona —idealmente del grupo del piloto— los etiquete **a ciegas** (sin ver la
   carpeta ni la predicción) con el criterio de la sección 3.
3. Informar el kappa de Cohen contra la etiqueta original.

Ese kappa dice cuánto del 0,74 es el techo de las etiquetas y cuánto es margen del modelo.
