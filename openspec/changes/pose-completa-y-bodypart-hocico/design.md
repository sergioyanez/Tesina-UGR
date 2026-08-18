## Context

El notebook 02 infiere pose con DeepLabCut, extrae de esa salida un par ojo/hocico y borra el resto.
La elección del hocico se resuelve hoy así:

```python
CANDIDATOS_HOCICO = ["nose", "snout", "muzzle", "upper_jaw", "mouth"]
# se toma el primero que exista en el modelo -> quedó `nose`
```

mientras que el ojo se resuelve tomando **el más confiable en cada frame** entre todos los bodyparts
cuyo nombre contiene `eye`. Esa asimetría se corresponde con la brecha medida: mediana de confianza
0,891 en el ojo contra 0,709 en el hocico, y 359 frames bloqueados por el hocico contra 10 por el
ojo.

La restricción que domina el diseño no es de cómputo sino de **tamaño muestral**: 34 videos, 92
clips, 22 caballos. Con esa escala, cada frame que se recupera cuenta, y cada decisión de
preprocesamiento tiene que ser defendible ante un tribunal, porque no hay volumen de datos que
absorba un criterio arbitrario. Un keypoint elegido por su posición en una lista escrita a mano es
exactamente el tipo de decisión que no se puede defender. Por la misma razón el trabajo se apoya en
modelos preentrenados —DeepLabCut para pose, transfer learning para el clasificador— y no en
entrenar nada desde cero.

## Goals / Non-Goals

**Goals:**

- Que la salida cruda de pose sobreviva a la corrida, de modo que revisar qué keypoints usar cueste
  segundos y no una hora de reinferencia.
- Elegir el keypoint del hocico con evidencia medida sobre este dataset, y dejar registrado el
  criterio.
- Cuantificar el efecto geométrico de alternar candidatos antes de adoptar esa estrategia, en vez de
  heredarla del ojo por analogía.
- Incorporar los videos nuevos de Disgustado en la misma reinferencia.

**Non-Goals:**

- **No** se resuelve que el pipeline ignore los cambios en los datos de entrada. Es un defecto real
  y va en su propio change; acá se lo sortea con `ETAPAS_A_FORZAR`.
- **No** se toca la anotación manual de ROI. Queda como último recurso para los 227 frames donde el
  modelo no ve la cara, y sólo si el barrido no alcanza el umbral.
- **No** se toca el clasificador ni la partición por caballo. Objetivos 3 y 4 siguen fuera.
- **No** se re-litiga bajar `CONF_MINIMA_KEYPOINT`: la confianza es bimodal y relajar el umbral sólo
  compra frames que el propio modelo da por malos.

## Decisions

### 1. La pose completa se persiste como CSV plano en formato largo

Una fila por `(clip, frame, bodypart)` con `x`, `y`, `confianza`. Formato largo y no ancho porque el
número de bodyparts lo fija el modelo y no el proyecto: en formato ancho, cambiar de modelo cambia el
esquema de columnas y rompe todo lo que lea el archivo.

*Alternativas descartadas:* conservar el `.h5` de DeepLabCut tal cual —es el formato más barato, pero
ata la lectura a tener DeepLabCut instalado y a su convención de MultiIndex `scorer/individuals/
bodyparts/coords`, justamente lo que el resto del paquete evita; y Parquet —mejor comprimido, pero
agrega dependencia y el resto de los artefactos del proyecto son CSV, sin razón para romper la
uniformidad en una tabla de este tamaño.

*A verificar en implementación:* con ~1500 frames y el número de bodyparts que exponga el modelo, la
tabla debería quedar en el orden de las decenas de miles de filas. Si resultara mucho mayor, hay que
reconsiderar el formato antes de fijarlo.

### 2. El bodypart del hocico pasa a ser parámetro de la etapa, con la lista como respaldo

`BODYPART_HOCICO` se declara en la celda `parameters` del notebook 02 y se inyecta desde
`00_pipeline.ipynb`, igual que el resto de los parámetros. Si no se especifica, se mantiene el
comportamiento actual —primer candidato de la lista— para que la etapa siga corriendo sola.

Esto convierte "qué keypoint se usó" en un dato de la corrida en vez de una consecuencia del código,
que es lo que el requisito de auditabilidad pide.

*Alternativa descartada:* fijar el bodypart directamente en el código. Más simple, pero obligaría a
resincronizar el repo con Colab para cada experimento — precisamente la fricción que el change busca
eliminar.

### 3. El barrido vive en el paquete `equinos`, no en el notebook

Módulo nuevo, invocable con dos líneas cortas al estilo de `reporte.confianzas()`. Hay dos razones.
La primera es operativa y está documentada: Sergio pega el contenido de las celdas y el copiado le
mutila el texto, así que todo lo que se ejecuta a mano tiene que caber en una llamada corta. La
segunda es que un barrido en el paquete es testeable con datos sintéticos sin GPU ni pesos, y el
entorno local no puede correr modelos.

### 4. La estrategia de máxima confianza por frame se mide, pero no se adopta por defecto

La transformada de similitud lleva el ojo a `(0.5, 0.34)` y el hocico a `(0.5, 0.74)` del recorte.
La escala queda determinada por la distancia entre ambos puntos en la imagen. Si el hocico alterna
entre `nose` (la punta) y un candidato más atrasado sobre el eje de la cabeza, esa distancia cambia
por razones anatómicas y no de pose, y el recorte se agranda o achica en consecuencia.

En el ojo el problema es menor: los candidatos oculares son aproximadamente simétricos respecto del
eje y a distancia similar del hocico. En el hocico no lo son. Por eso el barrido tiene que reportar
la **variación de la distancia ojo-hocico entre candidatos**, y la estrategia sólo se adopta si esa
variación es chica frente a la dispersión que ya introduce la pose.

*Criterio propuesto para decidirlo, a confirmar con el dato en mano:* adoptarla si la variación
mediana de la distancia queda por debajo del 10 %; si la supera, elegir un único bodypart fijo aunque
cubra menos frames. Un recorte con escala inconsistente entre frames del mismo clip es peor que un
frame no alineado, porque contamina la señal temporal que el objetivo 3 va a modelar.

### 5. Una sola reinferencia, con los videos nuevos adentro

Forzar `preparar_dataset` y `roi_normalizacion` juntas. `etapas.limpiar()` borra el checkpoint, así
que es re-corrida desde cero; no hay camino incremental hoy y construirlo pertenece al change del
dataset desactualizado.

## Risks / Trade-offs

- **El barrido no alcanza el umbral de 70 %** → Es el resultado posible más caro. Se hace falsable de
  entrada: el barrido informa la cobertura de cada candidato antes de tocar el notebook. Si ninguno
  llega, el change cierra con esa conclusión documentada —que es un resultado válido y defendible— y
  la anotación manual pasa a ser el camino, acotada a `031`, `053`, `054`, `055` y `920`.
- **La reinferencia falla a mitad** → El checkpoint permite reanudar, pero esa reanudación **nunca se
  ejercitó en una corrida real**. Conviene mirar el progreso en el primer latido en lugar de darla
  por buena.
- **Los videos nuevos de Disgustado cambian los porcentajes de referencia** → Todos los números
  citados (59,5 %, 359 frames, 83,9 %) son sobre 1472 frames. Al re-correr, el denominador cambia y
  no serán comparables sin recalcular. Hay que reportar ambos.
- **Disgustado sigue teniendo un solo caballo** → Los videos nuevos podrían no aportar ejemplares
  nuevos. Si es así, la clase sigue sin ser evaluable bajo partición por caballo y el plan B de 2
  clases sigue en pie. Conviene verificarlo apenas corra el inventario, antes de la inferencia.
- **`pose_completa.csv` se suma a los artefactos de la etapa** → Registrarlo en `etapas.PIPELINE`
  hace que su ausencia marque la etapa como incompleta. Es lo deseable, pero significa que una raíz
  de proyecto vieja quedará marcada como incompleta hasta re-correr. Es correcto: eso es exactamente
  lo que pasa.

## Migration Plan

1. Correr el barrido con la pose que se persista en la reinferencia y decidir el bodypart con el dato
   a la vista.
2. Fijar `BODYPART_HOCICO` como parámetro y re-derivar los recortes. Esto **no** requiere reinferir:
   la pose ya está persistida.
3. Verificar la cobertura contra el umbral de 70 % y contra el 59,5 % de partida.

Rollback: el bodypart es un parámetro, así que volver atrás es volver al valor anterior y re-derivar.
La pose persistida no se invalida.

## Open Questions

- **¿Qué bodyparts expone realmente el modelo?** La lista de candidatos es una conjetura: no está
  verificado cuáles de `nose`, `snout`, `muzzle`, `upper_jaw` y `mouth` existen. El notebook los
  imprime en tiempo de ejecución y es el primer dato que hay que mirar. Si `nose` resultara el único
  candidato de hocico, todo el barrido pierde sentido y el camino pasa directo a anotación manual.
- **¿Cuánto aportan los videos nuevos?** Cantidad, duración y —sobre todo— si suman ejemplares
  distintos de `037`.
- **¿El umbral de 70 % sigue siendo el criterio correcto?** Se fijó antes de saber que el piso
  irreducible es 15,4 %. No se toca en este change, pero si el barrido queda cerca sin cruzarlo,
  merece discutirse con fundamento en vez de tratarse como un número dado.
