# preprocesamiento-facial Specification

## Purpose
TBD - created by archiving change roi-normalizacion-pose. Update Purpose after archive.
## Requirements
### Requirement: Detección del equino en cada frame

El sistema DEBE (MUST) localizar al equino en cada frame de cada clip usando un detector preentrenado,
sin requerir anotación manual. Cuando haya más de un equino detectado, el sistema DEBE quedarse con
la detección de mayor confianza.

#### Scenario: Frame con un equino claramente visible

- **WHEN** se procesa un frame donde el detector encuentra un equino con confianza mayor al umbral
- **THEN** el sistema registra la caja delimitadora en coordenadas de píxel de la imagen original
- **AND** registra la confianza de la detección

#### Scenario: Frame con varios equinos

- **WHEN** el detector devuelve más de una caja de clase equino sobre el umbral
- **THEN** el sistema conserva únicamente la de mayor confianza
- **AND** registra la cantidad total de detecciones para poder auditar el caso después

#### Scenario: Frame sin detección

- **WHEN** el detector no encuentra ningún equino sobre el umbral de confianza
- **THEN** el sistema marca el frame como sin detección en lugar de inventar una caja
- **AND** el frame queda contabilizado en el reporte de cobertura

### Requirement: Suavizado temporal de las detecciones

El sistema DEBE suavizar las cajas detectadas a lo largo de cada clip mediante una mediana móvil,
de modo que una detección aislada mal ubicada no produzca un salto en la ROI. El suavizado NO DEBE (MUST NOT)
aplanar el movimiento sostenido del animal o de la cámara.

#### Scenario: Detección aislada errónea

- **WHEN** un clip tiene una secuencia de cajas estables con una sola caja muy desplazada en el medio
- **THEN** la caja suavizada de esa posición coincide con la de sus vecinas

#### Scenario: Movimiento sostenido del animal

- **WHEN** las cajas del clip se desplazan de forma progresiva y consistente entre frames
- **THEN** el suavizado conserva ese desplazamiento sin aplanarlo

#### Scenario: Huecos sin detección dentro del clip

- **WHEN** algunos frames del clip no tienen detección pero otros sí
- **THEN** el sistema rellena los huecos con la detección válida más cercana
- **AND** registra cuántos frames debieron rellenarse

### Requirement: Estimación de keypoints faciales

El sistema DEBE estimar keypoints faciales del equino —como mínimo un punto ocular y un punto en el
hocico— mediante un modelo preentrenado para cuadrúpedos, sin anotación manual. El sistema NO DEBE (MUST NOT)
asumir nombres fijos de keypoints: DEBE descubrir el esquema que expone el modelo en tiempo de
ejecución y seleccionar los faciales a partir de él.

Cuando el esquema del modelo ofrezca varios candidatos para un mismo punto facial, el sistema NO
DEBE (MUST NOT) resolver la elección por el orden de una lista fijada a priori. La selección DEBE
(MUST) sustentarse en la confianza medida sobre el propio dataset, y el keypoint elegido DEBE (MUST)
quedar registrado como parámetro explícito de la etapa, de modo que la corrida sea reproducible y la
decisión, auditable.

#### Scenario: Keypoints faciales disponibles

- **WHEN** el modelo devuelve el punto ocular y el del hocico con confianza suficiente
- **THEN** el sistema los registra en coordenadas de la imagen original
- **AND** los usa como referencia para la normalización de pose

#### Scenario: Esquema de keypoints desconocido

- **WHEN** se carga el modelo de pose por primera vez
- **THEN** el sistema imprime el listado completo de keypoints disponibles
- **AND** deja explícito cuáles seleccionó como ocular y como hocico

#### Scenario: Varios candidatos para el mismo punto facial

- **WHEN** el esquema del modelo expone más de un bodypart que podría cumplir el rol de hocico
- **THEN** el sistema usa el que se haya fijado como parámetro de la etapa
- **AND** deja registrado cuál usó junto a los resultados de la corrida

#### Scenario: Keypoints faciales ausentes o de baja confianza

- **WHEN** no se obtiene el par ocular/hocico con confianza suficiente en un frame
- **THEN** el sistema no normaliza la pose de ese frame
- **AND** recurre a la ROI derivada de la caja de detección, dejándolo marcado en el CSV

### Requirement: Normalización de pose por transformada de similitud

El sistema DEBE (MUST) alinear cada recorte facial mediante una transformada de similitud que lleve el
punto ocular y el del hocico a posiciones canónicas fijas. La transformación DEBE preservar la
relación de aspecto de la cara, es decir, escalar de forma uniforme en ambos ejes.

La preservación de la relación de aspecto es una propiedad de la **transformada**, no de la forma
del recorte de destino. Un destino rectangular NO deforma la cara mientras la escala siga siendo
isótropa: la relación de aspecto del destino determina el **campo de visión**, no el estiramiento.
Confundir ambas cosas es lo que ató históricamente el encuadre a un cuadrado.

#### Scenario: Cabezas en poses distintas

- **WHEN** se alinean dos frames cuya cabeza aparece con rotación, escala y posición muy distintas
- **THEN** en ambos recortes normalizados el ojo y el hocico caen en la misma posición

#### Scenario: Escalado uniforme

- **WHEN** se aplica la transformada de similitud a un recorte
- **THEN** la escala aplicada es idéntica en el eje horizontal y en el vertical, sin deformar la cara

#### Scenario: Destino rectangular sin deformación

- **WHEN** el recorte de destino es más alto que ancho
- **THEN** la escala aplicada sigue siendo idéntica en ambos ejes
- **AND** la cara no queda estirada ni comprimida respecto del frame de origen

#### Scenario: Puntos de referencia degenerados

- **WHEN** el punto ocular y el del hocico coinciden, de modo que no definen un eje
- **THEN** el sistema no produce una transformación inválida y trata el frame como no normalizable

### Requirement: Trazabilidad hasta el nivel de caballo

El sistema DEBE (MUST) arrastrar `id_caballo`, `video_origen` y `clip` en cada registro de frame alineado,
de manera que la partición por ejemplar siga siendo posible aguas abajo.

#### Scenario: Generación del CSV de frames alineados

- **WHEN** se completa el procesamiento de todos los clips
- **THEN** el CSV resultante contiene, por cada frame, su clase, `id_caballo`, `video_origen`,
  `clip`, la ruta del frame alineado y los indicadores de calidad de la detección

### Requirement: Reporte de cobertura de detección

El sistema DEBE (MUST) reportar qué proporción de frames obtuvo detección del equino y keypoints faciales
válidos, desglosado por clase y por caballo. Este reporte es el criterio que decide si el pipeline
automático es utilizable o si hay que recurrir a anotación manual.

El reporte DEBE incluir además el **vector de oreja legible** y la **distribución del relleno del
recorte**. La cobertura de alineación sola NO DEBE (MUST NOT) presentarse como criterio
suficiente de que el preprocesamiento es utilizable: mide cuántos frames se pudieron alinear, no
qué parte de la cara quedó dentro del recorte, y un encuadre puede tener cobertura alta dejando
fuera el rasgo más informativo de la clase de interés.

#### Scenario: Cobertura suficiente

- **WHEN** finaliza el procesamiento
- **THEN** el sistema informa el porcentaje de frames con detección y con keypoints, global y
  desglosado por clase y por caballo
- **AND** informa el vector de oreja legible y la distribución del relleno

#### Scenario: Cobertura insuficiente en una clase o caballo

- **WHEN** alguna clase o algún caballo queda por debajo del umbral de cobertura aceptable
- **THEN** el sistema lo señala explícitamente como advertencia
- **AND** no se continúa al entrenamiento sin decidir antes qué hacer con esos casos

#### Scenario: Cobertura alta con las orejas fuera del recorte

- **WHEN** la cobertura de alineación supera su umbral pero el vector de oreja legible queda por
  debajo del criterio declarado
- **THEN** el sistema señala que el preprocesamiento no cumple el criterio de encuadre
- **AND** no se considera cerrada la etapa por la sola cobertura de alineación

### Requirement: Relleno temporal de los keypoints faltantes

Cuando en un clip falte la detección de los keypoints de un frame y existan frames vecinos con
detección válida, el sistema DEBE (MUST) estimar la posición faltante interpolando temporalmente
entre esos vecinos, y construir la transformada de similitud con los puntos así obtenidos. La
estimación NO DEBE (MUST NOT) construirse por un camino distinto del que usa un frame con
detección propia: se interpolan los puntos, no la transformada.

El relleno DEBE estar acotado por un tope declarado de frames consecutivos sin detección. Superado
ese tope, el frame DEBE seguir cayendo al recorte de caja sin alinear. El tope DEBE ser un
parámetro auditable de la etapa y su valor DEBE surgir de la validación, no del criterio de quien
implementa.

Este requisito existe porque la unidad que consume el modelo es el clip y no el frame. Con
cobertura del 69,9 % por frame, sólo 80 de 342 clips tenían sus 16 frames alineados, y la clase
minoritaria quedaba en 5 clips de 4 ejemplares: un dataset que no admite partición por caballo. El
80 % de los huecos son de uno o dos frames rodeados de detecciones válidas, y la cabeza de un
caballo no se desplaza de forma arbitraria en ese lapso.

Los frames al principio o al final del clip, que no tienen vecinos detectados de ambos lados, DEBEN
tratarse por un mecanismo distinto y con su propio tope, y NO DEBEN contarse como interpolados.

#### Scenario: Hueco interior dentro del tope

- **WHEN** un frame no tiene detección de keypoints
- **AND** existen frames con detección antes y después, separados por no más que el tope declarado
- **THEN** el sistema estima el ojo y el hocico interpolando entre esos vecinos
- **AND** construye el recorte con la misma transformada de similitud que usaría con una detección
  propia

#### Scenario: Hueco interior más largo que el tope

- **WHEN** la cantidad de frames consecutivos sin detección supera el tope declarado
- **THEN** el sistema no interpola esos frames
- **AND** esos frames caen al recorte de caja sin alinear, como antes

#### Scenario: Clip sin ninguna detección

- **WHEN** ningún frame del clip tiene detección válida de keypoints
- **THEN** el sistema no estima ninguna posición
- **AND** el clip entero queda con recortes de caja sin alinear

### Requirement: Trazabilidad del origen de la alineación

Cada frame DEBE (MUST) registrar en los artefactos **cómo** quedó alineado: por detección propia,
por interpolación entre vecinos, por sostenimiento en un extremo del clip, o por recorte de caja
sin alinear. Una marca binaria de "alineado" NO DEBE ser el único dato disponible.

Los reportes de cobertura DEBEN poder desglosarse por ese origen, y NO DEBEN presentar como
equivalentes un frame detectado y uno estimado.

Sin esa distinción, los frames estimados se vuelven invisibles en cuanto se calcula la primera
métrica, no se puede contestar cuántas posiciones fueron observadas y cuántas inferidas, y no se
puede entrenar con y sin ellos para medir su efecto.

#### Scenario: Reporte de cobertura sobre un dataset con frames rescatados

- **WHEN** se informa la cobertura de alineación de una corrida que usó relleno temporal
- **THEN** el reporte distingue los frames alineados por detección de los estimados
- **AND** permite recalcular la cobertura considerando únicamente los detectados

#### Scenario: Ablación de los frames estimados

- **WHEN** se quiere entrenar excluyendo los frames cuya posición fue estimada
- **THEN** el dataset permite filtrarlos por el origen registrado, sin recomputar nada

### Requirement: Validación del relleno contra detecciones conocidas

El relleno temporal DEBE (MUST) validarse ocultando frames que **sí** tienen detección válida,
estimando su posición con el mismo procedimiento y midiendo el error contra la posición realmente
detectada. La validación NO DEBE (MUST NOT) medir la estimación contra los mismos datos con que se
construyó.

El error DEBE informarse en píxeles dentro del recorte final, que es el desplazamiento que ve el
modelo, y DEBE acompañarse de su distribución —al menos mediana, percentil 90 y máximo—, no
solamente de un promedio.

El umbral de aceptación DEBE declararse antes de medir, y el resultado DEBE informarse aunque no lo
cumpla.

El umbral DEBE anclarse en una **línea de base medida**: el desplazamiento del recorte entre dos
detecciones propias consecutivas. Ese valor es el piso físico del problema —lo que se mueve la
cabeza del caballo entre dos frames muestreados— y ninguna estimación puede ser sistemáticamente
más precisa que él sin estar sobreajustando. Un umbral fijado sin esa referencia no distingue "el
relleno no sirve" de "el criterio era imposible de cumplir".

**Umbral vigente.** Un largo de hueco se acepta si, medido en píxeles dentro del recorte de 224, su
**mediana** y su **percentil 90** no superan los de la línea de base. Se informa además el máximo,
que no forma parte del criterio pero sí del informe, porque lo que arruina una muestra es la cola.

> **Historia de este umbral, que hay que poder contar.** El 28/08/2026 se declaró, antes de medir,
> un umbral de mediana ≤ 3,0 px y p90 ≤ 8,0 px. Ningún largo de hueco lo cumplió: el mejor caso
> dio 13,99 px. Al medir después la línea de base —16,28 px de mediana, 78,22 de p90— quedó claro
> que el umbral exigía que la estimación fuera cinco veces más exacta que el propio movimiento del
> caballo entre frames, y que la interpolación de un hueco de 1 es de hecho **más precisa que usar
> la detección real del frame vecino**. El umbral se re-declaró el 28/08/2026 anclándolo en la
> línea de base. El resultado contra el umbral original queda registrado en `validacion.md`, y se
> informa junto con el nuevo: cambiar un criterio después de ver los números sólo es legítimo si
> queda escrito que se cambió, por qué, y qué daba antes.

La validación DEBE estratificarse por la cantidad de frames consecutivos que se está estimando, de
modo que el tope de relleno surja de la curva de error medida.

Este requisito existe porque una verificación que no puede fallar ya produjo una conclusión
equivocada en este proyecto: el error de alineación de 1e-13 px que se informaba comparaba la
transformada contra los mismos puntos con que se la había construido, y daba cero aunque el recorte
fuera inutilizable.

#### Scenario: Validación por omisión de una detección conocida

- **WHEN** se valida el relleno sobre un frame con detección válida
- **THEN** el sistema oculta esa detección, estima la posición desde los vecinos y la compara
  contra la real
- **AND** informa el error en píxeles del recorte final, con su distribución

#### Scenario: El error supera el umbral declarado

- **WHEN** la distribución de error medida no cumple el umbral declarado para un largo de hueco
- **THEN** ese largo de hueco queda por encima del tope de relleno
- **AND** los frames correspondientes siguen cayendo al recorte de caja

#### Scenario: Sesgo conocido de la validación

- **WHEN** se informa el resultado de la validación
- **THEN** se hace constar que los frames usados para validar son los que sí se detectaron, que no
  son una muestra aleatoria de los que fallaron

### Requirement: Orientación declarada en los metadatos

Toda lectura de video del pipeline DEBE (MUST) entregar el frame en su orientación de
visualización, aplicando la rotación declarada en los metadatos del archivo. El sistema NO DEBE
(MUST NOT) asumir que la biblioteca de video lo hace por su cuenta: el comportamiento varía entre
versiones y backends.

Las dimensiones que el sistema registre de cada video DEBEN (MUST) ser las de visualización y no
las de almacenamiento, que en un video rotado están intercambiadas.

La consecuencia de incumplirlo no es un fallo sino una pérdida silenciosa: un detector entrenado
sobre imágenes derechas no reconoce al animal girado 90 grados, no produce caja, y el frame se
descarta sin error. Y esa pérdida se concentra en los ejemplares filmados en vertical, con lo que
además sesga el dataset.

#### Scenario: Video con rotación declarada

- **WHEN** se lee un video cuyos metadatos declaran una rotación distinta de cero
- **THEN** el sistema entrega los frames ya rotados a su orientación de visualización
- **AND** registra las dimensiones correspondientes a esa orientación

#### Scenario: Video sin rotación declarada

- **WHEN** se lee un video sin rotación en sus metadatos
- **THEN** el sistema entrega los frames tal como están almacenados
- **AND** el resultado es idéntico al de no haber pedido corrección alguna

#### Scenario: Comportamiento de la biblioteca desconocido

- **WHEN** se quiere verificar qué hace la biblioteca de video del entorno ante un archivo rotado
- **THEN** el sistema informa las dimensiones almacenadas, las de visualización y la forma real
  del frame entregado
- **AND** distingue el caso en que ya aplica la rotación del caso en que no, porque aplicarla dos
  veces dejaría el dataset invertido

#### Scenario: Auditoría de la pérdida por orientación

- **WHEN** se consulta el reporte de frames perdidos
- **THEN** el sistema informa cuántos frames con detección no llegaron a recortarse
- **AND** los desglosa por rotación declarada, por clase y por video de origen

### Requirement: Independencia entre el inventario y la escritura de clips

El sistema DEBE (MUST) dimensionar cada clip con la forma del primer frame que efectivamente
escribe, y NO DEBE (MUST NOT) tomarla del inventario.

Son dos mediciones de la misma cosa hechas en momentos distintos, y cuando divergen el resultado
es un archivo vacío o corrupto en lugar de un error.

#### Scenario: Dimensiones coincidentes

- **WHEN** la forma del primer frame coincide con la registrada en el inventario
- **THEN** el clip se escribe con esa forma, sin observaciones

#### Scenario: Dimensiones divergentes

- **WHEN** la forma del primer frame difiere de la registrada en el inventario
- **THEN** el sistema usa la del frame
- **AND** lo informa, porque la divergencia indica que el inventario se construyó con otro
  criterio de lectura

### Requirement: Región de interés que incluye las orejas

El recorte facial canónico DEBE (MUST) contener las orejas del equino, no sólo el eje ojo-hocico.
La posición de las orejas es un rasgo informativo de la expresión —en la etología equina, el
indicador principal de disgusto y estrés—, y un recorte que las deja fuera impide que el
clasificador del objetivo 3 pueda usarlas, sin producir ningún error observable.

El encuadre DEBE (MUST) declararse como un **campo de visión en unidades de la distancia
ojo-hocico `d`**, con tres parámetros independientes: el aire por encima del ojo, el aire por
debajo del hocico y el semiancho lateral. NO DEBE (MUST NOT) declararse como dos fracciones sobre
un lado único, porque esa parametrización obliga a que el campo vertical y el lateral crezcan
juntos y oculta que el déficit del encuadre es asimétrico.

El recorte resultante NO DEBE (MUST NOT) exigirse cuadrado. Su relación de aspecto DEBE ser la que
resulte del campo de visión declarado, y las dimensiones en píxeles del destino DEBEN respetarla.

El par de anclaje DEBE seguir siendo el ojo y el hocico. Los keypoints de oreja NO DEBEN (MUST NOT)
usarse como puntos de alineación: su disponibilidad medida es de ~60 % por oreja contra la del par
facial, y anclarse en ellos cambiaría la cobertura de la etapa a cambio de nada — las orejas entran
por campo de visión, no por anclaje.

#### Scenario: Encuadre declarado por campo de visión

- **WHEN** se configura la etapa de normalización
- **THEN** el encuadre queda expresado como aire sobre el ojo, aire bajo el hocico y semiancho
  lateral, todos en unidades de `d`
- **AND** las dimensiones del recorte de destino se derivan de esa relación de aspecto

#### Scenario: Recorte no cuadrado

- **WHEN** el campo de visión declarado es más alto que ancho
- **THEN** el sistema produce recortes rectangulares de dimensiones uniformes en todo el dataset
- **AND** no reescala los ejes de forma independiente para forzarlos a un cuadrado

#### Scenario: El par de anclaje no cambia

- **WHEN** se alinea un frame bajo el nuevo encuadre
- **THEN** los puntos que definen la transformada siguen siendo el ojo y el hocico
- **AND** las posiciones anotadas manualmente bajo el encuadre anterior siguen siendo válidas sin
  volver a anotarse

### Requirement: Inclusión de orejas como criterio de aceptación del encuadre

La etapa DEBE (MUST) medir e informar qué proporción de las orejas cae dentro del recorte, y esa
proporción DEBE ser un criterio de aceptación del encuadre, al mismo nivel que la cobertura de
alineación. Una cobertura de alineación alta con las orejas fuera del cuadro NO DEBE (MUST NOT)
considerarse un preprocesamiento aceptable.

La medición DEBE hacerse sobre los keypoints de oreja ya persistidos en la pose completa,
proyectándolos por la misma transformada de similitud que produce el recorte. NO DEBE requerir
volver a inferir la pose: eso es lo que permite comparar varios encuadres antes de escribir una
sola imagen.

La unidad de medida DEBE ser el **vector de oreja** —la base y la punta de una misma oreja, las
dos dentro del recorte—, y NO DEBE (MUST NOT) ser el conteo suelto de keypoints de oreja. Lo que
informa el estado del animal es la **orientación** de la oreja, y eso se lee del vector
base→punta: una base sin su punta no dice hacia dónde apunta. El conteo suelto sobrestima de forma
sistemática, y no por poco — ver el registro del criterio, más abajo.

**Criterio declarado, antes de medir.** Un encuadre candidato es aceptable si cumple las dos
condiciones:

1. **El vector de oreja es legible en al menos la mitad de los frames** que tienen alguna oreja
   completa detectada. Anclado en la distribución medida: la punta de oreja está a 0,81 `d` por
   encima del ojo en la mediana, de modo que un campo que contenga la mediana contiene la mitad de
   los casos.
2. **No empeora el relleno mediano respecto del encuadre vigente.** La línea de base **no es
   cero**: el encuadre actual ya produce 14,1 % de relleno mediano. El candidato no debe superarlo.

Ambas condiciones se anclan en cantidades medidas sobre la corrida vigente y no en números
redondos. El resultado DEBE informarse aunque ningún candidato las cumpla, y si el criterio se
re-declara, DEBE quedar escrito que se cambió, por qué, y qué daba antes.

> **Registro de este criterio, que hay que poder contar.** Se declaró el 28/08/2026, antes de
> barrer, con dos diferencias respecto de la versión de arriba, y las dos se corrigieron con lo
> medido y no con lo conveniente:
>
> - **La unidad era el conteo suelto de keypoints de oreja.** Al barrer se vio que sobrestima: el
>   candidato C daba 64,4 % de keypoints dentro pero sólo 50,6 % de vectores legibles, porque
>   metía las bases (89 %) y se comía las puntas (38 %). Como la orientación de la oreja es el
>   rasgo informativo, la métrica correcta es el vector, y el criterio se re-ancló en él.
> - **La línea de base de relleno decía 22,2 %.** Ese número salía de aproximar el recorte por la
>   caja envolvente del rectángulo rotado, que lo sobrestima. Adoptando el muestreo por rejilla que
>   ya usaba `reporte.calidad_recortes` —que mide el rectángulo y no su envolvente— la línea de
>   base real es **14,1 %**. No se aflojó el criterio: se midieron mejor los dos lados de la
>   comparación, y el candidato adoptado empata la línea de base en vez de superarla.
>
> Ninguna de las dos correcciones cambió la decisión: el encuadre adoptado cumple el criterio
> viejo y el nuevo. Queda registrado igual, porque cambiar un criterio después de ver los números
> sólo es legítimo si queda escrito que se cambió, por qué, y qué daba antes.

Al informar la inclusión de orejas DEBE hacerse constar su **sesgo de denominador**: sólo se mide
sobre frames donde la oreja fue detectada con confianza suficiente, y una oreja fuera del cuadro
tiende a no detectarse, así que la métrica favorece a todos los encuadres por igual y sobrestima a
todos. Por eso la decisión del encuadre NO DEBE (MUST NOT) tomarse sólo con la tabla: DEBE
acompañarse de inspección visual de una muestra de recortes de la clase minoritaria, que es
evidencia independiente de ese sesgo.

#### Scenario: Barrido de encuadres sin reprocesar imágenes

- **WHEN** se comparan varios campos de visión candidatos
- **THEN** el sistema informa para cada uno el vector de oreja legible y la distribución del
  relleno
- **AND** no vuelve a inferir la pose ni escribe imágenes para producir esa comparación

#### Scenario: Encuadre que mete las bases de oreja pero no las puntas

- **WHEN** un candidato deja dentro del recorte la base de una oreja pero no su punta
- **THEN** ese frame NO cuenta como oreja legible, aunque el conteo suelto de keypoints suba
- **AND** el informe distingue ambas métricas, para que la diferencia entre ellas quede a la vista

#### Scenario: Encuadre que mejora las orejas empeorando el relleno

- **WHEN** un candidato supera el 50 % de vector de oreja legible pero su relleno mediano excede el
  de la línea de base
- **THEN** el sistema informa que no cumple el criterio declarado
- **AND** informa igualmente ambas métricas, para que la compensación quede a la vista

#### Scenario: Informe de la métrica de orejas

- **WHEN** se informa la inclusión de orejas de una corrida
- **THEN** se informa junto al denominador sobre el que se calculó
- **AND** se hace constar que los frames sin oreja detectada no son una muestra aleatoria

### Requirement: Trazabilidad del relleno del recorte

Cada frame DEBE (MUST) registrar qué fracción de su recorte se completó con píxeles inventados por
replicación de borde. Una marca binaria de "tuvo relleno" NO DEBE ser el único dato disponible.

Este requisito existe porque la cantidad de relleno depende de a qué distancia se filmó al animal,
que correlaciona con el ejemplar, que predice la clase. Es la misma clase de atajo que el encuadre
canónico existe para evitar, y ampliar el campo de visión lo aumenta. Sin la fracción registrada
por frame no se puede contestar cuánto del resultado del clasificador se apoya en ese artefacto.

Los reportes de cobertura DEBEN poder desglosarse por esa fracción, y el dataset DEBE permitir
filtrar por ella sin recomputar nada, del mismo modo que se filtra por `origen_alineacion`.

#### Scenario: Registro de la fracción de relleno

- **WHEN** se escribe un frame alineado cuyo rectángulo canónico se sale del cuadro de origen
- **THEN** el registro del frame incluye la fracción de su área completada por replicación

#### Scenario: Ablación por relleno

- **WHEN** se quiere entrenar excluyendo los frames con relleno por encima de un valor
- **THEN** el dataset permite filtrarlos por la fracción registrada, sin volver a procesar imágenes

### Requirement: Persistencia de la pose completa

El sistema DEBE (MUST) persistir la salida cruda del modelo de pose —todos los bodyparts que el
modelo expone, para todos los frames procesados, con sus coordenadas y su confianza— como artefacto
de la etapa, y NO DEBE (MUST NOT) descartarla junto con los intermedios del modelo.

La razón es que la elección de qué keypoints usar es una decisión analítica que se revisa varias
veces, mientras que la inferencia es la operación más cara del pipeline. Sin este artefacto, cada
revisión de esa decisión obliga a reinferir, y el costo desalienta justamente el análisis que la
haría correcta.

#### Scenario: Fin de la inferencia de pose

- **WHEN** termina la estimación de pose sobre el conjunto de clips
- **THEN** el sistema escribe una tabla con una fila por frame y por bodypart, o equivalente
- **AND** cada fila incluye el clip, el frame, el nombre del bodypart, sus coordenadas y su confianza
- **AND** la tabla queda en la raíz de proyecto, alcanzable sin volver a ejecutar el modelo

#### Scenario: Análisis posterior de keypoints

- **WHEN** se quiere evaluar un keypoint distinto del que se venía usando
- **THEN** el sistema responde a partir de la tabla persistida
- **AND** no requiere volver a ejecutar el modelo de pose

#### Scenario: Trazabilidad hacia el CSV de keypoints

- **WHEN** existen tanto la tabla de pose completa como el CSV de keypoints faciales
- **THEN** cada fila del CSV de keypoints es reconstruible desde la tabla completa aplicando el
  criterio de selección vigente
- **AND** esa correspondencia es verificable de forma automática

### Requirement: Barrido de candidatos a keypoint facial

El sistema DEBE (MUST) ofrecer, sobre la tabla de pose persistida, un barrido que informe la
cobertura de alineación alcanzable con cada candidato a keypoint facial y con la estrategia de
máxima confianza por frame, desglosada por clase y por caballo.

El barrido NO DEBE (MUST NOT) requerir ejecutar el modelo de pose.

#### Scenario: Comparación entre candidatos

- **WHEN** se solicita el barrido sobre los candidatos a hocico
- **THEN** el sistema informa, para cada candidato, la proporción de frames que superarían el umbral
  de confianza junto con el keypoint ocular
- **AND** informa el mismo dato para la estrategia de máxima confianza por frame

#### Scenario: Candidato ausente en el modelo

- **WHEN** un candidato del listado no figura entre los bodyparts que el modelo expone
- **THEN** el sistema lo informa como ausente
- **AND** continúa el barrido con los candidatos restantes en lugar de fallar

#### Scenario: Efecto geométrico de alternar keypoints

- **WHEN** el barrido evalúa la estrategia de máxima confianza por frame
- **THEN** el sistema informa además cuánto varía la distancia ojo-hocico entre los candidatos
  involucrados
- **AND** deja explícito que esa variación se traslada a la escala del recorte normalizado

