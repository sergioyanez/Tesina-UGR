# preprocesamiento-facial (delta)

> Convención de lenguaje normativo: el proyecto se escribe en español, pero conserva los términos
> RFC 2119 entre paréntesis en la primera cláusula normativa de cada requisito. `DEBE (MUST)`
> equivale a MUST y `NO DEBE (MUST NOT)` a MUST NOT.

## ADDED Requirements

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
