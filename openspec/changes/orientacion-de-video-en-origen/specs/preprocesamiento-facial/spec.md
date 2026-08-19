# preprocesamiento-facial (delta)

> Convención de lenguaje normativo: el proyecto se escribe en español, pero conserva los términos
> RFC 2119 entre paréntesis en la primera cláusula normativa de cada requisito. `DEBE (MUST)`
> equivale a MUST y `NO DEBE (MUST NOT)` a MUST NOT.

## ADDED Requirements

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
