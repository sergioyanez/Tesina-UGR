# Propuesta — interfaz de inferencia

## Why

**Objetivo específico 4** del plan de trabajo: *desarrollar una aplicación o interfaz para el uso
del modelo resultante*. Es el último objetivo, y el único que produce algo que puede usar alguien
que no seamos nosotros.

Hay un clasificador entrenado y evaluado con honestidad, pero hoy sólo existe como cuatro archivos
de pesos de validación cruzada y una cadena de notebooks. Nadie fuera del equipo puede darle un
video y obtener una respuesta.

**Y hay una restricción que ordena todo el diseño: el modelo NO cumple el criterio que se declaró
antes de entrenarlo** (F1 macro 0,626 contra 0,691), con recall de `Disgustado` de 0,346. La
interfaz no puede presentar como herramienta clínica confiable algo cuya propia evaluación dice que
no lo es. Construirla igual es correcto —el objetivo lo pide, y el modelo tiene señal real sobre la
línea de base—; **presentarla sin sus límites, no**.

## What Changes

- **Un modelo de producción entrenado sobre los 141 clips.** No existe: los cuatro pesos guardados
  son de validación cruzada, cada uno entrenado sobre 3 de 4 folds. Sirven para reportar la
  métrica, no para desplegar.
- **Una función de inferencia de punta a punta**, de un video crudo a una predicción, que reutiliza
  las etapas 01 y 02 en vez de reimplementarlas.
- **Una interfaz** que reciba un video y devuelva la clase predicha, sus probabilidades y —con el
  mismo peso— la confiabilidad medida de esa predicción.
- **Un contrato explícito de negativa.** El modelo consume clips íntegros: 16 frames alineados. Si
  el preprocesamiento no alcanza esa condición sobre el video del usuario, la interfaz **no
  predice**. Devolver una etiqueta sobre un clip contaminado es peor que no devolver nada.

### Lo que NO entra

Reentrenar desde la interfaz; cuentas de usuario; procesamiento por lote; despliegue en un servidor
público; y cualquier presentación del resultado como diagnóstico veterinario.

## Capabilities

### New Capabilities

- `inferencia-y-interfaz`: la inferencia de punta a punta sobre un video nuevo, el contrato de
  negativa cuando el preprocesamiento no alcanza, y cómo la interfaz comunica la predicción junto
  con su confiabilidad medida.

### Modified Capabilities

Ninguna. El preprocesamiento y la clasificación se consumen tal como quedaron; este change no
cambia ningún requisito de esas capacidades.

## Impact

- **Modelo de producción** en `modelos/objetivo3/`, entrenado sobre los 141 clips, con su
  procedencia registrada.
- **`equinos/inferencia.py`**: video → clip íntegro → características → predicción, o negativa
  explicada.
- **Notebook o script de la interfaz**, según lo que decida el diseño.
- **Dependencias:** la inferencia arrastra la cadena completa —YOLO11x y DeepLabCut— así que hereda
  la restricción de `numpy<2` y el costo medido de **~35 s por clip en GPU** (15 s de detector +
  20 s de pose). Eso condiciona qué tipo de interfaz es viable y hay que decidirlo, no descubrirlo
  al final.
- **Documentación:** `README.md` con cómo se usa, y `project.md` con el estado del objetivo 4.

## Riesgo principal

**Que la interfaz lave el resultado negativo.** Una pantalla que devuelve «Disgustado, 87 %» es
indistinguible, para quien la mira, de una herramienta validada — y esta no lo es: acierta un
tercio de los `Disgustado`, que es justamente la clase de interés clínico.

El riesgo no se mitiga con un descargo en letra chica. Se mitiga haciendo que **la confiabilidad
medida sea parte de la salida principal**, con el mismo peso visual que la etiqueta, y que la
interfaz diga qué no puede hacer. Es un requisito del producto, no una nota al pie.

Riesgo secundario: **el costo de inferencia**. 35 s por clip descarta una interfaz que aparente ser
instantánea, y obliga a que el diseño elija entre esperar con señales de progreso o preprocesar
por adelantado.
