# Propuesta — clasificación de la expresión a partir del clip

## A qué objetivo contribuye

**Objetivo específico 3** del plan de trabajo: *aplicar técnicas de clasificación de Aprendizaje
Automático orientadas a video para identificar estados emocionales*. Es el último objetivo
técnico antes del 4 (aplicación o interfaz), y el que pone a prueba la hipótesis del trabajo.

## Qué se construye

Un clasificador que, dado un clip —16 frames de 224×343 alineados por el eje ojo-hocico, con un
campo de visión que incluye las orejas desde el change `encuadre-que-incluye-las-orejas`—,
prediga una de tres clases: `Disgustado`, `Interesado`, `Sereno`. Y, con el mismo peso, **el
protocolo que permite creerle al número que devuelva.**

El entregable no es sólo el modelo entrenado. Es el modelo **más** su evaluación por ejemplar,
su comparación contra líneas de base, y la ablación que separa lo que aprendió de lo que
podría estar memorizando.

## Por qué ahora

El objetivo 2 quedó cerrado el 28/08/2026 y, por primera vez, el dataset admite una partición
honesta:

| | clips íntegros | ejemplares con clip íntegro |
|---|---:|---:|
| `Disgustado` | 21 | **15** |
| `Interesado` | 56 | 35 |
| `Sereno` | 64 | 36 |
| **total** | **141** | **73 únicos** |

Los 15 + 35 + 36 no suman 73 porque **10 ejemplares tienen clips íntegros en más de una clase**
(37, 54, 55, 178, 218, 230, 237, 906, 907, 910). Ese detalle no es trivia: obliga a que la
partición agrupe por ejemplar y no por par ejemplar-clase, o el mismo caballo termina de los dos
lados del corte.

`GroupKFold` por caballo quedó verificado con 3, 4 y 5 particiones: ninguna clase queda vacía en
ningún fold. Hasta el 27/08 `Disgustado` tenía 5 ejemplares y eso era imposible.

Con la entrega proyectada a septiembre de 2026, este es el trabajo que falta.

## La restricción que manda sobre todo lo demás

**Son 141 muestras.** No 141 mil. Ese número, y no la elección de arquitectura, es lo que
determina qué se puede hacer:

- Entrenar una red de video desde cero —I3D, SlowFast, un ViT temporal— se sobreajusta antes de
  la primera época útil. No es una opción disponible.
- El camino viable es **transfer learning con el backbone congelado**: un extractor de
  características preentrenado que procese cada frame, y una cabeza chica que modele la
  secuencia de 16. Así los parámetros entrenables bajan a un orden compatible con 141 muestras.
- La aumentación ayuda como regularizador, pero **no crea ejemplares nuevos**, que es lo único
  que rompe el confounder de individuo. No se cuenta como solución al tamaño del dataset.

## Qué hay que poder responder al final

Estas preguntas se van a hacer en la defensa, y el diseño existe para tener con qué contestarlas:

1. **¿El modelo aprendió la expresión, o aprendió a reconocer al caballo?** Se responde con la
   partición por ejemplar y con la comparación contra una línea de base que sólo ve la
   identidad del ejemplar.
2. **¿Es mejor que no hacer nada?** Se responde contra dos líneas de base declaradas antes de
   entrenar: la clase mayoritaria y un clasificador sobre un solo frame, sin información
   temporal. Si el modelo temporal no le gana al de un frame, la hipótesis del trabajo —que la
   evolución temporal es lo que permite clasificar— no queda sostenida por la evidencia.
3. **¿Cuánto de esto depende de posiciones que nadie midió?** Se responde con la ablación por
   `origen_alineacion`: entrenar con y sin los frames interpolados y anotados, y reportar
   ambos. La columna existe desde el change anterior justamente para esto.
4. **¿Qué tan estable es el resultado?** Se responde reportando el desvío entre folds y no sólo
   la media. Con 15 ejemplares en la clase minoritaria, una media sola es engañosa.

## Alcance

**Entra:** preparación del tensor de entrada desde `dataset_frames_alineados.csv`, partición por
ejemplar, líneas de base, arquitectura con backbone congelado, entrenamiento, evaluación
cruzada, ablación por origen, matriz de confusión y análisis de errores por clase y ejemplar.

**No entra:** la aplicación o interfaz del objetivo 4; la búsqueda extensiva de
hiperparámetros, que con 141 muestras es sobreajuste al conjunto de validación; y conseguir más
material, que sigue en curso en paralelo y no bloquea este change.

## Riesgo principal

Que el modelo no supere a las líneas de base. **Es un resultado posible y hay que poder
publicarlo.** Con 141 clips y 73 ejemplares, un resultado negativo bien medido vale más que uno
positivo obtenido eligiendo el corte que convenía. Por eso el criterio de aceptación se declara
antes de entrenar y se informa aunque no se cumpla, igual que se hizo con el umbral del relleno
temporal — donde el umbral declarado no se cumplió, se dijo, y eso destapó que el criterio
estaba mal fundado y no el método.
