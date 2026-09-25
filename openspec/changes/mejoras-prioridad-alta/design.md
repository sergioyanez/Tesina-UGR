## Context

El modelo vigente es el MIL multimodal de `scripts/evaluar_v2.py`: ResNet-18 congelada (512 por
frame) + canal geométrico (9 por frame) + atención sobre 16 frames, 132 355 parámetros
entrenables, evaluado con 4 folds por caballo, corte interno por caballo y early stopping. El
conjunto es de 359 clips, 244 videos y 133 caballos. La auditoría del 25/09 re-entrenó los cuatro
folds y reprodujo las 359 predicciones, así que el protocolo es sólido; lo que falta es poner
el número en contexto.

## Goals / Non-Goals

**Goals:**

- Que el F1 del modelo vigente no pueda citarse sin las líneas de base que lo ponen en contexto,
  medidas sobre los mismos folds y persistidas.
- Que el sobreajuste quede medido y registrado por el propio pipeline, no sólo en la auditoría.
- Que cada número de la tesis tenga un archivo, una rama y un comando que lo produzcan.
- Que los artefactos que ve el tribunal —figura 4, notebook 03— describan el modelo vigente.

**Non-Goals:**

- Cambiar el modelo de producción. Si la línea de base estática empata con el MIL, se informa;
  elegir cuál desplegar es una decisión de los autores que este change deja preparada, no tomada.
- Fusionar las ramas experimentales (`feat/enriquecer-canal-geometrico`,
  `feat/validacion-cruzada-anidada-NestedCV`, `feat/finetuning-visual`). Sus mediciones
  concluyeron «no se adopta» y tocan `clasificacion.py` de formas incompatibles entre sí.
- Recolectar datos nuevos o re-etiquetar.

## Decisions

### D1. Las líneas de base viven en un módulo propio con scikit-learn perezoso

`clasificacion.py` implementa sus métricas a mano para que se vea qué se promedia, y `particion`
evita el stack científico. Las líneas de base lineales sí usan scikit-learn (regresión logística
con estandarización): reimplementarla sería código sin valor de tesis. Se aísla en
`equinos/lineas_de_base.py` y se importa dentro de las funciones, así `import equinos` no
arrastra scikit-learn.

Las métricas se calculan con `clasificacion.f1_macro`, la misma función que el modelo, para que
la comparación no dependa de dos implementaciones.

**Descartado:** agregar las líneas de base como modelos de `clasificacion.evaluar`. Encajan en la
firma, pero `evaluar` recibe folds de diccionarios de clip y las lineales necesitan matrices; la
adaptación ensuciaba la función central sin ganar nada.

### D2. Hiperparámetros de las lineales fijados a priori

`C = 0,1` para entradas de 512 dimensiones y `C = 1` para las de baja dimensión,
`class_weight="balanced"`, estandarización ajustada sólo con el train del fold. No se barren:
barrerlos con el F1 de test las volvería optimistas, y barrerlos con validación interna
exigiría replicar todo el protocolo del MIL para un modelo de comparación. Fijarlos a priori es
conservador para la línea de base, que es la dirección segura.

### D3. Bootstrap por caballo, no por clip

Los 359 clips no son independientes. El intervalo se calcula remuestreando caballos con
reposición (2000 réplicas, semilla fija) y recalculando el F1 macro sobre las predicciones fuera
de muestra agrupadas. Las diferencias entre modelos se calculan **pareadas**, sobre la misma
réplica.

### D4. El control de confusores es un subconjunto, no un re-entrenamiento

Para saber cuánto depende el modelo del atajo de filmación se evalúan las predicciones fuera de
muestra ya existentes sobre el subconjunto sin los rasgos delatores (1080×1920 y duración ≥ 5 s).
No se re-entrena sobre el subconjunto: cambiaría el material de entrenamiento y mezclaría dos
efectos. El subconjunto tiene sólo 28 clips de `Disgustado` y así se informa.

**Descartado:** recortar todos los clips a la misma duración. Es la corrección de fondo, pero
cambia el dataset entero y exige re-ejecutar la etapa 02; queda como recomendación.

### D5. Registrar el train en la curva sin tocar el azar

Evaluar el train interno en modo `eval` después de cada época no consume el generador del
`DataLoader` ni el RNG global (dropout está apagado en `eval`). La auditoría lo verificó: con esa
evaluación agregada, las 359 predicciones y las curvas de validación son idénticas. Se agrega
igual una prueba que lo controle.

### D6. La huella se toma de la constante, no de un nombre de archivo

`_huella_caracteristicas` lee `caracteristicas.ARCHIVO_HUELLA`. El nombre literal fue lo que
envejeció en silencio cuando la caché pasó a `_v2`.

## Risks / Trade-offs

- [El re-entrenamiento de producción no es bit a bit idéntico en otra GPU] → se compara el hash
  de los pesos antes y después en esta máquina y se informa si difiere.
- [Re-ejecutar el notebook 03 sobrescribe artefactos] → el notebook no persiste pesos: el único
  escritor de `objetivo3_evaluacion_v2.json` y de los pesos es `scripts/evaluar_v2.py`.
- [La línea de base estática empata con el MIL y eso debilita la contribución] → es el objetivo
  de medirla; la spec ya exige informarlo sin buscar una configuración que lo revierta.
