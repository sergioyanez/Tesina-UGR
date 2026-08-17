# Diseño técnico: ROI facial y normalización de pose

## Restricción que gobierna todas las decisiones

**El dataset tiene 34 videos, 92 clips y 22 caballos.** Cualquier enfoque que requiera anotar
manualmente cientos de frames para entrenar un detector propio consume más tiempo del que el
proyecto puede absorber, y encima produciría un modelo entrenado sobre los mismos 22 ejemplares que
queremos generalizar. Por eso **todo el pipeline se apoya en modelos preentrenados de propósito
general, con cero anotación**.

La contrapartida es que la calidad de la detección no está garantizada, y de ahí que el reporte de
cobertura sea un requisito de primera clase y no un extra.

## Arquitectura

```
inventario_clips.csv  (92 clips, salida del Bloque 2)
        │
        ├─► [1] Muestreo de frames por clip
        │
        ├─► [2] Detección del equino        · YOLO preentrenado en COCO, clase "horse"
        │         └─ una caja por frame, la de mayor confianza
        │
        ├─► [3] Suavizado temporal          · mediana móvil sobre las cajas del clip
        │         └─ + relleno de huecos con la detección válida más cercana
        │
        ├─► [4] Keypoints faciales          · modelo de pose para cuadrúpedos
        │         └─ punto ocular + punto del hocico
        │
        ├─► [5] ROI cuadrada + alineación   · transformada de similitud
        │
        └─► [6] Recorte 224×224 + CSV + reporte de cobertura
```

## Decisiones

### Por qué un detector COCO y no un detector de caballos entrenado a medida

COCO incluye la clase `horse` entre sus 80 categorías, así que un YOLO preentrenado localiza al
animal sin una sola anotación nuestra. Entrenar un detector propio exigiría anotar cientos de cajas
para resolver un problema que ya está resuelto. La detección de cuerpo completo no es el objetivo
final, pero acota el espacio de búsqueda y sirve de respaldo cuando fallan los keypoints.

### Por qué keypoints y no recortar la cabeza por geometría

Se evaluó derivar la cabeza como una fracción fija de la caja del cuerpo (por ejemplo, el tercio
superior). **Se descartó:** con la cabeza a veces frontal, a veces de perfil y a veces baja
pastando, no hay una relación geométrica estable entre la caja del cuerpo y la ubicación de la
cara. Los keypoints resuelven esto directamente y además dan el eje necesario para alinear.

### Por qué dos puntos y no un conjunto grande de landmarks

Dos correspondencias determinan **exactamente** una similitud: cuatro grados de libertad —escala,
rotación y traslación en x e y— con cuatro ecuaciones. No hace falta mínimos cuadrados ni RANSAC.

Se eligen el **ojo** y el **hocico** porque:

- son los dos puntos faciales más robustos de detectar en un equino,
- el eje que definen coincide con el eje longitudinal de la cabeza, que es lo que hay que
  canonicalizar,
- siguen siendo visibles tanto de frente como de perfil, a diferencia del par ojo-ojo, que
  desaparece en perfil cerrado.

Es la misma idea que alinear rostros humanos por la línea entre los ojos, adaptada a una especie
cuya cara es alargada en vez de plana.

### Por qué mediana móvil y no promedio

Un promedio arrastra el outlier hacia el resultado: una caja disparatada corre los bordes de todas
las cajas de su ventana. La mediana lo descarta por completo. Y como el caballo y la cámara se
mueven de forma continua, la mediana conserva el desplazamiento sostenido —está verificado con una
prueba explícita sobre una rampa—, así que no se paga costo por la robustez.

### Por qué recortes cuadrados

La entrada de la red es cuadrada. Si la ROI no lo fuera, el reescalado deformaría la cara con un
factor distinto en cada eje, y esa deformación variaría según la pose de cada frame: se estaría
introduciendo exactamente la variabilidad que este cambio busca eliminar.

### Por qué se conservan los frames sin normalizar

Permite entrenar el mismo modelo con y sin normalización y comparar. Ese delta es un resultado
publicable en el capítulo 5 y, si diera negativo, sería evidencia de que el pipeline introduce un
problema, no una mejora.

## Estrategia de degradación

El pipeline nunca inventa datos. Cuando algo falla, lo registra y sigue:

| Situación | Respuesta | Marca en el CSV |
|---|---|---|
| Sin detección en un frame | Rellena con la detección válida más cercana del clip | `caja_rellenada` |
| Sin keypoints en un frame | ROI derivada de la caja, sin alinear | `alineado = False` |
| Sin detección en todo el clip | El clip queda excluido y se reporta | ausente del CSV |
| Ojo y hocico coincidentes | Frame no normalizable, se usa la ROI sin alinear | `alineado = False` |

## Verificación

Las funciones geométricas —suavizado, relleno, transformada de similitud y cálculo de ROI— se
mantienen **puras y sin dependencias de modelos**, y se prueban con casos sintéticos que no
requieren GPU ni pesos preentrenados. Están cubiertas las 21 propiedades relevantes, incluidos los
casos degenerados.

Lo que **no** se pudo verificar fuera de Colab es la calidad real de la detección y de los
keypoints, porque el entorno local no tiene acceso a los pesos preentrenados. Esa parte se valida
con el reporte de cobertura la primera vez que se ejecute en Colab.

## Parámetros

| Parámetro | Valor inicial | Justificación |
|---|---|---|
| `CONF_MINIMA_DETECCION` | 0.50 | Umbral estándar; se ajusta según lo que muestre la cobertura |
| `CONF_MINIMA_KEYPOINT` | 0.50 | Ídem, aplicado por keypoint |
| `VENTANA_SUAVIZADO` | 5 frames | Sobre 16 frames por clip, suaviza sin borrar la dinámica |
| `MARGEN_ROI` | 0.45 | Deja aire alrededor de la cara para no cortar orejas ni ollares |
| `LADO_RECORTE` | 224 px | Entrada estándar de los backbones preentrenados |
| `FRAC_OJO` / `FRAC_HOCICO` | 0.34 / 0.74 | Ubican la cabeza centrada verticalmente en el recorte |

Todos quedan como constantes en la primera celda para poder barrerlos sin tocar el resto.

## Alternativa si la cobertura resulta insuficiente

Anotación manual de una ROI por clip: 92 cajas, unas pocas horas de trabajo humano. Da una
referencia exacta y, además, sirve como conjunto de validación para medir cualquier detector
automático. **No es un fracaso del enfoque sino su red de contención**, y conviene decidirlo con el
reporte de cobertura sobre la mesa, no antes.
