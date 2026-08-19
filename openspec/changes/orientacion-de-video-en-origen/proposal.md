## Why

Contribuye al **objetivo específico 2** (ROI facial y normalización de pose), que estaba
bloqueado por una causa mal identificada.

La cobertura de alineación no llegaba al umbral de 70 % y la investigación apuntaba al keypoint
del hocico. El barrido de candidatos (19/08/2026) refutó esa hipótesis: `nose` ya era la mejor
opción disponible —`upper_jaw` da 54,4 % contra 62,4 %— y la estrategia de máxima confianza por
frame ganaba apenas 1,5 puntos. El bodypart movía 24 frames.

Medir la pérdida contra los frames **esperados** en vez de contra los generados destapó la causa
real. `cobertura()` informaba 68,1 %, pero dividía por frames generados: sobre los esperados era
51,9 %. Faltaban **400 frames (23,8 %) que nunca llegaron a recortarse**, y el cruce con los
metadatos de video fue inequívoco:

| rotación declarada | videos | frames | perdidos | tasa |
|---|---|---|---|---|
| 0° | 49 | 960 | 0 | **0,0 %** |
| 90° | 13 | 720 | 400 | **55,6 %** |

Los videos de celular guardan la orientación como metadato en lugar de rotar los píxeles, y
OpenCV entrega el frame como está almacenado salvo que se le pida lo contrario. El detector no
reconoce un caballo girado 90 grados, así que no produce caja; sin caja no hay recorte y el frame
no llega al dataset.

El daño excede esos 400 frames. Los 320 frames de videos rotados que **sí** pasaban la detección
también llegaban acostados al modelo de pose, lo que explica que los ejemplares dueños de esos
videos —`031`, `053`, `054`, `055`, `155`, `920`, `037`— estuvieran todos al fondo de la tabla de
cobertura, con `031` en 6,2 %.

## What Changes

- Toda lectura de video del pipeline pasa por un punto único que **aplica la orientación
  declarada** en los metadatos (`CAP_PROP_ORIENTATION_AUTO`).
- El inventario registra las dimensiones **de visualización** y no las almacenadas, que para un
  video rotado están intercambiadas.
- El `VideoWriter` de los clips se dimensiona con la forma del **primer frame real** en lugar de
  con las dimensiones del inventario, eliminando el acoplamiento entre ambos.
- La vista previa del Paso 3 usa el mismo punto de lectura, de modo que un frame acostado se vea
  en el primer minuto y no tres horas después como frames faltantes.

**No incluido:** el bodypart del hocico, que se decide en `pose-completa-y-bodypart-hocico` y
queda fijado en `nose` con su resultado negativo documentado.

## Capabilities

### New Capabilities

Ninguna.

### Modified Capabilities

- `preprocesamiento-facial`: se agrega que la lectura de video debe aplicar la orientación
  declarada en los metadatos, y que las dimensiones registradas deben ser las de visualización.
  Hoy ningún requisito lo menciona, y la consecuencia de no hacerlo no es un fallo sino una
  pérdida silenciosa concentrada en un subconjunto de ejemplares —que es peor, porque sesga.

## Impact

- **Notebooks:** `01_preparar_dataset_equinos.ipynb` (celdas de inventario, vista previa,
  segmentación y extracción de frames).
- **Paquete `equinos`:** `reporte.frames_perdidos()`, `reporte.orientacion_videos()` y
  `reporte.prueba_orientacion()`, que son las herramientas con que se diagnosticó y con las que se
  verifica que no reaparezca.
- **Requiere re-ejecutar el pipeline completo:** los clips ya generados tienen los píxeles
  acostados y ningún metadato que lo indique, así que no son recuperables sin regenerarlos.
- **Cambian todos los denominadores** de los reportes anteriores. Los números históricos
  —59,5 %, 68,1 %, 400 frames— dejan de ser comparables.
