# Tareas — ejecución local con GPU

## 1. Entorno local

- [x] 1.1 Crear el entorno aislado con Python 3.10 fuera del repo y del disco exFAT
- [x] 1.2 Instalar `torch 2.5.1+cu121` y verificar `torch.cuda.is_available()` contra el driver 535
- [x] 1.3 Instalar DeepLabCut 3.0.1 y el resto del stack, verificando que conviven en un mismo
      intérprete
- [x] 1.4 Prueba de humo sobre un clip real: detector, pose, bodyparts faciales, VRAM y tiempos
- [x] 1.5 Declarar el conjunto verificado en `requirements-local.txt`
- [x] 1.6 Registrar el kernel `tesina-equinos` para papermill y Jupyter

## 2. Notebooks

- [x] 2.1 `01_preparar_dataset_equinos.ipynb`: condicionar el Paso 1 a `EN_COLAB`; fuera de Colab
      verificar versiones en vez de instalar
- [x] 2.2 `02_roi_y_normalizacion.ipynb`: idem en su Paso 1, conservando el aviso de GPU ausente
- [x] 2.3 `00_pipeline.ipynb`: condicionar la instalación del Paso 0 y elegir el kernel según el
      entorno, verificando que existe antes de lanzar la primera etapa
- [x] 2.4 Verificar que las tres celdas de arranque siguen funcionando en Colab sin cambios de
      comportamiento — sólo comprobable ejecutando en Colab; en local se verificó que la rama
      `EN_COLAB` es la única que instala y que el resto del arranque no cambió

## 3. Configuración y datos

- [x] 3.1 Documentar `TESINA_EQUINOS_RAIZ` y `TESINA_EQUINOS_VIDEOS` para la ejecución local
- [x] 3.2 Copiar `tesina_equinos/` de Drive a la raíz local (clips, resultados, cache, checkpoints)
- [x] 3.3 Correr `etapas.diagnostico()` sobre la raíz local y comprobar que reconoce las etapas
      completas, sin reprocesar nada

## 4. Verificación

- [x] 4.1 Comprobar que un CSV producido en Colab resuelve sus rutas en local (`rutas.resolver`)
- [x] 4.2 Re-derivar los recortes con el encuadre nuevo en local (notebook 02, `LIMPIAR_ETAPA =
      False`) y confirmar `Clips pendientes: 0 de 210`
- [x] 4.3 Medir `reporte.calidad_recortes()` y `reporte.cobertura()` y comparar con los números de
      Colab: la migración no debe mover ninguno
- [x] 4.4 `openspec validate ejecucion-local-con-gpu --strict`

## 5. Documentación

- [x] 5.1 `README.md`: sección de ejecución local (entorno, variables, kernel, artefactos)
- [x] 5.2 `openspec/project.md`: registrar los dos entornos y actualizar el punto de partida
- [x] 5.3 Dejar asentado el resultado negativo útil: la GPU no era inservible, el `torch` era el
      equivocado

## 6. Resultados de la migración (22/08/2026)

Copiados de Drive 6,2 GiB (clips, resultados, cache, checkpoints), sin `videos/`. La corrida del
notebook 02 con papermill y el kernel local tardó **14 min**, no los 30-45 estimados.

| medición | Colab | local | |
|---|---:|---:|---|
| Clips procesados | 210 | 210 | = |
| Frames generados | 3360 (100 %) | 3360 (100 %) | = |
| Alineados por keypoints | 68,0 % | **68,0 %** | = |
| Recortes mayormente vacíos (>50 % relleno) | 97,5 % | **0,2 %** | encuadre corregido |

Los números que debían quedarse quietos se quedaron quietos, y el que debía moverse se movió: la
migración no alteró resultados y el change `encuadre-canonico-y-relleno` queda **verificado**.

**Hallazgo no buscado:** el espejo local `<repo>/videos` no es `Videos de prueba`. Comparten 135
nombres de archivo, al espejo le faltan 11 y le sobran 12. Regenerar clips desde ahí habría
producido un dataset distinto del que generó la pose, sin fallar. Documentado en el README.


## Cierre — 28/08/2026

**2.4 verificada ejecutando en Colab**, que era la única forma de comprobarla. Con la rama
`ejecucion-local-gpu` ya en el remoto y el clone apuntando a ella:

- el sync trajo el commit correcto — `Versión del código en Drive: 5c66ba1`, coincidente con el
  remoto, que es la comprobación que evita correr una versión vieja sin enterarse;
- las dependencias se instalaron sin romper el stack: los conflictos que informa pip son los
  esperados —DeepLabCut fija `numpy 1.26.4` y los paquetes preinstalados de Colab piden numpy 2—
  y no apareció ni `numpy.dtype size changed` ni `No module named 'numpy.strings'`, que son los
  síntomas de que el pin falló;
- GPU visible: Tesla T4 de 15,6 GB con torch 2.11.0+cu128;
- la carpeta compartida de videos se resolvió sola vía el acceso directo en Mi unidad.

La rama `EN_COLAB` sigue siendo la única que instala, y el resto del arranque se comporta igual
que en local.

**Dos cosas que la corrida destapó y que no son de este change:**

1. **Drive y el espejo local divergen: el `501_8v1_LY1ucwR9.avi` sigue en Drive en `Disgustado`
   y en `Interesado`.** Colab cataloga 251 videos contra los 249 de local, y los dos de
   diferencia son ese archivo contado dos veces. Se retiró del espejo local el 25/08 pero nunca
   de Drive. Hay que sacarlo de una de las dos clases **antes** de que alguien corra el pipeline
   allá; la decisión de cuál es de los autores.
2. El volcado crudo del `Inventario` tapaba el reporte legible. Corregido asignando el retorno
   de `informe()` a una variable, que suprime el auto-display de la última expresión.
