# Tareas: persistencia de artefactos y orquestación en Colab

## 1. Paquete `equinos/` — resolución de rutas

- [ ] 1.1 Crear el paquete `equinos/` con `__init__.py`
- [ ] 1.2 `rutas.py`: detectar el entorno (Colab vs local) sin que el notebook lo declare
- [ ] 1.3 `rutas.py`: montar Drive en Colab y resolver la raíz de proyecto
- [ ] 1.4 `rutas.py`: crear las subcarpetas de artefactos que falten
- [ ] 1.5 `rutas.py`: fallar de inmediato si la raíz no admite escritura
- [ ] 1.6 `rutas.py`: `resolver()` y `relativizar()` para las rutas guardadas en los CSV
- [ ] 1.7 `rutas.py`: informar la ruta absoluta resuelta al arrancar

## 2. Paquete `equinos/` — checkpointing

- [ ] 2.1 `checkpoint.py`: leer y escribir el estado por etapa en JSON
- [ ] 2.2 `checkpoint.py`: escritura atómica (temporal + rename) para resistir un corte
- [ ] 2.3 `checkpoint.py`: calcular `hash_config` sobre las claves que afectan la etapa
- [ ] 2.4 `checkpoint.py`: invalidar el checkpoint e informar qué claves cambiaron
- [ ] 2.5 `checkpoint.py`: anexar filas al CSV parcial por unidad completada
- [ ] 2.6 `checkpoint.py`: devolver las unidades pendientes dado el universo de trabajo

## 3. Paquete `equinos/` — caché comprimida

- [ ] 3.1 `cache.py`: empaquetar un directorio local a `cache/<etapa>.zip` sin compresión
- [ ] 3.2 `cache.py`: escribir el manifiesto con conteo de archivos y `hash_config`
- [ ] 3.3 `cache.py`: expandir al disco local verificando el conteo
- [ ] 3.4 `cache.py`: tratar la caché como inválida si el conteo no coincide

## 4. Paquete `equinos/` — registro de etapas

- [ ] 4.1 `etapas.py`: declarar las etapas del pipeline, su notebook y sus dependencias
- [ ] 4.2 `etapas.py`: predicado de completitud por etapa sobre sus artefactos de salida
- [ ] 4.3 `etapas.py`: estado pendiente / parcial / completa, con unidades restantes
- [ ] 4.4 `etapas.py`: limpieza explícita y dirigida de una etapa, informando qué borra

## 5. Pruebas locales (sin GPU ni pesos)

- [ ] 5.1 Rutas: resolución en modo local y `resolver()`/`relativizar()` como inversas
- [ ] 5.2 Rutas: error claro cuando la raíz no admite escritura
- [ ] 5.3 Checkpoint: ciclo guardar / releer / reanudar sobre unidades pendientes
- [ ] 5.4 Checkpoint: cambiar una clave relevante invalida; cambiar una irrelevante no
- [ ] 5.5 Checkpoint: un archivo truncado no destruye el estado anterior
- [ ] 5.6 Caché: empaquetar y expandir preserva el conteo y el contenido
- [ ] 5.7 Caché: zip con conteo alterado se detecta como inválido
- [ ] 5.8 Etapas: los tres estados se reportan correctamente
- [ ] 5.9 Verificar que las 21 pruebas de geometría siguen pasando

## 6. Celda de arranque (bootstrap)

- [ ] 6.1 Escribir la celda autocontenida: monta Drive y resuelve `sys.path`
- [ ] 6.2 Cascada de resolución del paquete: Drive → clone con token → error instructivo
- [ ] 6.3 Documentar cómo cargar `GITHUB_TOKEN` en los secrets de Colab
- [ ] 6.4 Insertar la celda como primer paso de `01` y `02`

## 7. Adaptar `01_preparar_dataset_equinos.ipynb`

- [ ] 7.1 Reemplazar el Paso 0 destructivo por limpieza explícita y dirigida
- [ ] 7.2 Sustituir la descarga con `gdown` por lectura de los videos montados en Drive
- [ ] 7.3 Fallar con mensaje claro si la carpeta de videos está vacía o no existe
- [ ] 7.4 Apuntar clips y CSV a la raíz de proyecto persistente
- [ ] 7.5 Guardar rutas **relativas** en los tres CSV
- [ ] 7.6 Persistir los frames extraídos como caché comprimida
- [ ] 7.7 Agregar la celda `parameters` con los valores por defecto

## 8. Adaptar `02_roi_y_normalizacion.ipynb`

- [ ] 8.1 Sustituir las rutas literales `/content` del Paso 2 por las del paquete
- [ ] 8.2 Resolver las rutas de `inventario_clips.csv` con `resolver()`
- [ ] 8.3 Envolver el bucle por clip con checkpointing reanudable
- [ ] 8.4 Declarar qué parámetros entran en el `hash_config` de esta etapa
- [ ] 8.5 Persistir los frames alineados como caché comprimida
- [ ] 8.6 Agregar la celda `parameters` con los umbrales del preprocesamiento
- [ ] 8.7 Verificar que la lógica geométrica quedó intacta

## 9. Orquestador `00_pipeline.ipynb`

- [ ] 9.1 Celda de arranque e instalación de `papermill`
- [ ] 9.2 Celda única de parámetros compartidos entre etapas
- [ ] 9.3 Diagnóstico del estado del pipeline sin ejecutar nada
- [ ] 9.4 Ejecución encadenada con salteo de etapas completas
- [ ] 9.5 Corte de la cadena ante fallo, sin ejecutar las dependientes
- [ ] 9.6 Guardar la copia ejecutada por etapa, con sufijo `_FALLIDA` si corresponde
- [ ] 9.7 Opción de reejecución forzada de una etapa concreta
- [ ] 9.8 Resumen final: qué se ejecutó, qué se salteó, cuánto tardó cada etapa

## 10. Documentación

- [ ] 10.1 Actualizar `openspec/project.md`: disposición en Drive y rutas ya no efímeras
- [ ] 10.2 Actualizar el diagrama del pipeline en `project.md` con las rutas nuevas
- [ ] 10.3 Actualizar `README.md`: cómo correr el pipeline desde el orquestador
- [ ] 10.4 Documentar la preparación inicial de la carpeta de Drive
- [ ] 10.5 Agregar `equinos/` al stack técnico de `project.md`

## 11. Cierre

- [ ] 11.1 Verificar la cuota disponible en Drive antes de la primera corrida completa
- [ ] 11.2 Primera corrida real en Colab desde el orquestador
- [ ] 11.3 Confirmar que una interrupción deliberada del `02` reanuda correctamente
- [ ] 11.4 Cerrar las tareas 9.3, 10.1 y 10.2 de `roi-normalizacion-pose` con la cobertura obtenida
