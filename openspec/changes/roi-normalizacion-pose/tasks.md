# Tareas: ROI facial y normalización de pose

## 1. Funciones geométricas (sin dependencias de modelos)

- [x] 1.1 Implementar `suavizar_cajas` con mediana móvil sobre las cajas del clip
- [x] 1.2 Implementar `rellenar_faltantes` para los frames sin detección
- [x] 1.3 Implementar `similitud_desde_dos_puntos` a partir de dos correspondencias
- [x] 1.4 Implementar `destino_canonico` con las posiciones fijas de ojo y hocico
- [x] 1.5 Implementar `roi_desde_keypoints` con caja cuadrada, margen y recorte al borde
- [x] 1.6 Probar las 21 propiedades, incluidos los casos degenerados, sin GPU ni pesos

## 2. Notebook `02_roi_y_normalizacion.ipynb` — estructura

- [x] 2.1 Bloque 1: instalación de dependencias y constantes de configuración
- [x] 2.2 Cargar `inventario_clips.csv` y validar que existan los clips referenciados
- [x] 2.3 Incorporar las funciones geométricas ya probadas en una celda propia

## 3. Detección del equino

- [x] 3.1 Cargar el detector preentrenado e **imprimir el índice de la clase `horse`**
      verificándolo contra los nombres del modelo, sin hardcodear el número
- [x] 3.2 Detectar por frame, quedándose con la caja de mayor confianza
- [x] 3.3 Registrar cantidad de detecciones y confianza por frame
- [x] 3.4 Marcar como sin detección los frames bajo umbral, sin inventar cajas

## 4. Suavizado temporal

- [x] 4.1 Aplicar `suavizar_cajas` por clip, nunca cruzando clips distintos
- [x] 4.2 Rellenar huecos y contabilizar cuántos frames se rellenaron
- [x] 4.3 Excluir y reportar los clips sin ninguna detección válida

## 5. Keypoints faciales

- [x] 5.1 Cargar el modelo de pose para cuadrúpedos (DeepLabCut SuperAnimal-Quadruped)
- [x] 5.2 **Descubrir el esquema de keypoints en tiempo de ejecución** e imprimirlo completo
- [x] 5.3 Seleccionar el punto ocular y el del hocico a partir del esquema, dejando explícito
      cuáles se eligieron
- [x] 5.6 Adaptar la extracción a la estructura real confirmada en Colab: dict indexado por ruta,
      DataFrame con columnas MultiIndex `scorer/individuals/bodyparts/coords`
- [x] 5.7 Elegir el ojo por frame según `likelihood`, porque en perfil sólo uno es visible
- [x] 5.8 Borrar los auxiliares que deja DeepLabCut (`.h5`, `.pickle`, video etiquetado)
- [x] 5.4 Filtrar keypoints por confianza mínima
- [x] 5.5 Degradar a la ROI de la caja cuando falte el par ojo/hocico

## 6. ROI y alineación

- [x] 6.1 Calcular la ROI cuadrada con margen a partir de los keypoints disponibles
- [x] 6.2 Construir la transformada de similitud hacia las posiciones canónicas
- [x] 6.3 Aplicar con `cv2.warpAffine` y producir el recorte de 224×224
- [x] 6.4 Marcar `alineado = False` en los frames que no se pudieron normalizar

## 7. Salidas

- [x] 7.1 Guardar los frames alineados en `frames_alineados/<clase>/<clip>/`
- [x] 7.2 Generar `dataset_frames_alineados.csv` arrastrando `id_caballo`, `video_origen` y `clip`
- [x] 7.3 Incluir las columnas de calidad: confianza, `caja_rellenada`, `alineado`

## 8. Control de calidad

- [x] 8.1 Reportar cobertura de detección y de keypoints, global y por clase y por caballo
- [x] 8.2 Advertir explícitamente las clases o caballos bajo el umbral aceptable
- [x] 8.3 Mostrar una grilla de recortes alineados por clase para inspección visual
- [x] 8.4 Verificar visualmente que el ojo y el hocico caen en la misma posición entre recortes

## 9. Documentación

- [x] 9.1 Actualizar `openspec/project.md` con el estado del objetivo específico 2
- [x] 9.2 Actualizar el `README.md` con el nuevo notebook y su salida
- [x] 9.3 Registrar en `project.md` la cobertura obtenida la primera vez que corra en Colab

## 10. Decisión posterior a la primera corrida

- [x] 10.1 Evaluar el reporte de cobertura — 59,5 % global, por debajo del umbral de 70 %
- [ ] 10.2 Si es insuficiente, decidir entre ajustar umbrales o pasar a anotación manual de las
      92 ROI
