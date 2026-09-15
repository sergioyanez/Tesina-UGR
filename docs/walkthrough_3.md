# Integración Inferencia Multimodal MIL

Se ha completado la actualización del pipeline de inferencia (`app_inferencia.py` y `equinos/inferencia.py`) para que utilice la nueva arquitectura **Multimodal MIL (Multiple Instance Learning)**.

> [!TIP]
> **Todo listo para probar**. Podés ejecutar `./correr_local.sh interfaz` en tu terminal para levantar la aplicación de Streamlit y probar el análisis con un video real.

## Cambios Implementados

### 1. Modelo de Producción en PyTorch
- **`equinos/clasificacion.py`**: Se refactorizó la función `entrenar_produccion` para que entrene el `MILFusionModel` sobre el dataset completo (sin validación cruzada) usando la misma rutina de PyTorch implementada en el notebook 03.
- Los pesos finales se guardaron exitosamente en `.pt` (en lugar del obsoleto `.npz` de scikit-learn).
- La función `cargar_produccion` fue actualizada para inicializar el modelo en GPU/CPU y cargar el estado (`state_dict`).

### 2. Pipeline de Inferencia
- **`equinos/inferencia.py`**:
  - Se modificó la extracción de keypoints (`_keypoints`) para que retorne los datos crudos en formato largo (`filas`), además de los pares de recorte.
  - La función `predecir_desde_caracteristicas` ahora acepta tanto los vectores visuales congelados como las filas de pose.
  - Se reutilizó el `MultimodalEquineDataset` pasándole un "clip dummy" (generado en el momento) para que aplique **exactamente la misma normalización geométrica y *validity mask*** usada en entrenamiento.
  - El modelo devuelve los **logits de la secuencia completa** y los **Pesos de Atención (Attention Weights)**.

### 3. Interfaz de Usuario (XAI)
- **`app_inferencia.py`**:
  - Se eliminó la sección que mostraba "Cómo quedó repartido el voto" ya que el nuevo modelo no vota por fotograma.
  - En su lugar, se agregó una sección de **Explicabilidad (XAI)** que utiliza `st.bar_chart` para graficar los **Pesos de Atención**. Esto permite entender rápidamente a qué fotogramas le prestó más atención la red neuronal para tomar su decisión.
  - Se actualizó el texto explicativo de la sección "Cómo se midió" para describir la naturaleza del modelo de secuencias (MIL).

## Validación
- Se ejecutó el script de entrenamiento de producción, confirmando la convergencia y el correcto guardado de `un_frame_produccion.pt`.
- El flujo de la tupla `Prediccion` fue ajustado sin romper la compatibilidad con el resto del código.
