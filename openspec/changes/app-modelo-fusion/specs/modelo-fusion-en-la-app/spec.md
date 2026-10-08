## ADDED Requirements

### Requirement: Modelo final entrenado con todos los clips y persistido con metadatos

El script de entrenamiento final DEBE (MUST) ajustar `FusionPoseVisual` con todos los clips íntegros (desarrollo y reserva) y persistir en `<raíz>/modelos/fusion_pose_visual/` un paquete `joblib` con el modelo de pose, el modelo visual, el peso `w` elegido, las clases y metadatos: versión de scikit-learn, huella del dataset, fecha y lista de clips. NO DEBE (MUST NOT) llamar a `evaluar_en_reserva` ni calcular una métrica sobre los clips con que se entrenó.

#### Scenario: entrenamiento final
- **WHEN** se corre `comparacion/entrenar_modelo_final.py`
- **THEN** el paquete contiene `peso` en {0; 0,25; 0,5; 0,75; 1}, la huella del dataset y los 359 clips, y el log de la reserva no cambia

#### Scenario: versión de sklearn distinta
- **WHEN** la versión de scikit-learn instalada difiere de la registrada
- **THEN** la carga emite una advertencia que nombra ambas versiones

### Requirement: La fusión es el modelo por omisión de la inferencia

`clasificar_video` DEBE (MUST) usar la fusión pose + visual cuando no se pide otro modelo, y DEBE (MUST) conservar la opción explícita `modelo="mil"` que usa el MIL multimodal sin cambios. La salida DEBE (MUST) mantener los campos `clase`, `probabilidades` (las tres clases, suma 1), `precision`, `acertadas`, `veces_que_lo_dijo`, `texto_confiabilidad` y `frames_alineados`, y la negativa DEBE (MUST) seguir siendo la misma `Negativa`.

#### Scenario: sin pedir modelo
- **WHEN** se llama `clasificar_video(ruta)`
- **THEN** la predicción lleva `modelo == "fusion"` y probabilidades por clase

#### Scenario: MIL explícito
- **WHEN** se llama `clasificar_video(ruta, modelo="mil")`
- **THEN** se ejecuta el MIL y la predicción conserva `atencion` y `clases_por_frame`

#### Scenario: modelo desconocido
- **WHEN** se pide un modelo que no es "fusion" ni "mil"
- **THEN** levanta `ValueError`

### Requirement: La confiabilidad informada corresponde al modelo servido

La precisión por clase y las métricas globales que acompañan a una predicción de la fusión DEBEN (MUST) leerse de los resultados de la comparación (`comparacion/resultados/fusion_tardia_pose_visual.json` y `..._reserva.json`) y NO DEBEN (MUST NOT) estar escritas a mano ni provenir de la evaluación del MIL. El texto DEBE (MUST) informar el F1 macro de la reserva, el del subconjunto limpio y que no es un diagnóstico veterinario.

#### Scenario: precisión por clase
- **WHEN** se pide la confiabilidad de la fusión
- **THEN** cada clase trae precisión, aciertos y veces que lo dijo, derivados de las predicciones fuera de muestra de desarrollo

#### Scenario: texto de confiabilidad
- **WHEN** la fusión predice una clase
- **THEN** el texto contiene el F1 macro de la reserva leído del archivo y la frase «NO es un diagnóstico veterinario»

### Requirement: La app presenta el modelo vigente sin atribuirle rasgos del MIL

`app_inferencia.py` DEBE (MUST) nombrar el modelo que ejecuta, tomar las métricas de la confiabilidad del modelo servido y mostrar el gráfico de atención sólo cuando la predicción trae atención.

#### Scenario: predicción de la fusión
- **WHEN** la predicción no trae pesos de atención
- **THEN** la app no dibuja el gráfico de atención y muestra las probabilidades de cada rama
