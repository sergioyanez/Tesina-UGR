# Tareas — interfaz de inferencia

Orden deliberado: **la negativa y la confiabilidad se construyen antes que la interfaz.** Son los
dos requisitos que el objetivo 4 puede incumplir sin que nada falle, y dejarlos para el final es
como terminan siendo una nota al pie.

## 1. El modelo de producción

- [ ] 1.1 Entrenar sobre los 141 clips íntegros con `random_state=42`, reutilizando
      `clasificacion.predecir_un_frame`, y persistirlo en `modelos/objetivo3/` con nombre que lo
      distinga de los cuatro de validación cruzada.
- [ ] 1.2 Registrar su procedencia junto a los pesos: cantidad de clips y ejemplares, corrida del
      preprocesamiento de la que salen las características, y la métrica de validación cruzada que
      le corresponde.
- [ ] 1.3 Verificar que sobre los 141 clips de entrenamiento predice mejor que 0,626 —tiene que
      hacerlo, los vio a todos— y **dejar escrito que ese número no se informa en ningún lado**.
      Es el control de que el modelo entrenó, no una métrica.

## 2. La confiabilidad, calculada desde los datos y no escrita a mano

- [ ] 2.1 Derivar la **precisión por clase** de las predicciones fuera de muestra guardadas en
      `objetivo3_evaluacion.json`, y exponerla desde `equinos/clasificacion.py`. Escribirla a mano
      la deja divergir del día que se re-evalúe.
- [ ] 2.2 Verificar contra la matriz de confusión: `Disgustado` 7/21, `Interesado` 49/62,
      `Sereno` 44/58.
- [ ] 2.3 Redactar el texto que acompaña a cada predicción —precisión de la clase, criterio no
      alcanzado, no es diagnóstico— y dejarlo en un solo lugar del código.

## 3. Inferencia de punta a punta

- [ ] 3.1 `equinos/inferencia.py`: de un video a un clip de 16 frames alineados, reutilizando las
      funciones de las etapas 01 y 02. **Sin reimplementar nada**: el encuadre y los umbrales
      salen de donde están declarados.
- [ ] 3.2 Devolver una **negativa explicada** cuando no se llegue a 16 frames alineados, con el
      motivo distinguido: sin detección, sin keypoints suficientes, video demasiado corto.
- [ ] 3.3 Extraer características con el backbone congelado y predecir, devolviendo la clase, las
      tres probabilidades y la precisión medida de la clase predicha.
- [ ] 3.4 Emitir señales de progreso por etapa, para que 35 s no parezcan un cuelgue.
- [ ] 3.5 Pruebas en `tests/test_inferencia.py`, sin GPU ni video: que la negativa se produzca con
      menos de 16 frames alineados, que distinga los motivos, y que la salida positiva traiga
      siempre la precisión de su clase.

## 4. Verificación contra el conjunto evaluado

- [ ] 4.1 Tomar un clip que YA está en el dataset, pasarlo por `inferencia` de punta a punta y
      comprobar que produce **las mismas características** que la caché. Si difieren, la interfaz
      está midiendo otra distribución y la confiabilidad informada no aplica.
- [ ] 4.2 Comprobar que la predicción de ese clip coincide con la que dio el fold en que estaba
      como test.
- [ ] 4.3 Medir la **tasa de negativa** sobre una muestra de videos crudos que no hayan entrado al
      dataset, e informarla. Si es muy alta, es un hallazgo sobre el preprocesamiento y va escrito
      como tal.

## 5. La interfaz

- [ ] 5.1 Elegir el framework y dejar escrito por qué. La lógica queda en `equinos/inferencia.py`:
      cambiar de framework no puede obligar a reimplementar nada.
- [ ] 5.2 Pantalla de carga: recibir el video, decir de antemano cuánto puede tardar y qué formatos
      admite.
- [ ] 5.3 Pantalla de resultado: clase, las tres probabilidades, **la precisión medida de la clase
      predicha con la misma prominencia que la etiqueta**, y el criterio no alcanzado.
- [ ] 5.4 Pantalla de negativa: el motivo, y dicho como resultado esperable y no como error.
- [ ] 5.5 Texto de alcance visible sin buscarlo: no es diagnóstico veterinario.
- [ ] 5.6 Decidir si la interfaz segmenta un video largo o exige un clip de 5–7 s, con el primer
      video de prueba real sobre la mesa.

## 6. Revisión de honestidad

- [ ] 6.1 **Mirar la pantalla de resultado y preguntarse si alguien que no leyó la tesina
      entendería que el modelo acierta un tercio de los `Disgustado`.** Si no, rehacerla. Es el
      riesgo principal declarado del change y merece una revisión propia.
- [ ] 6.2 Pasar la interfaz a una persona ajena al trabajo con un video, sin explicarle nada, y
      registrar qué entendió del resultado.
- [ ] 6.3 Corregir con lo que salga de 6.2, o dejar escrito por qué no se corrigió.

## 7. Documentación y cierre

- [ ] 7.1 `README.md`: cómo se levanta la interfaz, qué necesita y qué no hace.
- [ ] 7.2 `openspec/project.md`: estado del objetivo 4, con la tasa de negativa medida.
- [ ] 7.3 Dejar registrado el límite del entregable: es una demostración del pipeline, no una
      herramienta validada, y el motivo está medido —no alcanza el criterio y acierta un tercio de
      la clase de interés clínico—.
- [ ] 7.4 `openspec validate interfaz-de-inferencia` sin errores.
- [ ] 7.5 Commits separados por unidad lógica: modelo de producción; confiabilidad; inferencia;
      verificación; interfaz; documentación.
