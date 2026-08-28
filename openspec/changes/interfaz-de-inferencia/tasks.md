# Tareas — interfaz de inferencia

Orden deliberado: **la negativa y la confiabilidad se construyen antes que la interfaz.** Son los
dos requisitos que el objetivo 4 puede incumplir sin que nada falle, y dejarlos para el final es
como terminan siendo una nota al pie.

## 1. El modelo de producción

- [x] 1.1 Entrenar sobre los 141 clips íntegros con `random_state=42`, reutilizando
      `clasificacion.predecir_un_frame`, y persistirlo en `modelos/objetivo3/` con nombre que lo
      distinga de los cuatro de validación cruzada.
- [x] 1.2 Registrar su procedencia junto a los pesos: cantidad de clips y ejemplares, corrida del
      preprocesamiento de la que salen las características, y la métrica de validación cruzada que
      le corresponde.
- [x] 1.3 Verificar que sobre los 141 clips de entrenamiento predice mejor que 0,626. **Da F1
      macro 1,000**: memoriza el conjunto entero, que es lo esperable y la ilustración más clara
      de por qué una métrica de entrenamiento no se informa nunca. **Ese número no se informa en
      ningún lado**; es el control de que el modelo entrenó y se persistió bien.

## 2. La confiabilidad, calculada desde los datos y no escrita a mano

- [x] 2.1 Derivar la **precisión por clase** de las predicciones fuera de muestra guardadas en
      `objetivo3_evaluacion.json`, y exponerla desde `equinos/clasificacion.py`. Escribirla a mano
      la deja divergir del día que se re-evalúe.
- [x] 2.2 Verificar contra la matriz de confusión: `Disgustado` 7/21, `Interesado` 49/62,
      `Sereno` 44/58. **Las tres coinciden.** La verificación destapó además un error en el
      diseño: comparaba la precisión agregada contra el recall promediado por folds. En
      `Disgustado` precisión y recall coinciden (7/21 las dos), así que es la peor clase para
      ilustrar la distinción; se corrigió usando `Interesado` y `Sereno`, donde difieren en
      direcciones opuestas.
- [x] 2.3 Redactar el texto que acompaña a cada predicción —precisión de la clase, criterio no
      alcanzado, no es diagnóstico— y dejarlo en un solo lugar del código.

## 3. Inferencia de punta a punta

- [x] 3.1 `equinos/inferencia.py`: de un video a un clip de 16 frames alineados, reutilizando las
      funciones de las etapas 01 y 02. **Sin reimplementar nada**: el encuadre y los umbrales
      salen de donde están declarados.
- [x] 3.2 Devolver una **negativa explicada** cuando no se llegue a 16 frames alineados, con el
      motivo distinguido: sin detección, sin keypoints suficientes, video demasiado corto.
- [x] 3.3 Extraer características con el backbone congelado y predecir, devolviendo la clase, las
      tres probabilidades y la precisión medida de la clase predicha.
- [x] 3.4 Emitir señales de progreso por etapa, para que 35 s no parezcan un cuelgue.
- [x] 3.5 Pruebas en `tests/test_inferencia.py`, sin GPU ni video: que la negativa se produzca con
      menos de 16 frames alineados, que distinga los motivos, y que la salida positiva traiga
      siempre la precisión de su clase.

## 4. Verificación contra el conjunto evaluado

- [x] 4.1 Tomar un clip que YA está en el dataset y comprobar que produce **las mismas
      características** que la caché. **3 de 3 clips: diferencia exactamente 0.**
      `verificar_inferencia.py`. Destapó dos bugs que no daban error: DeepLabCut devuelve diez
      `individuals` y me quedaba con el vacío; y **faltaba el viaje por JPEG** —el conjunto
      evaluado pasó por compresión con pérdida y extraer del tensor en memoria daba
      características 9,4 % distintas—.
- [x] 4.2 ~~Comprobar que la predicción coincide con la del fold en que estaba como test.~~
      **La tarea estaba mal planteada y se deja constancia:** el modelo de producción vio ese clip
      al entrenarse, así que lo acierta por memorización; el del fold nunca lo vio. Son modelos
      distintos y difieren con razón. Lo que valida la cadena es la igualdad de características de
      la 4.1, no una coincidencia de predicciones.
- [x] 4.3 Medir la **tasa de negativa** sobre videos crudos e informarla. **Respondió en 5 de 12
      (42 %)**, muestra estratificada por clase. Coincide con lo que el diseño anticipó a partir
      del 38 % de clips íntegros. Motivos: alineación insuficiente 2, video largo 2, sin detección
      2, video corto 1. `tasa_de_negativa.py`, con el sesgo declarado en su encabezado: los videos
      contribuyeron al entrenamiento, lo que no afecta la tasa pero sí haría optimista cualquier
      medición de acierto — por eso no se informa ninguna.

## 5. La interfaz

- [x] 5.1 Elegir el framework y dejar escrito por qué. **Streamlit 1.62.0.** Verificado antes de
      adoptarlo que no toca el conjunto pineado: agrega 15 paquetes y deja numpy 1.26.4, torch
      2.5.1+cu121, pandas y scipy donde estaban. Era el riesgo real —subir numpy por encima de 2
      rompe DeepLabCut—. Fijado en `requirements-local.txt`.
- [x] 5.2 Pantalla de carga: recibir el video, decir de antemano cuánto puede tardar y qué formatos
      admite.
- [x] 5.3 Pantalla de resultado: clase, las tres probabilidades, **la precisión medida de la clase
      predicha con la misma prominencia que la etiqueta**, y el criterio no alcanzado.
- [x] 5.4 Pantalla de negativa: el motivo, y dicho como resultado esperable y no como error.
- [x] 5.5 Texto de alcance visible sin buscarlo: no es diagnóstico veterinario. Arriba de todo, en rojo,
      antes del cargador.
- [x] 5.6 Decidir si la interfaz segmenta un video largo o exige un clip de 5–7 s. **Exige una
      toma corta**, con un máximo de 15 s y un motivo de negativa propio (`VIDEO_LARGO`). Los 16
      frames se muestrean repartidos por todo el video: sobre uno de dos minutos darían fotogramas
      separados por ocho segundos, que es una colección de momentos y no una secuencia de
      expresión.

## 6. Revisión de honestidad

- [x] 6.1 **Mirar la pantalla** y preguntarse si alguien que no leyó la tesina entendería el
      acierto real. Hecho sobre la aplicación corriendo, con las tres pantallas: inicial, negativa
      y resultado. **Pasa**: la etiqueta y la precisión salen en columnas del mismo tamaño, y cada
      barra del voto lleva la precisión de su clase —«Disgustado — 0 % (acierta 33 % cuando lo
      dice)»—. **Destapó un defecto real**: la barra de progreso no se veía, así que durante 35 s
      el usuario sólo tenía un spinner. `st.progress` no renderiza mientras DeepLabCut bloquea el
      hilo; se reemplazó por `st.status` y se verificó en vivo que ahora sí muestra la etapa.
- [ ] 6.2 Pasar la interfaz a una persona ajena al trabajo con un video, sin explicarle nada, y
      registrar qué entendió del resultado. **Requiere a una persona: la hace Sergio.**
- [ ] 6.3 Corregir con lo que salga de 6.2, o dejar escrito por qué no se corrigió.

## 7. Documentación y cierre

- [x] 7.1 `README.md`: cómo se levanta la interfaz, qué necesita y qué no hace.
- [x] 7.2 `openspec/project.md`: estado del objetivo 4, con la tasa de negativa medida (42 % de
      respuesta).
- [x] 7.3 Dejar registrado el límite del entregable: es una demostración del pipeline, no una
      herramienta validada, y el motivo está medido —no alcanza el criterio y acierta un tercio de
      la clase de interés clínico—.
- [x] 7.4 `openspec validate interfaz-de-inferencia` sin errores.
- [ ] 7.5 Commits separados por unidad lógica: modelo de producción; confiabilidad; inferencia;
      verificación; interfaz; documentación.
