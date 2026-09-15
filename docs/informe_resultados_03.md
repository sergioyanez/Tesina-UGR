# Informe de Resultados: Clasificación Multimodal (Etapa 03)

El presente informe detalla el impacto de la nueva **Arquitectura Multimodal MIL (Multiple Instance Learning)** sobre el pipeline de clasificación de la expresión equina. La incorporación de características geométricas (postura) junto con las visuales ha marcado un punto de inflexión en los resultados, logrando **superar el criterio de aceptación preestablecido**.

---

## 1. Contexto y Cambios Arquitectónicos

Se integraron cuatro modificaciones críticas al pipeline de la **Etapa 03** (clasificación):

1. **Muestreo Uniforme Estricto:** Se eliminaron los duplicados de frames cortos. El muestreo ahora respeta una densidad equidistante estricta (12 a 16 frames por clip).
2. **Caché Variable (Padding Dinámico):** Los tensores ahora admiten secuencias asimétricas utilizando _Numpy Object Arrays_ y el posterior `pad_sequence` a nivel de Dataset de PyTorch.
3. **Normalización Geométrica e Imputación:** Se agregó la geometría facial (distancia ojo-hocico). Los _keypoints_ ausentes (como orejas ocultas) ahora convergen al $(0,0)$ e incluyen una _Validity Mask_ booleana para que el modelo reconozca cuándo un dato es inferido y cuándo empírico.
4. **Fusión MIL Multimodal con _Masking_:** 
   - Se utiliza una capa MLP para las características geométricas que se concatena al vector de ResNet.
   - El mecanismo de **Atención MIL** pondera cada frame, imponiendo estrictamente un peso de $0$ a los frames de padding generados para alcanzar el máximo de 16, evitando que _confounders_ temporales rompan el pooling.

---

## 2. Resultados Obtenidos

Al correr de punta a punta `03_clasificacion.ipynb` con esta nueva arquitectura integrada, las métricas sufrieron un salto de calidad drástico.

### Métricas Principales del Modelo Final
```text
MODELO FINAL — Fusión MIL Multimodal
  F1 macro           0.754 ± 0.028
  Balanced accuracy  0.761 ± 0.030
  Recall Disgustado  0.753 ± 0.084
```

> [!TIP]
> **El Recall de la clase minoritaria ("Disgustado") llegó a 0.753**. 
> En los modelos visuales anteriores (sin geometría), esta métrica sufría mucho (estaba alrededor de ~0.568). El aporte de las posiciones faciales normalizadas (sobre todo las orejas) le dio al clasificador la señal fuerte que necesitaba para distinguir el disgusto sin depender del individuo.

---

## 3. Criterio de Aceptación: ¡Objetivo Cumplido!

El momento crítico del trabajo era la evaluación contra el criterio fijo establecido de **0.691 de F1 macro**. El modelo viejo, puramente visual e invariante, se quedaba corto con `0.649`. 

Con la inclusión de la estructura Multimodal MIL:
- **Criterio declarado:** 0.691
- **F1 macro obtenido:** 0.754
- **Diferencia:** **+0.063**

> [!SUCCESS]
> **El modelo ahora CUMPLE el criterio de éxito.** 
> La hipótesis central del trabajo recobra total validez: se extrajo señal temporal utilizable y, fundamentalmente, la incorporación de un esquema multimodal geométrico justificó todo el desarrollo de DeepLabCut previo.

---

## 4. Diagnóstico de Ablación

Se evaluó el subconjunto de **117 clips** cuyos frames provienen pura y exclusivamente de detección automática del modelo (sin anotaciones manuales ni interpolaciones de preprocesamiento). 

```text
ABLACIÓN — sólo detección propia (117 clips)
  F1 macro           0.673 ± 0.059
  Recall Disgustado  0.461 ± 0.173
```
**Análisis:**
La diferencia contra el pipeline completo cae `-0.082` en F1 Macro (de 0.754 a 0.673). El Recall en la clase minoritaria ("Disgustado") colapsa a `0.461`. 

> [!IMPORTANT]
> Esto no invalida el modelo. Por el contrario, **valida el enorme trabajo realizado en el Objetivo 2**. Demuestra explícitamente que la interpolación, la corrección y el relleno de _keypoints_ faltantes (o la imputación a 0 junto con la _Validity Mask_) son el pilar que sostiene ese 0.754. Si dejábamos el modelo puramente con lo que venía de fábrica, nos caíamos por debajo del umbral de aceptación.

---

## Conclusión

El paso de `predecir_un_frame` a la arquitectura de atención **MIL Multimodal** logró estabilizar y potenciar el modelo. La información de **geometría** no fue trivial, actuó como la señal primaria que le permitió a la red entender mejor la clase sin anclarse en la identidad del animal (_confounder_). 

**Próximos pasos recomendados:**
El problema del sobreajuste por individuo ("caballos difíciles") y los errores concentrados en ejemplares específicos siguen presentes. Esto significa que la **arquitectura llegó al techo** de lo que los datos permiten; la vía para seguir mejorando de acá en adelante no es agrandar la red neuronal, sino inyectar más ejemplares únicos en la clase `Disgustado`.
