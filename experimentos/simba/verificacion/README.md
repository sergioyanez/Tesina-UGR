# Verificación adversarial del F1 0,851 (RF sobre features tipo SimBA)

Reproducir: `cd experimentos/simba && python verificacion/verificar.py` (venv `tesina-equinos`; ~10 min). Salida: `verificacion.json`.
El script reutiliza `evaluar.oof` y el bootstrap vigente, y reimplementa el bootstrap por separado.

| # | Punto | Veredicto | Evidencia |
|---|---|---|---|
| 1 | Partición | OK | Asserts re-ejecutados: fold == `particion.construir`, ningún caballo ni video en train y test. El grupo es el número de prefijo del archivo (`id_caballo`; 29 caballos tienen varias clases, 244 videos para 133 caballos). No se puede probar que no haya un mismo animal con dos ids; heurística de nombres y vecino más cercano: sin duplicados evidentes (vecino más cercano de otro caballo: 74 % misma clase entre folds). Límite no resuelto. |
| 2 | Preprocesamiento | OK | Features por clip sin estadística global (escala = mediana del propio clip); imputación por mediana dentro del pipeline por fold; sin selección de features ni de hiperparámetros; umbral 0,5 fijo. Etiquetas permutadas: F1 0,29-0,32 (control negativo correcto). |
| 3 | Filmación | ALERTA | Todas las features son ratios, ángulos o velocidades normalizadas; sin píxeles absolutos. Pero: (e) desde las 160 features se predice log-duración con R2 0,53 y resolución 1080x1920 con AUC 0,73 (no «casi perfecto», sí informativo; solo estáticas: R2 0,08, AUC 0,75). Metadatos solos 0,574; features+metadatos 0,842 (no suben). Solo disponibilidad 0,490. (a) Solo estáticas 0,826; (b) sin disponibilidad 0,848, solo dinámicas 0,726. (c) Residualizando todas las features por log-duración, fps, resolución y proporción (ajustado en train del fold): **0,777**. Por estrato (OOF base): dur >= 5 s 0,765; dur < 5 s 0,877 (71 de 95 son Disgustado); limpio 0,750; entrenado y evaluado solo en limpio 0,678. |
| 4 | Contexto | OK con matiz | Solo cara (orejas, ojos, boca, 122 feats): 0,848; cara estática 0,828; orejas 0,784; boca 0,648. Cabeza+cuello (29): 0,793; estáticas de cabeza+cuello 0,712; cuello solo 0,461. La señal está en la cara, no en el cuello; pero la inclinación/orientación de cabeza sí aporta (0,71), y el punto de vista/establecimiento no se puede descartar con estos datos. |
| 5 | Comparabilidad | OK | Mismos 359 clips, mismas etiquetas y caballos (asserts). MIL recomputado con la misma función: 0,7396 = 0,740; mismo cálculo (media de 4 folds, desvío ddof=1). Por clase RF 0,85/0,83/0,87 vs MIL 0,71/0,78/0,73. Asimetría: el MIL tiene early stopping con validación interna y no se afinó; el RF no se afinó pero tampoco tuvo búsqueda. |
| 6 | Semillas | ALERTA leve | Semillas 1-5: 0,839 / 0,842 / 0,836 / 0,842 / 0,844; media 0,841 ± 0,003. La semilla 0 (0,851) es la más alta: corresponde reportar ~0,84. |
| 7 | Bootstrap | OK | Remuestrea 133 caballos arrastrando sus clips, diferencia pareada por réplica. Reimplementación independiente (sklearn): +0,111 [0,045; 0,186]; bootstrap de clips (incorrecto) daría [0,055; 0,165], es decir un IC similar. |

## Conclusión
El 0,851 no es un artefacto de fuga: partición idéntica, sin ajuste global, comparación pareada válida, control de etiquetas permutadas en azar. Se sostiene **con reservas**:
- Cifra robusta a semilla: 0,841 ± 0,003 (no 0,851).
- Parte de la ventaja se apoya en la filmación/duración: Disgustado dura 3,2 s de mediana contra 6,0 s de las otras clases, y las features codifican duración (R2 0,53). Residualizando la filmación el F1 baja a 0,777 (sigue sobre el MIL 0,740, pero la diferencia de ~0,04 ya no es concluyente). Esa residualización también puede quitar señal legítima (la duración correlaciona con la clase), así que 0,777 es un piso, no una estimación.
- En el subconjunto limpio: 0,750 vs MIL 0,705, diferencia no concluyente (README principal).
- El MIL comparte el mismo confusor de duración/resolución, así que la comparación es justa pero ambos están inflados.
- Puntos sin resolver: identidad real del animal (mismo animal con ids distintos), punto de vista y establecimiento.
