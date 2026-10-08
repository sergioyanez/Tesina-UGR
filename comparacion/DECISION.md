# Decisión: ganador de la comparación de modelos

Fecha: 08/10/2026. Registrada ANTES de usar la reserva, como exige `PROTOCOLO.md` §6.

## Resultados en desarrollo (5 repeticiones × 4 folds por caballo, 113 caballos, 291 clips)

| Candidato | Rama | F1 macro | F1 limpio | Gap train−test |
|---|---|---|---|---|
| Fusión tardía pose + visual | `feat/candidato-fusion` | 0,868 ± 0,046 | 0,740 ± 0,028 | +0,13 |
| RF de pose tipo SimBA | `feat/candidato-rf-pose` | 0,844 ± 0,059 | 0,718 ± 0,020 | +0,15 |
| MIL multimodal corregido | `feat/candidato-mil` | 0,722 ± 0,102 | 0,618 ± 0,072 | +0,19 |
| Baseline LogReg media de frames | `feat/candidato-baseline` | 0,694 ± 0,062 | 0,597 ± 0,050 | +0,31 |

Los cuatro comparten la huella de particiones `089b347d4d…` y de reserva `3902d7fa22…`.

## IC95 bootstrap pareado por caballo (2 000 réplicas, semilla 7)

Detalle completo en `resultados/comparaciones_pareadas.json`.

| Diferencia | Todo el desarrollo | Subconjunto limpio |
|---|---|---|
| fusión − RF de pose | +0,026 [+0,002; +0,051] | +0,022 [−0,017; +0,056] |
| fusión − MIL | +0,147 [+0,100; +0,193] | +0,122 [+0,020; +0,183] |
| fusión − baseline | +0,166 [+0,117; +0,220] | +0,143 [+0,065; +0,213] |
| RF de pose − MIL | +0,121 [+0,070; +0,172] | +0,100 [−0,003; +0,168] |
| RF de pose − baseline | +0,140 [+0,079; +0,202] | +0,121 [+0,031; +0,210] |

## Aplicación del criterio (§5)

1. Ranking: la fusión tiene el mayor F1 macro medio (0,868).
2. No inferioridad en limpio: ningún IC de «fusión − otro» en limpio tiene su límite superior por
   debajo de 0. Se cumple.
3. Empate: el IC de «fusión − RF» en desarrollo es [+0,002; +0,051] y no incluye 0. No hay empate
   y no se aplica la regla de simplicidad.

**Ganador: fusión tardía pose + visual (`fusion_tardia_pose_visual`).**

## Salvedades que el informe debe declarar

- La ventaja sobre el RF de pose es mínima (límite inferior +0,002) y no es significativa en limpio.
- La fusión está dominada por la pose: el peso de la pose elegido en validación interna fue 0,75 en
  19 de 20 folds y 1,0 en uno.
- Todos los candidatos caen entre 10 y 13 puntos en el subconjunto limpio. Las features de pose
  predicen la duración del clip (auditoría, R² 0,53): parte de la ventaja puede ser de filmación.
- El MIL, que era el modelo de la tesina, queda tercero: con el protocolo común obtiene 0,722. La
  cifra previa de 0,740 se midió con otra partición (4 folds, sin repeticiones) y otro conjunto
  (133 caballos), por lo que no son directamente comparables y no debe leerse como una caída
  medida en igualdad de condiciones. Lo que sí muestra el protocolo común es que las tres
  correcciones no lo acercaron a los otros dos candidatos.
- Los JSON de los cuatro candidatos están ahora en `comparacion/resultados/` (el MIL y el RF de
  pose se trajeron desde sus ramas `feat/candidato-mil` y `feat/candidato-rf-pose`), junto con
  `comparaciones_pareadas.json`, que se reproduce con `comparacion.comparar`.

## Uso de la reserva

Autorizado por Sergio el 08/10/2026, una sola vez, con el ganador. Resultado en
`resultados/fusion_tardia_pose_visual__reserva.json` y uso registrado en `reserva_uso.log`.

### Resultado en la reserva (único uso, 08/10/2026)

20 caballos, 68 clips. El modelo se ajustó con todo el desarrollo y no se tocó después.

- F1 macro **0,834**, IC95 bootstrap por caballo [0,698; 0,925] (2 000 réplicas, semilla 7).
- Accuracy 0,853. F1 por clase: Disgustado 0,865, Interesado 0,727, Sereno 0,909.
- Subconjunto limpio (48 clips): F1 macro 0,767.
- Matriz de confusión (filas = real; Disgustado, Interesado, Sereno): [16, 1, 0], [3, 12, 1], [1, 4, 30].

Está dentro de lo esperado por el desarrollo (0,868 ± 0,046; limpio 0,740). Con 20 caballos el
intervalo es ancho: no se usa para comparar candidatos.

## Validación por sesión de filmación (análisis complementario)

Fecha: 08/10/2026. Diseño declarado antes de correr en la enmienda de `PROTOCOLO.md`; código en
`validacion_por_sesion.py`; resultados en `resultados/validacion_por_sesion.json`. **No cambia el
ganador ni toca la reserva**: el criterio de elección estaba fijado antes (§5 del protocolo).

**Qué se hizo.** Para cada una de las 20 sesiones de filmación con clips de desarrollo (de las 22 de
la tabla), el test son los clips de esa sesión y el train son los de las otras sesiones, sin los de
ningún caballo presente en la sesión de test. Ninguna sesión se saltó (el train siempre conservó las
tres clases). La exclusión por caballo quitó 40 clips de train en total (2,0 por sesión en promedio;
máximo 21, en S018). Los cuatro candidatos usaron exactamente la misma partición, con un candidato
nuevo por sesión. Se obtuvo una predicción fuera de muestra para cada uno de los 291 clips.

| Candidato | F1 macro agrupado [IC95] | Accuracy | F1 limpio (188 clips) [IC95] | 5×4 (F1 fuera de muestra) | Caída | 5×4 limpio | Caída limpio |
|---|---|---|---|---|---|---|---|
| Fusión pose + visual | 0,864 [0,793; 0,921] | 0,866 | 0,744 [0,665; 0,830] | 0,862 | −0,002 | 0,740 | −0,003 |
| RF de pose tipo SimBA | 0,831 [0,774; 0,885] | 0,832 | 0,715 [0,613; 0,781] | 0,836 | +0,005 | 0,718 | +0,003 |
| MIL multimodal corregido | 0,690 [0,606; 0,763] | 0,698 | 0,587 [0,481; 0,706] | 0,715 | +0,025 | 0,618 | +0,031 |
| Baseline LogReg media de frames | 0,628 [0,555; 0,696] | 0,632 | 0,534 [0,452; 0,613] | 0,696 | +0,068 | 0,597 | +0,063 |

La comparación con el 5×4 usa el F1 agrupado de las predicciones fuera de muestra de cada
repetición (media de las 5), que es la magnitud equivalente al F1 agrupado de este análisis; no es
el F1 medio por fold de la tabla de arriba (0,868 para la fusión). «Caída» = 5×4 − validación por
sesión; un valor negativo significa que no cayó. Los dos protocolos usan los mismos 291 clips pero
distintos entrenamientos (el train de cada sesión tiene entre 202 y 290 clips; el del 5×4, unos 218),
así que la comparación es orientativa.

F1 por clase de la fusión: Disgustado 0,823, Interesado 0,878, Sereno 0,891. Matriz de confusión
(filas = real; Disgustado, Interesado, Sereno): [72, 10, 4], [7, 90, 5], [10, 3, 90]. Las matrices y
los F1 por clase de los otros tres están en el JSON.

**Diferencias pareadas, fusión − otro (IC95 bootstrap por caballo, 2 000 réplicas, semilla 7).**

| Diferencia | Todo (291 clips, 113 caballos) | Limpio (188 clips, 71 caballos) |
|---|---|---|
| fusión − RF de pose | +0,032 [−0,006; +0,072] | +0,028 [−0,028; +0,117] |
| fusión − MIL | +0,174 [+0,105; +0,245] | +0,156 [+0,039; +0,253] |
| fusión − baseline | +0,236 [+0,177; +0,301] | +0,210 [+0,126; +0,294] |

**Interpretación.**

- **El ranking se sostiene**: fusión, RF de pose, MIL y baseline, en el mismo orden que en el 5×4. La
  fusión sigue primera y su ventaja sobre el MIL y el baseline es clara, también en el subconjunto limpio.
- **El nivel de acierto de los dos candidatos con pose se sostiene** (la fusión pierde 0 puntos y el
  RF 0,5); el MIL pierde 2,5 y el baseline 6,8. Los candidatos que dependen más de la apariencia
  visual son los que más caen ante sesiones no vistas, lo que es coherente con que la apariencia
  arrastre rasgos de cada filmación.
- **La ventaja de la fusión sobre el RF de pose ya no se distingue de cero** (+0,032, IC que
  incluye 0; en el 5×4 era +0,026 con límite inferior +0,002). Es una **limitación para la elección**:
  si la regla de empate del §5 (gana el más simple) se aplicara con este resultado, el RF de pose
  sería el elegido. Por mandato del protocolo el ganador no cambia, pero el informe no debe
  presentar a la fusión como claramente superior al RF de pose; la diferencia es mínima en ambos análisis.
- **Qué implica para el confusor de filmación**: ver una sesión nueva no baja el acierto de los
  modelos con pose, así que la ventaja **no se explica por rasgos propios de cada sesión** (un
  modelo que memorizara el contexto de una sesión habría caído). Pero **no descarta el confusor**: el
  subconjunto limpio sigue unos 12 puntos por debajo en los dos candidatos con pose (0,744 contra
  0,864 en la fusión; 0,715 contra 0,831 en el RF), igual que en el 5×4. Los rasgos que delatan la
  clase —sobre todo la duración del clip, que las features de pose predicen— parecen ser propiedades del
  protocolo de filmación de cada clase y no de una sesión en particular (la mediana de Disgustado es
  más corta, C-04 de `docs/caminos_explorados.md`), así que dejar afuera una sesión no los neutraliza. Lo que este análisis
  muestra es robustez ante sesiones nuevas, no ausencia de atajo de duración.
- **Límites del análisis**: 20 sesiones muy desiguales (S020 concentra 85 de 291 clips y 38 caballos;
  varias sesiones tienen 1 a 4 clips o una sola clase, por eso la métrica es agrupada); sesiones del
  mismo día o de fechas cercanas (S011/S012, S016/S017) pueden compartir condiciones aunque figuren
  como sesiones distintas; las sesiones se asocian con la clase en parte (S023 y S024 sólo tienen
  Disgustado), y el IC de la fusión sigue siendo ancho ([0,793; 0,921]). El análisis usa sólo el
  desarrollo; el único número fuera de muestra sigue siendo el de la reserva (0,834).

## Identidad duplicada: caballos 123 y 169 (08/10/2026)

La devolución del docente (observación D3) señaló que `123` y `169` podían ser el mismo caballo con
dos identificadores. Los dos videos son «Interesado» y se filmaron el 04/10/2025 con 34 s de
diferencia. Sergio comparó fotogramas de ambos y **confirmó que son el mismo caballo**.

No se rehacen las particiones, que estaban fijadas antes de comparar. Se mide el efecto:

- En el 5×4, el par quedó separado entre train y test en 4 de las 5 repeticiones (folds de 123 / 169:
  2/1, 2/2, 0/1, 3/2, 1/0).
- Quitar sus 5 clips del F1 agrupado fuera de muestra (media de las 5 repeticiones) lo cambia en:
  fusión 0,8620 → 0,8633 (+0,001); RF de pose 0,8361 → 0,8366 (+0,001); MIL 0,7152 → 0,7115
  (−0,004); baseline 0,6957 → 0,6930 (−0,003). El ranking y la elección no cambian.
- No afecta a la reserva (ninguno de los dos está en ella), a la validación por sesión (los dos son
  de la sesión S011, que siempre se prueba entera y cuyos caballos se excluyen del train) ni al
  modelo final de la app (se entrena con todos los clips).
- No se revisaron visualmente otros pares de identificadores.

