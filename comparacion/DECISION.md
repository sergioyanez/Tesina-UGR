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
