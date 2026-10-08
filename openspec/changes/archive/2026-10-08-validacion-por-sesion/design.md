## Context

291 clips de 113 caballos de desarrollo y 20 sesiones de filmación con clips de desarrollo (la tabla
`comparacion/sesiones_de_filmacion.csv` confirma 22 en total). La tabla está por video (274); los
clips son 359. El vínculo clip → video está en `video_origen` de `dataset_frames_alineados_v2.csv`
(la fuente de `particion.tabla_clips`), y la clave (clase, nombre de archivo) identifica unívocamente
cada video de la tabla de sesiones (que usa rutas con subcarpetas).

## Goals / Non-Goals

**Goals**
- Medir los cuatro candidatos con sesiones no vistas, con la misma partición para todos.
- Que no haya fuga ni por sesión ni por caballo.
- Declarar la partición y la métrica antes de correr.

**Non-Goals**
- Cambiar el ganador. El criterio de elección estaba fijado antes (PROTOCOLO §5).
- Tocar la reserva (ya usada una vez) o modificar `equinos/comparacion.py`.

## Decisions

1. **Partición**: para cada sesión `s` con clips de desarrollo, test = clips de `s`; train = clips de las
   otras sesiones menos los de cualquier caballo presente en `s`. Se registra cuántos clips de train
   se pierden. Si el train queda sin una clase, la sesión se saltea y se documenta.
2. **Métrica principal: F1 macro agrupado** sobre las predicciones fuera de muestra (una por clip):
   muchas sesiones tienen pocos clips o una sola clase, y un F1 por sesión sería inestable.
3. **Incertidumbre**: bootstrap por caballo (2 000 réplicas, semilla 7) sobre las predicciones
   agrupadas, para cada candidato y para la diferencia pareada fusión − otro.
4. **Instancia nueva por sesión** de cada candidato; reutiliza `Datos` y `f1_macro` sin modificarlos.

## Alternatives considered

- **`GroupKFold` por sesión (k folds de sesiones)**: descartada. Las sesiones son muy desiguales
  (una concentra 85 de 291 clips y 38 caballos), los folds serían inestables y no controlarían los
  caballos repetidos entre sesiones.
- **Componentes conexas caballo–sesión** (agrupar sesiones y caballos que comparten alguna relación
  y partir por componente): descartada. Con 38 caballos en la sesión mayor y caballos presentes en
  varias sesiones, casi todo quedaría en una sola componente gigante y no habría partición útil.
- **Dejar afuera la sesión sin excluir sus caballos del train**: descartada, porque deja fuga por
  caballo y mezcla dos efectos (sesión nueva y caballo ya visto).
- **Usar la reserva**: descartada, ya se usó una vez y es de un solo uso.

## Risks / Trade-offs

- El train pierde pocos clips (ver el JSON) pero cambia de una sesión a otra: la comparación es entre
  candidatos, con la misma partición, no entre sesiones.
- Sesiones de 1 clip aportan una predicción cada una; por eso el F1 es agrupado.
- Las sesiones están muy asociadas a la clase (varias tienen una sola clase): el F1 agrupado mezcla
  eso; se informa también el subconjunto limpio.
