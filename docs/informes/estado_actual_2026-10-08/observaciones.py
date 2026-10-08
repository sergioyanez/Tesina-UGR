"""Observaciones del docente → acción → resultado.

Las observaciones se extrajeron de las anotaciones del PDF `Devolucion_Juan_resumen_de_condiciones_cumplidas_o_no .pdf`
(autor de las anotaciones: «juan toloza»): 4 comentarios escritos (A), una nota sobre SimBA (B), una nota de diagnóstico con
cinco indicaciones (C) y los pasajes resaltados sin comentario (D). Los números del resultado salen de los JSON.
"""
from datos import CANDIDATOS, ic, n, par, pm, sg

FUS, RF, MIL, BAS = CANDIDATOS


def observaciones(D, ic_res):
    G = {c: D["cand"][c]["global"] for c in CANDIDATOS}
    LIM = {c: D["cand"][c]["limpio"]["f1_macro_oof_por_repeticion"] for c in CANDIDATOS}
    rep = [p["f1_macro_oof"] for p in D["cand"][FUS]["por_particion"]]
    p_rf = par(D, FUS, RF, "todo")
    ses_rf = D["sesion"]["intervalos"]["todo"]["diferencia_vs_fusion_tardia_pose_visual"][RF]
    return [
        # ------------------------------------------------------------------ A: comentarios escritos
        dict(obs="<b>A1.</b> «¿Cómo está la repetibilidad?» (junto a «reproducible bit a bit» y a las validaciones internas desbalanceadas).",
             accion="La auditoría v2 la midió en lugar de afirmarla. El protocolo común repite la validación 5 veces con particiones distintas de caballos y estratifica la validación interna.",
             resultado=f"MIL anterior, misma partición: 359/359 predicciones idénticas con la semilla 42; 6 semillas: 0,739 ± 0,015; otras particiones de caballos: 0,744 a 0,758. "
                       f"Fusión en el protocolo común: {n(min(rep))} a {n(max(rep))} entre las 5 repeticiones (fig. 4).",
             estado="Resuelta"),
        dict(obs="<b>A2.</b> «Me queda la duda si es espacio temporal o no porque es invariante al orden que se procesan los frames. Tal vez sea solo procesamiento del video.»",
             accion="Se verificó numéricamente: 0 de 359 predicciones cambian al barajar o invertir los 16 fotogramas. El requisito pasó a «parcial» y el texto dice «agregación espacial de varios fotogramas».",
             resultado="El docente tenía razón. Los dos modelos que lideran la comparación tampoco modelan el orden: resumen el movimiento en 21 rasgos dinámicos. La hipótesis temporal no queda sostenida.",
             estado="Resuelta"),
        dict(obs="<b>A3.</b> «No lo estamos haciendo, sacar» (requisito: la evolución temporal aporta robustez).",
             accion="Se retiró de la matriz de cumplimiento (18 requisitos en lugar de 19); la evidencia empírica se conserva como hallazgo.",
             resultado="Auditoría v2, §1.5 y §2.4.",
             estado="Resuelta"),
        dict(obs="<b>A4.</b> «¿Usaron el DeepLabCut con Horse-30? Subconjunto experto específico para caballos.»",
             accion="Se buscó en código, notebooks y documentos: no se usa. Se aplica el modelo generalista SuperAnimal-Quadruped sin ajuste.",
             resultado="Sin referencias a Horse-10 ni Horse-30 en el repositorio. No se midió la calidad de los puntos contra anotación humana en el conjunto vigente (sólo hay 14 puntos anotados a mano). Un modelo específico no se probó.",
             estado="Atendida en parte"),
        # ------------------------------------------------------------------ B: nota SimBA
        dict(obs="<b>B1.</b> Nota sobre SimBA: clasificadores supervisados (Random Forest) sobre los puntos de DeepLabCut, con rasgos calculados y explicabilidad (SHAP).",
             accion="Experimento comparativo (07/10): 160 rasgos de pose «tipo SimBA» con código propio y Random Forest; SimBA real instalado y probado; SHAP calculado en la auditoría v2.",
             resultado=f"RF de pose 0,841 ± 0,003 (5 semillas) contra 0,740 del MIL en el protocolo anterior; en el protocolo común queda segundo ({pm(G[RF]['f1_macro']['media'], G[RF]['f1_macro']['desvio'])}). SimBA real, para un solo animal, sólo da movimiento (0,720).",
             estado="Resuelta"),
        # ------------------------------------------------------------------ C: nota de diagnóstico
        dict(obs="<b>C1.</b> Aceptar la parsimonia: no ocultar el baseline estático (0,741) y preferir la solución más simple a igual resultado.",
             accion="El baseline es el candidato 1 del protocolo y la regla de empate («gana el más simple») se declaró antes de medir.",
             resultado=f"En el protocolo común el baseline queda último ({pm(G[BAS]['f1_macro']['media'], G[BAS]['f1_macro']['desvio'])}): sin la pose, la solución simple no alcanza. Fusión − RF no activa el desempate "
                       f"(IC {ic(p_rf['diferencia_ic95'], signo=True)}); por sesión sí lo activaría ({sg(ses_rf['diferencia'])} {ic(ses_rf['ic95'], signo=True)}).",
             estado="Resuelta"),
        dict(obs="<b>C2.</b> Blanquear el confusor de filmación: medir y reportar el rendimiento sólo en clips «limpios» (1080×1920, ≥ 5 s) y declarar el desbalance de duraciones.",
             accion="El subconjunto limpio es obligatorio en toda tabla; el criterio de elección exige no inferioridad en limpio; se agregó la validación dejando afuera cada sesión.",
             resultado=f"Fusión en limpio {pm(LIM[FUS]['media'], LIM[FUS]['desvio'])} (reserva {n(D['reserva']['limpio_f1_macro'])}). Todos los modelos caen entre 10 y 13 puntos: el confusor se midió y se declara, pero no se eliminó.",
             estado="Atendida en parte"),
        dict(obs="<b>C3.</b> Sincerar el sobreajuste: no afirmar que no hay; mostrar train contra validación.",
             accion="El gap train − test se mide y se informa por candidato; el MIL usa early stopping con validación interna estratificada por caballo y clase.",
             resultado=f"Gap: fusión +{n(G[FUS]['gap_train_test_f1']['media'], 2)}, RF +{n(G[RF]['gap_train_test_f1']['media'], 2)}, MIL +{n(G[MIL]['gap_train_test_f1']['media'], 2)}, baseline +{n(G[BAS]['gap_train_test_f1']['media'], 2)}. "
                       f"El F1 de train de la fusión es {n(G[FUS]['f1_macro_train']['media'])}: sobreajusta el train y el test es lo honesto.",
             estado="Resuelta"),
        dict(obs="<b>C4.</b> Documentar la falta de protocolo etológico (EquiFACS) y de acuerdo entre observadores; reconocerlo como limitación (resaltados: «no hay registro de quién etiquetó», «techo de rendimiento desconocido»).",
             accion="La etiquetadora (veterinaria, codificadora EquiFACS certificada) devolvió el formulario el 28/09; el criterio quedó escrito en <code>docs/protocolo_de_etiquetado.md</code>. Se declaró la limitación.",
             resultado="El acuerdo sólo se midió en un piloto sobre imágenes (kappa de Fleiss 0,99). En el dataset hay un único observador, que además filmó; no se midió el acuerdo interobservador (decisión del 07/10). Las alternativas de etiqueta débil y de unidades de acción se descartaron.",
             estado="Atendida en parte"),
        dict(obs="<b>C5.</b> Limpiar la trazabilidad: sin cifras «fantasma» (el 0,754) y corregir <code>_huella_caracteristicas</code> para que apunte a la versión v2; reproducibilidad baja de los experimentos recientes (resaltado).",
             accion="<code>_huella_caracteristicas</code> corregida (change <code>mejoras-prioridad-alta</code>); <code>docs/registro_de_experimentos.md</code> y <code>docs/caminos_explorados.md</code>; el 0,754 quedó marcado «no citar» (sin artefacto local).",
             resultado="Cada resultado de la comparación se regenera con un comando y lleva la huella del dataset, de las particiones y de la reserva; los cuatro JSON comparten las mismas huellas.",
             estado="Resuelta"),
        # ------------------------------------------------------------------ D: resaltados sin comentario
        dict(obs="<b>D1.</b> Resaltado: el criterio de aceptación de 0,691 se definió sobre otro conjunto y otro baseline.",
             accion="Se reemplazó por un criterio de elección declarado antes de comparar (§5 del protocolo): ranking, no inferioridad en limpio y desempate por simplicidad.",
             resultado="Aplicado a los cuatro candidatos (sección 5.5) y registrado en <code>DECISION.md</code> antes de usar la reserva.",
             estado="Resuelta"),
        dict(obs="<b>D2.</b> Resaltado: la validación anidada (0,784) no tiene script fuente; no se puede afirmar que el MIL sea «óptimo».",
             accion="La validación anidada quedó registrada con su rama y su commit; el MIL se comparó con otras tres arquitecturas en un protocolo común.",
             resultado=f"El MIL corregido queda tercero ({pm(G[MIL]['f1_macro']['media'], G[MIL]['f1_macro']['desvio'])}). «Óptimo» ya no se afirma: la comparación cubre cuatro candidatos, no todas las arquitecturas posibles.",
             estado="Resuelta"),
        dict(obs="<b>D3.</b> Resaltado: IDs duplicados de caballos (par 123/169); «no puede confirmarse que un mismo caballo no tenga dos IDs».",
             accion="Sergio comparó fotogramas de los dos videos (filmados con 34 s de diferencia el 04/10/2025) y confirmó el 08/10/2026 que son el mismo caballo. Se midió el efecto sin rehacer las particiones, que estaban fijadas antes de comparar.",
             resultado="El par quedó separado entre train y test en 4 de las 5 repeticiones del 5×4. Quitar sus 5 clips cambia el F1 agrupado en +0,001 (fusión), +0,001 (RF), −0,004 (MIL) y −0,003 (baseline): el ranking no cambia. No afecta a la reserva (ninguno de los dos está en ella), a la validación por sesión (los dos son de la misma sesión) ni al modelo final. Detalle en <code>DECISION.md</code>.",
             estado="Resuelta"),
        dict(obs="<b>D4.</b> Resaltado: la interfaz procesa el video entero y elige la mejor ventana de ≤ 7 s, que puede no ser el tramo etiquetado.",
             accion="La app informa el tramo analizado (segundo de inicio y de fin, y duración total del video) y, si el video supera los 7 s, recomienda subir un video recortado al momento de interés. La selección de la ventana no cambió; la rama de recortes largos no se fusionó.",
             resultado="La app informa el tramo analizado y recomienda recortar: el usuario sabe qué parte del video se clasificó. Rama <code>fix/app-tramo-analizado</code>; prueba <code>tests/test_tramo_analizado.py</code>.",
             estado="Resuelta"),
        dict(obs="<b>D5.</b> Resaltado: la utilidad para el público objetivo no fue medida; no hay test externo; el informe final no está en el repositorio.",
             accion="La reserva de 20 caballos da la única estimación fuera de muestra; la prueba de textos con una persona ajena (n = 1) no se repitió. El informe final se redacta para el 29/10.",
             resultado=f"Sin validación con veterinarios ni test externo. Reserva: F1 {n(D['reserva']['metricas']['f1_macro'])}, IC95 {ic(ic_res)}.",
             estado="Abierta"),
        dict(obs="<b>D6.</b> Resaltado: no se encontró un relevamiento bibliográfico formal en el repositorio.",
             accion="No se atendió en este tramo: corresponde al informe final.",
             resultado="Pendiente.",
             estado="Abierta"),
    ]
