# Cifras del informe (generado por `informe/generar_cifras.py`, no editar)

Cada fila es una macro de `cifras.tex`. Usarla como `\Macro{}`. Las cifras no se escriben a mano en el `.tex`.

| Macro | Valor | Qué es | Fuente |
|---|---|---|---|
| `\ClipsTotal` | 359 | Clips totales (desarrollo + reserva) | comparacion/resultados/validacion_por_sesion.json; comparacion/reserva_caballos.json |
| `\ClipsDesarrollo` | 291 | Clips de desarrollo | comparacion/resultados/validacion_por_sesion.json (intervalos.todo.clips) |
| `\ClipsReserva` | 68 | Clips de la reserva | comparacion/resultados/fusion_tardia_pose_visual__reserva.json (clips) |
| `\CaballosTotal` | 133 | Caballos totales | comparacion/reserva_caballos.json |
| `\CaballosDesarrollo` | 113 | Caballos de desarrollo | comparacion/reserva_caballos.json (desarrollo) |
| `\CaballosReserva` | 20 | Caballos de la reserva | comparacion/reserva_caballos.json (reserva) |
| `\VideosTotal` | 274 | Videos originales | comparacion/sesiones_resumen.json |
| `\SesionesTotal` | 22 | Sesiones de filmación confirmadas | comparacion/sesiones_de_filmacion.csv (sesion_confirmada) |
| `\SesionesEvaluadas` | 20 | Sesiones evaluadas (con clips de desarrollo) | comparacion/resultados/validacion_por_sesion.json (plan.evaluadas) |
| `\ClipsLimpioDesarrollo` | 188 | Clips del subconjunto limpio en desarrollo | comparacion/resultados/validacion_por_sesion.json (intervalos.limpio) |
| `\CaballosLimpioDesarrollo` | 71 | Caballos del subconjunto limpio en desarrollo | comparacion/resultados/validacion_por_sesion.json (intervalos.limpio) |
| `\ClipsLimpioReserva` | 48 | Clips limpios de la reserva | comparacion/DECISION.md (reserva, subconjunto limpio) |
| `\CriterioLimpio` | 1080\,$\times$\,1920 y duración $\geq$ 5\,s | Criterio del subconjunto limpio | comparacion/resultados/*.json (limpio.criterio) |
| `\ClipsDevDisg` | 86 | Clips de desarrollo de Disgustado | comparacion/resultados/fusion...json (matriz_confusion_suma / 5) |
| `\ClipsReservaDisg` | 17 | Clips de la reserva de Disgustado | comparacion/resultados/fusion...__reserva.json (matriz) |
| `\ClipsLimpioDisg` | 23 | Clips limpios de desarrollo de Disgustado | comparacion/resultados/fusion...json (limpio.clips_por_clase) |
| `\ClipsDevInt` | 102 | Clips de desarrollo de Interesado | comparacion/resultados/fusion...json (matriz_confusion_suma / 5) |
| `\ClipsReservaInt` | 16 | Clips de la reserva de Interesado | comparacion/resultados/fusion...__reserva.json (matriz) |
| `\ClipsLimpioInt` | 71 | Clips limpios de desarrollo de Interesado | comparacion/resultados/fusion...json (limpio.clips_por_clase) |
| `\ClipsDevSer` | 103 | Clips de desarrollo de Sereno | comparacion/resultados/fusion...json (matriz_confusion_suma / 5) |
| `\ClipsReservaSer` | 35 | Clips de la reserva de Sereno | comparacion/resultados/fusion...__reserva.json (matriz) |
| `\ClipsLimpioSer` | 94 | Clips limpios de desarrollo de Sereno | comparacion/resultados/fusion...json (limpio.clips_por_clase) |
| `\Repeticiones` | 5 | Repeticiones del 5×4 | comparacion/resultados/*.json (R) |
| `\Folds` | 4 | Folds por repetición | comparacion/resultados/*.json (K) |
| `\Evaluaciones` | 20 | Evaluaciones del 5×4 (R×K) | comparacion/resultados/*.json |
| `\ReplicasBootstrap` | 2000 | Réplicas del bootstrap por caballo | comparacion/resultados/comparaciones_pareadas.json |
| `\SemillaBootstrap` | 7 | Semilla del bootstrap | comparacion/resultados/comparaciones_pareadas.json |
| `\ClipsTrainSesionMin` | 202 | Mínimo de clips de train por sesión | comparacion/resultados/validacion_por_sesion.json (plan.evaluadas) |
| `\ClipsTrainSesionMax` | 290 | Máximo de clips de train por sesión | comparacion/resultados/validacion_por_sesion.json (plan.evaluadas) |
| `\ClipsTrainPerdidos` | 40 | Clips de train quitados por exclusión de caballos (total) | comparacion/resultados/validacion_por_sesion.json (plan) |
| `\ClipsTrainPerdidosMedia` | 2,0 | Ídem, promedio por sesión | comparacion/resultados/validacion_por_sesion.json (plan) |
| `\PoseRasgos` | 160 | Rasgos de pose | comparacion/resultados/fusion...json (descripcion) |
| `\Fotogramas` | 16 | Fotogramas seleccionados por clip | equinos/comparacion.py (Datos: 16 frames por clip) |
| `\PesoPose` | 0,75 | Peso w de la pose en la fusión | comparacion/DECISION.md (peso elegido en validación interna) |
| `\PesoFoldsA` | 19 | Folds en que w = 0,75 | comparacion/DECISION.md |
| `\PesoFoldsTotal` | 20 | Folds totales | comparacion/DECISION.md |
| `\PesoPoseUno` | 1,0 | Peso en el fold restante (1 de 20) | comparacion/DECISION.md |
| `\FusionDevF` | 0,868 | Fusión pose + visual: F1 macro 5×4 (media de 20 folds) | comparacion/resultados/fusion_tardia_pose_visual.json (global.f1_macro.media) |
| `\FusionDevDE` | 0,046 | Desvío del F1 macro (20 folds) | comparacion/resultados/fusion_tardia_pose_visual.json (global.f1_macro.desvio) |
| `\FusionDevAcc` | 87,1 % | Accuracy 5×4 (porcentaje) | comparacion/resultados/fusion_tardia_pose_visual.json (global.accuracy.media) |
| `\FusionDevAccDE` | 4,7 % | Desvío de la accuracy (porcentaje) | comparacion/resultados/fusion_tardia_pose_visual.json (global.accuracy.desvio) |
| `\FusionLimpioF` | 0,740 | F1 macro en el subconjunto limpio (media de 5 repeticiones) | comparacion/resultados/fusion_tardia_pose_visual.json (limpio.f1_macro_oof_por_repeticion.media) |
| `\FusionLimpioDE` | 0,028 | Desvío del F1 limpio (5 repeticiones) | comparacion/resultados/fusion_tardia_pose_visual.json (limpio.f1_macro_oof_por_repeticion.desvio) |
| `\FusionGap` | 0,13 | Brecha F1 train − test (2 decimales) | comparacion/resultados/fusion_tardia_pose_visual.json (global.gap_train_test_f1.media) |
| `\FusionGapDE` | 0,05 | Desvío de la brecha | comparacion/resultados/fusion_tardia_pose_visual.json (global.gap_train_test_f1.desvio) |
| `\FusionTrainF` | 1,000 | F1 macro en train | comparacion/resultados/fusion_tardia_pose_visual.json (global.f1_macro_train.media) |
| `\FusionAgrupF` | 0,862 | F1 agrupado fuera de muestra 5×4 (media de 5 repeticiones) | comparacion/resultados/comparaciones_pareadas.json (f1_a/f1_b) |
| `\FusionAgrupIC` | [0,803; 0,910] | IC95 bootstrap por caballo del F1 agrupado 5×4 | comparacion/resultados/comparaciones_pareadas.json |
| `\FusionAgrupLimpioF` | 0,740 | F1 agrupado 5×4, subconjunto limpio | comparacion/resultados/comparaciones_pareadas.json |
| `\FusionAgrupLimpioIC` | [0,650; 0,804] | IC95 del anterior | comparacion/resultados/comparaciones_pareadas.json |
| `\FusionSesionF` | 0,864 | F1 macro agrupado, una sesión afuera | comparacion/resultados/validacion_por_sesion.json (intervalos.todo.f1_agrupado) |
| `\FusionSesionIC` | [0,793; 0,921] | IC95 bootstrap por caballo | comparacion/resultados/validacion_por_sesion.json |
| `\FusionSesionAcc` | 86,6 % | Accuracy, una sesión afuera | comparacion/resultados/validacion_por_sesion.json (candidatos.*.metricas.accuracy) |
| `\FusionSesionLimpioF` | 0,744 | F1 agrupado por sesión, subconjunto limpio | comparacion/resultados/validacion_por_sesion.json (intervalos.limpio.f1_agrupado) |
| `\FusionSesionLimpioIC` | [0,665; 0,830] | IC95 del anterior | comparacion/resultados/validacion_por_sesion.json |
| `\FusionSesionCaida` | −0,002 | Caída del F1 al pasar de 5×4 a sesión (negativo = no cayó) | 5×4 agrupado − sesión (calculado) |
| `\FusionSesionCaidaLimpio` | −0,003 | Caída en el subconjunto limpio | ídem, limpio |
| `\FusionDevFDisg` | 0,832 | F1 de Disgustado 5×4 (media) | comparacion/resultados/fusion_tardia_pose_visual.json (global.por_clase.Disgustado.f1.media) |
| `\FusionDevFDisgDE` | 0,066 | Desvío del F1 de Disgustado | comparacion/resultados/fusion_tardia_pose_visual.json (global.por_clase.Disgustado.f1.desvio) |
| `\FusionSesionFDisg` | 0,823 | F1 de Disgustado, una sesión afuera | comparacion/resultados/validacion_por_sesion.json (candidatos.*.metricas.f1.Disgustado) |
| `\FusionDevFInt` | 0,889 | F1 de Interesado 5×4 (media) | comparacion/resultados/fusion_tardia_pose_visual.json (global.por_clase.Interesado.f1.media) |
| `\FusionDevFIntDE` | 0,073 | Desvío del F1 de Interesado | comparacion/resultados/fusion_tardia_pose_visual.json (global.por_clase.Interesado.f1.desvio) |
| `\FusionSesionFInt` | 0,878 | F1 de Interesado, una sesión afuera | comparacion/resultados/validacion_por_sesion.json (candidatos.*.metricas.f1.Interesado) |
| `\FusionDevFSer` | 0,884 | F1 de Sereno 5×4 (media) | comparacion/resultados/fusion_tardia_pose_visual.json (global.por_clase.Sereno.f1.media) |
| `\FusionDevFSerDE` | 0,041 | Desvío del F1 de Sereno | comparacion/resultados/fusion_tardia_pose_visual.json (global.por_clase.Sereno.f1.desvio) |
| `\FusionSesionFSer` | 0,891 | F1 de Sereno, una sesión afuera | comparacion/resultados/validacion_por_sesion.json (candidatos.*.metricas.f1.Sereno) |
| `\RFPoseDevF` | 0,844 | RF de pose tipo SimBA: F1 macro 5×4 (media de 20 folds) | comparacion/resultados/rf_pose_tipo_simba.json (global.f1_macro.media) |
| `\RFPoseDevDE` | 0,059 | Desvío del F1 macro (20 folds) | comparacion/resultados/rf_pose_tipo_simba.json (global.f1_macro.desvio) |
| `\RFPoseDevAcc` | 84,5 % | Accuracy 5×4 (porcentaje) | comparacion/resultados/rf_pose_tipo_simba.json (global.accuracy.media) |
| `\RFPoseDevAccDE` | 6,0 % | Desvío de la accuracy (porcentaje) | comparacion/resultados/rf_pose_tipo_simba.json (global.accuracy.desvio) |
| `\RFPoseLimpioF` | 0,718 | F1 macro en el subconjunto limpio (media de 5 repeticiones) | comparacion/resultados/rf_pose_tipo_simba.json (limpio.f1_macro_oof_por_repeticion.media) |
| `\RFPoseLimpioDE` | 0,020 | Desvío del F1 limpio (5 repeticiones) | comparacion/resultados/rf_pose_tipo_simba.json (limpio.f1_macro_oof_por_repeticion.desvio) |
| `\RFPoseGap` | 0,15 | Brecha F1 train − test (2 decimales) | comparacion/resultados/rf_pose_tipo_simba.json (global.gap_train_test_f1.media) |
| `\RFPoseGapDE` | 0,06 | Desvío de la brecha | comparacion/resultados/rf_pose_tipo_simba.json (global.gap_train_test_f1.desvio) |
| `\RFPoseTrainF` | 0,998 | F1 macro en train | comparacion/resultados/rf_pose_tipo_simba.json (global.f1_macro_train.media) |
| `\RFPoseAgrupF` | 0,836 | F1 agrupado fuera de muestra 5×4 (media de 5 repeticiones) | comparacion/resultados/comparaciones_pareadas.json (f1_a/f1_b) |
| `\RFPoseAgrupIC` | [0,775; 0,891] | IC95 bootstrap por caballo del F1 agrupado 5×4 | comparacion/resultados/comparaciones_pareadas.json |
| `\RFPoseAgrupLimpioF` | 0,718 | F1 agrupado 5×4, subconjunto limpio | comparacion/resultados/comparaciones_pareadas.json |
| `\RFPoseAgrupLimpioIC` | [0,621; 0,788] | IC95 del anterior | comparacion/resultados/comparaciones_pareadas.json |
| `\RFPoseSesionF` | 0,831 | F1 macro agrupado, una sesión afuera | comparacion/resultados/validacion_por_sesion.json (intervalos.todo.f1_agrupado) |
| `\RFPoseSesionIC` | [0,774; 0,885] | IC95 bootstrap por caballo | comparacion/resultados/validacion_por_sesion.json |
| `\RFPoseSesionAcc` | 83,2 % | Accuracy, una sesión afuera | comparacion/resultados/validacion_por_sesion.json (candidatos.*.metricas.accuracy) |
| `\RFPoseSesionLimpioF` | 0,715 | F1 agrupado por sesión, subconjunto limpio | comparacion/resultados/validacion_por_sesion.json (intervalos.limpio.f1_agrupado) |
| `\RFPoseSesionLimpioIC` | [0,613; 0,781] | IC95 del anterior | comparacion/resultados/validacion_por_sesion.json |
| `\RFPoseSesionCaida` | +0,005 | Caída del F1 al pasar de 5×4 a sesión (negativo = no cayó) | 5×4 agrupado − sesión (calculado) |
| `\RFPoseSesionCaidaLimpio` | +0,003 | Caída en el subconjunto limpio | ídem, limpio |
| `\RFPoseDevFDisg` | 0,838 | F1 de Disgustado 5×4 (media) | comparacion/resultados/rf_pose_tipo_simba.json (global.por_clase.Disgustado.f1.media) |
| `\RFPoseDevFDisgDE` | 0,060 | Desvío del F1 de Disgustado | comparacion/resultados/rf_pose_tipo_simba.json (global.por_clase.Disgustado.f1.desvio) |
| `\RFPoseSesionFDisg` | 0,830 | F1 de Disgustado, una sesión afuera | comparacion/resultados/validacion_por_sesion.json (candidatos.*.metricas.f1.Disgustado) |
| `\RFPoseDevFInt` | 0,850 | F1 de Interesado 5×4 (media) | comparacion/resultados/rf_pose_tipo_simba.json (global.por_clase.Interesado.f1.media) |
| `\RFPoseDevFIntDE` | 0,088 | Desvío del F1 de Interesado | comparacion/resultados/rf_pose_tipo_simba.json (global.por_clase.Interesado.f1.desvio) |
| `\RFPoseSesionFInt` | 0,830 | F1 de Interesado, una sesión afuera | comparacion/resultados/validacion_por_sesion.json (candidatos.*.metricas.f1.Interesado) |
| `\RFPoseDevFSer` | 0,844 | F1 de Sereno 5×4 (media) | comparacion/resultados/rf_pose_tipo_simba.json (global.por_clase.Sereno.f1.media) |
| `\RFPoseDevFSerDE` | 0,060 | Desvío del F1 de Sereno | comparacion/resultados/rf_pose_tipo_simba.json (global.por_clase.Sereno.f1.desvio) |
| `\RFPoseSesionFSer` | 0,835 | F1 de Sereno, una sesión afuera | comparacion/resultados/validacion_por_sesion.json (candidatos.*.metricas.f1.Sereno) |
| `\MILDevF` | 0,722 | MIL multimodal corregido: F1 macro 5×4 (media de 20 folds) | comparacion/resultados/mil_multimodal_corregido.json (global.f1_macro.media) |
| `\MILDevDE` | 0,102 | Desvío del F1 macro (20 folds) | comparacion/resultados/mil_multimodal_corregido.json (global.f1_macro.desvio) |
| `\MILDevAcc` | 73,0 % | Accuracy 5×4 (porcentaje) | comparacion/resultados/mil_multimodal_corregido.json (global.accuracy.media) |
| `\MILDevAccDE` | 9,6 % | Desvío de la accuracy (porcentaje) | comparacion/resultados/mil_multimodal_corregido.json (global.accuracy.desvio) |
| `\MILLimpioF` | 0,618 | F1 macro en el subconjunto limpio (media de 5 repeticiones) | comparacion/resultados/mil_multimodal_corregido.json (limpio.f1_macro_oof_por_repeticion.media) |
| `\MILLimpioDE` | 0,072 | Desvío del F1 limpio (5 repeticiones) | comparacion/resultados/mil_multimodal_corregido.json (limpio.f1_macro_oof_por_repeticion.desvio) |
| `\MILGap` | 0,19 | Brecha F1 train − test (2 decimales) | comparacion/resultados/mil_multimodal_corregido.json (global.gap_train_test_f1.media) |
| `\MILGapDE` | 0,11 | Desvío de la brecha | comparacion/resultados/mil_multimodal_corregido.json (global.gap_train_test_f1.desvio) |
| `\MILTrainF` | 0,911 | F1 macro en train | comparacion/resultados/mil_multimodal_corregido.json (global.f1_macro_train.media) |
| `\MILAgrupF` | 0,715 | F1 agrupado fuera de muestra 5×4 (media de 5 repeticiones) | comparacion/resultados/comparaciones_pareadas.json (f1_a/f1_b) |
| `\MILAgrupIC` | [0,647; 0,771] | IC95 bootstrap por caballo del F1 agrupado 5×4 | comparacion/resultados/comparaciones_pareadas.json |
| `\MILAgrupLimpioF` | 0,618 | F1 agrupado 5×4, subconjunto limpio | comparacion/resultados/comparaciones_pareadas.json |
| `\MILAgrupLimpioIC` | [0,542; 0,698] | IC95 del anterior | comparacion/resultados/comparaciones_pareadas.json |
| `\MILSesionF` | 0,690 | F1 macro agrupado, una sesión afuera | comparacion/resultados/validacion_por_sesion.json (intervalos.todo.f1_agrupado) |
| `\MILSesionIC` | [0,606; 0,763] | IC95 bootstrap por caballo | comparacion/resultados/validacion_por_sesion.json |
| `\MILSesionAcc` | 69,8 % | Accuracy, una sesión afuera | comparacion/resultados/validacion_por_sesion.json (candidatos.*.metricas.accuracy) |
| `\MILSesionLimpioF` | 0,587 | F1 agrupado por sesión, subconjunto limpio | comparacion/resultados/validacion_por_sesion.json (intervalos.limpio.f1_agrupado) |
| `\MILSesionLimpioIC` | [0,481; 0,706] | IC95 del anterior | comparacion/resultados/validacion_por_sesion.json |
| `\MILSesionCaida` | +0,025 | Caída del F1 al pasar de 5×4 a sesión (negativo = no cayó) | 5×4 agrupado − sesión (calculado) |
| `\MILSesionCaidaLimpio` | +0,031 | Caída en el subconjunto limpio | ídem, limpio |
| `\MILDevFDisg` | 0,658 | F1 de Disgustado 5×4 (media) | comparacion/resultados/mil_multimodal_corregido.json (global.por_clase.Disgustado.f1.media) |
| `\MILDevFDisgDE` | 0,171 | Desvío del F1 de Disgustado | comparacion/resultados/mil_multimodal_corregido.json (global.por_clase.Disgustado.f1.desvio) |
| `\MILSesionFDisg` | 0,620 | F1 de Disgustado, una sesión afuera | comparacion/resultados/validacion_por_sesion.json (candidatos.*.metricas.f1.Disgustado) |
| `\MILDevFInt` | 0,781 | F1 de Interesado 5×4 (media) | comparacion/resultados/mil_multimodal_corregido.json (global.por_clase.Interesado.f1.media) |
| `\MILDevFIntDE` | 0,109 | Desvío del F1 de Interesado | comparacion/resultados/mil_multimodal_corregido.json (global.por_clase.Interesado.f1.desvio) |
| `\MILSesionFInt` | 0,755 | F1 de Interesado, una sesión afuera | comparacion/resultados/validacion_por_sesion.json (candidatos.*.metricas.f1.Interesado) |
| `\MILDevFSer` | 0,726 | F1 de Sereno 5×4 (media) | comparacion/resultados/mil_multimodal_corregido.json (global.por_clase.Sereno.f1.media) |
| `\MILDevFSerDE` | 0,082 | Desvío del F1 de Sereno | comparacion/resultados/mil_multimodal_corregido.json (global.por_clase.Sereno.f1.desvio) |
| `\MILSesionFSer` | 0,696 | F1 de Sereno, una sesión afuera | comparacion/resultados/validacion_por_sesion.json (candidatos.*.metricas.f1.Sereno) |
| `\BaseDevF` | 0,694 | Baseline LogReg (media de frames): F1 macro 5×4 (media de 20 folds) | comparacion/resultados/baseline_logreg_media_frames.json (global.f1_macro.media) |
| `\BaseDevDE` | 0,062 | Desvío del F1 macro (20 folds) | comparacion/resultados/baseline_logreg_media_frames.json (global.f1_macro.desvio) |
| `\BaseDevAcc` | 70,5 % | Accuracy 5×4 (porcentaje) | comparacion/resultados/baseline_logreg_media_frames.json (global.accuracy.media) |
| `\BaseDevAccDE` | 6,1 % | Desvío de la accuracy (porcentaje) | comparacion/resultados/baseline_logreg_media_frames.json (global.accuracy.desvio) |
| `\BaseLimpioF` | 0,597 | F1 macro en el subconjunto limpio (media de 5 repeticiones) | comparacion/resultados/baseline_logreg_media_frames.json (limpio.f1_macro_oof_por_repeticion.media) |
| `\BaseLimpioDE` | 0,050 | Desvío del F1 limpio (5 repeticiones) | comparacion/resultados/baseline_logreg_media_frames.json (limpio.f1_macro_oof_por_repeticion.desvio) |
| `\BaseGap` | 0,31 | Brecha F1 train − test (2 decimales) | comparacion/resultados/baseline_logreg_media_frames.json (global.gap_train_test_f1.media) |
| `\BaseGapDE` | 0,06 | Desvío de la brecha | comparacion/resultados/baseline_logreg_media_frames.json (global.gap_train_test_f1.desvio) |
| `\BaseTrainF` | 1,000 | F1 macro en train | comparacion/resultados/baseline_logreg_media_frames.json (global.f1_macro_train.media) |
| `\BaseAgrupF` | 0,696 | F1 agrupado fuera de muestra 5×4 (media de 5 repeticiones) | comparacion/resultados/comparaciones_pareadas.json (f1_a/f1_b) |
| `\BaseAgrupIC` | [0,635; 0,749] | IC95 bootstrap por caballo del F1 agrupado 5×4 | comparacion/resultados/comparaciones_pareadas.json |
| `\BaseAgrupLimpioF` | 0,597 | F1 agrupado 5×4, subconjunto limpio | comparacion/resultados/comparaciones_pareadas.json |
| `\BaseAgrupLimpioIC` | [0,520; 0,657] | IC95 del anterior | comparacion/resultados/comparaciones_pareadas.json |
| `\BaseSesionF` | 0,628 | F1 macro agrupado, una sesión afuera | comparacion/resultados/validacion_por_sesion.json (intervalos.todo.f1_agrupado) |
| `\BaseSesionIC` | [0,555; 0,696] | IC95 bootstrap por caballo | comparacion/resultados/validacion_por_sesion.json |
| `\BaseSesionAcc` | 63,2 % | Accuracy, una sesión afuera | comparacion/resultados/validacion_por_sesion.json (candidatos.*.metricas.accuracy) |
| `\BaseSesionLimpioF` | 0,534 | F1 agrupado por sesión, subconjunto limpio | comparacion/resultados/validacion_por_sesion.json (intervalos.limpio.f1_agrupado) |
| `\BaseSesionLimpioIC` | [0,452; 0,613] | IC95 del anterior | comparacion/resultados/validacion_por_sesion.json |
| `\BaseSesionCaida` | +0,068 | Caída del F1 al pasar de 5×4 a sesión (negativo = no cayó) | 5×4 agrupado − sesión (calculado) |
| `\BaseSesionCaidaLimpio` | +0,063 | Caída en el subconjunto limpio | ídem, limpio |
| `\BaseDevFDisg` | 0,614 | F1 de Disgustado 5×4 (media) | comparacion/resultados/baseline_logreg_media_frames.json (global.por_clase.Disgustado.f1.media) |
| `\BaseDevFDisgDE` | 0,091 | Desvío del F1 de Disgustado | comparacion/resultados/baseline_logreg_media_frames.json (global.por_clase.Disgustado.f1.desvio) |
| `\BaseSesionFDisg` | 0,553 | F1 de Disgustado, una sesión afuera | comparacion/resultados/validacion_por_sesion.json (candidatos.*.metricas.f1.Disgustado) |
| `\BaseDevFInt` | 0,797 | F1 de Interesado 5×4 (media) | comparacion/resultados/baseline_logreg_media_frames.json (global.por_clase.Interesado.f1.media) |
| `\BaseDevFIntDE` | 0,088 | Desvío del F1 de Interesado | comparacion/resultados/baseline_logreg_media_frames.json (global.por_clase.Interesado.f1.desvio) |
| `\BaseSesionFInt` | 0,733 | F1 de Interesado, una sesión afuera | comparacion/resultados/validacion_por_sesion.json (candidatos.*.metricas.f1.Interesado) |
| `\BaseDevFSer` | 0,672 | F1 de Sereno 5×4 (media) | comparacion/resultados/baseline_logreg_media_frames.json (global.por_clase.Sereno.f1.media) |
| `\BaseDevFSerDE` | 0,076 | Desvío del F1 de Sereno | comparacion/resultados/baseline_logreg_media_frames.json (global.por_clase.Sereno.f1.desvio) |
| `\BaseSesionFSer` | 0,598 | F1 de Sereno, una sesión afuera | comparacion/resultados/validacion_por_sesion.json (candidatos.*.metricas.f1.Sereno) |
| `\FusionReservaF` | 0,834 | Fusión en la reserva: F1 macro | comparacion/resultados/fusion_tardia_pose_visual__reserva.json (metricas.f1_macro) |
| `\FusionReservaIC` | [0,698; 0,925] | IC95 del F1 macro de la reserva | comparacion/resultados/fusion_tardia_pose_visual__reserva.json (IC95 bootstrap por caballo) |
| `\FusionReservaAcc` | 85,3 % | Accuracy en la reserva (porcentaje) | comparacion/resultados/fusion_tardia_pose_visual__reserva.json (metricas.accuracy) |
| `\FusionReservaAccDec` | 0,853 | Accuracy en la reserva (decimal) | comparacion/resultados/fusion_tardia_pose_visual__reserva.json (metricas.accuracy) |
| `\FusionReservaLimpio` | 0,767 | F1 macro de la reserva, subconjunto limpio | comparacion/resultados/fusion_tardia_pose_visual__reserva.json (limpio_f1_macro) |
| `\FusionReservaFDisg` | 0,865 | F1 de Disgustado en la reserva | comparacion/resultados/fusion_tardia_pose_visual__reserva.json (metricas.f1.Disgustado) |
| `\FusionReservaPrecDisg` | 0,800 | Precisión de Disgustado en la reserva | comparacion/resultados/fusion_tardia_pose_visual__reserva.json (metricas.precision.Disgustado) |
| `\FusionReservaRecDisg` | 0,941 | Recall de Disgustado en la reserva | comparacion/resultados/fusion_tardia_pose_visual__reserva.json (metricas.recall.Disgustado) |
| `\FusionReservaFInt` | 0,727 | F1 de Interesado en la reserva | comparacion/resultados/fusion_tardia_pose_visual__reserva.json (metricas.f1.Interesado) |
| `\FusionReservaPrecInt` | 0,706 | Precisión de Interesado en la reserva | comparacion/resultados/fusion_tardia_pose_visual__reserva.json (metricas.precision.Interesado) |
| `\FusionReservaRecInt` | 0,750 | Recall de Interesado en la reserva | comparacion/resultados/fusion_tardia_pose_visual__reserva.json (metricas.recall.Interesado) |
| `\FusionReservaFSer` | 0,909 | F1 de Sereno en la reserva | comparacion/resultados/fusion_tardia_pose_visual__reserva.json (metricas.f1.Sereno) |
| `\FusionReservaPrecSer` | 0,968 | Precisión de Sereno en la reserva | comparacion/resultados/fusion_tardia_pose_visual__reserva.json (metricas.precision.Sereno) |
| `\FusionReservaRecSer` | 0,857 | Recall de Sereno en la reserva | comparacion/resultados/fusion_tardia_pose_visual__reserva.json (metricas.recall.Sereno) |
| `\ReservaAciertos` | 58 | Clips bien clasificados en la reserva | comparacion/resultados/fusion_tardia_pose_visual__reserva.json (traza de la matriz) |
| `\DifFusionRFDev` | +0,026 | Fusion − RF, 5×4 todo | comparacion/resultados/comparaciones_pareadas.json (fusion_tardia_pose_visual vs rf_pose_tipo_simba, todo) |
| `\DifFusionRFDevIC` | [+0,002; +0,051] | IC95 de Fusion − RF, 5×4 todo | comparacion/resultados/comparaciones_pareadas.json |
| `\DifFusionRFLimpio` | +0,022 | Fusion − RF, 5×4 limpio | comparacion/resultados/comparaciones_pareadas.json (fusion_tardia_pose_visual vs rf_pose_tipo_simba, limpio) |
| `\DifFusionRFLimpioIC` | [−0,017; +0,056] | IC95 de Fusion − RF, 5×4 limpio | comparacion/resultados/comparaciones_pareadas.json |
| `\DifFusionRFSesion` | +0,032 | Fusion − RF, una sesión afuera (todo) | comparacion/resultados/validacion_por_sesion.json (intervalos.todo.diferencia_vs_fusion...) |
| `\DifFusionRFSesionIC` | [−0,006; +0,072] | IC95 de Fusion − RF, sesión (todo) | comparacion/resultados/validacion_por_sesion.json |
| `\DifFusionRFSesionLimpio` | +0,028 | Fusion − RF, una sesión afuera (limpio) | comparacion/resultados/validacion_por_sesion.json (intervalos.limpio.diferencia_vs_fusion...) |
| `\DifFusionRFSesionLimpioIC` | [−0,028; +0,117] | IC95 de Fusion − RF, sesión (limpio) | comparacion/resultados/validacion_por_sesion.json |
| `\DifFusionMILDev` | +0,147 | Fusion − MIL, 5×4 todo | comparacion/resultados/comparaciones_pareadas.json (fusion_tardia_pose_visual vs mil_multimodal_corregido, todo) |
| `\DifFusionMILDevIC` | [+0,100; +0,193] | IC95 de Fusion − MIL, 5×4 todo | comparacion/resultados/comparaciones_pareadas.json |
| `\DifFusionMILLimpio` | +0,122 | Fusion − MIL, 5×4 limpio | comparacion/resultados/comparaciones_pareadas.json (fusion_tardia_pose_visual vs mil_multimodal_corregido, limpio) |
| `\DifFusionMILLimpioIC` | [+0,020; +0,183] | IC95 de Fusion − MIL, 5×4 limpio | comparacion/resultados/comparaciones_pareadas.json |
| `\DifFusionMILSesion` | +0,174 | Fusion − MIL, una sesión afuera (todo) | comparacion/resultados/validacion_por_sesion.json (intervalos.todo.diferencia_vs_fusion...) |
| `\DifFusionMILSesionIC` | [+0,105; +0,245] | IC95 de Fusion − MIL, sesión (todo) | comparacion/resultados/validacion_por_sesion.json |
| `\DifFusionMILSesionLimpio` | +0,156 | Fusion − MIL, una sesión afuera (limpio) | comparacion/resultados/validacion_por_sesion.json (intervalos.limpio.diferencia_vs_fusion...) |
| `\DifFusionMILSesionLimpioIC` | [+0,039; +0,253] | IC95 de Fusion − MIL, sesión (limpio) | comparacion/resultados/validacion_por_sesion.json |
| `\DifFusionBaseDev` | +0,166 | Fusion − Base, 5×4 todo | comparacion/resultados/comparaciones_pareadas.json (fusion_tardia_pose_visual vs baseline_logreg_media_frames, todo) |
| `\DifFusionBaseDevIC` | [+0,117; +0,220] | IC95 de Fusion − Base, 5×4 todo | comparacion/resultados/comparaciones_pareadas.json |
| `\DifFusionBaseLimpio` | +0,143 | Fusion − Base, 5×4 limpio | comparacion/resultados/comparaciones_pareadas.json (fusion_tardia_pose_visual vs baseline_logreg_media_frames, limpio) |
| `\DifFusionBaseLimpioIC` | [+0,065; +0,213] | IC95 de Fusion − Base, 5×4 limpio | comparacion/resultados/comparaciones_pareadas.json |
| `\DifFusionBaseSesion` | +0,236 | Fusion − Base, una sesión afuera (todo) | comparacion/resultados/validacion_por_sesion.json (intervalos.todo.diferencia_vs_fusion...) |
| `\DifFusionBaseSesionIC` | [+0,177; +0,301] | IC95 de Fusion − Base, sesión (todo) | comparacion/resultados/validacion_por_sesion.json |
| `\DifFusionBaseSesionLimpio` | +0,210 | Fusion − Base, una sesión afuera (limpio) | comparacion/resultados/validacion_por_sesion.json (intervalos.limpio.diferencia_vs_fusion...) |
| `\DifFusionBaseSesionLimpioIC` | [+0,126; +0,294] | IC95 de Fusion − Base, sesión (limpio) | comparacion/resultados/validacion_por_sesion.json |
| `\DifRFMILDev` | +0,121 | RF − MIL, 5×4 todo | comparacion/resultados/comparaciones_pareadas.json (rf_pose_tipo_simba vs mil_multimodal_corregido, todo) |
| `\DifRFMILDevIC` | [+0,070; +0,172] | IC95 de RF − MIL, 5×4 todo | comparacion/resultados/comparaciones_pareadas.json |
| `\DifRFMILLimpio` | +0,100 | RF − MIL, 5×4 limpio | comparacion/resultados/comparaciones_pareadas.json (rf_pose_tipo_simba vs mil_multimodal_corregido, limpio) |
| `\DifRFMILLimpioIC` | [−0,003; +0,168] | IC95 de RF − MIL, 5×4 limpio | comparacion/resultados/comparaciones_pareadas.json |
| `\DifRFBaseDev` | +0,140 | RF − Base, 5×4 todo | comparacion/resultados/comparaciones_pareadas.json (rf_pose_tipo_simba vs baseline_logreg_media_frames, todo) |
| `\DifRFBaseDevIC` | [+0,079; +0,202] | IC95 de RF − Base, 5×4 todo | comparacion/resultados/comparaciones_pareadas.json |
| `\DifRFBaseLimpio` | +0,121 | RF − Base, 5×4 limpio | comparacion/resultados/comparaciones_pareadas.json (rf_pose_tipo_simba vs baseline_logreg_media_frames, limpio) |
| `\DifRFBaseLimpioIC` | [+0,031; +0,210] | IC95 de RF − Base, 5×4 limpio | comparacion/resultados/comparaciones_pareadas.json |
| `\ParClips` | 5 | Clips del par 123/169 | equinos.comparacion.Datos (clips de los caballos 123 y 169) |
| `\ParRepsSeparado` | 4 | Repeticiones del 5×4 en que el par quedó separado entre train y test (de 5) | comparacion/DECISION.md |
| `\ParFusionAntes` | 0,8620 | Fusión pose + visual: F1 agrupado 5×4 con los 291 clips (4 decimales) | calculado de predicciones_oof (291 clips) |
| `\ParFusionDespues` | 0,8633 | Ídem sin los 5 clips del par | calculado de predicciones_oof (sin los 5 clips) |
| `\ParFusionDif` | +0,001 | Cambio al quitar el par | calculado |
| `\ParRFAntes` | 0,8361 | RF de pose tipo SimBA: F1 agrupado 5×4 con los 291 clips (4 decimales) | calculado de predicciones_oof (291 clips) |
| `\ParRFDespues` | 0,8366 | Ídem sin los 5 clips del par | calculado de predicciones_oof (sin los 5 clips) |
| `\ParRFDif` | +0,000 | Cambio al quitar el par | calculado |
| `\ParMILAntes` | 0,7152 | MIL multimodal corregido: F1 agrupado 5×4 con los 291 clips (4 decimales) | calculado de predicciones_oof (291 clips) |
| `\ParMILDespues` | 0,7115 | Ídem sin los 5 clips del par | calculado de predicciones_oof (sin los 5 clips) |
| `\ParMILDif` | −0,004 | Cambio al quitar el par | calculado |
| `\ParBaseAntes` | 0,6957 | Baseline LogReg (media de frames): F1 agrupado 5×4 con los 291 clips (4 decimales) | calculado de predicciones_oof (291 clips) |
| `\ParBaseDespues` | 0,6930 | Ídem sin los 5 clips del par | calculado de predicciones_oof (sin los 5 clips) |
| `\ParBaseDif` | −0,003 | Cambio al quitar el par | calculado |
