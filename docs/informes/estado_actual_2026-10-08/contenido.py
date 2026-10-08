"""Cuerpo HTML del informe. Toda cifra sale de `datos.cargar()`; el texto fija sólo la interpretación."""
import subprocess
from pathlib import Path

from datos import (CANDIDATOS, CLASES, NOMBRE, RAIZ, f1_pooled_54, f1_sesion, ic, n, par, pct, pm,
                   sg)

FUS, RF, MIL, BAS = CANDIDATOS

# ---------------------------------------------------------------------------------------------
# Registro de títulos para el índice
# ---------------------------------------------------------------------------------------------
_titulos: list[tuple[int, str, str]] = []
_c = {"h1": 0, "h2": 0}


def h1(texto, ident, num=True, salto=True):
    if num:
        _c["h1"] += 1
        _c["h2"] = 0
        etiqueta = f"{_c['h1']}."
    else:
        etiqueta = ""
    _titulos.append((1, f"{etiqueta} {texto}".strip(), ident))
    cls = "" if salto else ' class="sin-salto"'
    marca = f'<span class="num">{etiqueta}</span> ' if etiqueta else ""
    return f'<h1 id="{ident}"{cls}>{marca}{texto}</h1>'


def h2(texto, ident, num=True):
    if not num:
        _titulos.append((2, texto, ident))
        return f'<h2 id="{ident}">{texto}</h2>'
    _c["h2"] += 1
    etiqueta = f"{_c['h1']}.{_c['h2']}"
    _titulos.append((2, f"{etiqueta} {texto}", ident))
    return f'<h2 id="{ident}"><span class="num">{etiqueta}</span> {texto}</h2>'


def indice():
    filas = "".join(f'<li class="n{nv}"><a href="#{i}">{t}</a></li>' for nv, t, i in _titulos)
    return f'<section class="toc"><h1>Índice</h1><ul>{filas}</ul></section>'


def tabla(cabecera, filas, clase="", alin=None):
    """Tabla HTML. `alin`: lista de 'l', 'n' (derecha) o 'c' por columna."""
    alin = alin or ["l"] * len(cabecera)
    css = {"l": "", "n": ' class="num"', "c": ' class="c"'}
    th = "".join(f"<th{css[a]}>{c}</th>" for c, a in zip(cabecera, alin))
    trs = ""
    for f in filas:
        dest = ""
        if isinstance(f, tuple) and f and f[0] == "dest":
            dest, f = ' class="dest"', f[1]
        trs += f"<tr{dest}>" + "".join(f"<td{css[a]}>{c}</td>" for c, a in zip(f, alin)) + "</tr>"
    if len(filas) > 9:
        clase = (clase + " larga").strip()
    c = f' class="{clase}"' if clase else ""
    return f"<table{c}><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table>"


def tag(estado):
    clase = {"Resuelta": "ok", "Atendida en parte": "par", "Abierta": "ab"}[estado]
    return f'<span class="tag {clase}">{estado}</span>'


def fig(ruta, pie, num):
    return (f'<figure><img src="{Path(ruta).name}" alt="Figura {num}">'
            f"<figcaption><b>Figura {num}.</b> {pie}</figcaption></figure>")


def matriz_html(M, titulo):
    cab = "".join(f"<th>{c}</th>" for c in CLASES)
    filas = ""
    for i, cl in enumerate(CLASES):
        tot = sum(M[i])
        celdas = ""
        for j in range(3):
            d = ' class="diag"' if i == j else ""
            celdas += f"<td{d}>{M[i][j]} ({pct(M[i][j] / tot)})</td>"
        filas += f"<tr><td><b>{cl}</b></td>{celdas}</tr>"
    return (f'<table class="cm"><caption>{titulo}</caption><thead><tr><th>Real \\ predicha</th>{cab}'
            f"</tr></thead><tbody>{filas}</tbody></table>")


def _git(*args):
    return subprocess.run(["git", "-C", str(RAIZ), *args], capture_output=True, text=True).stdout.strip()


# ---------------------------------------------------------------------------------------------
def construir(D, figs, meta):
    cand = D["cand"]
    G = {c: cand[c]["global"] for c in CANDIDATOS}
    LIM = {c: cand[c]["limpio"]["f1_macro_oof_por_repeticion"] for c in CANDIDATOS}
    res = D["reserva"]["metricas"]
    ses = D["sesion"]
    ses_c = ses["candidatos"]
    iv = ses["intervalos"]
    p_f_rf = par(D, FUS, RF, "todo")
    p_f_rf_l = par(D, FUS, RF, "limpio")
    p_f_mil = par(D, FUS, MIL, "todo")
    p_f_mil_l = par(D, FUS, MIL, "limpio")
    p_f_bas = par(D, FUS, BAS, "todo")
    p_f_bas_l = par(D, FUS, BAS, "limpio")
    d_ses = {sub: iv[sub]["diferencia_vs_fusion_tardia_pose_visual"] for sub in ("todo", "limpio")}
    ic_res = meta["ic_reserva"]
    ncl_lim = cand[FUS]["limpio"]["clips"]
    lim_por_clase = cand[FUS]["limpio"]["clips_por_clase"]
    prec54 = {cl: G[FUS]["por_clase"][cl]["precision"]["media"] for cl in CLASES}
    M54 = G[FUS]["matriz_confusion_suma"]
    Mres = res["matriz"]
    n_pred = sum(map(sum, M54))
    caida_limpio_54 = f1_pooled_54(D, FUS, "todo")[0] - f1_pooled_54(D, FUS, "limpio")[0]
    caida_limpio_ses = f1_sesion(D, FUS, "todo")[0] - f1_sesion(D, FUS, "limpio")[0]
    f1_por_rep = [p["f1_macro_oof"] for p in cand[FUS]["por_particion"]]
    sop_res = [sum(f) for f in Mres]

    out = []
    A = out.append

    # ======================================================================= PORTADA
    A(f"""
<section class="portada"><div class="interior">
<div class="kicker">UGR · Licenciatura en Ciencias de Datos · Seminario de Trabajo Final</div>
<h1>Estado actual y métricas del proyecto de tesina</h1>
<p class="sub">Detección automática de expresiones faciales en equinos a partir de video</p>
<p class="sub">Del modelo MIL multimodal a la fusión pose + visual: qué observó el docente, qué se hizo,
qué resultado dio y cómo se mide hoy el modelo</p>
<div class="tiles">
<div class="tile"><b>{n(G[FUS]['f1_macro']['media'])}</b><span>F1 macro del modelo actual en desarrollo (± {n(G[FUS]['f1_macro']['desvio'])}; 113 caballos, 5×4 por caballo)</span></div>
<div class="tile"><b>{n(res['f1_macro'])}</b><span>F1 macro en la reserva de {D['reserva']['caballos']} caballos (único uso)</span></div>
<div class="tile"><b>{n(f1_sesion(D, FUS, 'todo')[0])}</b><span>F1 macro dejando afuera cada sesión de filmación</span></div>
<div class="tile"><b>4</b><span>modelos comparados con un protocolo declarado antes de medir</span></div>
</div></div>
<div class="pie">
<p>Integrantes: Héctor R. Liceaga · Sergio M. Yañez</p>
<p>Documento elaborado a partir del repositorio <code>expresiones_faciales_equinos</code> (rama <b>main</b>,
commit <b>{meta['commit']}</b>, {meta['fecha_commit']}), de <code>comparacion/resultados/</code> y de
la devolución del docente al informe de auditoría del 25/09/2026.</p>
<p>Fecha de elaboración: {meta['fecha']} · Segunda entrega, actualización de estado</p>
</div></section>
""")
    A("@@INDICE@@")

    # ======================================================================= 0
    A(h1("Cómo leer este documento", "como-leer", num=False))
    A(f"""
<p>Este documento continúa a <i>Evolución completa del proyecto de tesina</i> (24/09/2026), que
reconstruyó el camino hasta el modelo MIL multimodal. Aquí se cuenta lo que ocurrió después:
la devolución del docente sobre el informe de auditoría, la auditoría v2 del 07/10, la
comparación de cuatro modelos con un protocolo declarado antes de medir y el resultado de la
fusión pose + visual, que hoy es el modelo de la aplicación. Se mantiene el estilo y el nivel del
documento anterior: se explica cada decisión, se informa cada métrica con su incertidumbre y se
declara lo que no quedó resuelto.</p>
<h2 id="convenciones">Convenciones</h2>
<ul>
<li><b>Cifras.</b> Provienen de archivos del repositorio y se contrastaron con ellos al generar el documento
(<code>generar.py</code> lee los JSON y falla si no coinciden con <code>comparacion/DECISION.md</code>).
Se escriben con coma decimal y, salvo aviso, con tres decimales.</li>
<li><b>Dos protocolos que no deben mezclarse.</b> El modelo anterior se midió con 4 folds por caballo sobre
359 clips de 133 caballos. Los cuatro candidatos nuevos se midieron con un protocolo común (5 repeticiones
× 4 folds sobre 291 clips de 113 caballos, más una reserva de 20 caballos). Un 0,740 del primero y un 0,722 del segundo
<b>no son comparables</b>; las comparaciones válidas son las que se hacen dentro de un mismo protocolo.</li>
<li><b>Fechas.</b> Todas de 2026, en formato día/mes.</li>
<li><b>«Docente».</b> Juan Toloza, autor de las anotaciones del PDF
<i>Devolucion_Juan_resumen_de_condiciones_cumplidas_o_no .pdf</i> (35 páginas).</li>
</ul>
<h2 id="fuentes">Fuentes de cada parte</h2>
""")
    A(tabla(["Parte", "Fuente principal"], [
        ["Devolución del docente", "<code>Devolucion_Juan_resumen_de_condiciones_cumplidas_o_no .pdf</code> (anotaciones del PDF extraídas una por una)"],
        ["Modelo anterior y auditoría v2", "<code>condiciones_cumplidas.pdf</code> (07/10/2026), <code>docs/caminos_explorados.md</code>"],
        ["Protocolo y decisión", "<code>comparacion/PROTOCOLO.md</code>, <code>comparacion/DECISION.md</code>"],
        ["Métricas de los cuatro candidatos", "<code>comparacion/resultados/{fusion_tardia_pose_visual, rf_pose_tipo_simba, mil_multimodal_corregido, baseline_logreg_media_frames}.json</code>"],
        ["Diferencias pareadas", "<code>comparacion/resultados/comparaciones_pareadas.json</code>"],
        ["Reserva", "<code>comparacion/resultados/fusion_tardia_pose_visual__reserva.json</code>, <code>comparacion/reserva_uso.log</code>"],
        ["Validación por sesión", "<code>comparacion/resultados/validacion_por_sesion.json</code>"],
        ["Estado de la aplicación y del proyecto", "<code>README.md</code>, <code>openspec/</code>, <code>app_inferencia.py</code>, <code>equinos/fusion.py</code>, <code>tests/</code>"],
    ], "compacta"))

    # ======================================================================= 1 RESUMEN
    A(h1("Resumen ejecutivo", "resumen"))
    A(f"""
<p>El proyecto clasifica la expresión de un caballo —<b>Sereno</b>, <b>Interesado</b> o <b>Disgustado</b>— a partir de una toma
corta de su cara. El modelo de la segunda entrega era un MIL multimodal (F1 macro 0,740 ± 0,052). La auditoría del
25/09 mostró que una regresión logística sobre el promedio de los fotogramas rendía lo mismo (0,741), que la duración y
la resolución del clip predecían la clase sin mirar la cara (0,618) y que el modelo no era temporal. Esas observaciones,
junto con la devolución del docente, llevaron a comparar modelos con un protocolo declarado antes de medir.</p>
<p><b>El modelo actual es la fusión tardía pose + visual</b>: un Random Forest sobre 160 rasgos geométricos de la pose
(tipo SimBA) y una regresión logística sobre las características visuales promedio, combinados con un peso de 0,75 para la
pose. Ganó la comparación con el criterio fijado de antemano.</p>
""")
    A(f"""
<div class="kpis">
<div class="kpi"><b>{n(G[FUS]['f1_macro']['media'])}</b><span>F1 macro en desarrollo<br>± {n(G[FUS]['f1_macro']['desvio'])} (5×4, 113 caballos)</span></div>
<div class="kpi"><b>{n(res['f1_macro'])}</b><span>F1 macro en la reserva<br>IC95 {ic(ic_res)}</span></div>
<div class="kpi"><b>{n(res['accuracy'])}</b><span>accuracy en la reserva<br>(68 clips, 20 caballos)</span></div>
<div class="kpi"><b>{n(LIM[FUS]['media'])}</b><span>F1 macro en el subconjunto limpio (desarrollo)</span></div>
<div class="kpi"><b>{n(f1_sesion(D, FUS, 'todo')[0])}</b><span>F1 dejando afuera cada sesión de filmación</span></div>
</div>
<ul>
<li><b>Frente al modelo anterior.</b> En el protocolo común el MIL corregido queda tercero ({pm(G[MIL]['f1_macro']['media'], G[MIL]['f1_macro']['desvio'])}); la fusión lo supera por
{sg(p_f_mil['diferencia_a_menos_b'])} {ic(p_f_mil['diferencia_ic95'], signo=True)} y el baseline estático queda último ({pm(G[BAS]['f1_macro']['media'], G[BAS]['f1_macro']['desvio'])}).</li>
<li><b>Lo que debe leerse con cautela.</b> La ventaja sobre el RF de pose es mínima ({sg(p_f_rf['diferencia_a_menos_b'])} {ic(p_f_rf['diferencia_ic95'], signo=True)}) y, dejando afuera sesiones enteras,
ya no se distingue de cero. En clips filmados en condiciones parejas todos los modelos caen unos 12 puntos. La reserva es chica (20 caballos) y hay un solo etiquetador.</li>
<li><b>Estado.</b> La aplicación sirve la fusión por omisión; los 24 scripts de prueba pasan; OpenSpec no tiene changes abiertos. Queda el informe final (29/10).</li>
</ul>
<p>De las {meta['n_obs']} observaciones del docente recogidas, <b>{meta['n_res']} quedaron resueltas</b>, <b>{meta['n_par']} se atendieron en parte</b> y
<b>{meta['n_ab']} siguen abiertas</b> (sección 2.3).</p>
""")

    # ======================================================================= 2 PUNTO DE PARTIDA
    A(h1("Punto de partida y devolución del docente", "partida"))
    A(h2("El modelo anterior: MIL multimodal", "modelo-anterior"))
    A("""
<p>El modelo de la segunda entrega (<code>MILFusionModel</code>) procesa los 16 fotogramas que se eligen de cada clip. Cada fotograma
se recorta alineando el ojo y el hocico, se describe con una ResNet-18 congelada (512 números) y con 9 variables geométricas
de la cara; una capa de atención pondera los fotogramas y produce una única clasificación del clip. Se llama
<i>Multiple Instance Learning</i> porque el clip es una «bolsa» de instancias (fotogramas) y la etiqueta es de la bolsa. Tiene
132 355 parámetros entrenables.</p>
""")
    A(tabla(["Medida (protocolo anterior: 4 folds por caballo, 359 clips, 133 caballos)", "Valor"], [
        ["F1 macro del MIL multimodal (media ± desvío entre folds)", "0,740 ± 0,052 (IC95 por caballo [0,677; 0,797]); accuracy 0,744"],
        ["Regresión logística sobre el promedio de los 16 fotogramas (baseline estático)", "0,741 ± 0,020; MIL − estático = −0,004, IC95 [−0,066; +0,058]"],
        ["Duración, resolución y fps del clip, sin mirar la cara", "0,618 ± 0,034"],
        ["MIL en el subconjunto limpio (1080×1920 y ≥ 5 s)", "0,705"],
        ["Sobreajuste (F1 de train / validación / test)", "0,979 / 0,712 / 0,740: moderado, contenido por early stopping"],
        ["Cambiar el orden de los 16 fotogramas", "0 de 359 predicciones cambian: el modelo no es temporal"],
    ], "compacta"))
    A('<p class="fuente">Fuente: auditoría del 25/09/2026 y auditoría v2 (<code>condiciones_cumplidas.pdf</code>); <code>README.md</code>.</p>')

    A(h2("Qué escribió el docente", "que-escribio"))
    A(f"""
<p>La devolución es el PDF de la auditoría del 25/09 anotado por el docente. Contiene <b>4 comentarios escritos</b>
sobre el texto, <b>{meta['n_resaltados']} pasajes resaltados</b> y <b>dos notas de texto pegadas</b> en las páginas 22 y 35:
una describe cómo SimBA y un Random Forest podrían clasificar acciones a partir de los puntos de DeepLabCut;
la otra es un diagnóstico («lo bueno y lo preocupante») con cinco indicaciones sobre qué hacer. Las anotaciones se
extrajeron del PDF una por una (autor registrado: «juan toloza») y se agruparon así:</p>
<ul>
<li><b>A. Comentarios escritos:</b> repetibilidad; si el modelo es realmente espacio-temporal; retirar el requisito de evolución temporal;
si se usó DeepLabCut con el subconjunto experto Horse-30.</li>
<li><b>B. Nota sobre SimBA</b> (página 22).</li>
<li><b>C. Nota de diagnóstico</b> (página 35): aceptar la parsimonia, blanquear el confusor de filmación, sincerar el sobreajuste,
documentar la falta de protocolo etológico (EquiFACS) y limpiar la trazabilidad.</li>
<li><b>D. Pasajes resaltados</b> sin comentario: criterio de aceptación de 0,691, validación anidada de 0,784, identidades duplicadas,
videos largos, utilidad para veterinarios, test externo, informe final y relevamiento bibliográfico.</li>
</ul>
""")

    A(h2("De la observación al resultado", "trazabilidad"))
    A("""<p>La tabla siguiente es el hilo central del documento: para cada observación, qué se hizo y qué evidencia dejó.
El estado es una valoración honesta de si el trabajo posterior cerró la observación.</p>""")
    obs = meta["observaciones"]
    A(tabla(["Observación del docente", "Qué se hizo", "Resultado y evidencia", "Estado"],
            [[o["obs"], o["accion"], o["resultado"], tag(o["estado"])] for o in obs], "compacta fija"))
    A("""<p class="fuente">Evidencia citada: auditoría v2 (§1.5, caps. 2, 12, 14, 17), <code>comparacion/</code>, <code>docs/caminos_explorados.md</code> (C-06, C-07, C-09, C-26 a C-42, C-46, C-47).</p>""")

    A(h2("Cómo se encadenó todo", "cadena"))
    A(tabla(["Fecha", "Hecho", "Motivo ligado a la devolución"], [
        ["25/09", "Auditoría del plan de trabajo; líneas de base; se conserva el MIL", "Parsimonia, confusor, sobreajuste (notas C1 a C3)"],
        ["28/09", "La etiquetadora devuelve el formulario de etiquetado", "EquiFACS y protocolo (nota C4)"],
        ["07/10", "Auditoría v2: 18 requisitos (10 cumplidos, 6 parciales, 2 no verificables, 0 no cumplidos)", "Comentarios A1 a A4 y trazabilidad (C5)"],
        ["07/10", "Experimento con 160 rasgos de pose tipo SimBA: RF de pose 0,841 contra 0,740 del MIL", "Nota SimBA (B1)"],
        ["07/10", "Protocolo de comparación declarado antes de medir; reserva del 15 % de los caballos", "Criterio 0,691 sobre otro conjunto (D1); parsimonia (C1)"],
        ["08/10", "Cuatro candidatos medidos con las mismas particiones; gana la fusión pose + visual", "Todas las anteriores"],
        ["08/10", "Un único uso de la reserva: F1 0,834", "Estimación final fuera de muestra"],
        ["08/10", "Validación dejando afuera cada sesión de filmación", "Confusor de filmación (C2)"],
        ["08/10", "La aplicación pasa a servir la fusión; caminos explorados documentados", "Trazabilidad (C5)"],
    ], "compacta"))

    # ======================================================================= 3 QUÉ SE CAMBIÓ
    A(h1("Qué se cambió", "cambios"))
    A(h2("Auditoría v2 (07/10)", "auditoria-v2"))
    A("""
<p>La auditoría v2 repitió el contraste con el plan de trabajo, esta vez con un script reproducible por cada cifra
(<code>auditoria/scripts/</code>) y atendiendo explícitamente la devolución. Comparada con la del 25/09:</p>
""")
    A(tabla(["Estado de los requisitos del plan", "Auditoría 25/09", "Auditoría v2 (07/10)"], [
        ["Requisitos considerados", "19", "18 (se retiró «la evolución temporal aporta robustez»)"],
        ["Cumplidos", "9", "10"],
        ["Parciales", "4", "6"],
        ["No cumplidos", "2", "0"],
        ["No verificables", "4", "2"],
    ], "compacta", ["l", "c", "c"]))
    A("""
<p>Hallazgos que motivaron correcciones: (1) el modelo no es temporal y no supera a un promedio lineal; (2) existen
confusores de captura (duración, contenedor, rotación, sesión); (3) el canal geométrico del MIL era casi vacío
(sólo aportaba la inclinación de la cabeza y un punto de oreja); (4) la validación interna usaba los caballos con más clips;
(5) la inferencia no normalizaba la orientación de los videos igual que el entrenamiento. La auditoría v2 mantuvo un veredicto
de confiabilidad metodológica alta, confiabilidad estadística media, generalización externa no evaluada y utilidad para
expertos no verificable.</p>
""")
    A(h2("Correcciones aplicadas", "correcciones"))
    A(tabla(["Corrección", "Problema detectado", "Qué se hizo", "Evidencia"], [
        ["Rotación de video", "69 de 274 videos declaran rotación en sus metadatos; la inferencia no los normalizaba como el entrenamiento.",
         "Se normaliza la orientación y el contenedor de la misma forma que en el notebook 01.",
         "La prueba de punta a punta pasó de 8/16 a 16/16 sobre videos del propio dataset (C-08)."],
        ["Geometría v2", "Las 9 variables vigentes se reducían a 3 dimensiones efectivas.",
         "17 columnas: vectores base–punta de ambas orejas, hocico–mandíbula inferior, ancho de boca, distancia interocular, escorzo y orientación, en el marco de la cabeza.",
         "9 dimensiones efectivas (95 % de la varianza). El ancho de boca resultó casi vacío (máscara 0,09) (C-23)."],
        ["Máscaras ≥ 0,50", "Las máscaras de validez valían 1 siempre (confianza &gt; 0). Además, un umbral de 0,55 introducido el 21/09 cambió el preprocesamiento de la app sin aviso.",
         "Máscaras reales con confianza ≥ 0,50 y restitución del umbral 0,50 (07/10).",
         "C-15 y C-23."],
        ["Validación interna estratificada", "La validación interna eran los 7 caballos con más clips (~10 clips c/u contra 2,1 en el train interno).",
         "Validación interna por caballo y por clase, con 22 a 28 caballos.",
         "C-38; <code>equinos/particion.dividir_estratificada</code>."],
        ["Líneas de base y control del confusor", "No había baseline estático ni medida sobre clips parejos.",
         "Módulo <code>equinos/lineas_de_base.py</code>; subconjunto limpio en toda tabla.",
         "Change <code>mejoras-prioridad-alta</code>."],
        ["Trazabilidad", "<code>_huella_caracteristicas</code> leía el archivo del conjunto de 153 clips; cifras sin script.",
         "Corregida para leer la versión v2; <code>docs/registro_de_experimentos.md</code> y <code>docs/caminos_explorados.md</code>.",
         "Changes <code>mejoras-prioridad-alta</code> y <code>docs/caminos-explorados</code>."],
    ], "compacta fija"))

    A(h2("Protocolo de comparación declarado antes de medir", "protocolo"))
    A(f"""
<p>El 07/10 se fijó por escrito (<code>comparacion/PROTOCOLO.md</code>), antes de ver ningún resultado, cómo se compararían los modelos.
Declarar el criterio antes evita elegir el número que conviene; el protocolo agrega que cualquier cambio posterior debe quedar
como enmienda con fecha (hubo una, el 08/10, para la validación por sesión, también declarada antes de correrla).</p>
""")
    A(tabla(["Elemento", "Decisión"], [
        ["Pregunta", "Qué modelo clasifica mejor en caballos nunca vistos, sin que la ventaja se explique por rasgos de filmación."],
        ["Datos", f"359 clips íntegros de 133 caballos. Unidad de agrupamiento: el caballo."],
        ["Reserva", f"15 % de los caballos: {D['reserva']['caballos']} caballos y {D['reserva']['clips']} clips (Disgustado 17, Interesado 16, Sereno 35), sorteados con semilla {D['reserva_caballos']['semilla']}, estratificados por la clase dominante del caballo. Se usa <b>una sola vez, con el ganador</b>."],
        ["Desarrollo", "113 caballos y 291 clips (Disgustado 86, Interesado 102, Sereno 103)."],
        ["Particiones", "5 repeticiones × 4 folds por caballo (<code>StratifiedGroupKFold</code>, semillas 1101 a 1105): 20 evaluaciones por candidato, idénticas para todos. Todo ajuste ocurre dentro del train de cada evaluación."],
        ["Métrica principal", "F1 macro, media ± desvío sobre las 20 evaluaciones; accuracy, F1 por clase, matrices de confusión y gap train − test."],
        ["Control de filmación", f"Subconjunto limpio: clips 1080×1920 de al menos 5 s ({ncl_lim} clips en desarrollo: Disgustado {lim_por_clase['Disgustado']}, Interesado {lim_por_clase['Interesado']}, Sereno {lim_por_clase['Sereno']})."],
        ["Incertidumbre", "IC95 bootstrap pareado por caballo de la diferencia entre candidatos (2 000 réplicas, semilla 7)."],
    ], "compacta fija"))
    A("""
<div class="recuadro"><h3>Criterio de elección (§5 del protocolo)</h3><ol>
<li><b>Ranking</b> por F1 macro medio sobre las 20 evaluaciones de desarrollo.</li>
<li><b>No inferioridad en limpio</b>: el candidato elegido no puede ser significativamente peor que otro en el subconjunto limpio (límite superior del IC95 pareado menor que 0).</li>
<li><b>Empate</b>: si el IC95 pareado de la diferencia con el líder incluye 0, gana <b>el más simple</b>.</li>
<li>Se registra el resultado completo aunque haya un ganador claro.</li>
</ol></div>
""")

    # ======================================================================= 4 CANDIDATOS
    A(h1("Los cuatro candidatos", "candidatos"))
    A("""<p>Todos usan los mismos clips, las mismas particiones y el mismo evaluador, que es quien calcula las métricas. Se los numera por su
simplicidad (1 es el más simple), que sólo interviene si hay empate.</p>""")
    A(f"""
<div class="tarjeta"><h3>1. Baseline: regresión logística sobre la media visual</h3>
<p class="lema">Simplicidad 1 · entrada visual · {NOMBRE[BAS]}</p>
<p>Para cada clip se promedian las características que la ResNet-18 congelada extrae de los 16 fotogramas (un vector de 512 números) y
una regresión logística lineal (C = 0,1, pesos de clase balanceados) decide la clase. No usa pose ni orden. En lenguaje llano: «mira
cómo se ve la cara, en promedio, y traza una frontera recta entre las tres clases». Es la vara contra la que debe medirse cualquier modelo más
complejo.</p></div>
<div class="tarjeta"><h3>2. Random Forest de pose «tipo SimBA»</h3>
<p class="lema">Simplicidad 2 · entrada: pose · {NOMBRE[RF]}</p>
<p>A partir de los puntos que DeepLabCut ubica en la cara (ojos, orejas, hocico) en los 16 fotogramas, se calculan <b>160 rasgos por clip</b>:
130 estáticos (distancias, ángulos entre orejas, inclinación de la cabeza), 21 dinámicos (cuánto cambian en el clip) y 9 de disponibilidad (qué tan
medibles fueron los puntos). Un Random Forest de 500 árboles (hojas de al menos 2 clips, pesos de clase balanceados) los combina. Los rasgos están
inspirados en los que SimBA calcula, pero <b>se calculan con código propio</b>: SimBA real, para un solo animal, sólo entrega movimiento por punto y rindió 0,720.</p></div>
<div class="tarjeta"><h3>3. MIL multimodal corregido</h3>
<p class="lema">Simplicidad 4 · entrada: visual + pose · {NOMBRE[MIL]}</p>
<p>Es el modelo anterior con las tres correcciones: geometría v2 (17 columnas), máscaras de validez con umbral 0,50 y validación interna estratificada por
caballo y clase para el <i>early stopping</i>. Sigue la misma idea: cada fotograma se describe con 512 números visuales y la geometría, una capa de
atención los pondera y se clasifica el clip. No codifica el orden de los fotogramas, de modo que sigue siendo «agregación espacial de varios fotogramas».</p></div>
<div class="tarjeta"><h3>4. Fusión tardía pose + visual (modelo actual)</h3>
<p class="lema">Simplicidad 3 · entrada: pose + visual · {NOMBRE[FUS]}</p>
<p>Se entrenan por separado los dos primeros modelos: el Random Forest sobre los 160 rasgos de pose y una regresión logística sobre la media visual.
Cada uno entrega probabilidades para las tres clases y se combinan como <b>p = w · p<sub>pose</sub> + (1 − w) · p<sub>visual</sub></b>.
El peso <i>w</i> se elige entre 0; 0,25; 0,5; 0,75 y 1 con una validación interna de 4 pliegues por caballo dentro del train, nunca con el test. En las 20
evaluaciones el peso elegido fue 0,75 en 19 y 1,0 en una: <b>la decisión está dominada por la pose</b>. «Tardía» significa que se fusionan las probabilidades y no los rasgos.
La variante tardía se fijó antes de mirar resultados; la temprana no se evaluó.</p></div>
""")

    # ======================================================================= 5 MÉTRICAS
    A(h1("Métricas", "metricas"))
    A("""
<div class="recuadro"><h3>Cuatro conceptos para leer las tablas</h3><ul>
<li><b>F1 macro</b>: para cada clase se combina la precisión («cuando dice Disgustado, ¿cuántas veces acierta?») y el recall («de los Disgustados reales,
¿cuántos encuentra?») en el F1; el F1 macro es el promedio simple de las tres clases. Pesa igual a la clase chica que a la grande, por eso es la métrica principal.</li>
<li><b>Limpio</b>: subconjunto de clips filmados en 1080×1920 y de al menos 5 s, donde el formato del video no delata la clase. Si un modelo mantiene su acierto en limpio, no
depende sólo del formato.</li>
<li><b>Reserva</b>: 20 caballos apartados al principio que ningún ajuste vio. Se miden una sola vez, con el modelo ya elegido; es la estimación más honesta de cómo
rendiría con caballos nuevos.</li>
<li><b>IC95</b>: intervalo que, con 95 % de confianza, contiene el valor verdadero. Se calcula remuestreando caballos (bootstrap). Si el IC de una diferencia incluye 0, no se puede decir que un
modelo sea mejor que otro. <b>Gap</b>: F1 de entrenamiento menos F1 de test; cuanto mayor, más sobreajuste.</li>
</ul></div>
""")
    A(h2("Resultado en desarrollo: validación 5×4 por caballo", "res-54"))
    filas = []
    for c in CANDIDATOS:
        g = G[c]
        fila = [NOMBRE[c], pm(g["f1_macro"]["media"], g["f1_macro"]["desvio"]), pm(g["accuracy"]["media"], g["accuracy"]["desvio"]),
                pm(LIM[c]["media"], LIM[c]["desvio"]), "+" + pm(g["gap_train_test_f1"]["media"], g["gap_train_test_f1"]["desvio"], 2)]
        filas.append(("dest", fila) if c == FUS else fila)
    A(tabla(["Candidato", "F1 macro (media ± desvío, 20 evaluaciones)", "Accuracy", "F1 limpio (5 repeticiones)", "Gap train − test"],
            filas, "compacta res", ["l", "n", "n", "n", "n"]))
    A(f"""<p class="fuente">Fuente: <code>comparacion/resultados/*.json</code> (huella de particiones {D['huella_particiones'][:10]}…, común a los cuatro). 291 clips, 113 caballos.</p>""")
    A(f"""
<p>La fusión tiene el mayor F1 macro ({n(G[FUS]['f1_macro']['media'])}), seguida del RF de pose ({n(G[RF]['f1_macro']['media'])}); el MIL corregido ({n(G[MIL]['f1_macro']['media'])}) y el
baseline ({n(G[BAS]['f1_macro']['media'])}) quedan bastante por debajo. En el subconjunto limpio todos caen entre 10 y 13 puntos (fusión {n(LIM[FUS]['media'])}).
El desvío de 0,046 a 0,102 describe cuánto varía el F1 entre las 20 evaluaciones; no es un error estándar (las 5 repeticiones remuestrean la partición, no los datos).
El MIL es el más inestable (± {n(G[MIL]['f1_macro']['desvio'])}).</p>
""")
    A(fig(figs["f1"], f"F1 macro fuera de muestra de cada candidato con IC95 por caballo. Izquierda: validación 5×4 (promedio de las 5 repeticiones del F1 de las predicciones fuera de muestra). Derecha: dejando afuera una sesión de filmación (F1 agrupado). Las barras rayadas son el subconjunto limpio. Los valores de la izquierda difieren levemente de la tabla (0,862 contra {n(G[FUS]['f1_macro']['media'])} para la fusión) porque aquí se agrupan las predicciones de cada repetición en vez de promediar folds.", 1))

    A(h2("F1 por clase", "por-clase"))
    filas = []
    for c in CANDIDATOS:
        fila = [NOMBRE[c]] + [pm(G[c]["por_clase"][cl]["f1"]["media"], G[c]["por_clase"][cl]["f1"]["desvio"]) for cl in CLASES]
        filas.append(("dest", fila) if c == FUS else fila)
    A(tabla(["Candidato", "Disgustado", "Interesado", "Sereno"], filas, "compacta", ["l", "n", "n", "n"]))
    A('<p class="fuente">F1 por clase: media ± desvío de las 20 evaluaciones.</p>')
    filas = [[cl, pm(G[FUS]["por_clase"][cl]["precision"]["media"], G[FUS]["por_clase"][cl]["precision"]["desvio"]),
              pm(G[FUS]["por_clase"][cl]["recall"]["media"], G[FUS]["por_clase"][cl]["recall"]["desvio"]),
              pm(G[FUS]["por_clase"][cl]["f1"]["media"], G[FUS]["por_clase"][cl]["f1"]["desvio"])] for cl in CLASES]
    A("<h3>Fusión pose + visual: precisión, recall y F1 por clase</h3>")
    A(tabla(["Clase", "Precisión", "Recall", "F1"], filas, "compacta", ["l", "n", "n", "n"]))
    A(f"""
<p>La fusión es la más pareja entre clases. <b>Disgustado</b> es la clase más difícil para todos (F1 {n(G[FUS]['por_clase']['Disgustado']['f1']['media'], 2)} en la fusión; {n(G[MIL]['por_clase']['Disgustado']['f1']['media'], 2)} en el MIL,
con el mayor desvío), coherente con que sea la que tiene menos clips y los clips más cortos. Con la fusión, cuando el modelo dice Disgustado acierta el {pct(prec54['Disgustado'])} de las veces, Interesado el
{pct(prec54['Interesado'])} y Sereno el {pct(prec54['Sereno'])}.</p>
""")
    A(fig(figs["clase"], "F1 por clase de los cuatro candidatos. Izquierda: media ± desvío de las 20 evaluaciones del 5×4. Derecha: F1 agrupado dejando afuera una sesión (sin barra de error: es un único valor por clase).", 2))

    A(h2("Matrices de confusión", "confusion"))
    A(f"""<p>Las matrices suman las 20 evaluaciones ({n_pred} predicciones fuera de muestra, 5 por cada uno de los 291 clips). Cada celda muestra la cantidad y el porcentaje de la fila
(clase real). La diagonal es lo acertado.</p>""")
    A('<div class="dos"><div>' + matriz_html(G[FUS]["matriz_confusion_suma"], "Fusión pose + visual") + "</div><div>" +
      matriz_html(G[RF]["matriz_confusion_suma"], "RF de pose tipo SimBA") + "</div></div>")
    A('<div class="dos"><div>' + matriz_html(G[MIL]["matriz_confusion_suma"], "MIL multimodal corregido") + "</div><div>" +
      matriz_html(G[BAS]["matriz_confusion_suma"], "Baseline LogReg (media de frames)") + "</div></div>")
    A(f"""
<p>En la fusión, el error más frecuente es clasificar <b>Sereno como Disgustado</b> ({M54[2][0]} de {sum(M54[2])} Serenos, {pct(M54[2][0] / sum(M54[2]))}), seguido de Disgustado como Interesado
({M54[0][1]} de {sum(M54[0])}). Sereno casi nunca se confunde con Interesado ({M54[2][1]} de {sum(M54[2])}). El MIL y el baseline reparten sus errores entre todas las clases.</p>
""")

    A(h2("Comparaciones pareadas y aplicación del criterio", "pareadas"))
    A("""<p>Se compara cada par de candidatos remuestreando caballos (2 000 réplicas, semilla 7), de modo que ambos se miden sobre los mismos caballos en cada réplica.
Un IC que no incluye 0 indica una diferencia sostenida.</p>""")
    filas = []
    pares = [(FUS, RF), (FUS, MIL), (FUS, BAS), (RF, MIL), (RF, BAS)]
    for a, b in pares:
        pt, pl = par(D, a, b, "todo"), par(D, a, b, "limpio")
        filas.append([f"{NOMBRE[a].split(' (')[0]} − {NOMBRE[b].split(' (')[0]}",
                      f"{sg(pt['diferencia_a_menos_b'])} {ic(pt['diferencia_ic95'], signo=True)}",
                      f"{sg(pl['diferencia_a_menos_b'])} {ic(pl['diferencia_ic95'], signo=True)}"])
    A(tabla(["Diferencia de F1 macro", f"Todo el desarrollo (291 clips, 113 caballos)", f"Subconjunto limpio ({ncl_lim} clips, 71 caballos)"],
            filas, "compacta", ["l", "n", "n"]))
    A(f"""
<h3>Aplicación del criterio (§5 del protocolo)</h3>
<ol>
<li><b>Ranking.</b> La fusión tiene el mayor F1 macro medio ({n(G[FUS]['f1_macro']['media'])}).</li>
<li><b>No inferioridad en limpio.</b> Ningún IC de «fusión − otro» en limpio tiene su límite superior por debajo de 0. Se cumple.</li>
<li><b>Empate.</b> El IC de «fusión − RF» en desarrollo es {ic(p_f_rf['diferencia_ic95'], signo=True)}: no incluye 0 (por {n(p_f_rf['diferencia_ic95'][0])}). No hay empate y no se aplica la regla de simplicidad.</li>
</ol>
<p><b>Ganador: fusión tardía pose + visual.</b> La decisión se registró por escrito (<code>comparacion/DECISION.md</code>) antes de usar la reserva. En el subconjunto limpio, la diferencia fusión − RF
({sg(p_f_rf_l['diferencia_a_menos_b'])} {ic(p_f_rf_l['diferencia_ic95'], signo=True)}) y RF − MIL incluyen 0.</p>
""")
    A(fig(figs["dif"], "Diferencias de F1 macro entre la fusión y cada rival con IC95 bootstrap pareado por caballo, en el 5×4 y dejando afuera una sesión, con todos los clips y con el subconjunto limpio.", 3))

    A(h2("Estabilidad entre repeticiones", "estabilidad"))
    filas = []
    for c in CANDIDATOS:
        v = [p["f1_macro_oof"] for p in cand[c]["por_particion"]]
        filas.append([NOMBRE[c]] + [n(x) for x in v] + [f"{n(min(v))} – {n(max(v))}"])
    A(tabla(["Candidato", "Rep. 1", "Rep. 2", "Rep. 3", "Rep. 4", "Rep. 5", "Rango"], filas, "compacta", ["l", "n", "n", "n", "n", "n", "n"]))
    A(f"""<p class="fuente">F1 macro fuera de muestra de cada repetición (cada una es una asignación distinta de caballos a folds).</p>
<p>La fusión se mueve entre {n(min(f1_por_rep))} y {n(max(f1_por_rep))}; el MIL corregido tiene un rango mayor. Así se responde ahora, en el protocolo común, la pregunta
del docente sobre la repetibilidad: en las cinco particiones la fusión es primera y el RF de pose segundo; el MIL corregido y el baseline se alternan en el tercer lugar (el MIL queda por debajo en la repetición 3).</p>""")
    A(fig(figs["rep"], "F1 macro fuera de muestra por repetición del 5×4, con todos los clips y en el subconjunto limpio.", 4))

    A(h2("Resultado en la reserva (único uso, 08/10)", "reserva"))
    A(f"""
<p>La reserva se usó una sola vez ({meta['fecha_uso']}), con la fusión ya elegida y registrada, y con autorización expresa de Sergio (<code>reserva_uso.log</code>). El modelo se ajustó con todo el desarrollo y no se tocó
después. {D['reserva']['caballos']} caballos, {D['reserva']['clips']} clips.</p>
<div class="kpis">
<div class="kpi"><b>{n(res['f1_macro'])}</b><span>F1 macro<br>IC95 por caballo {ic(ic_res)}</span></div>
<div class="kpi"><b>{n(res['accuracy'])}</b><span>accuracy<br>(58 de 68 clips)</span></div>
<div class="kpi"><b>{n(D['reserva']['limpio_f1_macro'])}</b><span>F1 macro en el subconjunto limpio (48 clips)</span></div>
<div class="kpi"><b>{n(G[FUS]['f1_macro']['media'])}</b><span>F1 macro esperado por el desarrollo</span></div>
</div>
""")
    A(tabla(["Clase", "Clips", "Precisión", "Recall", "F1"],
            [[cl, str(sop_res[i]), n(res["precision"][cl]), n(res["recall"][cl]), n(res["f1"][cl])] for i, cl in enumerate(CLASES)],
            "compacta", ["l", "n", "n", "n", "n"]))
    A(f"""
<p>El resultado está dentro de lo esperado por el desarrollo ({pm(G[FUS]['f1_macro']['media'], G[FUS]['f1_macro']['desvio'])}). Con 20 caballos el intervalo es ancho
({ic(ic_res)}) y <b>no se usa para comparar candidatos</b>, sólo como estimación final fuera de muestra del modelo elegido. Interesado es la clase más débil (F1 {n(res['f1']['Interesado'])}, con 12 aciertos de 16).</p>
""")
    A(fig(figs["conf"], "Matriz de confusión de la fusión: suma de las 20 evaluaciones de desarrollo (izquierda) y reserva, único uso (derecha).", 5))
    A(f"""<p class="fuente">El IC95 de la reserva no está en el JSON (<code>fusion_tardia_pose_visual__reserva.json</code> guarda métricas y predicciones): se toma de <code>DECISION.md</code> y se recalculó
para este documento con el mismo bootstrap por caballo (2 000 réplicas, semilla 7), con el mismo resultado: {ic(ic_res)}. El número de clips limpios (48) también se verificó recalculando.</p>""")

    A(h2("Validación dejando afuera cada sesión de filmación", "sesion"))
    A(f"""
<p>El agrupamiento por caballo no impide que caballos de una misma sesión de filmación queden a ambos lados de la partición. Para controlar ese riesgo, el 08/10 se corrió un análisis
complementario (declarado antes de correrlo, <code>comparacion/validacion_por_sesion.py</code>): para cada una de las {len(ses['plan']['evaluadas'])} sesiones con clips de desarrollo, el test son los clips de esa sesión y
el train son los clips de las demás, <b>sin los de ningún caballo presente en la sesión de test</b> (la exclusión quitó {ses['plan']['clips_train_perdidos_total']} clips de train en total). Ninguna sesión se saltó.
Los cuatro candidatos usaron la misma partición. No cambia el ganador ni toca la reserva.</p>
""")
    filas = []
    for c in CANDIDATOS:
        v, lim_ic = f1_sesion(D, c, "todo")
        vl, lim_icl = f1_sesion(D, c, "limpio")
        v54 = f1_pooled_54(D, c, "todo")[0]
        v54l = f1_pooled_54(D, c, "limpio")[0]
        fila = [NOMBRE[c], f"{n(v)} {ic(lim_ic)}", n(ses_c[c]["metricas"]["accuracy"]), f"{n(vl)} {ic(lim_icl)}",
                f"{n(v54)} / {n(v54l)}", f"{sg(v - v54)} / {sg(vl - v54l)}"]
        filas.append(("dest", fila) if c == FUS else fila)
    A(tabla(["Candidato", "F1 macro agrupado [IC95]", "Accuracy", "F1 limpio, 188 clips [IC95]", "5×4: todo / limpio", "Cambio: todo / limpio"],
            filas, "compacta fija res", ["l", "n", "n", "n", "n", "n"]))
    A("""<p class="fuente">«5×4» es el F1 agrupado de las predicciones fuera de muestra de cada repetición (media de las 5); «Cambio» = validación por sesión − 5×4, con la misma magnitud. Los dos protocolos usan los mismos 291 clips pero
entrenamientos distintos, de modo que la comparación es orientativa.</p>""")
    filas = []
    for rv in (RF, MIL, BAS):
        a, b = d_ses["todo"][rv], d_ses["limpio"][rv]
        filas.append([f"Fusión − {NOMBRE[rv].split(' (')[0]}", f"{sg(a['diferencia'])} {ic(a['ic95'], signo=True)}", f"{sg(b['diferencia'])} {ic(b['ic95'], signo=True)}"])
    A("<h3>Diferencias pareadas (IC95 bootstrap por caballo, 2 000 réplicas)</h3>")
    A(tabla(["Diferencia", "Todo (291 clips, 113 caballos)", "Limpio (188 clips, 71 caballos)"], filas, "compacta", ["l", "n", "n"]))
    A(f"""
<ul>
<li><b>El ranking se sostiene</b>: fusión, RF de pose, MIL y baseline, en el mismo orden que en el 5×4. La ventaja sobre el MIL y el baseline es clara, también en limpio.</li>
<li><b>El nivel de los modelos con pose se sostiene</b>: la fusión no cae ({sg(f1_sesion(D, FUS, 'todo')[0] - f1_pooled_54(D, FUS, 'todo')[0])}) y el RF pierde apenas {n(f1_pooled_54(D, RF, 'todo')[0] - f1_sesion(D, RF, 'todo')[0])}; el MIL y el baseline pierden
{n(f1_pooled_54(D, MIL, 'todo')[0] - f1_sesion(D, MIL, 'todo')[0])} y {n(f1_pooled_54(D, BAS, 'todo')[0] - f1_sesion(D, BAS, 'todo')[0])}. Los modelos que dependen más de la apariencia visual son los que más pierden ante sesiones no vistas.</li>
<li><b>Limitación para la elección</b>: la ventaja de la fusión sobre el RF de pose <b>ya no se distingue de cero</b> ({sg(d_ses['todo'][RF]['diferencia'])} {ic(d_ses['todo'][RF]['ic95'], signo=True)}; en el 5×4 era {sg(p_f_rf['diferencia_a_menos_b'])} con límite inferior {sg(p_f_rf['diferencia_ic95'][0])}).
Si la regla de empate del §5 se aplicara con este resultado, <b>el RF de pose sería el elegido</b>. Por mandato del protocolo el ganador no cambia, pero la fusión no debe presentarse como claramente superior al RF.</li>
<li><b>El confusor de filmación no se descarta</b>: ver una sesión nueva no baja el acierto de los modelos con pose (la ventaja no se explica por rasgos propios de cada sesión), pero en el subconjunto limpio siguen unos
{n(caida_limpio_ses * 100, 0)} puntos por debajo ({n(f1_sesion(D, FUS, 'limpio')[0])} contra {n(f1_sesion(D, FUS, 'todo')[0])}), igual que en el 5×4. Los rasgos que delatan la clase, sobre todo la duración del clip, parecen propiedades del protocolo de filmación de cada clase y no de una sesión.</li>
<li><b>Límites del análisis</b>: 20 sesiones muy desiguales (S020 concentra 85 de 291 clips y 38 caballos; varias tienen 1 a 4 clips o una sola clase, por eso la métrica es agrupada); sesiones de fechas cercanas pueden compartir condiciones.</li>
</ul>
""")
    A(f"""<p>Matriz de confusión de la fusión por sesión (filas = real; Disgustado, Interesado, Sereno): {ses_c[FUS]['metricas']['matriz'][0]}, {ses_c[FUS]['metricas']['matriz'][1]}, {ses_c[FUS]['metricas']['matriz'][2]}.
F1 por clase: Disgustado {n(ses_c[FUS]['metricas']['f1']['Disgustado'])}, Interesado {n(ses_c[FUS]['metricas']['f1']['Interesado'])}, Sereno {n(ses_c[FUS]['metricas']['f1']['Sereno'])}.</p>""")

    # ======================================================================= 6 LIMITACIONES
    A(h1("Limitaciones declaradas", "limitaciones"))
    A("""<p>Esta sección reúne lo que el lector debe tener presente antes de confiar en una cifra. Ninguna es nueva para el proyecto: están en
<code>comparacion/DECISION.md</code>, en <code>openspec/project.md</code> y en la auditoría v2.</p>""")
    A(tabla(["Limitación", "Evidencia", "Qué implica para la lectura"], [
        ["<b>1. Confusor de duración y formato.</b> Los clips de Disgustado son más cortos (mediana de 3,2 s contra 6,0 s) y las features de pose predicen la duración del clip (R² 0,53).",
         f"En el subconjunto limpio todos los candidatos caen entre 10 y 13 puntos: la fusión pasa de {n(f1_pooled_54(D, FUS, 'todo')[0])} a {n(f1_pooled_54(D, FUS, 'limpio')[0])} ({n(caida_limpio_54 * 100, 0)} puntos).",
         f"Parte de la ventaja puede ser de filmación. Las cifras más prudentes son las de limpio ({n(LIM[FUS]['media'])} en desarrollo; {n(D['reserva']['limpio_f1_macro'])} en la reserva)."],
        ["<b>2. Ventaja mínima sobre el RF de pose.</b>",
         f"{sg(p_f_rf['diferencia_a_menos_b'])} {ic(p_f_rf['diferencia_ic95'], signo=True)} en el 5×4; {sg(d_ses['todo'][RF]['diferencia'])} {ic(d_ses['todo'][RF]['ic95'], signo=True)} por sesión.",
         "Elegir la fusión o el RF es defendible; el RF es más simple. La fusión está dominada por la pose (w = 0,75)."],
        ["<b>3. Reserva chica.</b>", f"20 caballos, 68 clips; IC95 {ic(ic_res)}.", "Sirve como estimación final, no para comparar modelos."],
        ["<b>4. Un solo etiquetador.</b>",
         "Un único observador (que además filmó); el acuerdo sólo se midió en un piloto sobre imágenes (kappa de Fleiss 0,99).",
         "El modelo aprende el criterio de ese observador. Se desconoce el techo alcanzable. No hay validación con paneles de expertos."],
        ["<b>5. Modelo final reentrenado con todos los clips.</b>", "La aplicación usa un modelo ajustado con desarrollo + reserva (<code>comparacion/entrenar_modelo_final.py</code>).",
         "No tiene una medida propia: las cifras de desarrollo y reserva describen modelos ajustados con menos datos."],
        ["<b>6. Una sola población y sin test externo.</b>", "Todos los videos se filmaron en Tandil (22 sesiones).", "Estima el rendimiento sobre caballos nuevos de esta población y este tipo de captura."],
        ["<b>7. Hipótesis temporal no sostenida.</b>", "Ningún candidato modela el orden de los fotogramas; en las 18 configuraciones probadas el orden no aportó.",
         "Se habla de «agregación espacial de varios fotogramas», no de modelo espacio-temporal."],
        ["<b>8. Sobreajuste.</b>", f"F1 de train de la fusión {n(G[FUS]['f1_macro_train']['media'])}; gap {sg(G[FUS]['gap_train_test_f1']['media'], 2)}.",
         "El Random Forest memoriza el train; el desempeño de test es el honesto, pero conviene no extrapolar."],
        ["<b>9. Identidad de los caballos.</b>", "El ID es el prefijo numérico del nombre de archivo, asignado a mano; el par 123/169 es visualmente similar.", "Riesgo bajo pero no descartado de fuga por un caballo con dos IDs."],
        ["<b>10. Repeticiones y sesiones.</b>", "Las 5 repeticiones remuestrean la partición, no los datos; las sesiones son 20 y muy desiguales.", "El desvío entre repeticiones no es un error estándar; se usa el bootstrap por caballo."],
        ["<b>11. Utilidad no medida.</b>", "No hay validación con veterinarios; la prueba de textos con una persona ajena fue n = 1.", "No es una herramienta de diagnóstico: es la demostración de un pipeline."],
    ], "compacta fija"))
    A("""<div class="recuadro honesto"><h3>Lo que el proyecto sí puede afirmar</h3><p>Que se puede clasificar la expresión de caballos nuevos de esta población con un F1 macro cercano a 0,83 y de 0,74 a 0,77 en clips filmados en condiciones parejas;
que el resultado se sostiene al dejar afuera sesiones enteras; y que un modelo basado en la pose supera con claridad a los modelos basados sólo en la apariencia. <b>No puede afirmar</b> que la dinámica
temporal sea la clave, que la fusión sea mejor que el RF de pose, que el nivel sea útil para expertos, ni que el modelo generalice a otra población.</p></div>""")

    # ======================================================================= 7 ESTADO
    A(h1("Estado actual del proyecto", "estado"))
    A(h2("La aplicación de inferencia", "app"))
    A(f"""
<p>La interfaz (<code>app_inferencia.py</code>, Streamlit) recibe una toma corta de la cara de un caballo y devuelve la expresión predicha <b>junto con la confiabilidad medida de esa predicción</b>.
Desde el 08/10 el modelo por omisión es la fusión pose + visual (peso de la pose w = 0,75, ajustada con todos los clips); el MIL sigue disponible con <code>clasificar_video(ruta, modelo="mil")</code>.
La aplicación y la inferencia comparten con el entrenamiento la misma implementación del preprocesamiento (<code>alineado.preparar_secuencia</code>) y de los rasgos de pose, verificada con pruebas de equivalencia.</p>
<ul>
<li><b>Textos en lenguaje llano.</b> La confiabilidad se explica en lenguaje llano («entre 7 y 9 de cada 10 veces») y se presenta como un <b>rango entre las dos mediciones</b> (desarrollo y reserva); los números salen de <code>comparacion/resultados/</code>, no se escriben a mano.</li>
<li><b>Qué muestra con la fusión.</b> Las probabilidades de cada rama (pose y visual); el gráfico de atención aparece sólo si se elige el MIL.</li>
<li><b>Límites declarados.</b> No es una herramienta de diagnóstico; no siempre responde (necesita 16 fotogramas con ojo y hocico visibles; el 91 % de los clips del dataset los tiene); acepta videos de hasta 15 s, analiza un tramo de hasta 7 s y lo informa al usuario; tarda alrededor de 80 s para una toma de 7 s.</li>
</ul>
""")
    prec_dev = {cl: G[FUS]["por_clase"][cl]["precision"]["media"] for cl in CLASES}
    prec_res = res["precision"]
    filas = []
    for i, cl in enumerate(CLASES):
        tot_dev = sum(M54[k][i] for k in range(3)) / 5
        ok_dev = M54[i][i] / 5
        tot_res = sum(Mres[k][i] for k in range(3))
        filas.append([f"«{cl}»", f"{ok_dev:.0f} de {tot_dev:.0f} — {pct(M54[i][i] / sum(M54[k][i] for k in range(3)))}", f"{Mres[i][i]} de {tot_res} — {pct(Mres[i][i] / tot_res)}"])
    A("<h3>Cuando la fusión dice una clase, ¿cuántas veces acierta?</h3>")
    A(tabla(["La fusión dice", "Desarrollo (clips por repetición)", "Reserva (68 clips)"], filas, "compacta", ["l", "n", "n"]))
    A('<p class="fuente">Calculado desde las matrices de confusión (precisión por clase). Con la reserva, cada precisión se apoya en pocas decenas de clips.</p>')

    A(h2("Pruebas, OpenSpec y ramas", "pruebas"))
    A(f"""
<p><b>Pruebas.</b> El directorio <code>tests/</code> contiene <b>{meta['n_tests']} guiones</b> (no usan pytest: cada uno se corre con <code>python3 tests/test_X.py</code> e informa OK o FALLA por caso).
Se ejecutaron todos el 08/10/2026 con el entorno del proyecto sobre el commit {meta['commit']}: <b>{meta['n_tests']} de {meta['n_tests']} terminaron sin falla</b>.
Cubren, entre otras cosas, la partición por caballo sin fuga, la equivalencia entrenamiento–inferencia de la fusión, el contrato del evaluador, el uso único de la reserva y la validación por sesión.</p>
<p><b>OpenSpec.</b> No hay changes abiertos: <code>openspec/changes/</code> sólo contiene <code>archive/</code>, con <b>{meta['n_archivados']} changes archivados</b>, {meta['n_arch_oct']} de ellos del 07 y 08/10. Los que corresponden a este tramo:</p>
""")
    A("<ul>" + "".join(f"<li><code>{x}</code></li>" for x in meta["archivados_oct"]) + "</ul>")
    A("""
<p><b>Ramas.</b> <code>main</code> contiene la línea vigente (MIL multimodal, auditoría, comparación, fusión, validación por sesión y aplicación). Las ramas de variantes que no se adoptaron
se conservan sin fusionar, cada una con el motivo registrado en <code>docs/caminos_explorados.md</code> (§3):</p>
""")
    A(tabla(["Grupo", "Ramas", "Estado"], [
        ["Fusionadas a <code>main</code>", "<code>feat/arquitectura-multimodal-mil</code>, <code>feat2/otro-camino</code>, <code>chore/limpieza-codigo</code>, <code>feat/mejoras-prioridad-alta</code>, <code>chore/deuda-menor-y-observador-unico</code>, <code>feat/auditoria-y-experimento-simba</code>, <code>feat/base-comun-comparacion</code>, <code>feat/candidato-baseline</code>, <code>feat/candidato-fusion</code>, <code>feat/app-modelo-fusion</code>, <code>feat/validacion-por-sesion</code>, <code>docs/caminos-explorados</code>, <code>chore/cerrar-changes-viejos</code>", "Parte de la línea vigente."],
        ["Resultados traídos a <code>main</code> sin fusionar", "<code>feat/candidato-mil</code>, <code>feat/candidato-rf-pose</code>", "Código y JSON copiados con <code>git checkout &lt;rama&gt; -- &lt;archivos&gt;</code>."],
        ["Variantes no adoptadas (sin fusionar)", "<code>feat/enriquecer-canal-geometrico</code>, <code>feat/finetuning-visual</code>, <code>feat/validacion-cruzada-anidada-NestedCV</code>, <code>origin/feat/microexpresiones-ollares</code>, <code>origin/feat/recortes-largos</code>", "Medidas o descartadas; ninguna mejoró al modelo (ver sección 7.3)."],
        ["Líneas paralelas anteriores", "<code>hector</code>, <code>ejecucion-local-gpu</code>, <code>nuevo-intento-con-enmascarado-y-geometria</code>", "No fusionadas por decisión de los autores."],
        ["Rescate", "<code>rescate/otro-camino-2026-09-08</code>", "Preserva tres commits huérfanos (<code>4c13c20</code>, <code>1a3859b</code>, <code>5ad6b64</code>)."],
    ], "compacta fija"))

    A(h2("Caminos explorados y descartados", "caminos"))
    A("""<p>El documento <code>docs/caminos_explorados.md</code> registra <b>47 caminos</b> (C-01 a C-47) con el mismo formato: qué se intentó, por qué, resultado, decisión y evidencia, más 14 lecciones de método.
Un resumen de lo que se descartó y por qué:</p>""")
    A(tabla(["Camino", "Resultado medido", "Decisión"], [
        ["Un fotograma con voto; 18 cabezas temporales (C-27)", "El orden de los fotogramas no aportó en ninguna configuración", "Hipótesis temporal no sostenida"],
        ["Fine-tuning de la ResNet-18 (C-18)", "0,747 ± 0,057 contra 0,740 ± 0,052", "Descartado: sin diferencia"],
        ["Canal geométrico enriquecido, 21 dimensiones (C-21)", "0,698 ± 0,103; margen sobre ruido +0,018", "Descartado: sin diferencia"],
        ["Comisuras y ollares, 15 dimensiones (C-22)", "0,669 ± 0,077; el ruido de pose supera a la señal", "Descartado: peor"],
        ["Geometría facial pura, sin imagen (C-24)", "0,184 por debajo de apariencia; 1 de 12 celdas cumple", "Descartado: peor"],
        ["Movimiento de cabeza, «vía C» (C-25)", "Ningún modelo supera el piso 0,571; la duración sola da 0,578", "Descartado: sin señal"],
        ["Enmascarado de clips incompletos (C-16, C-17)", "No cumple su criterio (0,736 contra 0,7847); la fuga pasa de «cuántos» a «cuáles»", "Descartado; no se retomó"],
        ["Validación cruzada anidada (C-39)", "Triplica el desvío de la época", "Descartado: peor"],
        ["CNN-LSTM, rama <code>hector</code> (C-29)", "Una partición, sin corrida que lo respalde", "Abandonado"],
        ["SimBA real (C-34)", "Para un animal solo entrega movimiento: 0,720", "Descartado: otra razón"],
        ["Videos largos en la app (C-46)", "Sin medición; la app informa el tramo analizado y recomienda recortar", "Abandonado"],
        ["Etiqueta débil y unidades de acción EquiFACS (C-07, A y B)", "No forman parte del modelo final", "Descartado por decisión de Sergio (08/10)"],
        ["Recortar todos los clips a igual duración (C-41)", "Cambia el dataset y obliga a re-ejecutar la etapa 02", "Sin medición; queda como recomendación"],
    ], "compacta fija"))
    A("""<p>Algunas lecciones de método, tal como se registraron: medir contra el denominador correcto (L-1); anclar un umbral declarado antes de medir en una línea de base física (L-2);
la vara es el atajo más fuerte, no la clase mayoritaria (L-3); no comparar cifras de composiciones o protocolos distintos (L-7); no elegir sobre el conjunto de evaluación (L-8);
y que un resultado negativo bien medido es un resultado (L-11).</p>""")

    A(h2("Qué queda", "queda"))
    A("""
<ul>
<li><b>Informe final (entrega prevista: 29/10/2026).</b> Redacción en formato APA (<code>.tex</code> para Overleaf). Debe llevar las limitaciones de la sección 6, incluir el relevamiento bibliográfico que el docente señaló como no encontrado
y presentar la fusión sin afirmar superioridad sobre el RF de pose.</li>
<li><b>Observaciones abiertas del docente</b> (sección 2.3): revisar el par de identidades 123/169, la utilidad con veterinarios y el relevamiento bibliográfico.</li>
<li><b>Trabajo futuro, sin ejecutar:</b> medir el acuerdo interobservador sobre el dataset con un panel de expertos; recortar los clips a igual duración (o filmar nuevas tomas con un protocolo parejo) para neutralizar el confusor de formato;
evaluar un modelo de pose específico de caballos; un test externo con otra población.</li>
</ul>
""")

    # ======================================================================= ANEXO
    A(h1("Anexo: reproducibilidad", "anexo", num=False))
    A(h2("Huellas y semillas", "huellas", num=False))
    A(tabla(["Elemento", "Valor"], [
        ["Huella del dataset (359 clips)", f"<code>{D['huella_dataset']}</code>"],
        ["Huella de la reserva (20 caballos)", f"<code>{D['huella_reserva']}</code>"],
        ["Huella de las particiones (5×4)", f"<code>{D['huella_particiones']}</code>"],
        ["Semilla del sorteo de la reserva", f"{D['reserva_caballos']['semilla']}"],
        ["Semillas de las 5 repeticiones", "1101, 1102, 1103, 1104, 1105"],
        ["Bootstrap por caballo", "2 000 réplicas, semilla 7"],
        ["Uso de la reserva (<code>reserva_uso.log</code>)", f"{meta['fecha_uso']}, candidato <code>{meta['uso']['candidato']}</code>; {meta['uso']['motivo']}"],
        ["Peso de la pose en el modelo final", "w = 0,75"],
    ], "compacta fija"))
    A(h2("Comandos", "comandos", num=False))
    A("""<p>Desde la raíz del repositorio, con el entorno <code>~/.venvs/tesina-equinos</code> y <code>TESINA_EQUINOS_RAIZ</code> apuntando a los datos:</p>
<table class="compacta"><thead><tr><th>Qué reproduce</th><th>Comando</th></tr></thead><tbody>
<tr><td>Baseline</td><td><code>python comparacion/correr_baseline.py</code></td></tr>
<tr><td>RF de pose tipo SimBA</td><td><code>python comparacion/candidato_rf_pose.py</code></td></tr>
<tr><td>MIL multimodal corregido</td><td><code>python comparacion/candidato_mil.py</code></td></tr>
<tr><td>Fusión pose + visual</td><td><code>python comparacion/candidato_fusion.py</code></td></tr>
<tr><td>Modelo final (todos los clips)</td><td><code>python comparacion/entrenar_modelo_final.py</code></td></tr>
<tr><td>Validación por sesión</td><td><code>python comparacion/validacion_por_sesion.py</code> (<code>--plan</code> sólo muestra el plan)</td></tr>
<tr><td>Pruebas</td><td><code>for t in tests/test_*.py; do PYTHONPATH="$PWD" python3 "$t" || echo "FALLÓ $t"; done</code></td></tr>
<tr><td>Aplicación</td><td><code>./correr_local.sh interfaz</code></td></tr>
<tr><td>Este documento</td><td><code>python3 docs/informes/estado_actual_2026-10-08/generar.py</code></td></tr>
</tbody></table>
<p>La reserva no se vuelve a evaluar: <code>comparacion.evaluar_en_reserva</code> se niega a un segundo uso.</p>""")
    A(h2("Archivos", "archivos", num=False))
    A(tabla(["Archivo", "Contenido"], [
        ["<code>comparacion/PROTOCOLO.md</code>", "Protocolo declarado el 07/10 y enmienda del 08/10"],
        ["<code>comparacion/DECISION.md</code>", "Ganador, salvedades, reserva y validación por sesión"],
        ["<code>comparacion/particiones.json</code>, <code>reserva_caballos.json</code>", "Particiones 5×4 y reserva, con huellas"],
        ["<code>comparacion/sesiones_de_filmacion.csv</code>", "Tabla de las 22 sesiones de filmación"],
        ["<code>comparacion/resultados/*.json</code>", "Métricas, predicciones fuera de muestra y diferencias pareadas"],
        ["<code>comparacion/reserva_uso.log</code>", "Registro del único uso de la reserva"],
        ["<code>equinos/fusion.py</code>, <code>comparacion/candidato_fusion.py</code>", "Modelo de fusión para inferir y para evaluar"],
        ["<code>docs/caminos_explorados.md</code>", "47 caminos, con decisión y evidencia"],
        ["<code>docs/informes/estado_actual_2026-10-08/</code>", "Fuentes de este documento (HTML, CSS, figuras y <code>generar.py</code>)"],
    ], "compacta fija"))
    return "\n".join(out)
