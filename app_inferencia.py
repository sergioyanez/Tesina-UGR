"""Interfaz del objetivo 4 — Streamlit sobre `equinos.inferencia`.

Se levanta con:

    ./correr_local.sh interfaz

**Es una envoltura delgada, a propósito.** Toda la lógica —el preprocesamiento, la negativa, el
modelo y la confiabilidad— vive en `equinos/`. Acá sólo hay presentación, y por eso cambiar de
framework no obliga a reimplementar nada.

**Lo que esta pantalla tiene que lograr, y es el requisito principal del change:** que alguien que
no leyó la tesina entienda que, cuando el modelo dice `Disgustado`, acierta una de cada tres veces.
Una pantalla que devuelve «Disgustado, 87 %» es indistinguible de una herramienta validada, y ésta
no lo es. Por eso la confiabilidad medida va arriba, en grande, y no en un descargo al pie.
"""

import tempfile
from pathlib import Path

import streamlit as st

from equinos import clasificacion, fusion, inferencia, rutas, secuencia

FORMATOS = ["mp4", "mov", "avi", "mkv", "webm"]

# Un color fijo por expresión, para que el gráfico de atención no cambie de paleta entre tomas.
# Las claves salen de `clasificacion.CLASES` para que no se desincronicen con el modelo.
COLOR_POR_CLASE = {"Sereno": "#2a9d8f", "Interesado": "#e9c46a", "Disgustado": "#e76f51"}
CLASES = list(clasificacion.CLASES)
COLORES_CLASE = [COLOR_POR_CLASE[c] for c in CLASES]

# La etapa `detectar` desapareció: la inferencia ya no corre YOLO sobre 16 fotogramas muestreados
# antes de la pose. Ese gateo rechazaba videos donde el detector propio de DeepLabCut sí habría
# encontrado la cara, y la generación del dataset nunca lo hizo.
ETAPAS = {
    "leer": "Leyendo el video",
    "pose": "Estimando keypoints en todos los fotogramas (es lo que más tarda)",
    "alinear": "Eligiendo el mejor tramo y alineando la cara",
    "caracteristicas": "Extrayendo características",
    "predecir": "Clasificando",
}

st.set_page_config(page_title="Expresiones faciales en equinos", page_icon="🐴")


@st.cache_resource
def _precisiones():
    """Precisión por clase del modelo que corre la app: la fusión (`inferencia.MODELO_POR_OMISION`)."""
    rutas.inicializar(verbose=False)
    return fusion.precision_por_clase()


@st.cache_resource
def _metricas():
    """Las métricas del modelo que esta app está ejecutando. **Ningún número escrito a mano.**

    La versión anterior tenía `F1 macro 0,754 sobre 0,691 exigido` en el texto, y ese 0,754 era
    de otro modelo medido en otra máquina sobre otro conjunto: la pantalla informaba una métrica
    que no correspondía a nada de lo que estaba corriendo. Leerlo del mismo archivo del que salen
    las precisiones por clase es lo que garantiza que hablen del mismo modelo.
    """
    rutas.inicializar(verbose=False)
    return fusion.metricas_globales()


st.title("Detección de expresiones faciales en equinos")

# --- Alcance, visible sin buscarlo (tarea 5.5) -------------------------------------------------
_m = _metricas()
st.error(
    "**Esto NO es una herramienta de diagnóstico veterinario.** Es la demostración de un "
    f"trabajo de tesina. El modelo (fusión de pose y apariencia) obtuvo F1 macro "
    f"**{_m['f1_macro_reserva']:.3f}** sobre {_m['clips_reserva']} tomas de "
    f"{_m['caballos_reserva']} caballos reservados, medido con una versión del modelo ajustada sin ellos "
    f"({_m['accuracy_reserva']:.0%} de aciertos), y {_m['f1_macro_desarrollo']:.3f} en validación "
    f"cruzada por caballo. Su acierto depende de la clase y baja si la toma no se filmó en "
    f"condiciones parejas (F1 {_m['f1_limpio_reserva']:.3f} en la reserva, tomas parejas): "
    "consultá el detalle que acompaña a cada resultado."
)

st.markdown(
    "Subí una toma corta de la cara de un caballo y el sistema intentará clasificar su expresión "
    "en **Sereno**, **Interesado** o **Disgustado**."
)

with st.expander("Antes de subir: qué esperar"):
    st.markdown(
        f"""
- **Duración:** una toma de **5 a 7 segundos** (máximo {inferencia.DURACION_MAXIMA_SEG:.0f} s).
  Es lo que el modelo vio al entrenarse; sobre un video largo mediría otra cosa.
- **Formatos:** {", ".join(FORMATOS)}.
- **Tarda entre 1 y 3 minutos** con GPU, y el costo crece con la duración del video: la pose se
  estima sobre TODOS los fotogramas, que es lo que permite elegir el mejor tramo en vez de
  muestrear a ciegas. Medido en una RTX 3070: 82 s para una toma de 7 s, 147 s para una de 13 s.
  La clasificación en sí es instantánea; lo lento es el preprocesamiento.
- **El sistema puede negarse a responder.** Necesita un tramo continuo del video —de hasta
  {secuencia.VENTANA_MAXIMA_SEG:.0f} s— donde el ojo y el hocico se midan con claridad
  en 16 fotogramas. Sobre el propio conjunto de datos del trabajo lo consigue el **91 %** de las
  tomas. Que se niegue es un resultado normal, no un error del video.
"""
    )

archivo = st.file_uploader("Video de la toma", type=FORMATOS)

if archivo is not None:
    st.video(archivo)
    if st.button("Analizar", type="primary"):
        with tempfile.NamedTemporaryFile(suffix=Path(archivo.name).suffix, delete=False) as tmp:
            tmp.write(archivo.getbuffer())
            ruta = tmp.name

        # `st.status` y no `st.progress`: con una barra suelta, la primera versión de esta
        # pantalla no mostraba NADA durante los 35 s —sólo el spinner del reproductor— porque las
        # llamadas a DeepLabCut bloquean el hilo del script. Verificado mirando la pantalla, que
        # es para lo que existe la revisión de honestidad del change.
        with st.status("Analizando la toma…", expanded=True) as estado:
            def avisar(etapa, detalle=""):
                estado.write(f"**{ETAPAS.get(etapa, etapa)}**" + (f" — {detalle}" if detalle else ""))
                estado.update(label=ETAPAS.get(etapa, etapa))

            try:
                resultado = inferencia.clasificar_video(ruta, progreso=avisar)
            finally:
                Path(ruta).unlink(missing_ok=True)
            estado.update(label="Análisis terminado", state="complete", expanded=False)

        # --- Negativa: un resultado de primera clase (tarea 5.4) ------------------------------
        if not resultado.hay_prediccion:
            st.warning(f"### No se puede clasificar esta toma\n\n{resultado.detalle}")
            st.markdown(
                "**Esto es un resultado esperable, no una falla.** El modelo sólo acepta tomas "
                "donde pueda medir el ojo y el hocico durante un tramo continuo; si no puede, "
                "prefiere no responder antes que dar una etiqueta cuya confiabilidad nadie midió."
            )
            if resultado.frames_alineados:
                st.caption(f"Se alinearon {resultado.frames_alineados} de "
                           f"{inferencia.FRAMES_POR_CLIP} fotogramas necesarios.")
            st.stop()

        # --- Resultado: la etiqueta y su confiabilidad, con el mismo peso (tarea 5.3) ---------
        precisiones = _precisiones()
        st.markdown("---")
        izquierda, derecha = st.columns(2)
        with izquierda:
            st.caption("Expresión predicha")
            st.markdown(f"# {resultado.clase}")
        with derecha:
            confianza = max(resultado.probabilidades.values())
            st.caption("Confianza del modelo en esta toma")
            st.markdown(f"# {confianza:.0%}")
            st.caption(f"Cuando dice «{resultado.clase}» acierta {resultado.precision:.0%}: "
                       f"{resultado.acertadas} de {resultado.veces_que_lo_dijo} veces, "
                       "en validación cruzada por caballo (tomas de desarrollo, fuera de muestra)")
            if resultado.detalle:
                pr = resultado.detalle["precision_reserva"]
                st.caption(f"En la reserva (20 caballos, 68 tomas, un solo uso): "
                           f"{pr['precision']:.0%}, {pr['acertadas']} de {pr['veces_que_lo_dijo']} veces")

        if resultado.precision < 0.5:
            st.error(
                f"**Atención: el modelo acierta menos de la mitad de las veces que dice "
                f"«{resultado.clase}».** Este resultado no debería usarse para tomar ninguna "
                "decisión sobre el animal."
            )

        st.markdown("---")
        st.markdown("#### Probabilidades del modelo")
        st.caption(
            + ("El modelo procesa toda la secuencia junta (arquitectura MIL) y emite esta "
               "distribución final. " if resultado.modelo == "mil" else
               "El modelo combina dos lecturas de la toma —la pose de la cara y su apariencia— y "
               "promedia sus probabilidades. ")
            + "Un reparto parejo significa que el modelo dudó, aunque igual haya tenido que "
            "elegir una etiqueta."
        )
        for clase, probabilidad in sorted(resultado.probabilidades.items(),
                                          key=lambda kv: -kv[1]):
            st.progress(probabilidad,
                        text=f"{clase} — {probabilidad:.0%}  "
                             f"(acierta {precisiones[clase]['precision']:.0%} cuando lo dice)")

        if resultado.detalle:
            st.markdown("---")
            st.markdown("#### Qué dijo cada lectura")
            w = resultado.detalle["peso_pose"]
            st.caption(
                f"La respuesta final pesa la pose con {w:.0%} y la apariencia con {1 - w:.0%}. "
                "Si las dos lecturas no coinciden, la toma es dudosa aunque el reparto final "
                "parezca claro.")
            col_pose, col_visual = st.columns(2)
            for col, titulo, clave in ((col_pose, "Pose de la cara", "pose"),
                                       (col_visual, "Apariencia", "visual")):
                with col:
                    st.caption(titulo)
                    for clase, probabilidad in sorted(resultado.detalle[clave].items(),
                                                      key=lambda kv: -kv[1]):
                        st.progress(probabilidad, text=f"{clase} — {probabilidad:.0%}")

        if resultado.atencion:
            st.markdown("---")
            st.markdown("#### Explicabilidad (XAI): Pesos de Atención")
            st.caption(
                "Muestra a qué fotogramas de la secuencia el modelo le prestó más atención "
                "para tomar su decisión. Los picos altos indican los momentos más informativos de la toma. "
                "El nombre dentro de cada barra es la expresión hacia la que empuja ese fotograma por "
                "separado; el modelo no decide fotograma a fotograma, decide sobre la secuencia entera."
            )
            import altair as alt
            import pandas as pd
            df_atencion = pd.DataFrame({
                "Fotograma": [str(i + 1) for i in range(len(resultado.atencion))],
                "Atención": resultado.atencion,
                "Expresión": resultado.clases_por_frame,
                "base": [0.0] * len(resultado.atencion),
            })
            colores = alt.Color(
                "Expresión:N",
                title="Empuja hacia",
                scale=alt.Scale(domain=CLASES, range=COLORES_CLASE),
            )
            base = alt.Chart(df_atencion).encode(
                x=alt.X("Fotograma:N", sort=None, title="Fotograma de la secuencia"),
            )
            barras = base.mark_bar().encode(
                y=alt.Y("Atención:Q", title="Peso de atención"),
                color=colores,
                tooltip=["Fotograma", alt.Tooltip("Atención:Q", format=".1%", title="Atención"),
                         "Expresión"],
            )
            # Texto oscuro y no blanco: sobre el amarillo de `Interesado` el blanco no se lee, y en
            # las barras bajas la etiqueta sobresale del relleno y queda sobre el fondo del gráfico.
            etiquetas = base.mark_text(
                align="left", baseline="middle", angle=270, dx=4, color="#1b2a32", fontSize=11,
            ).encode(
                y=alt.Y("base:Q"),
                text="Expresión:N",
            )
            st.altair_chart((barras + etiquetas).properties(height=280))

        st.markdown("---")
        with st.expander("Cómo se midió"):
            st.markdown(
                """
La confiabilidad sale de **validación cruzada agrupada por caballo**: ningún animal usado para
entrenar aparece en la evaluación, así que el modelo no puede acertar reconociendo al ejemplar en
lugar de su expresión.
"""
            )
            if resultado.modelo == "mil":
                st.markdown(
                    "Este sistema utiliza una arquitectura **Multimodal MIL (Multiple Instance "
                    "Learning)** que procesa la secuencia de fotogramas entera. Combina la postura "
                    "geométrica (orejas, hocico, ojos) y las características visuales para emitir "
                    "una única predicción.")
            else:
                st.markdown(
                    """
Este sistema es una **fusión tardía de pose y apariencia**. Un bosque aleatorio lee 160
medidas de la pose de la cara en los 16 fotogramas (distancias, ángulos de orejas, apertura de la
boca, velocidades) y una regresión logística lee el promedio de las características visuales de
una red ResNet-18 congelada; las probabilidades de ambos se promedian.

Dos límites que conviene tener presentes. **La pose domina la decisión**, y sus medidas también
delatan cómo se filmó la toma (por ejemplo, su duración), así que parte de la ventaja puede deberse
a la filmación y no a la expresión: sobre tomas filmadas en condiciones parejas el F1 baja. Y
**las cifras de arriba son las de la validación**; el modelo en uso se ajustó con todos los clips,
incluidos los reservados, y no tiene una medida propia.
"""
                )
