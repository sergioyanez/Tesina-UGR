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

from equinos import clasificacion, inferencia, rutas

FORMATOS = ["mp4", "mov", "avi", "mkv", "webm"]

# Un color fijo por expresión, para que el gráfico de atención no cambie de paleta entre tomas.
# Las claves salen de `clasificacion.CLASES` para que no se desincronicen con el modelo.
COLOR_POR_CLASE = {"Sereno": "#2a9d8f", "Interesado": "#e9c46a", "Disgustado": "#e76f51"}
CLASES = list(clasificacion.CLASES)
COLORES_CLASE = [COLOR_POR_CLASE[c] for c in CLASES]

ETAPAS = {
    "leer": "Leyendo el video",
    "detectar": "Buscando al equino (~15 s)",
    "pose": "Estimando keypoints faciales (~20 s)",
    "alinear": "Recortando y alineando la cara",
    "caracteristicas": "Extrayendo características",
    "predecir": "Clasificando",
}

st.set_page_config(page_title="Expresiones faciales en equinos", page_icon="🐴")


@st.cache_resource
def _precisiones():
    rutas.inicializar(verbose=False)
    return clasificacion.precision_por_clase()


st.title("Detección de expresiones faciales en equinos")

# --- Alcance, visible sin buscarlo (tarea 5.5) -------------------------------------------------
st.error(
    "**Esto NO es una herramienta de diagnóstico veterinario.** Es la demostración de un "
    "trabajo de tesina. El modelo **alcanzó el criterio que se declaró para considerarlo "
    "útil** (F1 macro 0,754 sobre 0,691 exigido)."
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
- **Tarda alrededor de 35 segundos** con GPU: 15 de detección y 20 de estimación de pose. La
  clasificación en sí es instantánea; lo lento es el preprocesamiento.
- **Es probable que el sistema no pueda responder.** Necesita 16 fotogramas donde se vean el ojo
  y el hocico con claridad. Sobre el propio conjunto de datos del trabajo, sólo el **38 %** de las
  tomas lo consiguió. Que se niegue es un resultado normal, no un error del video.
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
                "donde pueda alinear los 16 fotogramas por el eje ojo-hocico; si no puede, "
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
                       "medido fuera de muestra")

        if resultado.precision < 0.5:
            st.error(
                f"**Atención: el modelo acierta menos de la mitad de las veces que dice "
                f"«{resultado.clase}».** Este resultado no debería usarse para tomar ninguna "
                "decisión sobre el animal."
            )

        st.markdown("---")
        st.markdown("#### Probabilidades del modelo")
        st.caption(
            "El modelo procesa toda la secuencia junta (arquitectura MIL) y emite esta distribución final. "
            "Un reparto parejo significa que el modelo dudó, aunque igual haya tenido que elegir una etiqueta."
        )
        for clase, probabilidad in sorted(resultado.probabilidades.items(),
                                          key=lambda kv: -kv[1]):
            st.progress(probabilidad,
                        text=f"{clase} — {probabilidad:.0%}  "
                             f"(acierta {precisiones[clase]['precision']:.0%} cuando lo dice)")

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

Este sistema utiliza una arquitectura **Multimodal MIL (Multiple Instance Learning)** que procesa 
la secuencia de fotogramas entera. Combina la postura geométrica (orejas, hocico, ojos) y las 
características visuales para emitir una única predicción.
"""
            )
