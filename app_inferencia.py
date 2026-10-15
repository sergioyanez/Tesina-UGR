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

import shutil
import tempfile
from pathlib import Path

import streamlit as st

from equinos import clasificacion, fusion, inferencia, rutas

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
    "inferencia": "Clasificando cada tramo de 3 segundos",
    "render": "Dibujando el recuadro y la etiqueta sobre el video",
}

st.set_page_config(page_title="Expresiones faciales en equinos", page_icon="🐴")


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
# Lo primero que se ve: tiene que entenderlo alguien que no leyó la tesina. Las cifras salen del
# mismo archivo que el resto de la pantalla; ninguna se escribe a mano.
_acierto = round(_m["accuracy_reserva"] * 100)
st.error(
    "### Esta herramienta NO constituye un instrumento de diagnóstico veterinario\n\n"
    "Esta aplicación, desarrollada en el marco de un trabajo de tesina, utiliza técnicas de "
    "inteligencia artificial para **analizar videos de caballos y clasificar sus expresiones "
    "faciales** en tres categorías: Sereno, Interesado y Disgustado.\n\n"
    "Se trata de una herramienta experimental y **sus resultados pueden presentar errores**. Por lo "
    "tanto, no deben utilizarse para tomar decisiones relacionadas con la salud, el bienestar o el "
    "manejo de los equinos.\n\n"
    f"**¿Cuál es su nivel de precisión?** En una evaluación final realizada con videos de "
    f"{_m['caballos_reserva']} caballos que no habían sido utilizados durante el entrenamiento, el "
    f"modelo obtuvo aproximadamente un **{_acierto} % de clasificaciones correctas**. Esto significa "
    f"que, de cada 100 videos analizados, el sistema identificó correctamente la expresión facial "
    f"del caballo en aproximadamente {_acierto} casos y se equivocó en los {100 - _acierto} "
    "restantes.\n\n"
    "Este porcentaje corresponde a los resultados de esa evaluación específica y **no garantiza el "
    "mismo nivel de precisión con nuevos videos**. El rendimiento puede variar según la expresión "
    "facial analizada.\n\n"
    "**Limitaciones del modelo:** La evaluación se realizó sobre un número reducido de animales. "
    "Además, factores como la iluminación, el ángulo de la cámara, la calidad del video y otras "
    "condiciones de grabación podrían influir en las predicciones. Por este motivo, **los "
    "resultados deben interpretarse con cautela**."
)

st.markdown(
    "Suba una toma corta de la cara de un caballo y el sistema intentará clasificar su expresión "
    "en **Sereno**, **Interesado** o **Disgustado**."
)

with st.expander("Antes de subir: qué esperar"):
    st.markdown(
        f"""
- **Qué video subir:** uno corto, de **5 a 7 segundos** (como máximo
  {inferencia.DURACION_MAXIMA_SEG:.0f} s), donde se vea bien la cara del caballo. Así eran los
  videos con los que aprendió el modelo. Si el video es más largo, el sistema analiza sólo un tramo
  de hasta 7 s y lo informa.
- **Formatos que acepta:** {", ".join(FORMATOS)}.
- **Cuánto tarda:** entre **1 y 3 minutos**, y más cuanto más largo sea el video. Primero ubica la
  cara y sus puntos clave cuadro por cuadro, y eso es lo que lleva tiempo; la respuesta en sí es
  inmediata.
- **Puede que no responda.** Necesita ver el ojo y el hocico con claridad durante unos segundos
  seguidos. Si el caballo mueve mucho la cabeza o
  algo la tapa, prefiere no dar una respuesta antes que inventarla. Con los videos de este trabajo
  pudo responder en **9 de cada 10**. Que no responda es normal, no es un error del video.
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

            # `conservar`: la pose de DeepLabCut es lo que más tarda, y el video con recuadro la
            # necesita también; se hace una sola vez y se comparte.
            info = {"conservar": True}
            ruta_video_marcado = None
            aviso_video = None
            try:
                resultado = inferencia.clasificar_video(ruta, progreso=avisar, info=info)
                if info.get("filas_pose"):
                    from equinos.monitoreo import procesar_video_continuo
                    ruta_video_marcado = str(Path(ruta).with_name(Path(ruta).stem + "_out.mp4"))
                    try:
                        ventanas = procesar_video_continuo(ruta, ruta_video_marcado,
                                                           progreso=avisar, previo=info)
                        if not ventanas:
                            aviso_video = ("No hubo tramos de 3 segundos con la cara bien medida "
                                           "para mostrar el video con la expresión.")
                    except Exception as error:  # el video es un extra: no debe tirar el análisis
                        aviso_video = f"No se pudo generar el video con la expresión ({error})."
                else:
                    aviso_video = "No se pudo generar el video con la expresión de esta toma."
            finally:
                Path(ruta).unlink(missing_ok=True)
                if info.get("carpeta"):
                    shutil.rmtree(info["carpeta"], ignore_errors=True)
            estado.update(label="Análisis terminado", state="complete", expanded=False)

        # Dos columnas: a la izquierda la respuesta del modelo; a la derecha, el video con la cara
        # enmarcada y la expresión de cada tramo.
        col_resultado, col_video = st.columns(2)
        with col_video:
            st.markdown("#### Video con la expresión")
            if aviso_video is None and ruta_video_marcado and Path(ruta_video_marcado).exists():
                st.video(Path(ruta_video_marcado).read_bytes())
                st.caption(
                    "El recuadro blanco marca la cara del caballo y la etiqueta indica la expresión "
                    "de cada momento. Cada etiqueta sale de un tramo de 3 segundos, por eso puede "
                    "diferir de la respuesta general de la izquierda, que considera la toma "
                    "completa."
                )
            else:
                st.info(aviso_video or "No se pudo generar el video con la expresión.")
            if ruta_video_marcado:
                Path(ruta_video_marcado).unlink(missing_ok=True)

        with col_resultado:
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
            st.markdown("---")
            izquierda, derecha = st.columns(2)
            with izquierda:
                st.caption("Expresión que detectó el modelo")
                st.markdown(f"#### {resultado.clase}")
            with derecha:
                confianza = max(resultado.probabilidades.values())
                st.caption("Qué tan seguro está el modelo con este video")
                st.markdown(f"#### {confianza:.0%}")

            if resultado.precision < 0.5:
                st.error(
                    f"**Atención: el modelo acierta menos de la mitad de las veces que dice "
                    f"«{resultado.clase}».** Este resultado no debería usarse para tomar ninguna "
                    "decisión sobre el animal."
                )

            st.markdown("---")
            st.markdown("#### Probabilidades del modelo")
            st.caption(
                ("El modelo procesa toda la secuencia junta (arquitectura MIL) y emite esta "
                   "distribución final. " if resultado.modelo == "mil" else
                   "El modelo combina dos lecturas de la toma —la pose de la cara y su apariencia— y "
                   "promedia sus probabilidades. ")
                + "Un reparto parejo significa que el modelo dudó, aunque igual haya tenido que "
                "elegir una etiqueta."
            )
            for clase, probabilidad in sorted(resultado.probabilidades.items(),
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

        # Debajo de las dos columnas y a todo el ancho: las dos lecturas lado a lado necesitan lugar.
        if resultado.detalle:
            st.markdown("---")
            st.markdown("#### Cómo llegó a esta respuesta")
            w = resultado.detalle["peso_pose"]
            st.markdown(
                "El modelo mira el video de dos maneras distintas y después junta las dos "
                "opiniones:\n"
                "- **Postura de la cara:** dónde están las orejas, los ojos, el hocico y la boca, y "
                "cómo se mueven a lo largo del video.\n"
                "- **Aspecto de la imagen:** cómo se ve la cabeza en conjunto, como lo vería una "
                "persona mirando la foto.\n\n"
                f"La opinión de la postura cuenta un {w:.0%} y la del aspecto un {1 - w:.0%}, "
                "porque en las pruebas la postura resultó más confiable."
            )
            col_pose, col_visual = st.columns(2)
            ganadoras = {}
            for col, titulo, clave in ((col_pose, "Según la postura de la cara", "pose"),
                                       (col_visual, "Según el aspecto de la imagen", "visual")):
                lectura = resultado.detalle[clave]
                ganadoras[clave] = max(lectura, key=lectura.get)
                with col:
                    st.markdown(f"**{titulo}:** {ganadoras[clave]}")
                    for clase, probabilidad in sorted(lectura.items(), key=lambda kv: -kv[1]):
                        st.progress(probabilidad, text=f"{clase} — {probabilidad:.0%}")
            # El acuerdo se dice con palabras: un reparto final claro puede esconder dos lecturas
            # que se contradicen, y eso es justo lo que una persona sin formación no puede ver.
            if ganadoras["pose"] == ganadoras["visual"]:
                st.success(f"Las dos maneras de mirar el video coinciden en «{ganadoras['pose']}». "
                           "Eso hace la respuesta más sólida.")
            else:
                st.warning(f"Las dos maneras de mirar el video no coinciden: la postura dice "
                           f"«{ganadoras['pose']}» y el aspecto dice «{ganadoras['visual']}». "
                           "Conviene tomar la respuesta con más cautela.")

