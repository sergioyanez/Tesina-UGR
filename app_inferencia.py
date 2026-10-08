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
def _reserva():
    """Precisión por clase de la fusión en la reserva (20 caballos, un solo uso)."""
    rutas.inicializar(verbose=False)
    return fusion.precision_reserva()


def _rango_de_cada_diez(a: float, b: float) -> str:
    """Dos mediciones en una frase: «entre 7 y 9 de cada 10 veces», o una sola cifra si coinciden."""
    lo, hi = sorted((min(round(a * 10), 10), min(round(b * 10), 10)))
    if lo == hi:
        return _de_cada_diez(a)
    return f"entre {lo} y {hi} de cada 10 veces"


def _de_cada_diez(precision: float) -> str:
    """«unas 7 de cada 10 veces»: la proporción en palabras, para quien no lee porcentajes."""
    n = round(precision * 10)
    if n >= 10:
        return "más de 9 de cada 10 veces"
    if n <= 0:
        return "casi nunca"
    return f"unas {n} de cada 10 veces"


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
            st.caption("Expresión que detectó el modelo")
            st.markdown(f"# {resultado.clase}")
        with derecha:
            confianza = max(resultado.probabilidades.values())
            st.caption("Qué tan seguro está el modelo con este video")
            st.markdown(f"# {confianza:.0%}")

        # Cuánto confiar en la etiqueta, en palabras. Con la fusión hay dos mediciones fuera de
        # muestra (desarrollo 5×4 y reserva); cuando no coinciden se da el rango, no una sola cifra.
        if resultado.detalle:
            pr = resultado.detalle["precision_reserva"]
            rango = _rango_de_cada_diez(pr["precision"], resultado.precision)
            st.markdown(f"**¿Cuánto confiar en esta respuesta?** Cuando el modelo dice "
                        f"«{resultado.clase}», acierta **{rango}**.")
            st.caption(
                "Lo comprobamos dos veces, siempre con caballos que el modelo no había visto. "
                f"En la primera prueba dijo «{resultado.clase}» {resultado.veces_que_lo_dijo} veces "
                f"y acertó {resultado.acertadas}; en la prueba final, con otros 20 caballos, lo dijo "
                f"{pr['veces_que_lo_dijo']} veces y acertó {pr['acertadas']}."
                + (" Como las dos pruebas no dieron lo mismo, damos un rango y no un número exacto." if "entre" in rango else "")
            )
        else:
            st.markdown(f"**¿Cuánto confiar en esta respuesta?** Cuando el modelo dice "
                        f"«{resultado.clase}», acierta **{_de_cada_diez(resultado.precision)}**.")
            st.caption(
                f"Lo comprobamos con caballos que el modelo no había visto al entrenar: dijo "
                f"«{resultado.clase}» {resultado.veces_que_lo_dijo} veces y acertó "
                f"{resultado.acertadas} ({resultado.precision:.0%})."
            )

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
        # La misma frase que el recuadro de arriba: el rango entre desarrollo y reserva (fusión).
        reserva = _reserva() if resultado.detalle else None
        for clase, probabilidad in sorted(resultado.probabilidades.items(),
                                          key=lambda kv: -kv[1]):
            p_dev = precisiones[clase]["precision"]
            frase = (_rango_de_cada_diez(reserva[clase]["precision"], p_dev) if reserva
                     else _de_cada_diez(p_dev))
            st.progress(probabilidad,
                        text=f"{clase} — {probabilidad:.0%}  (cuando dice «{clase}», acierta {frase})")

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
                           "Tomá la respuesta con más cautela.")

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
        with st.expander("Cómo se probó el modelo y qué tener en cuenta"):
            m = _metricas()
            st.markdown(
                f"""
**Cómo se probó**

- El modelo aprendió con videos de caballos filmados en Tandil. Una persona miró cada video y lo
  etiquetó como Sereno, Interesado o Disgustado.
- Para saber cuánto acierta, **siempre se lo probó con caballos que no había visto al aprender**.
  Así no puede acertar porque reconoce al caballo: tiene que leer su expresión.
- Primero se lo probó varias veces con distintos grupos de caballos. Después, una única vez, con
  {m['caballos_reserva']} caballos que se habían guardado aparte desde el principio. Con esos
  caballos acertó en {m['accuracy_reserva']:.0%} de los videos. De esas dos pruebas salen las cifras
  de «¿Cuánto confiar en esta respuesta?».

**Qué tener en cuenta**

- **No es un diagnóstico veterinario.** Es una ayuda que puede equivocarse.
- Necesita ver el ojo y el hocico del caballo durante varios segundos seguidos. Si no puede,
  prefiere no responder.
- La postura de la cara es la que más pesa en la decisión, y una parte de lo que mide también
  depende de cómo se filmó el video (por ejemplo, cuánto dura). Con videos filmados todos de la
  misma manera, el modelo acierta algo menos.
- La versión que usa esta aplicación se volvió a entrenar al final con todos los caballos,
  incluidos los {m['caballos_reserva']} de la prueba final, para aprovechar todos los datos. Por eso
  las cifras son una estimación de cuánto acierta, no una medida exacta de esta versión.
"""
            )
            if resultado.modelo == "mil":
                st.caption("Esta respuesta la dio el modelo anterior (MIL), elegido a mano. Lo que "
                           "dice este recuadro describe al modelo principal de la aplicación.")
