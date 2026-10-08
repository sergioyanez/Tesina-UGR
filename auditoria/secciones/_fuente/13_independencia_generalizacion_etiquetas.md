# 13. Independencia de las muestras, generalización y calidad de las etiquetas

## 13.1 Independencia

Los 359 clips **no son 359 observaciones independientes** (`datos/independencia.json`, `datos/dataset.json`):

- 244 videos de origen: 175 aportan un solo clip, 69 aportan dos o más (hasta 7). **El 51,3 % de los clips pertenece a videos con más de un clip.**
- 133 caballos: 2,70 clips por caballo en promedio (mediana 2); 58 caballos tienen un único clip y los 10 con más clips reúnen el 27,6 %.
- **Correlación intra-caballo del acierto**: ICC (ANOVA de una vía sobre el indicador de acierto) = 0,31; efecto de diseño ≈ 1,52. El tamaño efectivo para estimar la accuracy está **entre ≈ 236 clips (corrección por diseño) y 133 (caballos)**; la estimación del ICC con muchos caballos de un clip es ruidosa, por lo que se usa el rango.
- Consecuencia sobre la incertidumbre: el IC 95 % de la accuracy es **[0,679; 0,801] con bootstrap por caballo** frente a [0,696; 0,786] con el intervalo de Wilson ingenuo por clip (ancho 0,122 frente a 0,090: el ingenuo subestima la incertidumbre en ≈ 26 %). Los JSON de evaluación informan el desvío entre 4 folds; el IC por caballo existe desde el 25/09 en `objetivo3_lineas_de_base.json` y coincide con el recalculado aquí.
- Los clips de un mismo video son contiguos y sin solape; entre videos de la misma sesión (misma fecha) hay dependencia de contexto (§6.4).

## 13.2 Generalización a caballos no vistos (interna)

El protocolo mide exactamente esto: cada clip se predice con un modelo que **nunca vio a su caballo**. El F1 macro 0,740 (IC 95 % [0,677; 0,797]) estima el rendimiento sobre **caballos nuevos de la misma población**, con la salvedad de dependencia de sesión de §6.4. Los 133 caballos son muy pocos para caracterizar subpoblaciones: el rendimiento por caballo es heterogéneo (55 caballos con algún error; 9 con ≥ 2 clips fallados en el 100 % de los casos).

## 13.3 Generalización externa

**No evaluada y, con los datos disponibles, no evaluable.** Todo el material lo filmó una sola persona, con dispositivos móviles, en establecimientos de la provincia de Buenos Aires (hipódromos de Tandil y Azul, haras, quintas y campus; `docs/protocolo_de_etiquetado.md` §3). No pudo verificarse con los archivos disponibles: la **raza** de los caballos (no hay metadato; el material proviene mayormente de haras de Sangre Pura de Carrera según el formulario), la **cámara/dispositivo** por video (sólo se pueden inferir contenedor, códec y rotación), ni la **iluminación**. Hay tres clases fuertemente asociadas al tipo de establecimiento y situación (el formulario lo confirma): `Interesado` en box de hipódromo, `Disgustado` en interacciones de manada en cría extensiva, `Sereno` en descanso. **El modelo no se probó en otro centro, otra cámara ni otro observador**, de modo que su rendimiento fuera de este conjunto es una incógnita.

## 13.4 Calidad de las etiquetas

| Aspecto | Hallazgo | Fuente |
|---|---|---|
| Quién etiquetó | Una sola persona, veterinaria **codificadora certificada de EquiFACS**, que además filmó y observó el contexto (**no ciega**) | `docs/protocolo_de_etiquetado.md` §3 |
| Criterio | Escrito por clase, con unidades de acción de EquiFACS y etogramas publicados; usa **contexto, comportamiento y postura**, no sólo la cara. Las unidades de acción son «pueden estar presentes», no obligatorias | ídem |
| Acuerdo entre observadores | **Kappa de Fleiss 0,9877 sólo en un piloto con imágenes y otros observadores**; sobre los videos del dataset **no se midió** (decisión del 07/10/2026) | ídem §3, §5 |
| Etiqueta por video, heredada a los clips | Supone que la expresión dura toda la toma; sostenido por el recorte de tramos de la etiquetadora | ídem §1, §4 |
| Clases excluidas por falta de acuerdo en el piloto | Miedo, intensidad elevada, conductas repetitivas, caballos atados | ídem §3 |
| Consistencia de nombres de clase | Idéntica en todos los artefactos; el prefijo del archivo coincide con `id_caballo` en 393/393 clips | `datos/dataset.json` |
| Posible ruido de etiqueta | 9 caballos con ≥ 2 clips son fallados en el 100 % de sus clips por el modelo (`013`, `053`, `044`, `251`, `916`, `224`, `920`, `215`, `905`): candidatos a revisión de etiqueta o de identidad | `datos/independencia.json` |
| Confusores de contexto | Box de hipódromo → `Interesado`; manada → `Disgustado`; rotación, contenedor y fecha asociados a la clase (§10.3) | ídem; formulario |

**Lo que cambió respecto del 25/09:** el protocolo existe y está documentado (antes «sin protocolo»), por lo que R14 pasa de NO CUMPLIDO a CUMPLIDO PARCIALMENTE. **Lo que no cambió:** un único observador no ciego, sin acuerdo medido sobre el dataset, y por lo tanto **techo de rendimiento desconocido**: no se puede separar cuánto del 0,740 es error del modelo y cuánto ambigüedad de la etiqueta.
