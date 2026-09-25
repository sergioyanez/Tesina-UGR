# Registro de experimentos del objetivo 3

Qué archivo de resultados sale de qué código, en qué rama y con qué comando. Existe porque en
`datos_equinos/resultados/` conviven doce JSON de evaluación con nombres parecidos, y varios los
producen scripts que **no están en `main`** sino en ramas experimentales cuyas conclusiones fueron
«no se adopta». Sin este registro, esos números se pueden citar sin poder reproducirlos, o
confundirse con el modelo vigente.

Regla para la tesis y la defensa: **el único número del modelo vigente es el de
`objetivo3_evaluacion_v2.json`**, y va siempre al lado de `objetivo3_lineas_de_base.json`.

## Modelo vigente (rama `main`)

| Archivo | Qué es | Código | Comando | Resultado |
|---|---|---|---|---|
| `objetivo3_evaluacion_v2.json` | **Evaluación del modelo que sirve la interfaz.** MIL multimodal, 359 clips, 133 caballos, 4 folds por caballo, validación interna por caballo, *early stopping* | `scripts/evaluar_v2.py` + `equinos/clasificacion.py` | `PYTHONPATH="$PWD" python3 scripts/evaluar_v2.py` | F1 macro **0,740 ± 0,052**; accuracy 0,744 |
| `objetivo3_lineas_de_base.json` | Líneas de base sobre los mismos folds, bootstrap por caballo y control del confusor de filmación | `scripts/evaluar_lineas_de_base.py` + `equinos/lineas_de_base.py` | `PYTHONPATH="$PWD" python3 scripts/evaluar_lineas_de_base.py` (después de `evaluar_v2.py`) | Promedio estático 0,741 ± 0,020; metadatos 0,618 |
| `modelos/objetivo3_v2/mil_v2_fold{0..3}.pt` | Pesos de cada fold | `scripts/evaluar_v2.py` | ídem | — |
| `modelos/objetivo3_v2/mil_multimodal_produccion.{pt,json}` | Modelo desplegado: 359 clips, 29 épocas (media de la mejor época de la CV) | `clasificacion.entrenar_produccion`, llamado por `evaluar_v2.py` | ídem | Su métrica es la de la CV; no se mide sobre su propio train |

Re-ejecutado el 25/09/2026: 359/359 predicciones y los pesos de los 4 folds y de producción
idénticos bit a bit a los del 18/09 (misma máquina, RTX 3070, torch 2.5.1+cu121).

## Experimentos no adoptados (ramas laterales)

Todos usan el mismo conjunto (359 clips, 133 caballos) y, salvo que se indique, el mismo protocolo
que el vigente. Para reproducirlos hay que hacer `git switch <rama>`: sus scripts dependen de
cambios en `equinos/clasificacion.py` que no están en `main` (`KEYPOINTS_ENRIQUECIDOS`,
`geom_ruido`, `modo=`, `unfreeze_visual`), y las tres ramas modifican ese archivo de formas
incompatibles entre sí. **No se fusionan** porque sus mediciones concluyeron que no se adoptan; se
conservan como evidencia.

| Archivo | Rama · commits | Script y comando | F1 macro | Conclusión |
|---|---|---|---:|---|
| `objetivo3_canal_baseline_evaluacion.json` | `feat/enriquecer-canal-geometrico` · `4c4f658`, `b0b1379` (22/09) | `scripts/evaluar_canal_geometrico.py --canal baseline` | 0,740 ± 0,052 | Control: reproduce el vigente |
| `objetivo3_canal_enriquecido_evaluacion.json` | ídem | `--canal enriquecido` (7 keypoints, 21 dims) | 0,698 ± 0,103 | **No se adopta**: no mejora y duplica el desvío |
| `objetivo3_canal_ruido21_evaluacion.json` | ídem | `--canal ruido --dims 21` | 0,680 ± 0,072 | Control de capacidad |
| `objetivo3_canal_ruido9_evaluacion.json` | ídem | `--canal ruido --dims 9` | 0,680 ± 0,102 | Control de capacidad: el canal real le gana por +0,059 |
| `objetivo3_nested_simple_baseline_evaluacion.json` | `feat/validacion-cruzada-anidada-NestedCV` · `23d4b25` (23/09) | `scripts/evaluar_nested_cv.py --modo simple` | 0,740 ± 0,052 | Control: reproduce el vigente |
| `objetivo3_nested_reentrenado_baseline_evaluacion.json` | ídem | `--modo reentrenado` | 0,755 ± 0,022 | Reentrenar sobre el 100 % del train_val: +0,016, menos desvío. No adoptado |
| `objetivo3_nested_anidado_baseline_evaluacion.json` | ídem | `--modo anidado --k 4` | 0,784 ± 0,036 | **No se adopta**: el desvío de la época elegida sube de 8,4 a 23,7 (la curva promediada es una meseta) |
| `objetivo3_nested_simple_ruido_evaluacion.json` | ídem | `--modo simple --canal ruido` | 0,680 ± 0,072 | Control |
| `objetivo3_nested_anidado_ruido_evaluacion.json` | ídem | `--modo anidado --canal ruido` | 0,737 ± 0,053 | Control |
| `objetivo3_finetuning_evaluacion.json` | `feat/finetuning-visual` · `c6d6f80`, `544d323` (22/09) | `scripts/evaluar_finetuning.py` | 0,747 ± 0,057 | **No se adopta** («NO LA HAY», commit `544d323`) |

### Advertencias sobre estos experimentos

- **El 0,784 de la CV anidada no es comparable con el 0,740.** Además de elegir la época de otra
  forma, reentrena sobre más datos; y fue una de varias variantes medidas contra el mismo test. Si
  se eligiera por ser la de mayor F1, el número del ganador sería optimista.
- **`docs/informe_finetuning.md` (en su rama) contradice su propio commit.** Presenta el
  *fine-tuning* como «la mejora arquitectónica más importante» con F1 0,7737, pero ese número es de
  **una sola partición de validación** (15 % hold-out, `scripts/entrenar_finetuning.py`), no de la
  validación cruzada por caballo. Con el protocolo de 4 folds (`evaluar_finetuning.py`) da 0,747 ±
  0,057, indistinguible del vigente, y el commit `544d323` concluye que no hay mejora. Vale el
  segundo número.
- Los `.pyc` en `scripts/__pycache__/` de esos scripts son restos de haberlos ejecutado desde esas
  ramas; no son una fuente.

## Históricos (conjuntos anteriores)

| Archivo o cifra | Modelo | Clips / caballos | Protocolo | Estado |
|---|---|---:|---|---|
| F1 0,626 ± 0,065 (`openspec/project.md`, 28/08) | Un frame + voto | 141 / 73 | GroupKFold, sin validación interna | Histórico. Origen del criterio 0,691 |
| `objetivo3_evaluacion.json` (15/09) | Cabeza «promedio» | 153 / 83 | GroupKFold, 50 épocas fijas | Histórico, F1 0,704 ± 0,044. Lo produce `openspec/changes/archive/2026-08-28-clasificacion-orientada-a-video/evaluacion_final.py` |
| F1 0,754 (citado en documentos viejos) | MIL | 257 / 113 | 50 épocas fijas, otra máquina | **Sin artefacto local**: no citar |
| F1 0,618 (`docs/informe_mejora_pipeline_mil.md` §6) | MIL con validación interna | 153 / 83 | protocolo vigente | Histórico |

Ninguna cifra histórica es comparable con 0,740: cambian el conjunto y, en las dos primeras, la
selección de épocas.
