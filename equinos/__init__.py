"""Infraestructura de ejecución del pipeline de expresiones faciales equinas.

Este paquete NO contiene lógica científica: no decide cómo se segmentan los clips, ni cómo se
recorta la ROI, ni qué umbrales se aplican. Sólo resuelve dónde viven los artefactos, cómo se
reanuda una etapa interrumpida y en qué estado está el pipeline.

Existe porque el pipeline corre en Google Colab, cuyo disco es efímero y cuya sesión se corta por
límite de tiempo o inactividad. Sin persistencia y reanudación, una etapa de una hora no se puede
terminar de ejecutar de forma confiable.

Módulos:
    rutas       raíz de proyecto persistente, montaje de Drive, resolución de rutas
    catalogo    qué videos de Drive pertenecen a cada clase, incluidas sus subcarpetas
    checkpoint  estado reanudable por etapa, invalidado ante cambio de configuración
    cache       empaquetado de artefactos numerosos, que Drive montado maneja mal
    etapas      registro del pipeline y diagnóstico de su estado
    pose        selección de keypoints faciales sobre la pose completa y su barrido
    reporte     lectura de los resultados del preprocesamiento facial
"""

from equinos import cache, catalogo, checkpoint, etapas, pose, reporte, rutas

__all__ = ["rutas", "catalogo", "checkpoint", "cache", "etapas", "pose", "reporte"]
