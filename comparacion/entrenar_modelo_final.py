"""Entrena el modelo final de la app: la fusión pose + visual con TODOS los clips.

El protocolo (`PROTOCOLO.md`) permite ajustar el modelo final con todo el material una vez que la
reserva se usó, y se usó (`reserva_uso.log`). Este script NO llama a `evaluar_en_reserva` y NO
calcula ninguna métrica sobre los clips con que se entrena: la confiabilidad del modelo es la de la
validación (`comparacion/resultados/`).

Persiste, bajo `<raíz>/modelos/fusion_pose_visual/`, un paquete joblib con los dos modelos
scikit-learn, el peso `w` (elegido con la misma validación interna por caballo de la evaluación),
las clases y metadatos (versión de scikit-learn, huella del dataset, fecha, lista de clips).

Uso: TESINA_EQUINOS_RAIZ=<datos> python comparacion/entrenar_modelo_final.py
"""
import sys
from pathlib import Path

RAIZ_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ_REPO))
sys.path.insert(0, str(RAIZ_REPO / "comparacion"))


def entrenar(guardar_en=None, verbose=True) -> tuple[dict, Path]:
    from datetime import datetime

    import sklearn

    import candidato_fusion as cf
    from equinos import comparacion, fusion, rutas

    rutas.inicializar(verbose=False)
    datos = comparacion.Datos()
    ids = sorted(datos.ids())
    if verbose:
        print(f"Ajustando con {len(ids)} clips de {len({datos.caballo(i) for i in ids})} caballos")

    modelo = cf.FusionPoseVisual()
    modelo.ajustar(ids, datos)

    paquete = {
        "modelo_pose": modelo.modelo_pose,
        "modelo_visual": modelo.modelo_visual,
        "peso": float(modelo.peso),
        "clases": [str(c) for c in modelo.clases],
        "metadatos": {
            "modelo": cf.FusionPoseVisual.nombre,
            "descripcion": "Fusión tardía: RF sobre 160 features de pose + LogReg sobre visual "
                           "medio; p = w·p_pose + (1-w)·p_visual",
            "peso_pose": float(modelo.peso),
            "puntajes_internos_f1": {str(k): float(v) for k, v in modelo.puntajes_internos.items()},
            "version_sklearn": sklearn.__version__,
            "huella_dataset": datos.huella,
            "fecha": datetime.now().isoformat(timespec="seconds"),
            "clips": ids,
            "n_clips": len(ids),
            "usa_reserva": True,
            "nota": ("Ajustado con desarrollo + reserva (la reserva ya se usó, una vez). Su "
                     "confiabilidad es la de la validación de comparacion/resultados/, no una "
                     "medida sobre estos clips."),
        },
    }
    destino = fusion.guardar_paquete(paquete, guardar_en)
    if verbose:
        print(f"w (peso de la pose) = {paquete['peso']}  puntajes internos: "
              f"{paquete['metadatos']['puntajes_internos_f1']}")
        print(f"Guardado en {destino}")
    return paquete, destino


if __name__ == "__main__":
    entrenar()
