"""La interfaz Streamlit arranca (AppTest, sin navegador) y muestra métricas leídas del JSON vigente."""
import os
from streamlit.testing.v1 import AppTest
from comun import *

os.chdir(WT)
at = AppTest.from_file(str(WT / "app_inferencia.py"), default_timeout=120).run()
txt = [e.value for e in at.error]
m = leer_json(RES / "objetivo3_evaluacion_v2.json")["resumen"]
res = {"excepciones_al_renderizar": [str(e.value)[:200] for e in at.exception],
       "titulo": [t.value for t in at.title], "aviso_visible": txt,
       "el_aviso_cita_accuracy_vigente": f"{m['accuracy']:.0%}" in " ".join(txt),
       "el_aviso_cita_f1_vigente": f"{m['f1_macro']:.3f}" in " ".join(txt),
       "alcance": "Se ejecuta el script de la interfaz sin subir un video; la inferencia sobre videos se mide en orientacion_inferencia.json"}
guardar_json(res, "app_streamlit.json")
print(json.dumps(res, ensure_ascii=False, indent=1))
