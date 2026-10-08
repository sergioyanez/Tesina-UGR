"""Candidato 4 del protocolo: MIL multimodal corregido (visual 16×512 + pose, atención sobre frames).

Es el `MILFusionModel` de `equinos.clasificacion` con las TRES correcciones de la auditoría, todas
opt-in en la rama base (la v1 sigue por omisión en la app):

  1. geometría v2 (17 columnas, `equinos.geometria`) en lugar de las 9 de v1;
  2. máscaras de validez con confianza >= 0,50 (`geometria.CONF_MINIMA_KEYPOINT`);
  3. validación interna estratificada por caballo (`particion.dividir_estratificada`), semilla 42.

Se reutilizan tal cual `MultimodalEquineDataset`, `collate_fn`, `MILFusionModel` y `pesos_de_clase`.
Mismos hiperparámetros que el original: Adam lr 1e-3, weight_decay 1e-4, batch 32, pesos de clase
inversos (calculados sobre el train interno), dropout 0,0, hasta 100 épocas, early stopping con
paciencia 10 sobre el F1 macro de la validación interna, semilla 42.

Desvíos respecto de `entrenar_mil_multimodal` (sólo de forma, no de método):
  - la clase guarda el mejor checkpoint (entrenado con el 75 % del train, como el original) para
    poder predecir luego sobre test Y sobre train (el evaluador pide el gap);
  - las características de cada clip se calculan una vez y se cachean (mismos valores; evita
    recomputar `extraer_v2` por época);
  - las predicciones de entrenamiento (para el gap) se hacen sobre TODO el train recibido, incluida
    la validación interna.

Estimación de tiempo (RTX 3070 compartida): ver `TIEMPO_ESTIMADO` abajo, medido con 1 fold.
Uso: TESINA_EQUINOS_RAIZ=<datos> python comparacion/correr_candidato_mil.py  (o este mismo archivo)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from equinos import comparacion  # noqa: E402

SEMILLA = 42
MAX_EPOCAS = 100
PACIENCIA = 10
FRACCION_VALIDACION = 0.25
BATCH = 32
GEOMETRIA = "v2"
UMBRAL_MASCARA = 0.50
TIEMPO_ESTIMADO = "1 fold (incl. carga de datos en frío) = 4,0 s; estimado 20 evaluaciones = ~1,5 min"

_cache_items: dict = {}
BITACORA: list = []     # `info` de cada ajuste (época elegida, etc.), en orden de evaluación


def _items(ids, datos):
    """(visual (16,512) tensor, geometría v2 (16,17) tensor, clase idx) por clip, cacheado."""
    import numpy as np
    import torch
    from equinos import geometria

    por_datos = _cache_items.setdefault(id(datos), {"pose": None, "items": {}})
    if por_datos["pose"] is None:
        from collections import defaultdict
        pose = defaultdict(lambda: defaultdict(list))
        for f in datos.pose():
            pose[f["clip"]][int(f["orden"])].append(f)
        por_datos["pose"] = pose
    salida = []
    for i in ids:
        if i not in por_datos["items"]:
            vis = np.asarray(datos.visual(i), dtype=np.float32)
            geo = geometria.extraer_v2(por_datos["pose"].get(i, {}), vis.shape[0], UMBRAL_MASCARA)
            por_datos["items"][i] = (torch.tensor(vis), torch.tensor(geo))
        salida.append(por_datos["items"][i])
    return salida


class _Conjunto:
    def __init__(self, ids, datos, clases):
        import torch
        self.items = _items(ids, datos)
        self.y = [clases.index(datos.etiqueta(i)) for i in ids]
        self._t = torch

    def __len__(self):
        return len(self.y)

    def __getitem__(self, k):
        v, g = self.items[k]
        return v, g, self._t.tensor(self.y[k], dtype=self._t.long)


class MILMultimodalCorregido:
    nombre = "mil_multimodal_corregido"

    def __init__(self, max_epocas=MAX_EPOCAS, paciencia=PACIENCIA, semilla=SEMILLA, verbose=False):
        self.max_epocas, self.paciencia, self.semilla, self.verbose = max_epocas, paciencia, semilla, verbose
        self.info = {}

    def ajustar(self, ids_train, datos):
        import random

        import numpy as np
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader
        from equinos import clasificacion as cl, geometria, particion

        torch.manual_seed(self.semilla); random.seed(self.semilla); np.random.seed(self.semilla)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.semilla)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        clases = list(cl.CLASES)

        clips = [{"clip": i, "clase": datos.etiqueta(i), "id_caballo": datos.caballo(i)} for i in ids_train]
        tr_int, val = particion.dividir_estratificada(clips, FRACCION_VALIDACION, particion.SEMILLA_VALIDACION)
        ds_tr = _Conjunto([c["clip"] for c in tr_int], datos, clases)
        ds_val = _Conjunto([c["clip"] for c in val], datos, clases)

        gen = torch.Generator(); gen.manual_seed(self.semilla)
        dl_tr = DataLoader(ds_tr, batch_size=BATCH, shuffle=True, collate_fn=cl.collate_fn, generator=gen)
        dl_val = DataLoader(ds_val, batch_size=BATCH, shuffle=False, collate_fn=cl.collate_fn)

        pesos = cl.pesos_de_clase([c["clase"] for c in tr_int], clases, "inversa")
        crit = nn.CrossEntropyLoss(weight=torch.tensor(pesos, dtype=torch.float32).to(self.device))
        self.modelo = cl.MILFusionModel(geom_dim=geometria.DIM[GEOMETRIA], dropout=0.0).to(self.device)
        opt = torch.optim.Adam(self.modelo.parameters(), lr=1e-3, weight_decay=1e-4)

        mejor = {"f1": -1.0, "epoca": -1, "estado": None}
        sin_mejorar, corridas = 0, 0
        for epoca in range(1, self.max_epocas + 1):
            self.modelo.train()
            for v, g, m, _, y in dl_tr:
                opt.zero_grad()
                crit(self.modelo(v.to(self.device), g.to(self.device), m.to(self.device)),
                     y.to(self.device)).backward()
                opt.step()
            corridas = epoca
            pred = self._predecir_cargador(dl_val)
            f1 = cl.f1_macro([clases[k] for k in ds_val.y], pred)
            if f1 > mejor["f1"]:
                mejor = {"f1": f1, "epoca": epoca,
                         "estado": {k: x.detach().cpu().clone() for k, x in self.modelo.state_dict().items()}}
                sin_mejorar = 0
            else:
                sin_mejorar += 1
                if sin_mejorar >= self.paciencia:
                    break
        self.modelo.load_state_dict(mejor["estado"])
        self.info = {"mejor_epoca": mejor["epoca"], "f1_validacion": mejor["f1"], "epocas_corridas": corridas,
                     "clips_train_interno": len(tr_int), "clips_validacion": len(val)}
        BITACORA.append(self.info)
        if self.verbose:
            print("   ", self.info, flush=True)
        del opt, dl_tr, dl_val, ds_tr, ds_val

    def _predecir_cargador(self, dl):
        import torch
        from equinos import clasificacion as cl
        self.modelo.eval()
        sal = []
        with torch.no_grad():
            for v, g, m, _, _ in dl:
                logits = self.modelo(v.to(self.device), g.to(self.device), m.to(self.device))
                sal.extend(cl.CLASES[k] for k in logits.argmax(dim=1).cpu().tolist())
        return sal

    def predecir(self, ids, datos):
        from torch.utils.data import DataLoader
        from equinos import clasificacion as cl
        ds = _Conjunto(list(ids), datos, list(cl.CLASES))
        return self._predecir_cargador(DataLoader(ds, batch_size=BATCH, shuffle=False,
                                                  collate_fn=cl.collate_fn))

    def __del__(self):
        try:
            import torch
            del self.modelo
            torch.cuda.empty_cache()
        except Exception:
            pass


def correr(un_fold: bool = False):
    import json
    import time
    datos = comparacion.Datos()
    if un_fold:
        reserva = comparacion.cargar_reserva(clips=datos.clips)
        dev = comparacion.clips_de_desarrollo(datos.clips, reserva)
        r, k, tr, te = next(comparacion.folds_de(dev, comparacion.cargar_particiones()))
        t = time.time()
        c = MILMultimodalCorregido(verbose=True)
        c.ajustar(tr, datos)
        p = c.predecir(te, datos)
        print(f"1 fold: {time.time() - t:.1f} s  F1 {comparacion.f1_macro(datos.etiquetas(te), p):.3f}")
        return
    t = time.time()
    res = comparacion.evaluar(
        lambda: MILMultimodalCorregido(verbose=True), MILMultimodalCorregido.nombre,
        "MIL multimodal (atención sobre 16 frames visuales + geometría v2, máscaras >= 0,50, "
        "validación interna estratificada por caballo)", datos=datos)
    (comparacion.CARPETA_RESULTADOS / f"{MILMultimodalCorregido.nombre}__epocas.json").write_text(
        json.dumps(BITACORA, indent=1), encoding="utf-8")
    g = res["global"]
    print(f"F1 macro {g['f1_macro']['media']:.3f} ± {g['f1_macro']['desvio']:.3f} | limpio "
          f"{res['limpio']['f1_macro_oof_por_repeticion']['media']:.3f} | gap "
          f"{g['gap_train_test_f1']['media']:+.3f} | {time.time() - t:.0f} s")


if __name__ == "__main__":
    correr(un_fold="--un-fold" in sys.argv)
