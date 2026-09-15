"""Clasificación de la expresión a partir del clip: métricas, líneas de base y modelo final.

Este módulo contiene lo que el notebook 03 necesita para reproducir el objetivo 3. Vive acá y no
en el directorio del change porque el change se archiva, y un notebook que importe desde
`openspec/changes/...` se rompe el día que eso pasa.

**El modelo final es el clasificador de un frame con voto.** Se lo diseñó como línea de base para
poner a prueba la hipótesis del trabajo —que la evolución temporal es lo que permite clasificar— y
terminó ganándole a las 18 configuraciones probadas. Es **invariante al orden**: barajar los
frames del clip no cambia una predicción. Ese hecho es el resultado central del objetivo 3, y no
un detalle de implementación.

La accuracy global no se calcula en ningún lado. Con el reparto de este dataset —`Disgustado`
15 %, `Interesado` 40 %, `Sereno` 45 %— un clasificador que nunca prediga la clase minoritaria
llega al 85 % sin haber aprendido nada.
"""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict

from equinos import rutas

CLASES = ("Disgustado", "Interesado", "Sereno")
SEMILLA = 42

# Criterio de aceptación declarado en el change, ANTES de entrenar: la línea de base más alta
# (un frame, 0,626) más un desvío entre folds (0,065). No se toca a posteriori.
CRITERIO_ACEPTACION = 0.691


# --- métricas -------------------------------------------------------------------------------
# Se implementan acá y no se importan de sklearn para que quede a la vista qué se promedia.

def f1_macro(reales, predichas, clases=CLASES):
    """Media NO ponderada del F1 por clase: las tres pesan igual, aunque no sean igual de
    frecuentes. Es la métrica principal del objetivo 3."""
    f1s = []
    for c in clases:
        vp = sum(1 for r, p in zip(reales, predichas) if r == c and p == c)
        fp = sum(1 for r, p in zip(reales, predichas) if r != c and p == c)
        fn = sum(1 for r, p in zip(reales, predichas) if r == c and p != c)
        precision = vp / (vp + fp) if (vp + fp) else 0.0
        recall = vp / (vp + fn) if (vp + fn) else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if (precision + recall) else 0.0)
    return sum(f1s) / len(f1s)


def balanced_accuracy(reales, predichas, clases=CLASES):
    """Media del recall por clase, sobre las clases presentes en el conjunto evaluado."""
    recalls = [r for r in (recall_de(reales, predichas, c) for c in clases) if r == r]
    return sum(recalls) / len(recalls) if recalls else float("nan")


def recall_de(reales, predichas, clase):
    presentes = [p for r, p in zip(reales, predichas) if r == clase]
    return sum(1 for p in presentes if p == clase) / len(presentes) if presentes else float("nan")


def resumen(valores):
    """Media y desvío. El desvío ENTRE FOLDS se informa siempre: con 3 a 7 clips de la clase
    minoritaria por fold, una media sola oculta cuánto se mueve el resultado."""
    limpios = [v for v in valores if v == v]
    if not limpios:
        return float("nan"), float("nan")
    return statistics.fmean(limpios), (statistics.stdev(limpios) if len(limpios) > 1 else 0.0)


# --- particiones ----------------------------------------------------------------------------

def folds_por_ejemplar(clips):
    """El protocolo del trabajo: los folds persistidos, que agrupan por `id_caballo`."""
    n = max(c["fold"] for c in clips) + 1
    return [([c for c in clips if c["fold"] != f], [c for c in clips if c["fold"] == f])
            for f in range(n)]


def folds_ingenuos(clips, n=4, semilla=SEMILLA):
    """El protocolo que NO se usa: reparte clips al azar, sin agrupar por ejemplar.

    Existe únicamente para medir la fuga que el protocolo real elimina. Ningún resultado del
    trabajo se reporta sobre estos folds.
    """
    import random

    mezclados = list(clips)
    random.Random(semilla).shuffle(mezclados)
    asignados = [(c, i % n) for i, c in enumerate(mezclados)]
    return [([c for c, f in asignados if f != k], [c for c, f in asignados if f == k])
            for k in range(n)]


# --- modelos --------------------------------------------------------------------------------

def predecir_mayoritaria(entrenamiento, prueba, guardar_en=None, **_):
    mayoritaria = Counter(c["clase"] for c in entrenamiento).most_common(1)[0][0]
    return [mayoritaria] * len(prueba), {"sin_ver": 0}


def predecir_por_identidad(entrenamiento, prueba, guardar_en=None, **_):
    """Predice la clase más frecuente del MISMO caballo en entrenamiento.

    Con partición por ejemplar el caballo de test nunca se vio, así que cae al voto mayoritario y
    empata con la línea de base trivial. Eso no es un defecto: es la demostración de que la
    partición neutraliza el confounder de individuo. El contador `sin_ver` lo hace explícito.
    """
    por_caballo = defaultdict(Counter)
    for c in entrenamiento:
        por_caballo[c["id_caballo"]][c["clase"]] += 1
    mayoritaria = Counter(c["clase"] for c in entrenamiento).most_common(1)[0][0]

    predichas, sin_ver = [], 0
    for c in prueba:
        visto = por_caballo.get(c["id_caballo"])
        if visto:
            predichas.append(visto.most_common(1)[0][0])
        else:
            predichas.append(mayoritaria)
            sin_ver += 1
    return predichas, {"sin_ver": sin_ver}


def predecir_un_frame(entrenamiento, prueba, vectores, indice, guardar_en=None):
    """EL MODELO FINAL. Clasifica cada frame por separado y vota entre los 16 del clip.

    Usa todos los frames pero es invariante a su orden. Cabeza lineal sobre las características
    congeladas, con pesos por clase balanceados porque la métrica es F1 macro.
    """
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    def aplanar(clips):
        filas, etiquetas, de_clip = [], [], []
        for c in clips:
            v = vectores[indice[c["clip"]]]
            filas.append(v)
            etiquetas.extend([c["clase"]] * len(v))
            de_clip.extend([c["clip"]] * len(v))
        return np.concatenate(filas), np.array(etiquetas), np.array(de_clip)

    X_ent, y_ent, _ = aplanar(entrenamiento)
    X_pru, _, clip_de_frame = aplanar(prueba)

    escalador = StandardScaler().fit(X_ent)
    modelo = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEMILLA)
    modelo.fit(escalador.transform(X_ent), y_ent)

    # Voto mayoritario, con desempate alfabético fijo para no depender del orden de iteración.
    votos = defaultdict(Counter)
    for clip, pred in zip(clip_de_frame, modelo.predict(escalador.transform(X_pru))):
        votos[clip][pred] += 1
    predichas = [sorted(votos[c["clip"]].items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
                 for c in prueba]

    if guardar_en is not None:
        guardar_en.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            guardar_en, coeficientes=modelo.coef_, intercepto=modelo.intercept_,
            clases=np.array(modelo.classes_), media=escalador.mean_, escala=escalador.scale_)
    return predichas, {"sin_ver": 0}

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import math

class MultimodalEquineDataset(Dataset):
    def __init__(self, clips, vectores, indice, pose_filas):
        self.clips = clips
        self.vectores = vectores
        self.indice = indice
        from collections import defaultdict
        self.pose_por_clip = defaultdict(lambda: defaultdict(list))
        for f in pose_filas:
            self.pose_por_clip[f['clip']][int(f['orden'])].append(f)
        self.clases = ["Disgustado", "Interesado", "Sereno"]
        self.clase_a_idx = {c: i for i, c in enumerate(self.clases)}

    def __len__(self):
        return len(self.clips)
        
    def _mejor_de(self, candidatos, frames_medidos):
        mejor = (float("nan"), float("nan"), -1.0)
        for c in candidatos:
            for f in frames_medidos:
                if f['bodypart'].lower() == c.lower() or c.lower() in f['bodypart'].lower():
                    conf = float(f['confianza'])
                    if conf > mejor[2]:
                        mejor = (float(f['x']), float(f['y']), conf)
        return mejor

    def __getitem__(self, idx):
        clip_info = self.clips[idx]
        clip_name = clip_info["clip"]
        clase_idx = self.clase_a_idx[clip_info["clase"]]
        v_vis = self.vectores[self.indice[clip_name]]
        seq_len = v_vis.shape[0]
        
        geom_features = []
        val_masks = []
        pose_del_clip = self.pose_por_clip.get(clip_name, {})
        
        for orden in range(seq_len):
            medidos = pose_del_clip.get(orden, [])
            x_ojo, y_ojo, c_ojo = self._mejor_de(["eye"], medidos)
            x_hoc, y_hoc, c_hoc = self._mejor_de(["nose", "snout", "muzzle", "upper_jaw", "mouth"], medidos)
            x_ore, y_ore, c_ore = self._mejor_de(["earbase", "earend"], medidos)
            
            if c_ojo > 0 and c_hoc > 0:
                cx, cy = (x_ojo + x_hoc) / 2, (y_ojo + y_hoc) / 2
                escala = math.hypot(x_ojo - x_hoc, y_ojo - y_hoc)
                if escala < 1e-6:
                    escala = 1.0
            else:
                cx, cy, escala = 0.0, 0.0, 1.0
                
            def normalizar(x, y, c):
                if c > 0.0 and not math.isnan(x) and not math.isnan(y):
                    return (x - cx) / escala, (y - cy) / escala, 1.0
                return 0.0, 0.0, 0.0
                
            nx_ojo, ny_ojo, val_ojo = normalizar(x_ojo, y_ojo, c_ojo)
            nx_hoc, ny_hoc, val_hoc = normalizar(x_hoc, y_hoc, c_hoc)
            nx_ore, ny_ore, val_ore = normalizar(x_ore, y_ore, c_ore)
            
            geom_features.append([nx_ojo, ny_ojo, nx_hoc, ny_hoc, nx_ore, ny_ore])
            val_masks.append([val_ojo, val_hoc, val_ore])
            
        v_vis = torch.tensor(v_vis, dtype=torch.float32)
        v_geom = torch.tensor(geom_features, dtype=torch.float32)
        v_val = torch.tensor(val_masks, dtype=torch.float32)
        v_geom_full = torch.cat([v_geom, v_val], dim=-1)
        return v_vis, v_geom_full, torch.tensor(clase_idx, dtype=torch.long)

def collate_fn(batch):
    v_vis_list, v_geom_list, labels = zip(*batch)
    lengths = torch.tensor([v.shape[0] for v in v_vis_list])
    v_vis_pad = nn.utils.rnn.pad_sequence(v_vis_list, batch_first=True, padding_value=0.0)
    v_geom_pad = nn.utils.rnn.pad_sequence(v_geom_list, batch_first=True, padding_value=0.0)
    batch_size = len(batch)
    max_len = v_vis_pad.shape[1]
    attn_mask = torch.arange(max_len).expand(batch_size, max_len) < lengths.unsqueeze(1)
    labels = torch.stack(labels)
    return v_vis_pad, v_geom_pad, attn_mask, lengths, labels

class MILFusionModel(nn.Module):
    def __init__(self, vis_dim=512, geom_dim=9, geom_embed_dim=128, num_classes=3):
        super().__init__()
        self.geom_mlp = nn.Sequential(
            nn.Linear(geom_dim, 64),
            nn.ReLU(),
            nn.Linear(64, geom_embed_dim),
            nn.ReLU()
        )
        fusion_dim = vis_dim + geom_embed_dim
        self.attention = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1, bias=False)
        )
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, v_vis, v_geom, attn_mask, return_attn=False):
        geom_embed = self.geom_mlp(v_geom)
        fusion = torch.cat([v_vis, geom_embed], dim=-1)
        attn_scores = self.attention(fusion).squeeze(-1)
        attn_scores = attn_scores.masked_fill(~attn_mask, float('-1e9'))
        attn_weights = torch.softmax(attn_scores, dim=1)
        bag_rep = torch.sum(fusion * attn_weights.unsqueeze(-1), dim=1)
        logits = self.classifier(bag_rep)
        if return_attn:
            return logits, attn_weights
        return logits

def entrenar_mil_multimodal(entrenamiento, prueba, vectores, indice, pose_filas, guardar_en=None):
    from torch.utils.tensorboard import SummaryWriter
    import time
    import matplotlib.pyplot as plt

    torch.manual_seed(SEMILLA)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ds_ent = MultimodalEquineDataset(entrenamiento, vectores, indice, pose_filas)
    ds_pru = MultimodalEquineDataset(prueba, vectores, indice, pose_filas)
    
    # Calculate class weights for CrossEntropyLoss
    clases = [c["clase"] for c in entrenamiento]
    from collections import Counter
    conteos = Counter(clases)
    pesos_clase = []
    for c in ds_ent.clases:
        pesos_clase.append(len(clases) / (len(ds_ent.clases) * max(1, conteos[c])))
    weight_tensor = torch.tensor(pesos_clase, dtype=torch.float32).to(device)
    
    dl_ent = DataLoader(ds_ent, batch_size=32, shuffle=True, collate_fn=collate_fn)
    dl_pru = DataLoader(ds_pru, batch_size=32, shuffle=False, collate_fn=collate_fn)
    
    model = MILFusionModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    
    # Setup TensorBoard
    log_dir = f"runs/mil_multimodal_{int(time.time())}"
    writer = SummaryWriter(log_dir=log_dir)
    
    for epoch in range(50):
        model.train()
        train_loss = 0.0
        for v_vis, v_geom, attn_mask, _, labels in dl_ent:
            optimizer.zero_grad()
            logits = model(v_vis.to(device), v_geom.to(device), attn_mask.to(device))
            loss = criterion(logits, labels.to(device))
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * v_vis.size(0)
            
        train_loss /= len(ds_ent)
        writer.add_scalar('Loss/train', train_loss, epoch)
            
    model.eval()
    predichas = []
    with torch.no_grad():
        for i, (v_vis, v_geom, attn_mask, lengths, _) in enumerate(dl_pru):
            logits, attn_weights = model(v_vis.to(device), v_geom.to(device), attn_mask.to(device), return_attn=True)
            preds = torch.argmax(logits, dim=1)
            predichas.extend([ds_ent.clases[p] for p in preds.cpu().numpy()])
            
            # Log attention weights for the first batch
            if i == 0:
                fig, ax = plt.subplots(figsize=(10, 4))
                sample_idx = 0
                seq_len = lengths[sample_idx].item()
                weights = attn_weights[sample_idx, :seq_len].cpu().numpy()
                ax.bar(range(seq_len), weights)
                ax.set_title(f"Pesos de Atención - Clip Validación")
                ax.set_xlabel("Frame")
                ax.set_ylabel("Atención")
                writer.add_figure('Attention/val_sample', fig, global_step=0)
                plt.close(fig)
                
    writer.close()
    if guardar_en is not None:
        guardar_en.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), guardar_en.with_suffix('.pt'))
        
    return predichas, {"sin_ver": 0}


# --- evaluación -----------------------------------------------------------------------------

def evaluar(modelo, particiones, etiqueta="", pesos_en=None, **kwargs):
    """Validación cruzada. Devuelve métricas por fold y predicciones FUERA DE MUESTRA.

    `pesos_en` es una función `fold -> ruta` para persistir el modelo de cada fold. Los modelos
    que no guardan pesos —las líneas de base triviales— ignoran el argumento.
    """
    por_fold, fuera_de_muestra, sin_ver = [], [], 0
    for i, (entrenamiento, prueba) in enumerate(particiones):
        if not prueba or not entrenamiento:
            continue
        predichas, extra = modelo(entrenamiento, prueba,
                                  guardar_en=(pesos_en(i) if pesos_en else None), **kwargs)
        reales = [c["clase"] for c in prueba]
        sin_ver += extra.get("sin_ver", 0)
        por_fold.append({
            "fold": i, "n": len(prueba),
            "f1_macro": f1_macro(reales, predichas),
            "balanced_accuracy": balanced_accuracy(reales, predichas),
            "recall_disgustado": recall_de(reales, predichas, "Disgustado"),
        })
        fuera_de_muestra.extend(
            {"clip": c["clip"], "id_caballo": c["id_caballo"], "real": c["clase"], "predicha": p}
            for c, p in zip(prueba, predichas))

    return {
        "etiqueta": etiqueta,
        "clips": sum(f["n"] for f in por_fold),
        "f1_macro": resumen([f["f1_macro"] for f in por_fold]),
        "balanced_accuracy": resumen([f["balanced_accuracy"] for f in por_fold]),
        "recall_disgustado": resumen([f["recall_disgustado"] for f in por_fold]),
        "sin_ver": sin_ver,
        "por_fold": por_fold,
        "predicciones": fuera_de_muestra,
    }


def imprimir(resultado):
    r = resultado
    print(f"\n{r['etiqueta']}")
    print(f"  F1 macro           {r['f1_macro'][0]:.3f} ± {r['f1_macro'][1]:.3f}")
    print(f"  Balanced accuracy  {r['balanced_accuracy'][0]:.3f} ± {r['balanced_accuracy'][1]:.3f}")
    print(f"  Recall Disgustado  {r['recall_disgustado'][0]:.3f} ± {r['recall_disgustado'][1]:.3f}")
    if r["sin_ver"]:
        print(f"  Cayó al voto mayoritario en {r['sin_ver']} de {r['clips']} clips "
              f"({r['sin_ver'] / r['clips']:.0%}): no había visto nunca a ese caballo")
    print("  por fold: " + "  ".join(f"{f['f1_macro']:.3f}" for f in r["por_fold"]))
    return r


def matriz_de_confusion(predicciones, clases=CLASES):
    print("\nMATRIZ DE CONFUSIÓN — fuera de muestra, agregada sobre los folds")
    encabezado = "real / predicha"
    print(f"{encabezado:<16}" + "".join(f"{c:>13}" for c in clases) + f"{'total':>8}")
    for real in clases:
        fila = [sum(1 for p in predicciones if p["real"] == real and p["predicha"] == pred)
                for pred in clases]
        print(f"{real:<16}" + "".join(f"{v:>13}" for v in fila) + f"{sum(fila):>8}")
    print(f"{'total':<16}" + "".join(
        f"{sum(1 for p in predicciones if p['predicha'] == c):>13}" for c in clases))


def errores_por_ejemplar(predicciones, peores=10):
    """Si los fallos se concentran en pocos caballos, eso es información sobre el DATASET."""
    print("\nERRORES POR EJEMPLAR")
    por_caballo = defaultdict(lambda: [0, 0])
    for p in predicciones:
        por_caballo[p["id_caballo"]][0] += 1
        por_caballo[p["id_caballo"]][1] += (p["real"] != p["predicha"])
    fallan = {k: v for k, v in por_caballo.items() if v[1]}
    total = sum(v[1] for v in por_caballo.values())
    if not total:
        print("  sin errores")
        return
    ranking = sorted(fallan.items(), key=lambda kv: (-kv[1][1], kv[0]))
    acumulado = sum(v[1] for _, v in ranking[:peores])
    print(f"  {len(fallan)} de {len(por_caballo)} ejemplares fallan alguna vez "
          f"({total} errores en total)")
    print(f"  los {min(peores, len(ranking))} peores concentran {acumulado} ({acumulado / total:.0%})")
    print("  " + "  ".join(f"{c}:{v[1]}/{v[0]}" for c, v in ranking[:peores]))


def contra_el_criterio(resultado, criterio=CRITERIO_ACEPTACION):
    """Contrasta contra el criterio declarado ANTES de entrenar, y lo informa aunque no se cumpla.

    Que se informe igual es la condición con la que se declaró: un criterio que sólo se reporta
    cuando se cumple no es un criterio.
    """
    medio = resultado["f1_macro"][0]
    cumple = medio > criterio
    print(f"\n  criterio declarado (tarea 2.4) : {criterio:.3f}")
    print(f"  F1 macro obtenido              : {medio:.3f} ± {resultado['f1_macro'][1]:.3f}")
    print(f"  {'CUMPLE' if cumple else 'NO CUMPLE'}  (diferencia {medio - criterio:+.3f})")
    return cumple


def ruta_pesos(fold):
    """Pesos de un fold de validación cruzada. Respaldan la métrica; NO se despliegan."""
    return rutas.carpeta("modelos") / "objetivo3" / f"un_frame_fold{fold}.npz"


# --- confiabilidad que se le muestra al usuario ----------------------------------------------

ARCHIVO_EVALUACION = "objetivo3_evaluacion.json"


def precision_por_clase(predicciones=None):
    """De cuántas veces que el modelo dijo una clase, cuántas acertó.

    **Es el número que hay que mostrarle a quien usa la interfaz**, y no el recall. El recall
    contesta «de los `Disgustado` reales, ¿cuántos encontró?», que le sirve a quien evalúa el
    modelo. Quien recibe una predicción se hace la pregunta inversa: «dijo esto, ¿cuántas veces
    acierta cuando lo dice?». Son números distintos y confundirlos engaña.

    Se deriva de las predicciones FUERA DE MUESTRA de la validación cruzada. No se escribe a
    mano: escrita a mano diverge el día que se re-evalúe, y quedaría una interfaz informando una
    confiabilidad que ya no corresponde a ningún modelo.
    """
    import json

    if predicciones is None:
        ruta = rutas.carpeta("resultados") / ARCHIVO_EVALUACION
        if not ruta.exists():
            raise FileNotFoundError(
                f"No existe {ruta}. La produce la evaluación del objetivo 3; sin ella no hay de "
                "dónde sacar la confiabilidad que la interfaz tiene que mostrar.")
        predicciones = json.loads(ruta.read_text(encoding="utf-8"))["predicciones_fuera_de_muestra"]

    salida = {}
    for clase in CLASES:
        dijo = [p for p in predicciones if p["predicha"] == clase]
        acerto = sum(1 for p in dijo if p["real"] == clase)
        salida[clase] = {
            "precision": acerto / len(dijo) if dijo else float("nan"),
            "acertadas": acerto,
            "veces_que_lo_dijo": len(dijo),
        }
    return salida


def texto_de_confiabilidad(clase, precisiones=None):
    """El texto que acompaña a toda predicción. Vive acá, en un solo lugar.

    Si estuviera repetido en la interfaz, cambiar el modelo dejaría el texto viejo en pantalla.
    """
    precisiones = precisiones or precision_por_clase()
    p = precisiones[clase]
    return (
        f"Cuando este modelo predijo «{clase}», acertó {p['acertadas']} de "
        f"{p['veces_que_lo_dijo']} veces ({p['precision']:.0%}) en validación cruzada.\n"
        f"El modelo NO alcanzó el criterio declarado para considerarse útil "
        f"(F1 macro 0,626 contra 0,691 exigido).\n"
        f"Este resultado NO es un diagnóstico veterinario."
    )


# --- modelo de producción -------------------------------------------------------------------

ARCHIVO_PRODUCCION = "un_frame_produccion.pt"
ARCHIVO_PROCEDENCIA = "un_frame_produccion.json"


def ruta_produccion():
    return rutas.carpeta("modelos") / "objetivo3" / ARCHIVO_PRODUCCION


def entrenar_produccion(clips, vectores, indice, pose_filas, metrica_validacion=None, verbose=True):
    """Entrena sobre TODOS los clips íntegros y persiste el modelo que va a usar la interfaz.

    **Su métrica no se mide acá.** Es la de validación cruzada, y no se vuelve a calcular sobre
    los datos con los que se entrenó. Entrenar con todo mejora el modelo, pero no hay forma
    honesta de medir cuánto sin dejar material afuera; informar una exactitud calculada sobre el
    conjunto de entrenamiento sería informar cuánto memorizó.
    """
    import json
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from collections import Counter

    torch.manual_seed(SEMILLA)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = MultimodalEquineDataset(clips, vectores, indice, pose_filas)

    clases_lista = [c["clase"] for c in clips]
    conteos = Counter(clases_lista)
    pesos_clase = []
    for c in ds.clases:
        pesos_clase.append(len(clases_lista) / (len(ds.clases) * max(1, conteos[c])))
    weight_tensor = torch.tensor(pesos_clase, dtype=torch.float32).to(device)

    dl = DataLoader(ds, batch_size=32, shuffle=True, collate_fn=collate_fn)

    modelo = MILFusionModel().to(device)
    optimizer = torch.optim.Adam(modelo.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    for epoch in range(50):
        modelo.train()
        for v_vis, v_geom, attn_mask, _, labels in dl:
            optimizer.zero_grad()
            logits = modelo(v_vis.to(device), v_geom.to(device), attn_mask.to(device))
            loss = criterion(logits, labels.to(device))
            loss.backward()
            optimizer.step()

    destino = ruta_produccion()
    destino.parent.mkdir(parents=True, exist_ok=True)
    torch.save(modelo.state_dict(), destino)

    procedencia = {
        "modelo": "MILFusionModel (Multimodal) sobre secuencia entera",
        "clips": len(clips),
        "ejemplares": len({c["id_caballo"] for c in clips}),
        "clases": ds.clases,
        "semilla": SEMILLA,
        "huella_caracteristicas": _huella_caracteristicas(),
        "metrica_validacion_cruzada": metrica_validacion,
        "advertencia": (
            "La métrica de este modelo es la de validación cruzada, no una calculada sobre el "
            "conjunto con el que se entrenó. No alcanza el criterio declarado de "
            f"{CRITERIO_ACEPTACION}."),
    }
    (destino.parent / ARCHIVO_PROCEDENCIA).write_text(
        json.dumps(procedencia, ensure_ascii=False, indent=1), encoding="utf-8")

    if verbose:
        print(f"Modelo de producción entrenado sobre {len(clips)} clips "
              f"({procedencia['ejemplares']} ejemplares)")
        print(f"  pesos      : {destino}")
        print(f"  procedencia: {destino.parent / ARCHIVO_PROCEDENCIA}")
    return modelo, procedencia


def _huella_caracteristicas():
    """De qué corrida del preprocesamiento salen las características que entrenaron el modelo."""
    import json

    ruta = rutas.carpeta("resultados") / "caracteristicas_backbone.json"
    if not ruta.exists():
        return None
    return json.loads(ruta.read_text(encoding="utf-8")).get("huella")


def cargar_produccion():
    """Carga el modelo de producción para inferir. Devuelve `(modelo, clases)`."""
    import torch
    
    ruta = ruta_produccion()
    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe {ruta}. Entrenarlo con `clasificacion.entrenar_produccion()`. "
            "Los pesos por fold NO se despliegan: cada uno vio sólo 3 de 4 partes del material.")
            
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo = MILFusionModel().to(device)
    modelo.load_state_dict(torch.load(ruta, map_location=device, weights_only=True))
    modelo.eval()
    
    return modelo, CLASES
