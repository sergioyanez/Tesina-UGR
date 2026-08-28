"""Cabezas temporales sobre las características congeladas, y la prueba de la hipótesis.

La hipótesis del trabajo dice que la expresión es un evento dinámico y que la **evolución
temporal** es lo que permite clasificarla. Para ponerla a prueba de verdad hay que separar dos
cosas que el diseño mencionaba juntas:

  - **Cabezas invariantes al orden.** El promedio de características lo es: barajar los 16 frames
    no cambia una sola predicción. Es una cabeza a nivel de clip, no un modelo temporal. Si gana,
    la hipótesis NO queda sostenida — sería la misma conclusión que si ganara la línea de base de
    un frame, sólo que disfrazada de modelo secuencial.
  - **Cabezas sensibles al orden.** La GRU y la atención con posición sí ven la secuencia.

La prueba de la hipótesis es entonces: **¿le gana una cabeza sensible al orden a una invariante,
sobre los mismos folds?** Esa comparación es más limpia que compararla sólo contra la línea de
base, porque las tres cabezas comparten backbone, proyección, presupuesto de parámetros y régimen
de entrenamiento. Lo único que cambia es cómo agregan el tiempo.

**Sin early stopping sobre el fold de test.** Elegir la época mirando el conjunto sobre el que se
va a reportar es fuga, y con 35 clips por fold la métrica se puede subir varios puntos así. El
presupuesto de épocas se fija de antemano, igual para todas las cabezas, y no se toca.

Uso:
    python3 openspec/changes/clasificacion-orientada-a-video/cabezas_temporales.py
"""

import statistics
import sys
from collections import Counter
from pathlib import Path

def _raiz_del_repo():
    """Sube hasta encontrar el paquete `equinos`.

    No se usa una profundidad fija porque este script se mueve: al archivar el change pasa de
    `openspec/changes/<n>/` a `openspec/changes/archive/<fecha>-<n>/`, un nivel más abajo.
    """
    for candidata in Path(__file__).resolve().parents:
        if (candidata / "equinos" / "__init__.py").exists():
            return candidata
    raise RuntimeError("No se encontró la raíz del repositorio desde " + __file__)


sys.path.insert(0, str(_raiz_del_repo()))

from equinos import caracteristicas, particion, rutas  # noqa: E402

SEMILLA = 42
CLASES = ("Disgustado", "Interesado", "Sereno")

# Presupuesto FIJO, declarado antes de correr, igual para las tres cabezas y para los dos
# regímenes. No se ajusta por fold ni por cabeza: sin un conjunto de validación interno,
# ajustarlo sería ajustarlo contra el test.
#
# Se cuentan PASOS DE GRADIENTE y no épocas a propósito. Con aumentación el conjunto de
# entrenamiento es 20 veces más grande, así que "la misma cantidad de épocas" daría 20 veces más
# pasos y la comparación mediría dos cosas mezcladas. Con pasos fijos, lo único que cambia entre
# regímenes es la DIVERSIDAD de los datos, que es lo que se quiere medir.
PASOS = 400
LOTE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-2
DROPOUT = 0.3

# Todas las cabezas comparten esta proyección: 512 -> 64. Así el presupuesto de parámetros es
# comparable y lo único que cambia entre variantes es cómo se agrega el tiempo.
PROYECCION = 64

# Largo de la sub-secuencia. De los 16 frames de un clip salen 16 - VENTANA + 1 ventanas
# contiguas, así que el modelo temporal pasa de 106 muestras de entrenamiento a ~954, comparable
# a las 1696 frames sueltos que ve la línea de base. Sin esto, comparar la cabeza de clip contra
# esa línea de base mide el TAMAÑO del conjunto y no la información temporal.
VENTANA = 8


def _construir(nombre, dispositivo, largo=16):
    import torch
    from torch import nn

    class Cabeza(nn.Module):
        def __init__(self, modo, largo=16):
            super().__init__()
            self.modo = modo
            self.proyeccion = nn.Linear(caracteristicas.DIMENSIONES, PROYECCION)
            self.dropout = nn.Dropout(DROPOUT)
            if modo == "promedio":
                self.salida = nn.Linear(PROYECCION, len(CLASES))
            elif modo == "gru":
                self.gru = nn.GRU(PROYECCION, 32, batch_first=True)
                self.salida = nn.Linear(32, len(CLASES))
            elif modo == "atencion":
                # Embedding de posición: sin él, la atención también sería invariante al orden.
                self.posicion = nn.Parameter(torch.zeros(largo, PROYECCION))
                nn.init.normal_(self.posicion, std=0.02)
                self.consulta = nn.Parameter(torch.zeros(PROYECCION))
                nn.init.normal_(self.consulta, std=0.02)
                self.salida = nn.Linear(PROYECCION, len(CLASES))
            else:
                raise ValueError(modo)

        def forward(self, x):                       # x: (B, 16, 512)
            z = torch.relu(self.proyeccion(x))      # (B, 16, 64)
            if self.modo == "promedio":
                h = z.mean(dim=1)
            elif self.modo == "gru":
                _, hn = self.gru(z)
                h = hn[-1]
            else:
                zp = z + self.posicion.unsqueeze(0)
                pesos = torch.softmax(zp @ self.consulta, dim=1)
                h = (zp * pesos.unsqueeze(-1)).sum(dim=1)
            return self.salida(self.dropout(h))

    return Cabeza(nombre, largo).to(dispositivo)


def _metricas(reales, predichas):
    f1s, recalls = [], []
    for i, _ in enumerate(CLASES):
        vp = sum(1 for r, p in zip(reales, predichas) if r == i and p == i)
        fp = sum(1 for r, p in zip(reales, predichas) if r != i and p == i)
        fn = sum(1 for r, p in zip(reales, predichas) if r == i and p != i)
        pr = vp / (vp + fp) if (vp + fp) else 0.0
        rc = vp / (vp + fn) if (vp + fn) else 0.0
        f1s.append(2 * pr * rc / (pr + rc) if (pr + rc) else 0.0)
        if (vp + fn):
            recalls.append(rc)
    return {"f1_macro": sum(f1s) / len(f1s),
            "balanced_accuracy": sum(recalls) / len(recalls) if recalls else float("nan"),
            "recall_disgustado": f1s and _recall(reales, predichas, 0)}


def _recall(reales, predichas, clase):
    presentes = [p for r, p in zip(reales, predichas) if r == clase]
    return sum(1 for p in presentes if p == clase) / len(presentes) if presentes else float("nan")


def entrenar_y_evaluar(nombre, X_ent, y_ent, X_pru, y_pru, dispositivo, semilla=SEMILLA):
    import torch
    from torch import nn

    torch.manual_seed(semilla)
    modelo = _construir(nombre, dispositivo, X_ent.shape[1])
    entrenables = sum(p.numel() for p in modelo.parameters() if p.requires_grad)

    conteo = Counter(int(v) for v in y_ent)
    pesos = torch.tensor([len(y_ent) / (len(CLASES) * max(conteo[i], 1)) for i in range(len(CLASES))],
                         dtype=torch.float32, device=dispositivo)
    criterio = nn.CrossEntropyLoss(weight=pesos)
    optimizador = torch.optim.AdamW(modelo.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    generador = torch.Generator(device="cpu").manual_seed(semilla)

    modelo.train()
    n = len(y_ent)
    for _ in range(PASOS):
        idx = torch.randint(0, n, (min(LOTE, n),), generator=generador).to(dispositivo)
        optimizador.zero_grad()
        criterio(modelo(X_ent[idx]), y_ent[idx]).backward()
        optimizador.step()

    modelo.eval()
    with torch.no_grad():
        predichas = modelo(X_pru).argmax(dim=1).cpu().tolist()
    return _metricas(y_pru.cpu().tolist(), predichas), entrenables, predichas


def preparar(clips, vectores, indice, dispositivo, aumentadas=None):
    """Arma el tensor de entrada. Con `aumentadas`, apila las K vistas de cada clip.

    La aumentación se aplica SÓLO al conjunto de entrenamiento. El de test se evalúa siempre
    sobre la vista limpia: evaluar sobre imágenes transformadas mediría el modelo en una
    distribución que no es la del problema.
    """
    import numpy as np
    import torch

    if aumentadas is None:
        X = np.stack([vectores[indice[c["clip"]]] for c in clips])
        y = np.array([CLASES.index(c["clase"]) for c in clips])
    else:
        bloques, etiquetas = [], []
        for c in clips:
            vistas = aumentadas[indice[c["clip"]]]          # (K, 16, 512)
            bloques.append(vistas)
            etiquetas.extend([CLASES.index(c["clase"])] * len(vistas))
        X = np.concatenate(bloques)
        y = np.array(etiquetas)
    return (torch.tensor(X, dtype=torch.float32, device=dispositivo),
            torch.tensor(y, dtype=torch.long, device=dispositivo))


def ventanas(X, largo=VENTANA, permutar=False, generador=None):
    """Parte cada secuencia de 16 en las ventanas contiguas de `largo`.

    Con `permutar`, baraja los frames DENTRO de cada ventana. Es el control que aísla el orden:
    misma arquitectura, mismas muestras, mismo presupuesto — lo único que se destruye es la
    secuencia. Si el orden real no le gana al permutado, el orden no está aportando.
    """
    import torch

    n, t, d = X.shape
    trozos = [X[:, i:i + largo] for i in range(t - largo + 1)]
    salida = torch.cat(trozos, dim=0)                       # (n * ventanas, largo, d)
    if permutar:
        # Una permutación distinta por ventana: destruye el orden sin tocar el contenido.
        for i in range(salida.shape[0]):
            orden_i = torch.randperm(largo, generator=generador).to(salida.device)
            salida[i] = salida[i][orden_i]
    return salida, len(trozos)


def comparar(cabezas=("promedio", "gru", "atencion"), aumentar=False):
    import torch

    rutas.inicializar(verbose=False)
    clips = particion.cargar()
    orden, vectores = caracteristicas.cargar()
    indice = {c: i for i, c in enumerate(orden)}
    vistas = None
    if aumentar:
        orden_a, vistas = caracteristicas.cargar_aumentadas()
        assert orden_a == orden, "el orden de clips de las vistas aumentadas no coincide"
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folds = max(c["fold"] for c in clips) + 1

    print("=" * 74)
    print("CABEZAS TEMPORALES SOBRE LAS CARACTERÍSTICAS CONGELADAS")
    print("=" * 74)
    regimen = f"CON aumentación ({vistas.shape[1]} vistas por clip)" if aumentar else "SIN aumentación"
    print(f"{len(clips)} clips · {folds} folds · {PASOS} pasos fijos · lote {LOTE} · sin early stopping")
    print(f"Régimen: {regimen}")
    print(f"Dispositivo: {dispositivo}\n")

    resultados = {}
    for nombre in cabezas:
        filas = []
        for f in range(folds):
            ent = [c for c in clips if c["fold"] != f]
            pru = [c for c in clips if c["fold"] == f]
            X_e, y_e = preparar(ent, vectores, indice, dispositivo, vistas)
            X_p, y_p = preparar(pru, vectores, indice, dispositivo)
            m, entrenables, _ = entrenar_y_evaluar(nombre, X_e, y_e, X_p, y_p, dispositivo)
            filas.append(m)
        f1 = [r["f1_macro"] for r in filas]
        ba = [r["balanced_accuracy"] for r in filas]
        rd = [r["recall_disgustado"] for r in filas if r["recall_disgustado"] == r["recall_disgustado"]]
        resultados[nombre] = {
            "f1_macro": (statistics.fmean(f1), statistics.stdev(f1)),
            "balanced_accuracy": (statistics.fmean(ba), statistics.stdev(ba)),
            "recall_disgustado": (statistics.fmean(rd), statistics.stdev(rd)) if rd else (float("nan"),) * 2,
            "entrenables": entrenables,
            "por_fold": f1,
        }

    orden_inv = {"promedio": "invariante al orden", "gru": "sensible al orden",
                 "atencion": "sensible al orden"}
    print(f'{"cabeza":<12}{"":<22}{"entrenables":>12}{"F1 macro":>18}{"bal. acc":>16}{"recall Disg.":>16}')
    print("-" * 96)
    for nombre in cabezas:
        r = resultados[nombre]
        print(f'{nombre:<12}{orden_inv[nombre]:<22}{r["entrenables"]:>12,}'
              f'{r["f1_macro"][0]:>11.3f} ± {r["f1_macro"][1]:.3f}'
              f'{r["balanced_accuracy"][0]:>9.3f} ± {r["balanced_accuracy"][1]:.3f}'
              f'{r["recall_disgustado"][0]:>9.3f} ± {r["recall_disgustado"][1]:.3f}')
    return resultados


if __name__ == "__main__":
    res = comparar()
    print()
    invariante = res["promedio"]["f1_macro"][0]
    mejor_secuencial = max(res["gru"]["f1_macro"][0], res["atencion"]["f1_macro"][0])
    print("=" * 74)
    print(f"Mejor cabeza SENSIBLE al orden : {mejor_secuencial:.3f}")
    print(f"Cabeza INVARIANTE al orden     : {invariante:.3f}")
    print(f"Diferencia                     : {mejor_secuencial - invariante:+.3f}")
    print()
    print("Línea de base de un frame (0,626) y criterio declarado (0,691) en lineas_de_base.md.")


def prueba_del_orden(cabeza="atencion", largo=VENTANA):
    """El experimento que aísla el orden: sub-secuencias reales contra sub-secuencias barajadas.

    Todo lo demás es idéntico —arquitectura, cantidad de muestras, pasos, semilla—, así que la
    diferencia entre las dos columnas es atribuible ÚNICAMENTE a la información de orden.

    En test se promedian las probabilidades de las ventanas de cada clip, igual que la línea de
    base promedia el voto de sus frames.
    """
    import torch

    rutas.inicializar(verbose=False)
    clips = particion.cargar()
    orden, vectores = caracteristicas.cargar()
    indice = {c: i for i, c in enumerate(orden)}
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folds = max(c["fold"] for c in clips) + 1

    print("=" * 74)
    print(f"PRUEBA DEL ORDEN — cabeza '{cabeza}', ventanas de {largo} frames")
    print("=" * 74)

    resultados = {}
    for permutar in (False, True):
        filas = []
        for f in range(folds):
            ent = [c for c in clips if c["fold"] != f]
            pru = [c for c in clips if c["fold"] == f]
            X_e, y_e = preparar(ent, vectores, indice, dispositivo)
            X_p, y_p = preparar(pru, vectores, indice, dispositivo)
            g = torch.Generator().manual_seed(SEMILLA)
            Xe_v, k = ventanas(X_e, largo, permutar, g)
            Xp_v, _ = ventanas(X_p, largo, permutar, g)
            ye_v = y_e.repeat(k)

            torch.manual_seed(SEMILLA)
            from torch import nn
            modelo = _construir(cabeza, dispositivo, largo)
            conteo = Counter(int(v) for v in ye_v)
            pesos = torch.tensor(
                [len(ye_v) / (len(CLASES) * max(conteo[i], 1)) for i in range(len(CLASES))],
                dtype=torch.float32, device=dispositivo)
            criterio = nn.CrossEntropyLoss(weight=pesos)
            opt = torch.optim.AdamW(modelo.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
            gen = torch.Generator(device="cpu").manual_seed(SEMILLA)
            modelo.train()
            for _ in range(PASOS):
                idx = torch.randint(0, len(ye_v), (LOTE,), generator=gen).to(dispositivo)
                opt.zero_grad()
                criterio(modelo(Xe_v[idx]), ye_v[idx]).backward()
                opt.step()

            # Promedio de probabilidades entre las ventanas de cada clip de test.
            modelo.eval()
            with torch.no_grad():
                probas = torch.softmax(modelo(Xp_v), dim=1).view(k, len(pru), len(CLASES))
                predichas = probas.mean(dim=0).argmax(dim=1).cpu().tolist()
            filas.append(_metricas(y_p.cpu().tolist(), predichas))

        f1 = [r["f1_macro"] for r in filas]
        rd = [r["recall_disgustado"] for r in filas]
        resultados["permutado" if permutar else "orden real"] = {
            "f1": (statistics.fmean(f1), statistics.stdev(f1)),
            "rd": (statistics.fmean(rd), statistics.stdev(rd)),
            "muestras": len(ye_v),
        }

    print(f'{"régimen":<16}{"muestras ent.":>15}{"F1 macro":>18}{"recall Disgustado":>22}')
    print("-" * 74)
    for k_, v in resultados.items():
        print(f'{k_:<16}{v["muestras"]:>15,}{v["f1"][0]:>11.3f} ± {v["f1"][1]:.3f}'
              f'{v["rd"][0]:>15.3f} ± {v["rd"][1]:.3f}')
    real = resultados["orden real"]["f1"]
    perm = resultados["permutado"]["f1"]
    print(f'\nDiferencia orden real - permutado: {real[0] - perm[0]:+.3f}  '
          f'(desvío entre folds: {real[1]:.3f})')
    return resultados
