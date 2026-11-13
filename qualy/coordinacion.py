"""Tarea B: detectar cuentas coordinadas y agruparlas por operación.

Una ráfaga sola no prueba nada: en una publicación viral mucha gente comenta a la vez, y hay
frases que se copian sin que nadie las ordene. Lo que delata a una operación es la repetición:
las mismas cuentas empujando el mismo bando de la misma narrativa, juntas, una y otra vez.

1. Coincidencia: dos cuentas coinciden cuando comentan en la misma publicación, del mismo lado de
   la misma narrativa, con menos de `VENTANA_S` segundos de diferencia. Se cuenta en cuántas
   publicaciones distintas coincidió cada par.
2. Contra lo esperable: ese número se normaliza por la actividad de las dos cuentas (coseno). Dos
   cuentas muy activas se cruzan seguido por azar; dos cuentas que casi solo aparecen juntas, no.
3. Repetición en grupo: un par aislado puede ser casualidad. Se conserva el núcleo donde cada
   cuenta tiene vínculo fuerte con varias otras (k-core) y cada componente es una operación.
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
import pandas as pd

VENTANA_S = 1800       # ventana de coincidencia
MIN_COINCIDENCIAS = 2  # publicaciones distintas en las que coincidió el par
COSENO_FUERTE = 0.30   # vínculo con 2 coincidencias
COSENO_REPETIDO = 0.15  # vínculo con 3 o más coincidencias
K_CORE = 3             # vínculos fuertes que debe tener cada cuenta dentro del grupo
MIN_GRUPO = 5          # tamaño mínimo de una operación
MIN_EVENTOS = 3        # veces que el grupo tiene que haber actuado en conjunto
FRACCION_EVENTO = 0.2  # parte del grupo que tiene que estar presente para contar un evento
FIRMA_EXCLUSIVA = 0.6  # una forma es "firma" del grupo si esta proporción de sus usos es del grupo
FIRMA_MIN_USOS = 3
FIRMA_MIN_PALABRAS = 4


def coincidencias(com: pd.DataFrame, ventana_s: int | None = None) -> pd.DataFrame:
    """com: cuenta_id, publicacion_id, narr, side, ts (segundos). Solo comentarios con narrativa.

    Devuelve un DataFrame con a, b (cuentas), n (publicaciones en que coincidieron) y coseno.
    """
    ventana_s = VENTANA_S if ventana_s is None else ventana_s
    com = com[com["narr"] != ""]
    acc_codes, acc_names = pd.factorize(com["cuenta_id"])
    pub_codes, _ = pd.factorize(com["publicacion_id"])
    grp_codes, _ = pd.factorize(pd.Series(pub_codes).astype(str) + "|" + com["narr"].to_numpy() + "|" + com["side"].astype(str).to_numpy())
    ts = com["ts"].to_numpy(dtype=np.int64)
    order = np.lexsort((ts, grp_codes))
    g, t, a, p = grp_codes[order], ts[order], acc_codes[order], pub_codes[order]
    # Para cada comentario, el primer comentario anterior del mismo grupo dentro de la ventana.
    key = g.astype(np.int64) * (int(t.max()) + ventana_s + 1 - int(t.min())) + (t - t.min())
    lo = np.searchsorted(key, key - ventana_s, side="left")
    idx = np.arange(len(key))
    width = idx - lo
    n_acc = len(acc_names)
    chunks = []
    step = 200_000
    for start in range(0, len(key), step):
        w = width[start:start + step]
        i = np.repeat(idx[start:start + step], w)
        offs = np.arange(w.sum()) - np.repeat(np.cumsum(w) - w, w)
        j = np.repeat(lo[start:start + step], w) + offs
        x, y = a[i], a[j]
        keep = x != y
        x, y, pp = x[keep], y[keep], p[i][keep]
        pair = np.minimum(x, y).astype(np.int64) * n_acc + np.maximum(x, y)
        chunks.append(np.unique(np.stack([pair, pp.astype(np.int64)], axis=1), axis=0))
    if not chunks:
        return pd.DataFrame(columns=["a", "b", "n", "coseno"])
    pp = np.unique(np.concatenate(chunks), axis=0)
    pair, n = np.unique(pp[:, 0], return_counts=True)
    keep = n >= MIN_COINCIDENCIAS
    pair, n = pair[keep], n[keep]
    pubs_per_acc = pd.DataFrame({"a": acc_codes, "p": pub_codes}).drop_duplicates().groupby("a").size()
    activity = pubs_per_acc.reindex(range(n_acc)).to_numpy()
    xa, xb = pair // n_acc, pair % n_acc
    cos = n / np.sqrt(activity[xa] * activity[xb])
    return pd.DataFrame({"a": acc_names[xa], "b": acc_names[xb], "n": n, "coseno": cos})


def _k_core(adj: dict, k: int) -> set:
    alive = set(adj)
    changed = True
    while changed:
        changed = False
        for node in list(alive):
            if sum(1 for m in adj[node] if m in alive) < k:
                alive.discard(node)
                changed = True
    return alive


def detectar(com: pd.DataFrame) -> tuple:
    """com: comentario_id, cuenta_id, publicacion_id, narr, side, ts, forma. Devuelve grupos, vínculos, eventos y firmas."""
    pares = coincidencias(com)
    fuerte = pares[((pares["n"] >= 3) & (pares["coseno"] >= COSENO_REPETIDO)) | (pares["coseno"] >= COSENO_FUERTE)]
    adj = defaultdict(set)
    for a, b in zip(fuerte["a"], fuerte["b"]):
        adj[a].add(b)
        adj[b].add(a)
    core = _k_core(adj, K_CORE)
    # Componentes conexas del núcleo.
    comp, groups = {}, []
    for node in sorted(core):
        if node in comp:
            continue
        stack, members = [node], []
        comp[node] = len(groups)
        while stack:
            x = stack.pop()
            members.append(x)
            for m in adj[x]:
                if m in core and m not in comp:
                    comp[m] = len(groups)
                    stack.append(m)
        groups.append(members)
    # Cuentas periféricas: fuera del núcleo pero con al menos dos vínculos fuertes hacia un mismo grupo.
    for node in sorted(set(adj) - core):
        votes = Counter(comp[m] for m in adj[node] if m in comp)
        if votes and votes.most_common(1)[0][1] >= 2:
            groups[votes.most_common(1)[0][0]].append(node)
    groups = [g for g in groups if len(g) >= MIN_GRUPO]

    # Repetición: el grupo tiene que haber actuado junto varias veces. Dos coincidencias masivas
    # son lo que deja un par de publicaciones virales, no una operación.
    narr = com[com["narr"] != ""]
    validos, eventos = [], []
    for g in groups:
        sub = narr[narr["cuenta_id"].isin(set(g))]
        ev = (sub.groupby(["publicacion_id", "narr", "side"])
              .agg(cuentas=("cuenta_id", "nunique"), t0=("ts", "min"), t1=("ts", "max"), comentarios=("ts", "size"))
              .reset_index())
        ev = ev[ev["cuentas"] >= max(3, FRACCION_EVENTO * len(g))]
        if len(ev) >= MIN_EVENTOS:
            validos.append(g)
            eventos.append(ev)
    orden = sorted(range(len(validos)), key=lambda i: (-len(validos[i]), min(validos[i])))
    asignacion = {acc: f"op{k + 1}" for k, i in enumerate(orden) for acc in validos[i]}
    tabla_eventos = [eventos[i].assign(grupo=f"op{k + 1}") for k, i in enumerate(orden)]
    tabla_eventos = pd.concat(tabla_eventos, ignore_index=True) if tabla_eventos else pd.DataFrame(
        columns=["publicacion_id", "narr", "side", "cuentas", "t0", "t1", "comentarios", "grupo"])

    # Firma: textos que casi solo usa el grupo. Quien los repite, o los usa durante un evento del
    # grupo, es parte de la operación aunque haya participado poco.
    firmas = {}
    if asignacion:
        grupo_com = com["cuenta_id"].map(asignacion).fillna("")
        uso = pd.DataFrame({"forma": com["forma"].to_numpy(), "grupo": grupo_com.to_numpy()})
        total = uso.groupby("forma").size()
        propio = uso[uso["grupo"] != ""].groupby(["forma", "grupo"]).size().reset_index(name="n")
        propio["total"] = propio["forma"].map(total)
        propio = propio[(propio["n"] >= FIRMA_MIN_USOS) & (propio["n"] / propio["total"] >= FIRMA_EXCLUSIVA)
                        & (propio["forma"].str.split().str.len() >= FIRMA_MIN_PALABRAS)]
        firmas = dict(zip(propio["forma"], propio["grupo"]))
        fuera = com[(grupo_com == "").to_numpy() & com["forma"].isin(firmas).to_numpy()].copy()
        if len(fuera):
            fuera["grupo"] = fuera["forma"].map(firmas)
            fuera = fuera.merge(tabla_eventos[["publicacion_id", "grupo", "t0", "t1"]], on=["publicacion_id", "grupo"], how="left")
            fuera["en_evento"] = (fuera["ts"] >= fuera["t0"] - VENTANA_S) & (fuera["ts"] <= fuera["t1"] + VENTANA_S)
            res = fuera.groupby(["cuenta_id", "grupo"]).agg(usos=("comentario_id", "nunique"), en_evento=("en_evento", "any")).reset_index()
            res = res[(res["usos"] >= 2) | res["en_evento"]].sort_values("usos", ascending=False).drop_duplicates("cuenta_id")
            asignacion.update(dict(zip(res["cuenta_id"], res["grupo"])))
    fuerte = fuerte[fuerte["a"].isin(asignacion) & fuerte["b"].isin(asignacion)].reset_index(drop=True)
    return {"grupos": asignacion, "vinculos": fuerte, "eventos": tabla_eventos, "firmas": firmas, "pares_evaluados": len(pares)}
