"""Tarea A: descubrir narrativas, separar ruido, asignar postura y fechar el origen.

No se parte de una lista de temas ni de un número de clusters. La estructura sale de la propia
conversación:

1. Cada comentario se reduce a su forma canónica (`texto.py`).
2. Las formas raras se pegan a la forma frecuente más parecida (n-gramas de caracteres).
3. Las formas que casi siempre aparecen como respuesta son "acompañamientos" (exacto, tal cual...):
   no dicen nada por sí solas y heredan narrativa y postura del comentario al que responden.
4. Con el resto se arma un grafo: dos formas quedan unidas cuando una se usa para responderle a la
   otra. La gente discute dentro de un mismo asunto, así que las componentes del grafo son las
   narrativas, y lo que nunca entra en discusión (saludos, consultas, spam) es ruido.
5. Una respuesta con contenido propio casi siempre contradice a su padre, así que cada narrativa
   se parte en dos bandos (bipartición del grafo). Falta decidir cuál bando afirma la narrativa y
   cuál la discute: eso lo hace `orientar`.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from .texto import canonical_forms

MIN_FREQ = 8            # una forma es "frecuente" si aparece al menos esta cantidad de veces
SIM_ATTACH = 0.72       # similitud mínima para pegar una forma rara a una frecuente
ACK_REPLY_SHARE = 0.8   # proporción de usos como respuesta para considerar "acompañamiento"
MIN_PARTICIPATION = 0.04  # proporción mínima de usos dentro de una discusión para entrar al grafo
MIN_NARRATIVE_SHARE = 0.002  # una narrativa tiene al menos este porcentaje de los comentarios
SHORT_REPLY_TOKENS = 3  # una respuesta desconocida de hasta N palabras se trata como acompañamiento

STOPWORDS = set("""a al algo algun alguna algunas alguno algunos ante antes aca ahi ahora asi aun aunque bien cada casi como con
contra cual cuando de del desde donde dos el ella ellas ellos en entre era eran es esa esas ese eso esos esta estaba estan
estas este esto estos fue fueron ha hace hacen hacer han hasta hay igual la las le les lo los mas me mi mis mucho muy nada
ni no nos nosotros o os otra otras otro otros para pero poco por porque que quien quienes se sea ser si sin sobre solo son
su sus tambien tan tanto te tiene tienen todo todos toda todas tu tus un una unas uno unos va van vez ya yo q x pa xq tmb
d k tdo eso esa les nos del al y e u mas menos despues siempre nunca aca alla cuando donde dice dicen decir dijo vamos voy
""".split())


class _ParityDSU:
    """Union-find con paridad: mantiene en qué bando queda cada nodo respecto de su raíz."""

    def __init__(self):
        self.parent, self.parity, self.size = {}, {}, {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x], self.parity[x], self.size[x] = x, 0, 1
            return x, 0
        path = []
        while self.parent[x] != x:
            path.append(x)
            x = self.parent[x]
        acc = 0
        for node in reversed(path):
            acc ^= self.parity[node]
            self.parent[node], self.parity[node] = x, acc
        return x, (self.parity[path[0]] if path else 0)

    def union_opposite(self, a, b) -> bool:
        ra, pa = self.find(a)
        rb, pb = self.find(b)
        if ra == rb:
            return pa != pb
        if self.size[ra] < self.size[rb]:
            ra, rb, pa, pb = rb, ra, pb, pa
        self.parent[rb] = ra
        self.parity[rb] = pa ^ pb ^ 1
        self.size[ra] += self.size[rb]
        return True


def _attach_rare(counts: pd.Series) -> dict:
    """Pega cada forma rara a la forma frecuente más parecida. Devuelve {forma_rara: forma_frecuente}."""
    frequent = [c for c, n in counts.items() if n >= MIN_FREQ and c]
    rare = [c for c, n in counts.items() if n < MIN_FREQ and c]
    if not frequent or not rare:
        return {}
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True, dtype=np.float32)
    xf = vec.fit_transform(frequent)
    xr = vec.transform(rare)
    out = {}
    for start in range(0, len(rare), 2000):
        sims = (xr[start:start + 2000] @ xf.T).toarray()
        best = sims.argmax(axis=1)
        for i, j in enumerate(best):
            if sims[i, j] >= SIM_ATTACH:
                out[rare[start + i]] = frequent[j]
    return out


def construir_formas(com: pd.DataFrame) -> tuple:
    """Agrega la columna `forma` (forma canónica con las raras ya pegadas a su vecina frecuente)."""
    cores, info = canonical_forms(com["texto"].tolist())
    counts = pd.Series(cores).value_counts()
    alias = _attach_rare(counts)
    info["formas_raras_pegadas"] = len(alias)
    formas = [alias.get(c, c) for c in cores]
    info["formas_finales"] = len(set(formas))
    return formas, info


def descubrir(com: pd.DataFrame) -> dict:
    """com: comentario_id, publicacion_id, cuenta_id, ts (datetime), texto, respuesta_a, forma.

    Devuelve un dict con: `tipo` por forma, `narrativa` y `bando` por forma, y columnas por comentario.
    """
    forma = com["forma"].to_numpy()
    ids = com["comentario_id"].to_numpy()
    parent_id = com["respuesta_a"].to_numpy()
    pos = {cid: i for i, cid in enumerate(ids)}
    parent = np.array([pos.get(p, -1) if p else -1 for p in parent_id])
    is_reply = parent >= 0
    counts = Counter(forma)
    reply_counts = Counter(forma[is_reply])
    ntok = {f: len(f.split()) for f in counts}

    acks = {f for f, n in counts.items()
            if n >= MIN_FREQ and reply_counts[f] / n >= ACK_REPLY_SHARE}

    # Padre efectivo: el primer ancestro que no es un acompañamiento.
    def effective_parent(i):
        j = parent[i]
        hops = 0
        while j >= 0 and forma[j] in acks and hops < 8:
            j = parent[j]
            hops += 1
        return j

    edges = Counter()
    for i in np.nonzero(is_reply)[0]:
        f = forma[i]
        if f in acks or counts[f] < MIN_FREQ:
            continue
        j = effective_parent(i)
        if j < 0:
            continue
        g = forma[j]
        if g in acks or counts[g] < MIN_FREQ or g == f:
            continue
        edges[(f, g) if f < g else (g, f)] += 1

    touched = Counter()
    for (a, b), w in edges.items():
        touched[a] += w
        touched[b] += w
    in_graph = {f for f, w in touched.items() if w / counts[f] >= MIN_PARTICIPATION and w >= 2}
    edges = {e: w for e, w in edges.items() if e[0] in in_graph and e[1] in in_graph}

    # Componentes: primero el esqueleto de aristas repetidas, después se cuelgan las de peso 1.
    dsu = _ParityDSU()
    ordered = sorted(edges.items(), key=lambda kv: (-kv[1], kv[0]))
    conflicts = 0
    for (a, b), w in ordered:
        if w >= 2:
            conflicts += not dsu.union_opposite(a, b)
    for (a, b), w in ordered:
        if w == 1:
            in_a, in_b = a in dsu.parent, b in dsu.parent
            if in_a and in_b and dsu.find(a)[0] != dsu.find(b)[0]:
                continue  # una sola respuesta no alcanza para fusionar dos narrativas
            if not in_a and not in_b:
                continue
            conflicts += not dsu.union_opposite(a, b)
    for (a, b), w in ordered:  # pares sueltos unidos solo por aristas de peso 1
        if w == 1 and a not in dsu.parent and b not in dsu.parent:
            dsu.union_opposite(a, b)

    comp_of, side_of = {}, {}
    for f in list(dsu.parent):
        root, par = dsu.find(f)
        comp_of[f], side_of[f] = root, par
    comp_size = Counter()
    for f, r in comp_of.items():
        comp_size[r] += counts[f]
    min_size = max(30, MIN_NARRATIVE_SHARE * len(com))
    keep = {r for r, n in comp_size.items() if n >= min_size}
    comp_of = {f: r for f, r in comp_of.items() if r in keep}
    side_of = {f: s for f, s in side_of.items() if f in comp_of}

    # Formas que nadie discutió pero que son casi iguales a una que sí (variantes de campaña, typos).
    loose = [f for f in counts if f not in comp_of and f not in acks and ntok[f] >= 3]
    graph_forms = sorted(comp_of)
    if loose and graph_forms:
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True, dtype=np.float32)
        xg = vec.fit_transform(graph_forms)
        sims = (vec.transform(loose) @ xg.T).toarray()
        for i, j in enumerate(sims.argmax(axis=1)):
            if sims[i, j] >= SIM_ATTACH:
                comp_of[loose[i]] = comp_of[graph_forms[j]]
                side_of[loose[i]] = side_of[graph_forms[j]]

    # Asignación por comentario, padres antes que hijos.
    n = len(com)
    narr = np.full(n, "", dtype=object)
    side = np.full(n, -1)
    how = np.full(n, "ruido", dtype=object)
    depth = np.zeros(n, dtype=int)
    for i in np.nonzero(is_reply)[0]:
        d, j = 0, parent[i]
        while j >= 0 and d < 10:
            d += 1
            j = parent[j]
        depth[i] = d
    for i in np.argsort(depth, kind="stable"):
        f = forma[i]
        if f in comp_of:
            narr[i], side[i], how[i] = comp_of[f], side_of[f], "forma"
            continue
        j = parent[i]
        if j < 0 or not narr[j]:
            continue
        if f in acks or ntok[f] <= SHORT_REPLY_TOKENS:
            narr[i], side[i], how[i] = narr[j], side[j], "acompaña"
        elif counts[f] < MIN_FREQ:
            narr[i], side[i], how[i] = narr[j], 1 - side[j], "responde"
    return {"narr": narr, "side": side, "how": how, "acks": acks, "comp_of": comp_of, "side_of": side_of,
            "edges": edges, "counts": counts, "conflictos_biparticion": conflicts}


# --- Postura: cuál de los dos bandos afirma la narrativa y cuál la discute -----------------------

# Marcas de réplica: quien discute una afirmación suele hablarle al que la sostiene ("dejen de",
# "lean", "muestren pruebas"), nombrarla de costado ("lo de", "eso del") o desmentirla ("es falso",
# "nada que ver", "no es"). Son marcas del español conversacional, no de un tema.
MARCAS_REPLICA = [
    r"\bdejen de\b", r"\bbasta de\b", r"\bnada que ver\b", r"\botra vez con\b", r"\b(lo|eso) (de|del)\b", r"\bno es\b",
    r"\b(falso|falsa|mentira|mentiras|invento|inventos|inventan|inventar|inventaron|verso|trucha|trucho|fake|rumor|rumores"
    r"|desmintio|desmintieron|aclaro|aclararon|exageran|exageren|exagerar|exagerado|difamar|estigmatizar|pruebas|humo)\b",
    r"\b(lean|informense|informarse|averiguen|miren|fijense|hablen|hablan|opinan|critican|criticando|criticar"
    r"|comparan|culpan|mezclan|mezclar|repiten|comparten|compartir|pasen)\b",
    r"\btambien\b", r"\bpero\b", r"\btampoco\b", r"\bcomparado\b", r"\bdepende\b", r"\bobvio\b",
]
_MARCAS_RE = None


def _es_replica(forma: str) -> bool:
    global _MARCAS_RE
    if _MARCAS_RE is None:
        import re
        _MARCAS_RE = [re.compile(p) for p in MARCAS_REPLICA]
    return any(r.search(forma) for r in _MARCAS_RE)


def orientar(com: pd.DataFrame, coordinadas: set | None = None) -> dict:
    """Decide, para cada narrativa, qué bando (0/1) es el que la rechaza.

    com: columnas narr, side, how, forma, cuenta_id. Combina tres evidencias independientes del tema:
    - réplica: el bando con más marcas de réplica es el que rechaza;
    - variedad: la afirmación circula con muchas más redacciones que su desmentida;
    - empuje coordinado: una operación infla la narrativa, no su desmentida;
    - mayoría: entre las cuentas genuinas, la afirmación junta algo más de voces que su desmentida.
    Devuelve {narrativa: {"rechaza": bando, "evidencia": {...}}}.
    """
    coordinadas = coordinadas or set()
    base = com[(com["narr"] != "") & (com["how"] == "forma")]
    replica = {f: _es_replica(f) for f in base["forma"].unique()}
    out = {}
    for narr, g in base.groupby("narr"):
        s0, s1 = g[g["side"] == 0], g[g["side"] == 1]
        if not len(s0) or not len(s1):
            lado = 1 if len(s0) else 0  # un solo bando: es el que afirma
            out[narr] = {"rechaza": lado, "evidencia": {"un_solo_bando": True}}
            continue
        r0 = s0["forma"].map(replica).mean()
        r1 = s1["forma"].map(replica).mean()
        f0, f1 = s0["forma"].nunique(), s1["forma"].nunique()
        c0 = int(s0["cuenta_id"].isin(coordinadas).sum())
        c1 = int(s1["cuenta_id"].isin(coordinadas).sum())
        e_replica = (r0 - r1) / 0.25
        e_variedad = math.log(f1 / f0) / 0.5
        e_coord = 3.0 * (c1 - c0) / (c1 + c0) * min(1.0, (c1 + c0) / 100) if c0 + c1 else 0.0
        org = com[(com["narr"] == narr) & ~com["cuenta_id"].isin(coordinadas)]
        e_mayoria = float((org["side"] == 1).mean() - (org["side"] == 0).mean()) / 0.05 if len(org) else 0.0
        score = float(e_replica + e_variedad + e_coord + e_mayoria)  # > 0: el bando 0 es el que rechaza
        out[narr] = {"rechaza": 0 if score > 0 else 1,
                     "evidencia": {"replica": round(float(e_replica), 2), "variedad": round(e_variedad, 2),
                                   "coordinacion": round(e_coord, 2), "mayoria": round(e_mayoria, 2),
                                   "puntaje": round(score, 2)}}
    return out


# --- Descripción legible de cada narrativa ------------------------------------------------------

def _texto_limpio(grupo: pd.DataFrame, forma: str) -> str:
    """El texto original más repetido de una forma, sin muletillas alrededor."""
    from .texto import tokens
    n = len(forma.split())
    cand = Counter(t.strip() for t in grupo["texto"] if len(tokens(t)) == n)
    if not cand:
        return forma.capitalize()
    best = max(cand.items(), key=lambda kv: (kv[1], sum(ch in "áéíóúñ" for ch in kv[0].lower()), kv[0][:1].isupper()))[0]
    best = "".join(ch for ch in best if ch.isalnum() or ch in " ,.;:¿?¡!%$#@'\"()-/").strip(" ,.;:")
    return best[:1].upper() + best[1:] if best else forma.capitalize()


def frases_clave(com: pd.DataFrame, top: int = 5) -> dict:
    """Términos y pares de palabras que más distinguen a cada narrativa del resto."""
    base = com[(com["narr"] != "") & (com["how"] == "forma")]
    conteo = base.groupby(["narr", "forma"]).size()
    por_narr = defaultdict(Counter)
    for (narr, forma), n in conteo.items():
        ws = [w for w in forma.split() if w != "@"]
        seen = set()
        for i, w in enumerate(ws):
            if w not in STOPWORDS and len(w) > 2:
                seen.add(w)
            if i + 1 < len(ws) and w not in STOPWORDS and ws[i + 1] not in STOPWORDS:
                seen.add(w + " " + ws[i + 1])
        for term in seen:
            por_narr[narr][term] += n
    df = Counter(t for c in por_narr.values() for t in c)
    total = len(por_narr)
    out = {}
    for narr, c in por_narr.items():
        size = int(conteo[narr].sum())
        scored = sorted(((n / size * math.log((total + 1) / df[t]) * (1.3 if " " in t else 1.0), t) for t, n in c.items()
                         if n / size >= 0.03), reverse=True)
        chosen = []
        for _, t in scored:
            if any(set(t.split()) & set(o.split()) for o in chosen):
                continue
            chosen.append(t)
            if len(chosen) == top:
                break
        out[narr] = chosen
    return out
