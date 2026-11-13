"""Normalización de texto: del comentario tal como se escribió a su "forma canónica".

La idea central es barata y no usa IA: en una conversación pública masiva la misma idea se repite
miles de veces con variaciones superficiales (mayúsculas, tildes, errores de tipeo, abreviaturas,
muletillas de apertura y cierre, menciones). Si se quitan esas variaciones, 200.000 comentarios se
reducen a unos pocos cientos de formas distintas, y todo el trabajo caro posterior se hace sobre
esas formas y no sobre cada comentario.

Nada de lo que hay acá depende del tema: las muletillas y los errores de tipeo se aprenden de los
propios datos.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict

TOKEN_RE = re.compile(r"#?[a-zñ0-9]+|@")
MENTION_RE = re.compile(r"@[\w.]+")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
REPEAT_RE = re.compile(r"(.)\1{2,}")
DOUBLE_RE = re.compile(r"(.)\1")
THOUSANDS_RE = re.compile(r"(?<=\d)[.,](?=\d{3}\b)")


def tokens(text: str) -> tuple:
    """Minúsculas, sin tildes (conserva la ñ), menciones como '@', letras repetidas colapsadas."""
    if not isinstance(text, str):
        return ()
    t = unicodedata.normalize("NFD", text.lower().replace("ñ", "\0"))
    t = "".join(ch for ch in t if unicodedata.category(ch) != "Mn").replace("\0", "ñ")
    t = URL_RE.sub(" ", t)
    t = THOUSANDS_RE.sub("", t)
    t = MENTION_RE.sub(" @ ", t)
    t = REPEAT_RE.sub(r"\1", t)
    return tuple(TOKEN_RE.findall(t))


def _dl1(a: str, b: str) -> bool:
    """True si la distancia de Damerau-Levenshtein entre a y b es <= 1."""
    if a == b:
        return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la > lb:
        a, b, la, lb = b, a, lb, la
    i = 0
    while i < la and a[i] == b[i]:
        i += 1
    if la == lb:
        if a[i + 1:] == b[i + 1:]:
            return True
        return i + 1 < la and a[i] == b[i + 1] and a[i + 1] == b[i] and a[i + 2:] == b[i + 2:]
    return a[i:] == b[i + 1:]


def _typo_kind(w: str, v: str) -> int:
    """0 = letra duplicada o transposición, 1 = inserción/borrado, 2 = sustitución, 9 = no es typo."""
    if DOUBLE_RE.sub(r"\1", w) == v:
        return 0
    if not _dl1(w, v):
        return 9
    if len(w) != len(v):
        return 0 if DOUBLE_RE.sub(r"\1", w) == DOUBLE_RE.sub(r"\1", v) else 1
    diff = [i for i in range(len(w)) if w[i] != v[i]]
    return 0 if len(diff) == 2 else 2


def typo_map(vocab: Counter, ratio: int = 15, min_len: int = 4) -> dict:
    """Mapea cada palabra rara a una palabra a distancia de edición 1 que sea `ratio` veces más frecuente.

    Se resuelve con el truco de "borrar una letra" (SymSpell): lineal en el tamaño del vocabulario.
    """
    by_del = defaultdict(set)
    for w in vocab:
        if len(w) < min_len - 1 or w[0] in "#@":
            continue
        by_del[w].add(w)
        for i in range(len(w)):
            by_del[w[:i] + w[i + 1:]].add(w)
    out = {}
    for w, c in vocab.items():
        if w[0] in "#@" or w.isdigit():
            continue
        cands = {DOUBLE_RE.sub(r"\1", w)}
        if len(w) >= min_len:
            cands |= by_del.get(w, set())
            for i in range(len(w)):
                cands |= by_del.get(w[:i] + w[i + 1:], set())
        best = None
        for v in cands:
            cv = vocab.get(v, 0)
            if v == w or cv < c * ratio:
                continue
            kind = _typo_kind(w, v)
            if kind == 9 or (kind == 2 and cv < c * ratio * 4):
                continue
            key = (kind, -cv, v)
            if best is None or key < best:
                best = key
        if best:
            out[w] = best[2]
    for w in list(out):  # cadenas a -> b -> c
        seen = {w}
        v = out[w]
        while v in out and out[v] not in seen:
            seen.add(v)
            v = out[v]
        out[w] = v
    return out


def _learn_side(universe: set, side: str, max_k: int, min_distinct: int) -> set:
    found = defaultdict(set)
    for t in universe:
        n = len(t)
        for k in range(1, min(max_k, n - 1) + 1):
            if side == "pre":
                if t[k:] in universe:
                    found[t[:k]].add(t[k:])
            elif t[:-k] in universe:
                found[t[-k:]].add(t[:-k])
    return {g for g, r in found.items() if len(r) >= min_distinct}


def _strip_side(t: tuple, grams: set, side: str, max_k: int = 6, min_core: int = 2) -> tuple:
    while True:
        for k in range(min(max_k, len(t) - min_core), 0, -1):
            g = t[:k] if side == "pre" else t[-k:]
            if g in grams:
                t = t[k:] if side == "pre" else t[:-k]
                break
        else:
            return t


def strip_wrappers(texts: list, max_k: int = 6, min_distinct: int = 12, rounds: int = 4):
    """Aprende y quita muletillas de apertura y cierre ("la verdad,", "a ver,", "che", "posta"...).

    Una secuencia de hasta `max_k` palabras es muletilla si envuelve a `min_distinct` textos distintos
    que además aparecen solos. No hay lista fija: sale de los datos.
    """
    prefixes, suffixes = set(), set()
    cur = list(texts)
    for _ in range(rounds):
        new_p = _learn_side(set(cur), "pre", max_k, min_distinct) - prefixes
        prefixes |= new_p
        cur = [_strip_side(t, prefixes, "pre", max_k) for t in cur]
        new_s = _learn_side(set(cur), "suf", max_k, min_distinct) - suffixes
        suffixes |= new_s
        cur = [_strip_side(t, suffixes, "suf", max_k) for t in cur]
        if not new_p and not new_s:
            break
    return cur, prefixes, suffixes


def canonical_forms(texts) -> tuple:
    """Devuelve (formas canónicas como string, info) para una lista de textos crudos."""
    toks = [tokens(x) for x in texts]
    vocab = Counter(w for t in toks for w in t)
    typos = typo_map(vocab)
    toks = [tuple(typos.get(w, w) for w in t) for t in toks]
    cores, prefixes, suffixes = strip_wrappers(toks)
    info = {"vocabulario": len(vocab), "typos_corregidos": len(typos),
            "muletillas_apertura": len(prefixes), "muletillas_cierre": len(suffixes),
            "formas_canonicas": len(set(cores))}
    return [" ".join(t) for t in cores], info
