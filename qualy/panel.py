"""Panel de demo: un único HTML autocontenido (sin servidor ni dependencias externas).

Responde en un minuto las tres preguntas del desafío: qué circula, desde cuándo y quién lo empuja.
Todo lo que muestra sale de las tablas que produjo el pipeline; no hay texto escrito a mano por set.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .narrativas import _texto_limpio

ZONA = "America/Argentina/Buenos_Aires"
CUBETA_H = 3
PLANTILLA = Path(__file__).with_name("panel_plantilla.html")


def _disposicion(nodos: list, aristas: list, iteraciones: int = 150, semilla: int = 7) -> dict:
    """Disposición de resortes (Fruchterman-Reingold) con semilla fija, para dibujar la red."""
    n = len(nodos)
    if n == 0:
        return {}
    rng = np.random.default_rng(semilla)
    pos = rng.uniform(-1, 1, size=(n, 2))
    idx = {x: i for i, x in enumerate(nodos)}
    e = np.array([(idx[a], idx[b]) for a, b in aristas if a in idx and b in idx], dtype=int).reshape(-1, 2)
    k = 1.6 / np.sqrt(n)
    temp = 0.2
    for _ in range(iteraciones):
        delta = pos[:, None, :] - pos[None, :, :]
        dist = np.maximum(np.linalg.norm(delta, axis=2), 1e-3)
        disp = (delta / dist[:, :, None] * (k * k / dist)[:, :, None]).sum(axis=1)
        if len(e):
            d = pos[e[:, 0]] - pos[e[:, 1]]
            ln = np.maximum(np.linalg.norm(d, axis=1), 1e-3)
            f = d / ln[:, None] * (ln * ln / k)[:, None]
            np.add.at(disp, e[:, 0], -f)
            np.add.at(disp, e[:, 1], f)
        disp -= pos * 0.3  # gravedad hacia el centro
        ln = np.maximum(np.linalg.norm(disp, axis=1), 1e-3)
        pos += disp / ln[:, None] * np.minimum(ln, temp)[:, None]
        temp *= 0.975
    pos -= pos.min(axis=0)
    pos /= np.maximum(pos.max(axis=0), 1e-9)
    return {x: [round(float(pos[i, 0]), 4), round(float(pos[i, 1]), 4)] for x, i in idx.items()}


def _frases(g: pd.DataFrame, top: int = 5) -> list:
    g = g[g["how"] == "forma"]
    out = []
    for forma, n in g["forma"].value_counts().head(top).items():
        out.append({"texto": _texto_limpio(g[g["forma"] == forma], forma), "n": int(n)})
    return out


def construir_datos(r: dict, nombre: str) -> dict:
    com = r["comentarios"].copy()
    com["local"] = com["ts"].dt.tz_convert(ZONA)
    narr = r["narrativas"]
    coord = r["coordinacion"]
    cuentas = r["cuentas"]
    pubs = r.get("publicaciones", pd.DataFrame())
    fuentes = r.get("fuentes", pd.DataFrame())
    fuente_nombre = dict(zip(fuentes["fuente_id"], fuentes["nombre"])) if len(fuentes) else {}
    pub_info = {}
    if len(pubs):
        for p in pubs.itertuples():
            pub_info[p.publicacion_id] = {"titular": p.titular, "fuente": fuente_nombre.get(p.fuente_id, p.fuente_id)}

    inicio = com["local"].min().floor("D")
    fin = com["local"].max()
    com["cubeta"] = ((com["local"] - inicio).dt.total_seconds() // (CUBETA_H * 3600)).astype(int)
    n_cubetas = int(com["cubeta"].max()) + 1
    com["coordinado"] = com["grupo"] != ""

    def serie(mask):
        return np.bincount(com.loc[mask, "cubeta"], minlength=n_cubetas).tolist()

    narrativas = []
    for fila in narr.itertuples():
        m = com["narrativa"] == fila.narrativa
        g = com[m]
        origen = pd.Timestamp(fila.fecha_origen)
        narrativas.append({
            "id": fila.narrativa, "titulo": fila.titulo, "frases": fila.frases_clave.split("|") if fila.frases_clave else [],
            "origen": fila.fecha_origen, "pub_origen": fila.publicacion_origen,
            "pub_origen_info": pub_info.get(fila.publicacion_origen, {}),
            "emergente": bool((origen - inicio).total_seconds() > 36 * 3600),
            "n": int(fila.comentarios), "apoya": int(fila.apoya), "rechaza": int(fila.rechaza),
            "coordinados": int(fila.comentarios_coordinados), "cuentas": int(g["cuenta_id"].nunique()),
            "publicaciones": int(g["publicacion_id"].nunique()),
            "serie_apoya": serie(m & (com["postura"] == "apoya") & ~com["coordinado"]),
            "serie_rechaza": serie(m & (com["postura"] == "rechaza") & ~com["coordinado"]),
            "serie_coord": serie(m & com["coordinado"]),
            "frases_apoya": _frases(g[(g["postura"] == "apoya") & ~g["coordinado"]]),
            "frases_rechaza": _frases(g[(g["postura"] == "rechaza") & ~g["coordinado"]]),
            "operaciones": [],
            "evidencia_postura": json.loads(fila.evidencia_postura),
        })

    grupos = coord["grupos"]
    vinc = coord["vinculos"]
    eventos = coord["eventos"]
    etiqueta = r["etiqueta"]
    cta = cuentas.set_index("cuenta_id") if len(cuentas) else pd.DataFrame()
    operaciones = []
    for op in sorted(set(grupos.values()), key=lambda x: int(x[2:])):
        miembros = sorted(a for a, g in grupos.items() if g == op)
        g = com[com["grupo"] == op]
        con_narr = g[g["narrativa"] != ""]
        principal = con_narr["narrativa"].value_counts()
        nid = principal.index[0] if len(principal) else ""
        postura = con_narr[con_narr["narrativa"] == nid]["postura"].value_counts().index[0] if nid else ""
        firmas = [f for f, o in coord["firmas"].items() if o == op]
        en_firma = g[g["forma"].isin(firmas)]
        ev = eventos[eventos["grupo"] == op].sort_values("t0")
        lista_ev = []
        for e in ev.itertuples():
            lista_ev.append({"pub": e.publicacion_id, **pub_info.get(e.publicacion_id, {}),
                             "inicio": pd.Timestamp(e.t0, unit="s", tz="UTC").tz_convert(ZONA).strftime("%d/%m %H:%M"),
                             "minutos": round((e.t1 - e.t0) / 60), "cuentas": int(e.cuentas), "comentarios": int(e.comentarios),
                             "cubeta": int(((pd.Timestamp(e.t0, unit="s", tz="UTC").tz_convert(ZONA) - inicio).total_seconds()) // (CUBETA_H * 3600))})
        creadas = pd.to_datetime(cta["fecha_creacion"].reindex(miembros), errors="coerce") if len(cta) else pd.Series(dtype="datetime64[ns]")
        antig = (inicio.tz_localize(None) - creadas).dt.days.dropna()
        resto = pd.to_datetime(cta["fecha_creacion"], errors="coerce") if len(cta) else creadas
        antig_resto = (inicio.tz_localize(None) - resto[~resto.index.isin(miembros)]).dt.days.dropna()
        horas = np.bincount(g["local"].dt.hour, minlength=24)
        e_op = vinc[vinc["a"].isin(miembros) & vinc["b"].isin(miembros)]
        aristas = list(zip(e_op["a"], e_op["b"]))
        pos = _disposicion(miembros, aristas)
        por_cuenta = g.groupby("cuenta_id").size()
        detalle = []
        for a in miembros:
            fila_c = cta.loc[a] if len(cta) and a in cta.index else None
            detalle.append({"id": a, "usuario": "" if fila_c is None else str(fila_c.get("usuario", "")),
                            "creada": "" if fila_c is None else str(fila_c.get("fecha_creacion", "")),
                            "seguidores": 0 if fila_c is None else int(float(fila_c.get("seguidores", 0) or 0)),
                            "comentarios": int(por_cuenta.get(a, 0)), "xy": pos.get(a, [0.5, 0.5])})
        nocturno = float(horas[0:6].sum() / max(horas.sum(), 1))
        mediana = float(antig.median()) if len(antig) else None
        rasgos = []
        if mediana is not None:
            rasgos.append("cuentas creadas durante el período analizado" if mediana <= 0
                          else f"cuentas nuevas: creadas {int(mediana)} días antes del período (mediana)" if mediana < 90
                          else f"cuentas antiguas (mediana {mediana / 365:.1f} años)")
        if nocturno >= 0.5:
            rasgos.append(f"{nocturno:.0%} de su actividad entre las 0 y las 6 h")
        if len(g):
            rasgos.append(f"{len(en_firma) / len(g):.0%} de sus comentarios usa textos que solo usa el grupo")
        entre_si = g["respuesta_a"].isin(set(g["comentario_id"])).sum()
        if entre_si:
            rasgos.append(f"{int(entre_si)} respuestas entre cuentas del mismo grupo")
        operaciones.append({
            "grupo": op, "cuentas": len(miembros), "comentarios": int(len(g)), "narrativa": nid, "postura": postura,
            "comentarios_en_narrativa": int(principal.iloc[0]) if len(principal) else 0,
            "antiguedad_mediana_dias": mediana,
            "antiguedad_mediana_resto_dias": float(antig_resto.median()) if len(antig_resto) else None,
            "horas": horas.tolist(), "eventos": lista_ev, "rasgos": rasgos,
            "firmas": [{"texto": _texto_limpio(g[g["forma"] == f], f), "n": int((g["forma"] == f).sum())}
                       for f in sorted(firmas, key=lambda f: -(g["forma"] == f).sum())[:6]],
            "detalle": detalle, "aristas": [[a, b] for a, b in aristas],
            "primera": g["local"].min().strftime("%d/%m %H:%M") if len(g) else "",
            "ultima": g["local"].max().strftime("%d/%m %H:%M") if len(g) else "",
        })

    # Una narrativa está "empujada" cuando es el objetivo principal de una operación; los comentarios
    # sueltos que esas cuentas dejan en otros temas son actividad de cobertura.
    for n in narrativas:
        n["operaciones"] = [{"grupo": o["grupo"], "comentarios": o["comentarios_en_narrativa"], "cuentas": o["cuentas"]}
                            for o in operaciones if o["narrativa"] == n["id"]]

    horas_todas = np.bincount(com.loc[~com["coordinado"], "local"].dt.hour, minlength=24)
    s = r["resumen"]
    return {
        "set": nombre, "inicio": inicio.isoformat(), "fin": fin.isoformat(), "cubeta_h": CUBETA_H, "cubetas": n_cubetas,
        "comentarios": int(len(com)), "cuentas": int(com["cuenta_id"].nunique()), "publicaciones": int(com["publicacion_id"].nunique()),
        "ruido": int((com["narrativa"] == "").sum()), "serie_ruido": serie(com["narrativa"] == ""),
        "horas_organico": horas_todas.tolist(),
        "costo": r["costo"], "tiempos": s["tiempos"], "texto": s["texto"], "ingesta": s["ingesta"],
        "pares_evaluados": s["pares_evaluados"],
        "asignado_por": {k: int(v) for k, v in com["how"].value_counts().items()},
        "narrativas": narrativas, "operaciones": operaciones,
    }


def generar(resultado: dict, destino: Path, nombre: str) -> Path:
    datos = construir_datos(resultado, nombre)
    html = PLANTILLA.read_text(encoding="utf-8").replace(
        "/*__DATOS__*/null", json.dumps(datos, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/"))
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(html, encoding="utf-8")
    return destino
