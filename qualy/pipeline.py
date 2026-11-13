"""Orquestación: de los CSV crudos a la carpeta de entrega, en una sola corrida."""
from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import almacen as alm
from . import coordinacion, narrativas

ZONA = "America/Argentina/Buenos_Aires"


def _iso(ts: pd.Timestamp) -> str:
    s = ts.tz_convert(ZONA).strftime("%Y-%m-%dT%H:%M:%S%z")
    return s[:-2] + ":" + s[-2:]


def _hardware() -> str:
    import os
    try:
        gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2 ** 30
        mem = f"{gb:.0f} GB RAM"
    except (ValueError, OSError, AttributeError):
        mem = "RAM no informada"
    return f"{platform.system()} {platform.machine()}, {os.cpu_count()} núcleos, {mem}, sin GPU (un solo proceso)"


def correr(datos: Path, salida: Path, almacen: Path, verbose: bool = True) -> dict:
    t0 = time.perf_counter()
    tiempos = {}

    def paso(nombre, t):
        tiempos[nombre] = round(time.perf_counter() - t, 2)
        if verbose:
            print(f"  [{tiempos[nombre]:6.2f}s] {nombre}")

    t = time.perf_counter()
    ingesta = alm.ingerir(datos, almacen)
    com = alm.leer(almacen, "comentarios")
    cuentas = alm.leer(almacen, "cuentas")
    ids = set(com["comentario_id"])
    huerfanas = ~com["respuesta_a"].isin(ids) & (com["respuesta_a"] != "")
    com.loc[huerfanas, "respuesta_a"] = ""  # respuesta a un comentario que no está en el set
    com.loc[com["respuesta_a"] == com["comentario_id"], "respuesta_a"] = ""
    ingesta["respuestas_huerfanas"] = int(huerfanas.sum())
    com["ts_s"] = com["ts"].astype("int64") // 10 ** (9 if str(com["ts"].dtype).find("[ns") >= 0 else 6)
    paso("ingesta a Parquet (DuckDB) y limpieza", t)

    t = time.perf_counter()
    com["forma"], info_texto = narrativas.construir_formas(com)
    paso("formas canónicas", t)

    t = time.perf_counter()
    res = narrativas.descubrir(com)
    com["narr"], com["side"], com["how"] = res["narr"], res["side"], res["how"]
    paso("grafo de discusión: narrativas, ruido y bandos", t)

    t = time.perf_counter()
    coord = coordinacion.detectar(com.rename(columns={"ts": "ts_dt", "ts_s": "ts"}))
    grupos = coord["grupos"]
    paso("coordinación: coincidencias repetidas, núcleos y firmas", t)

    t = time.perf_counter()
    orient = narrativas.orientar(com, set(grupos))
    con_narr = com[com["narr"] != ""]
    tam = con_narr.groupby("narr").size().sort_values(ascending=False)
    etiqueta = {raiz: f"n{i + 1:02d}" for i, raiz in enumerate(tam.index)}
    rech = com["narr"].map(lambda r: orient[r]["rechaza"] if r else -1).to_numpy()
    com["narrativa"] = com["narr"].map(etiqueta).fillna("")
    com["postura"] = np.where(com["narr"] == "", "", np.where(com["side"].to_numpy() == rech, "rechaza", "apoya"))
    com["grupo"] = com["cuenta_id"].map(grupos).fillna("")
    claves = narrativas.frases_clave(com[com["postura"] == "apoya"])

    filas = []
    for raiz in tam.index:
        g = com[com["narr"] == raiz]
        apoya = g[(g["postura"] == "apoya") & (g["how"] == "forma")]
        organico = apoya[apoya["grupo"] == ""]
        # Título: la redacción más repetida entre quienes sostienen la narrativa, prefiriendo las que
        # nombran sus términos distintivos (una frase suelta muy repetida no resume nada).
        cand = (organico if len(organico) else apoya if len(apoya) else g)["forma"].value_counts()
        largas = cand[[len(f.split()) >= 4 for f in cand.index]]
        cand = largas if len(largas) else cand
        terminos = claves.get(raiz, [])
        top = max(cand.items(), key=lambda kv: (kv[1] * (1 + sum(f" {t} " in f" {kv[0]} " for t in terminos)), kv[0]))[0]
        primero = g.loc[g["ts"].idxmin()]
        filas.append({"narrativa": etiqueta[raiz],
                      "titulo": narrativas._texto_limpio(g[g["forma"] == top], top),
                      "frases_clave": "|".join(claves.get(raiz, [])),
                      "fecha_origen": _iso(primero["ts"]),
                      "publicacion_origen": primero["publicacion_id"],
                      "comentarios": len(g), "apoya": int((g["postura"] == "apoya").sum()),
                      "rechaza": int((g["postura"] == "rechaza").sum()),
                      "comentarios_coordinados": int((g["grupo"] != "").sum()),
                      "evidencia_postura": json.dumps(orient[raiz]["evidencia"], default=float)})
    narr_df = pd.DataFrame(filas)
    paso("postura, origen y descripción de narrativas", t)

    t = time.perf_counter()
    salida.mkdir(parents=True, exist_ok=True)
    com[["comentario_id", "narrativa", "postura"]].sort_values("comentario_id").to_csv(salida / "comentarios.csv", index=False)
    narr_df[["narrativa", "titulo", "frases_clave", "fecha_origen"]].to_csv(salida / "narrativas.csv", index=False)
    todas = sorted(set(cuentas["cuenta_id"]) | set(com["cuenta_id"])) if len(cuentas) else sorted(set(com["cuenta_id"]))
    cuentas_df = pd.DataFrame({"cuenta_id": todas, "grupo": [grupos.get(c, "") for c in todas]})
    cuentas_df.to_csv(salida / "cuentas.csv", index=False)
    tablas = {"resultado_comentarios": com[["comentario_id", "forma", "narrativa", "postura", "how", "grupo"]].rename(columns={"how": "asignado_por"}),
              "resultado_narrativas": narr_df, "resultado_cuentas": cuentas_df,
              "resultado_vinculos": coord["vinculos"],
              "resultado_eventos": coord["eventos"].assign(narrativa=lambda d: d["narr"].map(etiqueta)).drop(columns=["narr", "side"])}
    db = alm.guardar_resultados(almacen, tablas)
    paso("escritura de entrega y resultados en el almacén", t)

    segundos = round(time.perf_counter() - t0, 2)
    costo = {"comentarios_procesados": int(len(com)), "segundos": segundos, "tokens_ia": 0, "usd_ia": 0.0,
             "usd_por_mil_comentarios": 0.0, "hardware": _hardware(),
             "almacenamiento": "DuckDB local sobre Parquet (zstd)",
             "notas": ("Sin IA generativa ni embeddings: normalización de texto aprendida de los datos, grafo de "
                       "respuestas para narrativas y postura, y coincidencias repetidas para coordinación. "
                       f"{len(com) / max(segundos, 1e-9):,.0f} comentarios por segundo en un solo proceso de CPU.")}
    (salida / "costo.json").write_text(json.dumps(costo, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    resumen = {"ingesta": ingesta, "texto": info_texto, "tiempos": tiempos, "segundos": segundos,
               "narrativas": int(len(narr_df)), "ruido": int((com["narrativa"] == "").sum()),
               "acompañamientos": sorted(res["acks"]), "conflictos_biparticion": int(res["conflictos_biparticion"]),
               "cuentas_coordinadas": len(grupos), "operaciones": int(len(set(grupos.values()))),
               "pares_evaluados": int(coord["pares_evaluados"]), "base": str(db)}
    return {"comentarios": com, "narrativas": narr_df, "cuentas": cuentas,
            "publicaciones": alm.leer(almacen, "publicaciones"), "fuentes": alm.leer(almacen, "fuentes"), "coordinacion": coord, "costo": costo,
            "resumen": resumen, "etiqueta": etiqueta, "orientacion": orient}
