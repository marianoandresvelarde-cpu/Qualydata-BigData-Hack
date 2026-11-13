"""Almacenamiento: DuckDB sobre Parquet.

Los CSV crudos se leen una sola vez con DuckDB, se limpian en SQL y se guardan como Parquet
(columnar, comprimido, tipado). Todo lo demás (pipeline, panel, consultas ad hoc) lee de ahí.
No hay servidor que levantar y el mismo SQL corre igual sobre 150 mil filas o sobre cientos de
millones particionando el Parquet por día.
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

LIMPIEZA_COMENTARIOS = """
    SELECT * EXCLUDE (rn) FROM (
        SELECT
            trim(comentario_id)                                   AS comentario_id,
            trim(coalesce(publicacion_id, ''))                    AS publicacion_id,
            trim(coalesce(cuenta_id, ''))                         AS cuenta_id,
            try_cast(fecha AS TIMESTAMPTZ)                        AS ts,
            coalesce(texto, '')                                   AS texto,
            trim(coalesce(respuesta_a, ''))                       AS respuesta_a,
            coalesce(try_cast(me_gusta AS INTEGER), 0)            AS me_gusta,
            row_number() OVER (PARTITION BY trim(comentario_id) ORDER BY fecha) AS rn
        FROM read_csv('{src}', header = true, all_varchar = true, ignore_errors = true)
        WHERE comentario_id IS NOT NULL AND trim(comentario_id) <> ''
    ) WHERE rn = 1 AND ts IS NOT NULL
    ORDER BY ts, comentario_id
"""


def _existe(datos: Path, nombre: str) -> Path | None:
    for ext in (".csv.gz", ".csv"):
        p = datos / (nombre + ext)
        if p.exists():
            return p
    return None


def ingerir(datos: Path, almacen: Path) -> dict:
    """CSV crudos -> Parquet limpio. Devuelve cuántas filas se leyeron y cuántas se descartaron."""
    almacen.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("SET TimeZone = 'America/Argentina/Buenos_Aires'")
    src = _existe(datos, "comentarios")
    if src is None:
        raise FileNotFoundError(f"No encuentro comentarios.csv(.gz) en {datos}")
    crudas = con.execute(f"SELECT count(*) FROM read_csv('{src}', header = true, all_varchar = true, ignore_errors = true)").fetchone()[0]
    con.execute(f"COPY ({LIMPIEZA_COMENTARIOS.format(src=src)}) TO '{almacen / 'comentarios.parquet'}' (FORMAT parquet, COMPRESSION zstd)")
    limpias = con.execute(f"SELECT count(*) FROM '{almacen / 'comentarios.parquet'}'").fetchone()[0]
    info = {"comentarios_leidos": crudas, "comentarios_validos": limpias, "comentarios_descartados": crudas - limpias}
    for nombre, clave in (("cuentas", "cuenta_id"), ("publicaciones", "publicacion_id"), ("fuentes", "fuente_id")):
        p = _existe(datos, nombre)
        if p is None:
            continue
        con.execute(f"""COPY (SELECT DISTINCT ON (trim({clave})) * REPLACE (trim({clave}) AS {clave})
                              FROM read_csv('{p}', header = true, all_varchar = true, ignore_errors = true)
                              WHERE {clave} IS NOT NULL AND trim({clave}) <> '' ORDER BY {clave})
                        TO '{almacen / (nombre + '.parquet')}' (FORMAT parquet, COMPRESSION zstd)""")
        info[nombre] = con.execute(f"SELECT count(*) FROM '{almacen / (nombre + '.parquet')}'").fetchone()[0]
    con.close()
    return info


def leer(almacen: Path, nombre: str) -> pd.DataFrame:
    p = almacen / (nombre + ".parquet")
    if not p.exists():
        return pd.DataFrame()
    con = duckdb.connect()
    con.execute("SET TimeZone = 'America/Argentina/Buenos_Aires'")
    df = con.execute(f"SELECT * FROM '{p}'").df()
    con.close()
    return df


def guardar_resultados(almacen: Path, tablas: dict) -> Path:
    """Guarda las tablas de resultado como Parquet y arma una base DuckDB con vistas sobre todo."""
    con = duckdb.connect()
    for nombre, df in tablas.items():
        con.register("tmp_df", df)
        con.execute(f"COPY (SELECT * FROM tmp_df) TO '{almacen / (nombre + '.parquet')}' (FORMAT parquet, COMPRESSION zstd)")
        con.unregister("tmp_df")
    con.close()
    db = almacen / "qualy.duckdb"
    if db.exists():
        db.unlink()
    con = duckdb.connect(str(db))
    for p in sorted(almacen.glob("*.parquet")):
        con.execute(f"CREATE VIEW {p.stem} AS SELECT * FROM read_parquet('{p.resolve().as_posix()}')")
    con.close()
    return db
