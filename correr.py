#!/usr/bin/env python3
"""Corre el pipeline completo con un solo comando.

    python correr.py --datos datos/final --salida entrega
    python correr.py --datos datos/practica --salida salida/practica --puntuar

Lee los CSV de --datos, los guarda limpios en Parquet (DuckDB), descubre narrativas, postura,
origen y operaciones coordinadas, escribe los cuatro archivos de la entrega y arma el panel.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from qualy import panel, pipeline


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datos", default="datos/final", type=Path, help="carpeta con comentarios.csv.gz, cuentas.csv.gz, etc.")
    ap.add_argument("--salida", default="entrega", type=Path, help="carpeta donde se escribe la entrega")
    ap.add_argument("--almacen", default=None, type=Path, help="carpeta del almacén Parquet/DuckDB (por defecto salida/almacen/<set>)")
    ap.add_argument("--panel", default=None, type=Path, help="archivo HTML del panel (por defecto panel/<set>.html)")
    ap.add_argument("--puntuar", action="store_true", help="si el set trae verdad/, corre el evaluador oficial al final")
    args = ap.parse_args()

    nombre = args.datos.resolve().name
    almacen = args.almacen or Path("salida") / "almacen" / nombre
    destino_panel = args.panel or Path("panel") / f"{nombre}.html"
    print(f"Procesando {args.datos} ...")
    r = pipeline.correr(args.datos, args.salida, almacen)
    panel.generar(r, destino_panel, nombre)
    s = r["resumen"]
    print(f"\n{r['costo']['comentarios_procesados']:,} comentarios en {s['segundos']} s  ·  USD 0 de IA")
    print(f"{s['narrativas']} narrativas, {s['ruido']:,} comentarios de ruido")
    print(f"{s['cuentas_coordinadas']} cuentas coordinadas en {s['operaciones']} operaciones")
    print(f"Entrega: {args.salida}/   Panel: {destino_panel}   Base: {s['base']}")
    (almacen / "resumen.json").write_text(json.dumps(s, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    verdad = args.datos / "verdad"
    if args.puntuar and verdad.exists():
        print()
        subprocess.run([sys.executable, "evaluacion/puntuar.py", "--verdad", str(verdad), "--entrega", str(args.salida)], check=False)


if __name__ == "__main__":
    main()
