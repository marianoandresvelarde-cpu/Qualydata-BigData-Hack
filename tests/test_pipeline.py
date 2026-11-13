"""Pruebas del pipeline. `python -m unittest` desde la raíz del repo (solo biblioteca estándar + dependencias)."""
import csv
import gzip
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from qualy import texto  # noqa: E402
from qualy.narrativas import _ParityDSU  # noqa: E402

PRACTICA = RAIZ / "datos" / "practica"


class Texto(unittest.TestCase):
    def test_tokens_normaliza(self):
        self.assertEqual(texto.tokens("¡¡Qué VERGÜENZAAA!! @juan.perez mirá https://x.co/a 1.850"),
                         ("que", "verguenza", "@", "mira", "1850"))
        self.assertEqual(texto.tokens("El niño #DiqueSi"), ("el", "niño", "#diquesi"))
        self.assertEqual(texto.tokens(None), ())

    def test_typos_van_a_la_palabra_frecuente(self):
        vocab = Counter({"comisaria": 500, "comisraia": 2, "comisariaa": 3, "camisaria": 1, "obra": 300, "otra": 280})
        m = texto.typo_map(vocab)
        self.assertEqual(m["comisraia"], "comisaria")
        self.assertEqual(m["comisariaa"], "comisaria")
        self.assertNotIn("otra", m)  # dos palabras frecuentes parecidas no se fusionan
        self.assertNotIn("obra", m)

    def test_muletillas_se_aprenden_de_los_datos(self):
        nucleos = [("frase", "numero", str(i), "sobre", "algo") for i in range(15)]
        textos = nucleos + [("la", "verdad") + n for n in nucleos] + [n + ("posta",) for n in nucleos]
        cores, pre, suf = texto.strip_wrappers(textos)
        self.assertIn(("la", "verdad"), pre)
        self.assertIn(("posta",), suf)
        self.assertEqual(set(cores), set(nucleos))


class Biparticion(unittest.TestCase):
    def test_bandos_opuestos(self):
        d = _ParityDSU()
        self.assertTrue(d.union_opposite("a", "b"))
        self.assertTrue(d.union_opposite("b", "c"))
        self.assertEqual(d.find("a")[1], d.find("c")[1])
        self.assertNotEqual(d.find("a")[1], d.find("b")[1])
        self.assertFalse(d.union_opposite("a", "c"))  # contradicción: ya estaban del mismo lado


@unittest.skipUnless((PRACTICA / "comentarios.csv.gz").exists(), "falta el set de práctica")
class PuntaACabo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        datos = cls.tmp / "datos"
        datos.mkdir()
        for f in ("cuentas.csv.gz", "publicaciones.csv.gz", "fuentes.csv"):
            shutil.copy(PRACTICA / f, datos / f)
        # Datos sucios a propósito: fila duplicada, fecha inválida, id vacío y respuesta a un comentario inexistente.
        with gzip.open(PRACTICA / "comentarios.csv.gz", "rt", encoding="utf-8", newline="") as fh:
            filas = list(csv.reader(fh))
        cls.validas = len(filas) - 1
        sucias = [filas[1], ["cX1", "p00001", "u000001", "no-es-fecha", "hola", "", "0"], ["", "p00001", "u000001", filas[1][3], "sin id", "", "0"],
                  ["cX2", "p00001", "u000001", filas[1][3], "Exacto", "c9999999", "abc"]]
        with gzip.open(datos / "comentarios.csv.gz", "wt", encoding="utf-8", newline="") as fh:
            csv.writer(fh).writerows(filas + sucias)
        cls.salida = cls.tmp / "entrega"
        cls.proc = subprocess.run([sys.executable, "correr.py", "--datos", str(datos), "--salida", str(cls.salida),
                                   "--almacen", str(cls.tmp / "almacen"), "--panel", str(cls.tmp / "panel.html")],
                                  cwd=RAIZ, capture_output=True, text=True)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_corre_y_escribe_la_entrega(self):
        self.assertEqual(self.proc.returncode, 0, self.proc.stderr[-2000:])
        for f in ("comentarios.csv", "narrativas.csv", "cuentas.csv", "costo.json"):
            self.assertTrue((self.salida / f).exists(), f)
        with open(self.salida / "comentarios.csv", encoding="utf-8") as fh:
            filas = list(csv.DictReader(fh))
        self.assertEqual(len(filas), self.validas + 1)  # las válidas + cX2; duplicada, sin id y sin fecha quedan afuera
        self.assertEqual(len({r["comentario_id"] for r in filas}), len(filas))
        self.assertTrue(all(r["postura"] in ("apoya", "rechaza") for r in filas if r["narrativa"]))
        self.assertTrue(all(r["postura"] == "" for r in filas if not r["narrativa"]))
        costo = json.loads((self.salida / "costo.json").read_text(encoding="utf-8"))
        self.assertEqual(costo["tokens_ia"], 0)
        self.assertIn("const D = {", (self.tmp / "panel.html").read_text(encoding="utf-8"))

    def test_puntaje_en_practica(self):
        sys.path.insert(0, str(RAIZ / "evaluacion"))
        import puntuar
        narr = puntuar.score_narratives(PRACTICA / "verdad", self.salida)
        coord = puntuar.score_coordination(PRACTICA / "verdad", self.salida)
        self.assertGreater(narr["bcubed_f1"], 0.98)
        self.assertGreater(narr["ruido_f1"], 0.98)
        self.assertGreater(narr["postura_macro_f1"], 0.85)
        self.assertGreater(narr["origen"], 0.95)
        self.assertGreater(coord["f1"], 0.95)
        self.assertGreater(coord["grupos_bcubed_f1"], 0.95)

    def test_es_determinista(self):
        otra = self.tmp / "entrega2"
        p = subprocess.run([sys.executable, "correr.py", "--datos", str(self.tmp / "datos"), "--salida", str(otra),
                            "--almacen", str(self.tmp / "almacen2"), "--panel", str(self.tmp / "panel2.html")],
                           cwd=RAIZ, capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr[-2000:])
        for f in ("comentarios.csv", "narrativas.csv", "cuentas.csv"):
            self.assertEqual((self.salida / f).read_bytes(), (otra / f).read_bytes(), f)


if __name__ == "__main__":
    unittest.main()
