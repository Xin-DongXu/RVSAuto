#!/usr/bin/env python3
"""Tests for complex assembly and affinity matrices."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rvsauto.docking import (  # noqa: E402
    create_complex_pdb,
    extract_best_ligand_pose,
)
from rvsauto.pdbio import format_pdb_line  # noqa: E402
from rvsauto.tables import parse_docking_label, write_docking_tables  # noqa: E402


def _atom(serial, name, resname, chain, resseq, x, y, z, element, record="ATOM"):
    return format_pdb_line(
        record, serial, name, " ", resname, chain, resseq, " ",
        x, y, z, 1.0, 0.0, element,
    )


class TestPoseAndComplex(unittest.TestCase):
    def test_extract_pose_without_model_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            pdbqt = os.path.join(tmp, "lig_out.pdbqt")
            out = os.path.join(tmp, "lig.pdb")
            with open(pdbqt, "w", encoding="utf-8") as fh:
                fh.write("REMARK VINA RESULT:    -7.500      0.000      0.000\n")
                fh.write(
                    "ATOM      1  C   UNL     1      10.000  20.000  30.000"
                    "  0.00  0.00     0.000 C \n"
                )
                fh.write(
                    "ATOM      2  OA  UNL     1      11.000  20.000  30.000"
                    "  0.00  0.00    -0.400 OA\n"
                )
            self.assertTrue(extract_best_ligand_pose(pdbqt, out))
            text = Path(out).read_text(encoding="utf-8")
            self.assertIn("HETATM", text)
            self.assertIn("10.000", text)
            self.assertIn(" O", text)

    def test_complex_uses_apo_and_rewrites_ligand_chain(self):
        with tempfile.TemporaryDirectory() as tmp:
            apo = os.path.join(tmp, "apo.pdb")
            lig = os.path.join(tmp, "lig.pdb")
            complex_pdb = os.path.join(tmp, "complex.pdb")
            with open(apo, "w", encoding="utf-8") as fh:
                fh.write(_atom(1, " N  ", "MET", "A", 1, 0, 0, 0, "N"))
                fh.write(_atom(2, " CA ", "MET", "A", 1, 1, 0, 0, "C"))
                fh.write("END\n")
            with open(lig, "w", encoding="utf-8") as fh:
                fh.write(_atom(1, " C  ", "UNL", "A", 1, 5, 5, 5, "C", "ATOM"))
                fh.write("END\n")
            self.assertTrue(create_complex_pdb(apo, lig, complex_pdb))
            text = Path(complex_pdb).read_text(encoding="utf-8")
            het = [ln for ln in text.splitlines() if ln.startswith("HETATM")]
            atm = [ln for ln in text.splitlines() if ln.startswith("ATOM")]
            self.assertEqual(len(atm), 2)
            self.assertEqual(len(het), 1)
            self.assertEqual(het[0][21], "Z")
            self.assertIn("LIG", het[0][17:20])
            self.assertGreater(int(het[0][6:11]), 2)

    def test_dirty_crystal_ligand_must_not_be_used_as_receptor(self):
        """Documents the v10.1 bug: original PDB still contains the crystal ligand."""
        with tempfile.TemporaryDirectory() as tmp:
            dirty = os.path.join(tmp, "crystal.pdb")
            apo = os.path.join(tmp, "apo.pdb")
            pose = os.path.join(tmp, "pose.pdb")
            bad = os.path.join(tmp, "bad_complex.pdb")
            good = os.path.join(tmp, "good_complex.pdb")
            with open(dirty, "w", encoding="utf-8") as fh:
                fh.write(_atom(1, " CA ", "MET", "A", 1, 0, 0, 0, "C"))
                fh.write(_atom(2, " C  ", "ATP", "A", 99, 8, 8, 8, "C", "HETATM"))
                fh.write("END\n")
            with open(apo, "w", encoding="utf-8") as fh:
                fh.write(_atom(1, " CA ", "MET", "A", 1, 0, 0, 0, "C"))
                fh.write("END\n")
            with open(pose, "w", encoding="utf-8") as fh:
                fh.write(_atom(1, " C  ", "LIG", "Z", 1, 4, 4, 4, "C", "HETATM"))
                fh.write("END\n")
            create_complex_pdb(dirty, pose, bad)
            create_complex_pdb(apo, pose, good)
            bad_het = [
                ln for ln in Path(bad).read_text(encoding="utf-8").splitlines()
                if ln.startswith("HETATM")
            ]
            good_het = [
                ln for ln in Path(good).read_text(encoding="utf-8").splitlines()
                if ln.startswith("HETATM")
            ]
            self.assertGreaterEqual(len(bad_het), 2)
            self.assertEqual(len(good_het), 1)


class TestMatrices(unittest.TestCase):
    def test_parse_label(self):
        self.assertEqual(
            parse_docking_label("config_AF-Q9ZWT3-F1-model_v6_pocket_2_atp", "atp"),
            ("AF-Q9ZWT3-F1-model_v6", "pocket_2"),
        )
        self.assertEqual(
            parse_docking_label("PROT_pocket_1_ligA", "ligA"),
            ("PROT", "pocket_1"),
        )

    def test_affinity_matrix_shape(self):
        affinities = {
            "ligA": [
                ("config_P1_pocket_1_ligA", -8.1),
                ("config_P1_pocket_2_ligA", -6.0),
                ("config_P2_pocket_1_ligA", -7.2),
            ],
            "ligB": [
                ("config_P1_pocket_1_ligB", -5.5),
                ("config_P2_pocket_1_ligB", -9.0),
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_docking_tables(affinities, ["ligA", "ligB"], tmp)
            pocket_txt = Path(paths["matrix_pockets"]).read_text(encoding="utf-8").splitlines()
            self.assertEqual(pocket_txt[0], "Protein_ID\tPocket\tligA\tligB")
            self.assertIn("P1\tpocket_1\t-8.1000\t-5.5000", pocket_txt)
            self.assertIn("P1\tpocket_2\t-6.0000\t", pocket_txt)
            prot_txt = Path(paths["matrix_proteins"]).read_text(encoding="utf-8").splitlines()
            self.assertEqual(prot_txt[0], "Protein_ID\tligA\tligB")
            self.assertIn("P1\t-8.1000\t-5.5000", prot_txt)
            self.assertIn("P2\t-7.2000\t-9.0000", prot_txt)
            choice = Path(paths["matrix_best_pocket"]).read_text(encoding="utf-8")
            self.assertIn("pocket_1", choice)


if __name__ == "__main__":
    unittest.main()
