#!/usr/bin/env python3
"""Unit tests for AF2BIND pocket parsing (no GPU / UniDock required)."""

from __future__ import annotations

import logging
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rvsauto.pockets import (
    P2RANK_DEFAULT_MIN_PROBABILITY,
    normalize_p2rank_min_probability,  # noqa: E402
    DEFAULT_BOX_MAX_SIDE,
    UNIDOCK_MAX_GRID_VOLUME,
    calculate_box_size,
    index_af2bind_dir,
    load_af2bind_table,
    match_af2bind_csv,
    pockets_from_af2bind,
    pockets_from_p2rank_csv,
    resolve_p2rank_predictions_dir,
    select_af2bind_residues,
    single_linkage_clusters,
    write_vina_config,
)
from rvsauto.pdbio import filter_pdb_by_plddt, format_pdb_line  # noqa: E402

AF2BIND_DIR = ROOT / "AF2BIND_out"


def _write_ca_pdb(path, residues):
    """residues: iterable of (chain, resi, x, y, z)."""
    with open(path, "w", encoding="utf-8") as fh:
        for i, (chain, resi, x, y, z) in enumerate(residues, 1):
            fh.write(
                format_pdb_line(
                    "ATOM", i, " CA ", " ", "ALA", chain, resi, " ",
                    x, y, z, 1.0, 0.0, "C",
                )
            )
        fh.write("END\n")


class TestAf2bindIndex(unittest.TestCase):
    @unittest.skipUnless(AF2BIND_DIR.is_dir(), "AF2BIND_out/ not present")
    def test_prefers_full_csv_over_top15(self):
        index = index_af2bind_dir(str(AF2BIND_DIR), recursive=True)
        self.assertIn("AF-Q9ZWT3-F1-model_v6", index)
        self.assertIn("Q9ZWT3", index)
        chosen = index["AF-Q9ZWT3-F1-model_v6"]
        self.assertTrue(chosen.endswith("_af2bind.csv"))
        self.assertFalse(chosen.endswith("_af2bind_top15.csv"))

    def test_match_transcript_id_with_dot_in_stem(self):
        """Regression: gene.1 IDs must not be truncated at the isoform dot."""
        with tempfile.TemporaryDirectory() as tmp:
            stem = "mescchra01g00156030.1_jaile_model_apo"
            csv_path = Path(tmp) / f"{stem}_af2bind.csv"
            csv_path.write_text(
                "chain,resi,resn,p(bind)\nA,1,ALA,0.5\n", encoding="utf-8"
            )
            index = index_af2bind_dir(tmp)
            self.assertIn(stem, index)
            self.assertIsNotNone(match_af2bind_csv(f"{stem}.pdb", index))

    @unittest.skipUnless(AF2BIND_DIR.is_dir(), "AF2BIND_out/ not present")
    def test_match_pdb_stem_and_uniprot(self):
        index = index_af2bind_dir(str(AF2BIND_DIR))
        self.assertIsNotNone(
            match_af2bind_csv("AF-X5JA13-F1-model_v6.pdb", index)
        )
        self.assertIsNotNone(match_af2bind_csv("X5JA13.pdb", index))

    @unittest.skipUnless(AF2BIND_DIR.is_dir(), "AF2BIND_out/ not present")
    def test_top15_fallback_filename_does_not_keep_suffix(self):
        """Regression: replace('_af2bind','') on *_af2bind_top15 left '_top15'."""
        with tempfile.TemporaryDirectory() as tmp:
            src = AF2BIND_DIR / "AF-Q9ZWT3-F1-model_v6_af2bind_top15.csv"
            dest = Path(tmp) / src.name
            dest.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
            index = index_af2bind_dir(tmp)
            self.assertIn("AF-Q9ZWT3-F1-model_v6", index)
            self.assertNotIn("AF-Q9ZWT3-F1-model_v6_top15", index)


class TestAf2bindTable(unittest.TestCase):
    @unittest.skipUnless(AF2BIND_DIR.is_dir(), "AF2BIND_out/ not present")
    def test_threshold_028_proteome_default(self):
        csv_path = AF2BIND_DIR / "AF-Q9ZWT3-F1-model_v6_af2bind.csv"
        df = load_af2bind_table(str(csv_path))
        selected = select_af2bind_residues(df, threshold=0.28)
        self.assertGreater(len(selected), 0)
        self.assertTrue((selected["p(bind)"] >= 0.28).all())

    @unittest.skipUnless(AF2BIND_DIR.is_dir(), "AF2BIND_out/ not present")
    def test_notebook_top_n_mode(self):
        csv_path = AF2BIND_DIR / "AF-Q9ZWT3-F1-model_v6_af2bind.csv"
        df = load_af2bind_table(str(csv_path))
        top = select_af2bind_residues(df, top_n=15, threshold=None)
        self.assertEqual(len(top), 15)
        self.assertGreaterEqual(top["p(bind)"].iloc[0], top["p(bind)"].iloc[-1])


class TestClusteringAndBox(unittest.TestCase):
    def test_two_spatial_clusters(self):
        import numpy as np

        coords = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [40.0, 0.0, 0.0],
                [41.0, 0.0, 0.0],
                [40.0, 1.0, 0.0],
                [41.0, 1.0, 0.0],
                [40.0, 0.0, 1.0],
            ]
        )
        groups = single_linkage_clusters(list(range(10)), coords, cutoff=12.0)
        sizes = sorted(len(g) for g in groups)
        self.assertEqual(sizes, [5, 5])

    def test_pockets_from_synthetic_pdb(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "toy_af2bind.csv")
            with open(csv_path, "w", encoding="utf-8") as fh:
                fh.write("chain,resi,resn,p(bind)\n")
                # site 1 (5 residues, all >= 0.28)
                for resi in (10, 11, 12, 13, 14):
                    fh.write(f"A,{resi},ALA,0.{80 - (resi - 10)}\n")
                # site 2, far away
                for resi in (50, 51, 52, 53, 54):
                    fh.write(f"A,{resi},ALA,0.{55 - (resi - 50)}\n")
            pdb_path = os.path.join(tmp, "toy.pdb")
            _write_ca_pdb(
                pdb_path,
                [
                    ("A", 10, 0, 0, 0),
                    ("A", 11, 1, 0, 0),
                    ("A", 12, 0, 1, 0),
                    ("A", 13, 1, 1, 0),
                    ("A", 14, 0, 0, 1),
                    ("A", 50, 40, 0, 0),
                    ("A", 51, 41, 0, 0),
                    ("A", 52, 40, 1, 0),
                    ("A", 53, 41, 1, 0),
                    ("A", 54, 40, 0, 1),
                ],
            )
            pockets = pockets_from_af2bind(
                pdb_path,
                csv_path,
                threshold=0.28,
                cluster_cutoff=12.0,
                min_residues=5,
                pocket_mode="cluster",
            )
            self.assertEqual(len(pockets), 2)
            self.assertEqual(pockets[0].engine, "af2bind")
            self.assertGreaterEqual(pockets[0].score, pockets[1].score)

            cx, cy, cz, sx, sy, sz = calculate_box_size(
                pdb_path, pockets[0].residue_ids, margin=2.0
            )
            self.assertGreater(sx, 0)
            conf = os.path.join(tmp, "pocket.conf")
            write_vina_config(conf, os.path.join(tmp, "toy.pdbqt"), (cx, cy, cz), (sx, sy, sz))
            text = Path(conf).read_text(encoding="utf-8")
            self.assertIn("center_x", text)
            self.assertIn("receptor", text)

    def test_single_mode_keeps_one_pocket(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "toy_af2bind.csv")
            Path(csv_path).write_text(
                "chain,resi,resn,p(bind)\n"
                "A,10,ALA,0.80\nA,11,ALA,0.70\nA,50,ALA,0.55\nA,51,ALA,0.50\n",
                encoding="utf-8",
            )
            pdb_path = os.path.join(tmp, "toy.pdb")
            _write_ca_pdb(
                pdb_path,
                [
                    ("A", 10, 0, 0, 0),
                    ("A", 11, 1, 0, 0),
                    ("A", 50, 40, 0, 0),
                    ("A", 51, 41, 0, 0),
                ],
            )
            pockets = pockets_from_af2bind(
                pdb_path, csv_path, top_n=15, threshold=None, pocket_mode="single", min_residues=2
            )
            self.assertEqual(len(pockets), 1)
            self.assertEqual(len(pockets[0].residues), 4)


class TestBoxLimits(unittest.TestCase):
    def test_oversized_auto_box_is_capped_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            pdb_path = os.path.join(tmp, "wide.pdb")
            _write_ca_pdb(
                pdb_path,
                [("A", 1, 0, 0, 0), ("A", 2, 80, 0, 0)],
            )
            _, _, _, sx, sy, sz = calculate_box_size(
                pdb_path, "A_1 A_2", margin=2.0,
            )
            self.assertLessEqual(sx, DEFAULT_BOX_MAX_SIDE)
            self.assertLessEqual(sy, DEFAULT_BOX_MAX_SIDE)
            self.assertLessEqual(sz, DEFAULT_BOX_MAX_SIDE)
            self.assertLessEqual(sx * sy * sz, UNIDOCK_MAX_GRID_VOLUME)

    def test_cap_can_be_disabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            pdb_path = os.path.join(tmp, "wide.pdb")
            _write_ca_pdb(
                pdb_path,
                [("A", 1, 0, 0, 0), ("A", 2, 80, 0, 0)],
            )
            _, _, _, sx, _, _ = calculate_box_size(
                pdb_path, "A_1 A_2", margin=2.0,
                max_box_side=None, max_box_volume=None,
            )
            self.assertGreater(sx, DEFAULT_BOX_MAX_SIDE)


class TestPlddtFilter(unittest.TestCase):
    def test_removes_low_plddt_residues(self):
        with tempfile.TemporaryDirectory() as tmp:
            pdb_path = os.path.join(tmp, "plddt.pdb")
            with open(pdb_path, "w", encoding="utf-8") as fh:
                fh.write(
                    format_pdb_line(
                        "ATOM", 1, " CA ", " ", "ALA", "A", 1, " ",
                        0, 0, 0, 1.0, 90.0, "C",
                    )
                    + "\n"
                    + format_pdb_line(
                        "ATOM", 2, " CA ", " ", "ALA", "A", 2, " ",
                        5, 0, 0, 1.0, 40.0, "C",
                    )
                    + "\nEND\n"
                )
            out = os.path.join(tmp, "filtered.pdb")
            removed, kept = filter_pdb_by_plddt(pdb_path, out, threshold=70.0)
            self.assertEqual(removed, 1)
            self.assertEqual(kept, 1)
            text = Path(out).read_text(encoding="utf-8")
            atom_lines = [ln for ln in text.splitlines() if ln.startswith("ATOM")]
            self.assertEqual(len(atom_lines), 1)
            self.assertIn("A   1", atom_lines[0])


class TestCliHelp(unittest.TestCase):
    def test_unidock_help(self):
        from rvsauto.cli_unidock import build_parser

        parser = build_parser()
        help_text = parser.format_help()
        self.assertIn("--pocket_engine", help_text)
        self.assertIn("--af2bind_dir", help_text)

    def test_redock_help(self):
        from rvsauto.cli_redock import build_parser

        parser = build_parser()
        help_text = parser.format_help()
        self.assertIn("--input_dir", help_text)
        self.assertIn("--unidock_path", help_text)

    def test_dry_run_af2bind_cli(self):
        from rvsauto.cli_unidock import main

        with tempfile.TemporaryDirectory() as tmp:
            pdb_dir = Path(tmp) / "pdb"
            pred_dir = Path(tmp) / "pred"
            out_dir = Path(tmp) / "out"
            pdb_dir.mkdir()
            pred_dir.mkdir()
            (pred_dir / "toy_af2bind.csv").write_text(
                "chain,resi,resn,p(bind)\n"
                "A,10,ALA,0.80\nA,11,ALA,0.70\nA,12,ALA,0.60\nA,13,ALA,0.55\nA,14,ALA,0.50\n"
                "A,50,ALA,0.55\nA,51,ALA,0.50\nA,52,ALA,0.45\nA,53,ALA,0.40\nA,54,ALA,0.35\n",
                encoding="utf-8",
            )
            _write_ca_pdb(
                str(pdb_dir / "toy.pdb"),
                [
                    ("A", 10, 0, 0, 0),
                    ("A", 11, 1, 0, 0),
                    ("A", 12, 0, 1, 0),
                    ("A", 13, 1, 1, 0),
                    ("A", 14, 0, 0, 1),
                    ("A", 50, 40, 0, 0),
                    ("A", 51, 41, 0, 0),
                    ("A", 52, 40, 1, 0),
                    ("A", 53, 41, 1, 0),
                    ("A", 54, 40, 0, 1),
                ],
            )
            try:
                rc = main(
                    [
                        "--pdb_dir", str(pdb_dir),
                        "--output_dir", str(out_dir),
                        "--pocket_engine", "af2bind",
                        "--af2bind_dir", str(pred_dir),
                        "--dry_run",
                    ]
                )
                self.assertEqual(rc, 0)
                summary = out_dir / "results" / "pockets_summary.tsv"
                self.assertTrue(summary.is_file())
                text = summary.read_text(encoding="utf-8")
                self.assertIn("af2bind", text)
                confs = list((out_dir / "workdir" / "receptors_Pocket").glob("*.conf"))
                self.assertGreaterEqual(len(confs), 1)
            finally:
                root = logging.getLogger()
                for handler in list(root.handlers):
                    handler.close()
                    root.removeHandler(handler)


class TestP2RankProbabilityFilter(unittest.TestCase):
    def test_default_threshold_constant(self):
        self.assertEqual(P2RANK_DEFAULT_MIN_PROBABILITY, 0.05)
        self.assertIsNone(normalize_p2rank_min_probability(0))
        self.assertEqual(normalize_p2rank_min_probability(0.05), 0.05)

    def test_default_threshold_filters_weak_pockets(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "prot.pdb_predictions.csv"
            csv_path.write_text(
                "name,rank,score,probability,residue_ids\n"
                "pocket1,1,10.0,0.10,A_10\n"
                "pocket2,2,5.0,0.05,A_20\n"
                "pocket3,3,3.0,0.04,A_30\n",
                encoding="utf-8",
            )
            filtered = pockets_from_p2rank_csv(
                str(csv_path), min_probability=P2RANK_DEFAULT_MIN_PROBABILITY
            )
            self.assertEqual(len(filtered), 2)
            probs = [float(p.extra["probability"]) for p in filtered]
            self.assertIn(0.10, probs)
            self.assertIn(0.05, probs)

    def test_min_probability_filters_pockets(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "prot.pdb_predictions.csv"
            csv_path.write_text(
                "name,rank,score,probability,residue_ids\n"
                "pocket1,1,10.0,0.69,A_10 A_20\n"
                "pocket2,2,5.0,0.20,A_30 A_40\n"
                "pocket3,3,3.0,0.42,A_50 A_60\n",
                encoding="utf-8",
            )
            all_pockets = pockets_from_p2rank_csv(str(csv_path))
            self.assertEqual(len(all_pockets), 3)
            filtered = pockets_from_p2rank_csv(
                str(csv_path), min_probability=0.25
            )
            self.assertEqual(len(filtered), 2)
            self.assertAlmostEqual(float(filtered[0].extra["probability"]), 0.69)
            self.assertAlmostEqual(float(filtered[1].extra["probability"]), 0.42)
            self.assertEqual(filtered[0].pocket_index, 1)
            self.assertEqual(filtered[1].pocket_index, 2)

    def test_max_pockets_after_probability_filter(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "prot.pdb_predictions.csv"
            csv_path.write_text(
                "name,rank,score,probability,residue_ids\n"
                "pocket1,1,10.0,0.50,A_10\n"
                "pocket2,2,5.0,0.40,A_20\n"
                "pocket3,3,3.0,0.35,A_30\n",
                encoding="utf-8",
            )
            pockets = pockets_from_p2rank_csv(
                str(csv_path), min_probability=0.30, max_pockets=1
            )
            self.assertEqual(len(pockets), 1)
            self.assertAlmostEqual(float(pockets[0].extra["probability"]), 0.50)


class TestP2RankOutputResolution(unittest.TestCase):
    def test_finds_predict_subdir(self):
        with tempfile.TemporaryDirectory() as tmp:
            pocket_dir = Path(tmp)
            pred_dir = pocket_dir / "predict_receptors"
            pred_dir.mkdir()
            (pred_dir / "AF-X.pdb_predictions.csv").write_text(
                "rank,score,residue_ids\n1,1.0,A_10\n", encoding="utf-8"
            )
            resolved = resolve_p2rank_predictions_dir(
                str(pocket_dir), ["AF-X.pdb"]
            )
            self.assertEqual(resolved, str(pred_dir))

    def test_finds_flat_parent_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            pocket_dir = Path(tmp)
            (pocket_dir / "AF-Y.pdb_predictions.csv").write_text(
                "rank,score,residue_ids\n1,1.0,B_20\n", encoding="utf-8"
            )
            resolved = resolve_p2rank_predictions_dir(
                str(pocket_dir), ["AF-Y.pdb"]
            )
            self.assertEqual(resolved, str(pocket_dir))


if __name__ == "__main__":
    unittest.main()
