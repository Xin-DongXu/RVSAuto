"""CLI: batch UniDock virtual screening with P2Rank or AF2BIND pockets."""

from __future__ import annotations

import argparse
import logging
import os
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from . import __version__
from .common import (
    REPO_ROOT,
    file_stem,
    find_p2rank,
    find_unidock,
    resolve_gpu_list,
    run_command,
    set_quiet_subprocesses,
    setup_logging,
)
from .docking import (
    create_complex_pdb,
    extract_best_ligand_pose,
    receptor_pdb_to_pdbqt,
    run_unidock_batch,
    _sanitize_receptor_pdb,
)
from .pdbfix import repair_pdb_with_pdbfixer, verify_pdbfixer_env
from .pockets import (
    DEFAULT_BOX_MAX_SIDE,
    P2RANK_DATASET_NAME,
    UNIDOCK_MAX_GRID_VOLUME,
    index_af2bind_dir,
    match_af2bind_csv,
    pockets_from_af2bind,
    pockets_from_p2rank_csv,
    normalize_p2rank_min_probability,
    P2RANK_DEFAULT_MIN_PROBABILITY,
    resolve_p2rank_predictions_dir,
    write_pocket_configs,
    write_pocket_summary,
)
from .progress import (
    PipelineProgress,
    silence_console_logging,
    use_tqdm_safe_console_logging,
)
from .tables import parse_docking_label, write_docking_tables


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rvsauto-unidock",
        description=(
            "Batch GPU docking with UniDock. Receptors are converted to PDBQT, "
            "binding pockets are predicted with P2Rank or reconstructed from a "
            "folder of AF2BIND CSVs, and every ligand is docked into every pocket."
        ),
        formatter_class=type(
            "HelpFormatter",
            (argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter),
            {},
        ),
        epilog=(
            "Examples:\n"
            "  rvsauto-unidock --pdb_dir receptors --ligand_dir ligands "
            "--output_dir out --search_mode detail\n"
            "  rvsauto-unidock --pdb_dir receptors --ligand_dir ligands "
            "--output_dir out --pocket_engine af2bind --af2bind_dir AF2BIND_out\n"
        ),
    )

    g_io = p.add_argument_group("Input / Output")
    g_io.add_argument("--pdb_dir", required=True, help="Directory of receptor PDB files.")
    g_io.add_argument(
        "--ligand_dir",
        default=None,
        help="Directory of ligand PDBQT files (required unless --dry_run).",
    )
    g_io.add_argument(
        "--output_dir", required=True, help="Directory for workdir/, results/ and logs/."
    )

    g_pocket = p.add_argument_group("Pocket prediction")
    g_pocket.add_argument(
        "--pocket_engine",
        choices=["p2rank", "af2bind"],
        default="p2rank",
        help="How to obtain docking pockets.",
    )
    g_pocket.add_argument(
        "--p2rank_path",
        default=None,
        help="P2Rank launcher (prank). Auto-detected from ./p2rank_*/ next to this repo if omitted.",
    )
    g_pocket.add_argument(
        "--use_alphafold",
        action="store_true",
        help='Pass "-c alphafold" to P2Rank (recommended for AF2/AF3 models).',
    )
    g_pocket.add_argument(
        "--af2bind_dir",
        default=None,
        help="Folder of batch AF2BIND CSVs (*_af2bind.csv and/or *_af2bind_topN.csv).",
    )
    g_pocket.add_argument(
        "--af2bind_top_n",
        type=int,
        default=None,
        help="AF2BIND notebook mode: keep only the N highest p(bind) residues "
             "(overrides --af2bind_threshold). Example: --af2bind_top_n 15.",
    )
    g_pocket.add_argument(
        "--af2bind_threshold",
        type=float,
        default=0.28,
        help="AF2BIND: keep residues with p(bind) >= this value (proteome "
             "screening default 0.28). Ignored when --af2bind_top_n is set.",
    )
    g_pocket.add_argument(
        "--af2bind_pocket_mode",
        choices=["cluster", "single"],
        default="cluster",
        help="AF2BIND: 'single' = one pocket from the selected residues "
             "(official PyMOL selection); 'cluster' = split spatially "
             "separated residues into multiple pockets.",
    )
    g_pocket.add_argument(
        "--af2bind_cluster_cutoff",
        type=float,
        default=12.0,
        help="AF2BIND: CA-CA distance cutoff (A) for single-linkage clustering.",
    )
    g_pocket.add_argument(
        "--af2bind_min_residues",
        type=int,
        default=5,
        help="AF2BIND: minimum residues per clustered site (>4, proteome default 5).",
    )
    g_pocket.add_argument(
        "--max_pockets",
        type=int,
        default=None,
        help="Use at most this many pockets per protein (highest ranked first).",
    )
    g_pocket.add_argument(
        "--p2rank_min_probability",
        type=float,
        default=P2RANK_DEFAULT_MIN_PROBABILITY,
        help="P2Rank only: drop pockets whose calibrated probability is below "
        "this value (applied before --max_pockets). Default follows common "
        "screening practice (probability > 0.05). Set 0 to disable.",
    )

    g_box = p.add_argument_group("Docking box")
    g_box.add_argument(
        "--box_margin",
        type=float,
        default=2.0,
        help="Padding (A) added on each side of the pocket-residue bounding box.",
    )
    g_box.add_argument(
        "--box_size",
        type=float,
        default=None,
        help="Force a cubic box of this side length (A) for every pocket "
             "(still subject to --box_max_size / --box_max_volume caps).",
    )
    g_box.add_argument(
        "--box_max_size",
        type=float,
        default=DEFAULT_BOX_MAX_SIDE,
        help="Maximum side length (A) per box axis (default: %(default)s). "
             "Large boxes reduce docking accuracy. Set 0 to disable.",
    )
    g_box.add_argument(
        "--box_max_volume",
        type=float,
        default=UNIDOCK_MAX_GRID_VOLUME,
        help="Maximum box volume (A^3); UniDock limit is 27000 (default: %(default)s). "
             "Set 0 to disable.",
    )
    g_box.add_argument(
        "--rewrite_pocket_configs",
        action="store_true",
        help="Rewrite existing pocket .conf files (required after changing box settings).",
    )

    g_env = p.add_argument_group("Environments / binaries")
    g_env.add_argument(
        "--adt_env_path",
        default=None,
        help="Conda env with prepare_receptor4.py (name or prefix).",
    )
    g_env.add_argument(
        "--pdbfixer_env_path",
        default=None,
        help="Conda env with pdbfixer + openmm (conda-forge). On prepare_receptor4 "
             "failure, repair the cleaned PDB and retry once.",
    )
    g_env.add_argument(
        "--pdbfixer_ph",
        type=float,
        default=7.0,
        help="pH passed to PDBFixer addMissingHydrogens when repairing receptors.",
    )
    g_env.add_argument(
        "--dock_env_path",
        default=None,
        help="Conda env with the unidock binary (name or prefix).",
    )
    g_env.add_argument(
        "--unidock_path",
        default=None,
        help="UniDock executable. Auto-detected from PATH / ./Uni-Dock-main if omitted.",
    )

    g_receptor = p.add_argument_group("Receptor PDBQT repair")
    g_receptor.add_argument(
        "--plddt_retry_threshold",
        type=float,
        default=70.0,
        help="On prepare_receptor4 failure, remove residues with CA pLDDT "
             "(B-factor) below this value and retry ADT. Set 0 to disable "
             "(default: 70).",
    )

    g_dock = p.add_argument_group("Docking parameters")
    g_dock.add_argument(
        "--search_mode",
        default="detail",
        choices=["fast", "balance", "detail"],
        help="UniDock search mode.",
    )
    g_dock.add_argument("--num_modes", type=int, default=None, help="Optional --num_modes.")
    g_dock.add_argument(
        "--exhaustiveness", type=int, default=None, help="Optional --exhaustiveness."
    )
    g_dock.add_argument("--seed", type=int, default=None, help="Optional UniDock random seed.")

    g_par = p.add_argument_group("Parallel execution")
    g_par.add_argument(
        "--total_cpu",
        type=int,
        default=os.cpu_count() or 4,
        help="CPU workers for parallel receptor PDBQT conversion and P2Rank -threads.",
    )
    g_par.add_argument(
        "--gpu_ids",
        type=str,
        default=None,
        help='Comma-separated GPU IDs (e.g. "0,1"). Default: all visible GPUs.',
    )
    g_par.add_argument(
        "--max_parallel",
        type=int,
        default=None,
        help="Maximum concurrent UniDock jobs (default: number of GPUs).",
    )

    g_misc = p.add_argument_group("Miscellaneous")
    g_misc.add_argument(
        "--make_complex",
        action="store_true",
        help="Write receptor + best-pose complex PDBs.",
    )
    g_misc.add_argument(
        "--dry_run",
        action="store_true",
        help="Build pockets and .conf files only; do not run UniDock.",
    )
    g_misc.add_argument(
        "--strict_receptors",
        action="store_true",
        help="Abort when any receptor PDBQT conversion fails "
             "(default: log failures and continue with successful receptors).",
    )
    g_misc.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most this many receptor PDB files.",
    )
    g_misc.add_argument(
        "--no_progress",
        action="store_true",
        help="Disable the terminal progress bar (logs only).",
    )
    g_misc.add_argument(
        "--quiet",
        action="store_true",
        help=(
            "Suppress console logs and external-tool stdout/stderr so only the "
            "progress bar is shown; details still go to log files. "
            "End-of-run summary is kept."
        ),
    )
    g_misc.add_argument(
        "--version", action="version", version=f"RVSAuto UniDock pipeline {__version__}"
    )
    return p


def _adt_log_tail(adt_log: str) -> str:
    if not os.path.isfile(adt_log):
        return ""
    try:
        text = Path(adt_log).read_text(encoding="utf-8", errors="replace")
        lines = [ln for ln in text.splitlines() if ln.strip()]
        if lines:
            return " | log tail: " + " / ".join(lines[-3:])
    except OSError:
        pass
    return ""


def _convert_one(
    pdb_file: str,
    pdbqt_dir: str,
    clean_dir: str,
    adt_env_path,
    pdbfixer_env_path=None,
    pdbfixer_ph: float = 7.0,
    plddt_retry_threshold: float = 70.0,
) -> Optional[str]:
    """Convert one receptor; return an error message or None on success."""
    stem = file_stem(pdb_file)
    out = os.path.join(pdbqt_dir, f"{stem}.pdbqt")
    keep = os.path.join(clean_dir, f"{stem}.pdb")
    fixed = os.path.join(clean_dir, f"{stem}_pdbfixer.pdb")

    def run_prepare(
        src: str,
        log_suffix: str,
        *,
        clean: bool,
        keep_path: Optional[str],
        adt_repairs: str = "hydrogens",
    ) -> None:
        logging.info(
            "Receptor %s: prepare_receptor4 -A %s on %s ...",
            stem, adt_repairs, os.path.basename(src),
        )
        adt_log = os.path.join(pdbqt_dir, f"{stem}_prepare_receptor{log_suffix}.log")
        receptor_pdb_to_pdbqt(
            src,
            out,
            adt_env_path=adt_env_path,
            clean=clean,
            keep_clean_pdb=keep_path,
            adt_log=adt_log,
            adt_repairs=adt_repairs,
        )
        if not os.path.exists(out) or os.path.getsize(out) == 0:
            adt_log = os.path.join(pdbqt_dir, f"{stem}_prepare_receptor{log_suffix}.log")
            raise RuntimeError(f"empty or missing PDBQT: {out}{_adt_log_tail(adt_log)}")

    try:
        run_prepare(pdb_file, "", clean=True, keep_path=keep)
        return None
    except Exception as exc:
        first_err = str(exc)
        adt_log = os.path.join(pdbqt_dir, f"{stem}_prepare_receptor.log")
        logging.error(
            "Receptor PDBQT failed for %s: %s%s", stem, exc, _adt_log_tail(adt_log)
        )

    if os.path.isfile(keep):
        _sanitize_receptor_pdb(keep)
    try:
        if os.path.isfile(out):
            os.remove(out)
        logging.info("Receptor %s: retry prepare_receptor4 with bonds_hydrogens ...", stem)
        run_prepare(keep, "_bonds", clean=False, keep_path=None, adt_repairs="bonds_hydrogens")
        logging.info("Receptor %s: PDBQT conversion succeeded after bonds_hydrogens.", stem)
        return None
    except Exception as exc_bonds:
        adt_log = os.path.join(pdbqt_dir, f"{stem}_prepare_receptor_bonds.log")
        logging.warning(
            "Receptor %s: bonds_hydrogens retry failed: %s%s",
            stem, exc_bonds, _adt_log_tail(adt_log),
        )

    if plddt_retry_threshold > 0:
        from .pdbio import clean_pdb, filter_pdb_by_plddt

        plddt_pdb = os.path.join(clean_dir, f"{stem}_plddt.pdb")
        if not os.path.isfile(keep):
            clean_pdb(pdb_file, keep)
        removed, kept = filter_pdb_by_plddt(
            keep, plddt_pdb, threshold=plddt_retry_threshold,
        )
        if kept == 0:
            logging.warning(
                "Receptor %s: pLDDT filter removed all residues; skip pLDDT retry.",
                stem,
            )
        else:
            _sanitize_receptor_pdb(plddt_pdb)
            for repairs, suffix in (("hydrogens", "_plddt"), ("bonds_hydrogens", "_plddt_bonds")):
                try:
                    if os.path.isfile(out):
                        os.remove(out)
                    logging.info(
                        "Receptor %s: retry after pLDDT>=%.0f filter (%d removed) "
                        "with -A %s ...",
                        stem, plddt_retry_threshold, removed, repairs,
                    )
                    run_prepare(
                        plddt_pdb, suffix, clean=False, keep_path=None, adt_repairs=repairs,
                    )
                    logging.info(
                        "Receptor %s: PDBQT succeeded after pLDDT filter + %s.",
                        stem, repairs,
                    )
                    return None
                except Exception as exc_plddt:
                    adt_log = os.path.join(
                        pdbqt_dir, f"{stem}_prepare_receptor{suffix}.log"
                    )
                    logging.warning(
                        "Receptor %s: pLDDT retry (%s) failed: %s%s",
                        stem, repairs, exc_plddt, _adt_log_tail(adt_log),
                    )

    if not pdbfixer_env_path:
        logging.warning("Receptor %s: skipping (no PDBFixer env).", stem)
        return first_err

    if not os.path.isfile(keep):
        from .pdbio import clean_pdb

        clean_pdb(pdb_file, keep)

    logging.info("Receptor %s: trying PDBFixer repair ...", stem)
    ok, msg = repair_pdb_with_pdbfixer(
        keep, fixed, env_path=pdbfixer_env_path, ph=pdbfixer_ph
    )
    if not ok:
        logging.error("PDBFixer failed for %s: %s", stem, msg)
        return f"{first_err} | pdbfixer: {msg}"

    _sanitize_receptor_pdb(fixed)
    try:
        if os.path.isfile(out):
            os.remove(out)
        run_prepare(fixed, "_after_pdbfixer", clean=False, keep_path=None)
        logging.info("Receptor %s: PDBQT conversion succeeded after PDBFixer.", stem)
        return None
    except Exception as exc2:
        adt_log = os.path.join(
            pdbqt_dir, f"{stem}_prepare_receptor_after_pdbfixer.log"
        )
        logging.error(
            "Receptor %s still failed after PDBFixer: %s%s",
            stem, exc2, _adt_log_tail(adt_log),
        )
        try:
            if os.path.isfile(out):
                os.remove(out)
            logging.info(
                "Receptor %s: retry pdbfixer output with bonds_hydrogens ...", stem
            )
            run_prepare(
                fixed,
                "_after_pdbfixer_bonds",
                clean=False,
                keep_path=None,
                adt_repairs="bonds_hydrogens",
            )
            logging.info(
                "Receptor %s: PDBQT succeeded after PDBFixer + bonds_hydrogens.", stem
            )
            return None
        except Exception as exc3:
            adt_log3 = os.path.join(
                pdbqt_dir, f"{stem}_prepare_receptor_after_pdbfixer_bonds.log"
            )
            logging.warning("Receptor %s: skipping after all repair attempts.", stem)
            return (
                f"{first_err} | after pdbfixer: {exc2}{_adt_log_tail(adt_log)}"
                f" | pdbfixer+bonds: {exc3}{_adt_log_tail(adt_log3)}"
            )


def _write_receptor_failures(results_dir: str, failures: list) -> None:
    if not failures:
        return
    path = os.path.join(results_dir, "receptor_pdbqt_failures.tsv")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("protein_id\terror\n")
        for stem, err in failures:
            fh.write(f"{stem}\t{err.replace(chr(9), ' ')}\n")
    logging.warning(
        "Wrote %d receptor conversion failure(s) to %s", len(failures), path
    )


def _filter_pockets_with_pdbqt(pockets, pdbqt_dir: str):
    kept = []
    skipped = []
    for pocket in pockets:
        pdbqt = os.path.join(pdbqt_dir, f"{pocket.protein_id}.pdbqt")
        if os.path.isfile(pdbqt) and os.path.getsize(pdbqt) > 0:
            kept.append(pocket)
        else:
            skipped.append(pocket.protein_id)
    if skipped:
        uniq = sorted(set(skipped))
        logging.warning(
            "Skipping %d pocket(s) from %d protein(s) without valid PDBQT: %s%s",
            len(skipped), len(uniq),
            ", ".join(uniq[:10]),
            " ..." if len(uniq) > 10 else "",
        )
    return kept


def _p2rank_output_dir(pocket_dir: str) -> str:
    """Default P2Rank output subdir for ``receptors.ds``."""
    return os.path.join(pocket_dir, f"predict_{P2RANK_DATASET_NAME}")


def _run_p2rank(
    pdb_files, pdb_dir, pocket_dir, p2rank_path, use_alphafold, threads, dock_env_path
) -> str:
    """Run P2Rank and return the directory that holds *_predictions.csv files."""
    p2rank_out = resolve_p2rank_predictions_dir(pocket_dir, pdb_files)
    expected = [
        os.path.join(p2rank_out, f"{file_stem(p)}.pdb_predictions.csv")
        for p in pdb_files
    ]
    if expected and all(os.path.exists(f) for f in expected):
        logging.info("P2Rank outputs already present; skip prediction.")
        return p2rank_out

    launcher = find_p2rank(p2rank_path)
    if launcher is None:
        raise FileNotFoundError(
            "P2Rank launcher not found. Place p2rank_* next to this repository "
            "or pass --p2rank_path. Download: https://github.com/rdk/p2rank/releases"
        )

    list_path = os.path.join(pocket_dir, f"{P2RANK_DATASET_NAME}.ds")
    with open(list_path, "w", encoding="utf-8") as fh:
        for pdb_file in pdb_files:
            fh.write(os.path.abspath(os.path.join(pdb_dir, pdb_file)) + "\n")

    cfg = "-c alphafold " if use_alphafold else ""
    if os.name == "nt" or launcher.suffix.lower() == ".bat":
        invoke = f'"{launcher}"'
    else:
        invoke = f'bash "{launcher}"'
    cmd = (
        f'{invoke} predict {cfg}"{list_path}" -o "{pocket_dir}" '
        f"-threads {threads} -visualizations 0"
    )
    logging.info("Running P2Rank: %s", cmd)
    logging.info("Expected P2Rank output directory: %s", _p2rank_output_dir(pocket_dir))
    # P2Rank needs Java on PATH / JAVA_HOME — not the UniDock conda env.
    run_command(cmd, env_path=None)

    p2rank_out = resolve_p2rank_predictions_dir(pocket_dir, pdb_files)
    if not os.path.isdir(p2rank_out):
        raise FileNotFoundError(
            f"P2Rank finished but no output directory found under {pocket_dir}. "
            "Check P2Rank logs in that folder."
        )
    missing = [
        os.path.join(p2rank_out, f"{file_stem(p)}.pdb_predictions.csv")
        for p in pdb_files
        if not os.path.exists(
            os.path.join(p2rank_out, f"{file_stem(p)}.pdb_predictions.csv")
        )
    ]
    if missing:
        run_log = os.path.join(p2rank_out, "run.log")
        if not os.path.exists(run_log):
            run_log = os.path.join(pocket_dir, "run.log")
        hint = f" See {run_log} for details." if os.path.exists(run_log) else ""
        listing = ""
        if os.path.isdir(pocket_dir):
            csvs = [
                f
                for f in os.listdir(pocket_dir)
                if f.endswith(".pdb_predictions.csv")
            ]
            if csvs:
                listing = f" CSV(s) in parent folder: {csvs[:5]}."
        raise FileNotFoundError(
            f"P2Rank produced no predictions for {len(missing)}/{len(pdb_files)} "
            f"structure(s). Resolved output dir: {p2rank_out}. "
            f"First missing: {missing[0]}.{hint}{listing}"
        )
    return p2rank_out


def _collect_p2rank_pockets(
    p2rank_out_dir: str,
    pdb_stems,
    max_pockets,
    min_probability=None,
):
    pockets = []
    for stem in pdb_stems:
        csv_path = os.path.join(p2rank_out_dir, f"{stem}.pdb_predictions.csv")
        if not os.path.exists(csv_path):
            logging.warning("P2Rank CSV missing for %s: %s", stem, csv_path)
            continue
        pockets.extend(
            pockets_from_p2rank_csv(
                csv_path,
                max_pockets=max_pockets,
                min_probability=min_probability,
            )
        )
    return pockets


def _collect_af2bind_pockets(pdb_dir, pdb_files, args, progress: PipelineProgress | None = None):
    if not args.af2bind_dir:
        raise SystemExit(
            "--pocket_engine af2bind requires --af2bind_dir pointing to a folder "
            "of AF2BIND CSV files (*_af2bind.csv / *_af2bind_top15.csv)."
        )
    index = index_af2bind_dir(args.af2bind_dir, recursive=True)
    if not index:
        raise SystemExit(f"No AF2BIND CSV files found in {args.af2bind_dir}")
    logging.info("Indexed %d AF2BIND key(s) from %s", len(index), args.af2bind_dir)

    pockets = []
    unmatched = []
    for pdb_file in pdb_files:
        pdb_path = os.path.join(pdb_dir, pdb_file)
        csv_path = match_af2bind_csv(pdb_path, index)
        if csv_path is None:
            unmatched.append(pdb_file)
        else:
            logging.debug(
                "AF2BIND %s <- %s", pdb_file, os.path.basename(csv_path)
            )
            pockets.extend(
                pockets_from_af2bind(
                    pdb_path,
                    csv_path,
                    top_n=args.af2bind_top_n,
                    threshold=args.af2bind_threshold,
                    cluster_cutoff=args.af2bind_cluster_cutoff,
                    min_residues=args.af2bind_min_residues,
                    pocket_mode=args.af2bind_pocket_mode,
                    max_pockets=args.max_pockets,
                )
            )
        if progress is not None:
            progress.step(1, phase="AF2BIND pockets")
    if unmatched:
        logging.warning(
            "%d receptor(s) have no matching AF2BIND CSV: %s",
            len(unmatched),
            ", ".join(unmatched[:10]) + (" ..." if len(unmatched) > 10 else ""),
        )
    return pockets


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    pdb_dir = os.path.abspath(args.pdb_dir)
    ligand_dir = os.path.abspath(args.ligand_dir) if args.ligand_dir else ""
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    log_file = setup_logging(output_dir, prefix="docking_run")
    progress = PipelineProgress(enabled=not args.no_progress)
    if args.quiet:
        silence_console_logging()
        set_quiet_subprocesses(True)
    elif progress.enabled:
        use_tqdm_safe_console_logging()
        set_quiet_subprocesses(True)
    logging.info("RVSAuto UniDock pipeline v%s", __version__)
    logging.info("Arguments: %s", vars(args))
    logging.info("Repository root: %s", REPO_ROOT)

    gpu_list, num_gpus = ["0"], 1
    parallel_tasks = 1
    unidock_bin = "unidock"
    if not args.dry_run:
        gpu_list, num_gpus = resolve_gpu_list(args.gpu_ids)
        logging.info("GPUs: %s (n=%d)", gpu_list, num_gpus)
        parallel_tasks = args.max_parallel if args.max_parallel else num_gpus
        unidock_bin = find_unidock(args.unidock_path)
        logging.info("UniDock binary: %s", unidock_bin)

    ligand_files = []
    if ligand_dir and os.path.isdir(ligand_dir):
        ligand_files = sorted(
            os.path.join(ligand_dir, f)
            for f in os.listdir(ligand_dir)
            if f.lower().endswith(".pdbqt")
        )
    if not ligand_files:
        if args.dry_run:
            logging.warning(
                "No ligand PDBQT files (allowed with --dry_run)."
            )
        else:
            logging.error(
                "No PDBQT ligands found. Pass --ligand_dir or use --dry_run "
                "to only build pocket configs."
            )
            return 1
    else:
        logging.info(
            "Found %d ligand(s): %s",
            len(ligand_files),
            [os.path.basename(f) for f in ligand_files],
        )

    pdb_files = sorted(f for f in os.listdir(pdb_dir) if f.lower().endswith(".pdb"))
    if args.limit:
        pdb_files = pdb_files[: args.limit]
    if not pdb_files:
        logging.error("No PDB files in %s", pdb_dir)
        return 1
    logging.info("Found %d receptor PDB(s).", len(pdb_files))

    workdir = os.path.join(output_dir, "workdir")
    results_dir = os.path.join(output_dir, "results")
    pdbqt_dir = os.path.join(workdir, "receptors_PDBQT")
    pocket_dir = os.path.join(workdir, "receptors_Pocket")
    clean_dir = os.path.join(workdir, "receptors_clean")
    docking_root = os.path.join(workdir, "docking")
    for d in (workdir, results_dir, pdbqt_dir, pocket_dir, clean_dir, docking_root):
        os.makedirs(d, exist_ok=True)

    pdb_paths = [os.path.join(pdb_dir, f) for f in pdb_files]
    planned_units = 0
    if not args.dry_run:
        planned_units += len(pdb_paths)
    if args.pocket_engine == "af2bind":
        planned_units += len(pdb_files)
    else:
        planned_units += 1
    if planned_units > 0:
        progress.begin(planned_units)

    receptor_failures = []
    if not args.dry_run:
        if args.pdbfixer_env_path:
            pdbfixer_err = verify_pdbfixer_env(args.pdbfixer_env_path)
            if pdbfixer_err:
                logging.warning(
                    "PDBFixer env check failed (repair retries will not work): %s",
                    pdbfixer_err,
                )
        logging.info(
            "Converting %d receptor(s) to PDBQT with %d worker(s).",
            len(pdb_paths), max(1, args.total_cpu),
        )
        if args.plddt_retry_threshold > 0:
            logging.info(
                "Failed receptors will retry after pLDDT filter (threshold %.1f).",
                args.plddt_retry_threshold,
            )
        with ThreadPoolExecutor(max_workers=max(1, args.total_cpu)) as pool:
            futures = {
                pool.submit(
                    _convert_one,
                    p,
                    pdbqt_dir,
                    clean_dir,
                    args.adt_env_path,
                    args.pdbfixer_env_path,
                    args.pdbfixer_ph,
                    args.plddt_retry_threshold,
                ): p
                for p in pdb_paths
            }
            errors = []
            for fut in as_completed(futures):
                pdb_path = futures[fut]
                try:
                    err = fut.result()
                except Exception as exc:
                    err = str(exc)
                errors.append((pdb_path, err))
                progress.step(1, phase="Receptor PDBQT")
        for pdb_path, err in errors:
            if err:
                receptor_failures.append((file_stem(pdb_path), err))
        ok = len(pdb_paths) - len(receptor_failures)
        logging.info(
            "Receptor PDBQT conversion: %d ok, %d failed.", ok, len(receptor_failures)
        )
        _write_receptor_failures(results_dir, receptor_failures)
        if receptor_failures and args.strict_receptors:
            logging.error(
                "--strict_receptors set; aborting after receptor conversion failures."
            )
            progress.close()
            return 1
    else:
        logging.info("Dry run: skipping receptor PDBQT conversion.")

    if args.pocket_engine == "p2rank":
        progress.step(0, phase="P2Rank")
        p2rank_out = _run_p2rank(
            pdb_files,
            pdb_dir,
            pocket_dir,
            args.p2rank_path,
            args.use_alphafold,
            args.total_cpu,
            args.dock_env_path,
        )
        progress.step(1, phase="P2Rank")
        pockets = _collect_p2rank_pockets(
            p2rank_out,
            [file_stem(f) for f in pdb_files],
            args.max_pockets,
            min_probability=normalize_p2rank_min_probability(
                args.p2rank_min_probability
            ),
        )
    else:
        pockets = _collect_af2bind_pockets(
            pdb_dir, pdb_files, args, progress=progress
        )

    if not args.dry_run:
        progress.add_tasks(1)
        progress.step(0, phase="Filter pockets")
        pockets = _filter_pockets_with_pdbqt(pockets, pdbqt_dir)
        progress.step(1, phase="Filter pockets")

    if not pockets:
        logging.error("No pockets were generated; aborting.")
        progress.close()
        return 1
    logging.info(
        "Prepared %d pocket(s) via %s", len(pockets), args.pocket_engine
    )

    if args.box_size is not None:
        logging.info("Fixed cubic box: %.1f A", args.box_size)
    max_side = args.box_max_size if args.box_max_size > 0 else None
    max_volume = args.box_max_volume if args.box_max_volume > 0 else None
    if max_side is not None:
        logging.info("Docking box per-axis cap: %.1f A", max_side)
    if max_volume is not None:
        logging.info("Docking box volume cap: %.0f A^3", max_volume)

    progress.add_tasks(len(pockets))
    progress.step(0, phase="Pocket configs")
    summary_rows = write_pocket_configs(
        pockets,
        pdb_dir,
        pdbqt_dir,
        pocket_dir,
        margin=args.box_margin,
        fixed_box_size=args.box_size,
        max_box_side=max_side,
        max_box_volume=max_volume,
        skip_existing=not args.rewrite_pocket_configs,
        require_pdbqt=not args.dry_run,
        progress=progress,
    )
    summary_tsv = os.path.join(results_dir, "pockets_summary.tsv")
    write_pocket_summary(summary_rows, summary_tsv)
    logging.info("Pocket summary: %s", summary_tsv)

    if args.dry_run:
        logging.info("Dry run: skipping UniDock. Configs are in %s", pocket_dir)
        progress.close()
        print(f"Dry run complete. Pocket configs: {pocket_dir}")
        print(f"Pocket summary: {summary_tsv}")
        return 0

    ligand_output_dirs = {}
    complex_dir = os.path.join(results_dir, "complexes")
    if args.make_complex:
        os.makedirs(complex_dir, exist_ok=True)
    for ligand_path in ligand_files:
        ligand_name = file_stem(ligand_path)
        docking_output_dir = os.path.join(docking_root, ligand_name)
        os.makedirs(docking_output_dir, exist_ok=True)
        ligand_output_dirs[ligand_name] = {"docking": docking_output_dir}

    all_affinities = {file_stem(lig): [] for lig in ligand_files}
    conf_files = sorted(f for f in os.listdir(pocket_dir) if f.endswith(".conf"))
    logging.info("Found %d configuration file(s).", len(conf_files))
    progress.add_tasks(len(conf_files))
    progress.step(0, phase="UniDock docking")
    missing_outputs = []
    gpu_cycle = 0

    with ThreadPoolExecutor(max_workers=max(1, parallel_tasks)) as executor:
        future_to_conf = {}
        for file in conf_files:
            conf_file = os.path.join(pocket_dir, file)
            base_filename = os.path.splitext(file)[0]
            task_gpu = gpu_list[gpu_cycle % num_gpus]
            gpu_cycle += 1
            output_pdbqt_files = {}
            for lig_path in ligand_files:
                lig_name = file_stem(lig_path)
                output_pdbqt_files[lig_path] = os.path.join(
                    ligand_output_dirs[lig_name]["docking"],
                    f"{base_filename}_{lig_name}_out.pdbqt",
                )
            log_file_path = os.path.join(
                workdir, "multi_ligand_logs", f"{base_filename}_multi_ligand.log"
            )
            future = executor.submit(
                run_unidock_batch,
                conf_file,
                ligand_files,
                output_pdbqt_files,
                log_file_path,
                args.search_mode,
                task_gpu,
                args.dock_env_path,
                unidock_bin,
                args.num_modes,
                args.exhaustiveness,
                args.seed,
            )
            future_to_conf[future] = base_filename

        for future in as_completed(future_to_conf):
            try:
                results = future.result()
                if results:
                    for output_file, affinity, lig_name in results:
                        filename = future_to_conf[future]
                        all_affinities[lig_name].append(
                            (f"{filename}_{lig_name}", affinity)
                        )
                else:
                    logging.warning("No results for %s", future_to_conf[future])
            except Exception as exc:
                logging.error("Task failed for %s: %s", future_to_conf[future], exc)
                logging.error(traceback.format_exc())
            progress.step(1, phase="UniDock docking")

    progress.close()

    for ligand_path in ligand_files:
        ligand_name = file_stem(ligand_path)
        affinities = all_affinities[ligand_name]
        logging.info(
            "Results for ligand %s: %d docking(s)", ligand_name, len(affinities)
        )
        docking_output_dir = ligand_output_dirs[ligand_name]["docking"]

        docking_result_file = os.path.join(
            results_dir, f"{ligand_name}_Docking_Result.tsv"
        )
        with open(docking_result_file, "w", encoding="utf-8") as result_file:
            result_file.write("Filename\tBinding Affinity (kcal/mol)\n")
            for filename, affinity in affinities:
                result_file.write(
                    f"{filename.replace('config_', '')}\t{affinity}\n"
                )

        if args.make_complex:
            logging.info("Generating complex PDBs for %s", ligand_name)
            complex_count = 0
            for filename, affinity in affinities:
                parsed = parse_docking_label(filename, ligand_name)
                if parsed is None:
                    continue
                receptor_id, pocket_id = parsed
                apo_pdb = os.path.join(clean_dir, f"{receptor_id}.pdb")
                if not os.path.exists(apo_pdb):
                    logging.warning(
                        "Cleaned receptor PDB missing (%s); skip complex.", apo_pdb
                    )
                    continue
                docking_pdbqt = os.path.join(
                    docking_output_dir, f"{filename}_out.pdbqt"
                )
                if not os.path.exists(docking_pdbqt):
                    logging.warning("Docking output missing: %s", docking_pdbqt)
                    missing_outputs.append(docking_pdbqt)
                    continue
                temp_ligand_pdb = os.path.join(
                    complex_dir, f"{receptor_id}_{pocket_id}_{ligand_name}_ligand_temp.pdb"
                )
                complex_pdb = os.path.join(
                    complex_dir,
                    f"{receptor_id}_{pocket_id}_{ligand_name}_complex.pdb",
                )
                if extract_best_ligand_pose(docking_pdbqt, temp_ligand_pdb):
                    if create_complex_pdb(apo_pdb, temp_ligand_pdb, complex_pdb):
                        complex_count += 1
                    if os.path.exists(temp_ligand_pdb):
                        os.remove(temp_ligand_pdb)
            logging.info(
                "Created %d complex structure(s) for %s in %s",
                complex_count, ligand_name, complex_dir,
            )

        best_affinities = {}
        for filename, affinity in affinities:
            parsed = parse_docking_label(filename, ligand_name)
            if parsed is None:
                continue
            protein_id, pocket_info = parsed
            if protein_id not in best_affinities or affinity < best_affinities[protein_id][1]:
                best_affinities[protein_id] = (
                    filename.replace("config_", ""),
                    affinity,
                    pocket_info,
                )

        best_result_file = os.path.join(
            results_dir, f"{ligand_name}_Best_Docking_Result.tsv"
        )
        with open(best_result_file, "w", encoding="utf-8") as result_file:
            result_file.write(
                "Protein_ID\tBest_Pocket\tBinding Affinity (kcal/mol)\t"
                "Full_Filename\tComplex_PDB\n"
            )
            for protein_id, (full_filename, affinity, pocket_info) in sorted(
                best_affinities.items(), key=lambda x: x[1][1]
            ):
                complex_name = (
                    f"complexes/{protein_id}_{pocket_info}_{ligand_name}_complex.pdb"
                    if args.make_complex else ""
                )
                result_file.write(
                    f"{protein_id}\t{pocket_info}\t{affinity}\t"
                    f"{full_filename}\t{complex_name}\n"
                )
        if not args.quiet:
            print(f"Completed ligand: {ligand_name}")
            print(f"  All results : {docking_result_file}")
            print(f"  Best results: {best_result_file}")
        logging.info(
            "Completed ligand %s: all=%s best=%s",
            ligand_name,
            docking_result_file,
            best_result_file,
        )

    ligand_names = [file_stem(p) for p in ligand_files]
    table_paths = write_docking_tables(
        all_affinities,
        ligand_names,
        results_dir,
        complex_relpath="complexes" if args.make_complex else None,
    )
    logging.info("Affinity matrix (pockets x ligands): %s", table_paths["matrix_pockets"])
    logging.info("Affinity matrix (proteins x ligands): %s", table_paths["matrix_proteins"])

    if missing_outputs:
        missing_log = os.path.join(results_dir, "missing_outputs.log")
        with open(missing_log, "w", encoding="utf-8") as fh:
            fh.write("Missing output files:\n")
            for path in missing_outputs:
                fh.write(f"{path}\n")
        logging.warning(
            "%d output file(s) missing; list saved to %s",
            len(missing_outputs), missing_log,
        )

    print("=" * 60)
    print("Mission accomplished.")
    print(f"Working files : {workdir}")
    print(f"Final results : {results_dir}")
    print(f"  Pocket x ligand matrix : {table_paths['matrix_pockets']}")
    print(f"  Protein x ligand matrix: {table_paths['matrix_proteins']}")
    print(f"Log file      : {log_file}")
    print("=" * 60)
    logging.info("Pipeline completed successfully.")
    return 0
