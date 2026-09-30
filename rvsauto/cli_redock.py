"""CLI: batch self-redocking + heavy-atom RMSD with UniDock."""

from __future__ import annotations

import argparse
import glob
import logging
import os
import shutil
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from . import __version__
from .common import (
    REPO_ROOT,
    file_stem,
    find_unidock,
    resolve_gpu_list,
    set_quiet_subprocesses,
    setup_logging,
)
from .docking import (
    create_complex_pdb,
    extract_best_ligand_pose,
    ligand_pdb_to_pdbqt,
    receptor_pdb_to_pdbqt,
    run_unidock_single,
    _sanitize_receptor_pdb,
)
from .pdbio import split_complex
from .pockets import write_vina_config
from .progress import (
    PipelineProgress,
    silence_console_logging,
    use_tqdm_safe_console_logging,
)
from .rmsd import heavy_atom_rmsd


def box_from_ligand_pdb(ligand_pdb, padding=4.0, fixed_box_size=None):
    coords = []
    with open(ligand_pdb, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith(("ATOM", "HETATM")):
                try:
                    coords.append(
                        (
                            float(line[30:38]),
                            float(line[38:46]),
                            float(line[46:54]),
                        )
                    )
                except (ValueError, IndexError):
                    continue
    if not coords:
        raise RuntimeError(f"no atoms in ligand PDB: {ligand_pdb}")

    arr = np.asarray(coords)
    mins = arr.min(axis=0)
    maxs = arr.max(axis=0)
    center = (mins + maxs) / 2.0
    if fixed_box_size is not None:
        size = np.array([fixed_box_size] * 3, dtype=float)
    else:
        size = np.maximum((maxs - mins) + 2 * padding, 15.0)
    return (
        (round(float(center[0]), 4), round(float(center[1]), 4), round(float(center[2]), 4)),
        (round(float(size[0]), 1), round(float(size[1]), 1), round(float(size[2]), 1)),
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rvsauto-redock",
        description=(
            "Self-redocking + RMSD pipeline for batch PDB / CIF complexes "
            "using UniDock. For every input complex the native ligand is "
            "extracted, the apo-protein is rebuilt, the ligand is re-docked "
            "into its own pocket, the lowest-affinity pose is combined with "
            "the apo-protein, and the heavy-atom RMSD to the native pose is "
            "reported."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g_io = p.add_argument_group("Input / Output")
    g_io.add_argument(
        "--input_dir",
        required=True,
        help="Directory holding the input PDB / CIF complex files.",
    )
    g_io.add_argument(
        "--output_dir",
        required=True,
        help="Directory where workdir/, results/ and logs/ will be created.",
    )
    g_io.add_argument(
        "--file_glob",
        default="*.pdb,*.cif,*.pdb.gz,*.cif.gz",
        help="Comma-separated glob patterns for input selection.",
    )

    g_env = p.add_argument_group("Environments / binaries")
    g_env.add_argument(
        "--adt_env_path",
        default=None,
        help="Conda env with prepare_receptor4.py / prepare_ligand4.py / obabel.",
    )
    g_env.add_argument(
        "--dock_env_path",
        default=None,
        help="Conda env with the unidock executable.",
    )
    g_env.add_argument(
        "--unidock_path",
        default=None,
        help="UniDock executable. Auto-detected from PATH / ./Uni-Dock-main if omitted.",
    )

    g_lig = p.add_argument_group("Ligand selection")
    g_lig.add_argument(
        "--ligand_resname",
        default=None,
        help="Force the ligand to be the residue with this 3-letter name (e.g. ATP).",
    )
    g_lig.add_argument(
        "--min_ligand_heavy_atoms",
        type=int,
        default=6,
        help="Residues with fewer heavy atoms than this are skipped.",
    )
    g_lig.add_argument(
        "--ligand_pdbqt",
        default=None,
        help=(
            "Path to a pre-built ligand PDBQT (e.g. ethylene). "
            "Copied into each case directory; skips ligand conversion."
        ),
    )

    g_box = p.add_argument_group("Docking box")
    g_box.add_argument(
        "--box_padding",
        type=float,
        default=4.0,
        help="Padding (A) added on each side of the native ligand bounding box.",
    )
    g_box.add_argument(
        "--box_size",
        type=float,
        default=None,
        help="Force a uniform cubic box of this side length (A).",
    )

    g_dock = p.add_argument_group("Docking parameters")
    g_dock.add_argument(
        "--search_mode",
        default="detail",
        choices=["fast", "balance", "detail"],
        help="UniDock search mode.",
    )
    g_dock.add_argument(
        "--num_modes", type=int, default=9, help="Number of poses per ligand."
    )
    g_dock.add_argument(
        "--exhaustiveness", type=int, default=None, help="Optional exhaustiveness override."
    )
    g_dock.add_argument("--seed", type=int, default=42, help="Random seed.")

    g_par = p.add_argument_group("Parallel execution")
    g_par.add_argument(
        "--total_cpu",
        type=int,
        default=os.cpu_count() or 4,
        help="CPU workers for preprocessing.",
    )
    g_par.add_argument(
        "--gpu_ids", type=str, default=None, help="Comma-separated GPU IDs (e.g. '0,1,2')."
    )
    g_par.add_argument(
        "--max_parallel",
        type=int,
        default=None,
        help="Maximum number of parallel UniDock jobs.",
    )

    g_misc = p.add_argument_group("Miscellaneous")
    g_misc.add_argument(
        "--make_complex",
        action="store_true",
        help="Also write a redocked complex PDB (apo-protein + best pose).",
    )
    g_misc.add_argument(
        "--symmetry_aware_rmsd",
        action="store_true",
        help="If RDKit is installed, also compute a symmetry-aware RMSD.",
    )
    g_misc.add_argument(
        "--purge_intermediates",
        action="store_true",
        help="Remove the per-case working directory after a successful redock.",
    )
    g_misc.add_argument(
        "--limit", type=int, default=None, help="Process at most this many input files."
    )
    g_misc.add_argument(
        "--skip_existing",
        action="store_true",
        help="Reuse cases whose redocked pose already exists; still score them.",
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
        "--version", action="version", version=f"RVSAuto redock pipeline {__version__}"
    )
    return p


def find_inputs(input_dir, file_glob):
    patterns = [pat.strip() for pat in file_glob.split(",") if pat.strip()]
    found = set()
    for pat in patterns:
        for path in glob.glob(os.path.join(input_dir, pat)):
            if os.path.isfile(path):
                found.add(os.path.abspath(path))
    return sorted(found)


def install_ligand_pdbqt(
    template_pdbqt: str,
    dest_pdbqt: str,
    *,
    stem: str = "",
) -> None:
    """Copy a user-supplied ligand PDBQT template into a case directory."""
    src = os.path.abspath(template_pdbqt)
    if not os.path.isfile(src) or os.path.getsize(src) == 0:
        raise FileNotFoundError(f"ligand PDBQT not found or empty: {src}")
    os.makedirs(os.path.dirname(os.path.abspath(dest_pdbqt)), exist_ok=True)
    shutil.copy2(src, dest_pdbqt)
    label = f"[{stem}] " if stem else ""
    logging.info("%susing ligand PDBQT template: %s", label, src)


def preprocess_one_complex(input_path, workdir, args):
    stem = os.path.splitext(os.path.basename(input_path.replace(".gz", "")))[0]
    if stem.lower().endswith(".pdb"):
        stem = stem[:-4]
    elif stem.lower().endswith(".cif") or stem.lower().endswith(".mmcif"):
        stem = os.path.splitext(stem)[0]

    case_dir = os.path.join(workdir, "cases", stem)
    os.makedirs(case_dir, exist_ok=True)

    apo_pdb = os.path.join(case_dir, f"{stem}_apo.pdb")
    native_lig_pdb = os.path.join(case_dir, f"{stem}_native_lig.pdb")
    apo_pdbqt = os.path.join(case_dir, f"{stem}_apo.pdbqt")
    lig_pdbqt = os.path.join(case_dir, f"{stem}_lig.pdbqt")
    conf_file = os.path.join(case_dir, f"{stem}.conf")

    resname, chain_id, res_seq, n_heavy = split_complex(
        input_path,
        apo_pdb,
        native_lig_pdb,
        min_heavy_atoms=args.min_ligand_heavy_atoms,
        ligand_resname_hint=args.ligand_resname,
    )
    logging.info(
        "[%s] ligand = %s (chain %s, resseq %s, %d heavy atoms)",
        stem, resname, chain_id, res_seq, n_heavy,
    )

    center, size = box_from_ligand_pdb(
        native_lig_pdb,
        padding=args.box_padding,
        fixed_box_size=args.box_size,
    )
    try:
        receptor_pdb_to_pdbqt(
            apo_pdb,
            apo_pdbqt,
            adt_env_path=args.adt_env_path,
            clean=False,
            adt_log=os.path.join(case_dir, f"{stem}_prepare_receptor.log"),
        )
    except Exception as exc:
        logging.warning(
            "[%s] prepare_receptor4 failed (%s); retry with bonds_hydrogens.",
            stem,
            exc,
        )
        _sanitize_receptor_pdb(apo_pdb)
        if os.path.isfile(apo_pdbqt):
            os.remove(apo_pdbqt)
        receptor_pdb_to_pdbqt(
            apo_pdb,
            apo_pdbqt,
            adt_env_path=args.adt_env_path,
            clean=False,
            adt_repairs="bonds_hydrogens",
            adt_log=os.path.join(case_dir, f"{stem}_prepare_receptor_bonds.log"),
        )
    if args.ligand_pdbqt:
        install_ligand_pdbqt(args.ligand_pdbqt, lig_pdbqt, stem=stem)
    else:
        ligand_pdb_to_pdbqt(native_lig_pdb, lig_pdbqt, adt_env_path=args.adt_env_path)
    write_vina_config(conf_file, apo_pdbqt, center, size)

    return dict(
        stem=stem,
        input_path=input_path,
        case_dir=case_dir,
        apo_pdb=apo_pdb,
        native_lig_pdb=native_lig_pdb,
        apo_pdbqt=apo_pdbqt,
        lig_pdbqt=lig_pdbqt,
        conf_file=conf_file,
        ligand_resname=resname,
        ligand_chain=chain_id,
        ligand_resseq=res_seq,
        ligand_heavy_atoms=n_heavy,
        box_center=center,
        box_size=size,
    )


def dock_and_score(task, args, gpu_id, results_dir, unidock_bin):
    stem = task["stem"]
    case_dir = task["case_dir"]
    docked_pdbqt = os.path.join(case_dir, f"{stem}_docked.pdbqt")
    docking_log = os.path.join(case_dir, f"{stem}_unidock.log")

    affinity = run_unidock_single(
        conf_file=task["conf_file"],
        ligand_pdbqt=task["lig_pdbqt"],
        output_pdbqt=docked_pdbqt,
        log_file=docking_log,
        search_mode=args.search_mode,
        gpu_id=gpu_id,
        dock_env_path=args.dock_env_path,
        unidock_bin=unidock_bin,
        num_modes=args.num_modes,
        exhaustiveness=args.exhaustiveness,
        seed=args.seed,
    )

    row = dict(
        structure_id=stem,
        input_path=task["input_path"],
        ligand_resname=task["ligand_resname"],
        ligand_chain=task["ligand_chain"],
        ligand_resseq=task["ligand_resseq"],
        ligand_heavy_atoms=task["ligand_heavy_atoms"],
        box_center_x=task["box_center"][0],
        box_center_y=task["box_center"][1],
        box_center_z=task["box_center"][2],
        box_size_x=task["box_size"][0],
        box_size_y=task["box_size"][1],
        box_size_z=task["box_size"][2],
        binding_affinity=affinity if affinity is not None else float("nan"),
        rmsd_heavy=float("nan"),
        apo_pdb=task["apo_pdb"],
        native_ligand_pdb=task["native_lig_pdb"],
        best_pose_pdb="",
        redocked_complex_pdb="",
        status="ok",
    )

    if affinity is None:
        row["status"] = "failed: docking produced no output"
        return row

    best_pose_pdb = os.path.join(case_dir, f"{stem}_best_pose.pdb")
    if not extract_best_ligand_pose(docked_pdbqt, best_pose_pdb, as_hetatm=True):
        row["status"] = "failed: best-pose extraction failed"
        return row
    row["best_pose_pdb"] = best_pose_pdb

    if args.make_complex:
        complex_pdb = os.path.join(case_dir, f"{stem}_redocked_complex.pdb")
        if create_complex_pdb(task["apo_pdb"], best_pose_pdb, complex_pdb):
            row["redocked_complex_pdb"] = complex_pdb
            complex_results_dir = os.path.join(results_dir, "redocked_complexes")
            os.makedirs(complex_results_dir, exist_ok=True)
            shutil.copy2(
                complex_pdb,
                os.path.join(complex_results_dir, f"{stem}_redocked_complex.pdb"),
            )

    rmsd = heavy_atom_rmsd(
        task["native_lig_pdb"],
        best_pose_pdb,
        symmetry_aware=args.symmetry_aware_rmsd,
    )
    row["rmsd_heavy"] = rmsd
    logging.info(
        "[%s] affinity = %.3f kcal/mol, heavy-atom RMSD = %.3f A",
        stem, affinity, rmsd,
    )
    return row


SUMMARY_COLUMNS = [
    "structure_id",
    "input_path",
    "ligand_resname",
    "ligand_chain",
    "ligand_resseq",
    "ligand_heavy_atoms",
    "box_center_x", "box_center_y", "box_center_z",
    "box_size_x", "box_size_y", "box_size_z",
    "binding_affinity",
    "rmsd_heavy",
    "apo_pdb",
    "native_ligand_pdb",
    "best_pose_pdb",
    "redocked_complex_pdb",
    "status",
]


def write_summary(rows, summary_tsv):
    rows_sorted = sorted(
        rows,
        key=lambda r: (r.get("status") != "ok", r.get("rmsd_heavy", float("inf"))),
    )
    with open(summary_tsv, "w", encoding="utf-8") as fh:
        fh.write("\t".join(SUMMARY_COLUMNS) + "\n")
        for r in rows_sorted:
            vals = []
            for col in SUMMARY_COLUMNS:
                v = r.get(col, "")
                if isinstance(v, float):
                    vals.append("NA" if v != v else f"{v:.4f}")
                else:
                    vals.append(str(v))
            fh.write("\t".join(vals) + "\n")


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    log_file = setup_logging(output_dir, prefix="redock_run")
    progress = PipelineProgress(enabled=not args.no_progress)
    if args.quiet:
        silence_console_logging()
        set_quiet_subprocesses(True)
    elif progress.enabled:
        use_tqdm_safe_console_logging()
        # ADT/UniDock chatter still breaks the bar if left on the terminal.
        set_quiet_subprocesses(True)
    logging.info("RVSAuto redock + RMSD pipeline v%s", __version__)
    logging.info("Arguments: %s", vars(args))
    logging.info("Repository root: %s", REPO_ROOT)
    logging.info("Log file: %s", log_file)

    inputs = find_inputs(input_dir, args.file_glob)
    if args.limit:
        inputs = inputs[: args.limit]
    if not inputs:
        logging.error(
            "No input files found in %s matching %s", input_dir, args.file_glob
        )
        return 1
    if args.ligand_pdbqt:
        args.ligand_pdbqt = os.path.abspath(args.ligand_pdbqt)
        if not os.path.isfile(args.ligand_pdbqt) or os.path.getsize(args.ligand_pdbqt) == 0:
            logging.error(
                "Ligand PDBQT not found or empty: %s", args.ligand_pdbqt
            )
            return 1
        logging.info("Ligand PDBQT template: %s", args.ligand_pdbqt)
    logging.info("Found %d input complex file(s).", len(inputs))

    workdir = os.path.join(output_dir, "workdir")
    results_dir = os.path.join(output_dir, "results")
    os.makedirs(workdir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    summary_tsv = os.path.join(results_dir, "redock_summary.tsv")

    gpu_list, num_gpus = resolve_gpu_list(args.gpu_ids)
    max_parallel = args.max_parallel or num_gpus
    unidock_bin = find_unidock(args.unidock_path)
    logging.info(
        "GPUs: %s (num_gpus=%d, max_parallel=%d)", gpu_list, num_gpus, max_parallel
    )
    logging.info("UniDock binary: %s", unidock_bin)

    progress.begin_phase("Preprocess", len(inputs))
    logging.info("Phase 1: preprocessing complexes (split + PDBQT + conf)")
    prepared = []
    failed_rows = []

    with ThreadPoolExecutor(max_workers=max(1, args.total_cpu)) as pool:
        future_to_input = {
            pool.submit(preprocess_one_complex, ip, workdir, args): ip
            for ip in inputs
        }
        for i, fut in enumerate(as_completed(future_to_input), 1):
            ip = future_to_input[fut]
            stem = file_stem(ip.replace(".gz", ""))
            try:
                prepared.append(fut.result())
            except Exception as exc:
                logging.error("[%s] preprocessing failed: %s", stem, exc)
                logging.debug(traceback.format_exc())
                failed_rows.append(
                    dict(
                        structure_id=stem,
                        input_path=ip,
                        status=f"failed (preprocess): {exc}",
                        binding_affinity=float("nan"),
                        rmsd_heavy=float("nan"),
                    )
                )
            progress.step(1)
            if i % 50 == 0 or i == len(inputs):
                logging.info("  preprocessed %d/%d", i, len(inputs))

    logging.info(
        "Phase 1 done: %d ready, %d failed.", len(prepared), len(failed_rows)
    )

    to_dock = prepared
    already_done = []
    if args.skip_existing:
        to_dock = []
        for task in prepared:
            docked = os.path.join(task["case_dir"], f"{task['stem']}_docked.pdbqt")
            if os.path.exists(docked) and os.path.getsize(docked) > 0:
                already_done.append(task)
            else:
                to_dock.append(task)
        logging.info(
            "--skip_existing: %d already docked, %d remaining",
            len(already_done), len(to_dock),
        )

    logging.info("Phase 2: redocking with UniDock")
    rows = list(failed_rows)
    gpu_cycle = 0
    progress.begin_phase("UniDock redock", len(already_done) + len(to_dock))

    def _submit(pool, task, gpu_id):
        return pool.submit(
            dock_and_score, task, args, gpu_id, results_dir, unidock_bin
        )

    with ThreadPoolExecutor(max_workers=max(1, max_parallel)) as pool:
        future_to_task = {}
        for task in already_done:
            future_to_task[_submit(pool, task, None)] = task
        for task in to_dock:
            gpu_id = gpu_list[gpu_cycle % num_gpus] if num_gpus else None
            gpu_cycle += 1
            future_to_task[_submit(pool, task, gpu_id)] = task

        total = len(future_to_task)
        for i, fut in enumerate(as_completed(future_to_task), 1):
            task = future_to_task[fut]
            try:
                rows.append(fut.result())
            except Exception as exc:
                logging.error("[%s] dock_and_score failed: %s", task["stem"], exc)
                logging.debug(traceback.format_exc())
                rows.append(
                    dict(
                        structure_id=task["stem"],
                        input_path=task["input_path"],
                        ligand_resname=task["ligand_resname"],
                        ligand_chain=task["ligand_chain"],
                        ligand_resseq=task["ligand_resseq"],
                        ligand_heavy_atoms=task["ligand_heavy_atoms"],
                        binding_affinity=float("nan"),
                        rmsd_heavy=float("nan"),
                        status=f"failed (dock): {exc}",
                    )
                )
            progress.step(1)
            if total and (i % 25 == 0 or i == total):
                logging.info("  scored %d/%d", i, total)
                write_summary(rows, summary_tsv)

    progress.close()
    write_summary(rows, summary_tsv)

    if args.purge_intermediates:
        for r in rows:
            if r.get("status") != "ok":
                continue
            case_dir = os.path.join(workdir, "cases", r["structure_id"])
            if os.path.isdir(case_dir):
                try:
                    shutil.rmtree(case_dir)
                except Exception as exc:
                    logging.warning("Could not purge %s: %s", case_dir, exc)

    n_ok = sum(1 for r in rows if r.get("status") == "ok")
    rmsds = [
        r["rmsd_heavy"]
        for r in rows
        if r.get("status") == "ok"
        and isinstance(r.get("rmsd_heavy"), float)
        and r["rmsd_heavy"] == r["rmsd_heavy"]
    ]
    affs = [
        r["binding_affinity"]
        for r in rows
        if r.get("status") == "ok"
        and isinstance(r.get("binding_affinity"), float)
        and r["binding_affinity"] == r["binding_affinity"]
    ]

    print("\n" + "=" * 60)
    print("REDOCK + RMSD PIPELINE -- SUMMARY")
    print("=" * 60)
    print(f"Inputs            : {len(inputs)}")
    print(f"Successful        : {n_ok}")
    print(f"Failed            : {len(rows) - n_ok}")
    if rmsds:
        rmsds_arr = np.asarray(rmsds)
        success_2A = float((rmsds_arr <= 2.0).mean()) * 100.0
        print(
            f"Mean heavy-atom RMSD : {rmsds_arr.mean():.3f} A  "
            f"(median {np.median(rmsds_arr):.3f}, max {rmsds_arr.max():.3f})"
        )
        print(f"% poses <= 2.0 A      : {success_2A:.1f} %")
    if affs:
        affs_arr = np.asarray(affs)
        print(
            f"Mean affinity     : {affs_arr.mean():.3f} kcal/mol  "
            f"(median {np.median(affs_arr):.3f})"
        )
    print(f"Summary table     : {summary_tsv}")
    print(f"Log file          : {log_file}")
    print("=" * 60)
    logging.info("Pipeline finished.")
    return 0
