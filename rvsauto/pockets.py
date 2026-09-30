"""Binding-pocket construction from P2Rank and AF2BIND outputs."""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from Bio import PDB

from .common import as_conf_path, file_stem


# AF-Q9ZWT3-F1-model_v6  ->  Q9ZWT3
_AF_STEM_RE = re.compile(r"AF-([A-Za-z0-9]+)-F\d+", re.IGNORECASE)
_TOPN_SUFFIX_RE = re.compile(r"_af2bind_top\d+$", re.IGNORECASE)
_AF2BIND_SUFFIX_RE = re.compile(r"_af2bind$", re.IGNORECASE)
_P2RANK_PRED_SUFFIX = ".pdb_predictions.csv"
P2RANK_DATASET_NAME = "receptors"
# Common P2Rank screening default: keep pockets with calibrated probability > 0.05.
P2RANK_DEFAULT_MIN_PROBABILITY = 0.05


def normalize_p2rank_min_probability(
    value: Optional[float],
) -> Optional[float]:
    """Return the threshold to apply, or ``None`` to skip filtering (``<= 0`` disables)."""
    if value is None or value <= 0:
        return None
    return float(value)


@dataclass
class Pocket:
    protein_id: str
    pocket_index: int
    residue_ids: str
    residues: List[Tuple[str, int]]
    score: float
    engine: str
    extra: dict = field(default_factory=dict)


def _strip_af2bind_suffix(stem: str) -> str:
    stem = _TOPN_SUFFIX_RE.sub("", stem)
    stem = _AF2BIND_SUFFIX_RE.sub("", stem)
    return stem


def _protein_id_stem(name: str) -> str:
    """Filename stem for protein IDs that contain dots (e.g. ``gene.1_jaile_model_apo``).

    ``os.path.splitext`` only removes the last dotted suffix, so IDs like
    ``mescchra01g00156030.1_jaile_model_apo`` must not go through plain splitext
    when the string is already a stem (no ``.pdb`` / ``.csv`` extension).
    """
    base = os.path.basename(name).strip()
    for suffix in (".pdb", ".csv", ".pdbqt", ".cif"):
        if base.lower().endswith(suffix):
            return base[: -len(suffix)]
    return base


def protein_keys(name: str) -> List[str]:
    """Return match keys for a PDB / AF2BIND filename (most specific first)."""
    stem = _strip_af2bind_suffix(_protein_id_stem(name))
    keys = [stem]
    match = _AF_STEM_RE.search(stem)
    if match:
        keys.append(match.group(1))
    return keys


def _pbind_column(columns: Sequence[str]) -> str:
    for col in columns:
        normalized = col.strip().lower().replace(" ", "")
        if normalized in {"p(bind)", "pbind", "probability", "score"}:
            return col
    raise ValueError(
        f"No p(bind)/probability column in AF2BIND CSV (columns={list(columns)})"
    )


def index_af2bind_dir(
    pred_dir: str, recursive: bool = True
) -> Dict[str, str]:
    """Map protein stem -> best AF2BIND CSV path.

    Prefers the full per-residue ``*_af2bind.csv`` over ``*_af2bind_topN.csv``
    because top-N can be reconstructed from the full table.  Both official
    Colab outputs and the accompanying ``generate_vina_pocket_configs.py``
    naming scheme are accepted.
    """
    pred_dir = os.path.abspath(pred_dir)
    if not os.path.isdir(pred_dir):
        raise FileNotFoundError(f"AF2BIND directory not found: {pred_dir}")

    iterator: Iterable[str]
    if recursive:
        iterator = (
            os.path.join(root, fname)
            for root, _dirs, files in os.walk(pred_dir)
            for fname in files
        )
    else:
        iterator = (
            os.path.join(pred_dir, fname) for fname in os.listdir(pred_dir)
        )

    ranked: Dict[str, Tuple[int, str]] = {}
    for path in iterator:
        fname = os.path.basename(path)
        if not fname.lower().endswith(".csv"):
            continue
        stem = os.path.splitext(fname)[0]
        is_topn = bool(_TOPN_SUFFIX_RE.search(stem))
        is_full = bool(_AF2BIND_SUFFIX_RE.search(_TOPN_SUFFIX_RE.sub("", stem)))
        if not (is_topn or is_full):
            # Accept a generic results.csv only when it has the expected columns
            if fname.lower() in {"results.csv", "af2bind.csv"}:
                protein_stem = os.path.basename(os.path.dirname(path))
                priority = 1
            else:
                continue
        else:
            protein_stem = _strip_af2bind_suffix(stem)
            # lower number = better; full table beats top-N
            priority = 1 if is_full and not is_topn else 2

        keys = protein_keys(protein_stem)
        for key in keys:
            prev = ranked.get(key)
            if prev is None or priority < prev[0]:
                ranked[key] = (priority, os.path.abspath(path))

    return {key: path for key, (_pri, path) in ranked.items()}


def match_af2bind_csv(pdb_path: str, index: Dict[str, str]) -> Optional[str]:
    for key in protein_keys(os.path.basename(pdb_path)):
        if key in index:
            return index[key]
    return None


def load_af2bind_table(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    required = {"chain", "resi"}
    missing = required - set(c.lower() for c in df.columns)
    if missing:
        # try case-insensitive remap
        lower_map = {c.lower(): c for c in df.columns}
        rename = {}
        for want in ("chain", "resi", "resn", "p(bind)", "rank"):
            if want in lower_map and want != lower_map[want]:
                rename[lower_map[want]] = want
        if rename:
            df = df.rename(columns=rename)
            df.columns = [c.strip() for c in df.columns]
        missing = {"chain", "resi"} - set(c.lower() for c in df.columns)
        if missing:
            raise ValueError(
                f"{csv_path} is not an AF2BIND residue table "
                f"(missing {missing}, columns={list(df.columns)})"
            )
    # Normalise column names we care about
    colmap = {c.lower(): c for c in df.columns}
    df = df.rename(columns={colmap["chain"]: "chain", colmap["resi"]: "resi"})
    if "resn" in colmap:
        df = df.rename(columns={colmap["resn"]: "resn"})
    pcol = _pbind_column(df.columns)
    if pcol != "p(bind)":
        df = df.rename(columns={pcol: "p(bind)"})
    df["chain"] = df["chain"].astype(str).str.strip()
    df["resi"] = pd.to_numeric(df["resi"], errors="coerce")
    df = df.dropna(subset=["resi"])
    df["resi"] = df["resi"].astype(int)
    df["p(bind)"] = pd.to_numeric(df["p(bind)"], errors="coerce")
    return df


def select_af2bind_residues(
    df: pd.DataFrame,
    top_n: Optional[int] = None,
    threshold: Optional[float] = 0.28,
) -> pd.DataFrame:
    """Select AF2BIND residues for pocket building.

    Default (proteome screening, Gazizov et al.): all residues with
    ``p(bind) >= 0.28``, then spatial clustering into multi-residue sites.

    Notebook / PyMOL mode: pass ``top_n=15`` (and ``threshold=None``) to keep
    only the highest-scoring residues regardless of absolute cutoff.
    """
    ranked = df.sort_values("p(bind)", ascending=False)
    if top_n is not None and top_n > 0:
        return ranked.head(top_n).reset_index(drop=True)
    if threshold is not None:
        selected = ranked[ranked["p(bind)"] >= threshold]
        if selected.empty:
            logging.warning(
                "No residues pass p(bind) >= %.3f in this table.", threshold
            )
        return selected.reset_index(drop=True)
    return ranked.reset_index(drop=True)


def _residue_anchor_coord(structure, chain_id: str, res_num: int):
    """CA coordinate, or centroid of all atoms if CA is missing."""
    for model in structure:
        preferred = []
        others = []
        for chain in model:
            (preferred if chain.id == chain_id else others).append(chain)
        for chain in preferred + others:
            for residue in chain:
                if residue.id[1] != res_num:
                    continue
                if "CA" in residue:
                    return np.asarray(residue["CA"].get_coord(), dtype=float)
                coords = [atom.get_coord() for atom in residue]
                if coords:
                    return np.mean(np.asarray(coords), axis=0)
        break
    return None


def single_linkage_clusters(
    labels: Sequence,
    coords: np.ndarray,
    cutoff: float,
) -> List[List[int]]:
    """Single-linkage clustering (union-find) on Euclidean distance.

    The AF2BIND helper ``generate_vina_pocket_configs.py`` called
    ``linkage(squareform(pdist(points)))``, which treats the *square* distance
    matrix as observation vectors and yields incorrect clusters.  This
    implementation clusters the 3D points directly.
    """
    n = len(labels)
    if n == 0:
        return []
    if n == 1:
        return [[0]]

    parent = list(range(n))
    rank = [0] * n

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    pts = np.asarray(coords, dtype=float)
    for i in range(n):
        delta = pts[i + 1:] - pts[i]
        dist = np.sqrt((delta * delta).sum(axis=1))
        hits = np.where(dist <= cutoff)[0]
        for h in hits:
            union(i, i + 1 + int(h))

    groups: Dict[int, List[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def _format_residue_ids(residues: Sequence[Tuple[str, int]]) -> str:
    return " ".join(f"{chain}_{resi}" for chain, resi in residues)


def pockets_from_af2bind(
    pdb_path: str,
    csv_path: str,
    top_n: Optional[int] = None,
    threshold: Optional[float] = 0.28,
    cluster_cutoff: float = 12.0,
    min_residues: int = 5,
    pocket_mode: str = "cluster",
    max_pockets: Optional[int] = None,
) -> List[Pocket]:
    """Build docking pockets from one AF2BIND CSV + the matching PDB.

    Defaults follow the human-proteome AF2BIND workflow: classifier cutoff
    ``p(bind) >= 0.28``, spatial clustering, keep sites with more than four
    residues (``min_residues=5``).

    *pocket_mode*:
      - ``single``: merge all selected residues into one docking box
      - ``cluster``: split spatially separated residues into multiple pockets
        (cutoff in Angstrom on CA/centroid distance)
    """
    protein_id = file_stem(pdb_path)
    df = load_af2bind_table(csv_path)
    selected = select_af2bind_residues(df, top_n=top_n, threshold=threshold)
    if selected.empty:
        logging.warning("No AF2BIND residues selected for %s", protein_id)
        return []

    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure(protein_id, pdb_path)

    records = []
    for _, row in selected.iterrows():
        chain = str(row["chain"]).strip() or "A"
        resi = int(row["resi"])
        coord = _residue_anchor_coord(structure, chain, resi)
        if coord is None:
            logging.debug("Residue %s_%s not found in %s", chain, resi, protein_id)
            continue
        records.append(
            {
                "chain": chain,
                "resi": resi,
                "coord": coord,
                "pbind": float(row["p(bind)"]) if pd.notna(row["p(bind)"]) else 0.0,
            }
        )
    if not records:
        logging.warning(
            "AF2BIND residues for %s were not found in the PDB; skip.", protein_id
        )
        return []

    if pocket_mode == "single" or len(records) == 1 or cluster_cutoff <= 0:
        groups = [list(range(len(records)))]
    else:
        coords = np.vstack([r["coord"] for r in records])
        groups = single_linkage_clusters(
            list(range(len(records))), coords, cutoff=cluster_cutoff
        )

    kept = [g for g in groups if len(g) >= min_residues]
    if not kept:
        logging.warning(
            "%s: no AF2BIND site with >= %d residues after clustering; skip.",
            protein_id, min_residues,
        )
        return []

    pockets: List[Pocket] = []
    scored_groups = []
    for group in kept:
        mean_p = float(np.mean([records[i]["pbind"] for i in group]))
        max_p = float(np.max([records[i]["pbind"] for i in group]))
        scored_groups.append((max_p, mean_p, group))
    scored_groups.sort(key=lambda t: (t[0], t[1], len(t[2])), reverse=True)

    if max_pockets is not None:
        scored_groups = scored_groups[: max(0, max_pockets)]

    for idx, (max_p, mean_p, group) in enumerate(scored_groups, start=1):
        residues = [(records[i]["chain"], records[i]["resi"]) for i in group]
        residues.sort(key=lambda t: (t[0], t[1]))
        pockets.append(
            Pocket(
                protein_id=protein_id,
                pocket_index=idx,
                residue_ids=_format_residue_ids(residues),
                residues=residues,
                score=max_p,
                engine="af2bind",
                extra={
                    "mean_pbind": round(mean_p, 6),
                    "max_pbind": round(max_p, 6),
                    "n_residues": len(residues),
                    "source_csv": os.path.basename(csv_path),
                },
            )
        )
    return pockets


def resolve_p2rank_predictions_dir(
    pocket_dir: str,
    pdb_names: Sequence[str],
    dataset_name: str = P2RANK_DATASET_NAME,
) -> str:
    """Find the folder containing P2Rank ``*_predictions.csv`` files.

    Standard layout: ``<pocket_dir>/predict_<dataset_stem>/`` (from ``.ds`` input).
    Legacy / some installs write CSVs directly under ``pocket_dir``.
    """
    pocket_dir = os.path.abspath(pocket_dir)
    stems = [file_stem(name) for name in pdb_names]

    def match_count(directory: str) -> int:
        if not os.path.isdir(directory):
            return 0
        return sum(
            1
            for stem in stems
            if os.path.isfile(
                os.path.join(directory, f"{stem}{_P2RANK_PRED_SUFFIX}")
            )
        )

    candidates: List[str] = []
    seen = set()

    def add(path: str) -> None:
        path = os.path.abspath(path)
        if path not in seen:
            seen.add(path)
            candidates.append(path)

    add(os.path.join(pocket_dir, f"predict_{dataset_name}"))
    add(pocket_dir)
    if os.path.isdir(pocket_dir):
        for entry in sorted(os.listdir(pocket_dir)):
            if entry.startswith("predict_"):
                add(os.path.join(pocket_dir, entry))

    best_dir = os.path.join(pocket_dir, f"predict_{dataset_name}")
    best_count = 0
    for cand in candidates:
        n = match_count(cand)
        if n > best_count:
            best_count = n
            best_dir = cand

    if best_count > 0:
        expected = os.path.join(pocket_dir, f"predict_{dataset_name}")
        if os.path.abspath(best_dir) != os.path.abspath(expected):
            logging.info(
                "P2Rank prediction CSVs in %s (%d/%d); default expected %s.",
                best_dir,
                best_count,
                len(stems),
                expected,
            )
    return best_dir


def _p2rank_probability_column(columns: Sequence[str]) -> Optional[str]:
    for col in columns:
        key = col.strip().lower().replace(" ", "")
        if key in {"probability", "prob", "pocket_probability"}:
            return col
    return None


def pockets_from_p2rank_csv(
    csv_path: str,
    max_pockets: Optional[int] = None,
    min_probability: Optional[float] = None,
) -> List[Pocket]:
    """Parse a P2Rank ``*.pdb_predictions.csv`` into Pocket objects.

    Pockets are kept in P2Rank rank order. When *min_probability* is set, rows
    with calibrated ``probability`` below the threshold are dropped before
    applying *max_pockets*.
    """
    fname = os.path.basename(csv_path)
    if fname.endswith(_P2RANK_PRED_SUFFIX):
        protein_id = fname[: -len(_P2RANK_PRED_SUFFIX)]
        if protein_id.endswith(".pdb"):
            protein_id = protein_id[:-4]
    else:
        protein_id = file_stem(fname)

    data = pd.read_csv(csv_path)
    data.columns = data.columns.str.strip()
    if "residue_ids" not in data.columns:
        raise ValueError(f"{csv_path} has no residue_ids column")

    prob_col = _p2rank_probability_column(data.columns)
    if min_probability is not None and prob_col is None:
        logging.warning(
            "%s: --p2rank_min_probability=%.3f ignored (no probability column).",
            fname,
            min_probability,
        )
        min_probability = None

    if min_probability is not None and prob_col is not None:
        probs = pd.to_numeric(data[prob_col], errors="coerce")
        before = len(data)
        data = data[probs >= min_probability].copy()
        dropped = before - len(data)
        if dropped:
            logging.info(
                "P2Rank %s: dropped %d pocket(s) with %s < %.3f (%d remain).",
                protein_id,
                dropped,
                prob_col,
                min_probability,
                len(data),
            )
        if data.empty:
            logging.warning(
                "P2Rank %s: no pockets pass probability >= %.3f.",
                protein_id,
                min_probability,
            )
            return []

    if max_pockets is not None:
        data = data.head(max(0, max_pockets))

    pockets: List[Pocket] = []
    for pocket_idx, (_, row) in enumerate(data.iterrows(), start=1):
        residue_ids = str(row.get("residue_ids", "")).strip()
        if not residue_ids or residue_ids.lower() == "nan":
            continue
        residues = []
        for token in residue_ids.split():
            parts = token.split("_")
            if len(parts) != 2:
                continue
            try:
                residues.append((parts[0], int(parts[1])))
            except ValueError:
                continue
        score = row.get("score", row.get(prob_col or "probability", np.nan))
        try:
            score = float(score)
        except (TypeError, ValueError):
            score = float("nan")
        prob_val = row.get(prob_col, row.get("probability", "")) if prob_col else row.get(
            "probability", ""
        )
        pockets.append(
            Pocket(
                protein_id=protein_id,
                pocket_index=pocket_idx,
                residue_ids=residue_ids,
                residues=residues,
                score=score,
                engine="p2rank",
                extra={
                    "probability": prob_val,
                    "center_x": row.get("center_x", ""),
                    "center_y": row.get("center_y", ""),
                    "center_z": row.get("center_z", ""),
                },
            )
        )
    return pockets


# Per-axis cap keeps search focused (accuracy); volume cap matches UniDock (vina.cpp).
DEFAULT_BOX_MAX_SIDE = 25.0
UNIDOCK_MAX_GRID_VOLUME = 27000.0


def _apply_box_limits(
    size_x: float,
    size_y: float,
    size_z: float,
    max_side: Optional[float] = DEFAULT_BOX_MAX_SIDE,
    max_volume: float = UNIDOCK_MAX_GRID_VOLUME,
) -> tuple[float, float, float]:
    """Clamp docking grid to accuracy / UniDock limits."""
    orig = (size_x, size_y, size_z)
    sx, sy, sz = float(size_x), float(size_y), float(size_z)

    if max_side is not None and max_side > 0:
        sx = min(sx, max_side)
        sy = min(sy, max_side)
        sz = min(sz, max_side)

    if max_volume is not None and max_volume > 0:
        vol = sx * sy * sz
        if vol > max_volume:
            scale = (max_volume / vol) ** (1.0 / 3.0)
            sx = round(sx * scale, 1)
            sy = round(sy * scale, 1)
            sz = round(sz * scale, 1)

    if (round(sx, 1), round(sy, 1), round(sz, 1)) != (
        round(orig[0], 1), round(orig[1], 1), round(orig[2], 1)
    ):
        side_note = f"{max_side:.1f}" if max_side and max_side > 0 else "off"
        vol_note = f"{max_volume:.0f}" if max_volume and max_volume > 0 else "off"
        logging.warning(
            "Docking box capped from %.1f x %.1f x %.1f A (volume %.0f A^3) "
            "to %.1f x %.1f x %.1f A (volume %.0f A^3); "
            "limits: max_side=%s A, max_volume=%s A^3.",
            orig[0], orig[1], orig[2], orig[0] * orig[1] * orig[2],
            sx, sy, sz, sx * sy * sz, side_note, vol_note,
        )
    return sx, sy, sz


def calculate_box_size(
    pdb_file: str,
    residue_ids: str,
    margin: float = 2.0,
    fixed_box_size: Optional[float] = None,
    max_box_side: Optional[float] = DEFAULT_BOX_MAX_SIDE,
    max_box_volume: Optional[float] = UNIDOCK_MAX_GRID_VOLUME,
):
    """Grid box from all atoms of the listed pocket residues (chain_resnum)."""
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)

    residue_list = residue_ids.strip().split()
    coords = []
    for residue_id in residue_list:
        parts = residue_id.split("_")
        if len(parts) != 2:
            continue
        chain_id, res_num_s = parts[0], parts[1]
        try:
            res_num = int(res_num_s)
        except ValueError:
            continue
        for model in structure:
            if chain_id not in model:
                continue
            chain = model[chain_id]
            for residue in chain:
                if residue.id[1] == res_num:
                    for atom in residue:
                        coords.append(atom.get_coord())

    if not coords:
        logging.warning(
            "No residues found for pocket in %s; using default 30 A box at origin.",
            os.path.basename(pdb_file),
        )
        return 0.0, 0.0, 0.0, DEFAULT_BOX_MAX_SIDE, DEFAULT_BOX_MAX_SIDE, DEFAULT_BOX_MAX_SIDE

    coords = np.asarray(coords, dtype=float)
    center = np.mean(coords, axis=0)
    if fixed_box_size is not None:
        size_x = size_y = size_z = round(float(fixed_box_size), 1)
    else:
        distances = np.abs(coords - center)
        size_x = max(15.0, round(float(2 * (distances[:, 0].max() + margin)), 1))
        size_y = max(15.0, round(float(2 * (distances[:, 1].max() + margin)), 1))
        size_z = max(15.0, round(float(2 * (distances[:, 2].max() + margin)), 1))
    size_x, size_y, size_z = _apply_box_limits(
        size_x, size_y, size_z, max_side=max_box_side, max_volume=max_box_volume,
    )

    return (
        round(float(center[0]), 4),
        round(float(center[1]), 4),
        round(float(center[2]), 4),
        size_x, size_y, size_z,
    )


def write_vina_config(conf_path: str, receptor_pdbqt: str, center, size) -> None:
    with open(conf_path, "w", encoding="utf-8") as fh:
        fh.write(
            f"receptor = {as_conf_path(receptor_pdbqt)}\n"
            f"center_x = {center[0]}\n"
            f"center_y = {center[1]}\n"
            f"center_z = {center[2]}\n"
            f"size_x = {size[0]}\n"
            f"size_y = {size[1]}\n"
            f"size_z = {size[2]}\n"
        )


def write_pocket_configs(
    pockets: Sequence[Pocket],
    pdb_dir: str,
    pdbqt_dir: str,
    pocket_output_dir: str,
    margin: float = 2.0,
    fixed_box_size: Optional[float] = None,
    max_box_side: Optional[float] = DEFAULT_BOX_MAX_SIDE,
    max_box_volume: Optional[float] = UNIDOCK_MAX_GRID_VOLUME,
    skip_existing: bool = True,
    require_pdbqt: bool = False,
    progress=None,
) -> List[dict]:
    """Write UniDock/Vina .conf files and return a summary table as dicts.

    *progress*, if given, should expose ``step(n=1, phase=...)`` (e.g.
    ``PipelineProgress``) and is advanced once per pocket.
    """
    os.makedirs(pocket_output_dir, exist_ok=True)
    rows = []
    for pocket in pockets:
        pdb_file = os.path.join(pdb_dir, f"{pocket.protein_id}.pdb")
        if not os.path.exists(pdb_file):
            logging.warning("PDB file %s not found; skip pocket.", pdb_file)
            if progress is not None:
                progress.step(1)
            continue
        pdbqt_file = os.path.join(pdbqt_dir, f"{pocket.protein_id}.pdbqt")
        if require_pdbqt and (
            not os.path.isfile(pdbqt_file) or os.path.getsize(pdbqt_file) == 0
        ):
            logging.warning(
                "Receptor PDBQT missing for %s; skip pocket %d.",
                pocket.protein_id, pocket.pocket_index,
            )
            if progress is not None:
                progress.step(1)
            continue
        config_filename = (
            f"config_{pocket.protein_id}_pocket_{pocket.pocket_index}.conf"
        )
        config_path = os.path.join(pocket_output_dir, config_filename)
        if skip_existing and os.path.exists(config_path):
            logging.info("Config %s already exists; skip rewrite.", config_filename)
            parsed = {}
            with open(config_path, encoding="utf-8") as fh:
                for line in fh:
                    if "=" in line:
                        key, val = line.split("=", 1)
                        parsed[key.strip()] = val.strip()
            try:
                cx, cy, cz = (
                    float(parsed["center_x"]),
                    float(parsed["center_y"]),
                    float(parsed["center_z"]),
                )
                sx, sy, sz = (
                    float(parsed["size_x"]),
                    float(parsed["size_y"]),
                    float(parsed["size_z"]),
                )
            except (KeyError, ValueError):
                cx = cy = cz = sx = sy = sz = float("nan")
        else:
            cx, cy, cz, sx, sy, sz = calculate_box_size(
                pdb_file, pocket.residue_ids,
                margin=margin, fixed_box_size=fixed_box_size,
                max_box_side=max_box_side, max_box_volume=max_box_volume,
            )
            write_vina_config(config_path, pdbqt_file, (cx, cy, cz), (sx, sy, sz))
        size_src = (
            f"fixed={fixed_box_size} A" if fixed_box_size is not None
            else f"auto (margin={margin} A)"
        )
        logging.info(
            "  %s pocket %d: center=(%.3f, %.3f, %.3f)  box=%.1f x %.1f x %.1f A (%s)",
            pocket.protein_id, pocket.pocket_index, cx, cy, cz, sx, sy, sz, size_src,
        )
        rows.append(
            {
                "protein_id": pocket.protein_id,
                "pocket_index": pocket.pocket_index,
                "engine": pocket.engine,
                "score": pocket.score,
                "n_residues": len(pocket.residues) or len(pocket.residue_ids.split()),
                "residue_ids": pocket.residue_ids,
                "center_x": cx, "center_y": cy, "center_z": cz,
                "size_x": sx, "size_y": sy, "size_z": sz,
                "config_file": config_filename,
                **{k: v for k, v in pocket.extra.items() if k not in {"center_x", "center_y", "center_z"}},
            }
        )
        if progress is not None:
            progress.step(1)
    return rows


def write_pocket_summary(rows: Sequence[dict], tsv_path: str) -> None:
    if not rows:
        return
    columns = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with open(tsv_path, "w", encoding="utf-8") as fh:
        fh.write("\t".join(columns) + "\n")
        for row in rows:
            fh.write("\t".join(str(row.get(c, "")) for c in columns) + "\n")
