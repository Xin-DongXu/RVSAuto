"""Result tables: long-form docking scores and ligand-by-pocket matrices."""

from __future__ import annotations

import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def parse_docking_label(label: str, ligand_name: str) -> Optional[Tuple[str, str]]:
    """Split a result key into (protein_id, pocket_id).

    Keys produced by the screening pipeline look like
    ``config_{protein}_pocket_{n}_{ligand}`` or ``{protein}_pocket_{n}_{ligand}``.
    """
    text = label[7:] if label.startswith("config_") else label
    suffix = "_" + ligand_name
    if text.endswith(suffix):
        text = text[: -len(suffix)]
    marker = "_pocket_"
    idx = text.rfind(marker)
    if idx < 0:
        return None
    protein_id = text[:idx]
    pocket_raw = text[idx + len(marker) :]
    token = pocket_raw.split("_")[0]
    try:
        pocket_id = f"pocket_{int(token)}"
    except ValueError:
        pocket_id = f"pocket_{pocket_raw}"
    if not protein_id:
        return None
    return protein_id, pocket_id


def _pocket_sort_key(pocket_id: str):
    token = pocket_id.replace("pocket_", "", 1)
    try:
        return (0, int(token))
    except ValueError:
        return (1, pocket_id)


def collect_records(
    all_affinities: Dict[str, Sequence[Tuple[str, float]]],
) -> List[dict]:
    rows = []
    for ligand_name, hits in all_affinities.items():
        for label, affinity in hits:
            parsed = parse_docking_label(label, ligand_name)
            if parsed is None:
                continue
            protein_id, pocket_id = parsed
            rows.append(
                {
                    "protein_id": protein_id,
                    "pocket": pocket_id,
                    "ligand": ligand_name,
                    "affinity": affinity,
                    "label": label.replace("config_", "", 1),
                }
            )
    rows.sort(key=lambda r: (r["protein_id"], _pocket_sort_key(r["pocket"]), r["ligand"]))
    return rows


def write_tsv(path: str, headers: Sequence[str], rows: Iterable[Sequence]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\t".join(headers) + "\n")
        for row in rows:
            fh.write("\t".join("" if v is None else str(v) for v in row) + "\n")


def write_docking_tables(
    all_affinities: Dict[str, Sequence[Tuple[str, float]]],
    ligand_names: Sequence[str],
    results_dir: str,
    complex_relpath: Optional[str] = "complexes",
) -> dict:
    """Write long table + two affinity matrices. Return output paths."""
    records = collect_records(all_affinities)
    ligands = list(ligand_names)

    long_path = os.path.join(results_dir, "docking_long.tsv")
    long_rows = []
    for rec in records:
        complex_name = ""
        if complex_relpath:
            complex_name = os.path.join(
                complex_relpath,
                f"{rec['protein_id']}_{rec['pocket']}_{rec['ligand']}_complex.pdb",
            ).replace("\\", "/")
        long_rows.append(
            (
                rec["protein_id"],
                rec["pocket"],
                rec["ligand"],
                rec["affinity"],
                rec["label"],
                complex_name,
            )
        )
    write_tsv(
        long_path,
        [
            "Protein_ID",
            "Pocket",
            "Ligand",
            "Binding_Affinity_kcal_mol",
            "Label",
            "Complex_PDB",
        ],
        long_rows,
    )

    pair_scores: Dict[Tuple[str, str, str], float] = {}
    for rec in records:
        key = (rec["protein_id"], rec["pocket"], rec["ligand"])
        prev = pair_scores.get(key)
        if prev is None or rec["affinity"] < prev:
            pair_scores[key] = rec["affinity"]

    pocket_rows = sorted(
        {(p, pk) for p, pk, _lig in pair_scores},
        key=lambda t: (t[0], _pocket_sort_key(t[1])),
    )
    pocket_matrix = os.path.join(results_dir, "affinity_matrix_pockets.tsv")
    matrix_rows = []
    for protein_id, pocket_id in pocket_rows:
        vals = [protein_id, pocket_id]
        for lig in ligands:
            aff = pair_scores.get((protein_id, pocket_id, lig))
            vals.append("" if aff is None else f"{aff:.4f}")
        matrix_rows.append(vals)
    write_tsv(
        pocket_matrix,
        ["Protein_ID", "Pocket"] + list(ligands),
        matrix_rows,
    )

    best: Dict[Tuple[str, str], Tuple[float, str]] = {}
    for rec in records:
        key = (rec["protein_id"], rec["ligand"])
        prev = best.get(key)
        if prev is None or rec["affinity"] < prev[0]:
            best[key] = (rec["affinity"], rec["pocket"])

    proteins = sorted({r["protein_id"] for r in records})
    protein_matrix = os.path.join(results_dir, "affinity_matrix_proteins.tsv")
    pocket_choice = os.path.join(results_dir, "affinity_matrix_proteins_best_pocket.tsv")
    prot_rows = []
    choice_rows = []
    for protein_id in proteins:
        aff_vals = [protein_id]
        pk_vals = [protein_id]
        for lig in ligands:
            hit = best.get((protein_id, lig))
            if hit is None:
                aff_vals.append("")
                pk_vals.append("")
            else:
                aff_vals.append(f"{hit[0]:.4f}")
                pk_vals.append(hit[1])
        prot_rows.append(aff_vals)
        choice_rows.append(pk_vals)
    write_tsv(protein_matrix, ["Protein_ID"] + list(ligands), prot_rows)
    write_tsv(pocket_choice, ["Protein_ID"] + list(ligands), choice_rows)

    return {
        "long": long_path,
        "matrix_pockets": pocket_matrix,
        "matrix_proteins": protein_matrix,
        "matrix_best_pocket": pocket_choice,
    }
