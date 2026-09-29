"""Heavy-atom RMSD between two ligand poses in the same coordinate frame."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


def read_heavy_atoms(pdb_path: str) -> List[Tuple[str, str, np.ndarray]]:
    """Return (atom_name, element, xyz) for every non-hydrogen atom."""
    atoms = []
    with open(pdb_path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            element = line[76:78].strip().upper()
            if not element:
                name = line[12:16].strip()
                element = name[0].upper() if name else "C"
            if element == "H":
                continue
            atom_name = line[12:16].strip()
            try:
                xyz = np.array(
                    [float(line[30:38]), float(line[38:46]), float(line[46:54])]
                )
            except (ValueError, IndexError):
                continue
            atoms.append((atom_name, element, xyz))
    return atoms


def heavy_atom_rmsd(
    native_pdb: str, pose_pdb: str, symmetry_aware: bool = False
) -> float:
    """Heavy-atom RMSD; optional RDKit symmetry-aware refinement."""
    a_atoms = read_heavy_atoms(native_pdb)
    b_atoms = read_heavy_atoms(pose_pdb)
    if not a_atoms or not b_atoms:
        return float("nan")

    a_by_name = {n: (e, c) for (n, e, c) in a_atoms}
    common = [n for (n, _, _) in b_atoms if n in a_by_name]
    if len(common) >= max(3, min(len(a_atoms), len(b_atoms)) * 0.8):
        pairs = []
        used = set()
        for n, _, cb in b_atoms:
            if n in a_by_name and n not in used:
                pairs.append((a_by_name[n][1], cb))
                used.add(n)
        ca = np.asarray([p[0] for p in pairs])
        cb = np.asarray([p[1] for p in pairs])
        diff = ca - cb
        rmsd = float(np.sqrt((diff * diff).sum() / len(pairs)))
    else:
        n = min(len(a_atoms), len(b_atoms))
        ca = np.asarray([a_atoms[i][2] for i in range(n)])
        cb = np.asarray([b_atoms[i][2] for i in range(n)])
        diff = ca - cb
        rmsd = float(np.sqrt((diff * diff).sum() / n))

    if symmetry_aware:
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem

            ma = Chem.MolFromPDBFile(native_pdb, removeHs=True, sanitize=False)
            mb = Chem.MolFromPDBFile(pose_pdb, removeHs=True, sanitize=False)
            if ma is not None and mb is not None:
                try:
                    rmsd = min(rmsd, float(AllChem.GetBestRMS(ma, mb)))
                except Exception:
                    pass
        except ImportError:
            pass
    return rmsd
