"""PDB / mmCIF I/O helpers used by both screening and redocking pipelines."""

from __future__ import annotations

import gzip
import logging
import os
import shutil
import tempfile
from typing import Optional, Tuple

from Bio import PDB
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning

import warnings

warnings.simplefilter("ignore", PDBConstructionWarning)

WATER_NAMES = {"HOH", "WAT", "H2O", "DOD", "D2O", "TIP", "SOL"}

STANDARD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "MSE", "SEC", "PYL",
}

# Side-chain atoms required for MGLTools prepare_receptor4 (incomplete → valence/H errors).
_SIDECHAIN_REQUIRED = {
    "ALA": {"CB"},
    "ARG": {"CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"},
    "ASN": {"CB", "CG", "OD1", "ND2"},
    "ASP": {"CB", "CG", "OD1", "OD2"},
    "CYS": {"CB", "SG"},
    "GLN": {"CB", "CG", "CD", "OE1", "NE2"},
    "GLU": {"CB", "CG", "CD", "OE1", "OE2"},
    "HIS": {"CB", "CG", "ND1", "CD2", "CE1", "NE2"},
    "HIE": {"CB", "CG", "ND1", "CD2", "CE1", "NE2"},
    "HID": {"CB", "CG", "ND1", "CD2", "CE1", "NE2"},
    "HIP": {"CB", "CG", "ND1", "CD2", "CE1", "NE2"},
    "ILE": {"CB", "CG1", "CG2", "CD1"},
    "LEU": {"CB", "CG", "CD1", "CD2"},
    "LYS": {"CB", "CG", "CD", "CE", "NZ"},
    "MET": {"CB", "CG", "SD", "CE"},
    "MSE": {"CB", "CG", "SE", "CE"},
    "PHE": {"CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"},
    "PRO": {"CB", "CG", "CD"},
    "SER": {"CB", "OG"},
    "THR": {"CB", "OG1", "CG2"},
    "TRP": {"CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"},
    "TYR": {"CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"},
    "VAL": {"CB", "CG1", "CG2"},
}

SOLVENT_ION_BUFFER = WATER_NAMES | {
    "NA", "K", "CL", "MG", "CA", "ZN", "FE", "MN", "CU", "NI", "CO",
    "CD", "HG", "BR", "I", "F", "LI", "CS", "RB", "SR", "BA", "AL",
    "SO4", "PO4", "ACT", "EDO", "GOL", "PEG", "PG4", "P6G", "1PE",
    "DMS", "TRS", "IMD", "MES", "HEP", "EPE", "BTB", "FMT", "ACE",
    "BCT", "CO3", "NO3", "MPD", "BME", "DTT", "DTV", "TPP", "PGE",
    "IOD", "CIT", "TLA", "MLI", "OXL",
}


def _maybe_decompress(input_path: str) -> Tuple[str, Optional[str]]:
    """If *input_path* is gzipped, write a temp uncompressed copy.

    Returns (path_to_read, temp_path_or_None). Caller must delete temp_path.
    """
    if input_path.endswith(".gz"):
        suffix = os.path.splitext(input_path[:-3])[1] or ".pdb"
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp.close()
        with gzip.open(input_path, "rb") as src, open(tmp.name, "wb") as dst:
            shutil.copyfileobj(src, dst)
        return tmp.name, tmp.name
    return input_path, None


def load_structure(input_path: str):
    """Load a PDB or mmCIF file (optionally .gz) into a Biopython Structure."""
    path, tmp = _maybe_decompress(input_path)
    try:
        base = path[:-3] if path.endswith(".gz") else path
        ext = os.path.splitext(base)[1].lower()
        if ext in (".cif", ".mmcif"):
            parser = MMCIFParser(QUIET=True)
        else:
            parser = PDB.PDBParser(QUIET=True)
        name = os.path.splitext(os.path.basename(input_path.replace(".gz", "")))[0]
        return parser.get_structure(name, path)
    finally:
        if tmp and os.path.exists(tmp):
            os.remove(tmp)


def clean_pdb(input_pdb: str, output_pdb: str) -> None:
    """Keep only standard protein ATOM records (drop ligands, waters, ions)."""
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein", input_pdb)

    removed_residues = []
    kept_residues = 0
    for model in structure:
        for chain in model:
            for residue in chain:
                resname = residue.get_resname().strip()
                het_flag = residue.id[0].strip()
                is_aa = PDB.is_aa(residue, standard=True)
                is_water = resname in WATER_NAMES or het_flag == "W"
                if is_aa and not is_water:
                    kept_residues += 1
                else:
                    removed_residues.append(
                        f"{chain.id}/{residue.id[1]}{residue.id[2].strip()} {resname}"
                    )

    if removed_residues:
        logging.info(
            "clean_pdb '%s': removing %d non-protein group(s): %s%s",
            os.path.basename(input_pdb),
            len(removed_residues),
            ", ".join(removed_residues[:20]),
            " ..." if len(removed_residues) > 20 else "",
        )
    else:
        logging.info(
            "clean_pdb '%s': no heteroatoms found, %d residue(s) kept.",
            os.path.basename(input_pdb),
            kept_residues,
        )

    io = PDB.PDBIO()
    io.set_structure(structure)

    class ProteinSelect(PDB.Select):
        def accept_residue(self, residue):
            resname = residue.get_resname().strip()
            het_flag = residue.id[0].strip()
            if resname in WATER_NAMES or het_flag == "W":
                return False
            return PDB.is_aa(residue, standard=True)

        def accept_atom(self, atom):
            parent_res = atom.get_parent()
            het_flag = parent_res.id[0].strip()
            return het_flag == ""

    io.save(output_pdb, ProteinSelect())


def _sqdist(a, b) -> float:
    return sum((float(a[k]) - float(b[k])) ** 2 for k in range(3))


def _prune_degenerate_atoms(residue, coord_tol: float = 0.05) -> tuple[int, bool]:
    """Drop duplicate atom names; flag residues with overlapping coordinates."""
    from collections import defaultdict

    by_name: dict = defaultdict(list)
    for atom in list(residue.get_atoms()):
        by_name[atom.get_name().strip()].append(atom)

    pruned = 0
    for group in by_name.values():
        if len(group) <= 1:
            continue
        group.sort(key=lambda a: (a.altloc or " "))
        for atom in group[1:]:
            residue.detach_child(atom.id)
            pruned += 1

    atoms = list(residue.get_atoms())
    tol2 = coord_tol * coord_tol
    for i, a1 in enumerate(atoms):
        for a2 in atoms[i + 1 :]:
            if _sqdist(a1.coord, a2.coord) <= tol2:
                return pruned, True
    return pruned, False


def _residue_incomplete_for_adt(residue) -> bool:
    """True if residue lacks atoms MGLTools needs (e.g. truncated HIS ring)."""
    atoms = list(residue.get_atoms())
    if not atoms:
        return True
    names = {a.get_name().strip() for a in atoms}
    if "CA" not in names or len(names) < 4:
        return True
    resname = residue.get_resname().strip().upper()
    required = _SIDECHAIN_REQUIRED.get(resname)
    if required and not required.issubset(names):
        return True
    return False


def sanitize_pdb_for_adt(input_pdb: str, output_pdb: str) -> int:
    """Drop residues/chains that crash MGLTools (IndexError in prepare_receptor4).

    AlphaFold models may contain empty placeholders, fragments without CA,
    truncated side chains, or overlapping atoms (same XYZ → ZeroDivisionError).
    Returns the number of removed residues.
    """
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein", input_pdb)
    removed_empty = 0
    removed_incomplete = 0
    removed_degenerate = 0
    pruned_atoms = 0

    for model in structure:
        for chain in list(model):
            for residue in list(chain):
                atoms = list(residue.get_atoms())
                if not atoms:
                    chain.detach_child(residue.id)
                    removed_empty += 1
                    continue
                n_pruned, drop_overlap = _prune_degenerate_atoms(residue)
                pruned_atoms += n_pruned
                if drop_overlap:
                    resname = residue.get_resname().strip()
                    logging.info(
                        "sanitize_pdb_for_adt '%s': drop %s%s (overlapping atoms).",
                        os.path.basename(input_pdb),
                        resname,
                        residue.id[1],
                    )
                    chain.detach_child(residue.id)
                    removed_degenerate += 1
                    continue
                if _residue_incomplete_for_adt(residue):
                    resname = residue.get_resname().strip()
                    logging.info(
                        "sanitize_pdb_for_adt '%s': drop %s%s (incomplete for ADT).",
                        os.path.basename(input_pdb),
                        resname,
                        residue.id[1],
                    )
                    chain.detach_child(residue.id)
                    removed_incomplete += 1
            if len(list(chain.get_residues())) == 0:
                model.detach_child(chain.id)

    removed = removed_empty + removed_incomplete + removed_degenerate
    if removed or pruned_atoms:
        logging.info(
            "sanitize_pdb_for_adt '%s': removed %d residue(s) "
            "(%d empty, %d incomplete, %d degenerate), pruned %d duplicate atom(s).",
            os.path.basename(input_pdb),
            removed,
            removed_empty,
            removed_incomplete,
            removed_degenerate,
            pruned_atoms,
        )

    os.makedirs(os.path.dirname(os.path.abspath(output_pdb)) or ".", exist_ok=True)
    io = PDB.PDBIO()
    io.set_structure(structure)
    io.save(output_pdb)
    return removed


def _residue_plddt(residue) -> Optional[float]:
    """CA B-factor in AlphaFold PDB; mean B-factor if CA is missing."""
    ca_b = None
    bfacs = []
    for atom in residue.get_atoms():
        bfacs.append(float(atom.get_bfactor()))
        if atom.get_name().strip() == "CA":
            ca_b = float(atom.get_bfactor())
    if ca_b is not None:
        return ca_b
    if bfacs:
        return sum(bfac) / len(bfac)
    return None


def filter_pdb_by_plddt(
    input_pdb: str,
    output_pdb: str,
    threshold: float = 70.0,
) -> tuple[int, int]:
    """Drop protein residues with pLDDT below *threshold* (AF B-factor column).

    Returns ``(removed_count, kept_count)``.
    """
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein", input_pdb)
    removed = 0
    kept = 0

    for model in structure:
        for chain in list(model):
            for residue in list(chain):
                if not PDB.is_aa(residue, standard=True):
                    continue
                plddt = _residue_plddt(residue)
                if plddt is None or plddt < threshold:
                    resname = residue.get_resname().strip()
                    logging.debug(
                        "filter_pdb_by_plddt '%s': drop %s%s (pLDDT=%s).",
                        os.path.basename(input_pdb),
                        resname,
                        residue.id[1],
                        "NA" if plddt is None else f"{plddt:.1f}",
                    )
                    chain.detach_child(residue.id)
                    removed += 1
                else:
                    kept += 1
            if len(list(chain.get_residues())) == 0:
                model.detach_child(chain.id)

    logging.info(
        "filter_pdb_by_plddt '%s': removed %d low-pLDDT residue(s) "
        "(threshold %.1f), kept %d.",
        os.path.basename(input_pdb),
        removed,
        threshold,
        kept,
    )
    os.makedirs(os.path.dirname(os.path.abspath(output_pdb)) or ".", exist_ok=True)
    io = PDB.PDBIO()
    io.set_structure(structure)
    io.save(output_pdb)
    return removed, kept


class ProteinSelect(PDB.Select):
    """Keep standard amino-acid ATOM records (MSE retained)."""

    def accept_model(self, model):
        return True

    def accept_residue(self, residue):
        resname = residue.get_resname().strip().upper()
        if resname in SOLVENT_ION_BUFFER:
            return False
        return PDB.is_aa(residue, standard=True) or resname in {"MSE"}

    def accept_atom(self, atom):
        parent_res = atom.get_parent()
        het_flag = parent_res.id[0].strip()
        if het_flag == "":
            return True
        return parent_res.get_resname().strip().upper() in {"MSE"}


def format_pdb_line(
    record, serial, atom_name, altloc, resname, chain_id,
    resseq, icode, x, y, z, occ, bfac, element,
) -> str:
    """Build an ATOM/HETATM line with fixed PDB column offsets.

    Column layout matches BioPython / common PDB files (0-indexed):
      0-5 record, 6-10 serial, 12-15 atom, 16 altLoc, 17-19 resName,
      21 chainID, 22-25 resSeq, 26 iCode, 30-37 X, 38-45 Y, 46-53 Z,
      54-59 occupancy, 60-65 B, 76-77 element.
    Residue names longer than 3 characters (typical of AF3 CIF ligands)
    are truncated so downstream columns stay aligned.
    """
    line = list(" " * 78)
    rec = record[:6].ljust(6)
    line[0:6] = list(rec)
    line[6:11] = list(f"{serial:>5d}")
    line[12:16] = list(atom_name[:4].ljust(4))
    line[16] = (altloc[:1] if altloc else " ")
    line[17:20] = list(resname[:3].rjust(3))
    line[21] = str(chain_id)[:1]
    line[22:26] = list(f"{int(resseq):>4d}")
    line[26] = icode[:1] if icode else " "
    line[30:38] = list(f"{float(x):8.3f}")
    line[38:46] = list(f"{float(y):8.3f}")
    line[46:54] = list(f"{float(z):8.3f}")
    line[54:60] = list(f"{float(occ):6.2f}")
    line[60:66] = list(f"{float(bfac):6.2f}")
    line[76:78] = list(str(element)[:2].rjust(2))
    return "".join(line).rstrip() + "\n"


def write_ligand_pdb_direct(residue, output_path: str) -> None:
    """Write a single residue to PDB, bypassing BioPython PDBIO (CIF-safe)."""
    chain_id = residue.get_parent().id
    resseq = residue.id[1]
    icode = residue.id[2] if len(residue.id) > 2 else ""
    resname = residue.get_resname().strip()
    het_flag = residue.id[0].strip()
    record = "HETATM" if (het_flag and het_flag != " ") else "ATOM"

    with open(output_path, "w", encoding="utf-8") as fh:
        serial = 1
        for atom in residue:
            element = atom.element.strip().upper() if atom.element else ""
            if not element or element in ("", "NONE"):
                name = atom.name.strip()
                element = next((ch.upper() for ch in name if ch.isalpha()), "C")

            fullname = (atom.fullname or atom.name or "").strip() or atom.name.strip()
            if len(fullname) < 4:
                if len(element) == 1 and fullname[:1].isalpha():
                    fullname = " " + fullname
                fullname = fullname.ljust(4)
            atom_name = fullname[:4]
            altloc = atom.altloc if atom.altloc else " "
            try:
                x, y, z = atom.coord[0], atom.coord[1], atom.coord[2]
            except (TypeError, IndexError):
                continue
            occ = atom.occupancy if atom.occupancy is not None else 1.0
            bfac = atom.bfactor if atom.bfactor is not None else 0.0
            fh.write(
                format_pdb_line(
                    record, serial, atom_name, altloc, resname, chain_id,
                    resseq, icode, x, y, z, occ, bfac, element,
                )
            )
            serial += 1
        fh.write("END\n")


def detect_ligand(structure, min_heavy_atoms: int = 6, ligand_resname_hint=None):
    """Return the residue that should be treated as the ligand, or None."""
    candidates = []
    hint = ligand_resname_hint.strip().upper() if ligand_resname_hint else None

    for model in structure:
        for chain in model:
            for residue in chain:
                resname = residue.get_resname().strip().upper()
                if resname in STANDARD_AA or resname in SOLVENT_ION_BUFFER:
                    continue
                n_heavy = sum(
                    1 for a in residue if a.element.strip().upper() != "H"
                )
                if n_heavy < 1:
                    continue
                candidates.append((residue, resname, n_heavy))
        break

    if hint is not None:
        matching = [c for c in candidates if c[1] == hint]
        if matching:
            matching.sort(key=lambda c: c[2], reverse=True)
            return matching[0][0]
        logging.warning(
            "Ligand resname hint '%s' not found; falling back to size-based detection.",
            hint,
        )

    candidates = [c for c in candidates if c[2] >= min_heavy_atoms]
    if not candidates:
        return None
    candidates.sort(key=lambda c: c[2], reverse=True)
    return candidates[0][0]


def split_complex(
    input_path: str,
    apo_out: str,
    lig_out: str,
    min_heavy_atoms: int = 6,
    ligand_resname_hint=None,
):
    """Split a complex into apo-protein PDB + native ligand PDB.

    Returns (ligand_resname, chain_id, res_seq, n_heavy_atoms).
    """
    structure = load_structure(input_path)
    ligand = detect_ligand(
        structure,
        min_heavy_atoms=min_heavy_atoms,
        ligand_resname_hint=ligand_resname_hint,
    )
    if ligand is None:
        raise RuntimeError("no ligand-like residue found")

    chain_id = ligand.get_parent().id
    res_id = ligand.id
    resname = ligand.get_resname().strip().upper()
    n_heavy = sum(1 for a in ligand if a.element.strip().upper() != "H")

    io = PDB.PDBIO()
    io.set_structure(structure)
    io.save(apo_out, ProteinSelect())

    with open(apo_out, encoding="utf-8") as fh:
        n_apo = sum(1 for line in fh if line.startswith(("ATOM", "HETATM")))
    if n_apo == 0:
        raise RuntimeError("apo-protein PDB is empty after splitting")

    write_ligand_pdb_direct(ligand, lig_out)

    n_lig = 0
    n_coord_ok = 0
    with open(lig_out, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith(("ATOM", "HETATM")):
                n_lig += 1
                try:
                    float(line[30:38])
                    float(line[38:46])
                    float(line[46:54])
                    n_coord_ok += 1
                except (ValueError, IndexError):
                    pass
    if n_lig == 0:
        raise RuntimeError(
            f"ligand PDB is empty after extraction "
            f"(resname={resname}, chain={chain_id}, resseq={res_id[1]}, "
            f"detected {n_heavy} heavy atoms in BioPython object)"
        )
    if n_coord_ok == 0:
        with open(lig_out, encoding="utf-8") as fh:
            sample_lines = [
                line.rstrip() for line in fh if line.startswith(("ATOM", "HETATM"))
            ][:3]
        raise RuntimeError(
            f"ligand PDB has {n_lig} ATOM lines but 0 with valid coords. "
            "Sample lines:\n" + "\n".join(f"  |{l}|" for l in sample_lines)
        )

    logging.debug(
        "[split] ligand PDB written: %d atoms, %d with valid coords, resname=%s",
        n_lig, n_coord_ok, resname,
    )
    return resname, chain_id, res_id[1], n_heavy
