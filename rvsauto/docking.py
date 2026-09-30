"""UniDock invocation, PDBQT conversion, pose extraction."""

from __future__ import annotations

import logging
import os
import shutil
from typing import Dict, List, Optional, Tuple

from .common import as_conf_path, file_stem, run_command
from .pdbio import clean_pdb, format_pdb_line, sanitize_pdb_for_adt
from .pockets import DEFAULT_BOX_MAX_SIDE, UNIDOCK_MAX_GRID_VOLUME


def _log_unidock_failure_tail(log_file_path: str, max_lines: int = 30) -> None:
    if not os.path.isfile(log_file_path):
        return
    try:
        with open(log_file_path, encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
        tail = "".join(lines[-max_lines:]).strip()
        if tail:
            logging.error("UniDock log tail (%s):\n%s", log_file_path, tail)
    except OSError as exc:
        logging.debug("Could not read UniDock log %s: %s", log_file_path, exc)



def _sanitize_receptor_pdb(pdb_path: str) -> None:
    """In-place sanitize via temp file."""
    tmp = pdb_path + ".sanitize_tmp.pdb"
    try:
        sanitize_pdb_for_adt(pdb_path, tmp)
        shutil.move(tmp, pdb_path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def receptor_pdb_to_pdbqt(
    pdb_file: str,
    pdbqt_out: str,
    adt_env_path: Optional[str] = None,
    clean: bool = True,
    keep_clean_pdb: Optional[str] = None,
    adt_log: Optional[str] = None,
    adt_repairs: str = "hydrogens",
    timeout: Optional[float] = None,
) -> None:
    """Convert a receptor PDB to PDBQT via prepare_receptor4.py.

    When *keep_clean_pdb* is set, the protein-only PDB used for docking is
    saved there (needed later to build apo + docked-ligand complexes).
    *timeout* (seconds) is passed to the ADT subprocess; ``None`` means wait
    forever (not recommended for large batch runs).
    """
    if os.path.exists(pdbqt_out) and os.path.getsize(pdbqt_out) > 0:
        logging.info("PDBQT already exists: %s", os.path.basename(pdbqt_out))
        return

    if keep_clean_pdb and clean:
        os.makedirs(os.path.dirname(os.path.abspath(keep_clean_pdb)), exist_ok=True)
        if not os.path.exists(keep_clean_pdb) or os.path.getsize(keep_clean_pdb) == 0:
            clean_pdb(pdb_file, keep_clean_pdb)
        _sanitize_receptor_pdb(keep_clean_pdb)

    src = pdb_file
    cleaned_tmp = None
    if clean:
        src = keep_clean_pdb if keep_clean_pdb else None
        if not src:
            cleaned_tmp = os.path.join(
                os.path.dirname(os.path.abspath(pdbqt_out)),
                f"{os.path.splitext(os.path.basename(pdb_file))[0]}_cleaned.pdb",
            )
            clean_pdb(pdb_file, cleaned_tmp)
            src = cleaned_tmp
            _sanitize_receptor_pdb(src)
    elif keep_clean_pdb and os.path.isfile(keep_clean_pdb):
        src = keep_clean_pdb
    try:
        # AlphaFold/NMR models usually lack explicit H; add them instead of
        # requiring pre-existing hydrogens (-A checkhydrogens often fails).
        cmd = (
            f'prepare_receptor4.py -r "{src}" -o "{pdbqt_out}" '
            f'-A {adt_repairs} -U deleteAltB'
        )
        # Never use MGLTools -v on a shared terminal: it floods progress bars
        # under parallel preprocess. Prefer a log file; otherwise discard.
        if adt_log:
            os.makedirs(os.path.dirname(os.path.abspath(adt_log)), exist_ok=True)
            cmd = f'{cmd} > "{adt_log}" 2>&1'
        else:
            null = "NUL" if os.name == "nt" else "/dev/null"
            cmd = f'{cmd} > "{null}" 2>&1'
        run_command(cmd, env_path=adt_env_path, timeout=timeout)
        if not os.path.exists(pdbqt_out):
            raise RuntimeError(
                f"prepare_receptor4.py produced no output for {pdb_file}"
            )
    finally:
        if cleaned_tmp and os.path.exists(cleaned_tmp):
            os.remove(cleaned_tmp)


def _count_heavy_atoms_pdb(pdb_path: str) -> int:
    count = 0
    with open(pdb_path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            element = line[76:78].strip().upper() if len(line) >= 78 else ""
            if not element:
                name = line[12:16].strip().upper()
                element = name[:1]
            if element and element != "H" and not element.startswith("D"):
                count += 1
    return count


def _pdbqt_output_ok(path: str) -> bool:
    return os.path.isfile(path) and os.path.getsize(path) > 0


def _run_obabel_ligand_to_pdbqt(
    ligand_pdb: str,
    pdbqt_out: str,
    adt_env_path: Optional[str] = None,
) -> bool:
    """Convert ligand PDB to PDBQT via Open Babel (OB 2.x option order)."""
    in_abs = os.path.abspath(ligand_pdb)
    out_abs = os.path.abspath(pdbqt_out)
    if os.path.isfile(out_abs):
        os.remove(out_abs)

    # OB 2.4.x rejects some flag orderings; try explicit -ipdb/-opdbqt first.
    commands = (
        f'obabel -ipdb "{in_abs}" -opdbqt -O "{out_abs}" -h --partialcharge gasteiger',
        f'obabel "{in_abs}" -O "{out_abs}" -opdbqt -h --partialcharge gasteiger',
        f'obabel -ipdb "{in_abs}" -opdbqt -O "{out_abs}" -h',
        f'obabel "{in_abs}" -O "{out_abs}" -opdbqt -h',
    )
    for cmd in commands:
        run_command(cmd, env_path=adt_env_path, check=False, capture=True)
        if _pdbqt_output_ok(out_abs):
            return True
    return False


def _run_prepare_ligand4(
    ligand_pdb: str,
    pdbqt_out: str,
    adt_env_path: Optional[str] = None,
) -> bool:
    lig_dir = os.path.dirname(os.path.abspath(ligand_pdb)) or "."
    lig_base = os.path.basename(ligand_pdb)
    out_abs = os.path.abspath(pdbqt_out)
    if os.path.isfile(out_abs):
        os.remove(out_abs)
    cmd = (
        f'cd "{lig_dir}" && prepare_ligand4.py -l "{lig_base}" '
        f'-o "{out_abs}" -A hydrogens'
    )
    run_command(cmd, env_path=adt_env_path, check=False, capture=True)
    return _pdbqt_output_ok(out_abs)


def _run_ligand_via_mol2(
    ligand_pdb: str,
    pdbqt_out: str,
    adt_env_path: Optional[str] = None,
) -> bool:
    in_abs = os.path.abspath(ligand_pdb)
    out_abs = os.path.abspath(pdbqt_out)
    mol2_path = out_abs + ".mol2"
    try:
        if os.path.isfile(out_abs):
            os.remove(out_abs)
        if os.path.isfile(mol2_path):
            os.remove(mol2_path)

        mol2_cmds = (
            f'obabel -ipdb "{in_abs}" -omol2 -O "{mol2_path}" -h',
            f'obabel "{in_abs}" -O "{mol2_path}" -omol2 -h',
        )
        for cmd in mol2_cmds:
            run_command(cmd, env_path=adt_env_path, check=False, capture=True)
            if os.path.isfile(mol2_path) and os.path.getsize(mol2_path) > 0:
                break
        else:
            return False

        mol2_dir = os.path.dirname(mol2_path) or "."
        mol2_base = os.path.basename(mol2_path)
        cmd = (
            f'cd "{mol2_dir}" && prepare_ligand4.py -l "{mol2_base}" '
            f'-o "{out_abs}" -A hydrogens'
        )
        run_command(cmd, env_path=adt_env_path, check=False, capture=True)
        return _pdbqt_output_ok(out_abs)
    finally:
        if os.path.isfile(mol2_path):
            os.remove(mol2_path)


def _vec_sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _vec_add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _vec_scale(v, s):
    return (v[0] * s, v[1] * s, v[2] * s)


def _vec_norm(v):
    mag = (v[0] ** 2 + v[1] ** 2 + v[2] ** 2) ** 0.5
    if mag <= 1e-8:
        return (0.0, 0.0, 0.0)
    return (v[0] / mag, v[1] / mag, v[2] / mag)


def _vec_dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _vec_cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _rotation_matrix_from_vectors(src, dst):
    src_u = _vec_norm(src)
    dst_u = _vec_norm(dst)
    dot = max(min(_vec_dot(src_u, dst_u), 1.0), -1.0)
    if dot > 0.999999:
        return ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    if dot < -0.999999:
        axis = _vec_norm(_vec_cross((1.0, 0.0, 0.0), src_u))
        if axis == (0.0, 0.0, 0.0):
            axis = _vec_norm(_vec_cross((0.0, 1.0, 0.0), src_u))
        x, y, z = axis
        return (
            (1 - 2 * (y * y + z * z), 2 * (x * y), 2 * (x * z)),
            (2 * (x * y), 1 - 2 * (x * x + z * z), 2 * (y * z)),
            (2 * (x * z), 2 * (y * z), 1 - 2 * (x * x + y * y)),
        )
    cross = _vec_cross(src_u, dst_u)
    s = (1.0 - dot) ** 0.5
    c = dot
    x, y, z = cross[0] / s, cross[1] / s, cross[2] / s
    return (
        (c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s),
        (y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s),
        (z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)),
    )


def _mat_vec(m, v):
    return (
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    )


def _parse_ligand_pdb_heavy_atoms(pdb_path: str):
    atoms = []
    meta = {"resname": "LIG", "chain": "A", "resseq": 1}
    with open(pdb_path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            name = line[12:16].strip()
            meta["resname"] = line[17:20].strip() or meta["resname"]
            meta["chain"] = (line[21] or meta["chain"]).strip() or meta["chain"]
            try:
                meta["resseq"] = int(line[22:26])
            except ValueError:
                pass
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            element = line[76:78].strip().upper() if len(line) >= 78 else ""
            if not element:
                element = name[:1].upper()
            if element == "H" or element.startswith("D"):
                continue
            atoms.append(
                {"name": name, "element": element, "coord": (x, y, z)}
            )
    return atoms, meta


def _format_pdbqt_atom_line(
    serial: int,
    atom_name: str,
    resname: str,
    chain_id: str,
    resseq: int,
    coord,
    charge: float,
    ad_type: str,
) -> str:
    x, y, z = coord
    return (
        f"ATOM  {serial:5d}  {atom_name[:4]:<4}{resname:>3} {chain_id:1}"
        f"{resseq:4d}    {x:8.3f}{y:8.3f}{z:8.3f}"
        f"  0.00  0.00    {charge:6.3f} {ad_type:<2}\n"
    )


def _write_alkene_pdbqt_fallback(ligand_pdb: str, pdbqt_out: str) -> bool:
    """Build a PDBQT for a 2-carbon alkene fragment (e.g. ethylene / LIG1)."""
    heavy, meta = _parse_ligand_pdb_heavy_atoms(ligand_pdb)
    if len(heavy) != 2:
        return False
    if any(atom["element"] != "C" for atom in heavy):
        return False

    c1 = heavy[0]["coord"]
    c2 = heavy[1]["coord"]
    if ((_vec_sub(c2, c1)[0] ** 2 + _vec_sub(c2, c1)[1] ** 2 + _vec_sub(c2, c1)[2] ** 2) ** 0.5) < 0.5:
        return False

    template = [
        ("C", (0.0, 0.0, 0.665), "A", -0.078),
        ("C", (0.0, 0.0, -0.665), "A", -0.078),
        ("H", (0.0, 0.926, 1.232), "HD", 0.039),
        ("H", (0.0, -0.926, 1.232), "HD", 0.039),
        ("H", (0.0, 0.926, -1.232), "HD", 0.039),
        ("H", (0.0, -0.926, -1.232), "HD", 0.039),
    ]
    t_bond = _vec_sub(template[1][1], template[0][1])
    u_bond = _vec_sub(c2, c1)
    rot = _rotation_matrix_from_vectors(t_bond, u_bond)
    t_center = _vec_scale(_vec_add(template[0][1], template[1][1]), 0.5)
    u_center = _vec_scale(_vec_add(c1, c2), 0.5)

    lines = ["REMARK  Built-in ethylene/alkene PDBQT fallback\n", "ROOT\n"]
    serial = 1
    c_names = [heavy[0]["name"], heavy[1]["name"]]
    h_idx = 0
    for idx, (element, coord, ad_type, charge) in enumerate(template):
        rel = _vec_sub(coord, t_center)
        aligned = _mat_vec(rot, rel)
        xyz = _vec_add(aligned, u_center)
        if element == "C":
            atom_name = c_names[idx][:4] if idx < len(c_names) else f"C{idx + 1}"
        else:
            h_idx += 1
            atom_name = f"H{h_idx}"
        lines.append(
            _format_pdbqt_atom_line(
                serial,
                atom_name,
                meta["resname"][:3],
                meta["chain"][:1],
                meta["resseq"],
                xyz,
                charge,
                ad_type,
            )
        )
        serial += 1
    lines.append("ENDROOT\nTORSDOF 0\n")

    out_dir = os.path.dirname(os.path.abspath(pdbqt_out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(pdbqt_out, "w", encoding="utf-8") as fh:
        fh.writelines(lines)
    return _pdbqt_output_ok(pdbqt_out)


def _is_ethylene_like(ligand_pdb: str) -> bool:
    """True for 2-carbon alkene fragments (Boltz ETH / LIG1)."""
    heavy, _ = _parse_ligand_pdb_heavy_atoms(ligand_pdb)
    if len(heavy) != 2:
        return False
    return all(atom["element"] == "C" for atom in heavy)


def ligand_pdb_to_pdbqt(
    ligand_pdb: str,
    pdbqt_out: str,
    adt_env_path: Optional[str] = None,
) -> None:
    """Convert ligand PDB to PDBQT (ADT, Open Babel, mol2 bridge, alkene fallback)."""
    if _pdbqt_output_ok(pdbqt_out):
        return

    n_heavy = _count_heavy_atoms_pdb(ligand_pdb)
    attempts = []

    # Ethylene (2 x C) never works with prepare_ligand4/obabel on many ADT installs.
    if _is_ethylene_like(ligand_pdb):
        attempts.append("alkene_fallback")
        if _write_alkene_pdbqt_fallback(ligand_pdb, pdbqt_out):
            logging.info(
                "Ligand PDBQT via alkene fallback (ethylene) for %s", ligand_pdb
            )
            return

    # Other tiny fragments: try obabel before ADT.
    if n_heavy <= 4:
        attempts.append("obabel")
        if _run_obabel_ligand_to_pdbqt(ligand_pdb, pdbqt_out, adt_env_path):
            logging.info("Ligand PDBQT via obabel for %s", ligand_pdb)
            return

    attempts.append("prepare_ligand4")
    if _run_prepare_ligand4(ligand_pdb, pdbqt_out, adt_env_path):
        return

    if not attempts.count("obabel"):
        logging.warning(
            "prepare_ligand4.py failed for %s; trying obabel.", ligand_pdb
        )
        attempts.append("obabel")
        if _run_obabel_ligand_to_pdbqt(ligand_pdb, pdbqt_out, adt_env_path):
            logging.info("Ligand PDBQT via obabel for %s", ligand_pdb)
            return

    logging.warning("obabel failed for %s; trying mol2 + prepare_ligand4.", ligand_pdb)
    attempts.append("mol2")
    if _run_ligand_via_mol2(ligand_pdb, pdbqt_out, adt_env_path):
        logging.info("Ligand PDBQT via mol2 bridge for %s", ligand_pdb)
        return

    if n_heavy == 2 and "alkene_fallback" not in attempts:
        attempts.append("alkene_fallback")
        if _write_alkene_pdbqt_fallback(ligand_pdb, pdbqt_out):
            logging.info("Ligand PDBQT via alkene fallback for %s", ligand_pdb)
            return

    raise RuntimeError(
        f"ligand PDBQT conversion failed for {ligand_pdb} "
        f"(tried: {', '.join(attempts)})"
    )


def extract_binding_affinity(docking_output_path: str) -> Optional[float]:
    """Read the first REMARK VINA/UNIDOCK RESULT line."""
    if not os.path.exists(docking_output_path):
        logging.warning("Docking output file not found: %s", docking_output_path)
        return None
    try:
        with open(docking_output_path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if line.startswith(
                    ("REMARK VINA RESULT:", "REMARK UNIDOCK RESULT:")
                ):
                    parts = line.split()
                    if len(parts) > 3:
                        return float(parts[3])
        logging.warning("No binding affinity remark in %s", docking_output_path)
        return None
    except Exception as exc:
        logging.error("Error reading %s: %s", docking_output_path, exc)
        return None


_PDBQT_TYPE_TO_ELEMENT = {
    "A": "C", "C": "C", "OA": "O", "OS": "O", "O": "O",
    "NA": "N", "NS": "N", "N": "N", "SA": "S", "S": "S",
    "HD": "H", "HS": "H", "H": "H", "P": "P", "F": "F",
    "CL": "CL", "BR": "BR", "I": "I", "MG": "MG", "ZN": "ZN",
    "MN": "MN", "CA": "CA", "FE": "FE",
}


def pdbqt_atom_type_to_element(atype: str) -> str:
    key = atype.strip().upper()
    if key in _PDBQT_TYPE_TO_ELEMENT:
        return _PDBQT_TYPE_TO_ELEMENT[key]
    if len(key) >= 2 and key[:2] in _PDBQT_TYPE_TO_ELEMENT:
        return _PDBQT_TYPE_TO_ELEMENT[key[:2]]
    return key[:1] if key else "C"


def _element_from_pdbqt_line(line: str) -> str:
    if len(line) >= 78 and line[76:78].strip():
        raw = line[76:78].strip()
        return pdbqt_atom_type_to_element(raw)
    parts = line.split()
    if parts:
        return pdbqt_atom_type_to_element(parts[-1])
    name = line[12:16].strip() if len(line) >= 16 else "C"
    return name[0].upper() if name else "C"


def extract_best_ligand_pose(
    pdbqt_file: str, output_pdb: str, as_hetatm: bool = True
) -> bool:
    """Extract MODEL 1 from a docked PDBQT and write a plain ligand PDB.

    UniDock/Vina PDBQT may omit MODEL records; atoms before the first ENDMDL
    (or the whole file) are then used. Ligand atoms are written as HETATM
    with the AutoDock atom type mapped back to an element symbol.
    """
    if not os.path.exists(pdbqt_file):
        logging.error("PDBQT file not found: %s", pdbqt_file)
        return False
    try:
        with open(pdbqt_file, encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
    except Exception as exc:
        logging.error("Read failed: %s: %s", pdbqt_file, exc)
        return False

    in_model = False
    ligand_lines = []
    saw_model = any(line.startswith("MODEL") for line in lines)
    for line in lines:
        if line.startswith("MODEL"):
            in_model = True
            continue
        if line.startswith("ENDMDL"):
            break
        take = in_model or not saw_model
        if take and line.startswith(("ATOM", "HETATM")):
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except (ValueError, IndexError):
                continue
            occ, bfac = 1.0, 0.0
            try:
                if len(line) >= 60:
                    occ = float(line[54:60])
                if len(line) >= 66:
                    bfac = float(line[60:66])
            except ValueError:
                pass
            atom_name = line[12:16] if len(line) >= 16 else " C  "
            altloc = line[16] if len(line) > 16 else " "
            resname = line[17:20].strip() if len(line) >= 20 else "LIG"
            chain = line[21] if len(line) > 21 else "Z"
            try:
                resseq = int(line[22:26])
            except (ValueError, IndexError):
                resseq = 1
            icode = line[26] if len(line) > 26 else " "
            element = _element_from_pdbqt_line(line)
            try:
                serial = int(line[6:11])
            except (ValueError, IndexError):
                serial = len(ligand_lines) + 1
            record = "HETATM" if as_hetatm else line[:6]
            ligand_lines.append(
                format_pdb_line(
                    record, serial, atom_name, altloc, resname or "LIG",
                    chain or "Z", resseq, icode, x, y, z, occ, bfac, element,
                )
            )

    if not ligand_lines:
        logging.warning("No ligand pose found in %s", pdbqt_file)
        return False
    with open(output_pdb, "w", encoding="utf-8") as fh:
        fh.writelines(ligand_lines)
        fh.write("END\n")
    return True


def _pick_ligand_chain(used_chains) -> str:
    for cand in "ZLYXWVUTSRQPONM":
        if cand not in used_chains:
            return cand
    return "Z"


def _receptor_serials_and_chains(receptor_pdb: str):
    max_serial = 0
    chains = set()
    n_atoms = 0
    with open(receptor_pdb, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith(("ATOM", "HETATM")):
                n_atoms += 1
                try:
                    max_serial = max(max_serial, int(line[6:11]))
                except (ValueError, IndexError):
                    max_serial = max(max_serial, n_atoms)
                if len(line) > 21:
                    ch = line[21].strip()
                    if ch:
                        chains.add(ch)
    return max_serial, chains


def create_complex_pdb(
    receptor_pdb: str,
    ligand_pdb: str,
    output_complex_pdb: str,
    ligand_chain: Optional[str] = None,
    ligand_resname: str = "LIG",
) -> bool:
    """Concatenate apo-protein and docked ligand into one complex PDB.

    The receptor file should already be protein-only (the cleaned PDB used
    for docking). The ligand is rewritten as HETATM on a free chain (default
    Z) with serial numbers continuing after the receptor, so PyMOL/ChimeraX
    do not merge it into a protein residue.
    """
    if not os.path.exists(receptor_pdb):
        logging.error("Receptor PDB not found: %s", receptor_pdb)
        return False
    if not os.path.exists(ligand_pdb):
        logging.error("Ligand PDB not found: %s", ligand_pdb)
        return False
    try:
        max_serial, used_chains = _receptor_serials_and_chains(receptor_pdb)
        chain = ligand_chain or _pick_ligand_chain(used_chains)
        os.makedirs(os.path.dirname(os.path.abspath(output_complex_pdb)) or ".", exist_ok=True)
        with open(output_complex_pdb, "w", encoding="utf-8") as outfile:
            with open(receptor_pdb, encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    if line.startswith(("END", "MASTER", "CONECT", "ENDMDL")):
                        continue
                    outfile.write(line)
            outfile.write("TER\n")
            serial = max_serial + 1
            with open(ligand_pdb, encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    if not line.startswith(("ATOM", "HETATM")):
                        continue
                    try:
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                    except (ValueError, IndexError):
                        continue
                    occ, bfac = 1.0, 0.0
                    try:
                        if len(line) >= 60:
                            occ = float(line[54:60])
                        if len(line) >= 66:
                            bfac = float(line[60:66])
                    except ValueError:
                        pass
                    atom_name = line[12:16] if len(line) >= 16 else " C  "
                    altloc = line[16] if len(line) > 16 else " "
                    icode = " "
                    element = line[76:78].strip() if len(line) >= 78 else ""
                    if not element:
                        element = atom_name.strip()[:1].upper() or "C"
                    outfile.write(
                        format_pdb_line(
                            "HETATM", serial, atom_name, altloc, ligand_resname,
                            chain, 1, icode, x, y, z, occ, bfac, element,
                        )
                    )
                    serial += 1
            outfile.write("END\n")
        return True
    except Exception as exc:
        logging.error("Error creating complex PDB %s: %s", output_complex_pdb, exc)
        return False


def _unidock_extra_flags(
    num_modes: Optional[int] = None,
    exhaustiveness: Optional[int] = None,
    seed: Optional[int] = None,
) -> str:
    extra = ""
    if num_modes is not None:
        extra += f" --num_modes {num_modes}"
    if exhaustiveness is not None:
        extra += f" --exhaustiveness {exhaustiveness}"
    if seed is not None:
        extra += f" --seed {seed}"
    return extra


def run_unidock_batch(
    conf_file: str,
    ligand_list: List[str],
    output_pdbqt_files: Dict[str, str],
    log_file_path: str,
    search_mode: str,
    gpu_id: Optional[str],
    dock_env_path: Optional[str] = None,
    unidock_bin: str = "unidock",
    num_modes: Optional[int] = None,
    exhaustiveness: Optional[int] = None,
    seed: Optional[int] = None,
) -> List[Tuple[str, float, str]]:
    """Dock several ligands against one receptor pocket (gpu_batch)."""
    all_exist = all(
        os.path.exists(f) and os.path.getsize(f) > 0
        for f in output_pdbqt_files.values()
    )
    base_filename = file_stem(conf_file)

    def _collect_existing() -> List[Tuple[str, float, str]]:
        results = []
        for ligand_path, output_pdbqt_file in output_pdbqt_files.items():
            lig_name = file_stem(ligand_path)
            if os.path.exists(output_pdbqt_file):
                affinity = extract_binding_affinity(output_pdbqt_file)
                if affinity is not None:
                    results.append((output_pdbqt_file, affinity, lig_name))
        return results

    if all_exist:
        logging.info("All outputs for %s already exist; skip UniDock.", base_filename)
        return _collect_existing()

    first_output = next(iter(output_pdbqt_files.values()))
    task_subdir = os.path.join(os.path.dirname(first_output), base_filename)
    os.makedirs(task_subdir, exist_ok=True)
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    ligand_batch = " ".join(f'"{lig}"' for lig in ligand_list)
    gpu_env = f"CUDA_VISIBLE_DEVICES={gpu_id} " if gpu_id is not None else ""
    extra = _unidock_extra_flags(num_modes, exhaustiveness, seed)
    docking_command = (
        f'{gpu_env}{unidock_bin} --config "{as_conf_path(conf_file)}" '
        f"--search_mode {search_mode} --gpu_batch {ligand_batch} "
        f'--dir "{task_subdir}"{extra} > "{log_file_path}" 2>&1'
    )
    logging.info(
        "UniDock %s: %d ligand(s) on GPU %s",
        base_filename, len(ligand_list), gpu_id,
    )
    try:
        proc = run_command(docking_command, env_path=dock_env_path, check=False)
    except Exception as exc:
        logging.error("UniDock command failed for %s: %s", base_filename, exc)
        proc = None

    if proc is not None and proc.returncode != 0:
        if proc.returncode in (139, -11):
            logging.error(
                "UniDock crashed (rc=%s) for %s — grid box likely too large "
                "(UniDock limit: volume <= %.0f A^3). Regenerate pocket configs "
                "with --rewrite_pocket_configs (default max side %.0f A).",
                proc.returncode, base_filename, UNIDOCK_MAX_GRID_VOLUME,
                DEFAULT_BOX_MAX_SIDE,
            )
        _log_unidock_failure_tail(log_file_path)

    results: List[Tuple[str, float, str]] = []
    for ligand_path, output_pdbqt_file in output_pdbqt_files.items():
        ligand_name = file_stem(ligand_path)
        temp_output = os.path.join(task_subdir, f"{ligand_name}_out.pdbqt")
        if not os.path.exists(temp_output):
            logging.warning("Expected output missing: %s", temp_output)
            try:
                logging.info("Files in %s: %s", task_subdir, os.listdir(task_subdir))
            except OSError:
                pass
            continue
        try:
            os.replace(temp_output, output_pdbqt_file)
        except OSError as exc:
            logging.error("Could not move %s -> %s: %s", temp_output, output_pdbqt_file, exc)
            continue
        if os.path.getsize(output_pdbqt_file) == 0:
            logging.warning("Output file is empty: %s", output_pdbqt_file)
            continue
        affinity = extract_binding_affinity(output_pdbqt_file)
        if affinity is not None:
            results.append((output_pdbqt_file, affinity, ligand_name))
            logging.info(
                "  %s + %s: %.3f kcal/mol", base_filename, ligand_name, affinity
            )
        else:
            logging.warning("Could not extract affinity from %s", output_pdbqt_file)

    try:
        os.rmdir(task_subdir)
    except OSError:
        pass
    return results


def run_unidock_single(
    conf_file: str,
    ligand_pdbqt: str,
    output_pdbqt: str,
    log_file: str,
    search_mode: str,
    gpu_id: Optional[str],
    dock_env_path: Optional[str] = None,
    unidock_bin: str = "unidock",
    num_modes: Optional[int] = None,
    exhaustiveness: Optional[int] = None,
    seed: Optional[int] = None,
) -> Optional[float]:
    """Dock one ligand; return the best-pose affinity or None."""
    if os.path.exists(output_pdbqt) and os.path.getsize(output_pdbqt) > 0:
        return extract_binding_affinity(output_pdbqt)

    out_dir = os.path.dirname(output_pdbqt)
    os.makedirs(out_dir, exist_ok=True)
    task_subdir = os.path.join(out_dir, f".tmp_{os.path.basename(output_pdbqt)}")
    os.makedirs(task_subdir, exist_ok=True)

    gpu_env = f"CUDA_VISIBLE_DEVICES={gpu_id} " if gpu_id is not None else ""
    extra = _unidock_extra_flags(num_modes, exhaustiveness, seed)
    cmd = (
        f'{gpu_env}{unidock_bin} --config "{as_conf_path(conf_file)}" '
        f"--search_mode {search_mode} --gpu_batch \"{ligand_pdbqt}\" "
        f'--dir "{task_subdir}"{extra} > "{log_file}" 2>&1'
    )
    logging.info("unidock: %s on GPU %s", os.path.basename(ligand_pdbqt), gpu_id)
    run_command(cmd, env_path=dock_env_path, check=False)

    lig_name = file_stem(ligand_pdbqt)
    produced = os.path.join(task_subdir, f"{lig_name}_out.pdbqt")
    if not os.path.exists(produced) or os.path.getsize(produced) == 0:
        logging.error("UniDock produced no output for %s", ligand_pdbqt)
        shutil.rmtree(task_subdir, ignore_errors=True)
        return None
    shutil.move(produced, output_pdbqt)
    shutil.rmtree(task_subdir, ignore_errors=True)
    return extract_binding_affinity(output_pdbqt)
