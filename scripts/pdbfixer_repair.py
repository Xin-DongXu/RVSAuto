#!/usr/bin/env python3
"""Repair a protein PDB with PDBFixer (missing atoms, non-standard residues, H).

Designed for AlphaFold models that fail MGLTools prepare_receptor4.py.
Internal missing-residue gaps are not rebuilt; only termini gaps may be filled.
"""

from __future__ import annotations

import argparse
import sys
import types


def _install_openmm_legacy_shim() -> None:
    """Map simtk.openmm.* to openmm.* for pdbfixer builds against OpenMM >= 7.6."""
    import sys

    try:
        import openmm
        from openmm import app
    except ImportError as exc:
        raise ImportError(
            "openmm is not installed. Use: "
            "conda install -c conda-forge 'openmm>=8.0' 'pdbfixer>=1.9'"
        ) from exc

    if "simtk.openmm" in sys.modules:
        return

    simtk = types.ModuleType("simtk")
    simtk.openmm = openmm
    sys.modules["simtk"] = simtk
    sys.modules["simtk.openmm"] = openmm
    sys.modules["simtk.openmm.app"] = app

    try:
        from openmm.app import internal as app_internal
    except ImportError:
        app_internal = None
    if app_internal is not None:
        sys.modules["simtk.openmm.app.internal"] = app_internal

    try:
        from openmm import unit
    except ImportError:
        unit = None
    if unit is not None:
        sys.modules["simtk.unit"] = unit
        simtk.unit = unit


def _import_pdbfixer():
    _install_openmm_legacy_shim()
    from openmm.app import PDBFile
    from pdbfixer import PDBFixer

    return PDBFile, PDBFixer


def self_test() -> None:
    PDBFile, PDBFixer = _import_pdbfixer()
    import openmm

    print(f"openmm {getattr(openmm, '__version__', '?')}")
    print(f"pdbfixer {getattr(PDBFixer, '__module__', 'pdbfixer')}")
    print("ok")


def repair_structure(input_pdb: str, output_pdb: str, ph: float = 7.0) -> None:
    PDBFile, PDBFixer = _import_pdbfixer()

    fixer = PDBFixer(filename=input_pdb)

    fixer.findMissingResidues()
    chains = list(fixer.topology.chains())
    for chain_index, chain in enumerate(chains):
        n_res = len(list(chain.residues()))
        for key in list(fixer.missingResidues.keys()):
            if key[0] == chain_index and key[1] != 0 and key[1] != n_res:
                del fixer.missingResidues[key]

    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(ph)

    with open(output_pdb, "w", encoding="utf-8") as fh:
        PDBFile.writeFile(fixer.topology, fixer.positions, fh, keepIds=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Repair a PDB with PDBFixer.")
    parser.add_argument("input_pdb", nargs="?", help="Input cleaned apo PDB.")
    parser.add_argument("output_pdb", nargs="?", help="Repaired PDB output path.")
    parser.add_argument("--ph", type=float, default=7.0, help="pH for protonation.")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Verify openmm + pdbfixer imports in this environment.",
    )
    args = parser.parse_args(argv)
    if args.self_test:
        try:
            self_test()
        except Exception as exc:
            print(f"PDBFixer self-test failed: {exc}", file=sys.stderr)
            return 1
        return 0
    if not args.input_pdb or not args.output_pdb:
        parser.error("input_pdb and output_pdb are required unless --self-test is set")
    try:
        repair_structure(args.input_pdb, args.output_pdb, ph=args.ph)
    except Exception as exc:
        print(f"PDBFixer repair failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
