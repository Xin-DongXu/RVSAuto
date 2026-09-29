"""PDBFixer-based structure repair for failed receptor preparation."""

from __future__ import annotations

import logging
import os
from typing import Optional, Tuple

from .common import REPO_ROOT, run_command

_REPAIR_SCRIPT = REPO_ROOT / "scripts" / "pdbfixer_repair.py"


def verify_pdbfixer_env(env_path: Optional[str]) -> Optional[str]:
    """Run --self-test in the pdbfixer conda env; return error text or None."""
    if not env_path:
        return None
    if not _REPAIR_SCRIPT.is_file():
        return f"repair script missing: {_REPAIR_SCRIPT}"
    cmd = f'python "{_REPAIR_SCRIPT.as_posix()}" --self-test'
    try:
        run_command(cmd, env_path=env_path)
    except Exception as exc:
        return (
            f"{exc}. Fix with: conda install -c conda-forge "
            "'openmm>=8.0' 'pdbfixer>=1.9' --update-deps "
            f"(env: {env_path})"
        )
    return None


def repair_pdb_with_pdbfixer(
    input_pdb: str,
    output_pdb: str,
    env_path: Optional[str] = None,
    ph: float = 7.0,
) -> Tuple[bool, str]:
    """Run PDBFixer in a conda env; return (success, message)."""
    if not os.path.isfile(input_pdb):
        return False, f"input not found: {input_pdb}"
    if not _REPAIR_SCRIPT.is_file():
        return False, f"repair script missing: {_REPAIR_SCRIPT}"

    os.makedirs(os.path.dirname(os.path.abspath(output_pdb)) or ".", exist_ok=True)
    cmd = (
        f'python "{_REPAIR_SCRIPT.as_posix()}" '
        f'"{input_pdb}" "{output_pdb}" --ph {ph}'
    )
    log_path = output_pdb + ".log"
    cmd = f'{cmd} > "{log_path}" 2>&1'
    try:
        run_command(cmd, env_path=env_path)
    except Exception as exc:
        tail = _read_tail(log_path)
        return False, f"{exc}{tail}"

    if not os.path.isfile(output_pdb) or os.path.getsize(output_pdb) == 0:
        tail = _read_tail(log_path)
        return False, f"empty output{tail}"
    return True, "ok"


def _read_tail(log_path: str, n: int = 3) -> str:
    if not os.path.isfile(log_path):
        return ""
    try:
        lines = [
            ln for ln in open(log_path, encoding="utf-8", errors="replace")
            if ln.strip()
        ]
        if lines:
            return " | log: " + " / ".join(lines[-n:])
    except OSError:
        pass
    return ""
