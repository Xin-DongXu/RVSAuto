"""Shared helpers: logging, conda wrapping, GPU detection, external binaries."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parent.parent


def setup_logging(output_dir: str, prefix: str = "run") -> str:
    """Log to both a timestamped file under output_dir/logs and stderr."""
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{prefix}_{timestamp}.log")

    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return log_file


def _candidate_conda_hooks() -> Iterable[Path]:
    env_exe = os.environ.get("CONDA_EXE")
    if env_exe:
        # .../condabin/conda or .../bin/conda
        prefix = Path(env_exe).resolve().parent.parent
        yield prefix / "etc" / "profile.d" / "conda.sh"

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        # env prefix -> base prefix is parent of envs/<name>
        p = Path(conda_prefix)
        yield p / "etc" / "profile.d" / "conda.sh"
        if p.parent.name == "envs":
            yield p.parent.parent / "etc" / "profile.d" / "conda.sh"

    home = Path.home()
    for dist in ("miniforge3", "mambaforge", "miniconda3", "anaconda3"):
        yield home / dist / "etc" / "profile.d" / "conda.sh"
        yield home / dist / "bin" / "activate"


def find_conda_sh() -> Optional[Path]:
    for path in _candidate_conda_hooks():
        if path.is_file():
            return path
    return None


def wrap_env_command(command: str, env_path: Optional[str] = None) -> str:
    """Prefix *command* so it runs inside a conda env when env_path is given.

    *env_path* may be an environment name or an absolute env prefix.  Uni-Dock
    is Linux-only; the wrapper prefers `conda.sh` and falls back to a direct
    `bin/activate` path for HPC images that ship miniforge.
    """
    if not env_path:
        return command
    env_path = os.path.expanduser(str(env_path))
    hook = find_conda_sh()
    if hook is not None:
        if hook.name == "activate":
            return f'source "{hook}" "{env_path}" && {command}'
        return (
            f'source "{hook}" && conda activate "{env_path}" && {command}'
        )
    # Last-resort HPC layout used by the original scripts
    fallback = Path.home() / "miniforge3" / "bin" / "activate"
    return f'source "{fallback}" "{env_path}" && {command}'


def bash_executable() -> Optional[str]:
    if os.name == "nt":
        return shutil.which("bash")
    for candidate in ("/bin/bash", "/usr/bin/bash"):
        if os.path.isfile(candidate):
            return candidate
    return shutil.which("bash")


def run_command(
    command: str,
    env_path: Optional[str] = None,
    check: bool = True,
    capture: bool = False,
) -> subprocess.CompletedProcess:
    """Run a shell command, optionally inside a conda environment."""
    full = wrap_env_command(command, env_path)
    bash = bash_executable()
    kwargs = dict(shell=True, check=False)
    if bash and os.name != "nt":
        kwargs["executable"] = bash
    if capture:
        kwargs.update(capture_output=True, text=True)
    logging.debug("Running: %s", full)
    try:
        proc = subprocess.run(full, **kwargs)
    except FileNotFoundError as exc:
        logging.error("Failed to start command: %s (%s)", full, exc)
        raise
    if proc.returncode != 0:
        logging.error("Command failed (rc=%s): %s", proc.returncode, full)
        if check and not capture:
            try:
                diag = subprocess.run(
                    full,
                    shell=True,
                    capture_output=True,
                    text=True,
                    executable=bash if bash and os.name != "nt" else None,
                )
                if diag.stdout and diag.stdout.strip():
                    logging.error("Command stdout:\n%s", diag.stdout.strip()[-4000:])
                if diag.stderr and diag.stderr.strip():
                    logging.error("Command stderr:\n%s", diag.stderr.strip()[-4000:])
            except Exception as exc:
                logging.debug("Could not capture failed command output: %s", exc)
        if check:
            raise subprocess.CalledProcessError(proc.returncode, full)
    return proc


def resolve_gpu_list(gpu_ids_arg: Optional[str]) -> Tuple[list, int]:
    """Return (gpu_id_strings, num_gpus)."""
    if gpu_ids_arg:
        gpu_list = [g.strip() for g in gpu_ids_arg.split(",") if g.strip()]
        if gpu_list:
            return gpu_list, len(gpu_list)
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            check=False,
        )
        gpus = [ln for ln in result.stdout.splitlines() if "GPU" in ln]
        if gpus:
            gpu_list = [str(i) for i in range(len(gpus))]
            return gpu_list, len(gpu_list)
    except Exception:
        pass
    logging.warning("No GPU detected via nvidia-smi; defaulting to GPU 0.")
    return ["0"], 1


def _is_executable(path: Path) -> bool:
    if not path.is_file():
        return False
    if os.name == "nt":
        return path.suffix.lower() in {".exe", ".bat", ".cmd", ""} or path.name in {
            "prank",
            "unidock",
        }
    return os.access(path, os.X_OK) or path.name in {"prank", "unidock"}


def find_p2rank(explicit: Optional[str] = None) -> Optional[Path]:
    """Locate the P2Rank launcher (prank / prank.bat) next to this repo or on PATH."""
    if explicit:
        p = Path(os.path.expanduser(explicit)).resolve()
        if p.exists():
            return p
        raise FileNotFoundError(f"P2Rank executable not found: {explicit}")

    names = ("prank.bat", "prank") if os.name == "nt" else ("prank", "prank.bat")
    search_roots = [REPO_ROOT, Path.cwd()]
    for root in search_roots:
        for child in sorted(root.iterdir()) if root.is_dir() else []:
            if child.is_dir() and child.name.lower().startswith("p2rank"):
                for name in names:
                    cand = child / name
                    if cand.exists():
                        return cand.resolve()
        for name in names:
            cand = root / name
            if cand.exists():
                return cand.resolve()

    which = shutil.which("prank")
    return Path(which).resolve() if which else None


def find_unidock(explicit: Optional[str] = None) -> str:
    """Return a UniDock binary path or the name `unidock` if it is expected on PATH."""
    if explicit:
        p = Path(os.path.expanduser(explicit)).resolve()
        if p.exists():
            return str(p)
        raise FileNotFoundError(f"UniDock executable not found: {explicit}")

    which = shutil.which("unidock")
    if which:
        return which

    hints = [
        REPO_ROOT / "unidock",
        REPO_ROOT / "bin" / "unidock",
        REPO_ROOT / "Uni-Dock-main" / "Uni-Dock-main" / "unidock" / "build" / "unidock",
        REPO_ROOT / "Uni-Dock-main" / "unidock" / "build" / "unidock",
    ]
    for cand in hints:
        if _is_executable(cand):
            return str(cand.resolve())
    return "unidock"


def file_stem(path: str) -> str:
    """Return the filename without the last suffix (foo.pdbqt -> foo)."""
    return os.path.splitext(os.path.basename(path))[0]


def as_conf_path(path: str) -> str:
    """Absolute POSIX path for UniDock/Vina config files (Linux GPU binaries)."""
    return Path(path).resolve().as_posix()
