# Maintainer & contributor notes

End users only need the root [README](../README.md). This file is for people who develop, test, or release RVSAuto.

## Repository layout (release hygiene)

Do **not** commit: `p2rank_*`, `Uni-Dock-main/`, `AF2BIND_out/`, `workdir/`, `results/`, `logs/`, `*.pdbqt`, or personal HPC workdirs (`xdxu_workdir/`, etc.). Run `git status` before every push.

## Development install

```bash
git clone https://github.com/Xin-DongXu/RVSAuto.git
cd RVSAuto
pip install -e ".[dev]"
```

Or use the dev conda env (pytest, editable install):

```bash
conda env create -f conda/environment-dev.yml
conda activate rvsauto-dev
```

## Tests

```bash
pip install -e ".[dev]"
pytest -v
```

No GPU required. Tests that need a local `AF2BIND_out/` sample folder are skipped if absent.

## GitHub releases

Tag versions that match `rvsauto.__version__` in `rvsauto/__init__.py`:

```bash
git tag -a v11.0.0 -m "RVSAuto 11.0.0"
git push origin main
git push origin v11.0.0
```

Draft the release on GitHub using [CHANGELOG.md](../CHANGELOG.md).

Helper scripts (Windows / Linux):

- `scripts/publish_final.ps1` — after `gh auth login`
- `scripts/release_to_github.sh` — first push from a clean tree

## Optional: PyPI upload

RVSAuto is **not** on PyPI by default. Users install from GitHub with `pip install .`.

If you publish to PyPI later:

```bash
pip install build twine
python -m build
twine upload dist/*
```
