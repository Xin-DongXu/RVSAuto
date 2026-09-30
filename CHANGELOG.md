# Changelog

All notable changes to this project are documented here.

## [1.1.1] - 2026-09-30

### Added

- **`rvsauto redock` progress bar** — overall completion (%), elapsed time, and ETA across preprocessing (split + PDBQT + conf) and UniDock redock/score jobs. Disable with `--no_progress`.

## [1.1.0] - 2026-09-29

### Added

- **`rvsauto screen` progress bar** — terminal display of overall completion (%), elapsed time, and ETA across receptor PDBQT conversion, pocket preparation (P2Rank / AF2BIND), and UniDock docking jobs. Disable with `--no_progress`.
- Dependency: `tqdm`.

## [1.0.0] - 2026-09-29

First public release.

### Features

- **`rvsauto screen`** — batch virtual screening with UniDock; pockets from **P2Rank** or precomputed **AF2BIND** CSV folders.
- **`rvsauto redock`** — self-redocking and heavy-atom RMSD for PDB/CIF complexes.
- Console aliases: `rvsauto-unidock`, `rvsauto-redock`.
- pip / conda-dev install (`pyproject.toml`, `conda/environment-*.yml`).
- AlphaFold-friendly receptor PDBQT pipeline: pLDDT retry, optional PDBFixer, docking box volume caps.
- P2Rank output layout auto-detection (`predict_receptors/` vs flat CSV).
- P2Rank pocket filter: default minimum calibrated **probability 0.05** (`--p2rank_min_probability 0` to disable).
- AF2BIND proteome-style clustering and notebook top-N modes; fix for transcript IDs with dots (e.g. `gene.1_jaile_model_apo`).

### Install

```bash
git clone https://github.com/Xin-DongXu/RVSAuto.git
cd RVSAuto
pip install .
rvsauto screen --help
```
