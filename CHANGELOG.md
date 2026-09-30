# Changelog

All notable changes to this project are documented here.

## [1.1.6] - 2026-09-30

### Fixed

- **Hung `prepare_receptor4.py` no longer blocks the whole PDBQT stage** — subprocesses now have a default **600 s** timeout (`--adt_timeout`); the process group is killed on expiry so batch jobs can continue.
- Failed shell commands are **no longer re-executed** just for diagnostics (that doubled ADT work and could hang twice).
- Receptor PDBQT progress shows **in-flight protein IDs** (`active=…`) so the last stuck structure is visible under `--quiet`.

## [1.1.5] - 2026-09-30

### Changed

- **Per-phase progress bars** for `rvsauto screen` and `rvsauto redock` — each stage (Receptor PDBQT, AF2BIND/P2Rank, Filter pockets, Pocket configs, UniDock, …) now has its **own** 0–100% bar instead of one combined counter that made mid-pipeline percentages misleading.

## [1.1.4] - 2026-09-30

### Added

- **`rvsauto screen` progress** for post-pocket stages: PDBQT pocket filtering and writing docking `.conf` files (`Pocket configs`), so long proteome runs no longer appear frozen after AF2BIND/P2Rank reaches 100%.

## [1.1.3] - 2026-09-30

### Fixed

- **`--quiet` / progress bar** — ADT `prepare_receptor4.py` no longer dumps verbose `-v` output to the terminal (that spam was breaking the bar even with `--quiet`). Output goes to per-case log files or `/dev/null`.
- External tool stdout/stderr is discarded when `--quiet` is set **or** the progress bar is enabled.

## [1.1.2] - 2026-09-30

### Added

- **`--quiet`** for `rvsauto screen` and `rvsauto redock` — suppress console log lines so only the progress bar updates on screen; full logs still go to the log file. End-of-run summary is kept.

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
