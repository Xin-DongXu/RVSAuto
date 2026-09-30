# RVSAuto

Batch UniDock pipelines for:

1. **Virtual screening** -- pocket prediction (P2Rank or precomputed AF2BIND) to docking box to GPU docking
2. **Self-redocking + RMSD** -- split a complex, re-dock the native ligand, report heavy-atom RMSD

UniDock is Linux-only (NVIDIA GPU, compute capability >= 7.0). Install RVSAuto with pip and set up UniDock, MGLTools/ADT, and (for screening) P2Rank as below. P2Rank and UniDock are **not** bundled in this repository.

**Commands:** `rvsauto screen` (virtual screening) and `rvsauto redock` (self-redocking + RMSD). Aliases: `rvsauto-unidock`, `rvsauto-redock`.

---

## Installation

### 1. pip (Python package)

From a git clone:

```bash
git clone https://github.com/Xin-DongXu/RVSAuto.git
cd RVSAuto
pip install .
```

After install, these commands are on your `PATH`:

```bash
rvsauto --help
rvsauto screen --help
rvsauto redock --help
python -m rvsauto screen --help   # same as above
```

Core Python dependencies (`numpy`, `pandas`, `biopython`) are installed automatically.

### 2. Conda environments (external tools)

RVSAuto orchestrates external binaries via conda env paths. Create all recommended envs:

```bash
bash scripts/setup_conda.sh
```

Or one at a time:

```bash
conda env create -f conda/environment-adt.yml       # MGLTools / obabel
conda env create -f conda/environment-unidock.yml   # UniDock GPU
conda env create -f conda/environment-pdbfixer.yml  # optional PDB repair
```

Pass conda env **names** or absolute prefixes to the pipeline, e.g. `--adt_env_path adt_env`, `--dock_env_path unidock_env`.

### 3. UniDock (conda)

UniDock must be installed in a conda environment. A GPU-enabled Linux box is required.

```bash
conda activate unidock_env
unidock --help
```

Pass that environment to the pipeline:

```bash
--dock_env_path unidock_env
# or an absolute prefix:
--dock_env_path /home/user/miniforge3/envs/unidock_env
```

### 4. Receptor / ligand PDBQT preparation (conda)

`prepare_receptor4.py` and `prepare_ligand4.py` come from MGLTools / ADFRsuite. The conda package is:

```bash
conda activate adt_env
prepare_receptor4.py -h
```

Pass it as `--adt_env_path adt_env` (or the env prefix). ADFRsuite from the Scripps CCSB download page also works if those scripts are on `PATH` inside the env you pass.

Optional **PDBFixer** for AlphaFold structures that fail `prepare_receptor4.py` (missing atoms, non-standard residues):

```bash
conda activate pdbfixer_env
python scripts/pdbfixer_repair.py --self-test
# or: bash scripts/check_pdbfixer_env.sh $CONDA_PREFIX
```

If you see `No module named 'simtk.openmm.app.internal'`, the env has mismatched packages (old pdbfixer + new openmm). Recreate the env with the pins in `conda/environment-pdbfixer.yml`, or run `conda install -c conda-forge "openmm>=8.0" "pdbfixer>=1.9" --update-deps`.

Pass `--pdbfixer_env_path pdbfixer_env`. After a failed MGLTools conversion the pipeline repairs the cleaned apo PDB once and retries. Repaired structures are saved as `workdir/receptors_clean/{stem}_pdbfixer.pdb`. Proteins that still fail are listed in `results/receptor_pdbqt_failures.tsv` and skipped (unless `--strict_receptors`).

### 5. P2Rank (not conda)

Download the **full** binary release and unpack it on the machine where you run the pipeline (project directory, or any path passed as `--p2rank_path`):

https://github.com/rdk/p2rank/releases/download/2.5.1/p2rank_2.5.1.tar.gz (about 263 MB)

Requires **Java 17-24** on the server (`module load java/17` or set `JAVA_HOME`). P2Rank does **not** need network access at runtime, but every model directory must contain `model.zst` (not just `features.txt`). A partial copy is the most common cause of `model not found` errors on offline clusters.

Verify before running the pipeline:

```bash
bash scripts/check_p2rank_install.sh /path/to/p2rank_2.5.1
java -version          # must be 17+
/path/to/p2rank_2.5.1/prank -v
```

P2Rank writes pocket CSVs to `workdir/receptors_Pocket/predict_receptors/` when given a `receptors.ds` dataset. Some runs (legacy `.txt` lists or certain P2Rank builds) place `*_predictions.csv` directly under `receptors_Pocket/`; RVSAuto detects both layouts automatically.

For AlphaFold structures always pass `--use_alphafold` (uses `config/alphafold.groovy` and `models/alphafold/model.zst`).

### 6. Optional

```bash
# optional symmetry-aware RMSD: rvsauto redock --symmetry_aware_rmsd
conda install -n unidock_env -c conda-forge rdkit
```

AF2BIND is **not** run by this tool (it needs JAX / ColabDesign / a GPU). Run AF2BIND separately, then pass the CSV folder with `--af2bind_dir`.

---

## Virtual screening — `rvsauto screen`

### P2Rank pockets (default)

```bash
rvsauto screen \
  --pdb_dir receptors \
  --ligand_dir ligands \
  --output_dir out_p2rank \
  --use_alphafold \
  --search_mode detail \
  --adt_env_path adt_env \
  --dock_env_path unidock_env \
  --gpu_ids 0,1
```

`--use_alphafold` passes P2Rank `-c alphafold`, recommended for AlphaFold models.

During long batch runs, `rvsauto screen` and `rvsauto redock` show a **separate terminal progress bar for each pipeline stage** (0–100% within that stage: receptor prep, pockets, pocket configs, docking, etc.). Use `--no_progress` to turn it off. Use `--quiet` to hide console log lines **and** external-tool chatter (ADT/P2Rank/etc.) so only the progress bar updates; details still go to log files under the case / `output_dir/logs/`.

By default, pockets with P2Rank calibrated `probability` below **0.05** are dropped (common screening default). Override with `--p2rank_min_probability` (e.g. `0.25` for stricter filtering); set `0` to keep all ranked pockets. Applied before `--max_pockets`.

### AF2BIND pockets (batch CSV folder)

AF2BIND writes one CSV per protein:

| File | Content |
|------|---------|
| `{stem}_af2bind.csv` | every residue: `chain,resi,resn,p(bind)` |
| `{stem}_af2bind_top15.csv` | official notebook ranking of the top 15 residues |

Point `--af2bind_dir` at that folder (nested directories are scanned). CSVs are matched to `{stem}.pdb` by AlphaFold name (`AF-Q9ZWT3-F1-model_v6`) or UniProt id (`Q9ZWT3`). If both the full table and `_top15` exist, the full table is used.

**Default residue selection (human proteome workflow, Gazizov et al.):**

1. Keep every residue with `p(bind) >= 0.28` (classifier threshold from the AF2BIND proteome paper).
2. Cluster selected residues by spatial proximity (`--af2bind_cluster_cutoff 12`, single-linkage on CA distance).
3. Keep sites with **more than four residues** (`--af2bind_min_residues 5`).

This can recover **multiple pockets** per protein when distant residue clusters all pass the threshold. Sites with five or fewer residues after clustering are dropped.

**Notebook / PyMOL mode** (top 15 only, one site for visualization):

```bash
--af2bind_top_n 15 --af2bind_min_residues 3
```

`--af2bind_pocket_mode cluster` (default) splits spatially separated residues; `single` merges them into one docking box.

```bash
rvsauto screen \
  --pdb_dir receptors \
  --ligand_dir ligands \
  --output_dir out_af2bind \
  --pocket_engine af2bind \
  --af2bind_dir /path/to/AF2BIND_out \
  --search_mode detail \
  --adt_env_path adt_env \
  --dock_env_path unidock_env \
  --make_complex
```

Build boxes only (no ligands, no UniDock):

```bash
rvsauto screen \
  --pdb_dir receptors \
  --output_dir out_af2bind \
  --pocket_engine af2bind \
  --af2bind_dir /path/to/AF2BIND_out \
  --dry_run
```

### Output layout

```
output_dir/
  logs/
  results/
    pockets_summary.tsv
    docking_long.tsv                         # protein, pocket, ligand, affinity
    affinity_matrix_pockets.tsv              # rows = protein+pocket, columns = ligands
    affinity_matrix_proteins.tsv             # rows = protein (best pocket), columns = ligands
    affinity_matrix_proteins_best_pocket.tsv # which pocket was best per ligand
    {ligand}_Docking_Result.tsv
    {ligand}_Best_Docking_Result.tsv
    complexes/                               # only with --make_complex
      {protein}_pocket_{n}_{ligand}_complex.pdb
  workdir/
    receptors_clean/                         # apo PDBs used for docking and complexes
    receptors_PDBQT/
    receptors_Pocket/                        # P2Rank CSVs and .conf files
    docking/{ligand}/                        # UniDock pose PDBQT
    unidock logs
```

`affinity_matrix_pockets.tsv` is the multi-ligand comparison table: one row per protein pocket, one column per ligand, values in kcal/mol (empty cell = docking failed or not run). `affinity_matrix_proteins.tsv` collapses to the best (lowest) affinity per protein.

Complexes are **apo protein + docked pose**. The crystal/input heteroatoms are stripped first; the ligand is written as `HETATM` on chain `Z` (or the first free chain) with serial numbers continuing after the receptor.

---

## Self-redocking — `rvsauto redock`

```bash
rvsauto redock \
  --input_dir complexes \
  --output_dir redock_out \
  --adt_env_path adt_env \
  --dock_env_path unidock_env \
  --search_mode detail \
  --box_padding 4.0 \
  --make_complex \
  --gpu_ids 0
```

Accepts `.pdb`, `.cif`, and gzipped variants. Summary: `results/redock_summary.tsv` (affinity + heavy-atom RMSD, plus percent of poses <= 2 A). Long runs show a terminal progress bar (disable with `--no_progress`; use `--quiet` for progress-only console output).

---

## Cite

If you publish results from these pipelines, cite the tools you actually used:

- **UniDock** -- Yu et al., *J. Chem. Theory Comput.* (2023). https://doi.org/10.1021/acs.jctc.2c01145
- **P2Rank** -- Krivak and Hoksza, *J. Cheminform.* (2018). https://doi.org/10.1186/s13321-018-0285-8
- **AF2BIND** -- Gazizov et al., *bioRxiv* (2023). https://doi.org/10.1101/2023.10.15.562410
