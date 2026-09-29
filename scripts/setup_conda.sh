#!/usr/bin/env bash
# Create all recommended conda environments for RVSAuto (Linux / HPC).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

create_or_update() {
  local file="$1"
  local name
  name="$(grep '^name:' "${file}" | awk '{print $2}')"
  echo "[setup] env: ${name} <= ${file}"
  if conda env list | awk '{print $1}' | grep -qx "${name}"; then
    conda env update -f "${file}" --name "${name}" --prune
  else
    conda env create -f "${file}"
  fi
}

create_or_update conda/environment-adt.yml
create_or_update conda/environment-unidock.yml
create_or_update conda/environment-pdbfixer.yml
create_or_update conda/environment-dev.yml

echo
echo "[setup] Done. Recommended usage:"
echo "  conda activate rvsauto-dev     # Python CLI + tests"
echo "  conda activate adt_env           # prepare_receptor4 / prepare_ligand4 / obabel"
echo "  conda activate unidock_env       # GPU docking"
echo "  conda activate pdbfixer_env      # optional AlphaFold repair"
echo
echo "  rvsauto screen --help"
echo "  rvsauto redock --help"
echo "  # aliases: rvsauto-unidock, rvsauto-redock"
