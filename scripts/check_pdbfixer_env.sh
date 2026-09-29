#!/usr/bin/env bash
# Verify pdbfixer + openmm in a conda env before long receptor batches.
# Usage: bash scripts/check_pdbfixer_env.sh /path/to/miniforge3/envs/pdbfixer_env

set -euo pipefail

ENV="${1:?conda env path required}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPAIR="${SCRIPT_DIR}/pdbfixer_repair.py"

echo "=== PDBFixer env check: ${ENV} ==="

if [[ ! -x "${ENV}/bin/python" ]]; then
  echo "FAIL: ${ENV}/bin/python not found"
  exit 1
fi

"${ENV}/bin/python" "${REPAIR}" --self-test

echo
echo "PDBFixer environment looks usable."
