#!/usr/bin/env bash
# Quick offline sanity check for a P2Rank 2.5.x installation.
# Usage: bash scripts/check_p2rank_install.sh /path/to/p2rank_2.5.1

set -euo pipefail

ROOT="${1:-./p2rank_2.5.1}"
PRANK="${ROOT}/prank"

echo "=== P2Rank install check: ${ROOT} ==="

if [[ ! -x "${PRANK}" && ! -f "${PRANK}" ]]; then
  echo "FAIL: prank launcher not found at ${PRANK}"
  exit 1
fi
echo "OK   prank launcher present"

if [[ ! -f "${ROOT}/bin/p2rank.jar" ]]; then
  echo "FAIL: missing ${ROOT}/bin/p2rank.jar"
  exit 1
fi
echo "OK   p2rank.jar present"

if command -v java >/dev/null 2>&1; then
  JAVA_VER="$(java -version 2>&1 | head -n1)"
  echo "OK   java on PATH: ${JAVA_VER}"
else
  echo "FAIL: java not found (P2Rank needs Java 17-24)"
  exit 1
fi

MODELS=(
  "models/alphafold/model.zst"
  "models/default/model.zst"
)
for rel in "${MODELS[@]}"; do
  if [[ ! -f "${ROOT}/${rel}" ]]; then
    echo "FAIL: missing ${ROOT}/${rel}"
    echo "      Re-download the full release tarball:"
    echo "      https://github.com/rdk/p2rank/releases/download/2.5.1/p2rank_2.5.1.tar.gz"
    exit 1
  fi
  echo "OK   ${rel} ($(du -h "${ROOT}/${rel}" | cut -f1))"
done

echo
echo "Smoke test (optional, needs a PDB in test_data):"
if [[ -f "${ROOT}/test_data/1fbl.pdb" ]]; then
  OUT="${ROOT}/_smoke_out"
  rm -rf "${OUT}"
  bash "${PRANK}" predict -c alphafold -f "${ROOT}/test_data/1fbl.pdb" -o "${OUT}" -visualizations 0
  CSV="$(find "${OUT}" -name '*1fbl.pdb_predictions.csv' | head -n1 || true)"
  if [[ -z "${CSV}" ]]; then
    echo "FAIL: smoke prediction produced no CSV under ${OUT}"
    exit 1
  fi
  echo "OK   smoke prediction -> ${CSV}"
  rm -rf "${OUT}"
else
  echo "SKIP no ${ROOT}/test_data/1fbl.pdb"
fi

echo
echo "P2Rank installation looks complete for offline use."
