#!/usr/bin/env bash
# First-time public release for https://github.com/Xin-DongXu/RVSAuto
# Prerequisite: create an EMPTY repo on GitHub (no README/license) named RVSAuto.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

REMOTE="https://github.com/Xin-DongXu/RVSAuto.git"
TAG="v11.0.0"

if ! command -v git >/dev/null 2>&1; then
  echo "git not found on PATH" >&2
  exit 1
fi

if [[ ! -d .git ]]; then
  git init -b main
fi

git add rvsauto tests scripts conda .github README.md LICENSE CHANGELOG.md \
  pyproject.toml MANIFEST.in requirements.txt docs .gitignore

if git diff --cached --quiet; then
  echo "Nothing to commit (already committed?)"
else
  git commit -m "Initial public release of RVSAuto ${TAG#v}"
fi

if ! git remote get-url origin >/dev/null 2>&1; then
  git remote add origin "${REMOTE}"
else
  current="$(git remote get-url origin)"
  if [[ "${current}" != "${REMOTE}" ]]; then
    echo "origin is ${current}; expected ${REMOTE}" >&2
    exit 1
  fi
fi

echo "Pushing main to ${REMOTE} ..."
git push -u origin main

if git rev-parse "${TAG}" >/dev/null 2>&1; then
  echo "Tag ${TAG} exists locally"
else
  git tag -a "${TAG}" -m "RVSAuto ${TAG#v} — first public release"
fi

git push origin "${TAG}"

echo "Done. Open: https://github.com/Xin-DongXu/RVSAuto/releases/new?tag=${TAG}"
