#!/usr/bin/env bash
# Download the public data the comparison needs into $SLM_BENCH_HOME.
#   LoCoMo (snap-research/locomo, CC BY-NC 4.0): pinned commit + sha256 check.
#   arXiv PDFs: optional; skipped with a message when arxiv.org is unreachable.
# Nothing downloaded here is ever committed to this repository.
set -euo pipefail

HOME_DIR="${SLM_BENCH_HOME:-$HOME/.cache/slm-embedder-compare}"
LOCOMO_REPO="https://github.com/snap-research/locomo.git"
LOCOMO_COMMIT="3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376"
LOCOMO_SHA256="79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4"
ARXIV_IDS=("2603.14588")

mkdir -p "$HOME_DIR"
dest="$HOME_DIR/locomo"
if [ ! -d "$dest/.git" ]; then
  git clone --quiet "$LOCOMO_REPO" "$dest"
fi
git -C "$dest" fetch --quiet origin "$LOCOMO_COMMIT" 2>/dev/null || true
git -C "$dest" checkout --quiet "$LOCOMO_COMMIT"
actual="$(sha256sum "$dest/data/locomo10.json" | cut -d' ' -f1)"
if [ "$actual" != "$LOCOMO_SHA256" ]; then
  echo "LoCoMo checksum mismatch: expected $LOCOMO_SHA256, got $actual" >&2
  exit 1
fi
echo "LoCoMo ok at $LOCOMO_COMMIT (licence: CC BY-NC 4.0, see $dest/LICENSE.txt)"

mkdir -p "$HOME_DIR/arxiv"
for id in "${ARXIV_IDS[@]}"; do
  out="$HOME_DIR/arxiv/$id.pdf"
  [ -s "$out" ] && continue
  if curl -fsSL --max-time 30 -o "$out" "https://arxiv.org/pdf/$id"; then
    echo "fetched arXiv $id"
  else
    rm -f "$out"
    echo "skipped arXiv $id: arxiv.org unreachable (generated PDFs are used instead)" >&2
  fi
done
