#!/usr/bin/env bash
# Download the public data the comparison needs into $SLM_BENCH_HOME.
#   LoCoMo (snap-research/locomo, CC BY-NC 4.0): pinned commit + sha256 check.
#   arXiv PDFs (the author's public papers): used by the build when all are present;
#   skipped with a message when arxiv.org is unreachable.
# Nothing downloaded here is ever committed to this repository.
set -euo pipefail

HOME_DIR="${SLM_BENCH_HOME:-$HOME/.cache/slm-embedder-compare}"
LOCOMO_REPO="https://github.com/snap-research/locomo.git"
LOCOMO_COMMIT="3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376"
LOCOMO_SHA256="79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4"
ARXIV_IDS=("2603.14588" "2603.02601" "2604.04514")

mkdir -p "$HOME_DIR"
dest="$HOME_DIR/locomo"
if [ ! -d "$dest/.git" ]; then
  git clone --quiet "$LOCOMO_REPO" "$dest"
fi
git -C "$dest" fetch --quiet origin "$LOCOMO_COMMIT" 2>/dev/null || true
git -C "$dest" checkout --quiet "$LOCOMO_COMMIT"
if command -v sha256sum >/dev/null 2>&1; then
  actual="$(sha256sum "$dest/data/locomo10.json" | cut -d' ' -f1)"
else
  actual="$(shasum -a 256 "$dest/data/locomo10.json" | cut -d' ' -f1)"
fi
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
    echo "skipped arXiv $id: arxiv.org unreachable (the build then leaves out the arXiv pages and queries)" >&2
  fi
done

# MS-COCO 2014 Karpathy test split (Hugging Face mirror, pinned revision): real photos with
# five human captions each. Captions CC BY 4.0; images under their Flickr terms. Only ids
# (golden/coco_selection.json) are committed; files stay in $SLM_BENCH_HOME/coco.
COCO_REPO="https://huggingface.co/datasets/nlphuji/mscoco_2014_5k_test_image_text_retrieval/resolve"
COCO_REV="551c4f7667f06fa82b4ef0a07617bfc4cf324ac3"
COCO_ZIP_SHA256="92e67aad1d818ba95103299f0c93426436c3cc37728b14d9b215d5b56722cbed"
COCO_CSV_SHA256="8c97309e06d7554174343d084f15feacc4d73e8c04cd920fed1f3edf0f328cec"
mkdir -p "$HOME_DIR/coco"
fetch_coco() {  # $1 remote name, $2 local name, $3 sha256 prefix
  out="$HOME_DIR/coco/$2"
  [ -s "$out" ] || curl -fsSL --max-time 900 -o "$out" "$COCO_REPO/$COCO_REV/$1" || { rm -f "$out"; echo "skipped COCO $1" >&2; return 0; }
  sum="$( (sha256sum "$out" 2>/dev/null || shasum -a 256 "$out") | cut -d' ' -f1)"
  case "$sum" in "$3"*) echo "COCO $2 ok ($sum)";; *) echo "COCO $2 checksum mismatch: $sum" >&2; rm -f "$out"; exit 1;; esac
}
fetch_coco test_5k_mscoco_2014.csv test_5k.csv "$COCO_CSV_SHA256"
fetch_coco images_mscoco_2014_5k_test.zip images.zip "$COCO_ZIP_SHA256"
