#!/usr/bin/env bash
# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
#
# build-vsix.sh — build and test the VS Code extension, then package
# dist/vscode/superlocalmemory-vscode-X.Y.Z.vsix with a pinned vsce.
#
#   scripts/release/build-vsix.sh [--dry-run] [--version X.Y.Z] [--dist dist]
#
# One build per release feeds both the GitHub release and Open VSX, so the two
# carry the same bytes. Needs Node >= 22 (vsce 4 requires it).

set -euo pipefail
SLM_CHANNEL=build-vsix
# shellcheck source=scripts/release/_lib.sh
source "$(dirname "$0")/_lib.sh"
slm_parse_args "$@"

VSCE_VERSION=4.0.0
EXT="$SLM_ROOT/vscode-extension"
OUT="$SLM_DIST/vscode/superlocalmemory-vscode-$SLM_VERSION.vsix"

manifest="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["version"])' "$EXT/package.json")"
[ "$manifest" = "$SLM_VERSION" ] || slm_die "vscode-extension/package.json is $manifest, expected $SLM_VERSION"

if [ "$SLM_DRY_RUN" -eq 1 ]; then
  slm_status ok "dry run: would build, test and package $OUT with @vscode/vsce@$VSCE_VERSION"
  exit 0
fi

mkdir -p "$(dirname "$OUT")"
(cd "$EXT" && npm ci --no-audit --no-fund && npm test)
(cd "$EXT" && npx --yes "@vscode/vsce@$VSCE_VERSION" package --out "$OUT")
packaged="$(unzip -p "$OUT" extension/package.json | python3 -c 'import json,sys; print(json.load(sys.stdin)["version"])')"
[ "$packaged" = "$SLM_VERSION" ] || slm_die "the packaged extension says $packaged"
slm_status ok "built $(basename "$OUT")"
