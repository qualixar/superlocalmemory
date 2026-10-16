#!/usr/bin/env bash
# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
#
# openvsx.sh — publish the VS Code extension to Open VSX (qualixar.superlocalmemory).
#
#   scripts/release/openvsx.sh [--dry-run] [--version X.Y.Z] [--dist dist]
#
# Publishes dist/vscode/superlocalmemory-vscode-X.Y.Z.vsix (built by
# build-vsix.sh) with the pinned ovsx CLI. ovsx reads the token from OVSX_PAT,
# so it never appears on a command line. A version Open VSX already has is a no-op.

set -euo pipefail
SLM_CHANNEL=openvsx
# shellcheck source=scripts/release/_lib.sh
source "$(dirname "$0")/_lib.sh"
slm_parse_args "$@"

OVSX_VERSION=1.2.0
API=https://open-vsx.org/api/qualixar/superlocalmemory
VSIX="$SLM_DIST/vscode/superlocalmemory-vscode-$SLM_VERSION.vsix"

code="$(slm_http_status "$API/$SLM_VERSION")"
case "$code" in
  200) slm_status skipped "qualixar.superlocalmemory $SLM_VERSION is already on Open VSX"; exit 0 ;;
  404) slm_log "$SLM_VERSION is not on Open VSX yet" ;;
  *) slm_die "Open VSX lookup returned HTTP $code" ;;
esac

if [ ! -f "$VSIX" ]; then
  if [ "$SLM_DRY_RUN" -eq 1 ]; then
    slm_status ok "dry run: would publish $VSIX (not built yet; build-vsix.sh makes it)"
    exit 0
  fi
  slm_die "$VSIX is missing (the vsix job did not produce it)"
fi
packaged="$(unzip -p "$VSIX" extension/package.json | python3 -c 'import json,sys; print(json.load(sys.stdin)["version"])')"
[ "$packaged" = "$SLM_VERSION" ] || slm_die "$(basename "$VSIX") contains version $packaged"

if [ "$SLM_DRY_RUN" -eq 1 ]; then
  slm_status ok "dry run: would publish $(basename "$VSIX") with ovsx@$OVSX_VERSION"
  exit 0
fi
slm_require_env OVSX_PAT
npx --yes "ovsx@$OVSX_VERSION" publish "$VSIX" --skip-duplicate
slm_status ok "published qualixar.superlocalmemory $SLM_VERSION"
