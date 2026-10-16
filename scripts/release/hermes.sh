#!/usr/bin/env bash
# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
#
# hermes.sh — confirm Hermes users get this release's plugin.
#
#   scripts/release/hermes.sh [--dry-run] [--version X.Y.Z] [--ref origin/main]
#
# Hermes installs 'https://github.com/qualixar/superlocalmemory.git#hermes-plugin';
# the "#hermes-plugin" fragment is a SUBDIRECTORY of the default branch
# (hermes_cli/plugins_cmd.py, "Explicit #subdir fragment"), not a branch. So
# there is nothing to push: the channel is current once the default branch's
# hermes-plugin/plugin.yaml carries this version (or a newer one). This script
# checks the tagged tree and the default branch, and fails when main is behind,
# which means the release commit was not merged to main.

set -euo pipefail
SLM_CHANNEL=hermes
SLM_ACCEPTS="--ref"
# shellcheck source=scripts/release/_lib.sh
source "$(dirname "$0")/_lib.sh"
slm_parse_args "$@"

REF="$(slm_extra --ref)"
REF="${REF:-origin/main}"
MANIFEST="hermes-plugin/plugin.yaml"

tagged="$(slm_tools yaml-scalar "$SLM_ROOT/$MANIFEST" version)" || slm_die "$MANIFEST has no version"
[ "$tagged" = "$SLM_VERSION" ] || slm_die "$MANIFEST in this checkout is $tagged, expected $SLM_VERSION"
slm_log "this checkout's hermes-plugin/ is $tagged"

branch="${REF#origin/}"
if ! git -C "$SLM_ROOT" fetch --quiet origin "$branch" 2>/dev/null; then
  if [ "$SLM_DRY_RUN" -eq 1 ]; then
    slm_warn "could not fetch $REF; checked the local tree only"
    slm_status skipped "dry run: hermes-plugin/ at $tagged locally; $REF not fetched"
    exit 0
  fi
  slm_die "could not fetch $REF to check what Hermes users install"
fi

served="$(git -C "$SLM_ROOT" show "FETCH_HEAD:$MANIFEST" | python3 -c '
import re, sys
m = re.search(r"^version:\s*([^\s#]+)", sys.stdin.read(), re.MULTILINE)
print(m.group(1) if m else "")')"
[ -n "$served" ] || slm_die "$REF has no $MANIFEST version"

if [ "$(slm_tools compare "$served" "$SLM_VERSION")" -lt 0 ]; then
  slm_die "Hermes users install $REF:hermes-plugin/ at $served, behind $SLM_VERSION; merge the release to main"
fi
slm_status ok "Hermes installs hermes-plugin/ from $REF at $served (served from the default branch, nothing to push)"
