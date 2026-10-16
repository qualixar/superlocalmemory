#!/usr/bin/env bash
# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
#
# listing-checklist.sh — open the issue "Release vX.Y.Z — manual listing
# checklist" naming every channel that needs a person (review queues, forms,
# PRs to other people's repositories). One issue per release: a re-run finds
# the existing issue and does nothing. The list itself lives in
# scripts/release/release_tools.py (MANUAL_LISTINGS). Needs GH_TOKEN with
# issues: write.
#
#   scripts/release/listing-checklist.sh [--dry-run] [--version X.Y.Z]

set -euo pipefail
SLM_CHANNEL=listing-checklist
# shellcheck source=scripts/release/_lib.sh
source "$(dirname "$0")/_lib.sh"
slm_parse_args "$@"

TITLE="Release v$SLM_VERSION — manual listing checklist"
command -v gh >/dev/null || slm_die "the gh CLI is required"

work=""
slm_tmpdir work
slm_tools checklist-body "$SLM_VERSION" >"$work/body.md"

existing="$(gh issue list --state all --search "\"$TITLE\" in:title" --json number,title \
  --jq "map(select(.title == \"$TITLE\")) | .[0].number // empty")" \
  || slm_die "could not search issues"
if [ -n "$existing" ]; then
  slm_status skipped "issue #$existing already tracks the manual listings"
  exit 0
fi
if [ "$SLM_DRY_RUN" -eq 1 ]; then
  cat "$work/body.md"
  slm_status ok "dry run: would open \"$TITLE\""
  exit 0
fi
url="$(gh issue create --title "$TITLE" --body-file "$work/body.md")"
slm_status ok "opened $url"
