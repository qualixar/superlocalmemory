#!/usr/bin/env bash
# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
#
# listing-checklist.sh — count the channels that need a person (review queues,
# forms, PRs to other people's repositories) and say so in the run log. It no
# longer opens a public issue: the to-do list is the maintainer's, not the
# project's front page (4.2.8 opened #288 and listed pending approvals in
# public). The list lives in scripts/release/release_tools.py (MANUAL_LISTINGS);
# `release_tools.py checklist-body X.Y.Z` prints it for the maintainer's notes.
#
#   scripts/release/listing-checklist.sh [--dry-run] [--version X.Y.Z]

set -euo pipefail
SLM_CHANNEL=listing-checklist
# shellcheck source=scripts/release/_lib.sh
source "$(dirname "$0")/_lib.sh"
slm_parse_args "$@"

command -v python3 >/dev/null || slm_die "python3 is required"

work=""
slm_tmpdir work
slm_tools checklist-body "$SLM_VERSION" >"$work/body.md"
count="$(grep -c '^- \[ \]' "$work/body.md" || true)"
if [ "$SLM_DRY_RUN" -eq 1 ]; then
  cat "$work/body.md"
fi
slm_status ok "$count manual listing(s) for v$SLM_VERSION are tracked by the maintainer; nothing published"
