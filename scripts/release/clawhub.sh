#!/usr/bin/env bash
# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
#
# clawhub.sh — republish the OpenClaw skill (clawhub-skill/superlocalmemory,
# MIT-0, slug "superlocalmemory") when, and only when, its files changed.
#
#   scripts/release/clawhub.sh [--dry-run] [--version X.Y.Z]
#
# The skill is versioned on its own (1.0.0, ...), not with SLM. Each local
# file's sha256 is compared with the latest published version's file list
# (`clawhub inspect --json`); files only ClawHub has (its generated
# skill-card.md) are ignored. Unchanged = no-op. On a change, the skill's own
# frontmatter version is used when it is newer than the published one,
# otherwise ClawHub picks the next patch. CLAWHUB_TOKEN is stored in a private
# temporary config for this run only.

set -euo pipefail
SLM_CHANNEL=clawhub
# shellcheck source=scripts/release/_lib.sh
source "$(dirname "$0")/_lib.sh"
slm_parse_args "$@"

CLAWHUB_CLI_VERSION=0.23.3
SLUG=superlocalmemory
SKILL_DIR="$SLM_ROOT/clawhub-skill/superlocalmemory"
[ -f "$SKILL_DIR/SKILL.md" ] || slm_die "$SKILL_DIR/SKILL.md is missing"

work=""
slm_tmpdir work
export CLAWHUB_CONFIG_PATH="$work/config.json"
export CLAWHUB_DISABLE_TELEMETRY=1
clawhub() { npx --yes "clawhub@$CLAWHUB_CLI_VERSION" --no-input "$@"; }

clawhub inspect "$SLUG" --files --json >"$work/published.json" \
  || slm_die "could not read the published skill from ClawHub"
changed="$(slm_tools skill-changed "$SKILL_DIR" "$work/published.json")"
published_version="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["latestVersion"]["version"])' "$work/published.json")"
if [ "$changed" = "unchanged" ]; then
  slm_status skipped "skill files match ClawHub $SLUG@$published_version"
  exit 0
fi

local_version="$(slm_tools yaml-scalar "$SKILL_DIR/SKILL.md" version || true)"
version_args=()
if [ -n "$local_version" ] && [ "$(slm_tools compare "$local_version" "$published_version")" -gt 0 ]; then
  version_args=(--version "$local_version")
fi
if [ "$SLM_DRY_RUN" -eq 1 ]; then
  slm_status ok "dry run: skill changed since $SLUG@$published_version; would publish ${version_args[1]:-the next patch after $published_version}"
  exit 0
fi

slm_require_env CLAWHUB_TOKEN
clawhub login --token "$CLAWHUB_TOKEN" --no-browser >/dev/null
clawhub skill publish "$SKILL_DIR" --slug "$SLUG" --name SuperLocalMemory \
  "${version_args[@]+"${version_args[@]}"}" \
  --changelog "Updated with SuperLocalMemory $SLM_VERSION" \
  --source-repo qualixar/superlocalmemory --source-path clawhub-skill/superlocalmemory \
  --source-commit "$(git -C "$SLM_ROOT" rev-parse HEAD)"
slm_status ok "published $SLUG (was $published_version)"
