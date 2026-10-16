#!/usr/bin/env bash
# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
#
# dsh-bundle.sh — keep github.com/qualixar/dsh-superlocalmemory (the DeepSeek
# Harness bundle) versioned with SuperLocalMemory.
#
#   scripts/release/dsh-bundle.sh [--dry-run] [--version X.Y.Z]
#
# The bundle is configuration only (package.json with dsh.bundle and
# cordis.patch.yml, which runs `slm mcp`); it carries a package.json version,
# which this sets to the release version so the bundle names the SLM release
# it was published with. When the version already matches there is no commit
# and no push. With SLM_DEPLOY_KEY set (CI: DEPLOY_KEY_DSH_SUPERLOCALMEMORY) git
# uses SSH through that key; otherwise your own GitHub access.
# SLM_DSH_REPO_URL overrides the target.

set -euo pipefail
SLM_CHANNEL=dsh-bundle
# shellcheck source=scripts/release/_lib.sh
source "$(dirname "$0")/_lib.sh"
slm_parse_args "$@"

if [ -n "${SLM_DEPLOY_KEY:-}" ]; then
  slm_use_deploy_key
  REPO_URL="${SLM_DSH_REPO_URL:-git@github.com:qualixar/dsh-superlocalmemory.git}"
else
  REPO_URL="${SLM_DSH_REPO_URL:-https://github.com/qualixar/dsh-superlocalmemory.git}"
fi

work=""
slm_tmpdir work
git clone --quiet --depth 1 "$REPO_URL" "$work/repo"
cd "$work/repo"
[ -f package.json ] || slm_die "the bundle repository has no package.json"
python3 -c 'import json,sys; d=json.load(open("package.json")); sys.exit(0 if "bundle" in d.get("dsh", {}) else 1)' \
  || slm_die "package.json has no dsh.bundle; is this still the DeepSeek Harness bundle?"

if ! python3 -c 'import json,sys; sys.exit(0 if "version" in json.load(open("package.json")) else 1)'; then
  slm_status skipped "the bundle carries no version (configuration only); nothing to update"
  exit 0
fi

check_flag=()
[ "$SLM_DRY_RUN" -eq 1 ] && check_flag=(--check)
result="$(slm_tools set-json-version package.json "$SLM_VERSION" "${check_flag[@]+"${check_flag[@]}"}")" \
  || slm_die "could not set package.json version"
if [ "$result" = "unchanged" ]; then
  slm_status skipped "dsh-superlocalmemory is already at $SLM_VERSION"
  exit 0
fi
if [ "$SLM_DRY_RUN" -eq 1 ]; then
  current="$(python3 -c 'import json; print(json.load(open("package.json"))["version"])')"
  slm_status ok "dry run: would bump dsh-superlocalmemory $current -> $SLM_VERSION and push"
  exit 0
fi

slm_git_identity
git add package.json
git commit --quiet -m "chore: SuperLocalMemory $SLM_VERSION"
git push --quiet origin HEAD
slm_status ok "dsh-superlocalmemory now at $SLM_VERSION ($(git rev-parse --short HEAD))"
