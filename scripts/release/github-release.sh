#!/usr/bin/env bash
# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
#
# github-release.sh — the GitHub release for tag vX.Y.Z, notes from CHANGELOG.md.
#
#   scripts/release/github-release.sh [--dry-run] [--version X.Y.Z] [--dist dist]
#
# Assets: dist/python/*, dist/npm/*.tgz, dist/release-evidence.json,
# dist/SHA256SUMS, dist/*.cdx.json and dist/vscode/*.vsix (whichever exist).
# A new release gets the CHANGELOG section as its notes. An existing release
# keeps its notes; each asset already on it must be byte-identical (the wheel,
# sdist and tarball are what PyPI and npm serve), except the .vsix, which is
# rebuilt per run and is left alone once attached. Missing assets are uploaded.
# Needs GH_TOKEN (the workflow's GITHUB_TOKEN) with contents: write.

set -euo pipefail
SLM_CHANNEL=github-release
# shellcheck source=scripts/release/_lib.sh
source "$(dirname "$0")/_lib.sh"
slm_parse_args "$@"

TAG="v$SLM_VERSION"
command -v gh >/dev/null || slm_die "the gh CLI is required"

work=""
slm_tmpdir work
slm_tools changelog-section "$SLM_VERSION" --changelog "$SLM_ROOT/CHANGELOG.md" >"$work/notes.md" \
  || slm_die "CHANGELOG.md has no usable section for $SLM_VERSION"
slm_log "notes: $(wc -l <"$work/notes.md" | tr -d ' ') lines from CHANGELOG.md [$SLM_VERSION]"

assets=()
for candidate in "$SLM_DIST"/python/* "$SLM_DIST"/npm/*.tgz "$SLM_DIST"/release-evidence.json \
  "$SLM_DIST"/SHA256SUMS "$SLM_DIST"/*.cdx.json "$SLM_DIST"/vscode/*.vsix; do
  [ -f "$candidate" ] && assets+=("$candidate")
done
slm_log "${#assets[@]} asset(s) found under $SLM_DIST"
if [ "${#assets[@]}" -eq 0 ] && [ "$SLM_DRY_RUN" -eq 0 ]; then
  slm_die "no release assets under $SLM_DIST"
fi

if ! gh release view "$TAG" >/dev/null 2>&1; then
  if [ "$SLM_DRY_RUN" -eq 1 ]; then
    slm_status ok "dry run: would create $TAG with ${#assets[@]} asset(s) and CHANGELOG notes"
    exit 0
  fi
  gh release create "$TAG" "${assets[@]}" --verify-tag --title "SuperLocalMemory $TAG" --notes-file "$work/notes.md"
  slm_status ok "created $TAG with ${#assets[@]} asset(s)"
  exit 0
fi

existing="$work/existing"
mkdir -p "$existing"
gh release download "$TAG" --dir "$existing" >/dev/null 2>&1 || true
upload=()
for asset in "${assets[@]+"${assets[@]}"}"; do
  published="$existing/$(basename "$asset")"
  if [ ! -f "$published" ]; then
    upload+=("$asset")
  elif [[ "$asset" != *.vsix ]] && ! cmp -s "$asset" "$published"; then
    slm_die "$(basename "$asset") on $TAG differs from this run's build; refusing to replace a published asset"
  fi
done

if [ "${#upload[@]}" -eq 0 ]; then
  slm_status skipped "$TAG already exists with every asset"
  exit 0
fi
if [ "$SLM_DRY_RUN" -eq 1 ]; then
  slm_status ok "dry run: $TAG exists; would upload ${#upload[@]} missing asset(s)"
  exit 0
fi
gh release upload "$TAG" "${upload[@]}"
slm_status ok "$TAG existed; uploaded ${#upload[@]} missing asset(s)"
