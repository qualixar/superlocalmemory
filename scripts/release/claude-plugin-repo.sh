#!/usr/bin/env bash
# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
#
# claude-plugin-repo.sh — copy the built Claude Code plugin (plugin/) to the root
# of github.com/qualixar/superlocalmemory-plugin, commit, tag vX.Y.Z, push.
#
#   scripts/release/claude-plugin-repo.sh [--dry-run] [--version X.Y.Z]
#
# WHY A SEPARATE REPOSITORY
#   Anthropic's plugin directory checks a plugin more strictly when it sits in
#   a subfolder of a larger repository: every .gitattributes above it counts
#   (this repository's Git LFS and line-ending rules stop validation), and a
#   launch command using any variable but ${CLAUDE_PLUGIN_ROOT} blocks instead
#   of going to a reviewer. The directory tracks this dedicated repository,
#   which holds exactly the plugin folder of each release and nothing else.
#
# AUTH
#   With SLM_DEPLOY_KEY set (CI: the DEPLOY_KEY_SUPERLOCALMEMORY_PLUGIN secret),
#   git talks SSH through that key; otherwise it uses your own GitHub access.
#   SLM_PLUGIN_REPO_URL overrides the target. A dry run clones, applies and
#   shows the change without committing or pushing.

set -euo pipefail
SLM_CHANNEL=claude-plugin-repo
# shellcheck source=scripts/release/_lib.sh
source "$(dirname "$0")/_lib.sh"
slm_parse_args "$@"

SRC="$SLM_ROOT/plugin"
if [ -n "${SLM_DEPLOY_KEY:-}" ]; then
  slm_use_deploy_key
  REPO_URL="${SLM_PLUGIN_REPO_URL:-git@github.com:qualixar/superlocalmemory-plugin.git}"
else
  REPO_URL="${SLM_PLUGIN_REPO_URL:-https://github.com/qualixar/superlocalmemory-plugin.git}"
fi

plugin_version="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["version"])' "$SRC/.claude-plugin/plugin.json")"
[ "$plugin_version" = "$SLM_VERSION" ] || slm_die "plugin/.claude-plugin/plugin.json is $plugin_version, expected $SLM_VERSION"
# A stale generated copy is how an old launcher shipped; never publish one.
node "$SLM_ROOT/scripts/build-all-plugins.mjs" --check >/dev/null \
  || slm_die "generated plugins differ from plugin-src; run node scripts/build-all-plugins.mjs"

work=""
slm_tmpdir work
git clone --quiet "$REPO_URL" "$work/repo"
# Replace everything but .git with the plugin folder, byte for byte.
find "$work/repo" -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +
cp -R "$SRC/." "$work/repo/"
# The directory blocks a computed launch command ("${ComSpec:-...}", which picks
# cmd.exe on Windows). Its copy runs the POSIX launcher by literal path; the
# GitHub marketplace plugin keeps the Windows-capable form.
python3 - "$work/repo/.mcp.json" <<'PY' || slm_die "unexpected plugin/.mcp.json launch command; refusing to publish"
import json, sys
path = sys.argv[1]
with open(path) as handle:
    data = json.load(handle)
server = data["mcpServers"]["superlocalmemory"]
if not server["command"].startswith("${ComSpec:-"):
    sys.exit(1)
server["command"] = "${CLAUDE_PLUGIN_ROOT}/scripts/slm-launch"
server["args"] = []
with open(path, "w") as handle:
    json.dump(data, handle, indent=2)
    handle.write("\n")
PY

cd "$work/repo"
git checkout --quiet -B main  # an empty repository has no branch yet
git add -A
if git diff --cached --quiet; then
  slm_status skipped "superlocalmemory-plugin already matches $SLM_VERSION"
  exit 0
fi
if [ "$SLM_DRY_RUN" -eq 1 ]; then
  git diff --cached --stat | tail -n 15
  slm_status ok "dry run: would commit, tag v$SLM_VERSION and push $(git diff --cached --name-only | wc -l | tr -d ' ') changed file(s)"
  exit 0
fi

slm_git_identity
git commit --quiet -m "SuperLocalMemory plugin $SLM_VERSION

Built from github.com/qualixar/superlocalmemory (plugin/) at $(git -C "$SLM_ROOT" rev-parse --short HEAD).
The MCP server starts through scripts/slm-launch by literal path."
git tag -f -a "v$SLM_VERSION" -m "SuperLocalMemory plugin $SLM_VERSION"
git push --quiet origin main
git push --quiet --force origin "v$SLM_VERSION"
slm_status ok "published superlocalmemory-plugin $SLM_VERSION ($(git rev-parse --short HEAD))"
