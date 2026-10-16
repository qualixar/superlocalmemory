# shellcheck shell=bash
# shellcheck disable=SC2034  # SLM_DIST and friends are read by the scripts that source this
# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
#
# _lib.sh — shared helpers for scripts/release/<channel>.sh. Source it; do not run it.
#
# Every channel script accepts:
#   --dry-run          read-only: check state and print what would happen
#   --version X.Y.Z    release version (default: package.json)
#   --dist DIR         release artifacts (default: dist)
# and reports its result through slm_status (ok | skipped | failed), which the
# release workflow turns into the run's channel table.
#
# Secrets arrive only through the environment and are never printed. Shell
# tracing is never turned on here or in any channel script.

SLM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SLM_TOOLS="$SLM_ROOT/scripts/release/release_tools.py"
SLM_DRY_RUN=0
SLM_VERSION=""
SLM_DIST="$SLM_ROOT/dist"
SLM_EXTRA_ARGS=()
SLM_CLEANUP_PATHS=()

slm_log() { printf '%s\n' "[$SLM_CHANNEL] $*"; }
slm_warn() { printf '%s\n' "::warning::[$SLM_CHANNEL] $*" >&2; }
slm_die() {
  printf '%s\n' "::error::[$SLM_CHANNEL] $*" >&2
  slm_status failed "$*"
  exit 1
}

# slm_status ok|skipped|failed "detail" — last call wins.
slm_status() {
  if [ -n "${SLM_CHANNEL_STATUS_FILE:-}" ]; then
    printf 'status=%s\ndetail=%s\n' "$1" "${2:-}" >"$SLM_CHANNEL_STATUS_FILE"
  fi
  slm_log "result: $1${2:+ ($2)}"
}

slm_cleanup() {
  local path
  for path in "${SLM_CLEANUP_PATHS[@]+"${SLM_CLEANUP_PATHS[@]}"}"; do
    [ -n "$path" ] && rm -rf -- "$path"
  done
}
trap slm_cleanup EXIT

# slm_tmpdir VAR — make a private temp dir, store its path in VAR, remove it on exit.
# (Not "$(slm_tmpdir)": a command substitution runs in a subshell, so the
# cleanup registration would be lost.)
slm_tmpdir() {
  local _slm_new_tmp
  _slm_new_tmp="$(mktemp -d "${TMPDIR:-/tmp}/slm-release.XXXXXX")"
  SLM_CLEANUP_PATHS+=("$_slm_new_tmp")
  printf -v "$1" '%s' "$_slm_new_tmp"
}

# slm_parse_args "$@" — sets SLM_DRY_RUN, SLM_VERSION, SLM_DIST; channel-specific
# "--name value" pairs listed in SLM_ACCEPTS (space separated) land in SLM_EXTRA_ARGS.
slm_parse_args() {
  while [ "$#" -gt 0 ]; do
    case "$1" in
      --dry-run) SLM_DRY_RUN=1 ;;
      --version) SLM_VERSION="${2:?--version needs a value}"; shift ;;
      --dist) SLM_DIST="${2:?--dist needs a value}"; shift ;;
      -h|--help) sed -n '2,/^$/p' "$0"; exit 0 ;;
      *)
        if [ -n "${SLM_ACCEPTS:-}" ] && [[ " $SLM_ACCEPTS " == *" $1 "* ]]; then
          SLM_EXTRA_ARGS+=("$1" "${2:?$1 needs a value}"); shift
        else
          printf '%s\n' "[$SLM_CHANNEL] unknown option: $1" >&2
          exit 2
        fi
        ;;
    esac
    shift
  done
  if [ -z "$SLM_VERSION" ]; then
    SLM_VERSION="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["version"])' "$SLM_ROOT/package.json")"
  fi
  if ! [[ "$SLM_VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+([-.][0-9A-Za-z.-]+)?$ ]]; then
    slm_die "not a release version: $SLM_VERSION"
  fi
  if [ "$SLM_DRY_RUN" -eq 1 ]; then slm_log "dry run for $SLM_VERSION (nothing is published)"; fi
}

# slm_extra NAME — value of a channel-specific option, or empty.
slm_extra() {
  local i
  for ((i = 0; i < ${#SLM_EXTRA_ARGS[@]}; i += 2)); do
    if [ "${SLM_EXTRA_ARGS[$i]}" = "$1" ]; then printf '%s\n' "${SLM_EXTRA_ARGS[$((i + 1))]}"; return 0; fi
  done
}

slm_tools() { python3 "$SLM_TOOLS" "$@"; }

# slm_require_env NAME — fail unless the variable is set; never prints its value.
slm_require_env() {
  if [ -z "${!1:-}" ]; then slm_die "$1 is not set"; fi
}

# slm_http_status URL — HTTP status code only (000 on network failure).
slm_http_status() {
  local code
  code="$(curl -sS -o /dev/null -w '%{http_code}' --retry 3 --retry-delay 2 "$1" 2>/dev/null)" || true
  printf '%s' "${code:-000}"
}

# slm_use_deploy_key — route git's ssh through the deploy key in $SLM_DEPLOY_KEY,
# pinned to GitHub's published host keys (fetched over TLS from api.github.com).
slm_use_deploy_key() {
  slm_require_env SLM_DEPLOY_KEY
  local dir
  slm_tmpdir dir
  (umask 077 && printf '%s\n' "$SLM_DEPLOY_KEY" >"$dir/key")
  curl -fsSL --retry 3 https://api.github.com/meta \
    | python3 -c 'import json,sys; [print("github.com", k) for k in json.load(sys.stdin)["ssh_keys"]]' \
      >"$dir/known_hosts"
  test -s "$dir/known_hosts" || slm_die "could not load GitHub's SSH host keys"
  export GIT_SSH_COMMAND="ssh -i $dir/key -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes -o UserKnownHostsFile=$dir/known_hosts"
}

# slm_git_identity — inside GitHub Actions, commits carry the Actions bot identity
# unless the workflow set one; locally the user's own git config applies.
slm_git_identity() {
  [ "${GITHUB_ACTIONS:-}" = "true" ] || return 0
  export GIT_AUTHOR_NAME="${GIT_AUTHOR_NAME:-github-actions[bot]}"
  export GIT_AUTHOR_EMAIL="${GIT_AUTHOR_EMAIL:-41898282+github-actions[bot]@users.noreply.github.com}"
  export GIT_COMMITTER_NAME="${GIT_COMMITTER_NAME:-$GIT_AUTHOR_NAME}"
  export GIT_COMMITTER_EMAIL="${GIT_COMMITTER_EMAIL:-$GIT_AUTHOR_EMAIL}"
}
