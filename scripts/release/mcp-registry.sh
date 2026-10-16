#!/usr/bin/env bash
# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
#
# mcp-registry.sh — publish server.json (io.github.qualixar/superlocalmemory)
# to the official MCP Registry.
#
#   scripts/release/mcp-registry.sh [--dry-run] [--version X.Y.Z]
#
# Auth is GitHub OIDC (`mcp-publisher login github-oidc`), so the job needs
# `id-token: write` and no stored secret. The registry checks that the npm
# package named in server.json carries the matching mcpName, so this runs only
# after npm has the release. A version the registry already lists is a no-op.
# mcp-publisher is pinned (MCP_PUBLISHER_VERSION) and its download is checked
# against the release's published SHA-256.

set -euo pipefail
SLM_CHANNEL=mcp-registry
# shellcheck source=scripts/release/_lib.sh
source "$(dirname "$0")/_lib.sh"
slm_parse_args "$@"

MCP_PUBLISHER_VERSION=1.8.1
REGISTRY=https://registry.modelcontextprotocol.io
SERVER_NAME="io.github.qualixar/superlocalmemory"

# sha256 of mcp-publisher_<os>_<arch>.tar.gz, from registry_1.8.1_checksums.txt.
publisher_sha256() {
  case "$1" in
    linux_amd64) echo a06c9096dcb9727c13555b6be26c7effa707b01f06a4c561ba7a3635443cf2cc ;;
    linux_arm64) echo 8dd75a6cf6845688b5d4e46df58d3ca26d5c8d233bb0626606e1db82c5e883e4 ;;
    darwin_amd64) echo 88126981225e7714fcc6b7a10cdba4a80ae5901e9740a8c06d0d5195c8bc294c ;;
    darwin_arm64) echo e45e520892460732a4bdf37255576415d4a53ec171f8b913faf15bb1aef7cb77 ;;
    *) return 1 ;;
  esac
}

# install_publisher — sets PUBLISHER to a pinned, checksum-verified binary.
install_publisher() {
  if command -v mcp-publisher >/dev/null \
    && mcp-publisher --version 2>&1 | grep -q "mcp-publisher $MCP_PUBLISHER_VERSION "; then
    PUBLISHER="$(command -v mcp-publisher)"
    return 0
  fi
  local os arch platform expected dir
  os="$(uname -s | tr '[:upper:]' '[:lower:]')"
  arch="$(uname -m | sed 's/x86_64/amd64/;s/aarch64/arm64/')"
  platform="${os}_${arch}"
  expected="$(publisher_sha256 "$platform")" || slm_die "no pinned mcp-publisher for $platform"
  slm_tmpdir dir
  curl -fsSL --retry 3 -o "$dir/p.tar.gz" \
    "https://github.com/modelcontextprotocol/registry/releases/download/v$MCP_PUBLISHER_VERSION/mcp-publisher_$platform.tar.gz"
  python3 - "$dir/p.tar.gz" "$expected" <<'PY' || slm_die "mcp-publisher download failed its checksum"
import hashlib, sys
digest = hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest()
sys.exit(0 if digest == sys.argv[2] else 1)
PY
  tar -xzf "$dir/p.tar.gz" -C "$dir" mcp-publisher
  PUBLISHER="$dir/mcp-publisher"
}

cd "$SLM_ROOT"
python3 - "$SLM_VERSION" "$SERVER_NAME" <<'PY' || slm_die "server.json does not describe $SLM_VERSION"
import json, sys
version, name = sys.argv[1], sys.argv[2]
server = json.load(open("server.json"))
problems = []
if server.get("name") != name:
    problems.append(f"name is {server.get('name')!r}")
if server.get("version") != version:
    problems.append(f"version is {server.get('version')!r}")
problems += [f"package {p.get('identifier')} is {p.get('version')!r}"
             for p in server.get("packages", []) if p.get("version") != version]
if problems:
    sys.exit("server.json: " + "; ".join(problems))
PY

encoded="${SERVER_NAME//\//%2F}"
code="$(slm_http_status "$REGISTRY/v0/servers/$encoded/versions/$SLM_VERSION")"
case "$code" in
  200) slm_status skipped "$SERVER_NAME $SLM_VERSION is already on the MCP Registry"; exit 0 ;;
  404) slm_log "$SLM_VERSION is not on the registry yet" ;;
  *) slm_die "registry lookup returned HTTP $code" ;;
esac

install_publisher
if [ "$SLM_DRY_RUN" -eq 1 ]; then
  "$PUBLISHER" validate
  slm_status ok "dry run: server.json valid; would publish $SLM_VERSION"
  exit 0
fi
[ -n "${ACTIONS_ID_TOKEN_REQUEST_URL:-}" ] || slm_die "GitHub OIDC is unavailable (the job needs id-token: write)"
"$PUBLISHER" login github-oidc
"$PUBLISHER" publish
slm_status ok "published $SERVER_NAME $SLM_VERSION"
