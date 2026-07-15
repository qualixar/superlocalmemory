#!/usr/bin/env bash
# Validate a SuperLocalMemory macOS DMG and its release provenance sidecars.
# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later; see LICENSE.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: test-dmg.sh --dmg PATH [--require-release-ready]

Validate container integrity, the complete mounted payload inventory, the
frozen wheel checksum/version, and final DMG checksum sidecars.

Options:
  --dmg PATH                  DMG to validate
  --require-release-ready     Require Developer ID signing and Apple
                              notarization; verify signature and staple
  --help, -h                  Show this help
EOF
}

fail() {
    printf 'Error: %s\n' "$1" >&2
    exit "${2:-2}"
}

DMG=""
REQUIRE_RELEASE_READY=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dmg)
            [[ $# -ge 2 ]] || fail "--dmg requires a path"
            DMG="$2"
            shift 2
            ;;
        --require-release-ready)
            REQUIRE_RELEASE_READY=true
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            fail "unknown option: $1"
            ;;
    esac
done

[[ -n "$DMG" ]] || fail "--dmg is required"
[[ -f "$DMG" ]] || fail "DMG not found: $DMG"
[[ "$(uname -s)" == "Darwin" ]] || fail "DMG validation requires macOS"
command -v python3 >/dev/null 2>&1 || fail "python3 is required" 127
python3 -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' || \
    fail "python3 3.11 or newer is required" 127
command -v hdiutil >/dev/null 2>&1 || fail "hdiutil is required" 127

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly HELPER="$SCRIPT_DIR/dmg_release.py"
DMG="$(cd -- "$(dirname -- "$DMG")" && pwd -P)/$(basename -- "$DMG")"

SIDECAR_ARGS=(validate-sidecars --dmg "$DMG")
if [[ "$REQUIRE_RELEASE_READY" == true ]]; then
    SIDECAR_ARGS+=(--require-release-ready)
fi
python3 "$HELPER" "${SIDECAR_ARGS[@]}" >/dev/null
hdiutil verify "$DMG"

ATTACH_OUTPUT="$(hdiutil attach "$DMG" -nobrowse -readonly -noverify)"
DEVICE="$(printf '%s\n' "$ATTACH_OUTPUT" | awk '/Apple_HFS|Apple_APFS/ {print $1; exit}')"
MOUNT_POINT="$(printf '%s\n' "$ATTACH_OUTPUT" | awk -F '\t' '/\/Volumes\// {print $NF; exit}')"
[[ -n "$DEVICE" && -n "$MOUNT_POINT" && -d "$MOUNT_POINT" ]] || \
    fail "could not identify mounted DMG"

detach() {
    hdiutil detach "$DEVICE" -quiet
}
trap detach EXIT HUP INT TERM

PAIR_ARGS=(validate-pair --volume "$MOUNT_POINT" --dmg "$DMG")
if [[ "$REQUIRE_RELEASE_READY" == true ]]; then
    PAIR_ARGS+=(--require-release-ready)
fi
python3 "$HELPER" "${PAIR_ARGS[@]}" >/dev/null

if [[ "$REQUIRE_RELEASE_READY" == true ]]; then
    command -v codesign >/dev/null 2>&1 || fail "codesign is required" 127
    command -v xcrun >/dev/null 2>&1 || fail "xcrun is required" 127
    command -v spctl >/dev/null 2>&1 || fail "spctl is required" 127
    codesign --verify --verbose=2 "$DMG"
    xcrun stapler validate "$DMG"
    spctl --assess --type open --context context:primary-signature --verbose=2 "$DMG"
fi

detach
trap - EXIT HUP INT TERM
printf 'DMG validation passed: %s\n' "$DMG"
if [[ "$REQUIRE_RELEASE_READY" != true ]]; then
    printf 'Candidate validation only. Use --require-release-ready for the distribution gate.\n'
fi
