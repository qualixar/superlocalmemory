#!/usr/bin/env bash
# Build a macOS DMG from one frozen, locally supplied SuperLocalMemory wheel.
# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later; see LICENSE.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: build-dmg.sh --wheel PATH [OPTIONS]

Build a macOS disk image from exactly one frozen local wheel. The wheel's
embedded package version must match pyproject.toml. This command never builds
or downloads a package and never publishes the result.

Required:
  --wheel PATH             Frozen superlocalmemory wheel to package

Options:
  --output-dir PATH        Artifact directory (default: dist/macos)
  --build-root PATH        Staging parent (default: build/dmg)
  --stage-only             Prepare and validate payload without creating a DMG
  --sign-identity NAME     Developer ID identity passed to codesign
  --notary-profile NAME    Existing notarytool keychain profile
  --help, -h               Show this help

Signing/notarization are explicit hooks. Credentials must already exist in the
macOS keychain; this script accepts no passwords, API keys, or private keys.
Without both options the output is an unsigned local candidate, not a release.
EOF
}

fail() {
    printf 'Error: %s\n' "$1" >&2
    exit "${2:-2}"
}

WHEEL=""
OUTPUT_DIR=""
BUILD_ROOT=""
SIGN_IDENTITY=""
NOTARY_PROFILE=""
STAGE_ONLY=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --wheel)
            [[ $# -ge 2 ]] || fail "--wheel requires a path"
            WHEEL="$2"
            shift 2
            ;;
        --output-dir)
            [[ $# -ge 2 ]] || fail "--output-dir requires a path"
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --build-root)
            [[ $# -ge 2 ]] || fail "--build-root requires a path"
            BUILD_ROOT="$2"
            shift 2
            ;;
        --stage-only)
            STAGE_ONLY=true
            shift
            ;;
        --sign-identity)
            [[ $# -ge 2 ]] || fail "--sign-identity requires an identity"
            SIGN_IDENTITY="$2"
            shift 2
            ;;
        --notary-profile)
            [[ $# -ge 2 ]] || fail "--notary-profile requires a profile"
            NOTARY_PROFILE="$2"
            shift 2
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

[[ -n "$WHEEL" ]] || fail "--wheel is required"
[[ -f "$WHEEL" ]] || fail "wheel not found: $WHEEL"
if [[ -n "$NOTARY_PROFILE" && -z "$SIGN_IDENTITY" ]]; then
    fail "--notary-profile requires --sign-identity"
fi
if [[ "$STAGE_ONLY" == true && ( -n "$SIGN_IDENTITY" || -n "$NOTARY_PROFILE" ) ]]; then
    fail "signing and notarization are not valid with --stage-only"
fi

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
readonly HELPER="$SCRIPT_DIR/dmg_release.py"
[[ -f "$HELPER" ]] || fail "DMG release helper is missing: $HELPER"

command -v python3 >/dev/null 2>&1 || fail "python3 is required" 127
python3 -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' || \
    fail "python3 3.11 or newer is required" 127

if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="$PROJECT_ROOT/dist/macos"
fi
if [[ -z "$BUILD_ROOT" ]]; then
    BUILD_ROOT="$PROJECT_ROOT/build/dmg"
fi
mkdir -p "$OUTPUT_DIR" "$BUILD_ROOT"
OUTPUT_DIR="$(cd -- "$OUTPUT_DIR" && pwd -P)"
BUILD_ROOT="$(cd -- "$BUILD_ROOT" && pwd -P)"
WHEEL="$(cd -- "$(dirname -- "$WHEEL")" && pwd -P)/$(basename -- "$WHEEL")"

VERSION="$(python3 - "$PROJECT_ROOT/pyproject.toml" <<'PY'
import sys, tomllib
with open(sys.argv[1], "rb") as stream:
    print(tomllib.load(stream)["project"]["version"])
PY
)"
[[ -n "$VERSION" ]] || fail "could not read project version"

STAGE_DIR="$(mktemp -d "$BUILD_ROOT/slm-dmg.${VERSION}.XXXXXX")"
# prepare_stage requires ownership of a path that does not yet exist.
rmdir "$STAGE_DIR"
python3 "$HELPER" prepare \
    --wheel "$WHEEL" \
    --project-root "$PROJECT_ROOT" \
    --stage-dir "$STAGE_DIR" >/dev/null
readonly VOLUME_DIR="$STAGE_DIR/SuperLocalMemory"
python3 "$HELPER" validate-stage --volume "$VOLUME_DIR" >/dev/null
printf 'Validated frozen payload: %s\n' "$VOLUME_DIR"

if [[ "$STAGE_ONLY" == true ]]; then
    printf 'Stage-only build complete. No DMG was created.\n'
    exit 0
fi

[[ "$(uname -s)" == "Darwin" ]] || fail "DMG creation requires macOS"
command -v hdiutil >/dev/null 2>&1 || fail "hdiutil is required" 127

readonly DMG_NAME="SuperLocalMemory-v${VERSION}-macos-universal.dmg"
readonly DMG_PATH="$OUTPUT_DIR/$DMG_NAME"
[[ ! -e "$DMG_PATH" ]] || fail "output already exists: $DMG_PATH"
[[ ! -e "$DMG_PATH.manifest.json" ]] || fail "manifest already exists: $DMG_PATH.manifest.json"
[[ ! -e "$DMG_PATH.sha256" ]] || fail "checksum already exists: $DMG_PATH.sha256"

hdiutil create \
    -srcfolder "$VOLUME_DIR" \
    -volname "SuperLocalMemory ${VERSION}" \
    -format UDZO \
    -imagekey zlib-level=9 \
    "$DMG_PATH"
hdiutil verify "$DMG_PATH"

SIGNED=false
NOTARIZED=false
if [[ -n "$SIGN_IDENTITY" ]]; then
    command -v codesign >/dev/null 2>&1 || fail "codesign is required" 127
    codesign --force --sign "$SIGN_IDENTITY" --timestamp "$DMG_PATH"
    codesign --verify --verbose=2 "$DMG_PATH"
    SIGNED=true
fi

if [[ -n "$NOTARY_PROFILE" ]]; then
    command -v xcrun >/dev/null 2>&1 || fail "xcrun is required" 127
    xcrun notarytool submit "$DMG_PATH" \
        --keychain-profile "$NOTARY_PROFILE" \
        --wait
    xcrun stapler staple "$DMG_PATH"
    xcrun stapler validate "$DMG_PATH"
    NOTARIZED=true
fi

SIDECAR_ARGS=(
    sidecars
    --dmg "$DMG_PATH"
    --version "$VERSION"
)
if [[ "$SIGNED" == true ]]; then
    SIDECAR_ARGS+=(--signed)
fi
if [[ "$NOTARIZED" == true ]]; then
    SIDECAR_ARGS+=(--notarized)
fi
python3 "$HELPER" "${SIDECAR_ARGS[@]}" >/dev/null

printf 'DMG candidate: %s\n' "$DMG_PATH"
printf 'Manifest: %s.manifest.json\n' "$DMG_PATH"
printf 'Checksum: %s.sha256\n' "$DMG_PATH"
if [[ "$SIGNED" != true || "$NOTARIZED" != true ]]; then
    printf 'CANDIDATE ONLY: strict release validation will reject this unsigned or unnotarized DMG.\n'
fi
