#!/usr/bin/env bash
# Verify the npm artifact without changing the machine's global npm state.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/slm-npm-pack.XXXXXX")"
TARBALL=""

cleanup() {
    if [ -n "$TARBALL" ] && [ -f "$TARBALL" ]; then
        rm -f -- "$TARBALL"
    fi
    rmdir -- "$TEMP_DIR" 2>/dev/null || true
}
trap cleanup EXIT

required_files=(
    package.json
    pyproject.toml
    bin/slm-npm
    scripts/postinstall.js
    scripts/preuninstall.js
    scripts/install.sh
    scripts/install.ps1
    README.md
    LICENSE
    NOTICE
    ATTRIBUTION.md
)

for relative_path in "${required_files[@]}"; do
    if [ ! -f "$REPO_DIR/$relative_path" ]; then
        echo "Missing required repository file: $relative_path" >&2
        exit 1
    fi
done

if [ -e "$REPO_DIR/install.sh" ] || [ -e "$REPO_DIR/install.ps1" ]; then
    echo "Repository lifecycle installers must stay scoped under scripts/." >&2
    exit 1
fi

node -e '
const pkg = require(process.argv[1]);
for (const field of ["name", "version", "description", "author", "license", "repository", "bin"]) {
  if (!pkg[field]) throw new Error(`package.json is missing ${field}`);
}
if (pkg.bin.slm !== "./bin/slm-npm") throw new Error("Unexpected slm wrapper");
if (pkg.scripts.postinstall !== "node scripts/postinstall.js") throw new Error("Unexpected postinstall");
' "$REPO_DIR/package.json"

PACK_JSON="$(cd "$REPO_DIR" && npm pack --json --ignore-scripts --pack-destination "$TEMP_DIR")"
TARBALL_NAME="$(printf '%s' "$PACK_JSON" | node -e '
let input = "";
process.stdin.on("data", chunk => input += chunk);
process.stdin.on("end", () => {
  const result = JSON.parse(input);
  if (!Array.isArray(result) || result.length !== 1 || !result[0].filename) process.exit(1);
  process.stdout.write(result[0].filename);
});
')"
TARBALL="$TEMP_DIR/$TARBALL_NAME"

if [ ! -f "$TARBALL" ]; then
    echo "npm pack did not create the declared artifact." >&2
    exit 1
fi

CONTENTS="$(tar -tzf "$TARBALL")"
for packaged_path in \
    package/bin/slm-npm \
    package/scripts/postinstall.js \
    package/pyproject.toml; do
    if ! grep -Fxq "$packaged_path" <<<"$CONTENTS"; then
        echo "Missing packaged runtime file: $packaged_path" >&2
        exit 1
    fi
done

for forbidden_path in \
    package/scripts/install.sh \
    package/scripts/install.ps1 \
    package/install.sh \
    package/install.ps1; do
    if grep -Fxq "$forbidden_path" <<<"$CONTENTS"; then
        echo "Repository lifecycle installer leaked into npm artifact: $forbidden_path" >&2
        exit 1
    fi
done

node --test "$REPO_DIR/tests/postinstall/test_npm_runtime_isolation.js"
echo "npm artifact contract verified: $TARBALL_NAME"
