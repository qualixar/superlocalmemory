# macOS DMG release contract

The macOS DMG is an optional distribution wrapper for one frozen
SuperLocalMemory wheel. It is not a source build, a repository snapshot, or a
different installation implementation. The wheel embedded in the image must
have the same version as `pyproject.toml`, and the bundled installer delegates
to the same isolated `uv` or `pipx` contract as `scripts/install.sh`.

## Candidate build

Build the wheel first using the normal release workflow. Then pass that exact
local artifact to the DMG builder:

```bash
./scripts/build-dmg.sh \
  --wheel /absolute/path/to/superlocalmemory-<version>-py3-none-any.whl
```

The builder does not download or rebuild the package. It writes the DMG, an
external JSON provenance manifest, and a SHA-256 sidecar under `dist/macos/`.
It refuses to overwrite an existing artifact. Without signing and notarization
options, the output is explicitly marked as an unsigned local candidate and is
not distributable.

For a platform-neutral inspection of the exact staged payload, use
`--stage-only`. This exercises version, license, inventory, executable-mode,
and checksum validation without creating a disk image.

## Signing and notarization hook

Release automation may opt into the macOS keychain-backed hooks:

```bash
./scripts/build-dmg.sh \
  --wheel /absolute/path/to/superlocalmemory-<version>-py3-none-any.whl \
  --sign-identity "Developer ID Application: <identity>" \
  --notary-profile "<notarytool-keychain-profile>"
```

The script accepts only identity/profile names. It does not accept, print, or
persist credentials. The profile must already exist in the macOS keychain.
Signing and notarization mutate the DMG, so its external manifest and checksum
are generated only after both operations finish.

## Verification gates

Candidate verification checks container integrity, every mounted file against
the internal artifact manifest, the embedded wheel metadata/checksum, and the
final DMG sidecars:

```bash
./scripts/test-dmg.sh --dmg dist/macos/SuperLocalMemory-v<version>-macos-universal.dmg
```

The distribution gate is stricter:

```bash
./scripts/test-dmg.sh \
  --dmg dist/macos/SuperLocalMemory-v<version>-macos-universal.dmg \
  --require-release-ready
```

That command fails unless the sidecar records Developer ID signing and Apple
notarization, the mounted wheel version matches the final DMG identity,
`codesign` verifies the final bytes, `stapler` validates the notarization ticket,
and Gatekeeper accepts the image through `spctl`. Candidate success must not be
reported as release success.

## Installation and uninstall semantics

`INSTALL.command` installs the embedded wheel into an isolated, user-scoped
`uv` or `pipx` environment. It does not bootstrap a package manager, contact a
package registry for the SuperLocalMemory package, or edit IDE settings. Run
`slm setup` afterward to choose integrations explicitly.

`UNINSTALL.command` removes application code through the owning package
manager. It preserves runtime memory data. The DMG contains the AGPL-3.0-or-
later license, `NOTICE`, and attribution files.
