# Release distribution

Pushing a `vX.Y.Z` tag runs `.github/workflows/pypi-publish.yml` ("Publish coordinated release"). The `release` job runs the test gates, builds the wheel, sdist and npm tarball, records release evidence, and publishes PyPI and npm. Only after both registries are verified does the fan-out start: a `vsix` job builds the VS Code extension once, then the `distribute` job runs one isolated job per channel. A failing channel never stops the others, and the `distribution-summary` job writes a table of every channel (ok, skipped or failed) to the run's summary page. The run is marked failed if any channel failed.

Every channel is a script in `scripts/release/<channel>.sh`. Each one accepts `--dry-run` (read-only: it checks the current state and says what it would do), `--version X.Y.Z` (defaults to `package.json`) and `--dist DIR`. Each is idempotent, so re-running the workflow for the same tag (Actions → "Publish coordinated release" → Run workflow, with the tag) finishes whatever failed and does nothing to channels that are already current. Secrets reach the scripts only through environment variables, each job receives only its own secret, and no script prints one.

## Channels

| Channel | Automatic or manual | How | Secret |
| --- | --- | --- | --- |
| PyPI | Automatic | `release` job, trusted publishing (OIDC) | none (environment `pypi`) |
| npm | Automatic | `release` job, trusted publishing (OIDC) on Node 24 with npm 11.21.1, `--provenance` | none (environment `pypi`) |
| GitHub release | Automatic | `github-release.sh`: creates the release with the CHANGELOG section as notes and attaches the wheel, sdist, npm tarball, evidence, SBOM and `.vsix`; on an existing release it only uploads missing assets and refuses to replace a different published one | `GITHUB_TOKEN` |
| MCP Registry | Automatic | `mcp-registry.sh`: pinned `mcp-publisher` 1.8.1 (checksum verified), `login github-oidc`, `publish server.json`; skipped when the registry already lists the version | none (GitHub OIDC) |
| Open VSX | Automatic | `build-vsix.sh` then `openvsx.sh`: `ovsx@1.2.0 publish`; skipped when Open VSX already has the version | `OVSX_PAT` |
| Claude plugin repository | Automatic | `claude-plugin-repo.sh`: copies `plugin/` to github.com/qualixar/superlocalmemory-plugin with the literal POSIX launch command, commits, tags `vX.Y.Z`, pushes over SSH. Refuses to publish if the generated plugins are stale | `DEPLOY_KEY_SUPERLOCALMEMORY_PLUGIN` |
| Claude plugin directory | Manual | Approve the new version at claude.ai/directory/manage (auto-publish is off). Listed in the checklist issue | none |
| Hermes plugin | Automatic (nothing to push) | Hermes installs `superlocalmemory.git#hermes-plugin`, which is the `hermes-plugin/` folder of the default branch, not a branch. `hermes.sh` fails the run if `main` still carries an older plugin version, which means the release was not merged to `main` | none |
| DeepSeek Harness bundle | Automatic | `dsh-bundle.sh`: sets the version in github.com/qualixar/dsh-superlocalmemory `package.json` to the release version and pushes over SSH; no change, no push | `DEPLOY_KEY_DSH_SUPERLOCALMEMORY` |
| ClawHub skill | Automatic when changed | `clawhub.sh`: compares `clawhub-skill/superlocalmemory/` file hashes with the published skill and republishes only when they differ (`clawhub@0.23.3`) | `CLAWHUB_TOKEN` |
| Manual listings | Manual, tracked | `listing-checklist.sh` opens one issue per release, "Release vX.Y.Z — manual listing checklist", listing the Claude directory, Cursor Marketplace, VS Code Marketplace, the Antigravity form, the Kimi curated tier, the Kilo, Cline, OpenCode and awesome-dsh-plugin PRs, the xAI marketplace and the Hermes catalog | `GITHUB_TOKEN` |

The manual list lives in `scripts/release/release_tools.py` (`MANUAL_LISTINGS`); edit it there when a listing is added or approved for good.

## Keeping generated plugins honest

The plugin folders are generated from `plugin-src/` by `node scripts/build-all-plugins.mjs` but committed, because hosts install them straight from git. The `plugin-drift` job in `.github/workflows/test.yml` rebuilds them on every pull request and fails if the result differs from what is committed. The Claude plugin channel runs the same check before it publishes.

## Running a channel by hand

```bash
scripts/release/openvsx.sh --dry-run --version 4.2.8
scripts/release/claude-plugin-repo.sh --dry-run          # clones, shows the diff, pushes nothing
OVSX_PAT=... scripts/release/openvsx.sh --version 4.2.8  # a real publish
```

Without `SLM_DEPLOY_KEY`, the two repository channels use your own GitHub access over HTTPS. `scripts/publish-claude-plugin-repo.sh` still works and calls `scripts/release/claude-plugin-repo.sh`.
