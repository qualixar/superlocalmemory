# Release checklist

Every registry, in order. A release is not done until each line is checked.

Since 4.2.8, pushing the tag (step 3) does the rest of steps 3 to 9 automatically,
except approving the version in the Claude directory (see
[release-distribution.md](release-distribution.md)): check the run's channel
table and work through the "manual listing checklist" issue it opens. The
commands below are the manual fallback for a channel that failed.

1. `python3 scripts/bump_version.py X.Y.Z`, then `node scripts/build-all-plugins.mjs`.
2. One full test suite, then the live-store gate (glance, session_init and hook recall on a real store).
3. Tag `vX.Y.Z`, push, GitHub release with the wheel, sdist, npm tgz and vsix attached.
4. PyPI: `uv build` then `uv publish`. Check the hashes PyPI shows against `dist/`.
5. npm (needs the owner's 2FA): `npm publish ./dist/superlocalmemory-X.Y.Z.tgz --loglevel notice`.
   The `./` is required: a bare `dist/x.tgz` is read as GitHub shorthand, and a
   `loglevel=silent` in `~/.npmrc` hides the error. Compare `npm view superlocalmemory dist.shasum`.
6. MCP Registry: `mcp-publisher login github --token "$(gh auth token)" && mcp-publisher publish`.
7. Open VSX: `npx ovsx@1.2.0 publish dist/superlocalmemory-vscode-X.Y.Z.vsix`.
8. Claude plugin directory: `scripts/publish-claude-plugin-repo.sh`, then approve the new
   version at claude.ai/directory/manage (auto-publish is off).
9. ClawHub, only if `clawhub-skill/superlocalmemory/SKILL.md` changed:
   `clawhub skill publish clawhub-skill/superlocalmemory --slug superlocalmemory --name SuperLocalMemory --changelog "<what changed>"`.
10. Upgrade the local hosts, run `slm doctor`, and reply on every issue the release closes.
