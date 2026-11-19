<!--
Thanks for contributing to SuperLocalMemory. Read CONTRIBUTING.md first.
Security fixes: don't open a public PR, report privately (see SECURITY.md).
-->

## What this changes

<!-- One or two sentences. What does a user or agent see differently after this merges? -->

Fixes #<!-- issue number; open one first for anything larger than a small fix -->

## Why

<!-- The problem, and why this is the right fix rather than a workaround. -->

## How it was tested

<!-- Commands you ran and what they showed. For bug fixes, a test that fails before this change and passes after. -->

- [ ] `pytest` on the tests this touches passes locally (the full suite runs in CI)
- [ ] `ruff check` is clean on the changed files
- [ ] Checked by hand where it matters (CLI, MCP tool call, dashboard or agent host), described above

## Type of change

- [ ] Bug fix (non-breaking)
- [ ] New feature (non-breaking)
- [ ] Breaking change (existing behaviour, config, CLI flags, MCP tool names or stored data change)
- [ ] Documentation only
- [ ] Build, CI or packaging

## Checklist

- [ ] Memory content never leaves the machine unless the user chose a feature that sends it, and no secrets are logged
- [ ] Upgrades migrate existing installs automatically, with no manual step for the user
- [ ] Docs, `CHANGELOG.md` and the plugin copies (`node scripts/build-all-plugins.mjs`) are updated where needed
- [ ] I disclosed any AI assistance below and can explain every line of this change myself

## AI assistance

<!-- Which tools helped (e.g. Claude Code, Codex), and for what. "None" is fine. -->

## Notes for reviewers

<!-- Design decisions, trade-offs, or places you'd like a closer look. -->
