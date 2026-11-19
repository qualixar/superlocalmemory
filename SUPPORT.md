# Getting help with SuperLocalMemory

SuperLocalMemory is open source, made by [Varun Pratap Bhardwaj](https://varunpratap.com) at [Qualixar](https://qualixar.com). Here is where to go, fastest route first.

## Fix it yourself in a minute

1. Run `slm doctor --fix`. It repairs what it can (missing models, `sqlite-vec`) and reports the rest.
2. Read the [troubleshooting guide](docs/troubleshooting.md). It covers the problems people actually hit, with exact fixes.
3. Upgrade. Many reports are already fixed in the [latest release](https://github.com/qualixar/superlocalmemory/releases/latest): `pipx upgrade superlocalmemory` or `npm update -g superlocalmemory`, then `slm restart`.

## Ask a question

Use [GitHub Discussions → Q&A](https://github.com/qualixar/superlocalmemory/discussions/categories/q-a) for setup help, "how do I…" questions and advice on using SLM with your agents. Answers there stay searchable for the next person.

## Report a bug or request a feature

Open an [issue](https://github.com/qualixar/superlocalmemory/issues/new/choose) and pick the matching form. Include `slm --version` and the output of `slm doctor`. Check pasted output for anything private first: logs can contain memory text and file paths. `slm diagnostics export ./slm-diagnostics.json` writes a counters-only summary with no memory content, which is safe to attach.

## Security problems

Never in a public issue. Use [private vulnerability reporting](https://github.com/qualixar/superlocalmemory/security/advisories/new) or email admin@superlocalmemory.com. Details in [SECURITY.md](SECURITY.md).

## Commercial and enterprise use

SuperLocalMemory is dual-licensed: [AGPL-3.0-or-later](LICENSE), or a [commercial licence](COMMERCIAL-LICENSE.md) for teams that can't adopt AGPL terms. The commercial licence page has the contact for licensing, team deployment and support agreements.

## What to expect

Issues and discussions are read by the maintainer and triaged as time allows; there is no guaranteed response time for community support. Clear reports with steps to reproduce get fixed first.
