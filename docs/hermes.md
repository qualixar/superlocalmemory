# Hermes native integration

SuperLocalMemory 4.1.12 ships a native Hermes plugin. It is a companion to,
not a replacement for, Hermes's built-in memory provider and configuration.

Install the owning runtime first:

```bash
python -m pip install --upgrade superlocalmemory==4.1.12
slm doctor
```

Then download the reviewed, immutable plugin pack from the `v4.1.12` GitHub
release. Hermes packs pin each monorepo plugin with its repository, subdirectory,
and exact 40-character commit SHA; the pack cannot grant capabilities or MCP
access on your behalf:

```bash
curl -LO https://github.com/qualixar/superlocalmemory/releases/download/v4.1.12/qualixar-agent-reliability-hermes-pack.yaml
hermes plugins pack show qualixar-agent-reliability-hermes-pack.yaml
hermes plugins pack install qualixar-agent-reliability-hermes-pack.yaml
```

The plugin registers 12 namespaced skills, four on-demand Hermes child-agent
roles, `/slm <command>`, and `/slm-<command>` aliases for the public SLM CLI.
It calls only the configured `superlocalmemory` MCP server. Grant that server
to this plugin when Hermes asks; no wildcard MCP grant is needed.

By default the plugin recalls bounded, untrusted evidence and records scrubbed
tool telemetry. Full user/assistant turn capture is off by default; enable it
only by setting `plugins.entries.superlocalmemory.capture_turns: true` in your
Hermes configuration.

The pack installs the two additive native plugins independently. The bridge
remains inactive unless both products are installed and their individual MCP
grants are approved.

`bounded-loops.dev/slm-bridge/v1` remains observation-only. Bridge v2 learns
only validated execution-reliability signals from eligible terminal receipts;
it never turns a passing gate into a semantic memory or a user preference.
