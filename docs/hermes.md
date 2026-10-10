# Hermes native integration

SuperLocalMemory 4.1.25 ships a native Hermes plugin. It is a companion to,
not a replacement for, Hermes's built-in memory provider and configuration.

Install the owning runtime first:

```bash
python -m pip install --upgrade superlocalmemory==4.1.25
slm doctor
```

Then install the plugin from this repository:

```bash
hermes plugins install 'https://github.com/qualixar/superlocalmemory.git#hermes-plugin'
```

To pin an exact release, add `--ref` with that release's 40-character commit.
Hermes asks before it enables the plugin (`--no-enable` installs it disabled);
the plugin cannot grant itself capabilities or MCP access.

The plugin registers 15 namespaced skills, four on-demand Hermes child-agent
roles, `/slm <command>`, and `/slm-<command>` aliases for the public SLM CLI.
It calls only the configured `superlocalmemory` MCP server. Grant that server
to this plugin when Hermes asks; no wildcard MCP grant is needed.

By default the plugin recalls bounded, untrusted evidence and records scrubbed
tool telemetry. Full user/assistant turn capture is off by default; enable it
only by setting `plugins.entries.superlocalmemory.settings.capture_turns: true`
in your Hermes configuration (Hermes reads plugin settings from the `settings`
block of the plugin's entry).

If you also install the Bounded Loops Hermes plugin, the two stay independent. The
bridge between them remains inactive unless both products are installed and their
individual MCP grants are approved.

`bounded-loops.dev/slm-bridge/v1` remains observation-only. Bridge v2 learns
only validated execution-reliability signals from eligible terminal receipts;
it never turns a passing gate into a semantic memory or a user preference.

## Use an SLM that runs on another computer

If your memory lives on a dedicated SLM server, the Hermes plugin can use it
directly. You do not need SuperLocalMemory installed on the Hermes computer.

**On the SLM server** (once):

```bash
slm remote tls init --name slm.lan --ip 192.168.50.144
slm remote enable --listen 192.168.50.144:8443
slm restart
slm remote keys add hermes-laptop --profile work   # prints the key once - copy it now
slm remote check
```

Copy `remote/tls/ca.pem` from the SLM data folder (by default
`~/.superlocalmemory/remote/tls/ca.pem`) to the Hermes computer. It is a public
certificate, not a secret. Add `--read-only` to `keys add` for a Hermes that
should only recall.

The key reaches one profile only: `--profile`, or the profile active when you
create the key. Hermes recalls from and saves into that profile and cannot
name another one; saves stay private to it. Whoever holds the key can read the
full text of every memory in that profile, including any paths or pasted output
written into them, so give Hermes a profile that holds only what it should see.
Every tool Hermes can call (recall, saves, search, list, update, delete,
session and lifecycle capture, status, kinds and the rest) works on the key's
profile whatever profile the server is using, and never moves the server's
active profile. A key made before 4.1.20 is bound to the profile active when
the server is upgraded; `slm remote keys list` shows which.

**On the Hermes computer**, put the key in your Hermes secrets as
`SLM_REMOTE_KEY`, then in `~/.hermes/config.yaml`:

```yaml
mcp_servers:
  superlocalmemory:
    url: "https://192.168.50.144:8443/mcp/hermes"
    headers:
      Authorization: "Bearer ${SLM_REMOTE_KEY}"
    ssl_verify: "/path/to/slm-ca.pem"
plugins:
  entries:
    superlocalmemory:
      mcp_allowlist: ["superlocalmemory"]
      settings:
        connection: remote
```

Recall, lifecycle capture, skills and the memory advisor now use the SLM
server. `/slm status`, `recall` (and `search`), `remember`, `list`, `delete`,
`update`, `summary`, `trace`, `kinds`, `health` and `help` work remotely.
Commands that manage the server (`serve`, `backup`, `profile`, `remote`, ...)
and the governance and loop advisors run on the server itself. The plugin
opens no network connection of its own: every call goes through the MCP server
you configured above, so the address, TLS check and key live only in Hermes's
configuration.

If the server cannot be reached, `/slm remember` says **NOT SAVED** and nothing
is stored. Automatic capture pauses for 30 seconds at a time, and `/slm status`
shows how many captures were skipped and the last error. Nothing is kept on the
Hermes computer to replay later; re-save anything that matters.

With a read-only key, recall works and automatic capture is switched off after
the first refused write (`/slm status` counts what was not sent).

Never set `ssl_verify: false`, and never put the key in the URL. To rotate a
key, add a new one, update `SLM_REMOTE_KEY`, then `slm remote keys revoke
hermes-laptop` on the server; revocation takes effect on the next request.
