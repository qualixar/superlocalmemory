# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""No module opens an outbound connection outside the gate unless it is reviewed.

Credentials stay in SLM, so every byte of memory text that leaves the machine
must pass ``core.outbound_http``. This test walks the source of the package and
the integration adapters, finds every call that opens a connection (httpx,
urllib, requests, aiohttp, http.client, raw sockets, the OpenAI/Anthropic SDKs,
Google API clients, IMAP/SMTP, and subprocess curl/wget), and fails when one is
not in the reviewed list below. Each entry names why it may skip the gate. The
list is exact: a stale entry fails too, and so does a second call of the same
kind added to a reviewed function.

To add an exit: send it through ``core.outbound_http``. If it truly carries no
memory text, add it here with the reason.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from tests.test_security._egress_scan import scan_source, scan_tree

_REPO = Path(__file__).resolve().parents[2]
_ROOTS = ("src/superlocalmemory", "integrations", "ide/integrations")
_GATE = "src/superlocalmemory/core/outbound_http.py"

#: Why an exit may skip the gate. Every reason starts with one of these.
_CATEGORIES = (
    "PROBE:",          # no memory text: a health/model-list GET or a fixed probe string
    "SDK:",            # receives only text the redacting dispatcher already screened
    "INBOUND:",        # reads the user's own mail/calendar; sends search parameters only
    "OAUTH:",          # an OAuth code exchange; no memory text
    "PROXY:",          # forwards the user's own agent traffic; never reads the store
    "MESH:",           # agent-to-agent mesh messages to a configured peer, not memory text
    "RELAY:",          # a byte relay between two loopback sockets
    "BACKUP:",         # cloud backup: uploads only client-side-encrypted bytes (L2-17)
)

_S = "src/superlocalmemory/"

#: (file, enclosing function, call) -> (number of call sites, reason)
REVIEWED: dict[tuple[str, str, str], tuple[int, str]] = {
    (_S + "cli/commands.py", "cmd_doctor", "httpx.get"):
        (1, "PROBE: `slm doctor` lists Ollama models (GET /api/tags)"),
    (_S + "cli/remote_commands.py", "_probe", "socket.create_connection"):
        (1, "PROBE: `slm remote check` TLS handshake to this SLM's own remote "
            "listener (configured host:port), verified with its CA; sends no data"),
    (_S + "cli/daemon.py", "_fetch_health", "urllib.request.urlopen"):
        (1, "PROBE: GET /health on 127.0.0.1, no body; identity checked after"),
    # cli/optimize_cmd.py, cli/proxy_cmd.py and optimize/proxy/lifecycle.py have
    # no entry: since 4.1.21 their loopback probes go through
    # cli.daemon.owned_daemon_answers (the _fetch_health probe above), so a port
    # answered by another account's SuperLocalMemory is not taken for this one.
    (_S + "cli/setup_wizard.py", "_ollama_installed_models", "httpx.get"):
        (1, "PROBE: is a local Ollama running (GET /api/tags)"),
    (_S + "core/component_registry.py", "probe_ollama", "httpx.get"):
        (1, "PROBE: Ollama model list for the component panel"),
    # core/mcp_embedder_proxy.py has no entry: since 4.1.20 it reaches only its
    # own data root's daemon through cli.daemon.daemon_request, which builds
    # http://127.0.0.1:<descriptor port> and sends via core.outbound_http.urlopen
    # (the gate: no proxy, no redirects). Nothing there opens a connection itself.
    (_S + "llm/ollama_reachability.py", "_probe", "httpx.get"):
        (1, "PROBE: is the configured Ollama answering (GET /api/tags, no body), cached"),
    (_S + "core/media_fetch.py", "_run", "httpx.Client"):
        (1, "INBOUND: downloads one picture or PDF the caller linked (GET only); sends no memory "
            "text. The client is built with trust_env=False and follow_redirects=False; the "
            "host is resolved once and every address must be public, the connection goes to "
            "that checked address (Host and TLS name stay the link's host), redirects are "
            "followed by hand (max 3, each re-checked), https only, 25 MB and 10 s caps (a PDF: "
            "100 MB, streamed to a file, 120 s). "
            "tests/test_security/test_media_fetch.py and test_media_fetch_file_hosts.py pin all of this."),
    (_S + "core/ollama_embedder.py", "OllamaEmbedder._check_availability", "httpx.get"):
        (1, "PROBE: Ollama model list (GET /api/tags)"),
    (_S + "core/ollama_validator.py", "validate_ollama_model", "httpx.post"):
        (2, "PROBE: one fixed probe string to test a model; no memory text"),
    (_S + "core/remember_runtime.py", "_slm_health_check", "urllib.request.urlopen"):
        (1, "PROBE: GET /health on 127.0.0.1, no body"),
    (_S + "evolution/llm_dispatch.py", "_call_claude_api_backend", "anthropic.Anthropic"):
        (1, "SDK: prompt arrives only from _dispatch_llm, hosted-redacted first"),
    (_S + "evolution/llm_dispatch.py", "_call_openai_api_backend", "openai.OpenAI"):
        (1, "SDK: prompt arrives only from _dispatch_llm, hosted-redacted first"),
    (_S + "evolution/skill_evolver.py", "_ollama_running", "urllib.request.urlopen"):
        (1, "PROBE: GET /api/tags on 127.0.0.1, no body"),
    (_S + "hooks/portable_kit.py", "_check_daemon_health", "urllib.request.urlopen"):
        (1, "PROBE: GET /api/v3/health on 127.0.0.1, no body"),
    (_S + "infra/cloud_backup.py", "_get_drive_service", "googleapiclient.discovery.build"):
        (1, "BACKUP: Drive client; uploads only client-side-encrypted bytes"),
    (_S + "infra/cloud_backup.py", "connect_github", "httpx.get"):
        (3, "BACKUP: GitHub backup repo setup; no memory text"),
    (_S + "infra/cloud_backup.py", "connect_github", "httpx.post"):
        (1, "BACKUP: GitHub backup repo setup; no memory text"),
    (_S + "infra/cloud_backup.py", "connect_github", "httpx.put"):
        (1, "BACKUP: GitHub backup repo README; no memory text"),
    (_S + "infra/cloud_backup.py", "connect_google_drive", "googleapiclient.discovery.build"):
        (1, "BACKUP: Google Drive backup setup; no memory text"),
    (_S + "infra/cloud_backup.py", "sync_to_github", "httpx.post"):
        (2, "BACKUP: uploads only client-side-encrypted bytes (release + assets)"),
    (_S + "infra/cloud_backup_github.py", "_cleanup_old_releases", "httpx.delete"):
        (2, "BACKUP: deletes old GitHub backup releases; no memory text"),
    (_S + "infra/cloud_backup_github.py", "_cleanup_old_releases", "httpx.get"):
        (1, "BACKUP: lists GitHub backup releases; no memory text"),
    (_S + "ingestion/calendar_adapter.py", "CalendarAdapter._fetch_oauth",
     "googleapiclient.discovery.build"):
        (1, "INBOUND: reads the user's calendar"),
    (_S + "ingestion/gmail_adapter.py", "GmailAdapter._fetch_imap", "imaplib.IMAP4_SSL"):
        (1, "INBOUND: reads the user's mailbox over IMAP"),
    (_S + "ingestion/gmail_adapter.py", "GmailAdapter._fetch_oauth",
     "googleapiclient.discovery.build"):
        (1, "INBOUND: reads the user's mailbox over the Gmail API"),
    (_S + "mesh/remote_sync.py", "RemoteSyncClient._http_client", "httpx.Client"):
        (2, "MESH: builds the shared-secret-authed client for peer-list sync, "
            "state/lock delta reads, AND mesh/send delivery. The reads carry "
            "no message content. The sends (send_to_remote, _drain_outbox) "
            "screen `content` through core.outbound_redaction.for_endpoint, "
            "keyed on the peer's resolved URL, before it is signed or handed "
            "to this client — loopback peers are exempt, same rule as the "
            "gate everywhere else."),
    (_S + "mesh/remote_sync.py", "_get_cert_sha256", "socket.create_connection"):
        (1, "MESH: reads the peer's TLS certificate for pinning; sends nothing"),
    (_S + "optimize/proxy/server.py", "ProxyApp.startup", "httpx.AsyncClient"):
        (1, "PROXY: forwards the agent's own LLM requests; never opens the store"),
    (_S + "server/routes/backup.py", "github_oauth_callback", "httpx.post"):
        (1, "OAUTH: exchanges the GitHub OAuth code"),
    (_S + "server/routes/config_api.py", "get_ollama_models", "httpx.get"):
        (1, "PROBE: Ollama model list for the settings page"),
    (_S + "server/routes/v3_api.py", "ollama_status", "httpx.Client"):
        (1, "PROBE: Ollama model list (GET /api/tags)"),
    (_S + "server/routes/v3_api.py", "test_embedding_endpoint", "httpx.Client"):
        (1, "PROBE: fixed string 'test embedding connection'"),
    (_S + "server/routes/v3_api.py", "test_provider", "httpx.Client"):
        (4, "PROBE: provider key test with the fixed message 'hi'"),
    (_S + "server/unified_daemon.py", "_register_dashboard_routes.v3_auto_detect",
     "httpx.get"):
        (1, "PROBE: is a local Ollama running (GET /api/tags)"),
    (_S + "remote_connections/gateway_provider.py", "CloudGatewayProvider._request",
     "httpx.AsyncClient"):
        (1, "OAUTH: owner enrollment calls to the fixed https://auth.superlocalmemory.com "
            "host (AUTH constant plus a fixed path allow-list, nothing caller-supplied). "
            "Bodies are OAuth client registration, PKCE code/refresh exchange and "
            "connection metadata; no memory text. Not sent through core.outbound_http "
            "because its credential screen would replace the refresh token and PKCE "
            "verifier with [redacted] and break the exchange. Instead the client is "
            "built with follow_redirects=False (a redirect is an error, so the owner "
            "token never reaches a second origin), trust_env=False (no environment "
            "proxy) and timeout=10; the response is capped at 64 KiB. "
            "tests/test_remote_gateway_provider.py pins all of this."),
    (_S + "remote_connections/origin.py", "CanonicalMcpOrigin.__call__", "httpx.AsyncClient"):
        (1, "RELAY: not a network call. httpx.ASGITransport hands the request to this "
            "daemon's own ASGI app in the same process (no socket, no DNS); the base_url "
            "is a virtual value for the Host header. Redirects are off, trust_env is off, "
            "and a 3xx is refused as origin_redirect_denied."),
    (_S + "server/legacy_port.py", "start_legacy_redirect._handle_client",
     "asyncio.open_connection"):
        (1, "RELAY: relays the legacy port to the daemon on 127.0.0.1"),
}


def _current_sites() -> Counter:
    sites: Counter = Counter()
    for root in _ROOTS:
        path = _REPO / root
        if path.exists():
            sites.update(scan_tree(path, _REPO))
    return Counter({site: n for site, n in sites.items() if site[0] != _GATE})


def test_no_module_opens_an_unreviewed_outbound_connection() -> None:
    sites = _current_sites()
    unreviewed = {
        site: n for site, n in sites.items()
        if site not in REVIEWED or n > REVIEWED[site][0]
    }
    assert not unreviewed, (
        "outbound connections outside core.outbound_http that are not reviewed "
        "(send them through the gate, or add a reviewed entry with its reason): "
        + "; ".join(f"{f} :: {q} :: {c} x{n}" for (f, q, c), n in sorted(unreviewed.items()))
    )


def test_the_reviewed_list_has_no_stale_entry() -> None:
    sites = _current_sites()
    stale = {site: n for site, (n, _r) in REVIEWED.items() if sites.get(site) != n}
    assert not stale, f"reviewed entries that no longer match the code: {sorted(stale)}"


def test_every_reviewed_entry_names_its_reason() -> None:
    for site, (count, reason) in REVIEWED.items():
        assert count >= 1, site
        assert reason.startswith(_CATEGORIES), f"{site}: {reason!r}"
        assert len(reason.split(":", 1)[1].strip()) > 10, site


def test_the_gate_itself_is_where_the_clients_are_built() -> None:
    gate = scan_tree(_REPO / "src/superlocalmemory/core", _REPO)
    calls = {c for (f, _q, c) in gate if f == _GATE}
    assert {"httpx.post", "httpx.Client", "httpx.AsyncClient",
            "urllib.request.build_opener"} <= calls


@pytest.mark.parametrize("source,expected", [
    ("import httpx\ndef leak(t):\n    httpx.post('https://x', json={'t': t})\n",
     ("leak", "httpx.post")),
    ("import httpx as h\ndef leak(t):\n    h.post('https://x', json={'t': t})\n",
     ("leak", "httpx.post")),
    ("from httpx import Client\nclass A:\n    def go(self):\n        Client().post('u')\n",
     ("A.go", "httpx.Client")),
    ("import urllib.request as u\ndef go(r):\n    u.urlopen(r)\n",
     ("go", "urllib.request.urlopen")),
    ("from urllib.request import urlopen\ndef go(r):\n    urlopen(r)\n",
     ("go", "urllib.request.urlopen")),
    ("import requests\ndef go(t):\n    requests.post('u', data=t)\n",
     ("go", "requests.post")),
    ("import subprocess\ndef go(t):\n    subprocess.run(['curl', '-d', t, 'https://x'])\n",
     ("go", "subprocess:curl")),
    ("def go():\n    import openai\n    openai.OpenAI()\n", ("go", "openai.OpenAI")),
])
def test_the_scanner_sees_through_aliases(source: str, expected: tuple[str, str]) -> None:
    assert scan_source(source, "m.py") == [("m.py", *expected)]


def test_a_new_raw_post_in_a_real_module_is_caught(tmp_path: Path) -> None:
    """A raw ``httpx.post`` added to a real, gated module fails the guard."""
    target = _REPO / _S / "core/summarizer.py"
    copy = tmp_path / "src/superlocalmemory/core/summarizer.py"
    copy.parent.mkdir(parents=True)
    copy.write_text(
        target.read_text(encoding="utf-8")
        + "\n\ndef _leak(text):\n    import httpx\n"
        "    return httpx.post('https://example.invalid', json={'t': text})\n",
        encoding="utf-8",
    )
    found = scan_tree(tmp_path / "src", tmp_path)
    site = ("src/superlocalmemory/core/summarizer.py", "_leak", "httpx.post")
    assert found[site] == 1
    assert site not in REVIEWED
