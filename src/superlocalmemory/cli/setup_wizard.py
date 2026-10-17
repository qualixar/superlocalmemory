# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com
"""Interactive setup wizard for first-time configuration.
Runs automatically on first use of any `slm` command, or via `slm setup`.
Downloads models, configures mode, verifies installation.
For npm: triggered by postinstall.js after dependency installation.
For pip: triggered on first `slm` command when .setup-complete is missing.
Part of Qualixar | Author: Varun Pratap Bhardwaj
"""
from __future__ import annotations
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# resolve via slm_home() so all 3 env aliases are honoured.
# Fallback keeps stdlib-only path if the import fails during early bootstrap.
def _resolve_slm_home() -> Path:
    from superlocalmemory.infra.data_root import canonical_data_root
    return canonical_data_root()
_EMBED_MODEL = "nomic-ai/nomic-embed-text-v1.5"
_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"
# v3.6.10: compulsory LLMLingua-2 prose compression model (~560MB, aggressive mode).
_COMPRESSOR_MODEL = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"
# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------
def is_interactive() -> bool:
    """True if running in a terminal (not CI, not piped, not MCP)."""
    if os.environ.get("CI"):
        return False
    if os.environ.get("SLM_NON_INTERACTIVE"):
        return False
    return sys.stdin.isatty() and sys.stdout.isatty()
def is_setup_complete() -> bool:
    """True if the setup wizard has been run at least once."""
    return (_resolve_slm_home() / ".setup-complete").exists()
def needs_setup() -> bool:
    """True if setup should auto-trigger (first use)."""
    return not is_setup_complete()
def _prompt(message: str, default: str = "") -> str:
    """Prompt user for input. Returns default if non-interactive."""
    if not is_interactive():
        return default
    try:
        return input(message).strip() or default
    except (EOFError, KeyboardInterrupt):
        print()
        return default
def _get_ram_gb() -> float:
    """Get total system RAM in GB (core.machine; kept here so tests can patch it)."""
    from superlocalmemory.core.machine import total_ram_gb
    return total_ram_gb()
def _ollama_installed_models() -> list[str] | None:
    """Names of the Ollama models actually installed on this machine.
    One bounded GET (``/api/tags``), 1.5s timeout — same call ``_ollama_available``
    used to make on its own. Returns ``None`` when Ollama is unreachable or the
    response cannot be parsed, so callers can fall back to the catalogue default
    instead of crashing the wizard.
    """
    try:
        import httpx
        resp = httpx.get("http://localhost:11434/api/tags", timeout=1.5)
        if resp.status_code != 200:
            return None
        return [m["name"] for m in resp.json().get("models", [])]
    except Exception:
        return None
def _ollama_available() -> bool:
    """True if a local Ollama server is reachable (or its binary is installed).
    v3.8.2: used to RECOMMEND Mode B at setup with a single keypress. Bounded
    and fail-safe — a slow/absent Ollama never blocks or slows the wizard.
    """
    if _ollama_installed_models() is not None:
        return True
    try:
        return shutil.which("ollama") is not None
    except Exception:
        return False
# ---------------------------------------------------------------------------
# Model download
# ---------------------------------------------------------------------------
#: First statement of every download child: exit when the process that started
#: it is gone. The daemon's self-heal starts these downloads (up to 10 minutes)
#: from a background thread; without this, stopping or killing the daemon left
#: the download running on its own, and on Windows nothing else reaps it.
#: Never a reason for the download itself to fail: if the watchdog cannot be
#: imported, the download still runs.
_EXIT_WITH_PARENT = (
    "import sys\n"
    "try:\n"
    "    from superlocalmemory.core.platform_utils import start_parent_watchdog\n"
    "    start_parent_watchdog()\n"
    "except Exception:\n"
    "    pass\n"
)
def _download_model(model_name: str, label: str) -> bool:
    """Download a HuggingFace model with visible progress.
    Runs in a subprocess so the main process never loads torch.
    stderr is inherited so the user sees download progress bars.
    Returns True on success.
    """
    print(f"\n  Downloading {label}: {model_name}")
    print(f"  (this may take a few minutes on first run)\n")
    # H-03: pass the model name as argv, never interpolated into executed
    # source, so a crafted model_name cannot become arbitrary Python.
    script = (
        _EXIT_WITH_PARENT
        + "from sentence_transformers import SentenceTransformer; "
        "m = SentenceTransformer(sys.argv[1], trust_remote_code=True); "
        "d = m.get_sentence_embedding_dimension(); "
        "print(f'OK dim={d}'); "
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", script, model_name],
            timeout=600,  # 10 min for large model downloads
            capture_output=False,  # Show download progress
            text=True,
            env={
                **os.environ,
                "CUDA_VISIBLE_DEVICES": "",
                "TOKENIZERS_PARALLELISM": "false",
                "TORCH_DEVICE": "cpu",
            },
        )
        if result.returncode == 0:
            print(f"  ✓ {label} ready")
            return True
        print(f"  ✗ {label} download failed (exit code {result.returncode})")
        return False
    except subprocess.TimeoutExpired:
        print(f"  ✗ {label} download timed out (10 min)")
        return False
    except FileNotFoundError:
        print(f"  ✗ Python not found: {sys.executable}")
        return False
    except Exception as exc:
        print(f"  ✗ {label} download error: {exc}")
        return False
def _download_reranker(model_name: str) -> bool:
    """Download cross-encoder reranker model."""
    print(f"\n  Downloading reranker: {model_name}")
    print(f"  (cross-encoder for result re-ranking)\n")
    # H-03: model name via argv, never interpolated into executed source.
    script = (
        _EXIT_WITH_PARENT
        + "from sentence_transformers import CrossEncoder; "
        "m = CrossEncoder(sys.argv[1], trust_remote_code=True); "
        "print('OK'); "
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", script, model_name],
            timeout=300,
            capture_output=False,
            text=True,
            env={
                **os.environ,
                "CUDA_VISIBLE_DEVICES": "",
                "TOKENIZERS_PARALLELISM": "false",
                "TORCH_DEVICE": "cpu",
            },
        )
        if result.returncode == 0:
            print(f"  ✓ Reranker ready")
            return True
        print(f"  ✗ Reranker download failed")
        return False
    except Exception as exc:
        print(f"  ✗ Reranker error: {exc}")
        return False
def _download_compressor(model_name: str) -> bool:
    """Download the LLMLingua-2 prose compression model (v3.6.10).
    Mirrors _download_reranker: a subprocess forces the HF download with visible
    progress. Fail-open — a network hiccup must NOT break setup; the model also
    lazy-downloads on first use in prose_llmlingua.py.
    """
    print(f"\n  Downloading compression model: {model_name}")
    print(f"  (LLMLingua-2 prose compressor, ~560MB — aggressive mode only)\n")
    # H-03: model name via argv, never interpolated into executed source.
    script = (
        "import sys; from llmlingua import PromptCompressor; "
        "PromptCompressor(model_name=sys.argv[1], use_llmlingua2=True, "
        "device_map='cpu'); "
        "print('OK')"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", script, model_name],
            timeout=900,  # 560MB on a slow link can exceed 5 min
            capture_output=False,
            text=True,
            env={
                **os.environ,
                "CUDA_VISIBLE_DEVICES": "",
                "TOKENIZERS_PARALLELISM": "false",
                "TORCH_DEVICE": "cpu",
            },
        )
        if result.returncode == 0:
            print(f"  ✓ Compression model ready")
            return True
        print(f"  ✗ Compression model download failed (will lazy-download on first use)")
        return False
    except ImportError:
        print(f"  ⚠ llmlingua not installed — compression model will download on first use")
        return False
    except Exception as exc:
        print(f"  ✗ Compression model error: {exc}")
        return False
# ---------------------------------------------------------------------------
# Laya — the local answer check (4.1.18)
# ---------------------------------------------------------------------------

def _laya_summary_line_applies(*, interactive: bool = True, no_laya: bool = False) -> bool:
    """True when the up-front download summary should mention Laya.
    Never raises and never imports anything heavier than the platform check
    it wraps — a broken optional import here must not break Step 3's summary.
    Since 4.1.25 unattended setup installs it too, unless opted out.
    """
    from superlocalmemory.cli import laya_offer

    return not laya_offer.opted_out(no_laya) and laya_offer.supported()


def _run_mode_b_server_step(config: Any, *, interactive: bool) -> None:
    """Step 2b: pick Mode B's local model server, then save (#112).
    Offers Ollama (the default — unchanged behavior) or another local
    OpenAI-compatible server (llama.cpp, vLLM, LM Studio, …), which the
    dashboard's Settings pane could already configure but the CLI wizard
    could not. The custom-server path prompts for the URL/model, applies
    the same endpoint-trust rules as the remote reranker, saves, and runs
    the same connection test ``slm provider set custom`` does — see
    ``cli.provider_custom_endpoint``.
    Non-interactive runs (``auto``/CI) always keep the Ollama default,
    matching the wizard's existing zero-prompt contract.
    """
    print()
    use_custom_server = False
    if interactive:
        print("  [1] Ollama (default)")
        print("  [2] Another local server (llama.cpp, vLLM, LM Studio, …)")
        server_choice = _prompt(
            "  Select local model server [1/2] (default: 1): ", "1",
        )
        use_custom_server = server_choice.strip() == "2"
    if use_custom_server:
        from superlocalmemory.cli.provider_custom_endpoint import (
            configure_custom_endpoint_provider,
        )
        configure_custom_endpoint_provider(
            config, endpoint=None, api_key=None, model=None,
            target_mode="b", interactive=True,
        )
        return
    if shutil.which("ollama"):
        print("  ✓ Ollama found!")
    else:
        print("  ⚠ Ollama not found. Install: https://ollama.ai")
        print("    After installing: ollama pull llama3.2")
    # #4.1.22: offer the models actually installed, best first, instead of
    # always wiring the catalogue default — logic lives in cli.model_choice
    # (setup_wizard.py is already past the 800-line guideline; see
    # .backup/4.1.22/LANE-RULES.md). ``_prompt`` is passed through BY NAME so
    # a test that monkeypatches ``sw._prompt`` is still honoured here.
    from superlocalmemory.cli.model_choice import pick_mode_b_model
    pick_mode_b_model(
        config, interactive=interactive, prompt=_prompt,
        installed=_ollama_installed_models(), ram_gb=_get_ram_gb(),
    )
    config.save(mode_change=True)


def _run_laya_step(config: Any, *, interactive: bool, no_laya: bool = False) -> None:
    """Step 4d: install the local answer check (Laya) — on by default (4.1.25).

    Interactive setup asks [Y/n]; unattended setup installs it unless opted
    out (``laya_offer.opted_out``). Never on hardware it cannot run on.
    Fails open on purpose: any problem here is one plain line, and setup
    keeps going. The same install is always offered again later in the
    dashboard (Settings -> Answer check), so nothing here is a last chance.
    """
    from superlocalmemory.cli import laya_offer

    print()
    print("─── Step 4d/10: Answer check (Laya) ───")
    print()

    if laya_offer.opted_out(no_laya):
        print(f"  ✓ Skipped ({laya_offer.SKIP_ENV} or --no-laya) — "
              "set it up later in the dashboard.")
        return
    try:
        if not laya_offer.supported():
            print("  The local answer check needs Apple Silicon. On this machine, set up")
            print("  the hosted option instead: Settings → Answer check, in the dashboard.")
            return
        print("  Lets your memory say \"I don't have that\" instead of guessing.")
        print("  About 1.1 GB, stays on this Mac.")
        print()
        if interactive:
            choice = _prompt("  Set it up now? [Y/n] (default: Y): ", "y").lower()
            if choice not in ("", "y", "yes"):
                laya_offer.remember_decline()
                print("  ✓ Skipped — set it up later in the dashboard: Settings → Answer check.")
                return
            laya_offer.forget_decline()
        else:
            print("  Setting it up (the default where it can run).")
            print(f"  {laya_offer.OPT_OUT_HINT}")

        from superlocalmemory.core.laya_runtime import install
        def _on_progress(fraction: float, step: str) -> None:
            print(f"    [{fraction * 100:5.1f}%] {step}")
        result = install(progress=_on_progress)
        if result.state != "ready":
            print(f"  ⚠ {result.error or 'The install did not finish.'}")
            print("  You can finish this later in the dashboard: Settings → Answer check.")
            return
        config.retrieval.sufficiency_judge = "laya"
        config.retrieval.sufficiency_python = result.python
        config.retrieval.sufficiency_hf_home = result.hf_home
        config.retrieval.sufficiency_model = result.model_path
        config.save()
        print("  ✓ Answer check installed and verified")
    except Exception as exc:  # noqa: BLE001 — this step must never abort setup.
        print(f"  ⚠ Answer check setup hit an unexpected error: {exc}")
        print("  You can finish this later in the dashboard: Settings → Answer check.")
# ---------------------------------------------------------------------------
# Embedding-provider detection (v3.7.6 #72)
# ---------------------------------------------------------------------------
def _embedding_is_remote(config: Any) -> bool:
    """True when embeddings come from a remote/OpenAI-compatible endpoint.
    In that case the local sentence-transformers download (768d nomic) is
    unnecessary and would only waste bandwidth/disk (#72). Detection is by
    provider name or the presence of a configured HTTP endpoint.
    """
    emb = getattr(config, "embedding", None)
    if emb is None:
        return False
    provider = (getattr(emb, "provider", "") or "").strip().lower()
    endpoint = (
        getattr(emb, "api_endpoint", "")
        or getattr(emb, "base_url", "")
        or ""
    )
    return provider in ("openai", "openai-compatible", "remote") or bool(endpoint)
def _reranker_is_remote(config: Any) -> bool:
    """True when reranking runs against a remote endpoint (v3.8.12, #105).
    In that case the local ~130MB English cross-encoder download is pointless
    — and worse, misleading, since it is not the model that will score recall.
    Unlike embeddings, the endpoint alone is NOT sufficient: a stray endpoint
    against a local backend is a misconfiguration the engine reports, not a
    remote setup.
    """
    rt = getattr(config, "retrieval", None)
    if rt is None:
        return False
    backend = (getattr(rt, "cross_encoder_backend", "") or "").strip().lower()
    endpoint = getattr(rt, "cross_encoder_endpoint", "") or ""
    return backend in ("openai", "remote") and bool(endpoint)
# Remote-reranker keys the wizard never asks about. ``retrieval`` is otherwise
# a mode-owned block that the wizard resets from the template, which would
# silently delete a working remote endpoint on every re-run (#105).
_USER_OWNED_RETRIEVAL_KEYS = (
    "cross_encoder_backend",
    "cross_encoder_endpoint",
    "cross_encoder_api_key",
    "cross_encoder_model",
    "cross_encoder_timeout_seconds",
)
#: Every answer-check setting (which check, the hosted check's consents and
#: provider, where the on-device install lives) is the person's choice, not a
#: mode preset: re-running setup must never reset it. By prefix, so a setting
#: added later is kept too.
_USER_OWNED_RETRIEVAL_PREFIXES = ("sufficiency_",)
def _user_owned_retrieval_keys(retrieval) -> tuple[str, ...]:
    from dataclasses import fields, is_dataclass
    names = [f.name for f in fields(retrieval)] if is_dataclass(retrieval) else []
    prefixed = [n for n in names if n.startswith(_USER_OWNED_RETRIEVAL_PREFIXES)]
    explicit = [k for k in _USER_OWNED_RETRIEVAL_KEYS if hasattr(retrieval, k)]
    return tuple(dict.fromkeys(explicit + prefixed))
def _build_wizard_config(mode):
    """Apply mode-owned presets without erasing existing user-owned blocks."""
    from superlocalmemory.core.config import SLMConfig
    from superlocalmemory.infra.data_root import state_path
    template = SLMConfig.for_mode(mode)
    if not state_path("config.json").exists():
        return template
    existing = SLMConfig.load()
    existing.mode = mode
    existing.llm = template.llm
    preserved = {
        key: getattr(existing.retrieval, key)
        for key in _user_owned_retrieval_keys(existing.retrieval)
    }
    existing.retrieval = template.retrieval
    for key, value in preserved.items():
        setattr(existing.retrieval, key, value)
    existing.math = template.math
    existing.channel_weights = template.channel_weights
    return existing
# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
def _verify_installation() -> bool:
    """Quick smoke test: embed a sentence, verify dimension."""
    print("\n  Running verification test...")
    script = (
        "from superlocalmemory.core.embeddings import EmbeddingService; "
        "from superlocalmemory.core.config import EmbeddingConfig; "
        "cfg = EmbeddingConfig(); "
        "svc = EmbeddingService(cfg); "
        "vec = svc.embed('SuperLocalMemory setup verification test'); "
        "print(f'OK dim={len(vec)}' if vec else 'FAIL'); "
        "svc.unload(); "
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            timeout=120,
            capture_output=True,
            text=True,
            env={
                **os.environ,
                "CUDA_VISIBLE_DEVICES": "",
                "TOKENIZERS_PARALLELISM": "false",
                "TORCH_DEVICE": "cpu",
            },
        )
        stdout = result.stdout.strip()
        if "OK dim=" in stdout:
            dim = stdout.split("dim=")[1]
            print(f"  ✓ Embedding verified (dimension={dim})")
            return True
        print(f"  ✗ Verification failed: {stdout}")
        if result.stderr:
            # Show last 3 lines of stderr for diagnosis
            lines = result.stderr.strip().split("\n")
            for line in lines[-3:]:
                print(f"    {line}")
        return False
    except subprocess.TimeoutExpired:
        print("  ✗ Verification timed out (120s)")
        return False
    except Exception as exc:
        print(f"  ✗ Verification error: {exc}")
        return False
# ---------------------------------------------------------------------------
# Mark setup complete
# ---------------------------------------------------------------------------
def _mark_complete() -> None:
    """Write .setup-complete marker file."""
    slm_root = _resolve_slm_home()
    slm_root.mkdir(parents=True, exist_ok=True)
    (slm_root / ".setup-complete").write_text(
        f"setup_completed={time.strftime('%Y-%m-%dT%H:%M:%S')}\n"
        f"python={sys.executable}\n"
        f"platform={platform.system()}\n"
        f"version={platform.python_version()}\n",
        encoding="utf-8",
    )
# ---------------------------------------------------------------------------
# Main wizard
# ---------------------------------------------------------------------------
def run_wizard(auto: bool = False, no_laya: bool = False) -> None:
    """Run the supported wizard phases with this module's replaceable helpers.

    ``no_laya``: never install the answer check (``slm setup --no-laya``).
    """
    from superlocalmemory.cli.setup_wizard_flow import run
    run(auto, sys.modules[__name__], no_laya=no_laya)

# ---------------------------------------------------------------------------
# First-use auto-trigger
# ---------------------------------------------------------------------------
def check_first_use(command: str) -> None:
    """Check if setup is needed before running a command.
    Called from main.py before dispatching any command.
    Skips for commands that don't need setup (setup, hook, --version, --help).
    First use may initialize SLM-owned defaults, but it never edits an IDE,
    installs a plugin, or creates an operating-system service. Those external
    mutations require explicit consent inside ``slm setup``.
    """
    # Commands that work without setup
    _SKIP_COMMANDS = {
        "setup", "init", "hook", "hooks", "reap", "mcp", "diagnostics",
        "backup",  # restoring on a new computer must not wait for setup
        "remote",  # keys and certificates are files; no setup needed
    }
    if command in _SKIP_COMMANDS:
        return
    if is_setup_complete():
        return
    # Non-interactive: use defaults silently, don't block the command.
    # CRIT-1: only save config when it does NOT already exist — lazy-init may
    # have already written a valid config.json; overwriting it here would clobber
    # any lazy-init content (e.g. a pre-existing mode-A skeleton).
    if not is_interactive():
        try:
            from superlocalmemory.core.config import SLMConfig
            from superlocalmemory.storage.models import Mode
            config_path = _resolve_slm_home() / "config.json"
            if not config_path.exists():
                cfg = SLMConfig.for_mode(Mode.A)
                cfg.save(mode_change=True)
            _mark_complete()
        except Exception:
            pass
        return
    # Interactive: run the full wizard
    print()
    print("  First time using SuperLocalMemory!")
    print("  Running setup wizard...\n")
    run_wizard()
def _configure_external_integrations(*, interactive: bool) -> bool:
    """Request consent before editing Claude Code configuration.
    v3.8.2: after the Claude Code step, a pip user can also wire every OTHER
    detected IDE in one step (previously only Claude Code was offered here;
    other IDEs required a manual `slm connect <ide>` each).
    """
    print()
    print("  Optional Claude Code integration can install the SLM plugin and hooks.")
    if not interactive:
        print("  Skipped in non-interactive setup. Run: slm connect claude-code")
        return False
    changed = False
    choice = _prompt(
        "  Install Claude Code plugin and hooks now? [y/N] (default: N): ",
        "n",
    ).lower()
    if choice in ("y", "yes"):
        changed = _install_external_integrations()
    else:
        print("  Skipped. Run `slm connect claude-code` when ready.")
    # v3.8.2: offer to connect all other detected IDEs (Cursor, VS Code, …).
    changed = _configure_other_ides(interactive=interactive) or changed
    return changed
def _configure_other_ides(*, interactive: bool) -> bool:
    """Consent-gated: connect all OTHER detected IDEs (Cursor, VS Code, …).
    Uses the same IDEConnector the dashboard uses. Best-effort — a failure to
    detect or connect any single IDE never blocks setup.
    """
    if not interactive:
        return False
    try:
        from superlocalmemory.hooks.ide_connector import IDEConnector
        connector = IDEConnector()
        installed = [s for s in connector.get_status() if s.get("installed")]
    except Exception:
        return False
    others = [s for s in installed if s.get("id") not in ("claude-code", "claude")]
    if not others:
        return False
    names = ", ".join(s.get("name", s.get("id")) for s in others)
    print()
    print(f"  Detected other IDEs: {names}")
    print("  SLM can wire them all so memory works across every editor.")
    choice = _prompt("  Connect them now? [Y/n] (default: Y): ", "y").lower()
    if choice not in ("", "y", "yes"):
        print("  Skipped. Connect any later with: slm connect <ide>")
        return False
    changed = False
    for s in others:
        ide_id = s.get("id")
        label = s.get("name", ide_id)
        try:
            if connector.connect(ide_id):
                print(f"    ✓ {label}")
                changed = True
            else:
                print(f"    ⚠ {label} — connect later: slm connect {ide_id}")
        except Exception:
            print(f"    ⚠ {label} — connect later: slm connect {ide_id}")
    return changed
def _install_external_integrations() -> bool:
    """Install Claude Code hooks/plugin after explicit setup consent.
    Installation rules:
      * Skip if the user explicitly opted out via ``slm hooks remove``
      * Skip hook configuration if Claude Code has no settings file.
      * Best-effort: setup remains usable when Claude Code is unavailable.
    """
    changed = False
    try:
        opt_out = _resolve_slm_home() / "hooks" / ".hooks-disabled"
        claude_settings = Path.home() / ".claude" / "settings.json"
        if not opt_out.exists() and claude_settings.exists():
            from superlocalmemory.hooks.claude_code_hooks import install_hooks
            changed = bool(install_hooks()) or changed
    except Exception:
        # Best-effort: never block setup over an optional integration.
        pass
    return _try_install_claude_plugin() or changed
def _try_install_claude_plugin() -> bool:
    """Install the Claude Code plugin after explicit setup consent.
    Runs ``claude plugin marketplace add qualixar/superlocalmemory`` then
    ``claude plugin install superlocalmemory@qualixar`` if the ``claude``
    CLI is found in PATH. Silent on any error — plugin install is a
    convenience, not a hard requirement for SLM to function.
    """
    import shutil
    import subprocess
    claude = shutil.which("claude")
    if not claude:
        print("  Claude Code CLI not found; plugin installation skipped.")
        return False
    _run = lambda cmd: subprocess.run(  # noqa: E731
        cmd, capture_output=True, timeout=30, check=False
    )
    try:
        marketplace = _run(
            [claude, "plugin", "marketplace", "add", "qualixar/superlocalmemory"]
        )
        installed = _run(
            [claude, "plugin", "install", "superlocalmemory@qualixar"]
        )
        return marketplace.returncode == 0 and installed.returncode == 0
    except Exception:
        return False
def _configure_autostart(*, interactive: bool) -> bool:
    """Request consent before creating a login-time service definition."""
    print()
    print("  Optional auto-start keeps the SLM daemon available after login.")
    if not interactive:
        print("  Skipped in non-interactive setup. Run: slm serve install")
        return False
    choice = _prompt(
        "  Install the user-level auto-start service now? [y/N] (default: N): ",
        "n",
    ).lower()
    if choice not in ("y", "yes"):
        print("  Skipped. Run `slm serve install` when ready.")
        return False
    return _install_autostart_service()
def _install_autostart_service() -> bool:
    """Install the user service after explicit setup consent."""
    try:
        from superlocalmemory.cli.service_installer import install_service
        installed = bool(install_service())
        if installed:
            print("  ✓ User-level SLM auto-start service installed.")
        else:
            print("  ⚠ Service was not installed (run: slm serve install)")
        return installed
    except Exception:
        print("  ⚠ Service was not installed (run: slm serve install)")
        return False
# ---------------------------------------------------------------------------
# Mode C provider config (preserved from original)
# ---------------------------------------------------------------------------
def _named_provider_mode(provider_name: str, target_mode: str | None):
    """The mode ``slm provider set <provider>`` saves: ``--mode`` when given,
    else B for a local model (Ollama) and C for a hosted one.

    4.2.9: named providers were always saved as Mode C and ``--mode`` was read
    only for ``custom``, so ``slm provider set ollama --mode b`` left Mode C.
    """
    from superlocalmemory.storage.models import Mode

    choice = (target_mode or ("b" if provider_name == "ollama" else "c")).strip().lower()
    if choice not in ("b", "c"):
        raise ValueError(f"--mode must be b or c for an LLM provider, not {target_mode!r}")
    return Mode.B if choice == "b" else Mode.C


def configure_provider(
    config: object,
    provider_name: str | None = None,
    *,
    endpoint: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    target_mode: str | None = None,
) -> None:
    """Configure an LLM provider for Mode B or Mode C.
    When ``provider_name`` is supplied by ``slm provider set <provider>``,
    configuration is non-interactive and resolves its credential from the
    provider's documented environment variable. Omitting it preserves the
    existing interactive picker.
    ``provider_name="custom"`` (#112) configures your own OpenAI-compatible
    endpoint instead of a named preset — llama.cpp, vLLM, LM Studio, or any
    other self-hosted server, with no key required. ``endpoint``/``api_key``/
    ``model``/``target_mode`` are only consulted for that case.
    """
    from superlocalmemory.core.config import SLMConfig
    from superlocalmemory.storage.models import Mode
    presets = SLMConfig.provider_presets()
    providers = list(presets.keys())
    interactive_selection = provider_name is None
    if interactive_selection:
        print()
        print("  Choose your LLM provider:")
        print()
        for i, name in enumerate(providers, 1):
            preset = presets[name]
            print(f"    [{i}] {name.capitalize()} — {preset['model']}")
        print(
            f"    [{len(providers) + 1}] Custom endpoint — your own "
            f"OpenAI-compatible server (llama.cpp, vLLM, LM Studio, …); "
            f"no key needed"
        )
        print()
        idx = _prompt(f"  Select provider [1-{len(providers) + 1}]: ", "1")
        try:
            choice_idx = int(idx)
            if choice_idx == len(providers) + 1:
                provider_name = "custom"
            else:
                provider_name = providers[choice_idx - 1]
        except (ValueError, IndexError):
            print("  Invalid choice. Using OpenAI.")
            provider_name = "openai"
    elif provider_name != "custom" and provider_name not in presets:
        raise ValueError(f"Unsupported provider: {provider_name}")
    if provider_name == "custom":
        from superlocalmemory.cli.provider_custom_endpoint import (
            configure_custom_endpoint_provider,
        )
        configure_custom_endpoint_provider(
            config,
            endpoint=endpoint,
            api_key=api_key,
            model=model,
            target_mode=target_mode,
            interactive=interactive_selection,
        )
        return
    preset = presets[provider_name]
    # Resolve API key
    env_key = preset.get("env_key", "")
    api_key = ""
    if env_key:
        existing = os.environ.get(env_key, "")
        if existing:
            print(f"  Found {env_key} in environment.")
            api_key = existing
        elif interactive_selection and is_interactive():
            api_key = _prompt(
                f"  Enter your {provider_name.capitalize()} API key: ",
            )
    # Offer the model instead of silently taking the preset's (#4.1.22):
    # OpenRouter users see SLM's hosted catalogue so a cheaper or stronger
    # tier doesn't require already knowing its id; direct presets show the
    # preset model as the default. Non-interactive runs (``slm provider set``,
    # CI) keep the preset model — unchanged zero-prompt behavior. Logic lives
    # in cli.model_choice (this file is already past the 800-line guideline).
    chosen_model = (model or "").strip() or preset["model"]
    if interactive_selection and is_interactive():
        from superlocalmemory.cli.model_choice import pick_mode_c_model
        chosen_model = pick_mode_c_model(
            provider_name=provider_name, default_model=chosen_model, prompt=_prompt,
        )
    # Provider selection is an additive configuration operation.  Rebuilding
    # via ``for_mode`` used to reset retrieval, scale-engine, evolution, and
    # user-tuned embedding settings — surprising and unsafe after a user had
    # configured a local or promoted Cozo/Lance deployment.  Keep every
    # unrelated setting intact and change only the mode/provider contract.
    from superlocalmemory.core.config import LLMConfig
    mode = _named_provider_mode(provider_name, target_mode)
    updated = config if isinstance(config, SLMConfig) else SLMConfig.load()
    updated.mode = mode
    updated.llm = LLMConfig(
        provider=provider_name,
        model=chosen_model,
        api_key=api_key,
        api_base=preset["base_url"],
    )
    updated.save(mode_change=True)
    # 4.2.9: the per-mode file too, or the next `slm mode` switch restores the
    # model this command just replaced.
    updated.save(SLMConfig._mode_config_path(updated.base_dir, mode))
    SLMConfig.write_current_mode(mode.value, updated.base_dir)
    print(f"  Mode: {mode.value.upper()}")
    print(f"  Provider: {provider_name}")
    print(f"  Model: {chosen_model}")
