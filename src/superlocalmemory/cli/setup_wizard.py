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

# WP-07: resolve via slm_home() so all 3 env aliases are honoured.
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
    """Get total system RAM in GB."""
    try:
        import psutil
        return psutil.virtual_memory().total / (1024 ** 3)
    except ImportError:
        pass
    # Fallback: macOS
    if platform.system() == "Darwin":
        try:
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
            return int(out.strip()) / (1024 ** 3)
        except Exception:
            pass
    # Fallback: Linux
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) / (1024 ** 2)
    except Exception:
        pass
    return 0.0


def _ollama_available() -> bool:
    """True if a local Ollama server is reachable (or its binary is installed).

    v3.8.2: used to RECOMMEND Mode B at setup with a single keypress. Bounded
    and fail-safe — a slow/absent Ollama never blocks or slows the wizard.
    """
    try:
        import httpx

        if httpx.get("http://localhost:11434/api/tags", timeout=1.5).status_code == 200:
            return True
    except Exception:
        pass
    try:
        return shutil.which("ollama") is not None
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Model download
# ---------------------------------------------------------------------------

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
        "import sys; "
        "from sentence_transformers import SentenceTransformer; "
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
        "import sys; from sentence_transformers import CrossEncoder; "
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
    existing.retrieval = template.retrieval
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
        f"version={platform.python_version()}\n"
    )


# ---------------------------------------------------------------------------
# Main wizard
# ---------------------------------------------------------------------------

def run_wizard(auto: bool = False) -> None:
    """Run the interactive setup wizard.

    Args:
        auto: If True, use defaults without prompting (for npm postinstall
              or CI environments).
    """
    interactive = is_interactive() and not auto
    slm_root = _resolve_slm_home()

    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  SuperLocalMemory V3 — The Unified Brain               ║")
    print("║  by Varun Pratap Bhardwaj / Qualixar                   ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    # -- Step 1: System check --
    print("─── Step 1/10: System Check ───")
    print()
    py_ver = platform.python_version()
    py_ok = sys.version_info >= (3, 11)
    ram_gb = _get_ram_gb()
    print(f"  Python:   {py_ver} {'✓' if py_ok else '✗ (3.11+ required)'}")
    print(f"  Platform: {platform.system()} {platform.machine()}")
    if ram_gb > 0:
        print(f"  RAM:      {ram_gb:.1f} GB {'✓' if ram_gb >= 4 else '⚠ (4GB+ recommended)'}")
    print(f"  Data dir: {slm_root}")

    # Check sentence-transformers
    st_ok = False
    try:
        import sentence_transformers  # noqa: F401
        st_ok = True
        print(f"  sentence-transformers: ✓")
    except ImportError:
        print(f"  sentence-transformers: ✗ (not installed)")
        print(f"    Run: pip install 'sentence-transformers>=4.0.0'")

    if not py_ok:
        print("\n  ✗ Python 3.11+ is required. Please upgrade Python.")
        print("    https://python.org/downloads/")
        return

    # -- Step 2: Mode selection --
    # v3.8.2: pre-detect Ollama so we can recommend Mode B with one keypress.
    # No detection → zero-config Mode A. (Non-interactive always stays A: Mode B
    # needs a pulled model, so we never assume it without the user confirming.)
    ollama_present = _ollama_available()
    default_mode = "b" if ollama_present else "a"
    print()
    print("─── Step 2/10: Choose Operating Mode ───")
    print()
    if ollama_present:
        print("  ✓ Ollama detected on this machine — Mode B recommended.")
        print("    Press Enter to accept, or pick another mode.")
        print()
    print("  [A] Local Guardian" + ("" if ollama_present else " (recommended)"))
    print("      No model-provider call in the core memory path.")
    print("      Review optional integrations and network policy for your deployment.")
    print()
    print("  [B] Smart Local" + (" (recommended — Ollama detected)" if ollama_present else ""))
    print("      Local LLM via Ollama for enrichment.")
    print("      Data stays on your machine.")
    print()
    print("  [C] Full Power")
    print("      Cloud LLM for maximum accuracy.")
    print("      Requires API key.")
    print()

    if interactive:
        choice = _prompt(
            f"  Select mode [A/B/C] (default: {default_mode.upper()}): ",
            default_mode,
        ).lower()
    else:
        choice = "a"
        print("  Auto-selecting Mode A (non-interactive, zero-config)")

    if choice not in ("a", "b", "c"):
        print(f"  Invalid choice '{choice}', using Mode A.")
        choice = "a"

    from superlocalmemory.core.config import SLMConfig
    from superlocalmemory.storage.models import Mode

    mode_map = {"a": Mode.A, "b": Mode.B, "c": Mode.C}
    config = _build_wizard_config(mode_map[choice])

    # -- Multi-scope (shared memory) opt-in — v3.6.15 --
    # OFF by default: your memories stay private to this profile (3.6.14 behaviour).
    # Enabling only turns ON recall VISIBILITY of other profiles' shared/global facts;
    # your own writes still default to 'personal' (mark a memory shared/global per call
    # with --scope). Existing users can flip this later in the `scope` section of
    # mode_a/b/c.json. See docs/shared-memory.md.
    print()
    print("  Shared memory lets other local profiles' 'shared'/'global' memories")
    print("  appear in your recall. It is OFF by default — your memories stay private.")
    if interactive:
        sm_choice = _prompt(
            "  See shared/global memories from other profiles by default? [y/N] (default: N): ",
            "n",
        ).lower()
    else:
        sm_choice = "n"
    if sm_choice in ("y", "yes"):
        from superlocalmemory.core.config import ScopeConfig
        config.scope = ScopeConfig(
            default_scope="personal",
            recall_include_global=True,
            recall_include_shared=True,
        )
        print("  ✓ Shared-memory recall ENABLED (your own writes still default to personal).")
    else:
        print("  ✓ Shared memory OFF (default) — enable later in mode_*.json if needed.")

    if choice == "b":
        print()
        if shutil.which("ollama"):
            print("  ✓ Ollama found!")
        else:
            print("  ⚠ Ollama not found. Install: https://ollama.ai")
            print("    After installing: ollama pull llama3.2")

    if choice == "c" and interactive:
        configure_provider(config)
    else:
        config.save(mode_change=True)

    mode_names = {"a": "Local Guardian", "b": "Smart Local", "c": "Full Power"}
    print(f"\n  ✓ Mode {choice.upper()} ({mode_names[choice]}) configured")

    # -- Step 3: Code Knowledge Graph --
    print()
    print("─── Step 3/10: Code Knowledge Graph ───")
    print()
    print("  CodeGraph builds a structural map of your codebase using Tree-sitter.")
    print("  It gives your AI assistant blast-radius analysis, call graphs,")
    print("  and connects code structure to your session memories.")
    print()
    print("  [Y] Enable CodeGraph (recommended for developers)")
    print("  [N] Disable CodeGraph (can enable later via config)")
    print()

    if interactive:
        cg_choice = _prompt("  Enable Code Knowledge Graph? [Y/n] (default: Y): ", "y").lower()
    else:
        cg_choice = "y"
        print("  Auto-enabling CodeGraph (non-interactive)")

    code_graph_enabled = cg_choice in ("", "y", "yes")

    # Write code graph config
    slm_root.mkdir(parents=True, exist_ok=True)
    cg_config_path = slm_root / "code_graph_config.json"
    import json
    cg_config_data = {"enabled": code_graph_enabled, "bridge_enabled": code_graph_enabled}
    cg_config_path.write_text(json.dumps(cg_config_data, indent=2))

    if code_graph_enabled:
        print(f"\n  ✓ CodeGraph enabled")
        print(f"    Run `slm code-graph build` in any repo to index it")
    else:
        print(f"\n  ✓ CodeGraph disabled (enable later in {cg_config_path})")

    # v3.8.2: up-front download summary so a large pull is never a surprise on
    # a metered/slow link. Sizes approximate, one-time, cached locally.
    if st_ok and not _embedding_is_remote(config):
        print()
        print("  Models to download (one-time, cached locally):")
        print("    • Embedding model   ~500 MB  — required for semantic recall")
        print("    • Reranker model    ~130 MB  — result quality")
        print("    • Compression model ~560 MB  — optional (you'll be asked)")

    # -- Step 4: Download models --
    print()
    # H-06 (CVE-2025-14926): a malicious HuggingFace checkpoint can execute code
    # at load time. SLM only downloads its pinned defaults, but warn users who
    # point config at a custom model.
    print("  ⚠ Only install models from sources you trust — a malicious model")
    print("    checkpoint can run code on your machine at load time. See SECURITY.md.")
    print("─── Step 4/10: Download Embedding Model ───")

    if _embedding_is_remote(config):
        # v3.7.6 (#72): a remote/OpenAI-compatible embedding endpoint supplies
        # its own vectors (any width, e.g. 1024d Qwen3-Embedding), so pulling the
        # local 768d nomic model would be wasted bandwidth and could imply a
        # dimension that does not match the endpoint.
        emb = config.embedding
        print("  ✓ Skipped — remote/OpenAI-compatible embedding endpoint configured")
        print(f"    provider={getattr(emb, 'provider', '?')}, "
              f"dimension={getattr(emb, 'dimension', '?')}")
        print("    No local embedding model needed.")
    elif not st_ok:
        print("  ⚠ Skipped (sentence-transformers not installed)")
        print("    Models will download on first use.")
    else:
        embed_ok = _download_model(_EMBED_MODEL, "Embedding model")
        if not embed_ok:
            print("  ⚠ Model will download on first use (may take a few minutes)")

    print()
    print("─── Step 4b/10: Download Reranker Model ───")

    if not st_ok:
        print("  ⚠ Skipped (sentence-transformers not installed)")
    else:
        _download_reranker(_RERANKER_MODEL)

    print()
    print("─── Step 4c/10: Compression Model (LLMLingua-2, ~560MB) ───")
    print("  Only used for aggressive prompt compression. It also downloads")
    print("  automatically the first time you enable that feature — so skipping")
    print("  here is safe.")
    # v3.8.2: consent-gate the 560MB compressor (was unconditional). Skip by
    # default; it lazy-downloads on first real use of compression.
    if not st_ok:
        print("  ⚠ Skipped (sentence-transformers not installed)")
    elif interactive:
        comp_choice = _prompt(
            "  Download the ~560MB compression model now? [y/N] (default: N): ",
            "n",
        ).lower()
        if comp_choice in ("y", "yes"):
            _download_compressor(_COMPRESSOR_MODEL)
        else:
            print("  ✓ Skipped — downloads automatically when you enable compression.")
    else:
        print("  ✓ Skipped (non-interactive) — downloads on first use.")

    # -- Step 5: Daemon Configuration (v3.4.3) --
    print()
    print("─── Step 5/10: Daemon Configuration ───")
    print()
    print("  The SLM daemon runs in the background for instant memory access.")
    print()
    print("  [1] 24/7 Always-On (recommended — brain never sleeps)")
    print("  [2] Auto-shutdown after idle (saves RAM when not coding)")
    print()

    if interactive:
        daemon_choice = _prompt("  Select daemon mode [1/2] (default: 1): ", "1")
    else:
        daemon_choice = "1"
        print("  Auto-selecting 24/7 mode (non-interactive)")

    if daemon_choice == "2":
        if interactive:
            timeout_choice = _prompt("  Idle timeout [30m/1h/2h] (default: 30m): ", "30m")
        else:
            timeout_choice = "30m"
        timeout_map = {"30m": 1800, "1h": 3600, "2h": 7200}
        config.daemon_idle_timeout = timeout_map.get(timeout_choice, 1800)
        print(f"\n  ✓ Auto-shutdown after {timeout_choice} idle")
    else:
        config.daemon_idle_timeout = 0
        print("\n  ✓ 24/7 Always-On mode")

    config.save(mode_change=True)

    # -- Step 6: Mesh Communication (v3.4.3) --
    print()
    print("─── Step 6/10: Mesh Communication ───")
    print()
    print("  SLM Mesh enables agent-to-agent P2P communication.")
    print("  Multiple AI sessions can share knowledge in real-time.")
    print()
    print("  [Y] Enable Mesh (recommended)")
    print("  [N] Disable Mesh")
    print()

    if interactive:
        mesh_choice = _prompt("  Enable Mesh? [Y/n] (default: Y): ", "y").lower()
    else:
        mesh_choice = "y"
        print("  Auto-enabling Mesh (non-interactive)")

    config.mesh_enabled = mesh_choice in ("", "y", "yes")
    config.save(mode_change=True)
    print(f"\n  ✓ Mesh {'enabled' if config.mesh_enabled else 'disabled'}")

    # -- Step 7: Ingestion Adapters (v3.4.3) --
    print()
    print("─── Step 7/10: Ingestion Adapters ───")
    print()
    print("  These let SLM learn from your email, calendar, and meetings.")
    print("  All adapters are OFF by default. You can enable them later.")
    print()
    print("  Available adapters:")
    print("    • Gmail Ingestion     — requires Google OAuth setup")
    print("    • Google Calendar     — shares Gmail credentials")
    print("    • Meeting Transcripts — watches a folder for .srt/.vtt files")
    print()

    if interactive:
        adapter_input = _prompt("  Enable any now? [Enter to skip, or type: gmail,calendar,transcript]: ", "")
    else:
        adapter_input = ""

    # Save adapter preferences (actual setup happens via `slm adapters enable X`)
    adapters_config = {"gmail": False, "calendar": False, "transcript": False}
    if adapter_input:
        for name in adapter_input.split(","):
            name = name.strip().lower()
            if name in adapters_config:
                adapters_config[name] = True

    adapters_path = slm_root / "adapters.json"
    import json as _json
    adapters_path.write_text(_json.dumps(
        {k: {"enabled": v, "tier": "polling"} for k, v in adapters_config.items()},
        indent=2,
    ))

    enabled_adapters = [k for k, v in adapters_config.items() if v]
    if enabled_adapters:
        print(f"\n  ✓ Enabled: {', '.join(enabled_adapters)}")
        print("    Run `slm adapters start <name>` to begin ingestion")
    else:
        print("\n  ✓ All adapters disabled (enable later: slm adapters enable gmail)")

    # -- Step 8: Entity Compilation (v3.4.3) --
    print()
    print("─── Step 8/10: Entity Compilation ───")
    print()
    print("  Entity compilation builds knowledge summaries per person,")
    print("  project, and concept. Runs automatically during consolidation.")
    print()
    print("  [Y] Enable entity compilation (recommended)")
    print("  [N] Disable")
    print()

    if interactive:
        ec_choice = _prompt("  Enable entity compilation? [Y/n] (default: Y): ", "y").lower()
    else:
        ec_choice = "y"
        print("  Auto-enabling entity compilation (non-interactive)")

    config.entity_compilation_enabled = ec_choice in ("", "y", "yes")
    config.save(mode_change=True)
    print(f"\n  ✓ Entity compilation {'enabled' if config.entity_compilation_enabled else 'disabled'}")

    # -- Step 9: Skill Evolution (v3.4.11) --
    print()
    print("─── Step 9/10: Skill Evolution ───")
    print()
    print("  SLM can automatically evolve skills that underperform.")
    print("  It detects degradation, generates improvements, and verifies them blindly.")
    print("  Requires an LLM backend (Claude CLI, Ollama, or API key).")
    print()
    print("  OFF by default — when enabled it makes background LLM calls.")
    print("  [Y] Enable Skill Evolution")
    print("  [N] Keep disabled (enable later: slm config set evolution.enabled true)")
    print()

    # v3.8.2: default OFF (was Y) — consistent with the npm profiles and no
    # surprise outbound LLM calls for a fresh user. Opt-in, not opt-out.
    if interactive:
        evo_choice = _prompt("  Enable Skill Evolution? [y/N] (default: N): ", "n").lower()
    else:
        evo_choice = "n"
        print("  Skill Evolution stays OFF (non-interactive) — enable later with:")
        print("    slm config set evolution.enabled true")

    evolution_enabled = evo_choice in ("y", "yes")

    # Write evolution config to config.json directly
    # (SLMConfig.save() doesn't serialize evolution)
    slm_root.mkdir(parents=True, exist_ok=True)
    evo_config_path = slm_root / "config.json"
    evo_cfg: dict = {}
    if evo_config_path.exists():
        try:
            evo_cfg = json.loads(evo_config_path.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    evo_cfg["evolution"] = {
        "enabled": evolution_enabled,
        "backend": "auto",
    }
    evo_config_path.write_text(json.dumps(evo_cfg, indent=2) + "\n")

    if evolution_enabled:
        print(f"\n  ✓ Skill Evolution enabled (backend: auto-detect)")
    else:
        print(f"\n  ✓ Skill Evolution disabled")

    # -- Step 10: Verification --
    print()
    print("─── Step 10/10: Verification ───")

    if st_ok:
        verified = _verify_installation()
    else:
        print("  ⚠ Skipped (sentence-transformers not installed)")
        verified = False

    # v3.4.26 options — data-dir safety + queue toggle. One prompt max.
    try:
        from superlocalmemory.cli.wizard_v3426_options import (
            persist_v3426_options,
            prompt_v3426_options,
            validate_install_data_dir,
        )
        ok, reason = validate_install_data_dir(slm_root)
        if not ok:
            print()
            print("  ⚠ Data directory check failed:")
            for line in reason.splitlines():
                print(f"    {line}")
            print()
        v3426_opts = prompt_v3426_options(interactive=interactive)
        persist_v3426_options(v3426_opts, slm_root)
    except Exception as exc:
        # Wizard must never crash over an advisory feature.
        print(f"  (v3.4.26 options step skipped: {exc})")

    # External IDE configuration and login-time services cross ownership
    # boundaries. They require explicit consent even inside the setup wizard.
    _configure_external_integrations(interactive=interactive)
    _configure_autostart(interactive=interactive)

    # -- Done --
    _mark_complete()

    print()
    print("╔══════════════════════════════════════════════════════════╗")
    if verified:
        print("║  ✓ Setup Complete — The Unified Brain is ready!        ║")
    else:
        print("║  ✓ Setup Complete — basic config saved                 ║")
        print("║    Models will auto-download on first use              ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    # Summary of choices
    daemon_mode = "24/7" if config.daemon_idle_timeout == 0 else f"auto-shutdown ({config.daemon_idle_timeout}s)"
    print(f"  Enabled: Mode {choice.upper()}, Daemon ({daemon_mode})", end="")
    if config.mesh_enabled:
        print(", Mesh", end="")
    if config.entity_compilation_enabled:
        print(", Entity Compilation", end="")
    if code_graph_enabled:
        print(", CodeGraph", end="")
    if evolution_enabled:
        print(", Skill Evolution", end="")
    print()
    if enabled_adapters:
        print(f"  Adapters: {', '.join(enabled_adapters)}")
    else:
        print("  Adapters: none (enable via: slm adapters enable gmail)")
    print()
    # v3.8.2: surface component self-heal status so the user leaves setup
    # KNOWING everything they need is present — or exactly what to run if not.
    try:
        from superlocalmemory.core import component_registry as _cr

        _sum = _cr.snapshot(config)["summary"]
        if _sum.get("healthy"):
            print("  ✓ All components ready — models and dependencies present.")
        else:
            print(f"  ⚠ {_sum.get('missing', 0)} component(s) need attention.")
            print("    SLM auto-repairs fixable ones on daemon start. To check/fix now:")
            print("      slm doctor          (auto-fix: slm doctor --fix)")
    except Exception:
        pass
    print()
    print("  Quick start:")
    print('    slm remember "your first memory"')
    print('    slm recall "search query"')
    print("    slm dashboard              → http://localhost:8765")
    print("    slm adapters enable gmail  → start Gmail ingestion")
    print()
    print("  Need help?")
    print("    slm help       — every command, grouped (try: slm help self-heal)")
    print("    slm doctor     — diagnose issues (auto-fix: slm doctor --fix)")
    print("    slm serve install — install auto-start service")
    print("    https://github.com/qualixar/superlocalmemory")
    print()


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

def configure_provider(config: object, provider_name: str | None = None) -> None:
    """Configure an LLM provider for Mode C.

    When ``provider_name`` is supplied by ``slm provider set <provider>``,
    configuration is non-interactive and resolves its credential from the
    provider's documented environment variable. Omitting it preserves the
    existing interactive picker.
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
        print()

        idx = _prompt(f"  Select provider [1-{len(providers)}]: ", "1")
        try:
            provider_name = providers[int(idx) - 1]
        except (ValueError, IndexError):
            print("  Invalid choice. Using OpenAI.")
            provider_name = "openai"
    elif provider_name not in presets:
        raise ValueError(f"Unsupported provider: {provider_name}")

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

    # Provider selection is an additive configuration operation.  Rebuilding
    # via ``for_mode`` used to reset retrieval, scale-engine, evolution, and
    # user-tuned embedding settings — surprising and unsafe after a user had
    # configured a local or promoted Cozo/Lance deployment.  Keep every
    # unrelated setting intact and change only the mode/provider contract.
    from superlocalmemory.core.config import LLMConfig

    updated = config if isinstance(config, SLMConfig) else SLMConfig.load()
    updated.mode = Mode.C
    updated.llm = LLMConfig(
        provider=provider_name,
        model=preset["model"],
        api_key=api_key,
        api_base=preset["base_url"],
    )
    updated.save(mode_change=True)
    SLMConfig.write_current_mode(Mode.C, updated.base_dir)
    print(f"  Provider: {provider_name}")
    print(f"  Model: {preset['model']}")
