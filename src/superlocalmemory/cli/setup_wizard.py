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


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SLM_HOME = Path(os.environ.get("SL_MEMORY_PATH", Path.home() / ".superlocalmemory"))
_SETUP_MARKER = _SLM_HOME / ".setup-complete"
_EMBED_MODEL = "nomic-ai/nomic-embed-text-v1.5"
_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"


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
    return _SETUP_MARKER.exists()


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

    script = (
        f"import sys; "
        f"from sentence_transformers import SentenceTransformer; "
        f"m = SentenceTransformer('{model_name}', trust_remote_code=True); "
        f"d = m.get_sentence_embedding_dimension(); "
        f"print(f'OK dim={{d}}'); "
    )

    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
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

    script = (
        f"from sentence_transformers import CrossEncoder; "
        f"m = CrossEncoder('{model_name}', trust_remote_code=True); "
        f"print('OK'); "
    )

    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
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
    _SLM_HOME.mkdir(parents=True, exist_ok=True)
    _SETUP_MARKER.write_text(
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
    print(f"  Data dir: {_SLM_HOME}")

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
    print()
    print("─── Step 2/10: Choose Operating Mode ───")
    print()
    print("  [A] Local Guardian (recommended)")
    print("      Zero cloud. Zero LLM. Full privacy.")
    print("      EU AI Act compliant. Works immediately.")
    print()
    print("  [B] Smart Local")
    print("      Local LLM via Ollama for enrichment.")
    print("      Data stays on your machine.")
    print()
    print("  [C] Full Power")
    print("      Cloud LLM for maximum accuracy.")
    print("      Requires API key.")
    print()

    if interactive:
        choice = _prompt("  Select mode [A/B/C] (default: A): ", "a").lower()
    else:
        choice = "a"
        print("  Auto-selecting Mode A (non-interactive)")

    if choice not in ("a", "b", "c"):
        print(f"  Invalid choice '{choice}', using Mode A.")
        choice = "a"

    from superlocalmemory.core.config import SLMConfig
    from superlocalmemory.storage.models import Mode

    mode_map = {"a": Mode.A, "b": Mode.B, "c": Mode.C}
    config = SLMConfig.for_mode(mode_map[choice])

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
        config.save()

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
    _SLM_HOME.mkdir(parents=True, exist_ok=True)
    cg_config_path = _SLM_HOME / "code_graph_config.json"
    import json
    cg_config_data = {"enabled": code_graph_enabled, "bridge_enabled": code_graph_enabled}
    cg_config_path.write_text(json.dumps(cg_config_data, indent=2))

    if code_graph_enabled:
        print(f"\n  ✓ CodeGraph enabled")
        print(f"    Run `slm code-graph build` in any repo to index it")
    else:
        print(f"\n  ✓ CodeGraph disabled (enable later in {cg_config_path})")

    # -- Step 4: Download models --
    print()
    print("─── Step 4/10: Download Embedding Model ───")

    if not st_ok:
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

    config.save()

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
    config.save()
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

    adapters_path = _SLM_HOME / "adapters.json"
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
    config.save()
    print(f"\n  ✓ Entity compilation {'enabled' if config.entity_compilation_enabled else 'disabled'}")

    # -- Step 9: Skill Evolution (v3.4.11) --
    print()
    print("─── Step 9/10: Skill Evolution ───")
    print()
    print("  SLM can automatically evolve skills that underperform.")
    print("  It detects degradation, generates improvements, and verifies them blindly.")
    print("  Requires an LLM backend (Claude CLI, Ollama, or API key).")
    print()
    print("  [Y] Enable Skill Evolution")
    print("  [N] Disable (can enable later: slm config set evolution.enabled true)")
    print()

    if interactive:
        evo_choice = _prompt("  Enable Skill Evolution? [Y/n] (default: Y): ", "y").lower()
    else:
        evo_choice = "y"
        print("  Auto-enabling Skill Evolution (non-interactive)")

    evolution_enabled = evo_choice in ("", "y", "yes")

    # Write evolution config to config.json directly
    # (SLMConfig.save() doesn't serialize evolution)
    _SLM_HOME.mkdir(parents=True, exist_ok=True)
    evo_config_path = _SLM_HOME / "config.json"
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
    print("  Quick start:")
    print('    slm remember "your first memory"')
    print('    slm recall "search query"')
    print("    slm dashboard              → http://localhost:8765")
    print("    slm adapters enable gmail  → start Gmail ingestion")
    print()
    # V3.4.4: Auto-install OS service for daemon persistence (survive reboots)
    try:
        from superlocalmemory.cli.service_installer import install_service
        print("  Installing OS service for auto-start...")
        if install_service():
            print("  ✓ SLM will auto-start on login — zero friction.")
        else:
            print("  ⚠ OS service not installed (run: slm serve install)")
    except Exception:
        print("  ⚠ Could not install OS service (run: slm serve install)")

    print()
    print("  Need help?")
    print("    slm doctor     — diagnose issues")
    print("    slm --help     — all commands")
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

    On first use, also auto-installs Claude Code hooks so pip installs have
    the same "it just works" experience as npm installs (npm does this via
    postinstall; pip has no postinstall, so we do it here).
    """
    # Commands that work without setup
    _SKIP_COMMANDS = {"setup", "init", "hook", "hooks", "reap", "mcp"}
    if command in _SKIP_COMMANDS:
        return

    if is_setup_complete():
        return

    # Non-interactive: use defaults silently, don't block the command
    if not is_interactive():
        # Just create config with defaults and mark complete
        try:
            from superlocalmemory.core.config import SLMConfig
            from superlocalmemory.storage.models import Mode
            config = SLMConfig.for_mode(Mode.A)
            config.save()
            _mark_complete()
        except Exception:
            pass
        _maybe_install_hooks_on_first_use()
        return

    # Interactive: run the full wizard
    print()
    print("  First time using SuperLocalMemory!")
    print("  Running setup wizard...\n")
    run_wizard()
    _maybe_install_hooks_on_first_use()


def _maybe_install_hooks_on_first_use() -> None:
    """Install Claude Code hooks on first SLM run — matches npm postinstall.

    Install/uninstall parity rules:
      * Skip if the user explicitly opted out via ``slm hooks remove``
        (creates ``~/.superlocalmemory/hooks/.hooks-disabled``).
      * Skip if Claude Code isn't installed (no ``~/.claude/settings.json``
        to merge into).
      * Silent + best-effort: never fail a CLI command because of this.
    """
    try:
        opt_out = _SLM_HOME / "hooks" / ".hooks-disabled"
        if opt_out.exists():
            return
        claude_settings = Path.home() / ".claude" / "settings.json"
        if not claude_settings.exists():
            return  # Claude Code not installed — nothing to hook into.
        from superlocalmemory.hooks.claude_code_hooks import install_hooks
        install_hooks()
    except Exception:
        # Best-effort: parity-fallback, never block CLI.
        pass


# ---------------------------------------------------------------------------
# Mode C provider config (preserved from original)
# ---------------------------------------------------------------------------

def configure_provider(config: object) -> None:
    """Configure LLM provider for Mode C."""
    from superlocalmemory.core.config import SLMConfig
    from superlocalmemory.storage.models import Mode

    presets = SLMConfig.provider_presets()

    print()
    print("  Choose your LLM provider:")
    print()
    providers = list(presets.keys())
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

    preset = presets[provider_name]

    # Resolve API key
    env_key = preset.get("env_key", "")
    api_key = ""
    if env_key:
        existing = os.environ.get(env_key, "")
        if existing:
            print(f"  Found {env_key} in environment.")
            api_key = existing
        elif is_interactive():
            api_key = _prompt(
                f"  Enter your {provider_name.capitalize()} API key: ",
            )

    updated = SLMConfig.for_mode(
        Mode.C,
        llm_provider=provider_name,
        llm_model=preset["model"],
        llm_api_key=api_key,
        llm_api_base=preset["base_url"],
    )
    updated.save()
    print(f"  Provider: {provider_name}")
    print(f"  Model: {preset['model']}")
