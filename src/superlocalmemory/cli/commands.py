# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""CLI command implementations.

Each function handles one CLI command. Dispatch routes by name.
All data-returning commands support --json for agent-native output.

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

import os
import sys
from argparse import Namespace


def _cmd_db_dispatch(args: Namespace) -> None:
    """Route ``slm db ...`` subcommands. LLD-06 §7.2."""
    sub = getattr(args, "db_command", None)
    if sub == "migrate":
        from superlocalmemory.cli.db_migrate import cmd_db_migrate
        rc = cmd_db_migrate(args)
        if rc:
            sys.exit(rc)
        return
    print("Usage: slm db migrate [--status] [--dry-run]")
    sys.exit(2)


def _cmd_escape_disable(args: Namespace) -> None:
    from superlocalmemory.cli.escape_hatch import cmd_disable
    cmd_disable(args)


def _cmd_escape_enable(args: Namespace) -> None:
    from superlocalmemory.cli.escape_hatch import cmd_enable
    cmd_enable(args)


def _cmd_escape_clear_cache(args: Namespace) -> None:
    from superlocalmemory.cli.escape_hatch import cmd_clear_cache
    cmd_clear_cache(args)


def _cmd_escape_reconfigure(args: Namespace) -> None:
    from superlocalmemory.cli.escape_hatch import cmd_reconfigure
    cmd_reconfigure(args)


def _cmd_escape_benchmark(args: Namespace) -> None:
    from superlocalmemory.cli.escape_hatch import cmd_benchmark
    cmd_benchmark(args)


def _cmd_escape_rotate_token(args: Namespace) -> None:
    """S-M07: rotate the install token."""
    from superlocalmemory.cli.escape_hatch import cmd_rotate_token
    cmd_rotate_token(args)


def dispatch(args: Namespace) -> None:
    """Route CLI command to the appropriate handler."""
    # Auto-install/upgrade hooks on version change (single file read, ~0.1ms)
    if args.command not in ("hooks", "init", "mcp"):
        try:
            from superlocalmemory.hooks.claude_code_hooks import auto_install_if_needed
            auto_install_if_needed()
        except Exception:
            pass

    handlers = {
        "init": cmd_init,
        "setup": cmd_setup,
        "mode": cmd_mode,
        "provider": cmd_provider,
        "connect": cmd_connect,
        "migrate": cmd_migrate,
        "list": cmd_list,
        "remember": cmd_remember,
        "recall": cmd_recall,
        "forget": cmd_forget,
        "delete": cmd_delete,
        "update": cmd_update,
        "status": cmd_status,
        "health": cmd_health,
        "doctor": cmd_doctor,
        "trace": cmd_trace,
        "mcp": cmd_mcp,
        "warmup": cmd_warmup,
        "dashboard": cmd_dashboard,
        "profile": cmd_profile,
        "hooks": cmd_hooks,
        "session-context": cmd_session_context,
        "observe": cmd_observe,
        # V3.3 commands
        "decay": cmd_decay,
        "quantize": cmd_quantize,
        "consolidate": cmd_consolidate,
        "soft-prompts": cmd_soft_prompts,
        "reap": cmd_reap,
        # V3.3.21 daemon
        "serve": cmd_serve,
        # V3.4.9 nuclear restart
        "restart": cmd_restart,
        # V3.4.3 ingestion adapters
        "adapters": cmd_adapters,
        # V3.4.8 external observation ingestion
        "ingest": cmd_ingest,
        # V3.4.11 skill evolution
        "config": cmd_config,
        "evolve": cmd_evolve,
        # V3.4.21 LLD-05 context pre-staging
        "context": _cmd_context_dispatch,
        # V3.4.21 LLD-06 additive schema migrations
        "db": _cmd_db_dispatch,
        # V3.4.21 Stage 8 SB-5 — MASTER-PLAN §8 escape hatches.
        "disable": _cmd_escape_disable,
        "enable": _cmd_escape_enable,
        "clear-cache": _cmd_escape_clear_cache,
        "reconfigure": _cmd_escape_reconfigure,
        "benchmark": _cmd_escape_benchmark,
        "rotate-token": _cmd_escape_rotate_token,
    }
    handler = handlers.get(args.command)
    if handler:
        handler(args)
    else:
        print(f"Unknown command: {args.command}")
        sys.exit(1)


# -- Daemon serve mode (V3.3.21) ------------------------------------------

def cmd_serve(args: Namespace) -> None:
    """Start/stop the SLM daemon for instant CLI response."""
    from superlocalmemory.cli.daemon import is_daemon_running, ensure_daemon, stop_daemon

    action = getattr(args, 'action', 'start')

    if action == 'stop':
        if stop_daemon():
            print("Daemon stopped.")
        else:
            print("Daemon was not running.")
        return

    if action == 'status':
        if is_daemon_running():
            from superlocalmemory.cli.daemon import daemon_request
            status = daemon_request("GET", "/status")
            if status:
                print(f"Daemon: RUNNING (PID {status['pid']}, "
                      f"mode={status['mode']}, facts={status['fact_count']}, "
                      f"uptime={status['uptime_s']}s, idle={status['idle_s']}s)")
            else:
                print("Daemon: RUNNING (could not get status)")
        else:
            print("Daemon: NOT RUNNING")
        # Also show OS service status
        try:
            from superlocalmemory.cli.service_installer import service_status
            svc = service_status()
            installed = svc.get("installed", False)
            print(f"OS Service: {'INSTALLED' if installed else 'NOT INSTALLED'} "
                  f"({svc.get('service_type', svc.get('platform', '?'))})")
        except Exception:
            pass
        return

    if action == 'install':
        # Install OS-level service for auto-start on boot/login
        from superlocalmemory.cli.service_installer import install_service
        print("Installing SLM as OS service (auto-start on login)...")
        if install_service():
            print("Service installed \u2713 — SLM will auto-start on login.")
            print("  slm serve status    — check service status")
            print("  slm serve uninstall — remove auto-start")
        else:
            print("Failed to install service. Check logs.")
        return

    if action == 'uninstall':
        from superlocalmemory.cli.service_installer import uninstall_service
        if uninstall_service():
            print("OS service removed \u2713 — SLM will no longer auto-start.")
        else:
            print("Failed to remove service.")
        return

    # Default: start
    if is_daemon_running():
        print("Daemon already running.")
        return

    print("Starting SLM daemon (engine warming up)...")
    if ensure_daemon():
        print("Daemon started \u2713 — CLI commands are now instant.")
        print("  slm serve status  — check daemon status")
        print("  slm serve stop    — stop daemon and free RAM")
    else:
        print("Failed to start daemon. Check ~/.superlocalmemory/logs/daemon.log")


# -- Ingestion Adapters (V3.4.3) ------------------------------------------


def cmd_restart(args: Namespace) -> None:
    """Nuclear restart: kill ALL orphans, clean state, start fresh, verify health.

    5-step pipeline:
      1. Kill ALL SLM processes (daemon + workers + orphans)
      2. Clean stale PID/port/lock files
      3. Start fresh daemon
      4. Wait for engine warmup + verify health
      5. Optionally open dashboard
    """
    import os
    import time
    from pathlib import Path

    use_json = getattr(args, "json", False)
    open_dashboard = getattr(args, "dashboard", False)
    slm_dir = Path.home() / ".superlocalmemory"
    steps: list[dict] = []

    def _log(step: int, name: str, status: str, detail: str = ""):
        entry = {"step": step, "name": name, "status": status, "detail": detail}
        steps.append(entry)
        if not use_json:
            icon = {"ok": "+", "warn": "!", "fail": "x"}.get(status, " ")
            print(f"  [{icon}] Step {step}: {name}" + (f" — {detail}" if detail else ""))

    if not use_json:
        print()
        print("  SLM Full System Restart")
        print("  " + "=" * 40)
        print()

    # Step 1: Kill ALL SLM processes
    killed = 0
    try:
        import psutil
        my_pid = os.getpid()
        targets = [
            "superlocalmemory.server.unified_daemon",
            "superlocalmemory.core.embedding_worker",
            "superlocalmemory.core.recall_worker",
            "superlocalmemory.core.reranker_worker",
            "superlocalmemory.cli.daemon",
        ]
        for proc in psutil.process_iter(["pid", "cmdline"]):
            try:
                if proc.pid == my_pid:
                    continue
                cmdline = " ".join(proc.info.get("cmdline") or [])
                if any(t in cmdline for t in targets):
                    for child in proc.children(recursive=True):
                        try:
                            child.kill()
                            killed += 1
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                    proc.kill()
                    killed += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    except ImportError:
        # Fallback: pkill
        import subprocess as _sp
        for pattern in [
            "superlocalmemory.server.unified_daemon",
            "superlocalmemory.core.embedding_worker",
            "superlocalmemory.core.recall_worker",
            "superlocalmemory.core.reranker_worker",
        ]:
            try:
                r = _sp.run(["pkill", "-9", "-f", pattern], capture_output=True, timeout=5)
                if r.returncode == 0:
                    killed += 1
            except Exception:
                pass

    _log(1, "Kill all SLM processes", "ok", f"{killed} processes killed")
    time.sleep(3)

    # Step 2: Clean stale files + HOLD the lock to prevent races
    # v3.4.13: Do NOT delete daemon.lock — HOLD it instead.
    # If we delete it, `slm mcp` (still running in Claude) will see no lock,
    # acquire a NEW lock, and start a second daemon during our restart.
    cleaned = []
    for fname in ("daemon.pid", "daemon.port", ".embedding-worker.pid", ".reranker-worker.pid"):
        fpath = slm_dir / fname
        if fpath.exists():
            fpath.unlink(missing_ok=True)
            cleaned.append(fname)

    # Hold the lock file to block other processes from starting a daemon
    _LOCK_FILE = slm_dir / "daemon.lock"
    _LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    restart_lock_fd = None
    try:
        restart_lock_fd = open(_LOCK_FILE, "w")
        if sys.platform != "win32":
            import fcntl
            fcntl.flock(restart_lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except Exception:
        pass  # Best-effort — don't block restart if lock fails

    _log(2, "Clean stale state files", "ok",
         f"removed: {', '.join(cleaned)}" if cleaned else "already clean")

    # Step 3: Start fresh daemon (lock still held — no races)
    time.sleep(1)
    from superlocalmemory.cli.daemon import ensure_daemon
    started = ensure_daemon()

    # Release restart lock — daemon is now running with its own lock
    if restart_lock_fd:
        try:
            restart_lock_fd.close()
        except Exception:
            pass
    _log(3, "Start fresh daemon", "ok" if started else "fail",
         "daemon started" if started else "failed to start — check slm doctor")

    if not started:
        if use_json:
            from superlocalmemory.cli.json_output import json_print
            json_print("restart", data={"steps": steps, "success": False},
                       next_actions=[{"command": "slm doctor", "description": "Diagnose issues"}])
        else:
            print("\n  Restart FAILED at step 3. Run: slm doctor")
        return

    # Step 4: Wait for warmup + verify health
    if not use_json:
        print("  [ ] Step 4: Waiting for engine warmup (up to 30s)...", end="", flush=True)

    health = None
    engine_ok = False
    for attempt in range(15):
        time.sleep(2)
        try:
            from superlocalmemory.cli.daemon import daemon_request
            health = daemon_request("GET", "/health")
            if health and health.get("engine") == "initialized":
                engine_ok = True
                break
        except Exception:
            pass

    if not use_json:
        print("\r", end="")  # Clear the waiting line

    if engine_ok:
        version = health.get("version", "?")
        pid = health.get("pid", "?")
        _log(4, "Engine health verified", "ok", f"v{version}, PID {pid}, engine=initialized")
    else:
        engine_state = health.get("engine", "unknown") if health else "unreachable"
        _log(4, "Engine health check", "warn",
             f"engine={engine_state} — may still be warming up. Try again in 30s.")

    # Step 5: Database integrity check
    try:
        import sqlite3
        db_path = slm_dir / "memory.db"
        if db_path.exists():
            conn = sqlite3.connect(str(db_path))
            integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
            fact_count = conn.execute("SELECT COUNT(*) FROM atomic_facts").fetchone()[0]
            entity_count = conn.execute("SELECT COUNT(*) FROM canonical_entities").fetchone()[0]
            conn.close()
            _log(5, "Database integrity", "ok" if integrity == "ok" else "fail",
                 f"integrity={integrity}, {fact_count} facts, {entity_count} entities")
        else:
            _log(5, "Database check", "warn", "no database yet — will create on first use")
    except Exception as exc:
        _log(5, "Database check", "warn", str(exc))

    # Step 6 (optional): Open dashboard
    if open_dashboard:
        try:
            import webbrowser
            from superlocalmemory.cli.daemon import _get_port
            port = _get_port()
            url = f"http://localhost:{port}"
            webbrowser.open(url)
            _log(6, "Dashboard opened", "ok", url)
        except Exception as exc:
            _log(6, "Dashboard open", "fail", str(exc))

    # Summary
    all_ok = all(s["status"] == "ok" for s in steps)

    if use_json:
        from superlocalmemory.cli.json_output import json_print
        json_print("restart", data={
            "steps": steps, "success": all_ok,
            "processes_killed": killed,
            "version": health.get("version") if health else None,
        }, next_actions=[
            {"command": "slm serve status", "description": "Check daemon status"},
            {"command": "slm dashboard", "description": "Open dashboard"},
        ])
        return

    print()
    if all_ok:
        print("  All systems operational.")
    else:
        warnings = [s for s in steps if s["status"] != "ok"]
        print(f"  {len(warnings)} issue(s) — check details above.")
    print()


def cmd_ingest(args: Namespace) -> None:
    """Import external observations into SLM learning pipeline."""
    from superlocalmemory.cli.ingest_cmd import cmd_ingest as _ingest
    _ingest(args)


# -- Config & Evolution (V3.4.11) -----------------------------------------------


def cmd_config(args: Namespace) -> None:
    """Get or set config values (dot-notation).

    Usage:
      slm config set evolution.enabled true
      slm config set evolution.backend auto
      slm config get evolution.enabled
      slm config get evolution.backend
    """
    import json
    from pathlib import Path

    use_json = getattr(args, "json", False)
    action = getattr(args, "action", "get")
    key = getattr(args, "key", "")
    value = getattr(args, "value", None)

    config_path = Path.home() / ".superlocalmemory" / "config.json"

    # Read existing config
    cfg: dict = {}
    if config_path.exists():
        try:
            cfg = json.loads(config_path.read_text())
        except (json.JSONDecodeError, OSError):
            pass

    if action == "get":
        # Parse dot-notation key (e.g. "evolution.enabled")
        parts = key.split(".") if key else []
        node = cfg
        for part in parts:
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                node = None
                break

        if use_json:
            from superlocalmemory.cli.json_output import json_print
            json_print("config", data={"key": key, "value": node})
        else:
            if node is None:
                print(f"{key}: (not set)")
            else:
                print(f"{key}: {node}")

    elif action == "set":
        _ALLOWED_CONFIG_KEYS = {
            "evolution.enabled", "evolution.backend", "evolution.max_evolutions_per_cycle",
            "mesh_enabled", "daemon_idle_timeout", "entity_compilation_enabled",
        }
        if key not in _ALLOWED_CONFIG_KEYS:
            if use_json:
                from superlocalmemory.cli.json_output import json_print
                json_print("config", error={
                    "code": "DISALLOWED_KEY",
                    "message": f"'{key}' is not a configurable key. Allowed: {', '.join(sorted(_ALLOWED_CONFIG_KEYS))}",
                })
            else:
                print(f"Error: '{key}' is not a configurable key. Allowed: {', '.join(sorted(_ALLOWED_CONFIG_KEYS))}")
            sys.exit(1)

        if not key or value is None:
            if use_json:
                from superlocalmemory.cli.json_output import json_print
                json_print("config", error={
                    "code": "INVALID_INPUT",
                    "message": "Usage: slm config set <key> <value>",
                })
            else:
                print("Usage: slm config set <key> <value>")
            sys.exit(1)

        # Parse value: booleans, numbers, strings
        parsed_value: object
        if value.lower() in ("true", "yes", "on"):
            parsed_value = True
        elif value.lower() in ("false", "no", "off"):
            parsed_value = False
        else:
            try:
                parsed_value = int(value)
            except ValueError:
                try:
                    parsed_value = float(value)
                except ValueError:
                    parsed_value = value

        # Set via dot-notation (e.g. "evolution.enabled" -> cfg["evolution"]["enabled"])
        parts = key.split(".")
        node = cfg
        for part in parts[:-1]:
            if part not in node or not isinstance(node.get(part), dict):
                node[part] = {}
            node = node[part]
        old_value = node.get(parts[-1])
        node[parts[-1]] = parsed_value

        # Write back
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(cfg, indent=2) + "\n")

        if use_json:
            from superlocalmemory.cli.json_output import json_print
            json_print("config", data={
                "key": key, "old_value": old_value, "new_value": parsed_value,
            })
        else:
            print(f"{key}: {old_value} -> {parsed_value}")

    else:
        if use_json:
            from superlocalmemory.cli.json_output import json_print
            json_print("config", error={
                "code": "UNKNOWN_ACTION",
                "message": f"Unknown action: {action}. Use 'get' or 'set'.",
            })
        else:
            print(f"Unknown config action: {action}. Use 'get' or 'set'.")
        sys.exit(1)


def cmd_evolve(args: Namespace) -> None:
    """Run skill evolution for a session (called from Stop hook).

    Reads config.json to check if evolution is enabled.
    If enabled, imports SkillEvolver and runs run_post_session().
    If disabled, exits silently (zero output for fire-and-forget).
    """
    import json
    from pathlib import Path

    session_id = getattr(args, "session", "") or ""
    profile = getattr(args, "profile", "default") or "default"

    if not session_id:
        return  # Silent exit — nothing to do without a session

    # Check if evolution is enabled via config.json
    config_path = Path.home() / ".superlocalmemory" / "config.json"
    try:
        cfg = json.loads(config_path.read_text()) if config_path.exists() else {}
    except (json.JSONDecodeError, OSError):
        return  # Config unreadable — silent exit

    evolution_cfg = cfg.get("evolution", {})
    if not evolution_cfg.get("enabled", False):
        return  # Disabled — silent exit

    # Heavy imports only if enabled (this runs as a Popen child)
    try:
        from superlocalmemory.evolution.skill_evolver import SkillEvolver

        db_path = Path.home() / ".superlocalmemory" / "memory.db"
        if not db_path.exists():
            return

        evolver = SkillEvolver(db_path)
        evolver.run_post_session(session_id, profile)
    except Exception:
        pass  # Best-effort — don't crash the Stop hook


def cmd_adapters(args: Namespace) -> None:
    """Manage ingestion adapters (Gmail, Calendar, Transcript).

    Usage:
      slm adapters list                — show all adapters
      slm adapters enable <name>       — enable an adapter
      slm adapters disable <name>      — disable and stop
      slm adapters start <name>        — start running
      slm adapters stop <name>         — stop running
      slm adapters status              — detailed status
    """
    from superlocalmemory.ingestion.adapter_manager import handle_adapters_cli
    # args.rest contains everything after "adapters"
    rest = getattr(args, 'rest', []) or []
    handle_adapters_cli(rest)


# -- Setup & Config (no --json — interactive commands) ---------------------


def cmd_setup(args: Namespace) -> None:
    """Run the interactive setup wizard."""
    from superlocalmemory.cli.setup_wizard import run_wizard

    run_wizard(auto=getattr(args, "auto", False))
    sys.exit(0)  # Force clean exit (background threads from imports may linger)


def cmd_mode(args: Namespace) -> None:
    """Get or set the operating mode."""
    from superlocalmemory.core.config import SLMConfig
    from superlocalmemory.storage.models import Mode

    config = SLMConfig.load()

    if getattr(args, 'json', False):
        from superlocalmemory.cli.json_output import json_print
        if args.value:
            old_mode = config.mode.value.upper()
            updated = SLMConfig.for_mode(
                Mode(args.value),
                llm_provider=config.llm.provider,
                llm_model=config.llm.model,
                llm_api_key=config.llm.api_key,
                llm_api_base=config.llm.api_base,
            )
            updated.save()
            json_print("mode", data={
                "previous_mode": old_mode, "current_mode": args.value.upper(),
            }, next_actions=[
                {"command": "slm status --json", "description": "Check system status"},
            ])
        else:
            json_print("mode", data={"current_mode": config.mode.value.upper()},
                       next_actions=[
                           {"command": "slm mode a --json", "description": "Switch to zero-cloud mode"},
                           {"command": "slm mode c --json", "description": "Switch to full-power mode"},
                       ])
        return

    if args.value:
        updated = SLMConfig.for_mode(
            Mode(args.value),
            llm_provider=config.llm.provider,
            llm_model=config.llm.model,
            llm_api_key=config.llm.api_key,
            llm_api_base=config.llm.api_base,
        )
        updated.save()
        print(f"Mode set to: {args.value.upper()}")

        # V3.3: Check if embedding model changed — inform about re-indexing
        if (config.embedding.provider != updated.embedding.provider
                or config.embedding.model_name != updated.embedding.model_name):
            print("  ⚠ Embedding model changed. Re-indexing will run on next recall.")

        # V3.3.4: Warn if Mode C lacks cloud API key
        if args.value == "c" and not updated.llm.api_key:
            print("  ⚠ Mode C requires a cloud API key. Run: slm provider set")
    else:
        print(f"Current mode: {config.mode.value.upper()}")


def cmd_provider(args: Namespace) -> None:
    """Get or set the LLM provider."""
    from superlocalmemory.core.config import SLMConfig

    config = SLMConfig.load()

    if args.action == "set":
        from superlocalmemory.cli.setup_wizard import configure_provider

        configure_provider(config)
    else:
        print(f"Provider: {config.llm.provider or 'none (Mode A)'}")
        if config.llm.model:
            print(f"Model: {config.llm.model}")


def _cmd_context_dispatch(args: Namespace) -> None:
    """V3.4.21 LLD-05: ``slm context prestage``."""
    from superlocalmemory.cli.context_commands import cmd_context
    cmd_context(args)


def cmd_connect(args: Namespace) -> None:
    """Configure IDE integrations. V3.4.21: ``--cross-platform`` uses LLD-05."""
    # Route --disable <name> and --cross-platform to the LLD-05 orchestrator.
    if getattr(args, "disable", None) or getattr(args, "cross_platform", False):
        from superlocalmemory.cli.context_commands import (
            cmd_connect_cross_platform,
        )
        cmd_connect_cross_platform(args)
        return
    from superlocalmemory.hooks.ide_connector import IDEConnector

    connector = IDEConnector()

    if getattr(args, 'json', False):
        from superlocalmemory.cli.json_output import json_print
        if getattr(args, "list", False):
            json_print("connect", data={"ides": connector.get_status()},
                       next_actions=[
                           {"command": "slm connect --json", "description": "Auto-configure all IDEs"},
                       ])
        elif getattr(args, "ide", None):
            success = connector.connect(args.ide)
            json_print("connect", data={"ide": args.ide, "connected": success})
        else:
            json_print("connect", data={"results": connector.connect_all()},
                       next_actions=[
                           {"command": "slm status --json", "description": "Check system status"},
                       ])
        return

    if getattr(args, "list", False):
        status = connector.get_status()
        for s in status:
            mark = "[+]" if s["installed"] else "[-]"
            print(f"  {mark} {s['name']:20s} {s['config_path']}")
        return
    if getattr(args, "ide", None):
        success = connector.connect(args.ide)
        print(f"{'Connected' if success else 'Failed'}: {args.ide}")
    else:
        results = connector.connect_all()
        for ide_id, ide_status in results.items():
            print(f"  {ide_id}: {ide_status}")


def cmd_migrate(args: Namespace) -> None:
    """Run V2 to V3 migration."""
    from superlocalmemory.cli.migrate_cmd import cmd_migrate as _migrate

    _migrate(args)


# -- Memory Operations (all support --json) --------------------------------


def cmd_list(args: Namespace) -> None:
    """List recent memories chronologically."""
    from superlocalmemory.core.config import SLMConfig
    from superlocalmemory.core.engine import MemoryEngine

    use_json = getattr(args, 'json', False)
    try:
        config = SLMConfig.load()
        engine = MemoryEngine(config)
        engine.initialize()

        limit = getattr(args, "limit", 20)
        facts = engine._db.get_all_facts(engine.profile_id)
        facts.sort(key=lambda f: f.created_at or "", reverse=True)
        facts = facts[:limit]
    except Exception as exc:
        if use_json:
            from superlocalmemory.cli.json_output import json_print
            json_print("list", error={"code": "ENGINE_ERROR", "message": str(exc)})
            sys.exit(1)
        raise

    if use_json:
        from superlocalmemory.cli.json_output import json_print
        items = []
        for f in facts:
            ftype_raw = getattr(f, "fact_type", "")
            ftype = ftype_raw.value if hasattr(ftype_raw, "value") else str(ftype_raw)
            items.append({
                "fact_id": f.fact_id, "content": f.content,
                "fact_type": ftype, "created_at": (f.created_at or "")[:19],
            })
        json_print("list", data={"results": items, "count": len(items)},
                   next_actions=[
                       {"command": "slm recall '<query>' --json", "description": "Search memories"},
                       {"command": "slm delete <fact_id> --json --yes", "description": "Delete a memory"},
                   ])
        return

    if not facts:
        print("No memories stored yet.")
    else:
        print(f"Recent memories ({len(facts)}):\n")
        for i, f in enumerate(facts, 1):
            date = (f.created_at or "")[:19]
            ftype_raw = getattr(f, "fact_type", "")
            ftype = ftype_raw.value if hasattr(ftype_raw, "value") else str(ftype_raw)
            content = f.content[:100] + ("..." if len(f.content) > 100 else "")
            print(f"  {i:3d}. [{date}] ({ftype}) {content}")

    # V3.3.21: Show pending memories (store-first pattern)
    try:
        from superlocalmemory.cli.pending_store import get_pending
        pending = get_pending(limit=10)
        if pending:
            print(f"\nPending (processing in background): {len(pending)}")
            for p in pending:
                content = p["content"][:80] + ("..." if len(p["content"]) > 80 else "")
                print(f"  \u23f3 [{p['created_at'][:19]}] {content}")
    except Exception:
        pass


def cmd_remember(args: Namespace) -> None:
    """Store a memory via the engine."""
    from superlocalmemory.core.config import SLMConfig

    use_json = getattr(args, 'json', False)
    sync_mode = getattr(args, 'sync_mode', False)

    # V3.3.21: Route through daemon for instant remember (no cold start).
    # If daemon is running, send request directly (~0.1s).
    # If not, use store-first pattern (pending.db) as fallback.
    if not sync_mode:
        # Try daemon first
        try:
            from superlocalmemory.cli.daemon import is_daemon_running, daemon_request, ensure_daemon
            if is_daemon_running() or ensure_daemon():
                result = daemon_request("POST", "/remember", {
                    "content": args.content,
                    "tags": args.tags or "",
                })
                if result and "fact_ids" in result:
                    if use_json:
                        from superlocalmemory.cli.json_output import json_print
                        json_print("remember", data=result)
                    else:
                        print(f"Stored \u2713 {result['count']} facts (via daemon).")
                    return
        except Exception:
            pass  # Fall through to pending store

        # v3.4.13: Store to pending DB (zero data loss) — daemon processes in background.
        # NO subprocess spawn. Daemon's background loop picks up pending memories.
        from superlocalmemory.cli.pending_store import store_pending

        row_id = store_pending(
            content=args.content,
            tags=args.tags or "",
        )

        if use_json:
            from superlocalmemory.cli.json_output import json_print
            json_print("remember", data={"queued": True, "async": True,
                                         "pending_id": row_id, "safe": True})
        else:
            print(f"Stored \u2713 (pending_id={row_id}) \u2014 processing in background.")
        return

    from superlocalmemory.core.engine import MemoryEngine

    try:
        config = SLMConfig.load()
        engine = MemoryEngine(config)
        engine.initialize()

        metadata = {"tags": args.tags} if args.tags else {}
        fact_ids = engine.store(args.content, metadata=metadata)
    except Exception as exc:
        if use_json:
            from superlocalmemory.cli.json_output import json_print
            json_print("remember", error={"code": "STORE_ERROR", "message": str(exc)})
            sys.exit(1)
        raise

    if use_json:
        from superlocalmemory.cli.json_output import json_print
        json_print("remember", data={"fact_ids": fact_ids, "count": len(fact_ids)},
                   next_actions=[
                       {"command": "slm recall '<query>' --json", "description": "Search your memories"},
                       {"command": "slm list --json -n 5", "description": "See recent memories"},
                   ])
        return

    print(f"Stored {len(fact_ids)} facts.")


def cmd_recall(args: Namespace) -> None:
    """Search memories via the engine — routes through daemon if available."""
    use_json = getattr(args, 'json', False)

    # V3.3.21: Route through daemon for instant response (no cold start).
    # Falls back to direct engine if daemon not running.
    # S9-DASH-02: pass a stable session_id derived from the shell's
    # parent PID so a sequence of CLI recalls in one terminal can be
    # grouped. The Stop hook on session end won't fire for CLI, so
    # these outcomes are closed by the reaper (TTL → neutral reward).
    try:
        from superlocalmemory.cli.daemon import is_daemon_running, daemon_request, ensure_daemon
        if is_daemon_running() or ensure_daemon():
            from urllib.parse import quote
            session_id = f"cli:{os.getppid()}"
            result = daemon_request(
                "GET",
                f"/recall?q={quote(args.query)}&limit={args.limit}"
                f"&session_id={quote(session_id)}",
            )
            if result and "results" in result:
                # Format daemon response same as engine response
                if use_json:
                    from superlocalmemory.cli.json_output import json_print
                    json_print("recall", data=result, next_actions=[
                        {"command": "slm list --json", "description": "List recent memories"},
                    ])
                    return
                if not result["results"]:
                    print("No matching memories found.")
                    return
                # Text output
                print(f"SpreadingActivation.search completed via daemon ({result.get('retrieval_time_ms', 0):.0f}ms)")
                for i, r in enumerate(result["results"], 1):
                    print(f"  {i}. [{r['score']:.2f}] {r['content']}")
                return
    except Exception:
        pass  # Fall through to direct engine

    from superlocalmemory.core.config import SLMConfig
    from superlocalmemory.core.engine import MemoryEngine

    try:
        config = SLMConfig.load()
        engine = MemoryEngine(config)
        engine.initialize()

        response = engine.recall(args.query, limit=args.limit)
    except Exception as exc:
        if use_json:
            from superlocalmemory.cli.json_output import json_print
            json_print("recall", error={"code": "RECALL_ERROR", "message": str(exc)})
            sys.exit(1)
        raise

    if use_json:
        from superlocalmemory.cli.json_output import json_print
        items = []
        for r in response.results:
            item = {
                "fact_id": r.fact.fact_id, "content": r.fact.content,
                "score": round(r.score, 3),
            }
            if hasattr(r, "channel_scores") and r.channel_scores:
                item["channel_scores"] = {k: round(v, 3) for k, v in r.channel_scores.items()}
            items.append(item)
        json_print("recall", data={
            "results": items, "count": len(items),
            "query_type": getattr(response, "query_type", "unknown"),
        }, next_actions=[
            {"command": "slm list --json", "description": "List recent memories"},
        ])
        return

    # Record learning signals (CLI path — works without MCP)
    try:
        _cli_record_signals(config, args.query, response.results)
    except Exception:
        pass

    if not response.results:
        print("No memories found.")
        return
    for i, r in enumerate(response.results, 1):
        print(f"  {i}. [{r.score:.2f}] {r.fact.content[:120]}")


def _cli_record_signals(config, query, results):
    """Record learning signals from CLI recall (no MCP dependency)."""
    from pathlib import Path
    from superlocalmemory.learning.feedback import FeedbackCollector
    from superlocalmemory.learning.signals import LearningSignals
    slm_dir = Path.home() / ".superlocalmemory"
    pid = config.active_profile
    fact_ids = [r.fact.fact_id for r in results[:10]]
    if not fact_ids:
        return
    FeedbackCollector(slm_dir / "learning.db").record_implicit(
        profile_id=pid, query=query,
        fact_ids_returned=fact_ids, fact_ids_available=fact_ids,
    )
    signals = LearningSignals(slm_dir / "learning.db")
    signals.record_co_retrieval(pid, fact_ids)
    for fid in fact_ids[:5]:
        LearningSignals.boost_confidence(str(slm_dir / "memory.db"), fid)


def cmd_forget(args: Namespace) -> None:
    """Delete memories matching a query."""
    from superlocalmemory.core.engine import MemoryEngine
    from superlocalmemory.core.config import SLMConfig

    use_json = getattr(args, 'json', False)
    try:
        config = SLMConfig.load()
        engine = MemoryEngine(config)
        engine.initialize()
        facts = engine._db.get_all_facts(engine.profile_id)
        query_lower = args.query.lower()
        matches = [f for f in facts if query_lower in f.content.lower()]
    except Exception as exc:
        if use_json:
            from superlocalmemory.cli.json_output import json_print
            json_print("forget", error={"code": "ENGINE_ERROR", "message": str(exc)})
            sys.exit(1)
        raise

    dry_run = getattr(args, 'dry_run', False)

    if use_json:
        from superlocalmemory.cli.json_output import json_print
        if not matches:
            json_print("forget", data={"matched_count": 0, "deleted_count": 0, "matches": []})
            return
        match_items = [{"fact_id": f.fact_id, "content": f.content[:120]} for f in matches[:20]]
        if dry_run:
            json_print("forget", data={
                "matched_count": len(matches), "deleted_count": 0,
                "dry_run": True, "matches": match_items,
            })
            return
        if getattr(args, 'yes', False):
            for f in matches:
                engine._db.delete_fact(f.fact_id)
            json_print("forget", data={
                "matched_count": len(matches), "deleted_count": len(matches),
                "deleted": [f.fact_id for f in matches],
            }, next_actions=[
                {"command": "slm list --json", "description": "Verify remaining memories"},
            ])
        else:
            json_print("forget", data={
                "matched_count": len(matches), "deleted_count": 0,
                "matches": match_items,
                "hint": "Add --yes to confirm deletion",
            }, next_actions=[
                {"command": f"slm forget '{args.query}' --json --yes", "description": "Confirm deletion"},
            ])
        return

    if not matches:
        print(f"No memories matching '{args.query}'")
        return
    print(f"Found {len(matches)} matching memories:")
    for f in matches[:10]:
        print(f"  - {f.fact_id[:8]}... {f.content[:80]}")
    if dry_run:
        print(f"(dry run — {len(matches)} would be deleted)")
        return
    if getattr(args, 'yes', False):
        for f in matches:
            engine._db.delete_fact(f.fact_id)
        print(f"Deleted {len(matches)} memories.")
        return
    confirm = input(f"Delete {len(matches)} memories? [y/N] ").strip().lower()
    if confirm in ("y", "yes"):
        for f in matches:
            engine._db.delete_fact(f.fact_id)
        print(f"Deleted {len(matches)} memories.")
    else:
        print("Cancelled.")


def cmd_delete(args: Namespace) -> None:
    """Delete a specific memory by exact fact ID."""
    from superlocalmemory.core.config import SLMConfig
    from superlocalmemory.core.engine import MemoryEngine

    use_json = getattr(args, 'json', False)
    try:
        config = SLMConfig.load()
        engine = MemoryEngine(config)
        engine.initialize()

        fact_id = args.fact_id.strip()
        rows = engine._db.execute(
            "SELECT content FROM atomic_facts WHERE fact_id = ? AND profile_id = ?",
            (fact_id, engine.profile_id),
        )
    except Exception as exc:
        if use_json:
            from superlocalmemory.cli.json_output import json_print
            json_print("delete", error={"code": "ENGINE_ERROR", "message": str(exc)})
            sys.exit(1)
        raise

    if use_json:
        from superlocalmemory.cli.json_output import json_print
        if not rows:
            json_print("delete", error={
                "code": "NOT_FOUND", "message": f"Memory not found: {fact_id}",
            })
            sys.exit(1)
        content = dict(rows[0]).get("content", "")
        if getattr(args, "yes", False):
            engine._db.delete_fact(fact_id)
            json_print("delete", data={"deleted": fact_id, "content": content[:120]},
                       next_actions=[
                           {"command": "slm list --json", "description": "Verify remaining memories"},
                       ])
        else:
            json_print("delete", data={
                "fact_id": fact_id, "content": content[:120], "deleted": False,
                "hint": "Add --yes to confirm deletion",
            }, next_actions=[
                {"command": f"slm delete {fact_id} --json --yes", "description": "Confirm deletion"},
            ])
        return

    if not rows:
        print(f"Memory not found: {fact_id}")
        return

    content_preview = dict(rows[0]).get("content", "")[:120]
    print(f"Memory: {content_preview}")

    if not getattr(args, "yes", False):
        confirm = input("Delete this memory? [y/N] ").strip().lower()
        if confirm not in ("y", "yes"):
            print("Cancelled.")
            return

    engine._db.delete_fact(fact_id)
    print(f"Deleted: {fact_id}")


def cmd_update(args: Namespace) -> None:
    """Update the content of a specific memory by exact fact ID."""
    from superlocalmemory.core.config import SLMConfig
    from superlocalmemory.core.engine import MemoryEngine

    use_json = getattr(args, 'json', False)
    fact_id = args.fact_id.strip()
    new_content = args.content.strip()

    if not new_content:
        if use_json:
            from superlocalmemory.cli.json_output import json_print
            json_print("update", error={"code": "INVALID_INPUT", "message": "content cannot be empty"})
            sys.exit(1)
        print("Error: content cannot be empty")
        return

    try:
        config = SLMConfig.load()
        engine = MemoryEngine(config)
        engine.initialize()

        rows = engine._db.execute(
            "SELECT content FROM atomic_facts WHERE fact_id = ? AND profile_id = ?",
            (fact_id, engine.profile_id),
        )
    except Exception as exc:
        if use_json:
            from superlocalmemory.cli.json_output import json_print
            json_print("update", error={"code": "ENGINE_ERROR", "message": str(exc)})
            sys.exit(1)
        raise

    if not rows:
        if use_json:
            from superlocalmemory.cli.json_output import json_print
            json_print("update", error={
                "code": "NOT_FOUND", "message": f"Memory not found: {fact_id}",
            })
            sys.exit(1)
        print(f"Memory not found: {fact_id}")
        return

    old_content = dict(rows[0]).get("content", "")
    engine._db.execute(
        "UPDATE atomic_facts SET content = ? WHERE fact_id = ?",
        (new_content, fact_id),
    )

    if use_json:
        from superlocalmemory.cli.json_output import json_print
        json_print("update", data={
            "fact_id": fact_id,
            "old_content": old_content[:120],
            "new_content": new_content[:120],
        }, next_actions=[
            {"command": "slm list --json", "description": "List recent memories"},
        ])
        return

    print(f"Old: {old_content[:100]}")
    print(f"New: {new_content[:100]}")
    print(f"Updated: {fact_id}")


# -- Diagnostics (all support --json) -------------------------------------


def cmd_status(args: Namespace) -> None:
    """Show system status."""
    from superlocalmemory.core.config import SLMConfig

    config = SLMConfig.load()

    if getattr(args, 'json', False):
        from superlocalmemory.cli.json_output import json_print
        data = {
            "mode": config.mode.value.upper(),
            "provider": config.llm.provider or "none",
            "base_dir": str(config.base_dir),
            "db_path": str(config.db_path),
        }
        if config.db_path.exists():
            data["db_size_mb"] = round(config.db_path.stat().st_size / 1024 / 1024, 2)
        json_print("status", data=data, next_actions=[
            {"command": "slm health --json", "description": "Check math layer health"},
            {"command": "slm list --json", "description": "List recent memories"},
        ])
        return

    print("SuperLocalMemory V3")
    print(f"  Mode: {config.mode.value.upper()}")
    print(f"  Provider: {config.llm.provider or 'none'}")
    print(f"  Base dir: {config.base_dir}")
    print(f"  Database: {config.db_path}")
    if config.db_path.exists():
        size_mb = round(config.db_path.stat().st_size / 1024 / 1024, 2)
        print(f"  DB size: {size_mb} MB")

    # S9-UX-07 / S9-UX-13: --verbose surfaces the disabled marker,
    # last-version marker, and daemon port so users who are debugging
    # "slm seems broken" get the signal without opening three files.
    if getattr(args, "verbose", False):
        home = config.base_dir
        disabled_path = home / ".disabled"
        version_path = home / ".last_version"
        print()
        print("  --verbose --")
        print(
            f"  Disabled marker: "
            f"{'YES (slm enable to reactivate)' if disabled_path.exists() else 'no'}",
        )
        if version_path.exists():
            try:
                last_ver = version_path.read_text(encoding="utf-8").strip()
            except OSError:
                last_ver = "(unreadable)"
            print(f"  Last booted version: {last_ver}")
        else:
            print("  Last booted version: (never booted)")
        port = os.environ.get("SLM_DAEMON_PORT") or "8765"
        print(f"  Daemon port: {port}")


def cmd_health(args: Namespace) -> None:
    """Show math layer health status."""
    from superlocalmemory.core.engine import MemoryEngine
    from superlocalmemory.core.config import SLMConfig

    use_json = getattr(args, 'json', False)
    try:
        config = SLMConfig.load()
        engine = MemoryEngine(config)
        engine.initialize()
        facts = engine._db.get_all_facts(engine.profile_id)
        fisher_count = sum(1 for f in facts if f.fisher_mean is not None)
        langevin_count = sum(1 for f in facts if f.langevin_position is not None)
    except Exception as exc:
        if use_json:
            from superlocalmemory.cli.json_output import json_print
            json_print("health", error={"code": "ENGINE_ERROR", "message": str(exc)})
            sys.exit(1)
        raise

    if use_json:
        from superlocalmemory.cli.json_output import json_print
        json_print("health", data={
            "total_facts": len(facts),
            "similarity_indexed": fisher_count,
            "lifecycle_positioned": langevin_count,
            "mode": config.mode.value.upper(),
        }, next_actions=[
            {"command": "slm status --json", "description": "Check system status"},
            {"command": "slm recall '<query>' --json", "description": "Test retrieval"},
        ])
        return

    print("Math Layer Health:")
    print(f"  Total facts: {len(facts)}")
    print(f"  Fisher-Rao indexed: {fisher_count}/{len(facts)}")
    print(f"  Langevin positioned: {langevin_count}/{len(facts)}")
    print(f"  Mode: {config.mode.value.upper()}")


def cmd_doctor(args: Namespace) -> None:
    """Comprehensive pre-flight check — verify everything works.

    S9-UX-10: ``--quick`` skips the slow probes (embedding worker
    subprocess, Ollama roundtrip) so first-run installer hooks can
    surface a sub-second PASS/FAIL line before letting the user go.
    Full doctor remains the default for ``slm doctor`` with no flag.
    """
    import shutil
    from pathlib import Path

    use_json = getattr(args, "json", False)
    quick = getattr(args, "quick", False)
    checks: list[dict] = []
    passed = warned = failed = 0

    def _check(name: str, status: str, detail: str, fix: str = ""):
        nonlocal passed, warned, failed
        checks.append({"name": name, "status": status, "detail": detail, "fix": fix})
        if status == "PASS":
            passed += 1
        elif status == "WARN":
            warned += 1
        else:
            failed += 1
        if not use_json:
            tag = {"PASS": "[PASS]", "WARN": "[WARN]", "FAIL": "[FAIL]"}[status]
            line = f"  {tag} {name}: {detail}"
            if fix:
                line += f"\n         Fix: {fix}"
            print(line)

    if not use_json:
        print("SuperLocalMemory V3 — Doctor (Pre-flight Check)")
        print("=" * 50)
        print()

    # 1. Python version
    v = sys.version_info
    if v >= (3, 11):
        _check("Python", "PASS", f"{v.major}.{v.minor}.{v.micro} (>= 3.11)")
    else:
        _check("Python", "FAIL", f"{v.major}.{v.minor}.{v.micro} (need >= 3.11)",
               "Install Python 3.11+ from https://python.org/downloads/")

    # 2. Core deps
    core_modules = {
        "numpy": "numpy", "scipy": "scipy", "networkx": "networkx",
        "httpx": "httpx", "dateutil": "python-dateutil",
        "rank_bm25": "rank-bm25", "vaderSentiment": "vadersentiment",
        "einops": "einops",
    }
    core_ok, core_versions = [], []
    for mod, pkg in core_modules.items():
        try:
            m = __import__(mod)
            ver = getattr(m, "__version__", "?")
            core_ok.append(mod)
            core_versions.append(f"{mod} {ver}")
        except ImportError:
            pass
    if len(core_ok) == len(core_modules):
        _check("Core deps", "PASS", ", ".join(core_versions[:4]) + "...")
    else:
        missing = set(core_modules) - set(core_ok)
        _check("Core deps", "FAIL", f"Missing: {', '.join(missing)}",
               "pip install " + " ".join(core_modules[m] for m in missing))

    # 3. Search deps
    search_mods = {"sentence_transformers": "sentence-transformers", "torch": "torch",
                   "sklearn": "scikit-learn", "geoopt": "geoopt"}
    search_ok = []
    for mod, pkg in search_mods.items():
        try:
            __import__(mod)
            search_ok.append(mod)
        except ImportError:
            pass
    if len(search_ok) == len(search_mods):
        _check("Search deps", "PASS", "sentence-transformers, torch, sklearn, geoopt")
    else:
        missing = set(search_mods) - set(search_ok)
        _check("Search deps", "WARN", f"Missing: {', '.join(missing)}",
               "pip install 'superlocalmemory[search]'")

    # 4. Dashboard deps
    dash_ok = True
    for mod in ["fastapi", "uvicorn", "websockets"]:
        try:
            __import__(mod)
        except ImportError:
            dash_ok = False
            break
    if dash_ok:
        _check("Dashboard deps", "PASS", "fastapi, uvicorn, websockets")
    else:
        _check("Dashboard deps", "WARN", "Missing dashboard deps",
               "pip install 'fastapi[all]' uvicorn websockets")

    # 5. Learning deps
    try:
        import lightgbm
        _check("Learning deps", "PASS", f"lightgbm {lightgbm.__version__}")
    except ImportError:
        _check("Learning deps", "WARN", "lightgbm not installed",
               "pip install lightgbm")
    except OSError as exc:
        _check("Learning deps", "WARN", f"lightgbm installed but broken: {exc}",
               "brew install libomp && pip install --force-reinstall lightgbm")

    # 6. Performance deps
    perf_ok = []
    for mod in ["diskcache", "orjson"]:
        try:
            __import__(mod)
            perf_ok.append(mod)
        except ImportError:
            pass
    if len(perf_ok) == 2:
        _check("Performance deps", "PASS", "diskcache, orjson")
    else:
        missing = {"diskcache", "orjson"} - set(perf_ok)
        _check("Performance deps", "WARN", f"Missing: {', '.join(missing)}",
               "pip install diskcache orjson")

    # 7. Embedding worker functional test — skipped under --quick.
    if quick:
        _check("Embedding worker", "PASS", "skipped (--quick)")
    else:
        try:
            import subprocess as _sp
            import json as _json

            env = {
                **__import__("os").environ,
                "CUDA_VISIBLE_DEVICES": "",
                "PYTORCH_MPS_HIGH_WATERMARK_RATIO": "0.0",
                "TOKENIZERS_PARALLELISM": "false",
                "TORCH_DEVICE": "cpu",
            }
            proc = _sp.Popen(
                [sys.executable, "-m",
                 "superlocalmemory.core.embedding_worker"],
                stdin=_sp.PIPE, stdout=_sp.PIPE, stderr=_sp.DEVNULL,
                text=True, bufsize=1, env=env,
            )
            proc.stdin.write(_json.dumps({"cmd": "ping"}) + "\n")
            proc.stdin.flush()

            import select as _sel
            ready, _, _ = _sel.select([proc.stdout], [], [], 30)
            if ready:
                resp = _json.loads(proc.stdout.readline())
                if resp.get("ok"):
                    _check(
                        "Embedding worker", "PASS",
                        f"responsive (PID {proc.pid}, "
                        f"Python {sys.executable})",
                    )
                else:
                    _check(
                        "Embedding worker", "FAIL",
                        f"error: {resp.get('error', 'unknown')}",
                        "pip install sentence-transformers einops torch",
                    )
            else:
                _check("Embedding worker", "FAIL", "timed out (30s)",
                       "slm warmup")
            proc.stdin.write(_json.dumps({"cmd": "quit"}) + "\n")
            proc.stdin.flush()
            proc.wait(timeout=5)
        except FileNotFoundError:
            _check(
                "Embedding worker", "FAIL",
                "embedding_worker module not found",
                "Reinstall: npm install -g superlocalmemory",
            )
        except Exception as exc:
            _check("Embedding worker", "FAIL", str(exc), "slm warmup")

    # 8. Ollama connectivity (Mode B only) — skipped under --quick.
    if quick:
        _check("Ollama / API key", "PASS", "skipped (--quick)")
    else:
        try:
            from superlocalmemory.core.config import SLMConfig
            config = SLMConfig.load()
            if config.mode.value == "b":
                import httpx
                try:
                    resp = httpx.get(
                        f"{config.llm.api_base}/api/tags", timeout=5.0,
                    )
                    if resp.status_code == 200:
                        models = [
                            m["name"].split(":")[0]
                            for m in resp.json().get("models", [])
                        ]
                        has_llm = config.llm.model.split(":")[0] in models
                        if has_llm:
                            _check(
                                "Ollama", "PASS",
                                f"running, {len(models)} models, "
                                f"'{config.llm.model}' available",
                            )
                        else:
                            _check(
                                "Ollama", "WARN",
                                f"running but '{config.llm.model}' "
                                f"not pulled",
                                f"ollama pull {config.llm.model}",
                            )
                    else:
                        _check(
                            "Ollama", "WARN",
                            f"HTTP {resp.status_code}",
                            "brew services start ollama",
                        )
                except Exception:
                    _check(
                        "Ollama", "WARN",
                        "not reachable at " + config.llm.api_base,
                        "brew services start ollama",
                    )
            elif config.mode.value == "c":
                if config.llm.api_key:
                    _check(
                        "API key", "PASS",
                        f"provider={config.llm.provider}, "
                        f"key=***{config.llm.api_key[-4:]}",
                    )
                else:
                    _check(
                        "API key", "WARN", "no API key configured",
                        "slm provider set",
                    )
        except Exception:
            pass  # Config load failed — already caught above

    # 9. Disk space
    slm_home = Path.home() / ".superlocalmemory"
    try:
        usage = shutil.disk_usage(slm_home if slm_home.exists() else Path.home())
        free_gb = usage.free / (1024 ** 3)
        if free_gb >= 2.0:
            _check("Disk space", "PASS", f"{free_gb:.1f} GB free")
        else:
            _check("Disk space", "WARN", f"{free_gb:.1f} GB free (< 2 GB)",
                   "Free up disk space")
    except Exception:
        pass

    # 10. Database integrity
    db_path = slm_home / "memory.db"
    if db_path.exists():
        try:
            import sqlite3
            conn = sqlite3.connect(str(db_path))
            result = conn.execute("PRAGMA integrity_check").fetchone()
            conn.close()
            if result and result[0] == "ok":
                size_mb = db_path.stat().st_size / (1024 * 1024)
                _check("Database", "PASS", f"OK ({size_mb:.2f} MB)")
            else:
                _check("Database", "FAIL", f"integrity check: {result}",
                       "Backup and recreate database")
        except Exception as exc:
            _check("Database", "FAIL", str(exc))
    else:
        _check("Database", "PASS", "not yet created (will initialize on first use)")

    # Summary
    if use_json:
        from superlocalmemory.cli.json_output import json_print
        next_actions = []
        for c in checks:
            if c["fix"]:
                next_actions.append({"command": c["fix"], "description": f"Fix {c['name']}"})
        json_print("doctor", data={
            "checks": checks,
            "summary": {"passed": passed, "warned": warned, "failed": failed},
        }, next_actions=next_actions)
    else:
        print(f"\nSummary: {passed} passed, {warned} warnings, {failed} failed")
        if failed > 0:
            print("Run the suggested fix commands above, then re-run: slm doctor")


def cmd_trace(args: Namespace) -> None:
    """Recall with per-channel score breakdown."""
    from superlocalmemory.core.engine import MemoryEngine
    from superlocalmemory.core.config import SLMConfig

    use_json = getattr(args, 'json', False)
    try:
        config = SLMConfig.load()
        engine = MemoryEngine(config)
        limit = getattr(args, 'limit', 10)
        response = engine.recall(args.query, limit=limit)
    except Exception as exc:
        if use_json:
            from superlocalmemory.cli.json_output import json_print
            json_print("trace", error={"code": "ENGINE_ERROR", "message": str(exc)})
            sys.exit(1)
        raise

    if use_json:
        from superlocalmemory.cli.json_output import json_print
        items = []
        for r in response.results:
            item = {
                "fact_id": r.fact.fact_id, "content": r.fact.content[:200],
                "score": round(r.score, 3),
            }
            if hasattr(r, "channel_scores") and r.channel_scores:
                item["channel_scores"] = {
                    k: round(v, 3) for k, v in r.channel_scores.items()
                }
            items.append(item)
        json_print("trace", data={
            "query": args.query,
            "query_type": getattr(response, "query_type", "unknown"),
            "retrieval_time_ms": round(getattr(response, "retrieval_time_ms", 0), 1),
            "results": items, "count": len(items),
        }, next_actions=[
            {"command": "slm recall '<query>' --json", "description": "Standard recall"},
        ])
        return

    print(f"Query: {args.query}")
    print(f"Type: {response.query_type} | Time: {response.retrieval_time_ms:.0f}ms")
    print(f"Results: {len(response.results)}")
    for i, r in enumerate(response.results, 1):
        print(f"\n  {i}. [{r.score:.3f}] {r.fact.content[:100]}")
        if hasattr(r, "channel_scores") and r.channel_scores:
            for ch, sc in r.channel_scores.items():
                print(f"       {ch}: {sc:.3f}")


# -- Services (no --json — these start long-running processes) -------------


def cmd_mcp(_args: Namespace) -> None:
    """Start the V3 MCP server (stdio transport for IDE integration)."""
    # Auto-install hooks on MCP startup (fast path: ~0.1ms if already current)
    # CRITICAL: No stdout — MCP uses stdio transport, any print corrupts protocol
    try:
        from superlocalmemory.hooks.claude_code_hooks import auto_install_if_needed
        auto_install_if_needed()
    except Exception:
        pass

    from superlocalmemory.mcp.server import server

    server.run(transport="stdio")


def cmd_warmup(_args: Namespace) -> None:
    """Pre-download the embedding model so first use is instant."""
    import superlocalmemory.core.embeddings as _emb_mod

    print("SuperLocalMemory V3 — Embedding Model Warmup")
    print("=" * 50)
    print(f"  Python: {sys.executable}")
    print(f"  Model:  nomic-ai/nomic-embed-text-v1.5 (~500MB)")
    print()

    # Increase timeout for first-time download
    original_timeout = _emb_mod._SUBPROCESS_RESPONSE_TIMEOUT
    _emb_mod._SUBPROCESS_RESPONSE_TIMEOUT = 180  # 3 min for cold start

    try:
        from superlocalmemory.core.config import EmbeddingConfig
        from superlocalmemory.core.embeddings import EmbeddingService

        config = EmbeddingConfig()

        print("Step 1/3: Spawning embedding worker subprocess...")
        svc = EmbeddingService(config)

        if not svc.is_available:
            print("\n[FAIL] Embedding service not available.")
            _warmup_diagnose()
            return

        print("Step 2/3: Loading model (may download ~500MB on first run)...")
        emb = svc.embed("warmup test")

        if emb and len(emb) == config.dimension:
            print("Step 3/3: Verifying embedding output...")
            print(f"\n[PASS] Model ready: {config.model_name} ({config.dimension}-dim)")
            print("Semantic search is fully operational.")
        else:
            print("\n[FAIL] Model loaded but embedding verification failed.")
            _warmup_diagnose()

    except ImportError as exc:
        print(f"\n[FAIL] Missing dependency: {exc}")
        print("Fix: pip install sentence-transformers einops torch")
    except Exception as exc:
        print(f"\n[FAIL] Warmup failed: {exc}")
        _warmup_diagnose()
    finally:
        _emb_mod._SUBPROCESS_RESPONSE_TIMEOUT = original_timeout


def _warmup_diagnose() -> None:
    """Diagnostic helper when warmup fails."""
    print("\nDiagnosing...")
    print(f"  Python executable: {sys.executable}")
    try:
        from sentence_transformers import SentenceTransformer
        print("  sentence-transformers: importable")
        m = SentenceTransformer(
            "nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True, device="cpu",
        )
        v = m.encode(["test"], normalize_embeddings=True)
        print(f"  Direct embed: OK (dim={v.shape[1]})")
        print("\n  Issue: Subprocess worker failed but direct import works.")
        print("  This is likely a Python path mismatch between Node.js wrapper")
        print("  and your current shell. Run: slm doctor")
    except ImportError as ie:
        print(f"  sentence-transformers: NOT importable ({ie})")
        print("  Fix: pip install sentence-transformers einops torch")
    except Exception as de:
        print(f"  Direct embed failed: {de}")
        print("  Run: slm doctor")


def cmd_dashboard(args: Namespace) -> None:
    """Open the web dashboard in the browser.

    v3.4.3: Dashboard is now served by the unified daemon. This command
    ensures the daemon is running and opens the browser. It does NOT
    start a separate server (saves ~500MB RAM from duplicate engine).
    """
    from superlocalmemory.cli.daemon import ensure_daemon, _get_port

    port = getattr(args, "port", None) or _get_port()

    print("  SuperLocalMemory V3 — Web Dashboard")
    print(f"  Starting daemon if needed...")

    if not ensure_daemon():
        print("  ✗ Could not start daemon. Run `slm doctor` to diagnose.")
        sys.exit(1)

    url = f"http://localhost:{port}"
    print(f"  ✓ Daemon running")
    print(f"  Dashboard: {url}")
    print(f"  API Docs:  {url}/docs")

    # Open browser
    import webbrowser
    webbrowser.open(url)
    print("\n  Dashboard opened in browser. Daemon continues running in background.")


# -- Profiles (supports --json) -------------------------------------------


def cmd_profile(args: Namespace) -> None:
    """Profile management (list, switch, create).

    Writes to BOTH SQLite and profiles.json so CLI, Dashboard, and
    MCP all see the same profiles.
    """
    from superlocalmemory.core.config import SLMConfig
    from superlocalmemory.storage.database import DatabaseManager
    from superlocalmemory.storage import schema
    from superlocalmemory.server.routes.helpers import (
        ensure_profile_in_json, set_active_profile_everywhere,
    )

    config = SLMConfig.load()
    db = DatabaseManager(config.db_path)
    db.initialize(schema)

    if getattr(args, 'json', False):
        from superlocalmemory.cli.json_output import json_print
        if args.action == "list":
            rows = db.execute("SELECT profile_id, name FROM profiles")
            profiles = [
                {"profile_id": dict(r)["profile_id"], "name": dict(r).get("name", "")}
                for r in rows
            ]
            json_print("profile", data={"profiles": profiles, "count": len(profiles)},
                       next_actions=[
                           {"command": "slm profile switch <name> --json", "description": "Switch profile"},
                       ])
        elif args.action == "switch":
            set_active_profile_everywhere(args.name)
            config.active_profile = args.name
            config.save()
            json_print("profile", data={"action": "switched", "profile": args.name})
        elif args.action == "create":
            db.execute(
                "INSERT OR IGNORE INTO profiles (profile_id, name) VALUES (?, ?)",
                (args.name, args.name),
            )
            ensure_profile_in_json(args.name)
            json_print("profile", data={"action": "created", "profile": args.name},
                       next_actions=[
                           {"command": f"slm profile switch {args.name} --json",
                            "description": "Switch to new profile"},
                       ])
        return

    if args.action == "list":
        rows = db.execute("SELECT profile_id, name FROM profiles")
        print("Profiles:")
        for r in rows:
            d = dict(r)
            print(f"  - {d['profile_id']}: {d.get('name', '')}")
    elif args.action == "switch":
        set_active_profile_everywhere(args.name)
        config.active_profile = args.name
        config.save()
        print(f"Switched to profile: {args.name}")
    elif args.action == "create":
        db.execute(
            "INSERT OR IGNORE INTO profiles (profile_id, name) VALUES (?, ?)",
            (args.name, args.name),
        )
        ensure_profile_in_json(args.name)
        print(f"Created profile: {args.name}")


# -- Active Memory commands (V3.1) ------------------------------------------


def cmd_init(args: Namespace) -> None:
    """One-command setup: mode + hooks + IDE connect + warmup."""
    from pathlib import Path
    from superlocalmemory.core.config import SLMConfig

    force = getattr(args, "force", False)

    config_exists = (Path.home() / ".superlocalmemory" / "config.json").exists()

    print()
    print("SuperLocalMemory — One-Time Setup")
    print("=" * 40)

    # Step 1: Mode selection (interactive)
    if force or not config_exists:
        print()
        from superlocalmemory.cli.setup_wizard import run_wizard
        run_wizard()
    else:
        config = SLMConfig.load()
        print(f"\n  Already configured: Mode {config.mode.value.upper()}")
        print(f"  Profile: {config.active_profile}")

    # Step 2: Install hooks (gate always OFF by default)
    print()
    print("Installing Claude Code hooks...")
    from superlocalmemory.hooks.claude_code_hooks import install_hooks, check_status

    status = check_status()

    if status["installed"] and not force:
        if status["needs_upgrade"]:
            from superlocalmemory.hooks.claude_code_hooks import upgrade_hooks
            result = upgrade_hooks()
            if result.get("upgraded"):
                print(f"  Hooks upgraded: {result['from_version']} -> {result['to_version']}")
            else:
                print(f"  Upgrade issue: {result.get('reason', result.get('errors', ''))}")
        else:
            print(f"  Hooks already installed (v{status['version']})")
    else:
        result = install_hooks(include_gate=False)
        if result["success"]:
            print(f"  Hooks installed: {', '.join(result['hooks_added'])}")
            print("  SLM: Hooks installed into Claude Code (slm hooks remove to undo)")
        else:
            print(f"  Hook install failed: {result['errors']}")

    # Step 3: IDE connection
    print()
    print("Detecting IDEs...")
    try:
        from superlocalmemory.hooks.ide_connector import IDEConnector
        connector = IDEConnector()
        results = connector.connect_all()
        for ide_id, ide_status in results.items():
            print(f"  {ide_id}: {ide_status}")
    except Exception as exc:
        print(f"  IDE detection skipped: {exc}")

    # Step 4: Warmup (embedding model)
    print()
    print("Checking embedding model...")
    try:
        from superlocalmemory.core.config import SLMConfig as _Cfg
        cfg = _Cfg.load()
        model_name = cfg.embedding.model_name
        print(f"  Model: {model_name}")
        # Quick check: try creating embedding service (auto-downloads if needed)
        from superlocalmemory.core.embeddings import EmbeddingService
        svc = EmbeddingService(cfg.embedding)
        test_result = svc.embed_text("test")
        if test_result is not None and len(test_result) > 0:
            print("  Status: ready")
        else:
            print("  Status: model not available (run: slm warmup)")
    except Exception as exc:
        print(f"  Warmup skipped: {exc}")
        print("  Run 'slm warmup' later to download the embedding model.")

    # Done
    print()
    print("=" * 40)
    print("SLM is active. Your AI now remembers you.")
    print()
    print("What happens next:")
    print("  - Open Claude Code in any project")
    print("  - SLM auto-injects your memory context")
    print("  - Decisions, bugs, preferences are captured automatically")
    print("  - Session summaries saved when you close")
    print()


def cmd_hooks(args: Namespace) -> None:
    """Manage Claude Code hooks for invisible memory injection."""
    from superlocalmemory.hooks.claude_code_hooks import (
        install_hooks, remove_hooks, check_status,
    )

    action = getattr(args, "action", "status")
    # Gate is OFF by default. --gate opts in (for brave users).
    include_gate = getattr(args, "gate", False)

    if action == "install":
        result = install_hooks(include_gate=include_gate)
        if result["success"]:
            print("SLM hooks installed in Claude Code.")
            print(f"  Hook types: {', '.join(result['hooks_added'])}")
            if include_gate:
                print("  Gate: ON (enforces session_init — experimental)")
            print("  SLM: Hooks installed into Claude Code (slm hooks remove to undo)")
        else:
            print(f"Installation failed: {result['errors']}")
    elif action == "remove":
        result = remove_hooks()
        if result["success"]:
            print("SLM hooks removed from Claude Code.")
        else:
            print(f"Removal failed: {result['errors']}")
    else:
        result = check_status()
        if result["installed"]:
            print(f"SLM hooks: INSTALLED (v{result['version']})")
            print(f"  Hook types: {', '.join(result['hook_types'])}")
            print(f"  Gate: {'ON' if result['gate_enabled'] else 'OFF'}")
            if result["needs_upgrade"]:
                print(f"  Update available: {result['version']} -> {result['latest_version']}")
                print("  Run: slm hooks install")
        else:
            print("SLM hooks: NOT INSTALLED")
            print("  Run: slm hooks install")
            print("  Or:  slm init  (full setup)")


def cmd_session_context(args: Namespace) -> None:
    """Print session context (for hook scripts and piping).

    Uses a FAST PATH that queries SQLite directly (no engine/Ollama needed).
    This ensures the SessionStart hook completes within its 15s timeout even
    when Ollama requires a 60s+ cold start.  The fast path returns:
      - Core Memory blocks (always-on context)
      - Recent high-importance memories (last 7 days)
      - Session summary from last session
    Falls back to the full engine path only if --full is passed explicitly.
    """
    import sqlite3
    from pathlib import Path
    from superlocalmemory.core.config import SLMConfig

    use_full = getattr(args, "full", False)

    if use_full:
        # Full engine path (slow, requires Ollama) — for explicit CLI use
        try:
            from superlocalmemory.hooks.auto_recall import AutoRecall
            from superlocalmemory.core.engine import MemoryEngine
            config = SLMConfig.load()
            engine = MemoryEngine(config)
            engine.initialize()
            auto = AutoRecall(
                engine=engine,
                config={"enabled": True, "max_memories_injected": 10, "relevance_threshold": 0.3},
            )
            context = auto.get_session_context(
                query=getattr(args, "query", "") or "recent decisions and important context",
            )
            if context:
                print(context)
        except Exception as exc:
            logger.debug("session-context (full) failed: %s", exc)
        return

    # ── FAST PATH: direct SQLite, no engine, <500ms ──────────────
    try:
        config = SLMConfig.load()
        db_path = config.base_dir / "memory.db"
        if not db_path.exists():
            return

        pid = config.active_profile
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        sections = []

        # 1. Core Memory blocks (compiled high-value context)
        try:
            rows = conn.execute(
                "SELECT block_type, content FROM core_memory_blocks "
                "WHERE profile_id = ? ORDER BY block_type",
                (pid,),
            ).fetchall()
            if rows:
                blocks = [f"[{r['block_type']}] {r['content']}" for r in rows]
                sections.append("## Core Memory\n" + "\n".join(blocks))
        except sqlite3.OperationalError:
            pass

        # 2. Recent important memories (last 7 days, top 10 by importance)
        try:
            rows = conn.execute(
                "SELECT content, fact_type, created_at FROM atomic_facts "
                "WHERE profile_id = ? "
                "AND created_at >= datetime('now', '-7 days') "
                "AND lifecycle = 'active' "
                "ORDER BY importance DESC, created_at DESC LIMIT 10",
                (pid,),
            ).fetchall()
            if rows:
                items = []
                for r in rows:
                    content = r["content"][:200]
                    items.append(f"- [{r['fact_type'] or 'fact'}] {content}")
                sections.append("## Recent Context (7 days)\n" + "\n".join(items))
        except sqlite3.OperationalError:
            pass

        # 3. Session markers (last session summary)
        try:
            rows = conn.execute(
                "SELECT content, created_at FROM atomic_facts "
                "WHERE profile_id = ? AND content LIKE 'Session%' "
                "ORDER BY created_at DESC LIMIT 3",
                (pid,),
            ).fetchall()
            if rows:
                items = [f"- {r['content'][:150]}" for r in rows]
                sections.append("## Recent Sessions\n" + "\n".join(items))
        except sqlite3.OperationalError:
            pass

        # 4. V3.3 Soft prompts (auto-learned patterns)
        try:
            rows = conn.execute(
                "SELECT category, content FROM soft_prompt_templates "
                "WHERE profile_id = ? AND active = 1 "
                "ORDER BY confidence DESC LIMIT 5",
                (pid,),
            ).fetchall()
            if rows:
                items = [f"- [{r['category']}] {r['content'][:150]}" for r in rows]
                sections.append("## Learned Patterns\n" + "\n".join(items))
        except sqlite3.OperationalError:
            pass

        conn.close()

        if sections:
            header = f"# SLM Session Context — {config.active_profile}"
            print(header + "\n\n" + "\n\n".join(sections))
    except Exception as exc:
        logger.debug("session-context (fast) failed: %s", exc)


def cmd_observe(args: Namespace) -> None:
    """Evaluate and auto-capture content from stdin or argument.

    V3.3.28: Routes through daemon to prevent embedding worker memory blast.
    Previously each `slm observe` spawned its own MemoryEngine + embedding
    worker (~1.4 GB each). With 20 parallel edits = 28+ GB = system crash.
    Now uses the daemon's singleton engine (1 worker total).
    """
    import sys

    content = getattr(args, "content", "") or ""
    if not content and not sys.stdin.isatty():
        content = sys.stdin.read().strip()

    if not content:
        print("No content to observe.")
        return

    # V3.3.28: Route through daemon (singleton engine, single embedding worker).
    # This is the P0 fix for the memory blast incident of April 7, 2026.
    try:
        from superlocalmemory.cli.daemon import is_daemon_running, daemon_request, ensure_daemon
        if is_daemon_running() or ensure_daemon():
            result = daemon_request("POST", "/observe", {"content": content})
            if result is not None:
                if result.get("captured"):
                    cat = result.get("category", "unknown")
                    conf = result.get("confidence", 0)
                    print(f"Auto-captured: {cat} (confidence: {conf:.2f}) (via daemon)")
                else:
                    reason = result.get("reason", "no patterns matched")
                    print(f"Not captured: {reason}")
                return
    except Exception:
        pass  # Fall through to direct engine

    # Fallback: direct engine (only if daemon unavailable).
    # Acquires a system-wide file lock to prevent concurrent worker spawns.
    try:
        from superlocalmemory.hooks.auto_capture import AutoCapture
        from superlocalmemory.core.config import SLMConfig
        from superlocalmemory.core.engine import MemoryEngine
        from superlocalmemory.core.embeddings import acquire_embedding_lock

        if not acquire_embedding_lock():
            logger.debug("observe: another embedding worker active, skipping")
            print("Not captured: system busy (another embedding in progress)")
            return

        try:
            config = SLMConfig.load()
            engine = MemoryEngine(config)
            engine.initialize()

            auto = AutoCapture(engine=engine)
            decision = auto.evaluate(content)

            if decision.capture:
                stored = auto.capture(content, category=decision.category)
                if stored:
                    print(f"Auto-captured: {decision.category} (confidence: {decision.confidence:.2f})")
                else:
                    print(f"Detected {decision.category} but store failed.")
            else:
                print(f"Not captured: {decision.reason}")
        finally:
            from superlocalmemory.core.embeddings import release_embedding_lock
            release_embedding_lock()
    except Exception as exc:
        logger.debug("observe failed: %s", exc)


# -- V3.3 Commands -----------------------------------------------------------


def cmd_decay(args: Namespace) -> None:
    """Run Ebbinghaus forgetting decay cycle."""
    from superlocalmemory.core.config import SLMConfig
    from superlocalmemory.core.engine import MemoryEngine

    use_json = getattr(args, "json", False)
    dry_run = getattr(args, "dry_run", True)
    profile = getattr(args, "profile", "")

    try:
        config = SLMConfig.load()
        engine = MemoryEngine(config)
        engine.initialize()
        pid = profile or engine.profile_id

        from superlocalmemory.math.ebbinghaus import EbbinghausCurve
        from superlocalmemory.learning.forgetting_scheduler import (
            ForgettingScheduler,
        )

        ebbinghaus = EbbinghausCurve(config.forgetting)
        scheduler = ForgettingScheduler(
            engine._db, ebbinghaus, config.forgetting,
        )
        result = scheduler.run_decay_cycle(pid, force=True)
    except Exception as exc:
        if use_json:
            from superlocalmemory.cli.json_output import json_print
            json_print("decay", error={"code": "DECAY_ERROR", "message": str(exc)})
            sys.exit(1)
        raise

    if use_json:
        from superlocalmemory.cli.json_output import json_print
        json_print("decay", data={"dry_run": dry_run, **result},
                   next_actions=[
                       {"command": "slm decay --execute --json", "description": "Apply transitions"},
                       {"command": "slm status --json", "description": "Check system status"},
                   ])
        return

    if result.get("skipped"):
        print(f"Skipped: {result.get('reason', 'unknown')}")
        return

    total = result.get("total", 0)
    print(f"Decay cycle complete (dry_run={dry_run})")
    print(f"  Total facts:  {total}")
    print(f"  Active:       {result.get('active', 0)}")
    print(f"  Warm:         {result.get('warm', 0)}")
    print(f"  Cold:         {result.get('cold', 0)}")
    print(f"  Archive:      {result.get('archive', 0)}")
    print(f"  Forgotten:    {result.get('forgotten', 0)}")
    print(f"  Transitions:  {result.get('transitions', 0)}")


def cmd_quantize(args: Namespace) -> None:
    """Run EAP embedding quantization cycle."""
    from superlocalmemory.core.config import SLMConfig
    from superlocalmemory.core.engine import MemoryEngine

    use_json = getattr(args, "json", False)
    dry_run = getattr(args, "dry_run", True)
    profile = getattr(args, "profile", "")

    try:
        config = SLMConfig.load()
        engine = MemoryEngine(config)
        engine.initialize()
        pid = profile or engine.profile_id

        from superlocalmemory.math.ebbinghaus import EbbinghausCurve
        from superlocalmemory.dynamics.eap_scheduler import EAPScheduler
        from superlocalmemory.storage.quantized_store import (
            QuantizedEmbeddingStore,
        )

        from superlocalmemory.math.polar_quant import PolarQuantEncoder
        from superlocalmemory.math.qjl import QJLEncoder

        ebbinghaus = EbbinghausCurve(config.forgetting)
        polar = PolarQuantEncoder(config.quantization.polar)
        qjl = QJLEncoder(config.quantization.qjl)
        qstore = QuantizedEmbeddingStore(
            engine._db, polar, qjl, config.quantization,
        )
        scheduler = EAPScheduler(
            engine._db, ebbinghaus, qstore, config.quantization,
        )
        result = scheduler.run_eap_cycle(pid)
    except Exception as exc:
        if use_json:
            from superlocalmemory.cli.json_output import json_print
            json_print("quantize", error={"code": "EAP_ERROR", "message": str(exc)})
            sys.exit(1)
        raise

    if use_json:
        from superlocalmemory.cli.json_output import json_print
        json_print("quantize", data={"dry_run": dry_run, **result},
                   next_actions=[
                       {"command": "slm quantize --execute --json", "description": "Apply changes"},
                       {"command": "slm status --json", "description": "Check status"},
                   ])
        return

    print(f"EAP quantization cycle (dry_run={dry_run})")
    print(f"  Total:       {result.get('total', 0)}")
    print(f"  Downgrades:  {result.get('downgrades', 0)}")
    print(f"  Upgrades:    {result.get('upgrades', 0)}")
    print(f"  Skipped:     {result.get('skipped', 0)}")
    print(f"  Errors:      {result.get('errors', 0)}")


def cmd_consolidate(args: Namespace) -> None:
    """Run cognitive consolidation pipeline."""
    from superlocalmemory.core.config import SLMConfig
    from superlocalmemory.core.engine import MemoryEngine

    use_json = getattr(args, "json", False)
    cognitive = getattr(args, "cognitive", False)
    dry_run = getattr(args, "dry_run", False)
    profile = getattr(args, "profile", "")

    if not cognitive:
        if use_json:
            from superlocalmemory.cli.json_output import json_print
            json_print("consolidate", error={
                "code": "MISSING_FLAG",
                "message": "Use --cognitive to run CCQ pipeline",
            })
            sys.exit(1)
        print("Use --cognitive to run CCQ consolidation pipeline.")
        print("  slm consolidate --cognitive")
        return

    try:
        config = SLMConfig.load()
        engine = MemoryEngine(config)
        engine.initialize()
        pid = profile or engine.profile_id

        from superlocalmemory.encoding.cognitive_consolidator import (
            CognitiveConsolidator,
        )

        consolidator = CognitiveConsolidator(db=engine._db)
        result = consolidator.run_pipeline(pid, dry_run=dry_run)
    except Exception as exc:
        if use_json:
            from superlocalmemory.cli.json_output import json_print
            json_print("consolidate", error={
                "code": "CCQ_ERROR", "message": str(exc),
            })
            sys.exit(1)
        raise

    if use_json:
        from superlocalmemory.cli.json_output import json_print
        json_print("consolidate", data={
            "clusters_processed": result.clusters_processed,
            "blocks_created": result.blocks_created,
            "facts_archived": result.facts_archived,
            "compression_ratio": round(result.compression_ratio, 3),
        }, next_actions=[
            {"command": "slm list --json", "description": "List recent memories"},
            {"command": "slm status --json", "description": "Check status"},
        ])
        return

    print("CCQ Cognitive Consolidation")
    print(f"  Clusters processed: {result.clusters_processed}")
    print(f"  Blocks created:     {result.blocks_created}")
    print(f"  Facts archived:     {result.facts_archived}")
    print(f"  Compression ratio:  {result.compression_ratio:.3f}")


def cmd_soft_prompts(args: Namespace) -> None:
    """List active soft prompts (auto-learned user patterns)."""
    from superlocalmemory.core.config import SLMConfig
    from superlocalmemory.core.engine import MemoryEngine

    use_json = getattr(args, "json", False)
    profile = getattr(args, "profile", "")

    try:
        config = SLMConfig.load()
        engine = MemoryEngine(config)
        engine.initialize()
        pid = profile or engine.profile_id

        rows = engine._db.execute(
            "SELECT prompt_id, category, content, confidence, "
            "  effectiveness, token_count, version, created_at "
            "FROM soft_prompt_templates "
            "WHERE profile_id = ? AND active = 1 "
            "ORDER BY confidence DESC",
            (pid,),
        )
        prompts = []
        for row in rows:
            r = dict(row)
            prompts.append({
                "prompt_id": r["prompt_id"],
                "category": r["category"],
                "content": r["content"],
                "confidence": round(float(r["confidence"]), 3),
                "effectiveness": round(float(r["effectiveness"]), 3),
                "token_count": int(r["token_count"]),
                "version": int(r["version"]),
                "created_at": r["created_at"],
            })
    except Exception as exc:
        if use_json:
            from superlocalmemory.cli.json_output import json_print
            json_print("soft-prompts", error={
                "code": "QUERY_ERROR", "message": str(exc),
            })
            sys.exit(1)
        raise

    if use_json:
        from superlocalmemory.cli.json_output import json_print
        json_print("soft-prompts", data={
            "prompts": prompts, "count": len(prompts), "profile": pid,
        }, next_actions=[
            {"command": "slm status --json", "description": "Check status"},
        ])
        return

    if not prompts:
        print("No active soft prompts.")
        return

    print(f"Active soft prompts ({len(prompts)}):\n")
    for i, p in enumerate(prompts, 1):
        print(f"  {i}. [{p['category']}] (conf={p['confidence']:.2f})")
        content_preview = p["content"][:100]
        if len(p["content"]) > 100:
            content_preview += "..."
        print(f"     {content_preview}")


def cmd_reap(args: Namespace) -> None:
    """Find and kill orphaned SLM processes."""
    use_json = getattr(args, "json", False)
    dry_run = not getattr(args, "force", False)

    try:
        from superlocalmemory.infra.process_reaper import (
            cleanup_all_orphans,
            ReaperConfig,
        )

        config = ReaperConfig()
        result = cleanup_all_orphans(config, dry_run=dry_run)
    except Exception as exc:
        if use_json:
            from superlocalmemory.cli.json_output import json_print
            json_print("reap", error={
                "code": "REAP_ERROR", "message": str(exc),
            })
            sys.exit(1)
        raise

    if use_json:
        from superlocalmemory.cli.json_output import json_print
        json_print("reap", data={
            "dry_run": dry_run,
            "total_found": result.get("total_found", 0),
            "orphans_found": result.get("orphans_found", 0),
            "killed": result.get("killed", 0),
            "skipped": result.get("skipped", 0),
        }, next_actions=[
            {"command": "slm reap --force --json", "description": "Kill orphans"},
            {"command": "slm status --json", "description": "Check status"},
        ])
        return

    total = result.get("total_found", 0)
    orphans = result.get("orphans_found", 0)
    killed = result.get("killed", 0)
    skipped = result.get("skipped", 0)

    if dry_run:
        print(f"Process reaper (dry run)")
    else:
        print(f"Process reaper")
    print(f"  Total SLM processes: {total}")
    print(f"  Orphans found:       {orphans}")
    print(f"  Killed:              {killed}")
    print(f"  Skipped:             {skipped}")
    if dry_run and orphans > 0:
        print("\n  Use --force to kill orphaned processes.")
