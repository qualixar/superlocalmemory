# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""CLI command implementations.

Each function handles one CLI command. Dispatch routes by name.
All data-returning commands support --json for agent-native output.

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

import logging
import os
import sys
from argparse import Namespace
from pathlib import Path

logger = logging.getLogger(__name__)


def _cmd_db_dispatch(args: Namespace) -> None:
    """Route ``slm db ...`` subcommands. LLD-06 §7.2."""
    sub = getattr(args, "db_command", None)
    if sub == "migrate":
        from superlocalmemory.cli.db_migrate import cmd_db_migrate
        rc = cmd_db_migrate(args)
        if rc:
            sys.exit(rc)
        return
    if sub == "scale":
        from superlocalmemory.cli.scale_engine_cmd import cmd_db_scale
        rc = cmd_db_scale(args)
        if rc:
            sys.exit(rc)
        return
    print("Usage: slm db migrate [--status] [--dry-run] | slm db scale <action>")
    sys.exit(2)


def _cmd_mesh_dispatch(args: Namespace) -> None:
    """Route ``slm mesh ...`` inspection subcommands (M-03)."""
    from superlocalmemory.cli.mesh_cmd import cmd_mesh
    rc = cmd_mesh(args)
    if rc:
        sys.exit(rc)


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


# ---- SLM v3.6 Optimize dispatch functions (additive) ----

def _cmd_optimize(args: Namespace) -> None:
    from superlocalmemory.cli.optimize_cmd import cmd_optimize
    cmd_optimize(args)


def _cmd_cache(args: Namespace) -> None:
    from superlocalmemory.cli.cache_cmd import cmd_cache
    cmd_cache(args)


def _cmd_compress(args: Namespace) -> None:
    from superlocalmemory.cli.compress_cmd import cmd_compress
    cmd_compress(args)


def _cmd_proxy(args: Namespace) -> None:
    from superlocalmemory.cli.proxy_cmd import cmd_proxy
    cmd_proxy(args)


def _cmd_help_optimize(args: Namespace) -> None:
    from superlocalmemory.cli.help_cmd import cmd_help_optimize
    cmd_help_optimize(args)


# ---- end SLM v3.6 Optimize dispatch functions ----


def cmd_session(args: Namespace) -> None:
    """#49: Open/close a session locally via the daemon — no model roundtrip,
    so a shell hook (e.g. Claude session-start / /quit) can call it directly.
    """
    from superlocalmemory.cli.daemon import (
        daemon_request, ensure_daemon, is_daemon_running,
    )

    action = getattr(args, "session_command", None)
    if action not in ("open", "close"):
        print("Usage: slm session {open|close} "
              "[--session-id ID] [--project-path PATH] [--query Q]")
        return

    if not is_daemon_running():
        ensure_daemon()

    if action == "open":
        body = {
            "project_path": getattr(args, "project_path", "") or "",
            "query": getattr(args, "query", "") or "",
            "max_results": int(getattr(args, "max_results", 10) or 10),
        }
        resp = daemon_request("POST", "/session/open", body)
        if resp and resp.get("ok"):
            print(f"Session opened — warmed {resp.get('warmed', 0)} memories "
                  f"(query: {resp.get('query', '')})")
        else:
            print("Session open failed (daemon unreachable?)")
        return

    # close
    body = {"session_id": getattr(args, "session_id", "") or ""}
    resp = daemon_request("POST", "/session/close", body)
    if resp and resp.get("ok"):
        sid = resp.get("session_id") or "(most recent)"
        print(f"Session closed: {sid} — "
              f"{resp.get('summary_events_created', 0)} summary event(s) created")
    else:
        print("Session close failed (daemon unreachable?)")


def dispatch(args: Namespace) -> None:
    """Route CLI command to the appropriate handler."""
    # Auto-install/upgrade hooks on version change (single file read, ~0.1ms)
    if args.command not in ("hooks", "codex", "init", "mcp"):
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
        "search": cmd_recall,  # v3.6.12 (parity-3): MCP exposes a `search` verb; give the CLI parity (recall is multi-channel incl. BM25/keyword).
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
        "codex": cmd_codex,
        "session-context": cmd_session_context,
        "session": cmd_session,  # #49: local session open/close for hooks
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
        # V3.4.22 LLD-05 context pre-staging
        "context": _cmd_context_dispatch,
        # V3.4.22 LLD-06 additive schema migrations
        "db": _cmd_db_dispatch,
        # V3.7.9 M-03 — terminal mesh inspection
        "mesh": _cmd_mesh_dispatch,
        # V3.4.22 Stage 8 SB-5 — MASTER-PLAN §8 escape hatches.
        "disable": _cmd_escape_disable,
        "enable": _cmd_escape_enable,
        "clear-cache": _cmd_escape_clear_cache,
        "reconfigure": _cmd_escape_reconfigure,
        "benchmark": _cmd_escape_benchmark,
        "rotate-token": _cmd_escape_rotate_token,
        "evidence": _cmd_evidence,
        "diagnostics": _cmd_diagnostics,
        # LLD-06 — `slm wrap <agent> [args...]` activates the Optimize proxy.
        "wrap": _cmd_wrap,
        # V3.6 Optimize subcommands (additive)
        "optimize": _cmd_optimize,
        "cache": _cmd_cache,
        "compress": _cmd_compress,
        "proxy": _cmd_proxy,
        "help-optimize": _cmd_help_optimize,
    }
    handler = handlers.get(args.command)
    if handler:
        handler(args)
    else:
        print(f"Unknown command: {args.command}")
        sys.exit(1)


def _cmd_evidence(args: Namespace) -> None:
    """Lazy-load the evidence/rebuild command surface."""
    from superlocalmemory.cli.evidence_cmd import cmd_evidence

    cmd_evidence(args)


def _cmd_diagnostics(args: Namespace) -> None:
    """Lazy-load the local aggregate diagnostics export surface."""
    from superlocalmemory.cli.diagnostics_cmd import cmd_diagnostics

    cmd_diagnostics(args)


def _cmd_wrap(args: Namespace) -> None:
    """LLD-06 §6.6 — `slm wrap <agent> [args...]` activates the Optimize proxy.

    Routes the Optimize layer's per-agent launch/activation. See
    optimize.adapters._agent_registry for the full agent table.
    """
    if getattr(args, "list", False):
        from superlocalmemory.optimize.adapters.wrap import list_agents
        from superlocalmemory.optimize.adapters._agent_registry import AGENT_REGISTRY
        print("Registered agents for `slm wrap`:")
        for key in list_agents():
            spec = AGENT_REGISTRY.get(key, {})
            mech = spec.get("mechanism", "unknown")
            print(f"  {key:20s}  mechanism={mech}")
        return

    agent = getattr(args, "agent", None)
    if not agent:
        print("Usage: slm wrap <agent> [args...]\n"
              "       slm wrap --list\n"
              "       slm wrap --help")
        sys.exit(2)

    agent_args = list(getattr(args, "agent_args", []) or [])
    persistent = bool(getattr(args, "persistent", False))
    dry_run = bool(getattr(args, "dry_run", False))

    from superlocalmemory.optimize.adapters.wrap import wrap_agent
    rc = wrap_agent(agent, agent_args, persistent=persistent, dry_run=dry_run)
    if rc:
        sys.exit(rc)


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
        from superlocalmemory.infra.data_root import state_path
        print(f"Failed to start daemon. Check {state_path('logs', 'daemon.log')}")
        # INT-H-02: exit non-zero so callers can detect the failure. The plugin
        # launcher (plugin/scripts/slm-launch) guards on this exit code and
        # refuses to open a direct MCP writer against a broken daemon; without
        # the non-zero exit that guard was a dead no-op.
        sys.exit(1)


# -- Ingestion Adapters (V3.4.3) ------------------------------------------


def cmd_restart(args: Namespace) -> None:
    """Restart the one daemon owned by the current SLM data namespace.

    5-step pipeline:
      1. Capability-stop the owned daemon and its children
      2. Acquire the namespace start lock
      3. Start fresh daemon
      4. Wait for engine warmup + verify health
      5. Optionally open dashboard
    """
    import time
    from superlocalmemory.infra.daemon_identity import canonical_data_root

    use_json = getattr(args, "json", False)
    open_dashboard = getattr(args, "dashboard", False)
    slm_dir = canonical_data_root()
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

    # Acquire the namespace lock before requesting shutdown.  Otherwise an
    # auto-starting hook can observe the brief offline window and start a
    # second daemon while this command is still waiting for the old process.
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

    # Step 1: stop only the descriptor-owned daemon. Its graceful shutdown
    # owns worker termination; process-name-wide scans are forbidden.
    from superlocalmemory.cli.daemon import (
        is_daemon_running,
        read_descriptor,
        stop_daemon,
        wait_for_owned_daemon_shutdown,
    )

    was_running = is_daemon_running()
    owned_descriptor = read_descriptor() if was_running else None
    stopped = stop_daemon() if was_running else True
    if stopped and was_running:
        stopped = wait_for_owned_daemon_shutdown(owned_descriptor)
    killed = 1 if was_running and stopped else 0
    _log(
        1,
        "Stop owned SLM daemon",
        "ok" if stopped else "fail",
        "owned daemon stopped" if killed else (
            "already stopped" if stopped else "owned daemon did not stop"
        ),
    )
    if not stopped:
        if restart_lock_fd:
            restart_lock_fd.close()
        if use_json:
            from superlocalmemory.cli.json_output import json_print
            json_print(
                "restart",
                data={"steps": steps, "success": False},
                next_actions=[{
                    "command": "slm doctor",
                    "description": "Diagnose the owned daemon",
                }],
            )
        else:
            print("\n  Restart FAILED at step 1. The owned daemon was not stopped.")
        return

    # Step 2: the namespace lock was acquired before shutdown so hooks cannot
    # auto-start another daemon inside the offline transition.
    _log(2, "Acquire namespace start lock", "ok", str(_LOCK_FILE))

    # Step 3: Start fresh daemon (lock still held — no races)
    # v3.4.42: Call _start_daemon_subprocess() directly instead of
    # ensure_daemon(). The latter tries to acquire daemon.lock itself,
    # which the SAME PROCESS holds via restart_lock_fd above — BSD-style
    # flock blocks per-fd even within one process, so ensure_daemon would
    # fall into its lock-fail branch and time out after 60s while the
    # actual daemon never gets started. Calling the helper directly
    # bypasses that self-deadlock and starts the daemon as intended.
    from superlocalmemory.cli.daemon import _start_daemon_subprocess
    started = _start_daemon_subprocess()

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

    from superlocalmemory.infra.data_root import state_path

    use_json = getattr(args, "json", False)
    action = getattr(args, "action", "get")
    key = getattr(args, "key", "")
    value = getattr(args, "value", None)

    config_path = state_path("config.json")

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
            "evolution.mutation_model", "evolution.verify_model",
            "evolution.confirm_model",
            "mesh_enabled", "daemon_idle_timeout", "entity_compilation_enabled",
            "graph_backend", "vector_backend", "scale_engine_state",
            "scope.default_scope", "scope.recall_include_global",
            "scope.recall_include_shared",
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

        if key == "scope.default_scope" and parsed_value not in {
            "personal", "shared", "global",
        }:
            message = "scope.default_scope must be personal, shared, or global"
            if use_json:
                from superlocalmemory.cli.json_output import json_print
                json_print("config", error={
                    "code": "INVALID_VALUE", "message": message,
                })
            else:
                print(f"Error: {message}")
            sys.exit(1)
        if key in {
            "scope.recall_include_global", "scope.recall_include_shared",
        } and not isinstance(parsed_value, bool):
            message = f"{key} must be true or false"
            if use_json:
                from superlocalmemory.cli.json_output import json_print
                json_print("config", error={
                    "code": "INVALID_VALUE", "message": message,
                })
            else:
                print(f"Error: {message}")
            sys.exit(1)

        _MODEL_KEYS = {
            "evolution.mutation_model", "evolution.verify_model",
            "evolution.confirm_model",
        }
        if key in _MODEL_KEYS:
            from superlocalmemory.evolution.model_selection import _MODEL_ALIASES

            accepted = set(_MODEL_ALIASES) | {"", "auto"}
            sval = str(parsed_value)
            if sval not in accepted:
                allowed = ", ".join(["auto", *sorted(_MODEL_ALIASES)])
                message = f"{key} must be one of: {allowed}"
                if use_json:
                    from superlocalmemory.cli.json_output import json_print
                    json_print("config", error={
                        "code": "INVALID_VALUE", "message": message,
                    })
                else:
                    print(f"Error: {message}")
                sys.exit(1)
            # Normalise the "auto" sentinel to the empty string the
            # resolver treats as "pick the cheapest for the backend".
            if sval == "auto":
                parsed_value = ""

        if key.startswith("scope."):
            from superlocalmemory.cli.daemon import daemon_request, is_daemon_running

            if is_daemon_running():
                field = key.split(".", 1)[1]
                result = daemon_request(
                    "PUT", "/api/v3/scope/config", {field: parsed_value},
                )
                if not isinstance(result, dict) or result.get("success") is not True:
                    message = "resident daemon rejected the scope configuration"
                    if use_json:
                        from superlocalmemory.cli.json_output import json_print
                        json_print("config", error={
                            "code": "CONFIG_APPLY_FAILED", "message": message,
                        })
                    else:
                        print(f"Error: {message}")
                    sys.exit(1)
                old_value = None
                if use_json:
                    from superlocalmemory.cli.json_output import json_print
                    json_print("config", data={
                        "key": key, "old_value": old_value,
                        "new_value": result.get(field), "runtime": "daemon",
                    })
                else:
                    print(f"{key}: applied to resident daemon -> {result.get(field)}")
                return

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
            if key == "evolution.enabled" and parsed_value is True:
                print(
                    "\n⚠  Skill evolution is now ON. It makes background "
                    "LLM calls during consolidation\n"
                    "   (capped at 10 calls/cycle, 3 cycles/day). It defaults "
                    "to the lowest-cost model\n"
                    "   for your backend (Claude → Haiku, Ollama → "
                    "local/free).\n"
                    "   Pick models: slm config set evolution.mutation_model "
                    "<auto|haiku|sonnet|ollama>\n"
                    "   Turn off:    slm config set evolution.enabled false"
                )

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

    from superlocalmemory.infra.data_root import state_path

    session_id = getattr(args, "session", "") or ""
    profile = getattr(args, "profile", "default") or "default"

    if not session_id:
        return  # Silent exit — nothing to do without a session

    # Check if evolution is enabled via config.json
    config_path = state_path("config.json")
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

        db_path = state_path("memory.db")
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
    """Get or set the operating mode.

    v3.4.43 behavior change: switching modes via this CLI now PRESERVES the
    user's existing embedding, retrieval, evolution, forgetting, and math
    settings. Previously the CLI called ``SLMConfig.for_mode(...)`` which
    re-derived every field from mode defaults — silently clobbering user
    customizations (e.g. a tuned cross-encoder model, a custom embedding
    endpoint, or custom forgetting half-lives). The v3.4.34 ``mode_change=True``
    guard only protected the ``mode`` field itself; everything else was lost.

    New rules:
      - Only ``config.mode`` changes.
      - If the user has NO LLM provider configured AND is switching to a mode
        that typically needs one (B or C), mode-appropriate LLM defaults are
        populated to avoid the daemon coming up dead. Existing LLM config
        is preserved as-is.
      - Embedding / retrieval / evolution / forgetting / math: untouched.
    """
    from superlocalmemory.core.config import SLMConfig
    from superlocalmemory.storage.models import Mode

    config = SLMConfig.load()

    if getattr(args, 'json', False):
        from superlocalmemory.cli.json_output import json_print
        if args.value:
            old_mode = config.mode.value.upper()
            updated = SLMConfig.switch_mode(args.value)
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
        updated = SLMConfig.switch_mode(args.value)
        print(f"Mode set to: {args.value.upper()}")
        print(f"  Embedding: {updated.embedding.provider}/{updated.embedding.model_name}")
        if args.value.lower() != "a":
            print(f"  LLM: {updated.llm.provider}/{updated.llm.model}")
        print(f"  Reranker: ONNX cross-encoder (enabled)")

        # V3.3.4: Warn if Mode C lacks cloud API key
        if args.value == "c" and not updated.llm.api_key:
            print("  ⚠ Mode C requires a cloud API key. Run: slm provider set")
        print("  ℹ Run `slm restart` to apply the new mode.")
    else:
        print(f"Current mode: {config.mode.value.upper()}")


def cmd_provider(args: Namespace) -> None:
    """Get or set the LLM provider."""
    from superlocalmemory.core.config import SLMConfig

    config = SLMConfig.load()

    if args.action == "set":
        from superlocalmemory.cli.setup_wizard import configure_provider

        configure_provider(config, provider_name=getattr(args, "provider", None))
    else:
        print(f"Provider: {config.llm.provider or 'none (Mode A)'}")
        if config.llm.model:
            print(f"Model: {config.llm.model}")


def _cmd_context_dispatch(args: Namespace) -> None:
    """V3.4.22 LLD-05: ``slm context prestage``."""
    from superlocalmemory.cli.context_commands import cmd_context
    cmd_context(args)


def _agents_md_source_factory():
    """Return a callable that reads the WP-05 AGENTS.md content, or None on failure.

    Source: plugin-src/rules/AGENTS.md (relative to package root).
    Gracefully skips if absent — never fails the MCP write.
    """
    from pathlib import Path

    # Resolve relative to the package root (src/superlocalmemory/../../)
    _pkg_root = Path(__file__).resolve().parents[3]
    _agents_src = _pkg_root / "plugin-src" / "rules" / "AGENTS.md"

    def _read() -> str | None:
        if _agents_src.exists():
            return _agents_src.read_text(encoding="utf-8")
        logger.warning(
            "WP-05 AGENTS.md not found at %s — skipping AGENTS.md write", _agents_src
        )
        return None

    return _read


def cmd_connect(args: Namespace) -> None:
    """Configure IDE integrations.

    Dispatch priority (WP-08):
    1. ``slm connect <ide>`` where ide ∈ IDE_MATRIX → portable_kit.connect_ide
       (MCP-wiring + AGENTS.md; includes claude-code short-circuit to WP-06).
    2. ``--cross-platform`` / ``--disable`` → LLD-05 CrossPlatformConnector.
    3. Bare ``slm connect`` / ``--list`` → legacy IDEConnector (markdown-rules).
    """
    ide_arg = getattr(args, "ide", None)

    # WP-08: intercept known IDE_MATRIX ids before legacy branches (CRIT-1)
    if ide_arg is not None:
        from superlocalmemory.hooks.portable_kit import (
            IDE_MATRIX,
            connect_ide,
            supported_ides,
        )

        if ide_arg in IDE_MATRIX:
            here = getattr(args, "here", False)
            profile = getattr(args, "profile", None)
            project = None
            if here:
                import pathlib
                project = pathlib.Path.cwd()

            result = connect_ide(
                ide_arg,
                home=None,
                project=project,
                here=here,
                profile=profile,
                agents_md_source=_agents_md_source_factory(),
            )

            if not result.get("error"):
                from superlocalmemory.infra.local_diagnostics import record_operation

                record_operation("activation", client=ide_arg)

            if getattr(args, "json", False):
                from superlocalmemory.cli.json_output import json_print
                json_print("connect", data=result)
                return

            if result["error"]:
                print(f"Error: {result['error']}", file=sys.stderr)
                print(
                    f"Supported IDEs: {', '.join(supported_ides())}",
                    file=sys.stderr,
                )
                sys.exit(1)

            status_sym = {"wrote": "[+]", "merged": "[~]", "unchanged": "[=]",
                          "skipped": "[s]", "error": "[!]"}.get(
                result["mcp_config"], "[?]"
            )
            print(
                f"{status_sym} {ide_arg}: mcp_config={result['mcp_config']} "
                f"path={result['mcp_path']}"
            )
            print(f"    agents_md={result['agents_md']}")
            return

        # Unknown ide — list supported and exit non-zero
        from superlocalmemory.hooks.portable_kit import supported_ides
        print(
            f"Unknown IDE '{ide_arg}'.\nSupported: {', '.join(supported_ides())}",
            file=sys.stderr,
        )
        sys.exit(1)

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
    # v3.6.15 multi-scope: scope=None means "not specified" → resolve to the
    # configured default_scope (personal) at the daemon / engine boundary.
    # Shared memory is opt-in, so an unset --scope always stays private.
    scope = getattr(args, 'scope', None)
    # v3.6.15: --shared-with is a comma-separated string on the CLI, but the
    # daemon (RememberRequest) and engine expect list[str]. Parse here so an
    # explicit `--shared-with a,b` doesn't 422 at the daemon and silently fall
    # back to a personal write.
    _sw_raw = getattr(args, 'shared_with', None)
    shared_with = (
        [s.strip() for s in _sw_raw.split(",") if s.strip()]
        if isinstance(_sw_raw, str) and _sw_raw.strip() else _sw_raw
    )

    # Both paths use the one owned daemon.  A second local engine for --sync
    # duplicates heavyweight workers and can block for minutes on cold models.
    daemon_owned = False
    try:
        from superlocalmemory.cli.daemon import (
            daemon_request, ensure_daemon, is_daemon_running,
        )
        daemon_owned = is_daemon_running() or ensure_daemon()
        if daemon_owned:
            path = "/remember?wait=true" if sync_mode else "/remember"
            result = daemon_request(
                "POST", path, {
                    "content": args.content,
                    "tags": args.tags or "",
                    "scope": scope,
                    "shared_with": shared_with,
                },
                timeout_seconds=30,
            )
            if result and "fact_ids" in result:
                if use_json:
                    from superlocalmemory.cli.json_output import json_print
                    json_print("remember", data=result)
                else:
                    state = result.get("materialization_state", "queryable")
                    operation_id = result.get("operation_id", "unknown")
                    print(
                        f"{state.capitalize()} \u2713 {result['count']} facts "
                        f"(operation={operation_id})."
                    )
                return
            if sync_mode:
                if use_json:
                    from superlocalmemory.cli.json_output import json_print
                    json_print("remember", error={
                        "code": "SYNC_TIMEOUT",
                        "message": (
                            "Canonical ingestion did not complete within 30s; "
                            "the durable operation remains available for retry."
                        ),
                    })
                else:
                    print(
                        "Synchronous ingestion did not complete within 30s; "
                        "the durable operation remains queued.",
                        file=sys.stderr,
                    )
                sys.exit(1)
    except SystemExit:
        raise
    except Exception:
        if sync_mode and daemon_owned:
            if use_json:
                from superlocalmemory.cli.json_output import json_print
                json_print("remember", error={
                    "code": "SYNC_TIMEOUT",
                    "message": "Owned daemon request failed before completion.",
                })
            sys.exit(1)
        # Receipt-first writes may use the authenticated local fallback when
        # no owned daemon exists.

    from superlocalmemory.core.engine import MemoryEngine

    try:
        config = SLMConfig.load()
        engine = MemoryEngine(config)
        engine.initialize()

        # v3.6.15: resolve an unset scope to the configured default_scope.
        _scope = scope or getattr(getattr(config, "scope", None), "default_scope", "personal")
        from superlocalmemory.core.engine_ingestion import (
            canonical_store,
            local_trusted_actor_id,
        )

        metadata = {"tags": args.tags} if args.tags else {}
        operation = canonical_store(
            engine,
            args.content,
            source_type="cli-sync" if sync_mode else "cli-offline-canonical",
            trusted_actor_id=local_trusted_actor_id("cli"),
            metadata=metadata,
            scope=_scope,
            shared_with=shared_with,
            return_receipt=True,
        )
    except Exception as exc:
        if use_json:
            from superlocalmemory.cli.json_output import json_print
            json_print("remember", error={"code": "STORE_ERROR", "message": str(exc)})
            sys.exit(1)
        raise

    fact_ids = list(operation.fact_ids) if hasattr(operation, "fact_ids") else list(operation)
    operation_data = {
        "fact_ids": fact_ids,
        "count": len(fact_ids),
        "materialization_state": getattr(
            getattr(operation, "state", None), "value", "complete"
        ),
    }
    if getattr(operation, "operation_id", None):
        operation_data["operation_id"] = operation.operation_id

    if use_json:
        from superlocalmemory.cli.json_output import json_print
        json_print("remember", data=operation_data,
                   next_actions=[
                       {"command": "slm recall '<query>' --json", "description": "Search your memories"},
                       {"command": "slm list --json -n 5", "description": "See recent memories"},
                   ])
        return

    print(
        f"Complete \u2713 {len(fact_ids)} facts "
        f"(operation={operation_data.get('operation_id', 'none')})."
    )


def cmd_recall(args: Namespace) -> None:
    """Search memories via the engine — routes through daemon if available."""
    use_json = getattr(args, 'json', False)
    # v3.6.15: None = "not specified" → daemon/engine resolves the configured
    # default (shared-off). Only an explicit --include-global / --no-global
    # produces True/False here.
    include_global = getattr(args, 'include_global', None)
    include_shared = getattr(args, 'include_shared', None)

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
            fast_qs = "&fast=true" if getattr(args, "fast", False) else ""
            # Only send scope flags when the user set them explicitly; absent =
            # let the daemon resolve the configured default. (Never emit
            # "none" — that would parse as a missing/false value.)
            scope_qs = ""
            if include_global is not None:
                scope_qs += f"&include_global={str(include_global).lower()}"
            if include_shared is not None:
                scope_qs += f"&include_shared={str(include_shared).lower()}"
            _window = getattr(args, "window", "") or ""
            window_qs = f"&window={quote(_window)}" if _window else ""
            result = daemon_request(
                "GET",
                f"/recall?q={quote(args.query)}&limit={args.limit}"
                f"&session_id={quote(session_id)}{fast_qs}{scope_qs}{window_qs}",
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
                    print("No confident match."
                          if result.get("no_confident_match")
                          else "No matching memories found.")
                    return
                # Text output
                print(f"SpreadingActivation.search completed via daemon ({result.get('retrieval_time_ms', 0):.0f}ms)")
                for i, r in enumerate(result["results"], 1):
                    print(f"  {i}. [{r['score']:.2f}] {r['content']}")
                return
    except Exception as _exc:  # noqa: BLE001
        logger.warning(
            "Daemon recall failed, falling back to direct engine: %s", _exc
        )

    from superlocalmemory.core.config import SLMConfig
    from superlocalmemory.core.engine import MemoryEngine

    try:
        config = SLMConfig.load()
        engine = MemoryEngine(config)
        engine.initialize()

        response = engine.recall(
            args.query, limit=args.limit,
            fast=getattr(args, "fast", False),
            include_global=include_global,
            include_shared=include_shared,
            window=getattr(args, "window", "") or None,
        )
    except Exception as exc:
        if use_json:
            from superlocalmemory.cli.json_output import json_print
            json_print("recall", error={"code": "RECALL_ERROR", "message": str(exc)})
            sys.exit(1)
        raise

    # v3.6.6: route the direct-fallback path through the SAME shared
    # serializer the daemon uses, so CLI-without-daemon output is identical
    # to CLI/MCP-with-daemon (budget + source discipline + no_confident_match).
    from superlocalmemory.server.recall_serializer import (
        recall_response_metadata,
        serialize_recall_response,
    )
    _rc = getattr(config, "retrieval", None)
    _ser, _no_match = serialize_recall_response(
        response,
        limit=args.limit,
        per_fact_max=getattr(_rc, "recall_per_fact_max_chars", 2400),
        total_max=getattr(_rc, "recall_total_max_chars", 12000),
        full=getattr(args, "full", False),
    )

    if use_json:
        from superlocalmemory.cli.json_output import json_print
        items = []
        for d in _ser:
            items.append(dict(d))
        json_print("recall", data={
            "results": items, "count": len(items),
            "query_type": getattr(response, "query_type", "unknown"),
            "no_confident_match": _no_match,
            **recall_response_metadata(response),
        }, next_actions=[
            {"command": "slm list --json", "description": "List recent memories"},
        ])
        return

    # Record learning signals (CLI path — works without MCP)
    try:
        _cli_record_signals(config, args.query, response.results)
    except Exception:
        pass

    if not _ser:
        print("No confident match." if _no_match else "No memories found.")
        return
    for i, d in enumerate(_ser, 1):
        print(f"  {i}. [relevance {d['relevance_score']:.2f}] {d['content']}")


def _cli_record_signals(config, query, results):
    """Record learning signals from CLI recall (no MCP dependency)."""
    from pathlib import Path

    from superlocalmemory.learning.feedback import FeedbackCollector
    from superlocalmemory.learning.signals import LearningSignals
    configured_root = getattr(config, "base_dir", None)
    if configured_root is not None:
        slm_dir = Path(configured_root)
    else:
        from superlocalmemory.infra.data_root import canonical_data_root
        slm_dir = canonical_data_root()
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
    dry_run = getattr(args, 'dry_run', False)
    raw_query = getattr(args, 'query', None)

    # F3: `query` is optional so `slm forget --dry-run` can preview every memory.
    # Deletion ALWAYS requires an explicit query — a bare `slm forget` must never
    # mass-delete. In dry-run mode a missing query means "match all" (preview).
    if raw_query is None and not dry_run:
        msg = (
            "A query is required to delete. Preview everything with "
            "'slm forget --dry-run', or delete matches with 'slm forget <query>'."
        )
        if use_json:
            from superlocalmemory.cli.json_output import json_print
            json_print("forget", error={"code": "QUERY_REQUIRED", "message": msg})
        else:
            print(msg)
        sys.exit(2)

    query_lower = "" if raw_query is None else raw_query.lower()
    query_label = raw_query if raw_query is not None else "*all*"

    try:
        config = SLMConfig.load()
        engine = MemoryEngine(config)
        engine.initialize()
        facts = engine._db.get_all_facts(engine.profile_id)
        matches = [f for f in facts if query_lower in f.content.lower()]
    except Exception as exc:
        if use_json:
            from superlocalmemory.cli.json_output import json_print
            json_print("forget", error={"code": "ENGINE_ERROR", "message": str(exc)})
            sys.exit(1)
        raise

    def delete_fact_authorized_for_cli(fact_id: str) -> None:
        from superlocalmemory.core.engine_ingestion import local_trusted_actor_id
        from superlocalmemory.core.mutations import delete_fact_authorized

        result = delete_fact_authorized(
            engine,
            fact_id,
            trusted_actor_id=local_trusted_actor_id("cli"),
            source_agent_id="cli",
        )
        if not result.get("ok"):
            raise RuntimeError(result.get("error", "delete failed"))

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
                delete_fact_authorized_for_cli(f.fact_id)
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
                {"command": f"slm forget '{query_label}' --json --yes", "description": "Confirm deletion"},
            ])
        return

    if not matches:
        print(f"No memories matching '{query_label}'")
        return
    print(f"Found {len(matches)} matching memories:")
    for f in matches[:10]:
        print(f"  - {f.fact_id[:8]}... {f.content[:80]}")
    if dry_run:
        print(f"(dry run — {len(matches)} would be deleted)")
        return
    if getattr(args, 'yes', False):
        for f in matches:
            delete_fact_authorized_for_cli(f.fact_id)
        print(f"Deleted {len(matches)} memories.")
        return
    confirm = input(f"Delete {len(matches)} memories? [y/N] ").strip().lower()
    if confirm in ("y", "yes"):
        for f in matches:
            delete_fact_authorized_for_cli(f.fact_id)
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
            from superlocalmemory.core.engine_ingestion import local_trusted_actor_id
            from superlocalmemory.core.mutations import delete_fact_authorized

            delete_fact_authorized(
                engine,
                fact_id,
                trusted_actor_id=local_trusted_actor_id("cli"),
                source_agent_id="cli",
            )
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

    from superlocalmemory.core.engine_ingestion import local_trusted_actor_id
    from superlocalmemory.core.mutations import delete_fact_authorized

    delete_fact_authorized(
        engine,
        fact_id,
        trusted_actor_id=local_trusted_actor_id("cli"),
        source_agent_id="cli",
    )
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
    from superlocalmemory.core.engine_ingestion import local_trusted_actor_id
    from superlocalmemory.core.mutations import update_fact_authorized

    update_fact_authorized(
        engine,
        fact_id,
        new_content,
        trusted_actor_id=local_trusted_actor_id("cli"),
        source_agent_id="cli",
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
    daemon_status = None
    try:
        from superlocalmemory.cli.daemon import (
            daemon_request,
            is_daemon_running,
        )

        if is_daemon_running():
            candidate = daemon_request("GET", "/status")
            if isinstance(candidate, dict) and candidate.get("profile"):
                daemon_status = candidate
    except Exception:
        logger.debug(
            "cmd_status: daemon runtime status unavailable; using offline view",
            exc_info=True,
        )

    if getattr(args, 'json', False):
        from superlocalmemory.cli.json_output import json_print

        if daemon_status is not None:
            data = {
                "mode": str(daemon_status.get("mode", "unknown")).upper(),
                "provider": daemon_status.get("provider", "none"),
                "profile": daemon_status["profile"],
                "base_dir": daemon_status.get("base_dir", str(config.base_dir)),
                "db_path": daemon_status.get("db_path", str(config.db_path)),
                "db_size_mb": float(daemon_status.get("db_size_mb", 0.0)),
                "fact_count": int(daemon_status.get("fact_count", 0)),
                "entity_count": int(daemon_status.get("entity_count", 0)),
                "edge_count": int(daemon_status.get("edge_count", 0)),
                "profile_generation": int(
                    daemon_status.get("profile_generation", 0)
                ),
            }
            json_print("status", data=data, next_actions=[
                {"command": "slm health --json", "description": "Check math layer health"},
                {"command": "slm list --json", "description": "List recent memories"},
            ])
            return

        # WP-02 D8: canonical key set — db_size_mb always present (0.0 if absent).
        db_size_mb = 0.0
        if config.db_path.exists():
            db_size_mb = round(config.db_path.stat().st_size / 1024 / 1024, 2)

        # Open engine for counts (json branch only — LLD Decision B).
        # Fail-open to 0 on any error; status must never crash.
        # Guard on db existence: `slm status --json` must stay observational —
        # opening the engine on a fresh install would create + migrate the db
        # (MemoryEngine.initialize → DatabaseManager mkdir/connect/DDL). A
        # previously read-only command must not acquire a write side-effect.
        fact_count = 0
        entity_count = 0
        edge_count = 0
        eng = None
        if config.db_path.exists():
            try:
                from superlocalmemory.core.engine import MemoryEngine
                from superlocalmemory.core.engine_capabilities import Capabilities
                eng = MemoryEngine(config, capabilities=Capabilities.LIGHT)
                eng.initialize()
                pid = config.active_profile
                fact_count = eng._db.get_fact_count(pid)
                rows = eng._db.execute(
                    "SELECT COUNT(*) AS c FROM canonical_entities WHERE profile_id = ?",
                    (pid,),
                )
                entity_count = int(dict(rows[0])["c"]) if rows else 0
                rows2 = eng._db.execute(
                    "SELECT COUNT(*) AS c FROM graph_edges WHERE profile_id = ?",
                    (pid,),
                )
                edge_count = int(dict(rows2[0])["c"]) if rows2 else 0
            except Exception:
                logger.debug("cmd_status: engine count query failed; using 0s", exc_info=True)
            finally:
                if eng is not None:
                    try:
                        eng.close()
                    except Exception:
                        pass

        data = {
            "mode": config.mode.value.upper(),
            "provider": config.llm.provider or "none",
            "profile": config.active_profile,
            "base_dir": str(config.base_dir),
            "db_path": str(config.db_path),
            "db_size_mb": db_size_mb,
            "fact_count": fact_count,
            "entity_count": entity_count,
            "edge_count": edge_count,
            "profile_generation": 0,
        }
        json_print("status", data=data, next_actions=[
            {"command": "slm health --json", "description": "Check math layer health"},
            {"command": "slm list --json", "description": "List recent memories"},
        ])
        return

    print("SuperLocalMemory V3")
    print(f"  Mode: {config.mode.value.upper()}")
    print(f"  Provider: {config.llm.provider or 'none'}")
    print(
        f"  Profile: "
        f"{daemon_status.get('profile') if daemon_status else config.active_profile}"
    )
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
    from superlocalmemory.core.config import SLMConfig

    use_json = getattr(args, 'json', False)
    try:
        config = SLMConfig.load()
        from superlocalmemory.cli.daemon import is_daemon_running
        if is_daemon_running():
            # A running daemon owns the writable SQLite connections. Opening a
            # second MemoryEngine re-runs schema initialization and can lock
            # the user's database. Health only needs aggregate counts, so use
            # a read-only snapshot connection instead.
            import sqlite3
            db_path = config.db_path
            conn = sqlite3.connect(
                f"file:{db_path}?mode=ro", uri=True, timeout=5,
            )
            try:
                row = conn.execute(
                    "SELECT COUNT(*), "
                    "SUM(CASE WHEN fisher_mean IS NOT NULL THEN 1 ELSE 0 END), "
                    "SUM(CASE WHEN langevin_position IS NOT NULL THEN 1 ELSE 0 END) "
                    "FROM atomic_facts WHERE profile_id = ?",
                    (config.active_profile,),
                ).fetchone()
                total_facts, fisher_count, langevin_count = row or (0, 0, 0)
            finally:
                conn.close()
            facts = [None] * int(total_facts or 0)
            fisher_count = int(fisher_count or 0)
            langevin_count = int(langevin_count or 0)
        else:
            from superlocalmemory.core.engine import MemoryEngine
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
    print(f"  Math layer indexed: {fisher_count}/{len(facts)}")
    print(f"  Lifecycle positioned: {langevin_count}/{len(facts)}")
    print(f"  Mode: {config.mode.value.upper()}")


def _gather_optimize_surface_b() -> dict:
    """Gather Surface-B health data for slm doctor.

    Pure data-gather — never raises, never prints, never starts the
    ConfigStore watchdog thread. Reads daemon-persisted metrics only
    (CacheDB.metrics_load), never the in-process KV counters.

    Returns a dict with keys:
        enabled, cache_enabled, compress_enabled, proxy_enabled,
        compress_runs, tokens_saved, cache_hits, cache_misses,
        db_present, error
    """
    from superlocalmemory.infra.data_root import state_path
    from superlocalmemory.optimize.storage.db import CacheDB

    result: dict = {
        "enabled": False,
        "cache_enabled": False,
        "compress_enabled": False,
        "proxy_enabled": False,
        "compress_runs": 0,
        "tokens_saved": 0,
        "cache_hits": 0,
        "cache_misses": 0,
        "db_present": False,
        "error": "",
    }

    # Step 1: read optimize config — NO watchdog start.
    try:
        from superlocalmemory.optimize.config.store import ConfigStore
        cfg = ConfigStore().get()
        result["enabled"] = cfg.enabled
        result["cache_enabled"] = cfg.cache_enabled
        result["compress_enabled"] = cfg.compress_enabled
        result["proxy_enabled"] = cfg.proxy_enabled
    except Exception as exc:  # noqa: BLE001
        result["error"] = str(exc)
        # enabled stays False — safe default

    # Step 2: read persisted metrics from llmcache.db (daemon-flushed, ≤60s stale).
    try:
        db_path = state_path("llmcache.db")
        result["db_present"] = db_path.exists()
        if result["db_present"]:
            snap = CacheDB.get_default().metrics_load()
            result["compress_runs"] = snap.compress_runs
            result["tokens_saved"] = snap.tokens_saved_compress
            result["cache_hits"] = snap.hits
            result["cache_misses"] = snap.misses
    except Exception as exc:  # noqa: BLE001
        prior = result["error"]
        result["error"] = (prior + "; " if prior else "") + "metrics read failed"

    return result


def _readline_with_timeout(
    stream, timeout_sec: float,
) -> tuple[str | None, Exception | None]:
    """Read one line from a pipe-like stream without POSIX-only select().

    Windows select() only accepts sockets, not subprocess pipes. A bounded
    helper thread keeps the embedding-worker probe cross-platform while
    preserving the existing timeout behavior.
    """
    import threading

    result: dict[str, object] = {}

    def _read() -> None:
        try:
            result["line"] = stream.readline()
        except Exception as exc:  # noqa: BLE001 - returned as probe failure
            result["exc"] = exc

    reader = threading.Thread(target=_read, daemon=True)
    reader.start()
    reader.join(timeout_sec)
    if reader.is_alive():
        return None, None
    line = result.get("line")
    exc = result.get("exc")
    return (
        line if isinstance(line, str) else None,
        exc if isinstance(exc, Exception) else None,
    )


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
        except Exception:  # dependency import may fail after module discovery
            pass
    if len(core_ok) == len(core_modules):
        _check("Core deps", "PASS", ", ".join(core_versions[:4]) + "...")
    else:
        missing = set(core_modules) - set(core_ok)
        _check("Core deps", "FAIL", f"Missing: {', '.join(missing)}",
               "pip install " + " ".join(core_modules[m] for m in missing))

    # 3. Search deps
    search_mods = {"sentence_transformers": "sentence-transformers", "torch": "torch",
                   "sklearn": "scikit-learn"}
    search_ok = []
    for mod, pkg in search_mods.items():
        try:
            __import__(mod)
            search_ok.append(mod)
        except Exception:  # dependency import may fail after module discovery
            pass
    if len(search_ok) == len(search_mods):
        _check("Search deps", "PASS", "sentence-transformers, torch, sklearn")
    else:
        missing = set(search_mods) - set(search_ok)
        _check("Search deps", "WARN", f"Missing: {', '.join(missing)}",
               "pip install 'superlocalmemory[search]'")

    # 4. Dashboard deps
    dash_ok = True
    for mod in ["fastapi", "uvicorn", "websockets"]:
        try:
            __import__(mod)
        except Exception:  # dependency import may fail after module discovery
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
    except Exception:  # dependency import may fail after module discovery
        _check("Learning deps", "WARN", "lightgbm not installed",
               "pip install lightgbm")
    except OSError as exc:
        _check("Learning deps", "WARN", f"lightgbm installed but broken: {exc}",
               "brew install libomp && pip install --force-reinstall lightgbm")

    # 6. Performance deps
    # v3.4.43: diskcache removed from this check — it was a phantom dependency
    # (declared in pyproject.toml but never imported anywhere in src/ or tests/).
    # Dropping it closes CVE-2025-69872 (pickle deserialization RCE) without any
    # behavior change. orjson remains a real performance dep.
    perf_ok = []
    for mod in ["orjson"]:
        try:
            __import__(mod)
            perf_ok.append(mod)
        except Exception:  # dependency import may fail after module discovery
            pass
    if perf_ok:
        _check("Performance deps", "PASS", "orjson")
    else:
        _check("Performance deps", "WARN", "Missing: orjson",
               "pip install orjson")

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

            line, read_exc = _readline_with_timeout(proc.stdout, 30)
            if read_exc is not None:
                _check("Embedding worker", "FAIL", str(read_exc), "slm warmup")
            elif line is not None:
                resp = _json.loads(line or "{}")
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
    from superlocalmemory.infra.data_root import canonical_data_root
    slm_home = canonical_data_root()
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

    # 11. PEP 668 advisory — WP-07: detect EXTERNALLY-MANAGED marker and
    #     recommend pipx when the system Python is managed by the OS package
    #     manager (e.g. Homebrew, Debian/Ubuntu, Fedora 38+).
    try:
        import sysconfig as _sc
        _stdlib = _sc.get_path("stdlib")
        if _stdlib:
            _em_marker = Path(_stdlib) / "EXTERNALLY-MANAGED"
            if _em_marker.exists():
                _check(
                    "PEP 668 / Install method",
                    "WARN",
                    "System Python is externally managed (EXTERNALLY-MANAGED marker found). "
                    "pip install may fail with PEP 668 error.",
                    "Use an isolated install: pipx install superlocalmemory  "
                    "or uv tool install superlocalmemory",
                )
            else:
                _check(
                    "PEP 668 / Install method",
                    "PASS",
                    "No EXTERNALLY-MANAGED marker — standard pip install supported",
                )
    except Exception:
        pass  # advisory only — never fail doctor on this check

    # 12. Optimize (Surface B) — reads daemon-persisted metrics (≤60s stale).
    info = _gather_optimize_surface_b()
    _enabled = info["enabled"]
    _error = info.get("error", "")
    if not _enabled:
        _check(
            "Optimize (Surface B)",
            "WARN",
            "disabled (optimize.json enabled=false) — caching/compression not active"
            + (f" [{_error}]" if _error else ""),
            fix="Enable via dashboard Optimize tab or set enabled=true"
            f" in {slm_home / 'optimize.json'}",
        )
    else:
        _surfaces = []
        if info["cache_enabled"]:
            _surfaces.append("cache")
        if info["compress_enabled"]:
            _surfaces.append("compress")
        if info["proxy_enabled"]:
            _surfaces.append("proxy")
        _stats = (
            f"compress_runs={info['compress_runs']}"
            f" tokens_saved={info['tokens_saved']}"
            f" cache_hits={info['cache_hits']}"
            f" cache_misses={info['cache_misses']}"
        )
        _surface_str = ",".join(_surfaces) if _surfaces else "(none)"
        if not _surfaces:
            _check(
                "Optimize (Surface B)",
                "WARN",
                f"enabled [{_surface_str}] but no surface active"
                + (f" [{_error}]" if _error else ""),
            )
        elif not info["db_present"]:
            _check(
                "Optimize (Surface B)",
                "WARN",
                f"enabled [{_surface_str}] {_stats}"
                " but no metrics yet (llmcache.db not created)"
                + (f" [{_error}]" if _error else ""),
                fix="slm serve start",
            )
        elif _error:
            _check(
                "Optimize (Surface B)",
                "WARN",
                f"enabled [{_surface_str}] {_stats} (partial: {_error})",
            )
        else:
            _check(
                "Optimize (Surface B)",
                "PASS",
                f"enabled [{_surface_str}] {_stats}",
            )

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
    use_json = getattr(args, 'json', False)
    limit = getattr(args, 'limit', 10)

    # Trace must use the same daemon-owned engine as recall. A direct CLI
    # engine cannot attach to the machine-wide embedding worker already owned
    # by the daemon; it then silently loses semantic, Hopfield, and spreading
    # activation channels. The daemon trace route keeps the loaded model,
    # graph, and retrieval state intact while returning the same score detail.
    try:
        from superlocalmemory.cli.daemon import (
            daemon_request, ensure_daemon, is_daemon_running,
        )
        if is_daemon_running() or ensure_daemon():
            result = daemon_request(
                "POST", "/api/v3/recall/trace",
                {"query": args.query, "limit": limit},
            )
            if result and "results" in result:
                if use_json:
                    from superlocalmemory.cli.json_output import json_print
                    json_print("trace", data={
                        "query": result.get("query", args.query),
                        "query_type": result.get("query_type", "unknown"),
                        "retrieval_time_ms": round(
                            float(result.get("retrieval_time_ms", 0)), 1,
                        ),
                        "results": result["results"],
                        "count": len(result["results"]),
                        "no_confident_match": bool(
                            result.get("no_confident_match", False),
                        ),
                        "score_contract_version": result.get(
                            "score_contract_version", "2",
                        ),
                        "calibration_status": result.get(
                            "calibration_status", "uncalibrated",
                        ),
                        "calibration_id": result.get("calibration_id"),
                        "answer_confidence": result.get("answer_confidence"),
                        "abstained": bool(result.get("abstained", False)),
                        "abstention_reason": result.get("abstention_reason"),
                    }, next_actions=[
                        {
                            "command": "slm recall '<query>' --json",
                            "description": "Standard recall",
                        },
                    ])
                    return
                print(f"Query: {result.get('query', args.query)}")
                print(
                    f"Type: {result.get('query_type', 'unknown')} | Time: "
                    f"{float(result.get('retrieval_time_ms', 0)):.0f}ms"
                )
                print(f"Results: {len(result['results'])}")
                for i, item in enumerate(result["results"], 1):
                    print(
                        f"\n  {i}. [relevance "
                        f"{float(item.get('relevance_score', item.get('score', 0))):.3f}] "
                        f"{str(item.get('content', ''))[:100]}"
                    )
                    if item.get("ranking_score") is not None:
                        print(
                            "       ranking utility: "
                            f"{float(item['ranking_score']):.6f}"
                        )
                    for channel, score in (item.get("channel_scores") or {}).items():
                        print(f"       {channel}: {float(score):.3f}")
                return
    except Exception:
        # The direct path remains the offline escape hatch when a daemon is
        # unavailable or a local transport error occurs.
        pass

    from superlocalmemory.core.engine import MemoryEngine
    from superlocalmemory.core.config import SLMConfig
    try:
        config = SLMConfig.load()
        engine = MemoryEngine(config)
        engine.initialize()
        response = engine.recall(args.query, limit=limit)
    except Exception as exc:
        if use_json:
            from superlocalmemory.cli.json_output import json_print
            json_print("trace", error={"code": "ENGINE_ERROR", "message": str(exc)})
            sys.exit(1)
        raise

    if use_json:
        from superlocalmemory.cli.json_output import json_print
        from superlocalmemory.server.recall_serializer import (
            recall_response_metadata,
            serialize_recall_response,
        )
        items, no_confident_match = serialize_recall_response(
            response,
            limit=limit,
            per_fact_max=200,
            total_max=max(200, limit * 200),
        )
        json_print("trace", data={
            "query": args.query,
            "query_type": getattr(response, "query_type", "unknown"),
            "retrieval_time_ms": round(getattr(response, "retrieval_time_ms", 0), 1),
            "results": items, "count": len(items),
            "no_confident_match": no_confident_match,
            **recall_response_metadata(response),
        }, next_actions=[
            {"command": "slm recall '<query>' --json", "description": "Standard recall"},
        ])
        return

    print(f"Query: {args.query}")
    print(f"Type: {response.query_type} | Time: {response.retrieval_time_ms:.0f}ms")
    print(f"Results: {len(response.results)}")
    for i, r in enumerate(response.results, 1):
        print(f"\n  {i}. [relevance {r.relevance_score:.3f}] {r.fact.content[:100]}")
        if r.ranking_score is not None:
            print(f"       ranking utility: {r.ranking_score:.6f}")
        if hasattr(r, "channel_scores") and r.channel_scores:
            for ch, sc in r.channel_scores.items():
                print(f"       {ch}: {sc:.3f}")


# -- Services (no --json — these start long-running processes) -------------


def cmd_mcp(_args: Namespace) -> None:
    """Start the V3 MCP server (stdio transport for IDE integration)."""
    # SINGLETON GUARD (v3.5.8): Reap fresh orphans before binding stdio.
    # Root cause: every IDE session spawns a new `slm mcp`; dead sessions'
    # processes survive indefinitely (~400 MB each) because the reaper's
    # 1-hour age gate misses them. Fix: run find_orphans with age_threshold=0
    # so any `slm mcp` whose parent IDE/Claude process has died is killed
    # immediately. Only orphans (dead parent) are killed — live sessions safe.
    # CRITICAL: No stdout anywhere below — MCP stdio transport, any print
    # corrupts the JSON-RPC protocol.
    try:
        from superlocalmemory.infra.process_reaper import (
            ReaperConfig,
            find_orphans,
            is_mcp_server_process,
            kill_orphan,
        )
        _reaper_cfg = ReaperConfig(orphan_age_threshold_hours=0.0)
        for _orphan in find_orphans(_reaper_cfg):
            # A unified daemon is expected to be detached from the launching
            # shell and can therefore have PPID 1.  The MCP reaper must never
            # treat that healthy shared daemon as an orphaned stdio server.
            if is_mcp_server_process(_orphan):
                kill_orphan(_orphan.pid, graceful_timeout_seconds=1.0)
    except Exception:
        pass  # Never block MCP startup on cleanup failure

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
    """Pre-download the embedding model so first use is instant.

    v3.4.42: daemon-aware. The embedding worker is a machine-wide
    singleton (`_is_embedding_worker_alive` + PID file), so when the
    unified daemon is running it OWNS the worker. A fresh
    `EmbeddingService` started here would see the singleton, set
    `_available = False`, return None from `_subprocess_embed`, and
    print "embedding verification failed" — even though the daemon's
    worker is already happily serving the same model. The fix: detect
    the daemon, verify via its health endpoint, and skip the local
    spawn. Only fall through to the original local-worker path when
    the daemon is genuinely unreachable.
    """
    import superlocalmemory.core.embeddings as _emb_mod

    print("SuperLocalMemory V3 — Embedding Model Warmup")
    print("=" * 50)
    print(f"  Python: {sys.executable}")
    print(f"  Model:  nomic-ai/nomic-embed-text-v1.5 (~500MB)")
    print()

    # v3.4.42 — daemon-aware fast path. If the daemon is up and reports
    # engine=initialized, the embedding model is already loaded inside
    # the daemon's worker subprocess. No need to spawn a redundant one;
    # in fact, the machine-wide singleton would refuse to do so anyway.
    try:
        from superlocalmemory.cli.daemon import (
            is_daemon_running, daemon_request,
        )
        if is_daemon_running():
            health = daemon_request("GET", "/health")
            if health and health.get("engine") == "initialized":
                from superlocalmemory.core.config import EmbeddingConfig
                cfg = EmbeddingConfig()
                print("[PASS] Daemon is running with embedding model loaded.")
                print(f"       Model: {cfg.model_name} ({cfg.dimension}-dim)")
                print("Semantic search is fully operational.")
                return
            # Daemon up but engine not yet initialized — warn and return
            # rather than racing the daemon for the singleton lock.
            engine_state = (health or {}).get("engine", "unknown")
            print(f"[INFO] Daemon is up but engine state is '{engine_state}'.")
            print("       Wait ~30s and retry, or run: slm doctor")
            return
    except Exception:
        # Any failure in the daemon path falls through to local warmup —
        # better to spawn a local worker than block warmup entirely.
        pass

    # Local-warmup fallback path: daemon is unreachable, so it's safe
    # to spawn our own embedding worker (no singleton conflict).
    # Increase timeout for first-time download.
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
    os.environ["ORT_DISABLE_COREML"] = "1"
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


def _switch_profile_runtime(config, profile_name: str) -> dict:
    """Switch through the resident daemon, or persist an offline fallback."""
    from superlocalmemory.cli.daemon import daemon_request, is_daemon_running

    if is_daemon_running():
        result = daemon_request(
            "POST",
            f"/api/profiles/{profile_name}/switch",
        )
        if not result or not result.get("success"):
            raise RuntimeError(
                "resident daemon did not acknowledge the profile switch"
            )
        acknowledged = str(result.get("active_profile", ""))
        if acknowledged != profile_name:
            raise RuntimeError(
                "resident daemon acknowledged a different active profile"
            )
        return {
            "action": "switched",
            "profile": acknowledged,
            "generation": int(result.get("generation", 0)),
            "runtime": "daemon",
        }

    from superlocalmemory.server.profile_runtime import persist_active_profile

    persist_active_profile(profile_name)
    config.active_profile = profile_name
    return {
        "action": "switched",
        "profile": profile_name,
        "runtime": "offline",
    }


def cmd_profile(args: Namespace) -> None:
    """Profile management (list, switch, create).

    Writes to BOTH SQLite and profiles.json so CLI, Dashboard, and
    MCP all see the same profiles.
    """
    from superlocalmemory.core.config import SLMConfig
    from superlocalmemory.storage.database import DatabaseManager
    from superlocalmemory.storage import schema
    from superlocalmemory.server.routes.helpers import (
        ensure_profile_in_json,
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
            rows = db.execute(
                "SELECT 1 FROM profiles WHERE profile_id = ?",
                (args.name,),
            )
            if not rows:
                json_print("profile", error={
                    "code": "PROFILE_NOT_FOUND",
                    "message": f"Profile '{args.name}' does not exist.",
                })
                sys.exit(1)
            try:
                result = _switch_profile_runtime(config, args.name)
            except Exception as exc:
                json_print("profile", error={
                    "code": "PROFILE_SWITCH_FAILED",
                    "message": str(exc),
                })
                sys.exit(1)
            json_print("profile", data=result)
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
        rows = db.execute(
            "SELECT 1 FROM profiles WHERE profile_id = ?",
            (args.name,),
        )
        if not rows:
            print(f"Profile '{args.name}' does not exist.", file=sys.stderr)
            sys.exit(1)
        try:
            result = _switch_profile_runtime(config, args.name)
        except Exception as exc:
            print(f"Profile switch failed: {exc}", file=sys.stderr)
            sys.exit(1)
        generation = result.get("generation")
        suffix = f" (generation {generation})" if generation is not None else ""
        print(f"Switched to profile: {args.name}{suffix}")
    elif args.action == "create":
        db.execute(
            "INSERT OR IGNORE INTO profiles (profile_id, name) VALUES (?, ?)",
            (args.name, args.name),
        )
        ensure_profile_in_json(args.name)
        print(f"Created profile: {args.name}")


# -- Active Memory commands (V3.1) ------------------------------------------


def _cmd_init_auto(
    args: Namespace,
    slm_data_dir: "Path",
    config_exists: bool,
    force: bool,
) -> None:
    """WP-07: non-interactive --auto branch for slm init.

    Best-effort at every step; only exits non-zero when config save fails.
    No TTY required.  Does NOT run IDE connect (AC6).
    """
    from pathlib import Path
    from superlocalmemory.core.config import SLMConfig
    from superlocalmemory.storage.models import Mode

    # Step 1: write mode-A config (create-if-absent or --force).
    # Pass slm_data_dir explicitly so env-overridden paths are respected even
    # when DEFAULT_BASE_DIR was evaluated before the env var was set (e.g. tests).
    if force or not config_exists:
        try:
            cfg = SLMConfig.for_mode(Mode.A, base_dir=slm_data_dir)
            cfg.save(mode_change=True)
        except Exception as exc:
            print(f"[ERROR] slm init --auto: config save failed: {exc}", file=sys.stderr)
            sys.exit(1)

    # Step 2: mark complete (write .setup-complete sentinel).
    # Write the sentinel directly using the already-selected namespace.
    try:
        import platform
        import time as _time
        sentinel = slm_data_dir / ".setup-complete"
        slm_data_dir.mkdir(parents=True, exist_ok=True)
        sentinel.write_text(
            f"setup_completed={_time.strftime('%Y-%m-%dT%H:%M:%S')}\n"
            f"python={sys.executable}\n"
            f"platform={platform.system()}\n"
            f"version={platform.python_version()}\n"
        )
    except Exception:
        pass  # best-effort — sentinel is advisory only

    # Step 3: install hooks if ~/.claude exists.
    try:
        claude_dir = Path.home() / ".claude"
        if claude_dir.exists():
            from superlocalmemory.hooks.claude_code_hooks import install_hooks
            install_hooks(include_gate=getattr(args, "gate", False))
    except Exception:
        pass  # best-effort

    # Step 4: warmup best-effort (don't block if models not present).
    # Skipped in --auto to keep startup fast for CI; user can run slm warmup.

    print("[OK] slm init --auto: setup complete (mode A, non-interactive)", file=sys.stderr)


def cmd_init(args: Namespace) -> None:
    """One-command setup: mode + hooks + IDE connect + warmup."""
    from superlocalmemory.cli._lazy_init import slm_home
    from superlocalmemory.core.config import SLMConfig

    force = getattr(args, "force", False)
    auto = getattr(args, "auto", False)

    slm_data_dir = slm_home()
    config_exists = (slm_data_dir / "config.json").exists()

    # WP-07: --auto branch — fully non-interactive, no TTY required (AC6).
    if auto:
        os.environ["SLM_NON_INTERACTIVE"] = "1"
        _cmd_init_auto(args, slm_data_dir, config_exists, force)
        return

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
    """Manage additive Claude Code or Codex memory lifecycle hooks."""
    from superlocalmemory.hooks.claude_code_hooks import (
        install_hooks, remove_hooks, check_status,
    )

    action = getattr(args, "action", "status")
    agent = getattr(args, "agent", "claude")
    dry_run = getattr(args, "dry_run", False)
    if agent == "codex":
        from superlocalmemory.hooks.codex_hooks import (
            install_hooks as install_codex_hooks,
            remove_hooks as remove_codex_hooks,
            check_status as check_codex_hooks,
        )
        if action == "install":
            result = install_codex_hooks(dry_run=dry_run)
            if result["success"]:
                prefix = "would be installed" if dry_run else "installed"
                print(f"SLM hooks {prefix} in Codex: {result['path']}")
                if result.get("hooks_added"):
                    print(f"  Hook types: {', '.join(result['hooks_added'])}")
                print("  Review and trust the new hooks in Codex with /hooks.")
            else:
                print(f"Installation failed: {result['errors']}")
            return
        if action == "remove":
            result = remove_codex_hooks(dry_run=dry_run)
            if result["success"]:
                prefix = "would be removed" if dry_run else "removed"
                print(f"SLM hooks {prefix} from Codex: {result['path']}")
            else:
                print(f"Removal failed: {result['errors']}")
            return
        result = check_codex_hooks()
        if result["installed"] is None:
            print(f"SLM Codex hooks: INDETERMINATE ({result['error']})")
        elif result["installed"]:
            print("SLM Codex hooks: INSTALLED")
            print(f"  Hook types: {', '.join(result['hook_types'])}")
        else:
            print("SLM Codex hooks: NOT INSTALLED")
            print("  Run: slm hooks install --agent codex")
        return
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


def cmd_codex(args: Namespace) -> None:
    """Manage explicit, SLM-owned Codex skills, agents, and lifecycle hooks."""
    from superlocalmemory.hooks.codex_assets import install_assets, remove_assets, status_assets
    from superlocalmemory.hooks.codex_hooks import install_hooks, remove_hooks, check_status

    action, dry_run = getattr(args, "action", "status"), getattr(args, "dry_run", False)
    if action == "install":
        assets, hooks = install_assets(dry_run=dry_run), install_hooks(dry_run=dry_run)
        if assets.get("success") and hooks.get("success"):
            print(f"SLM Codex add-ons {'would be installed' if dry_run else 'installed'}: 7 skills, 2 subagents, 4 lifecycle hooks.")
            print("MCP wiring remains explicit: run `slm connect codex` if it is not already configured.")
            print("Review and trust newly installed hooks in Codex with /hooks.")
        else:
            print(f"Codex integration failed: {assets.get('errors', []) + hooks.get('errors', [])}")
        return
    if action == "remove":
        assets, hooks = remove_assets(dry_run=dry_run), remove_hooks(dry_run=dry_run)
        if assets.get("success") and hooks.get("success"):
            print("SLM-owned Codex add-ons removed; your MCP and non-SLM settings were left intact.")
        else:
            print(f"Codex removal failed: {assets.get('errors', []) + hooks.get('errors', [])}")
        return
    assets, hooks = status_assets(), check_status()
    print(f"SLM Codex add-ons: {'INSTALLED' if assets['installed'] and hooks['installed'] else 'NOT INSTALLED'}")
    print(f"  Skills: {len(assets['skills'])}/7; subagents: {len(assets['agents'])}/2")
    print(f"  Hooks: {', '.join(hooks.get('hook_types', [])) or 'none'}")


def cmd_session_context(args: Namespace) -> None:
    """Print session context (for hook scripts and piping).

    Uses a FAST PATH that queries SQLite directly (no engine/Ollama needed).
    This ensures the SessionStart hook completes within its 15s timeout even
    when Ollama requires a 60s+ cold start.  The fast path returns:
      - Core Memory blocks (always-on context)
      - Recent high-importance memories (last N days)
      - Session summary from last session
    Falls back to the full engine path only if --full is passed explicitly.

    v3.4.65: uses shared injection formatter (render_context) for identical
    output across MCP and CLI surfaces. --json flag returns structured JSON.
    """
    import sqlite3
    from pathlib import Path
    from superlocalmemory.core.config import SLMConfig

    use_json = getattr(args, "json", False)
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
                if use_json:
                    from superlocalmemory.cli.json_output import json_print
                    json_print("session-context", data={"context": context}, next_actions=[
                        {"command": "slm recall --json <query>", "description": "Search memories"},
                    ])
                else:
                    print(context)
        except Exception as exc:
            logger.debug("session-context (full) failed: %s", exc)
        return

    # ── FAST PATH: direct SQLite, no engine, <500ms ──────────────
    try:
        from superlocalmemory.core.injection import InjectableMemory, render_context

        config = SLMConfig.load()
        db_path = config.base_dir / "memory.db"
        if not db_path.exists():
            return

        pid = config.active_profile
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row

        # Collect facts for injection — same queries as pre-v3.4.65 but
        # mapped into InjectableMemory for the shared formatter.
        inj_mems: list[InjectableMemory] = []

        # Core Memory blocks (compiled high-value context)
        try:
            cm_rows = conn.execute(
                "SELECT block_type, content FROM core_memory_blocks "
                "WHERE profile_id = ? ORDER BY block_type",
                (pid,),
            ).fetchall()
            for r in cm_rows:
                content = f"[{r['block_type']}] {r['content']}"
                inj_mems.append(InjectableMemory(
                    content=content, score=1.0, fact_id="",
                    importance=1.0, access_count=10,
                    pinned=True,
                ))
        except sqlite3.OperationalError:
            pass

        # Recent important memories — age gate from --max-age-days (default 30)
        max_age = getattr(args, "max_age_days", 30)
        age_clause = (
            f"AND created_at >= datetime('now', '-{int(max_age)} days') "
            if max_age > 0 else ""
        )
        try:
            fact_rows = conn.execute(
                "SELECT fact_id, content, importance, access_count, fact_type FROM atomic_facts "
                "WHERE profile_id = ? "
                f"{age_clause}"
                "AND lifecycle = 'active' "
                "ORDER BY importance DESC, created_at DESC LIMIT 10",
                (pid,),
            ).fetchall()
            for r in fact_rows:
                inj_mems.append(InjectableMemory(
                    content=r["content"],
                    score=r["importance"] or 0.5,
                    fact_id=r["fact_id"],
                    importance=r["importance"] or 0.0,
                    access_count=r["access_count"] or 0,
                ))
        except sqlite3.OperationalError:
            pass

        # Session markers (last session summary)
        try:
            sess_rows = conn.execute(
                "SELECT fact_id, content, importance, access_count FROM atomic_facts "
                "WHERE profile_id = ? AND content LIKE 'Session%' "
                "ORDER BY created_at DESC LIMIT 3",
                (pid,),
            ).fetchall()
            for r in sess_rows:
                inj_mems.append(InjectableMemory(
                    content=r["content"],
                    score=r["importance"] or 0.3,
                    fact_id=r["fact_id"],
                    importance=r["importance"] or 0.0,
                    access_count=r["access_count"] or 0,
                ))
        except sqlite3.OperationalError:
            pass

        conn.close()

        if not inj_mems:
            return

        # V3.3 Soft prompts (auto-learned patterns) — append as high-importance
        try:
            conn2 = sqlite3.connect(str(db_path))
            conn2.row_factory = sqlite3.Row
            sp_rows = conn2.execute(
                "SELECT category, content FROM soft_prompt_templates "
                "WHERE profile_id = ? AND active = 1 "
                "ORDER BY confidence DESC LIMIT 5",
                (pid,),
            ).fetchall()
            conn2.close()
            for r in sp_rows:
                inj_mems.append(InjectableMemory(
                    content=f"[{r['category']}] {r['content']}",
                    score=0.7, fact_id="",
                    importance=0.7, access_count=5,
                ))
        except Exception:
            pass

        cfg_inj = getattr(config, "injection", None)
        context = render_context(inj_mems, mode=config.mode.value.upper(), cfg=cfg_inj, wrap=True)
        if context:
            if use_json:
                from superlocalmemory.cli.json_output import json_print
                json_print("session-context", data={
                    "context": context,
                    "memory_count": len(inj_mems),
                    "mode": config.mode.value.upper(),
                }, next_actions=[
                    {"command": "slm recall --json <query>", "description": "Search memories"},
                ])
            else:
                print(context)

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

            from superlocalmemory.core.engine_ingestion import (
                canonical_store_fn,
                local_trusted_actor_id,
            )
            auto = AutoCapture(store_fn=canonical_store_fn(
                engine,
                source_type="cli-observe",
                trusted_actor_id=local_trusted_actor_id("cli"),
            ))
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
        result = scheduler.run_decay_cycle(pid, force=True, dry_run=dry_run)
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
        result = scheduler.run_eap_cycle(pid, dry_run=dry_run)
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
    # V3.5.9: --all bypasses orphan detection and kills every slm mcp process
    # except the current one. Use after switching IDEs to clear stale sessions.
    use_force_all = getattr(args, "all", False)
    if use_force_all:
        dry_run = False  # --all always kills; --force is implied

    try:
        from superlocalmemory.infra.process_reaper import (
            cleanup_all_orphans,
            ReaperConfig,
        )

        config = ReaperConfig()
        result = cleanup_all_orphans(config, dry_run=dry_run, force=use_force_all)
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
            {"command": "slm reap --force --json", "description": "Kill orphan processes"},
            {"command": "slm reap --all --json", "description": "Kill ALL slm mcp sessions (IDE switch)"},
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
        print("  Use --all to kill ALL slm mcp sessions (after IDE switch).")
