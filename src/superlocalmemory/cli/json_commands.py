# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V4 | https://qualixar.com | https://varunpratap.com

"""The JSON form of commands whose plain form is a wizard, a server or a launcher.

``dispatch`` sends a command here when ``args.json`` is set and the command
is listed in :data:`HANDLERS`. Every other command prints its own JSON (or
its text, which ``json_mode.run_json`` wraps in the standard envelope).

What each long-running or interactive command does under ``--json``:

* ``mcp``: describes how to connect (stdio command, HTTP URL, whether the
  daemon is up) and exits 0. It never starts the stdio server.
* ``dashboard``: the URL, port and whether the daemon is running. It starts
  nothing and opens no browser.
* ``serve``: ``status`` reports; ``start``/``stop``/``install``/``uninstall``
  do what they always do (the daemon runs in the background, so nothing
  streams) and report the state afterwards.
* ``warmup``: one object with ``ready`` after the same wait as without JSON.
* ``wrap``: ``--list`` lists agents; otherwise the planned action is reported
  and the agent is not launched (``--persistent`` still writes its config).
* ``setup`` / ``init``: need ``--auto`` (the non-interactive path);
  without it the answer is ``input_required``.
* ``reconfigure``: needs ``--profile=NAME`` among the passed-through flags.
* ``migrate``: needs ``--yes`` to migrate (``confirmation_required`` otherwise).
"""

from __future__ import annotations

import sys
from argparse import Namespace
from collections.abc import Callable
from typing import Any

from superlocalmemory.cli.json_mode import EXIT_FAILED, EXIT_USAGE, fail
from superlocalmemory.cli.json_output import json_print


def _daemon_state() -> dict[str, Any]:
    from superlocalmemory.cli.daemon import is_daemon_running
    from superlocalmemory.cli.deployment import runtime_port

    port = runtime_port()
    try:
        running = bool(is_daemon_running())
    except Exception:  # noqa: BLE001 - a probe failure reads as "not running"
        running = False
    return {"running": running, "port": port, "url": f"http://127.0.0.1:{port}"}


# -- servers and launchers ---------------------------------------------------------


def json_mcp(args: Namespace) -> None:
    state = _daemon_state()
    json_print("mcp", data={
        "started": False,
        "stdio": {"command": "slm", "args": ["mcp"],
                  "config": {"mcpServers": {"superlocalmemory": {"command": "slm",
                                                                  "args": ["mcp"]}}}},
        "http": {"url": f"{state['url']}/mcp/", "requires_daemon": True,
                 "daemon_running": state["running"]},
        "note": "`slm mcp` (without --json) is the stdio server an MCP client launches.",
    }, next_actions=[
        {"command": "slm connect --list --json", "description": "Apps SLM can configure"},
        {"command": "slm tool list --json", "description": "Call the same tools from a shell"},
    ])


def json_dashboard(args: Namespace) -> None:
    from superlocalmemory.cli.deployment import runtime_port

    state = _daemon_state()
    port = getattr(args, "port", None) or runtime_port()
    url = f"http://localhost:{port}"
    data = {"url": url, "api_docs": f"{url}/docs", "port": port,
            "running": state["running"], "opened_browser": False}
    actions = [] if state["running"] else [
        {"command": "slm serve start --json", "description": "Start the daemon that serves it"}]
    json_print("dashboard", data=data, next_actions=actions or None)


def _serve_status() -> dict[str, Any]:
    from superlocalmemory.cli.daemon import daemon_request

    state = _daemon_state()
    data: dict[str, Any] = {"running": state["running"], "port": state["port"],
                            "url": state["url"]}
    if state["running"]:
        status = daemon_request("GET", "/status") or {}
        data["status"] = status
        data["pid"] = status.get("pid")
    try:
        from superlocalmemory.cli.service_installer import service_status

        svc = service_status()
        data["service"] = {"installed": bool(svc.get("installed", False)),
                           "type": svc.get("service_type", svc.get("platform"))}
    except Exception:  # noqa: BLE001 - the service line is extra information
        data["service"] = None
    return data


def json_serve(args: Namespace) -> None:
    from superlocalmemory.cli import daemon

    action = getattr(args, "action", "start") or "start"
    command = f"serve {action}"
    if action == "status":
        json_print(command, data=_serve_status())
        return
    if action == "stop":
        stopped = daemon.stop_daemon()
        alive = daemon.owned_daemon_process_alive()
        data = {"stopped": bool(stopped), "running": bool(alive)}
        if alive:
            fail(command, "daemon_still_running", "The daemon is still running; run "
                 "`slm serve stop --json` again, or `slm doctor --json`.", **data)
        json_print(command, data=data)
        return
    if action in ("install", "uninstall"):
        from superlocalmemory.cli import service_installer

        fn = service_installer.install_service if action == "install" \
            else service_installer.uninstall_service
        if not fn():
            fail(command, f"{action}_failed", f"The OS service {action} did not succeed; "
                 "the reason is on stderr.")
        json_print(command, data={"installed": action == "install"})
        return
    if daemon.is_daemon_running():
        json_print(command, data={**_serve_status(), "already_running": True})
        return
    blocked = daemon._recovery_block()
    if blocked:
        fail(command, "recovery_pending", str(blocked))
    if not daemon.ensure_daemon():
        from superlocalmemory.infra.data_root import state_path

        fail(command, "start_failed", "The daemon did not start.",
             log=str(state_path("logs", "daemon.log")))
    json_print(command, data={**_serve_status(), "already_running": False})


def json_warmup(args: Namespace) -> None:
    from superlocalmemory.cli import commands
    from superlocalmemory.cli.daemon import daemon_request, is_daemon_running
    from superlocalmemory.core.config import EmbeddingConfig

    cfg = EmbeddingConfig()
    model = {"name": cfg.model_name, "dimension": cfg.dimension}
    try:
        daemon_up = is_daemon_running()
    except Exception:  # noqa: BLE001
        daemon_up = False
    if daemon_up:
        from superlocalmemory.cli.warmup_readiness import DEFAULT_WAIT_SECONDS, wait_until_warm

        timeout = getattr(args, "timeout", None)
        timeout = DEFAULT_WAIT_SECONDS if timeout is None else max(0.0, float(timeout))
        verdict = wait_until_warm(lambda: _safe(daemon_request, "GET", "/health"), timeout)
        data = {"ready": bool(verdict.ready), "via": "daemon", "model": model,
                "reachable": bool(verdict.reachable), "detail": getattr(verdict, "detail", ""),
                "waited_seconds": round(float(getattr(verdict, "waited_seconds", 0.0)), 1)}
        if not verdict.ready:
            fail("warmup", "not_ready", "The daemon is running but recall is not ready yet."
                 if verdict.reachable else "The daemon stopped answering.", **data)
        json_print("warmup", data=data)
        return
    # No daemon: the local check prints its steps; they go to stderr here.
    real = sys.stdout
    sys.stdout = sys.stderr
    try:
        commands.cmd_warmup(args)
        rc = 0
    except SystemExit as exc:
        rc = exc.code if isinstance(exc.code, int) else 1
    finally:
        sys.stdout = real
    if rc:
        fail("warmup", "not_ready", "The embedding model did not load; details on stderr.",
             ready=False, via="local", model=model)
    json_print("warmup", data={"ready": True, "via": "local", "model": model})


def _safe(fn: Callable[..., Any], *a: Any) -> Any:
    try:
        return fn(*a)
    except Exception:  # noqa: BLE001
        return None


def json_wrap(args: Namespace) -> None:
    from superlocalmemory.optimize.adapters._agent_registry import AGENT_REGISTRY
    from superlocalmemory.optimize.adapters.wrap import list_agents

    if getattr(args, "list", False):
        json_print("wrap", data={"agents": [
            {"agent": k, "mechanism": AGENT_REGISTRY.get(k, {}).get("mechanism", "unknown")}
            for k in list_agents()]})
        return
    agent = getattr(args, "agent", None)
    if not agent:
        fail("wrap", "input_required", "Name the agent: slm wrap --json <agent>.",
             exit_code=EXIT_USAGE, missing_flags=["agent"])
    if agent not in AGENT_REGISTRY:
        fail("wrap", "unknown_agent", f"Unknown agent {agent!r}.",
             exit_code=EXIT_USAGE, known=sorted(AGENT_REGISTRY))
    spec = AGENT_REGISTRY[agent]
    if getattr(args, "persistent", False) and not getattr(args, "dry_run", False):
        from superlocalmemory.optimize.adapters.wrap import wrap_agent

        real = sys.stdout
        sys.stdout = sys.stderr
        try:
            rc = wrap_agent(agent, [], persistent=True, dry_run=False)
        finally:
            sys.stdout = real
        if rc:
            fail("wrap", "wrap_failed", "The agent's config was not written; see stderr.")
        json_print("wrap", data={"agent": agent, "persistent": True, "launched": False,
                                 "mechanism": spec.get("mechanism")})
        return
    from superlocalmemory.optimize.adapters.wrap import proxy_port

    json_print("wrap", data={
        "agent": agent, "launched": False, "mechanism": spec.get("mechanism"),
        "proxy_port": proxy_port(), "env_vars": spec.get("env_vars", {}),
        "agent_args": list(getattr(args, "agent_args", []) or []),
        "note": "JSON never launches the agent; run `slm wrap <agent>` in a terminal.",
    })


# -- sessions, mesh, db --------------------------------------------------------------


def json_session(args: Namespace) -> None:
    from superlocalmemory.cli.daemon import daemon_request, ensure_daemon, is_daemon_running

    action = getattr(args, "session_command", None)
    command = f"session {action}"
    if not is_daemon_running():
        ensure_daemon()
    if action == "open":
        resp = daemon_request("POST", "/session/open", {
            "project_path": getattr(args, "project_path", "") or "",
            "query": getattr(args, "query", "") or "",
            "max_results": int(getattr(args, "max_results", 10) or 10)})
    else:
        resp = daemon_request("POST", "/session/close",
                              {"session_id": getattr(args, "session_id", "") or ""})
    if not (resp and resp.get("ok")):
        fail(command, "daemon_unreachable", f"Session {action} failed: the daemon did not answer.")
    json_print(command, data=resp)


def json_mesh(args: Namespace) -> None:
    from superlocalmemory.cli.daemon import daemon_request

    action = getattr(args, "mesh_action", None)
    result = daemon_request("GET", f"/mesh/{action}")
    if result is None:
        fail(f"mesh {action}", "daemon_unreachable",
             "Cannot reach the daemon broker. Start it: slm serve start --json")
    json_print(f"mesh {action}", data=result if isinstance(result, dict) else {"result": result})


def json_db_compact(args: Namespace) -> None:
    from io import StringIO

    from superlocalmemory.cli import commands

    real = sys.stdout
    buf = StringIO()
    sys.stdout = buf
    try:
        rc = commands._cmd_db_compact(args)
    finally:
        sys.stdout = real
    text = buf.getvalue().strip()
    if rc:
        fail("db compact", "compact_failed", text or "Compaction failed; see stderr.")
    json_print("db compact", data={"offline": bool(getattr(args, "offline", False)),
                                   "detail": text})


def json_db_migrate(args: Namespace) -> None:
    from superlocalmemory.cli import db_migrate
    from superlocalmemory.storage.migration_runner import apply_all, status

    learning_db, memory_db = db_migrate._resolve_paths(args)
    if getattr(args, "status", False):
        report = status(learning_db, memory_db) or {}
        json_print("db migrate", data={
            "migrations": report,
            "end_state_disagreements": db_migrate._end_state_disagreements(learning_db, memory_db)
            if report else {}})
        return
    result = apply_all(learning_db, memory_db, dry_run=bool(getattr(args, "dry_run", False)))
    data = {"dry_run": bool(getattr(args, "dry_run", False)), **result}
    if result.get("failed"):
        fail("db migrate", "migration_failed", f"{len(result['failed'])} migration(s) failed.",
             **data)
    json_print("db migrate", data=data)


# -- hooks and add-ons ----------------------------------------------------------------


def json_hooks(args: Namespace) -> None:
    action = getattr(args, "action", "status") or "status"
    agent = getattr(args, "agent", "claude") or "claude"
    dry_run = bool(getattr(args, "dry_run", False))
    if agent == "codex":
        from superlocalmemory.hooks import codex_hooks as h

        result = {"install": lambda: h.install_hooks(dry_run=dry_run),
                  "remove": lambda: h.remove_hooks(dry_run=dry_run)}.get(action, h.check_status)()
    else:
        from superlocalmemory.hooks import claude_code_hooks as h

        result = {"install": lambda: h.install_hooks(include_gate=bool(getattr(args, "gate", False))),
                  "remove": h.remove_hooks}.get(action, h.check_status)()
    _result_or_fail(f"hooks {action}", {"agent": agent, "action": action, "dry_run": dry_run,
                                        **dict(result)})


def json_codex(args: Namespace) -> None:
    from superlocalmemory.hooks import codex_assets as a
    from superlocalmemory.hooks import codex_hooks as h

    action = getattr(args, "action", "status") or "status"
    dry_run = bool(getattr(args, "dry_run", False))
    if action == "install":
        assets, hooks = a.install_assets(dry_run=dry_run), h.install_hooks(dry_run=dry_run)
    elif action == "remove":
        assets, hooks = a.remove_assets(dry_run=dry_run), h.remove_hooks(dry_run=dry_run)
    else:
        assets, hooks = a.status_assets(), h.check_status()
        json_print("codex status", data={
            "installed": bool(assets.get("installed") and hooks.get("installed")),
            "assets": assets, "hooks": hooks,
            "skills_total": len(a.SKILLS), "agents_total": len(a.AGENTS)})
        return
    data = {"action": action, "dry_run": dry_run, "assets": assets, "hooks": hooks,
            "success": bool(assets.get("success") and hooks.get("success"))}
    _result_or_fail(f"codex {action}", data)


def _result_or_fail(command: str, data: dict) -> None:
    ok = data.pop("success", True)
    if ok is False:
        errors = data.get("errors") or (data.get("assets", {}).get("errors", [])
                                        + data.get("hooks", {}).get("errors", []))
        fail(command, "failed", f"{command} did not succeed: {errors}", **data)
    json_print(command, data=data)


def json_adapters(args: Namespace) -> None:
    from superlocalmemory.ingestion import adapter_manager as am

    rest = list(getattr(args, "rest", []) or []) or ["list"]
    action, name = rest[0], (rest[1] if len(rest) > 1 else "")
    command = f"adapters {action}"
    if action in ("list", "status"):
        rows = am.list_adapters() if action == "list" else am.status_adapters()
        json_print(command, data={"adapters": rows})
        return
    actions = {"enable": am.enable_adapter, "disable": am.disable_adapter,
               "start": am.start_adapter, "stop": am.stop_adapter}
    if action not in actions:
        am.handle_adapters_cli(rest)  # allow-folder / folders print text; run_json wraps it
        return
    if not name:
        fail(command, "input_required", f"Name the adapter: slm adapters {action} <name>.",
             exit_code=EXIT_USAGE, missing_flags=["name"])
    result = dict(actions[action](name))
    if result.get("error"):
        fail(command, "failed", str(result["error"]), **result)
    json_print(command, data=result)


# -- wizards and confirmations ------------------------------------------------------


def json_provider(args: Namespace) -> None:
    from superlocalmemory.core.config import SLMConfig

    config = SLMConfig.load()
    if getattr(args, "action", None) != "set":
        json_print("provider", data={"provider": config.llm.provider or None,
                                     "model": config.llm.model or None,
                                     "mode": config.mode.value})
        return
    if not getattr(args, "provider", None):
        fail("provider set", "input_required", "Name the provider: slm provider set "
             "<openai|anthropic|azure|ollama|openrouter|custom> --json.",
             missing_flags=["provider"])
    from superlocalmemory.cli.json_mode import PromptBlocked
    from superlocalmemory.cli.setup_wizard import configure_provider

    real = sys.stdout
    sys.stdout = sys.stderr  # the configurator narrates; the narration is not the answer
    try:
        configure_provider(config, provider_name=args.provider,
                           endpoint=getattr(args, "endpoint", None),
                           api_key=getattr(args, "api_key", None),
                           model=getattr(args, "model", None),
                           target_mode=getattr(args, "mode", None))
    except PromptBlocked as blocked:
        sys.stdout = real
        fail("provider set", "input_required",
             f"The provider needs more answers ({' '.join(blocked.prompt.split())!r}).",
             missing_flags=["--key", "--model", "--endpoint"], prompt=blocked.prompt)
    finally:
        sys.stdout = real
    after = SLMConfig.load()
    json_print("provider set", data={"provider": after.llm.provider or None,
                                     "model": after.llm.model or None, "mode": after.mode.value})


def _needs_auto(command: str, args: Namespace) -> None:
    if not getattr(args, "auto", False):
        fail(command, "input_required", f"`slm {command}` is a wizard; with --json pass "
             "--auto for the non-interactive setup (mode A + hooks).",
             missing_flags=["--auto"])


def json_setup(args: Namespace) -> None:
    from superlocalmemory.cli import commands

    _needs_auto("setup", args)
    _quietly(lambda: commands.cmd_setup(args), "setup")


def json_init(args: Namespace) -> None:
    from superlocalmemory.cli import commands

    _needs_auto("init", args)
    _quietly(lambda: commands.cmd_init(args), "init")


def _quietly(fn: Callable[[], Any], command: str) -> None:
    """Run a narrating command with its narration on stderr; report how it ended."""
    real = sys.stdout
    sys.stdout = sys.stderr
    rc = 0
    try:
        fn()
    except SystemExit as exc:
        rc = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
    finally:
        sys.stdout = real
    if rc:
        fail(command, "failed", f"`slm {command}` did not finish; details on stderr.",
             exit_code=EXIT_FAILED)
    from superlocalmemory.core.config import SLMConfig

    config = SLMConfig.load()
    json_print(command, data={"completed": True, "mode": config.mode.value,
                              "profile": config.active_profile})


def json_reconfigure(args: Namespace) -> None:
    import subprocess

    from superlocalmemory.cli.escape_hatch import _find_postinstall_script

    extras = list(getattr(args, "extras", None) or [])
    if not any(e.startswith("--profile=") for e in extras):
        fail("reconfigure", "input_required", "With --json pass the answers through: "
             "slm --json reconfigure -- --profile=<minimal|light|balanced|...>.",
             missing_flags=["--profile=NAME"])
    script = _find_postinstall_script()
    if script is None:
        fail("reconfigure", "not_available", "postinstall-interactive.js is not installed; "
             "reinstall the npm package.", exit_code=EXIT_USAGE)
    run = subprocess.run(["node", str(script), "--reconfigure", *extras], check=False,
                         stdin=subprocess.DEVNULL, capture_output=True, text=True)
    if run.stderr:
        print(run.stderr, file=sys.stderr, end="")
    if run.returncode:
        fail("reconfigure", "failed", "The reconfigure script failed.",
             exit_code=run.returncode if run.returncode in (1, 2) else EXIT_FAILED,
             output=run.stdout)
    json_print("reconfigure", data={"extras": extras, "output": run.stdout.rstrip("\n")})


def json_migrate(args: Namespace) -> None:
    from superlocalmemory.storage.v2_migrator import V2Migrator

    migrator = V2Migrator()
    if getattr(args, "rollback", False):
        result = migrator.rollback()
        if not result.get("success"):
            fail("migrate", "rollback_failed", str(result.get("error", "unknown")), **result)
        json_print("migrate", data={"rolled_back": True, **result})
        return
    if not migrator.detect_v2():
        json_print("migrate", data={"migrated": False, "reason": "no_v2_installation"})
        return
    if migrator.is_already_migrated():
        json_print("migrate", data={"migrated": False, "reason": "already_migrated"})
        return
    stats = migrator.get_v2_stats()
    if not getattr(args, "yes", False):
        fail("migrate", "confirmation_required", "A V2 installation was found. Pass --yes "
             "to migrate it.", flag="--yes", v2=stats)
    result = migrator.migrate()
    if not result.get("success"):
        fail("migrate", "migration_failed", str(result.get("error", "unknown")), **result)
    json_print("migrate", data={"migrated": True, "v2": stats, **result})


# -- help ------------------------------------------------------------------------------


def json_help(args: Namespace) -> None:
    from superlocalmemory.cli import commands, help_tree
    from superlocalmemory.cli.main import build_parser

    topics = sorted(set(commands._HELP_TOPICS))
    topic = (getattr(args, "topic", None) or "").strip().lower()
    if topic:
        text = commands._HELP_TOPICS.get(topic)
        if text is None:
            fail("help", "unknown_topic", f"No help topic {topic!r}.", exit_code=EXIT_USAGE,
                 topics=topics)
        json_print("help", data={"topic": topic, "text": text, "topics": topics})
        return
    json_print("help", data={
        "commands": help_tree.command_tree(build_parser()),
        "global_options": help_tree.global_options(),
        "groups": [{"title": t, "commands": [{"command": c, "summary": d} for c, d in rows]}
                   for t, rows in commands._COMMAND_GROUPS],
        "topics": topics,
        "json_contract": {
            "switch": ["slm --json <command>", "SLM_JSON=1", "slm <command> --json"],
            "exit_codes": {"0": "success", "1": "the command ran and failed",
                           "2": "bad usage or could not run"},
            "errors": ["confirmation_required", "input_required", "usage"],
        },
    })


def json_help_optimize(args: Namespace) -> None:
    from superlocalmemory.cli import help_cmd

    sections = help_cmd._HELP_SECTIONS
    topics = [k for k in sections if k != "footer"]
    topic = getattr(args, "topic", None)
    text = sections[topic] if topic else "\n".join(sections.values())
    json_print("help-optimize", data={"topic": topic, "topics": topics, "text": text})


HANDLERS: dict[str, Callable[[Namespace], None]] = {
    "mcp": json_mcp, "dashboard": json_dashboard, "serve": json_serve, "warmup": json_warmup,
    "wrap": json_wrap, "session": json_session, "mesh": json_mesh, "hooks": json_hooks,
    "codex": json_codex, "adapters": json_adapters, "provider": json_provider,
    "setup": json_setup, "init": json_init, "reconfigure": json_reconfigure,
    "migrate": json_migrate, "help": json_help, "help-optimize": json_help_optimize,
    "db compact": json_db_compact, "db migrate": json_db_migrate,
}


def handler_for(args: Namespace) -> Callable[[Namespace], None] | None:
    """The JSON handler for a parsed command, or None (the command speaks JSON itself)."""
    command = str(getattr(args, "command", "") or "")
    if command == "db":
        return HANDLERS.get(f"db {getattr(args, 'db_command', '')}")
    if command == "session" and getattr(args, "session_command", None) not in ("open", "close"):
        return None
    return HANDLERS.get(command)


__all__ = ["HANDLERS", "handler_for"]
