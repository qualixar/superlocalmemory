"""Consented, non-destructive upgrades for existing host integrations.

Package installation owns the SLM executable.  This module owns the separate
operator-approved step that refreshes SLM-owned integration assets.  It never
discovers a host and edits it implicitly: preview is the default and apply
requires either explicit hosts or an explicit all-detected acknowledgement.
"""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from typing import Any


def validate_upgrade_request(*, apply: bool, hosts: list[str], all_detected: bool) -> None:
    """Reject unbounded mutations before any host filesystem access."""
    if apply and not hosts and not all_detected:
        raise ValueError("--apply requires at least one --host or --all-detected")
    if hosts and all_detected:
        raise ValueError("choose explicit --host values or --all-detected, not both")


def cmd_upgrade_hosts(args: Namespace) -> None:
    """Preview or explicitly apply host integration refreshes."""
    hosts = list(getattr(args, "hosts", []) or [])
    apply = bool(getattr(args, "apply", False))
    all_detected = bool(getattr(args, "all_detected", False))
    try:
        validate_upgrade_request(apply=apply, hosts=hosts, all_detected=all_detected)
    except ValueError as exc:
        print(f"Error: {exc}")
        return

    from superlocalmemory.hooks.portable_kit import supported_ides

    detected = set(_detected_hosts())
    targets = sorted(detected) if all_detected else (hosts or sorted(detected))
    managed_hosts = set(supported_ides()) | {"claude-code", "hermes"}
    unknown = sorted(set(targets) - managed_hosts)
    if unknown:
        print(f"Error: unsupported host(s): {', '.join(unknown)}")
        return
    if not targets:
        print("No existing SLM host integrations were detected.")
        print("Run `slm setup` for first-time setup, or use `--host <name>` to target one host.")
        _report_reconnect_apps([])
        return

    from superlocalmemory.cli.measurement_cmd import announce_if_pending
    announce_if_pending()  # 4.2.30 D-28: an upgrade shows the measurement notice once
    operation = "Applying" if apply else "Previewing"
    print(f"{operation} SLM host upgrades: {', '.join(targets)}")
    if not apply:
        print("No host files will be changed. Re-run with --apply and explicit targets to proceed.")

    for host in targets:
        result = _upgrade_host(host, apply=apply, already_integrated=host in detected)
        status = result.get("status", "error")
        detail = result.get("detail", "")
        print(f"  [{status}] {host}: {detail}")
        commands = plugin_update_commands(host)
        if commands:
            note = ("if you use its plugin, its hooks and skills update only through the host: "
                    if host != "hermes" else "its plugin code (and the overlap hook) updates only by reinstalling it: ")
            print(f"      {note}{'  &&  '.join(commands)}")

    _report_reconnect_apps(targets if apply else [])


def _report_reconnect_apps(targets: list[str]) -> None:
    from superlocalmemory.infra.data_root import canonical_data_root
    from superlocalmemory.infra.mcp_processes import HOST_NAMES, snapshot

    running = snapshot(canonical_data_root())
    restart = set(targets) | {
        app["host"] for app in running["apps"]
        if app["stale_version"] or app.get("version_reported") is False
    }
    if restart:
        names = ", ".join(HOST_NAMES.get(host, host) for host in sorted(restart))
        print(f"Apps to restart or reconnect: {names}")
        print("Use Dashboard → Connected apps to reconnect live MCP sessions, "
              "then run `slm doctor`.")


def _detected_hosts(home: Path | None = None) -> list[str]:
    """Return only hosts already containing an SLM-owned integration block."""
    from superlocalmemory.hooks.portable_kit import (
        IDE_MATRIX,
        _load_config,
        _ParseError,
        global_config_path,
    )

    effective_home = home or Path.home()
    detected: list[str] = []
    for host, desc in IDE_MATRIX.items():
        if not desc.fmt:
            continue
        path = global_config_path(desc, effective_home, environ=None if home is None else {})
        try:
            data = _load_config(path, desc.fmt)
        except _ParseError:
            continue
        if _contains_slm_block(data, desc):
            detected.append(host)

    # Claude Code's plugin is host-managed. A local SLM hook installation is
    # the durable evidence that its SLM integration exists; upgrading the
    # plugin remains an explicit Claude Code marketplace action.
    try:
        from superlocalmemory.hooks.claude_code_hooks import (
            _read_settings,
            check_status,
            settings_have_slm_hooks,
        )

        if check_status().get("installed") or settings_have_slm_hooks(_read_settings()):
            detected.append("claude-code")
    except Exception:
        pass
    # A skills-only install (SLM's skills and agents in ~/.claude, no hooks) is the one this refresh
    # exists for; it is found by the same test the refresh itself uses.
    try:
        from superlocalmemory.hooks.claude_code_assets import is_integrated

        if is_integrated():
            detected.append("claude-code")
    except Exception:
        pass

    # Hermes has a native YAML mapping rather than the portable-kit schema.
    # Detect it for reporting, but never overwrite it with an invented format.
    hermes_path = effective_home / ".hermes" / "config.yaml"
    try:
        hermes = _load_config(hermes_path, "yaml")
        servers = hermes.get("mcp_servers", {})
        if isinstance(servers, dict) and "superlocalmemory" in servers:
            detected.append("hermes")
    except _ParseError:
        pass
    return sorted(set(detected))


def _contains_slm_block(data: dict[str, Any], desc: Any) -> bool:
    if desc.fmt == "yaml":
        providers = data.get(desc.server_key, [])
        return isinstance(providers, list) and any(
            isinstance(item, dict)
            and item.get("name") == "superlocalmemory"
            for item in providers
        )
    servers = data.get(desc.server_key, {})
    return isinstance(servers, dict) and "superlocalmemory" in servers


#: Hermes's documented grant for a plugin's MCP access; it has no prompt for it.
_HERMES_GRANT_LINE = "plugins.entries.superlocalmemory.mcp_allowlist: [superlocalmemory]"


def _hermes_config_path() -> Path:
    import os

    return Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes") / "config.yaml"


def hermes_plugin_grant(path: Path, *, apply: bool) -> dict[str, str]:
    """Grant the SLM MCP server to the SLM Hermes plugin (``mcp_allowlist``).

    Hermes refuses every plugin MCP call without it, so recall, session
    tracking and turn capture silently did nothing on installs that followed
    the guide. Writes only that key, keeps a backup, and never rewrites a
    config holding hand-written comments (a YAML rewrite would drop them).
    """
    import re
    import time

    import yaml

    fix = f"set {_HERMES_GRANT_LINE} in {path}, then restart Hermes"
    if not path.is_file():
        return {"status": "not-installed", "detail": "no Hermes config"}
    try:
        text = path.read_text(encoding="utf-8")
        data = yaml.safe_load(text) or {}
    except (OSError, yaml.YAMLError) as exc:
        return {"status": "blocked", "detail": f"cannot read {path}: {exc}"}
    plugins = data.get("plugins") if isinstance(data, dict) else None
    enabled = plugins.get("enabled") if isinstance(plugins, dict) else None
    if not isinstance(enabled, list) or "superlocalmemory" not in enabled:
        return {"status": "not-installed", "detail": "the SLM Hermes plugin is not enabled"}
    entries = plugins.get("entries") if isinstance(plugins.get("entries"), dict) else {}
    entry = entries.get("superlocalmemory") if isinstance(entries.get("superlocalmemory"), dict) else {}
    allow = entry.get("mcp_allowlist") if isinstance(entry.get("mcp_allowlist"), list) else []
    if "superlocalmemory" in allow:
        return {"status": "verified", "detail": "the Hermes plugin may reach SLM"}
    if not apply:
        return {"status": "preview", "detail": "would grant SLM to the Hermes plugin: " + fix}
    if re.search(r"(?m)(^\s*#|\s#)", text):
        return {"status": "blocked", "detail": "config has hand-written comments; " + fix}
    entry = {**entry, "mcp_allowlist": [*allow, "superlocalmemory"]}
    data["plugins"] = {**plugins, "entries": {**entries, "superlocalmemory": entry}}
    backup = path.with_name(f"{path.name}.bak-slm-{time.strftime('%Y%m%dT%H%M%S')}")
    backup.write_text(text, encoding="utf-8")
    tmp = path.with_name(path.name + ".slm-tmp")
    tmp.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    tmp.replace(path)
    reread = yaml.safe_load(path.read_text(encoding="utf-8"))
    granted = reread["plugins"]["entries"]["superlocalmemory"].get("mcp_allowlist", [])
    if "superlocalmemory" not in granted:
        backup.replace(path)
        return {"status": "blocked", "detail": "the grant did not persist; " + fix}
    return {"status": "updated",
            "detail": f"granted SLM to the Hermes plugin (backup {backup.name}); restart Hermes"}


def hermes_plugin_finding() -> tuple[str, str, str] | None:
    """``slm doctor``: whether the SLM Hermes plugin may reach SLM (None: no plugin)."""
    grant = hermes_plugin_grant(_hermes_config_path(), apply=False)
    if grant["status"] == "not-installed":
        return None
    if grant["status"] == "verified":
        return "PASS", grant["detail"], ""
    return ("WARN", "Hermes has not granted SLM to the SLM plugin, so its recall, "
            "session tracking and turn capture are off", "slm upgrade-hosts --host hermes --apply")


def _hook_upgrade():
    from superlocalmemory.hooks import host_hook_upgrade

    return host_hook_upgrade


def _version() -> str:
    from superlocalmemory import __version__

    return __version__


#: What a hook-file result means, in plain words, for the upgrade line.
_HOOK_WORDS = {
    "updated": "SLM entries replaced or added, your other hooks untouched",
    "would-update": "SLM entries would be replaced or added, your other hooks untouched",
    "current": "already current",
    "not-integrated": "no SLM hooks there, so none added (`slm codex install` adds them)",
    "skipped": "left alone (you removed SLM hooks)",
    "error": "could not be read",
    "plugin-managed": ("left untouched: SLM's hooks come from the Claude Code plugin, which "
                       "updates with `claude plugin update superlocalmemory@qualixar`"),
    "changed-meanwhile": "changed while it was being upgraded, so nothing was written (try again)",
}

_HOOK_STATUS = {"updated": "updated", "would-update": "preview", "current": "verified",
                "not-integrated": "verified", "skipped": "verified", "error": "blocked",
                "plugin-managed": "verified", "changed-meanwhile": "blocked"}


def _hook_file_result(result: dict[str, Any], label: str, add_hint: str) -> dict[str, str]:
    status = result["status"]
    detail = f"{label}: {_HOOK_WORDS[status]}"
    if status == "not-integrated":
        detail = f"{label}: no SLM hooks there, so none added ({add_hint})"
    if status == "error":
        detail = f"{label}: {result['detail']}; the file was not changed"
    if result.get("backup"):
        detail += f" (backup {Path(result['backup']).name})"
    return {"status": _HOOK_STATUS[status], "detail": detail}


#: Hosts whose hooks come from a host-managed plugin. SLM cannot write those files, so the
#: exact command that moves each to the running version is printed instead (docs/plugins.md).
def plugin_update_commands(host: str, version: str | None = None) -> list[str]:
    """The commands that pick up this release's plugin hooks on ``host`` (empty: none)."""
    version = version or _version()
    repo = "https://github.com/qualixar/superlocalmemory.git"
    table = {
        "claude-code": ["claude plugin marketplace update qualixar",
                        "claude plugin update superlocalmemory@qualixar"],
        # `add` keeps an existing snapshot as it was ("already added"); `upgrade` refreshes it.
        "codex": ["codex plugin marketplace add qualixar/superlocalmemory --ref main",
                  "codex plugin marketplace upgrade qualixar",
                  "codex plugin add superlocalmemory-codex@qualixar"],
        "antigravity": ['agy plugin install "$(npm root -g)/superlocalmemory/antigravity-plugin"'],
        "muse": ['muse plugins install "$(npm root -g)/superlocalmemory/muse-plugin"',
                 "muse plugins approve superlocalmemory"],
        "hermes": [f"hermes plugins install '{repo}#hermes-plugin' --force --ref "
                   f"\"$(git ls-remote {repo} 'refs/tags/v{version}^{{}}' | cut -f1)\""],
    }
    return list(table.get(host, []))


def host_hooks_finding() -> tuple[str, str, str]:
    """``slm doctor``: whether SLM's own hook entries are current, per host file (read-only)."""
    results = _hook_upgrade().upgrade_all(version=_version(), apply=False)
    names = {"claude-code": "Claude Code settings.json", "codex": "Codex hooks.json"}
    stale = [h for h, r in results.items() if r["status"] == "would-update"]
    broken = [h for h, r in results.items() if r["status"] == "error"]
    current = [h for h, r in results.items() if r["status"] == "current"]
    if stale:
        fixes = "  &&  ".join(f"slm upgrade-hosts --host {h} --apply" for h in stale)
        return ("WARN", "SLM hooks are older than this release in: "
                + ", ".join(names[h] for h in stale)
                + " (the daemon refreshes them by itself once per version; your own hooks are never touched)",
                fixes)
    if broken:
        return ("WARN", "could not read " + ", ".join(names[h] for h in broken)
                + "; SLM hooks there were not checked", "fix the file, then run `slm upgrade-hosts --apply --host "
                + broken[0] + "`")
    if current:
        return "PASS", "SLM hooks are current in: " + ", ".join(names[h] for h in current), ""
    return ("PASS", "no SLM hooks in Claude Code settings.json or Codex hooks.json "
            "(plugin hosts take hooks from their plugin)", "")


def claude_assets_finding() -> tuple[str, str, str] | None:
    """``slm doctor``: are SLM's skills, subagents and ``/slm-loop`` current in a non-plugin Claude Code.

    Read-only. ``None`` when there is nothing to say: the plugin owns them, or SLM is not there.
    """
    from superlocalmemory.hooks.claude_code_assets import upgrade_claude_assets

    result = upgrade_claude_assets(version=_version(), apply=False, adopt_unrecorded=True)
    status = result["status"]
    if status in ("plugin-managed", "not-integrated"):
        return None
    fix = "slm upgrade-hosts --host claude-code --apply"
    if status == "error":
        return "WARN", f"Claude Code skills and agents could not be checked: {result['detail']}", ""
    if status == "would-update":
        count = len(result["added"]) + len(result["refreshed"]) + len(result["adopted"])
        return ("WARN", f"{count} SLM skill, agent or command file(s) in ~/.claude are older than this "
                "release or missing (the daemon adds missing ones by itself; yours are never "
                "replaced without a backup)", fix)
    return "PASS", "SLM skills, agents and command in ~/.claude are current" + _asset_notes(result), ""


def plugin_hosts_finding() -> tuple[str, str, str] | None:
    """``slm doctor``: the exact command each connected plugin host needs for this release."""
    lines = []
    for host in _detected_hosts():
        commands = plugin_update_commands(host)
        if commands:
            lines.append(f"{host}: " + "  &&  ".join(commands))
    if not lines:
        return None
    return ("PASS", "plugin hosts pick up this release's hooks only through their own plugin; "
            "if you use a plugin there, run: " + " | ".join(lines), "")


#: Severity of a per-part status, highest first: the line reports the worst part.
_STATUS_ORDER = ("blocked", "updated", "preview", "verified")


def _asset_words(result: dict[str, Any], apply: bool) -> str:
    """Plain words for the skills, subagents and command part of Claude Code's upgrade line."""
    status = result["status"]
    if status == "plugin-managed":
        return "come from the Claude Code plugin, so ~/.claude was left untouched"
    if status == "not-integrated":
        return "no SLM skills or hooks there, so none added"
    if status == "error":
        return result["detail"]
    added, refreshed, adopted = result["added"], result["refreshed"], result["adopted"]
    parts = []
    if added:
        parts.append(f"{'would add' if not apply else 'added'} {len(added)}")
    if refreshed or adopted:
        parts.append(f"{'would refresh' if not apply else 'refreshed'} {len(refreshed) + len(adopted)}")
    text = " and ".join(parts) if parts else (
        "already current" if status == "current" else "nothing changed")
    if adopted:
        text += (f" ({len(adopted)} from before SLM kept a record, "
                 f"each copied aside as .bak-slm-{_version()} first)")
    return text + _asset_notes(result)


def _asset_notes(result: dict[str, Any]) -> str:
    notes = ""
    if result["preserved"]:
        names = ", ".join(result["preserved"])
        notes += (f"; kept {len(result['preserved'])} file(s) you changed after SLM wrote them "
                  f"({names}); move one aside and re-run to take the new version")
    if result["held"]:
        notes += (f"; {len(result['held'])} older file(s) not written by a recorded SLM version were "
                  "left for `slm upgrade-hosts --apply`, which keeps a backup of each")
    if result["outside"]:
        notes += (f"; {len(result['outside'])} path(s) resolve outside ~/.claude and were NOT "
                  f"refreshed ({result['outside'][0]})")
    if result["removed"]:
        notes += f"; {len(result['removed'])} file(s) you removed were not put back"
    if result["blocked"]:
        notes += (f"; {len(result['blocked'])} path(s) are a folder or a broken link where SLM "
                  f"expects a file and were left alone ({result['blocked'][0]})")
    return notes


def _asset_listing(result: dict[str, Any], apply: bool) -> str:
    """One indented line per file that would be, or was, added or refreshed."""
    tense = {"added": "added" if apply else "would add",
             "refreshed": "refreshed" if apply else "would refresh",
             "adopted": "refreshed" if apply else "would refresh"}
    lines = [f"\n        {tense[kind]}: {rel}" for kind in tense for rel in result[kind]]
    return "".join(lines)


def _asset_status(result: dict[str, Any]) -> str:
    return {"updated": "updated", "would-update": "preview", "error": "blocked",
            "changed-meanwhile": "blocked"}.get(result["status"], "verified")


def _upgrade_claude_code(*, apply: bool) -> dict[str, str]:
    """Claude Code's hooks in settings.json plus its skills, subagents and ``/slm-loop``.

    Both parts are always looked at, so the line cannot say "already current" for one while the
    other is stale. With SLM's plugin installed both are the plugin's and nothing here is written.
    """
    from superlocalmemory.hooks.claude_code_assets import upgrade_claude_assets

    settings = _hook_upgrade().upgrade_claude_settings(version=_version(), apply=apply)
    assets = upgrade_claude_assets(version=_version(), apply=apply, adopt_unrecorded=True)
    if settings["status"] == "plugin-managed":
        return {"status": "verified", "detail": "Claude Code settings.json, skills, agents and "
                "commands: come from the Claude Code plugin, so ~/.claude was left untouched"}
    part = _hook_file_result(
        settings, "Claude Code settings.json",
        "`slm hooks install` adds them, or use the Claude Code plugin")
    detail = f"{part['detail']}; skills/agents/commands: {_asset_words(assets, apply)}"
    detail += _asset_listing(assets, apply)
    statuses = (part["status"], _asset_status(assets))
    status = next(s for s in _STATUS_ORDER if s in statuses)
    return {"status": status, "detail": detail}


def _upgrade_host(host: str, *, apply: bool, already_integrated: bool) -> dict[str, str]:
    """Refresh assets without degrading a working, host-specific MCP block."""
    if host == "claude-code":
        return _upgrade_claude_code(apply=apply)

    if host == "hermes":
        if already_integrated:
            # Connected is not enough: without the grant the plugin is inert.
            grant = hermes_plugin_grant(_hermes_config_path(), apply=apply)
            if grant["status"] in ("updated", "preview", "blocked"):
                return grant
            return {
                "status": "verified",
                "detail": "existing native Hermes SLM mapping preserved",
            }
        return {
            "status": "blocked",
            "detail": "Hermes is not connected; follow docs/ide-setup.md before retrying",
        }

    if host == "codex" and already_integrated:
        from superlocalmemory.hooks.codex_assets import install_assets

        assets = install_assets(dry_run=not apply)
        hooks = _hook_upgrade().upgrade_codex_hooks(version=_version(), apply=apply)
        if not assets.get("success") or hooks["status"] == "error":
            return {"status": "blocked", "detail": "could not refresh SLM-owned Codex assets"}
        action = "would refresh" if not apply else "refreshed"
        # Name what is actually written.  Saying "refreshed skills" while the
        # copies under ~/.agents/skills are not what this Codex reads turns a
        # no-op into a success report.
        written = assets.get("skills_written") or []
        agents_written = assets.get("agents_written") or []
        preserved = assets.get("agents_preserved") or []
        elsewhere = assets.get("skills_read_elsewhere") or []

        detail = f"{action} {len(written)} skill file(s) and {len(agents_written)} subagent file(s)"
        if written:
            detail += f" under {Path(written[0]).parent.parent}"
        detail += f"; hooks: {_HOOK_WORDS[hooks['status']]}; MCP block preserved"
        if preserved:
            names = ", ".join(Path(x).name for x in preserved)
            detail += f"; preserved {len(preserved)} subagent file(s) not written by SLM ({names})"
        if elsewhere:
            detail += (
                f"; {len(elsewhere)} skill path(s) under ~/.codex/skills resolve outside "
                f"the written location and were NOT refreshed ({Path(elsewhere[0]).parent})"
            )
        return {"status": "updated" if apply else "preview", "detail": detail}

    if already_integrated:
        return {
            "status": "verified",
            "detail": "existing SLM MCP block preserved; no portable host assets require refresh",
        }

    # An explicitly named, not-yet-connected host is a first-time connection.
    # This branch is never reached by --all-detected.
    from superlocalmemory.hooks.portable_kit import connect_ide

    result = connect_ide(host, dry_run=not apply)
    if result.get("error"):
        return {"status": "blocked", "detail": str(result["error"])}
    action = result.get("mcp_config", "unknown")
    if not apply:
        return {"status": "preview", "detail": f"would refresh SLM MCP block ({action})"}
    return {"status": "updated", "detail": f"SLM MCP block {action}"}


__all__ = ("cmd_upgrade_hosts", "plugin_update_commands", "validate_upgrade_request")
