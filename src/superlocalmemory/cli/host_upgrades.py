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
        return

    operation = "Applying" if apply else "Previewing"
    print(f"{operation} SLM host upgrades: {', '.join(targets)}")
    if not apply:
        print("No host files will be changed. Re-run with --apply and explicit targets to proceed.")

    for host in targets:
        result = _upgrade_host(host, apply=apply, already_integrated=host in detected)
        status = result.get("status", "error")
        detail = result.get("detail", "")
        print(f"  [{status}] {host}: {detail}")

    if apply:
        print("Restart affected host applications, then run `slm doctor` to verify SLM itself.")


def _detected_hosts(home: Path | None = None) -> list[str]:
    """Return only hosts already containing an SLM-owned integration block."""
    from superlocalmemory.hooks.portable_kit import (
        IDE_MATRIX,
        global_config_path,
        _load_config,
        _ParseError,
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
        from superlocalmemory.hooks.claude_code_hooks import check_status

        if check_status().get("installed"):
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
            and item.get("params", {}).get("serverName") == "superlocalmemory"
            for item in providers
        )
    servers = data.get(desc.server_key, {})
    return isinstance(servers, dict) and "superlocalmemory" in servers


def _upgrade_host(host: str, *, apply: bool, already_integrated: bool) -> dict[str, str]:
    """Refresh assets without degrading a working, host-specific MCP block."""
    if host == "claude-code":
        return {
            "status": "plugin-managed",
            "detail": "run `claude plugin update superlocalmemory@qualixar` in Claude Code",
        }

    if host == "hermes":
        if already_integrated:
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
        from superlocalmemory.hooks.codex_hooks import install_hooks

        assets, hooks = install_assets(dry_run=not apply), install_hooks(dry_run=not apply)
        if not assets.get("success") or not hooks.get("success"):
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
        detail += "; hooks refreshed; MCP block preserved"
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


__all__ = ("cmd_upgrade_hosts", "validate_upgrade_request")
