# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Merge-only upgrade of SLM's own hook entries in host configuration files (4.2.30).

The owner's rule: people already have hooks in their harnesses, so an SLM upgrade may only
replace SLM-owned entries and add missing SLM entries. It never replaces a whole file, never
touches another entry, key or the key order, and never creates hooks in a file that holds no
SLM hook yet (that person never opted in there).

Two files are SLM's to maintain: Claude Code's ``settings.json`` and Codex's ``hooks.json``. A
third kind, the ``slm-*`` skills, subagents and ``/slm-loop`` of a Claude Code wired without the
plugin, is handled by :mod:`superlocalmemory.hooks.claude_code_assets` in the same once-per-version step.
SLM tracks no project-level hook file, so there is none to visit. Plugin-managed copies (the
Claude, Codex, Antigravity, Muse and Copilot plugin caches, the Hermes plugin) belong to their
host and are never written here.

It runs by itself: the daemon calls :func:`run` once after start-up settles, and
:func:`run_once` does the work only once per installed version (``hooks-upgraded.json`` in the
SLM data folder). ``SLM_NO_HOOK_UPGRADE=1`` switches it off. ``slm upgrade-hosts --apply``
calls the same functions on demand.

Per file: one backup (``<name>.bak-slm-<version>``) before the first change, then an atomic
write (temp file, rename). A file that already holds the current entries is not written.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import stat
import time
from pathlib import Path
from typing import Any, Callable, Mapping

logger = logging.getLogger(__name__)

OPT_OUT_ENV = "SLM_NO_HOOK_UPGRADE"
MARKER_NAME = "hooks-upgraded.json"
#: Seconds the daemon waits after start-up before it looks at host files.
FIRST_DELAY_S = 90.0

Result = dict[str, Any]


def disabled(env: Mapping[str, str] | None = None) -> bool:
    """``SLM_NO_HOOK_UPGRADE=1``: no automatic hook upgrade in this process."""
    return (os.environ if env is None else env).get(OPT_OUT_ENV) == "1"


# -- files ----------------------------------------------------------------------------------

def _indent_of(text: str) -> str | int:
    """The indent the file already uses, so a rewrite keeps its look."""
    match = re.search(r'\n([ \t]+)\S', text)
    return match.group(1) if match else 2


def _backup_once(target: Path, version: str) -> Path | None:
    backup = target.with_name(f"{target.name}.bak-slm-{version}")
    if backup.exists():
        return backup
    try:
        shutil.copy2(target, backup)
    except OSError as exc:
        raise OSError(f"could not write the backup {backup.name}: {exc}") from exc
    return backup


class FileChanged(OSError):
    """The file is no longer what was read: someone wrote it while the upgrade worked."""


def _write_atomic(target: Path, data: dict[str, Any], original_text: str) -> None:
    """Replace ``target`` with ``data`` only if it still holds ``original_text`` (F-13).

    The file is read once, merged and replaced; Claude Code or the person may write it in between.
    The text is read again just before the replace, and a difference raises :class:`FileChanged`
    with nothing written, so the other writer's change is never lost.
    """
    text = json.dumps(data, indent=_indent_of(original_text), ensure_ascii=False)
    text += "\n"
    tmp = target.with_name(target.name + ".slm_tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        try:
            os.chmod(tmp, stat.S_IMODE(target.stat().st_mode))
        except OSError:
            pass
        if target.read_text(encoding="utf-8") != original_text:
            raise FileChanged(f"{target.name} changed while it was being upgraded")
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)


def _result(status: str, detail: str, path: Path, **extra: Any) -> Result:
    return {"status": status, "detail": detail, "path": str(path), **extra}


def _read_json(path: Path) -> tuple[dict[str, Any], str] | Result:
    """The parsed file and its text, or a ready ``error`` result."""
    try:
        text = path.read_text(encoding="utf-8")
        data = json.loads(text)
    except (OSError, ValueError) as exc:
        return _result("error", f"cannot read {path.name}: {exc}", path)
    if not isinstance(data, dict):
        return _result("error", f"{path.name} is not a JSON object", path)
    return data, text


def _upgrade_file(
    path: Path, *, version: str, apply: bool,
    has_slm: Callable[[dict], bool], merge: Callable[[dict], dict],
) -> Result:
    """The shared flow: read, skip if no SLM entries, merge, compare, back up, write."""
    if not path.is_file():
        return _result("not-integrated", "no file; none created", path)
    read = _read_json(path)
    if isinstance(read, dict) and "status" in read:
        return read
    data, text = read  # type: ignore[misc]
    if not has_slm(data):
        return _result("not-integrated", "no SLM hooks in this file; left untouched", path)
    try:
        merged = merge(data)
    except (ValueError, TypeError) as exc:
        return _result("error", f"{path.name}: {exc}", path)
    if merged == data:
        return _result("current", "SLM hooks are current", path)
    if not apply:
        return _result("would-update", "would replace or add SLM hook entries only", path)
    target = path.resolve()  # a symlinked file keeps its link
    try:
        backup = _backup_once(target, version)
        _write_atomic(target, merged, text)
    except FileChanged:
        return _result("changed-meanwhile", f"{path.name} changed while it was being upgraded; "
                       "nothing was written, the next start tries again", path)
    except OSError as exc:
        return _result("error", f"could not write {path.name}: {exc}", path)
    return _result("updated", "replaced and added SLM hook entries; other entries untouched",
                   path, backup=str(backup) if backup else None)


# -- the two hosts --------------------------------------------------------------------------

_PLUGIN_PREFIX = "superlocalmemory@"


def claude_plugin_present(settings_path: Path) -> bool:
    """True when SLM's Claude Code plugin is enabled in this settings file or installed beside it.

    Reads only: ``enabledPlugins`` in the same file, then ``plugins/installed_plugins.json`` in the
    same Claude folder. An unreadable file answers False here (the merge path then reads it again
    and reports its own error).
    """
    try:
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
        enabled = settings.get("enabledPlugins") if isinstance(settings, dict) else None
        if isinstance(enabled, dict) and any(
                str(k).startswith(_PLUGIN_PREFIX) and v for k, v in enabled.items()):
            return True
    except (OSError, ValueError):
        pass
    try:
        listed = json.loads((settings_path.parent / "plugins" / "installed_plugins.json")
                            .read_text(encoding="utf-8"))
        plugins = listed.get("plugins", listed) if isinstance(listed, dict) else {}
        return isinstance(plugins, dict) and any(str(k).startswith(_PLUGIN_PREFIX) for k in plugins)
    except (OSError, ValueError):
        return False


def upgrade_claude_settings(path: Path | None = None, *, version: str, apply: bool = True) -> Result:
    """Claude Code ``settings.json``: SLM entries replaced and added, the rest untouched."""
    from superlocalmemory.hooks import claude_code_hooks as claude

    target = path or claude.CLAUDE_SETTINGS
    if claude._disabled_file().exists():
        return _result("skipped", "hooks were removed with `slm hooks remove`", target)
    if claude_plugin_present(target):
        # Found on the owner's Mac, 19 Nov 2026: a leftover `slm session close` entry made the
        # file count as SLM-installed and the full set was added beside the plugin's own hooks,
        # so every SLM hook would have run twice. With the plugin, the plugin owns SLM's hooks.
        return _result("plugin-managed", "SLM's hooks come from the Claude Code plugin; "
                       "settings.json left untouched", target)

    def merge(settings: dict) -> dict:
        return claude._merge_hooks(
            settings, claude._hook_definitions(include_gate=claude.settings_have_gate(settings)))

    return _upgrade_file(target, version=version, apply=apply,
                         has_slm=claude.settings_have_slm_hooks, merge=merge)


def upgrade_codex_hooks(path: Path | None = None, *, version: str, apply: bool = True) -> Result:
    """Codex user ``hooks.json``: SLM-marked entries replaced and added, the rest untouched."""
    from superlocalmemory.hooks import codex_hooks as codex

    return _upgrade_file(path or codex.DEFAULT_HOOKS_PATH, version=version, apply=apply,
                         has_slm=codex.has_slm_entries, merge=codex.merged_hooks_data)


def upgrade_all(*, version: str, apply: bool = True, claude_settings: Path | None = None,
                codex_hooks_path: Path | None = None) -> dict[str, Result]:
    """Every host file SLM maintains, by host name."""
    return {
        "claude-code": upgrade_claude_settings(claude_settings, version=version, apply=apply),
        "codex": upgrade_codex_hooks(codex_hooks_path, version=version, apply=apply),
    }


# -- once per version -----------------------------------------------------------------------

def _data_root(data_root: Path | str | None) -> Path:
    if data_root is not None:
        return Path(data_root)
    from superlocalmemory.infra.data_root import canonical_data_root

    return canonical_data_root()


#: A host with one of these answers is tried again at the next start (the version is not marked done).
_RETRY_STATUSES = ("error", "changed-meanwhile")


def _killed(root: Path, env: Mapping[str, str] | None) -> bool:
    """The 4.2.30 kill switch is on (environment or ``features.json``). Unreadable counts as on:
    a background default writing into a person's files does not guess."""
    try:
        from superlocalmemory.core.feature_flags import FeatureFlags

        flags = FeatureFlags(root) if env is None else FeatureFlags(root, environ=env)
        return bool(flags.killed)
    except Exception:  # noqa: BLE001
        logger.warning("hook upgrade: kill switch unreadable; skipped this start", exc_info=True)
        return True


def marker_version(data_root: Path | str | None = None) -> str | None:
    """The version whose upgrade already ran, or ``None``."""
    try:
        value = json.loads((_data_root(data_root) / MARKER_NAME).read_text(encoding="utf-8"))
        return str(value.get("version")) if isinstance(value, dict) else None
    except (OSError, ValueError):
        return None


def _write_marker(root: Path, version: str, hosts: Mapping[str, Result]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    body = {"version": version, "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "hosts": {name: res["status"] for name, res in hosts.items()}}
    tmp = root / (MARKER_NAME + ".tmp")
    tmp.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, root / MARKER_NAME)


def _upgrade_claude_assets_quietly(claude_settings: Path | None, version: str) -> Result:
    """Skills, subagents and ``/slm-loop`` of a non-plugin Claude Code.

    Only what nobody has to be asked about: missing files are added and files SLM wrote and the
    person never changed are refreshed. A file from before the manifest existed is left for the
    explicit ``slm upgrade-hosts --apply``, which keeps a backup of it.
    """
    from superlocalmemory.hooks import claude_code_assets

    try:
        return claude_code_assets.upgrade_claude_assets(
            claude_settings, version=version, apply=True, adopt_unrecorded=False)
    except Exception as exc:  # noqa: BLE001 - a background default never takes the daemon down
        logger.warning("claude asset upgrade hit an error", exc_info=True)
        return _result("error", f"claude assets: {exc}", claude_settings or Path("."))


def run_once(*, version: str | None = None, data_root: Path | str | None = None,
             env: Mapping[str, str] | None = None, claude_settings: Path | None = None,
             codex_hooks_path: Path | None = None) -> Result | None:
    """Upgrade the host files once per installed version; ``None`` when nothing was due.

    A version is marked done only when no host reported an error, so a locked or broken
    file is tried again at the next daemon start.
    """
    if disabled(env):
        return None
    if version is None:
        from superlocalmemory import __version__ as version
    root = _data_root(data_root)
    if marker_version(root) == version:
        return None
    if _killed(root, env):
        return None  # "turn off everything new in 4.2.30" (F-13): not marked done, it runs once lifted
    hosts = upgrade_all(version=version, claude_settings=claude_settings,
                        codex_hooks_path=codex_hooks_path)
    hosts["claude-code-assets"] = _upgrade_claude_assets_quietly(claude_settings, version)
    if not any(res["status"] in _RETRY_STATUSES for res in hosts.values()):
        try:
            _write_marker(root, version, hosts)
        except OSError:
            logger.warning("hook upgrade: could not write %s", MARKER_NAME, exc_info=True)
    return {"version": version, "hosts": hosts}


async def run(*, first_delay_s: float = FIRST_DELAY_S,
              sleep: Callable[[float], Any] = asyncio.sleep,
              run_once_fn: Callable[..., Result | None] = run_once, **kwargs: Any) -> Result | None:
    """The daemon's task: wait for start-up to settle, then upgrade once. Cancelled with the daemon."""
    if disabled(kwargs.get("env")):
        return None
    await sleep(first_delay_s)
    try:
        outcome = await asyncio.to_thread(run_once_fn, **kwargs)
    except Exception:  # noqa: BLE001 - a background default never takes the daemon down
        logger.warning("automatic hook upgrade hit an error", exc_info=True)
        return None
    if outcome:
        logger.info("automatic hook upgrade: %s", {k: v["status"] for k, v in outcome["hosts"].items()})
    return outcome


__all__ = ("MARKER_NAME", "OPT_OUT_ENV", "disabled", "marker_version", "run", "run_once",
           "upgrade_all", "upgrade_claude_settings", "upgrade_codex_hooks")
