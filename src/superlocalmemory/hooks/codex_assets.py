"""Explicit installer for SLM-owned Codex skills and subagents."""

from __future__ import annotations

import shutil
import sysconfig
from pathlib import Path

SKILLS = ("slm-cache", "slm-compress", "slm-graph", "slm-recall", "slm-remember", "slm-session", "slm-status")

# Codex subagent files written to ~/.codex/agents (content built by _agent_files()).
AGENTS = ("slm-memory-advisor.toml", "slm-optimize-advisor.toml")

_MEMORY_ADVISOR_TOML = (
    'name = "slm-memory-advisor"\n'
    'description = "Use SuperLocalMemory safely: initialize once, recall before remember, and store only durable atomic facts."\n'
    'instructions = "Use SLM for memory discipline only. Check results before claiming success; preserve private scope unless the user explicitly asks to share."\n'
)


def _source_root() -> Path:
    development = Path(__file__).resolve().parents[3] / "plugin-src" / "skills"
    if development.exists():
        return development
    installed = Path(sysconfig.get_path("data")) / "share" / "superlocalmemory" / "codex" / "skills"
    if installed.exists():
        return installed
    raise FileNotFoundError("Bundled Codex skills were not found in this installation")


def _agents_source_root() -> Path | None:
    development = Path(__file__).resolve().parents[3] / "plugin-src" / "agents"
    if development.exists():
        return development
    installed = Path(sysconfig.get_path("data")) / "share" / "superlocalmemory" / "codex" / "agents"
    return installed if installed.exists() else None


def _optimize_advisor_toml() -> str:
    """Build the optimize-advisor TOML from the canonical advisor doc so Codex
    ships the FULL decision rules (the 8-rule tree), not a one-line stub. Falls
    back to a short instruction only if the source doc is unavailable.
    """
    description = (
        "Apply SuperLocalMemory's no-proxy context-optimization rules — reversible "
        "compression of large tool output and KV-caching of repeated reads/searches."
    )
    body = ""
    root = _agents_source_root()
    if root is not None:
        src = root / "slm-optimize-advisor.md"
        if src.exists():
            text = src.read_text(encoding="utf-8")
            if text.startswith("---"):  # strip YAML frontmatter, keep the guidance body
                end = text.find("\n---", 3)
                if end != -1:
                    text = text[end + 4:]
            body = text.strip()
    if not body:
        body = (
            "Reduce context-window pressure with the Surface-B tools (reversible CCR "
            "compression + a per-agent KV cache); fail-open — never block the task."
        )
    # TOML literal multi-line string ('''...'''): no escape processing, and the
    # advisor body contains no ''' sequence.
    return (
        'name = "slm-optimize-advisor"\n'
        f'description = "{description}"\n'
        f"instructions = '''\n{body}\n'''\n"
    )


def _agent_files() -> dict:
    """Return {filename: TOML content} for the Codex subagents."""
    return {
        "slm-memory-advisor.toml": _MEMORY_ADVISOR_TOML,
        "slm-optimize-advisor.toml": _optimize_advisor_toml(),
    }


def install_assets(*, home: Path | None = None, dry_run: bool = False) -> dict:
    """Copy only named SLM assets; never rewrite user-owned assets."""
    home = home or Path.home()
    source = _source_root()
    missing = [skill for skill in SKILLS if not (source / skill / "SKILL.md").exists()]
    if missing:
        return {"success": False, "errors": [f"missing bundled skills: {', '.join(missing)}"]}
    if dry_run:
        return {"success": True, "skills": list(SKILLS), "agents": list(AGENTS), "dry_run": True}
    skills_root, agents_root = home / ".agents" / "skills", home / ".codex" / "agents"
    skills_root.mkdir(parents=True, exist_ok=True)
    agents_root.mkdir(parents=True, exist_ok=True)
    for skill in SKILLS:
        target = skills_root / skill
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / skill / "SKILL.md", target / "SKILL.md")
    for filename, content in _agent_files().items():
        (agents_root / filename).write_text(content, encoding="utf-8")
    return {"success": True, "skills": list(SKILLS), "agents": list(AGENTS), "dry_run": False}


def remove_assets(*, home: Path | None = None, dry_run: bool = False) -> dict:
    """Remove only the known SLM directories and files."""
    home = home or Path.home()
    targets = [home / ".agents" / "skills" / skill for skill in SKILLS]
    targets += [home / ".codex" / "agents" / agent for agent in AGENTS]
    existing = [target for target in targets if target.exists()]
    if not dry_run:
        for target in existing:
            shutil.rmtree(target) if target.is_dir() else target.unlink()
    return {"success": True, "removed": [str(x) for x in existing], "dry_run": dry_run}


def status_assets(*, home: Path | None = None) -> dict:
    home = home or Path.home()
    skills = [x for x in SKILLS if (home / ".agents" / "skills" / x / "SKILL.md").exists()]
    agents = [x for x in AGENTS if (home / ".codex" / "agents" / x).exists()]
    return {"installed": len(skills) == len(SKILLS) and len(agents) == len(AGENTS), "skills": skills, "agents": agents}
