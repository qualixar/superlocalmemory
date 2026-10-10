"""Explicit installer for SLM-owned Codex skills and subagents."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import sysconfig
from pathlib import Path

SKILLS = (
    "slm-cache",
    "slm-compress",
    "slm-governance",
    "slm-graph",
    "slm-loop",
    "slm-media",
    "slm-mesh",
    "slm-profile",
    "slm-recall",
    "slm-remember",
    "slm-scope",
    "slm-session",
    "slm-status",
    "slm-web-access",
)

# Codex subagent files written to ~/.codex/agents (content built by _agent_files()).
AGENTS = ("slm-memory-advisor.toml", "slm-optimize-advisor.toml",
          "slm-governance-advisor.toml", "slm-loop-runner.toml")

#: Digests of the agent files this installer last wrote, kept beside them.  An
#: agent file whose current content still matches its recorded digest is ours to
#: refresh; anything else belongs to whoever changed it and is left alone.  A
#: file with no recorded digest predates this manifest and is treated as theirs,
#: which is the safe reading: this installer once replaced two hand-maintained
#: 4.9 KB advisors with its own one-line stubs and there was nothing to restore
#: from.
MANIFEST_NAME = ".slm-managed.json"

#: Used only when the advisor source document is unavailable — an installation
#: that ships no agent sources still gets a usable, if terse, subagent.
_FALLBACKS = {
    "slm-memory-advisor.toml": (
        "Use SuperLocalMemory safely: initialize once, recall before remember, "
        "and store only durable atomic facts.",
        "Use SLM for memory discipline only. Check results before claiming success; "
        "preserve private scope unless the user explicitly asks to share.",
    ),
    "slm-optimize-advisor.toml": (
        "Apply SuperLocalMemory's no-proxy context-optimization rules — reversible "
        "compression of large tool output and KV-caching of repeated reads/searches.",
        "Reduce context-window pressure with the Surface-B tools (reversible CCR "
        "compression + a per-agent KV cache); fail-open — never block the task.",
    ),
    "slm-governance-advisor.toml": (
        "Advise on SuperLocalMemory scope, roles, retention and GDPR erasure.",
        "Keep memories personal unless the user asks to share; never bypass a governance "
        "control; prefer the least-permissive safe action.",
    ),
    "slm-loop-runner.toml": (
        "Run a task as a bounded loop until an independent gate passes.",
        "Iterate only until the gate passes or the bound is reached, and report the exact "
        "terminal status; never report success the gate did not confirm.",
    ),
}


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


def _split_frontmatter(text: str) -> tuple[str, str]:
    """Return (frontmatter, body); frontmatter is "" when the doc has none."""
    if not text.startswith("---"):
        return "", text.strip()
    end = text.find("\n---", 3)
    if end == -1:
        return "", text.strip()
    return text[3:end], text[end + 4:].strip()


def _description(frontmatter: str, default: str) -> str:
    """Read `description:`, joining YAML folded (`>`) blocks onto one line."""
    folded = re.search(r"^description:\s*>[-+]?\s*\n((?:[ \t]+\S.*\n?)+)", frontmatter, re.M)
    if folded:
        return " ".join(line.strip() for line in folded.group(1).splitlines() if line.strip())
    inline = re.search(r"^description:\s*(\S.*)$", frontmatter, re.M)
    return inline.group(1).strip() if inline else default


def _agent_toml(filename: str) -> str:
    """Build one subagent's TOML, preferring the canonical advisor document so
    Codex ships the advisor's full decision rules rather than a summary of them.
    """
    name = filename.removesuffix(".toml")
    default_description, default_body = _FALLBACKS[filename]
    description, body = default_description, default_body

    root = _agents_source_root()
    if root is not None:
        source = root / f"{name}.md"
        if source.exists():
            frontmatter, source_body = _split_frontmatter(source.read_text(encoding="utf-8"))
            if source_body:
                description = _description(frontmatter, default_description)
                body = source_body

    # description is a TOML basic string, so quotes and backslashes must be
    # escaped — the advisor descriptions really do contain quoted questions.
    escaped = description.replace("\\", "\\\\").replace('"', '\\"')
    # A literal multi-line string ('''...''') performs no escape processing,
    # which keeps the advisor markdown byte-exact.  It cannot contain the
    # sequence that closes it, so fall back to an escaped basic string in that
    # case rather than editing the advisor's own text.
    if "'''" in body:
        basic = body.replace("\\", "\\\\").replace('"', '\\"')
        instructions = '"""\n' + basic + '\n"""'
    else:
        instructions = "'''\n" + body + "\n'''"
    return f'name = "{name}"\ndescription = "{escaped}"\ninstructions = {instructions}\n'


def _agent_files() -> dict:
    """Return {filename: TOML content} for the Codex subagents."""
    return {filename: _agent_toml(filename) for filename in AGENTS}


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_manifest(agents_root: Path) -> dict:
    try:
        loaded = json.loads((agents_root / MANIFEST_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _is_ours(target: Path, manifest: dict) -> bool:
    """True when this installer may overwrite `target`.

    Either it does not exist yet, or its content is byte-identical to what this
    installer last recorded writing there.
    """
    if not target.exists():
        return True
    recorded = manifest.get(target.name)
    if not recorded:
        return False
    try:
        return _digest(target.read_text(encoding="utf-8")) == recorded
    except OSError:
        return False


def _skills_read_elsewhere(home: Path, skills_root: Path) -> list[str]:
    """Skill paths Codex may read that this installer does not write.

    Some setups point ~/.codex/skills/<name> at a checkout instead of using the
    copies under ~/.agents/skills, in which case writing the copies refreshes
    nothing Codex will load.  Report those paths so the caller can say so rather
    than claiming a refresh it did not perform.
    """
    elsewhere = []
    for skill in SKILLS:
        candidate = home / ".codex" / "skills" / skill
        if not candidate.exists() and not candidate.is_symlink():
            continue
        resolved = candidate.resolve() if candidate.is_symlink() else candidate
        if resolved != (skills_root / skill).resolve():
            elsewhere.append(str(candidate))
    return elsewhere


def install_assets(*, home: Path | None = None, dry_run: bool = False, force: bool = False) -> dict:
    """Copy only named SLM assets; never rewrite user-owned assets.

    An agent file that this installer did not write, or that has been edited
    since it did, is preserved and reported under ``agents_preserved`` rather
    than overwritten.  Pass ``force=True`` to overwrite anyway, which first
    copies the existing file aside with a ``.bak`` suffix.
    """
    home = home or Path.home()
    source = _source_root()
    missing = [skill for skill in SKILLS if not (source / skill / "SKILL.md").exists()]
    if missing:
        return {"success": False, "errors": [f"missing bundled skills: {', '.join(missing)}"]}

    skills_root, agents_root = home / ".agents" / "skills", home / ".codex" / "agents"
    manifest = _read_manifest(agents_root)
    planned = _agent_files()

    writable, preserved = [], []
    for filename, content in planned.items():
        target = agents_root / filename
        if force or _is_ours(target, manifest):
            writable.append((filename, content, target))
        else:
            preserved.append(str(target))

    skill_targets = [skills_root / skill / "SKILL.md" for skill in SKILLS]
    result = {
        "success": True,
        "dry_run": dry_run,
        "skills": list(SKILLS),
        "agents": [filename for filename, _, _ in writable],
        "skills_written": [str(path) for path in skill_targets],
        "agents_written": [str(target) for _, _, target in writable],
        "agents_preserved": preserved,
        "skills_read_elsewhere": _skills_read_elsewhere(home, skills_root),
    }
    if dry_run:
        return result

    skills_root.mkdir(parents=True, exist_ok=True)
    agents_root.mkdir(parents=True, exist_ok=True)
    for skill in SKILLS:
        target = skills_root / skill
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / skill / "SKILL.md", target / "SKILL.md")
    for filename, content, target in writable:
        if force and target.exists():
            shutil.copy2(target, target.with_suffix(target.suffix + ".bak"))
        target.write_text(content, encoding="utf-8")
        manifest[filename] = _digest(content)
    if writable:
        (agents_root / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return result


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
