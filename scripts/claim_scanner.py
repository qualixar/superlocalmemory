#!/usr/bin/env python3
"""Fail closed on public SuperLocalMemory claims rejected by the V3.7 audit."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Rule:
    rule_id: str
    pattern: re.Pattern[str]
    replacement: str


RULES = (
    Rule(
        "peer-review-status",
        re.compile(r"peer[ -]?review(?:ed)?", re.IGNORECASE),
        "Call the three papers public arXiv preprints; do not imply venue review.",
    ),
    Rule(
        "unproven-scale",
        re.compile(
            r"(?:1\s*M\+?|one million|million)\s+(?:memor(?:y|ies)|entries|facts)|"
            r"(?:five|5)\s+years?\s+(?:of\s+)?daily\s+use|zero\s+slowdown",
            re.IGNORECASE,
        ),
        "Remove million-memory, five-year, and zero-slowdown release claims.",
    ),
    Rule(
        "aggregate-compression",
        re.compile(r"60\s*[-–—]\s*95\s*%", re.IGNORECASE),
        "Use content-specific measured behavior; safe JSON/code paths may yield no reduction.",
    ),
    Rule(
        "embedding-compression-as-product",
        re.compile(
            r"(?:up\s+to\s+)?32\s*[x×].{0,45}(?:storage|memory|cold|compression)|"
            r"(?:storage|memory|cold|compression).{0,45}(?:up\s+to\s+)?32\s*[x×]",
            re.IGNORECASE,
        ),
        "Limit 32x wording to a historical selected-embedding representation experiment.",
    ),
    Rule(
        "integration-count",
        re.compile(
            r"(?:16|17)\+\s+(?:[A-Za-z-]+\s+){0,2}(?:tools|IDEs|clients|integrations|environments)|"
            r"(?:tools|IDEs|clients|integrations|environments).{0,80}(?:16|17)\+",
            re.IGNORECASE,
        ),
        "Name documented clients; publish a count only from the release client matrix.",
    ),
    Rule(
        "current-channel-count",
        re.compile(r"\b(?:6|7|six|seven)[ -]channel(?:s)?\b", re.IGNORECASE),
        "Describe five candidate producers plus graph-based score enhancement.",
    ),
    Rule(
        "backend-production-overclaim",
        re.compile(
            r"CozoDB\s*\+\s*LanceDB|"
            r"(?:CozoDB|LanceDB).{0,45}(?:scale[- ]ready|production|million)|"
            r"(?:scale[- ]ready|production|million).{0,45}(?:CozoDB|LanceDB)",
            re.IGNORECASE,
        ),
        "Label CozoDB and LanceDB as optional or experimental until normal-path proof exists.",
    ),
    Rule(
        "absolute-network-privacy",
        re.compile(
            r"100\s*%\s+local|(?:no|zero)\s+(?:cloud|network)\s+(?:calls?|dependency)|"
            r"(?:no|zero)\s+telemetry|nothing\s+(?:is\s+|ever\s+)?sent|"
            r"(?:data|memories?|content|history|code context)\s+never\s+leaves|"
            r"never\s+leaves\s+(?:your|the)\s+(?:machine|device|laptop|network)",
            re.IGNORECASE,
        ),
        "Qualify local Mode A memory-content behavior and disclose optional networked features.",
    ),
    Rule(
        "compliance-certification",
        re.compile(
            r"(?:full\s+)?EU\s+AI\s+Act(?:\s*\([^)]*\))?\s+(?:compliant|compliance)|"
            r"compliant\s+(?:with|in)\s+(?:the\s+)?EU\s+AI\s+Act|"
            r"Modes?\s+[AB](?:\s+and\s+[AB])?\s+pass(?:es)?\s+all\s+checks",
            re.IGNORECASE,
        ),
        "Describe technical controls; compliance depends on the deployment and operator.",
    ),
    Rule(
        "unqualified-locomo-74",
        re.compile(r"74\.8\s*%", re.IGNORECASE),
        "Mark this historical and state local retrieval plus GPT-4.1-mini answer construction.",
    ),
    Rule(
        "unqualified-locomo-87",
        re.compile(r"87\.7\s*%", re.IGNORECASE),
        "Mark this historical: 81 questions, one conversation, cloud-assisted components.",
    ),
    Rule(
        "fisher-absolute",
        re.compile(
            r"every\s+recall\s+uses.{0,100}Fisher|"
            r"Fisher.{0,100}(?:instead\s+of|not)\s+cosine|"
            r"replac(?:e|es|ing).{0,60}cosine.{0,60}Fisher",
            re.IGNORECASE,
        ),
        "Dense candidates use cosine; Fisher-derived terms inform later scoring.",
    ),
    Rule(
        "single-sqlite-portability",
        re.compile(r"(?:one|single|a)\s+SQLite\s+(?:file|database)", re.IGNORECASE),
        "Core memory is SQLite-backed; document additional state, indexes, models, and logs.",
    ),
    Rule(
        "cache-default",
        re.compile(
            r"exact[- ]match\s+(?:cache\s+)?default|"
            r"(?:engine\s+active|facts\s+indexed).{0,80}cache:\s*on",
            re.IGNORECASE,
        ),
        "Optimize caching is available only when explicitly enabled.",
    ),
    Rule(
        "unsupported-superlative",
        re.compile(
            r"world['’]s\s+first|best[- ]in[- ]class|highest\s+(?:local|score)|"
            r"only\s+(?:zero|local|publicly|local-first)|the\s+first\s+AI\s+agent\s+memory",
            re.IGNORECASE,
        ),
        "Replace market superlatives with a release-linked, testable contract.",
    ),
    Rule(
        "mathematical-guarantees",
        re.compile(r"mathematical\s+guarantees", re.IGNORECASE),
        "Use information-geometric foundations; do not imply system-wide guarantees.",
    ),
    Rule(
        "license-inconsistency",
        re.compile(
            r"Elastic(?:-|\s+)2\.0|Elastic\s+License\s+2\.0",
            re.IGNORECASE,
        ),
        "Use AGPL-3.0-or-later on current shipped surfaces; preserve old licenses only in clearly historical archives.",
    ),
)

TEXT_SUFFIXES = {".md", ".mdx", ".astro", ".tsx", ".ts", ".js", ".mjs", ".py", ".toml", ".json", ".txt"}
SKIP_PARTS = {
    ".astro",
    ".backup",
    ".claude",
    ".git",
    ".slm-venv",
    ".venv",
    "node_modules",
    "dist",
    "build",
    "__pycache__",
    "v2-archive",
    "audits",
    "superpowers",
}
HISTORICAL_FILES = {"CHANGELOG.md", "package-lock.json"}


@dataclass(frozen=True)
class Finding:
    path: Path
    line: int
    rule: Rule
    text: str


def _is_public_surface(path: Path, root: Path) -> bool:
    rel = path.relative_to(root)
    if any(part in SKIP_PARTS for part in rel.parts):
        return False
    if path.suffix.lower() not in TEXT_SUFFIXES:
        return False
    if "AUDIT" in path.name.upper():
        return False

    root_name = root.name.lower()
    if "website" in root_name:
        return rel.parts[0] in {"src", "public"}
    if root_name.endswith(".wiki"):
        return len(rel.parts) == 1

    if rel.as_posix() in {"README.md", "package.json", "pyproject.toml"}:
        return True
    if rel.parts[0] == "ide":
        return path.suffix.lower() in {".md", ".toml", ".json", ".js", ".mjs"}
    return rel.parts[0] in {"docs", "wiki-content", "plugin-src"} or rel.parts[:3] == (
        "src",
        "superlocalmemory",
        "cli",
    )


def iter_public_files(roots: Iterable[Path]) -> Iterable[tuple[Path, Path]]:
    for root in roots:
        resolved = root.resolve()
        if resolved.is_file():
            yield resolved, resolved.parent
            continue
        for path in sorted(resolved.rglob("*")):
            if path.is_file() and _is_public_surface(path, resolved):
                yield path, resolved


def _benchmark_is_qualified(rule_id: str, lines: list[str], index: int) -> bool:
    context = " ".join(lines[max(0, index - 5) : index + 6]).lower()
    if "historical" not in context:
        return False
    if rule_id == "unqualified-locomo-74":
        return "gpt-4.1-mini" in context and "answer" in context
    if rule_id == "unqualified-locomo-87":
        return "81 questions" in context and "one conversation" in context
    return False


def scan_paths(roots: Iterable[Path]) -> list[Finding]:
    findings: list[Finding] = []
    scanned: set[Path] = set()
    for path, _root in iter_public_files(roots):
        scanned.add(path)
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            continue
        for index, line in enumerate(lines):
            for rule in RULES:
                if not rule.pattern.search(line):
                    continue
                if (
                    rule.rule_id == "license-inconsistency"
                    and "website" in _root.name.lower()
                    and path.stem == "skillfortify"
                ):
                    # SkillFortify is a separate product with its own license policy.
                    continue
                if rule.rule_id.startswith("unqualified-locomo-") and _benchmark_is_qualified(rule.rule_id, lines, index):
                    continue
                findings.append(Finding(path=path, line=index + 1, rule=rule, text=line.strip()))

    # License metadata is shipped in source headers and runtime responses, not
    # only in marketing pages. Scan the complete current tree for this one rule
    # so package metadata cannot silently disagree with LICENSE again.
    license_rule = next(rule for rule in RULES if rule.rule_id == "license-inconsistency")
    for root in roots:
        resolved = root.resolve()
        candidates = [resolved] if resolved.is_file() else sorted(resolved.rglob("*"))
        for path in candidates:
            if not path.is_file() or path in scanned or path.name in HISTORICAL_FILES:
                continue
            if "website" in resolved.name.lower() and path.stem == "skillfortify":
                # SkillFortify is a separate product with its own license policy.
                continue
            try:
                rel = path.relative_to(resolved) if resolved.is_dir() else Path(path.name)
            except ValueError:
                rel = Path(path.name)
            if any(part in SKIP_PARTS for part in rel.parts):
                continue
            if path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except (OSError, UnicodeDecodeError):
                continue
            for index, line in enumerate(lines):
                if license_rule.pattern.search(line):
                    findings.append(
                        Finding(path=path, line=index + 1, rule=license_rule, text=line.strip())
                    )
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path, help="Repository roots or public text files")
    args = parser.parse_args()
    findings = scan_paths(args.roots)
    for finding in findings:
        print(f"{finding.path}:{finding.line}: {finding.rule.rule_id}: {finding.text}")
        print(f"  remedy: {finding.rule.replacement}")
    if findings:
        print(f"claim scanner: {len(findings)} unsupported public claim(s)")
        return 1
    print("claim scanner: green")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
