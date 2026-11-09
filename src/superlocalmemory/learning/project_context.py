# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Project context detection via filesystem analysis.

Scans a directory for project manifest files (package.json, pyproject.toml,
Cargo.toml, etc.) and extracts project metadata: name, languages, frameworks,
build tools.  No database access -- pure filesystem analysis.

Ported from V2 ProjectContextManager.  V2 memory.db multi-signal detection
replaced with direct filesystem scanning (the DB-based signals are handled
by the retrieval layer in V3).

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Manifest definitions
# ---------------------------------------------------------------------------

# Each manifest maps a filename to a parser function name.
# The parser extracts (project_name, languages, frameworks) from the file.

_MANIFEST_FILES: list[dict[str, Any]] = [
    {
        "filename": "package.json",
        "languages": ["javascript", "typescript"],
        "parser": "_parse_package_json",
    },
    {
        "filename": "pyproject.toml",
        "languages": ["python"],
        "parser": "_parse_pyproject_toml",
    },
    {
        "filename": "Cargo.toml",
        "languages": ["rust"],
        "parser": "_parse_cargo_toml",
    },
    {
        "filename": "go.mod",
        "languages": ["go"],
        "parser": "_parse_go_mod",
    },
    {
        "filename": "pom.xml",
        "languages": ["java"],
        "parser": "_parse_pom_xml",
    },
    {
        "filename": "build.gradle",
        "languages": ["java", "kotlin"],
        "parser": "_parse_gradle",
    },
    {
        "filename": "build.gradle.kts",
        "languages": ["kotlin"],
        "parser": "_parse_gradle",
    },
    {
        "filename": "Gemfile",
        "languages": ["ruby"],
        "parser": "_parse_gemfile",
    },
    {
        "filename": "composer.json",
        "languages": ["php"],
        "parser": "_parse_composer_json",
    },
    {
        "filename": "pubspec.yaml",
        "languages": ["dart"],
        "parser": "_parse_pubspec_yaml",
    },
]

# Framework detection via dependency names
_FRAMEWORK_HINTS: dict[str, str] = {
    "react": "React",
    "react-dom": "React",
    "vue": "Vue",
    "next": "Next.js",
    "nuxt": "Nuxt",
    "angular": "Angular",
    "svelte": "Svelte",
    "@sveltejs/kit": "SvelteKit",
    "express": "Express",
    "fastapi": "FastAPI",
    "django": "Django",
    "flask": "Flask",
    "fastify": "Fastify",
    "spring-boot": "Spring Boot",
    "rails": "Rails",
    "nestjs": "NestJS",
    "@nestjs/core": "NestJS",
    "tailwindcss": "Tailwind CSS",
    "pytest": "pytest",
    "jest": "Jest",
    "vitest": "Vitest",
    "torch": "PyTorch",
    "tensorflow": "TensorFlow",
    "transformers": "Hugging Face",
}


class ProjectContextManager:
    """Detect project context from the filesystem.

    No database needed -- scans the target directory for manifest files
    and extracts project metadata.
    """

    def detect(self, path: Path | str) -> dict[str, Any]:
        """Detect project context from a directory.

        Args:
            path: Directory to scan.

        Returns:
            Dict with keys: project_name, languages, frameworks,
            build_tool, manifest_file. All values may be None/empty
            when no manifest is found.
        """
        directory = Path(path)
        if not directory.is_dir():
            return _empty_context()

        # Try each manifest file
        for manifest_def in _MANIFEST_FILES:
            manifest_path = directory / manifest_def["filename"]
            if not manifest_path.exists():
                continue

            parser_name = manifest_def["parser"]
            parser = _PARSERS.get(parser_name)
            if parser is None:
                continue

            try:
                info = parser(manifest_path)
            except Exception as exc:
                logger.debug(
                    "Failed to parse %s: %s", manifest_path, exc
                )
                continue

            # Merge manifest-defined languages
            languages = list(
                set(info.get("languages", []) + manifest_def["languages"])
            )
            return {
                "project_name": info.get("project_name") or directory.name,
                "languages": languages,
                "frameworks": info.get("frameworks", []),
                "build_tool": info.get("build_tool"),
                "manifest_file": manifest_def["filename"],
            }

        # Fallback: infer from file extensions
        return self._infer_from_extensions(directory)

    # ------------------------------------------------------------------
    # Fallback extension scanning
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_from_extensions(directory: Path) -> dict[str, Any]:
        """Scan for source files and infer language from extensions."""
        ext_map: dict[str, str] = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".jsx": "javascript",
            ".rs": "rust",
            ".go": "go",
            ".java": "java",
            ".kt": "kotlin",
            ".rb": "ruby",
            ".php": "php",
            ".dart": "dart",
            ".swift": "swift",
            ".cs": "csharp",
            ".cpp": "cpp",
            ".c": "c",
        }
        detected: set[str] = set()
        try:
            for child in directory.iterdir():
                if child.is_file() and child.suffix in ext_map:
                    detected.add(ext_map[child.suffix])
        except PermissionError:
            pass

        return {
            "project_name": directory.name,
            "languages": sorted(detected),
            "frameworks": [],
            "build_tool": None,
            "manifest_file": None,
        }


# ----------------------------------------------------------------------
# Parsers (one per manifest type)
# ----------------------------------------------------------------------


def _parse_package_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    deps = set(data.get("dependencies", {}).keys())
    deps |= set(data.get("devDependencies", {}).keys())
    frameworks = [
        _FRAMEWORK_HINTS[d] for d in deps if d in _FRAMEWORK_HINTS
    ]
    lang = ["typescript"] if "typescript" in deps else ["javascript"]
    return {
        "project_name": data.get("name", ""),
        "languages": lang,
        "frameworks": sorted(set(frameworks)),
        "build_tool": "npm",
    }


def _parse_pyproject_toml(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    name = ""
    # Simple TOML name extraction (no toml library required)
    m = re.search(r'name\s*=\s*["\']([^"\']+)["\']', text)
    if m:
        name = m.group(1)

    # Detect frameworks from dependencies section
    frameworks: list[str] = []
    for hint_key, hint_name in _FRAMEWORK_HINTS.items():
        if hint_key in text.lower():
            frameworks.append(hint_name)

    build_tool = "poetry" if "[tool.poetry]" in text else "setuptools"
    if "[tool.hatch]" in text:
        build_tool = "hatch"

    return {
        "project_name": name,
        "languages": ["python"],
        "frameworks": sorted(set(frameworks)),
        "build_tool": build_tool,
    }


def _parse_cargo_toml(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    m = re.search(r'name\s*=\s*"([^"]+)"', text)
    return {
        "project_name": m.group(1) if m else "",
        "languages": ["rust"],
        "frameworks": [],
        "build_tool": "cargo",
    }


def _parse_go_mod(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    m = re.search(r"^module\s+(\S+)", text, re.MULTILINE)
    module_name = m.group(1) if m else ""
    project_name = module_name.split("/")[-1] if module_name else ""
    return {
        "project_name": project_name,
        "languages": ["go"],
        "frameworks": [],
        "build_tool": "go",
    }


def _parse_pom_xml(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    m = re.search(r"<artifactId>([^<]+)</artifactId>", text)
    frameworks = []
    if "spring-boot" in text.lower():
        frameworks.append("Spring Boot")
    return {
        "project_name": m.group(1) if m else "",
        "languages": ["java"],
        "frameworks": frameworks,
        "build_tool": "maven",
    }


def _parse_gradle(path: Path) -> dict[str, Any]:
    return {
        "project_name": path.parent.name,
        "languages": ["java", "kotlin"],
        "frameworks": [],
        "build_tool": "gradle",
    }


def _parse_gemfile(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    frameworks = ["Rails"] if "rails" in text.lower() else []
    return {
        "project_name": path.parent.name,
        "languages": ["ruby"],
        "frameworks": frameworks,
        "build_tool": "bundler",
    }


def _parse_composer_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        "project_name": data.get("name", "").split("/")[-1],
        "languages": ["php"],
        "frameworks": [],
        "build_tool": "composer",
    }


def _parse_pubspec_yaml(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    m = re.search(r"^name:\s*(\S+)", text, re.MULTILINE)
    frameworks = ["Flutter"] if "flutter" in text.lower() else []
    return {
        "project_name": m.group(1) if m else "",
        "languages": ["dart"],
        "frameworks": frameworks,
        "build_tool": "pub",
    }


# Parser dispatch table
_PARSERS: dict[str, Any] = {
    "_parse_package_json": _parse_package_json,
    "_parse_pyproject_toml": _parse_pyproject_toml,
    "_parse_cargo_toml": _parse_cargo_toml,
    "_parse_go_mod": _parse_go_mod,
    "_parse_pom_xml": _parse_pom_xml,
    "_parse_gradle": _parse_gradle,
    "_parse_gemfile": _parse_gemfile,
    "_parse_composer_json": _parse_composer_json,
    "_parse_pubspec_yaml": _parse_pubspec_yaml,
}


def _empty_context() -> dict[str, Any]:
    return {
        "project_name": None,
        "languages": [],
        "frameworks": [],
        "build_tool": None,
        "manifest_file": None,
    }
