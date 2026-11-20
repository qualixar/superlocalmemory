# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V4 | https://qualixar.com | https://varunpratap.com

"""Arguments for ``slm tool <name>``: read, merged and checked against the tool's schema.

Sources, any combination: ``--json '<object>'``, ``--args-file PATH``, ``-``
(an object on stdin) and ``--key value`` for one argument. A key given twice
is an error, not a silent override. The merged object is validated against the
input schema the MCP server publishes for the tool, and an argument the tool
does not take is refused: an MCP client may send extra keys and have them
ignored, but a typo or a smuggled field (``channel``, ``agent_id_source``)
should be loud on a command line.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Iterable


class ArgumentProblem(ValueError):
    """The arguments cannot be sent; ``details`` says why, one line each."""

    def __init__(self, message: str, details: list[str] | None = None) -> None:
        super().__init__(message)
        self.details = list(details or [])


def _object(text: str, source: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except ValueError as exc:
        raise ArgumentProblem(f"{source} is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ArgumentProblem(f"{source} must be a JSON object, got {type(value).__name__}")
    return value


def _merge(into: dict[str, Any], extra: dict[str, Any], source: str) -> None:
    for key, value in extra.items():
        if key in into:
            raise ArgumentProblem(f"argument {key!r} is given more than once ({source})")
        into[key] = value


def _types(prop: dict[str, Any]) -> set[str]:
    """The JSON types a property accepts (``anyOf`` with ``null`` included)."""
    found: set[str] = set()
    kind = prop.get("type")
    if isinstance(kind, str):
        found.add(kind)
    elif isinstance(kind, list):
        found.update(k for k in kind if isinstance(k, str))
    for option in prop.get("anyOf") or prop.get("oneOf") or []:
        if isinstance(option, dict):
            found |= _types(option)
    return found


def coerce_flag(name: str, raw: str, prop: dict[str, Any] | None) -> Any:
    """A ``--key value`` string as the type the schema asks for."""
    kinds = _types(prop or {}) - {"null"}
    if not kinds or kinds == {"string"}:
        return raw
    if "string" in kinds and not raw.strip().startswith(("{", "[")):
        return raw
    if kinds <= {"boolean"}:
        lowered = raw.strip().lower()
        if lowered in ("true", "1", "yes", "on"):
            return True
        if lowered in ("false", "0", "no", "off"):
            return False
        raise ArgumentProblem(f"--{name} takes true or false, got {raw!r}")
    try:
        return json.loads(raw)
    except ValueError:
        if "string" in kinds:
            return raw
        raise ArgumentProblem(
            f"--{name} needs a {'/'.join(sorted(kinds))} value, got {raw!r}") from None


def flag_pairs(tokens: list[str]) -> list[tuple[str, str]]:
    """``--key value`` and ``--key=value`` tokens as ``(key, value)`` pairs."""
    pairs: list[tuple[str, str]] = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if not token.startswith("--") or token == "--":
            raise ArgumentProblem(f"unexpected argument {token!r}; use --key value or --json")
        key, sep, value = token[2:].partition("=")
        if not sep:
            if i + 1 >= len(tokens):
                raise ArgumentProblem(f"--{key} needs a value")
            i += 1
            value = tokens[i]
        pairs.append((key.replace("-", "_"), value))
        i += 1
    return pairs


def gather(
    *,
    json_text: str | None,
    args_file: str | None,
    use_stdin: bool,
    pairs: Iterable[tuple[str, str]],
    schema: dict[str, Any],
    stdin: Any = None,
) -> dict[str, Any]:
    """The tool arguments from every source, merged (no key twice)."""
    merged: dict[str, Any] = {}
    if json_text is not None:
        _merge(merged, _object(json_text, "--json"), "--json")
    if args_file is not None:
        try:
            text = Path(args_file).read_text(encoding="utf-8")
        except OSError as exc:
            raise ArgumentProblem(f"cannot read --args-file {args_file}: {exc.strerror}") from exc
        _merge(merged, _object(text, "--args-file"), "--args-file")
    if use_stdin:
        _merge(merged, _object((stdin or sys.stdin).read(), "stdin"), "stdin")
    props = (schema or {}).get("properties") or {}
    for key, raw in pairs:
        _merge(merged, {key: coerce_flag(key, raw, props.get(key))}, f"--{key}")
    return merged


def validate(arguments: dict[str, Any], schema: dict[str, Any]) -> None:
    """Refuse arguments the tool's input schema does not accept."""
    import jsonschema

    props = (schema or {}).get("properties") or {}
    unknown = sorted(set(arguments) - set(props))
    problems = [
        f"unknown argument {key!r} (this tool takes: {', '.join(sorted(props)) or 'nothing'})"
        for key in unknown
    ]
    validator_cls = jsonschema.validators.validator_for(schema or {})
    validator = validator_cls(schema or {"type": "object"})
    for error in sorted(validator.iter_errors(arguments), key=lambda e: list(e.path)):
        where = ".".join(str(p) for p in error.path) or "arguments"
        problems.append(f"{where}: {error.message}")
    if problems:
        raise ArgumentProblem("the arguments do not match the tool's input schema", problems)


__all__ = ["ArgumentProblem", "coerce_flag", "flag_pairs", "gather", "validate"]
