"""Validated SLM 4.0.2 contract boundaries.

These validators are the required pre-persistence boundary for forthcoming
Agent Experience and host-certificate writers. They prevent host/self-reported
telemetry from masquerading as independently verified learning evidence.
"""

from __future__ import annotations

import json
from datetime import datetime
from importlib import resources
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker  # type: ignore[import-untyped]


class ContractValidationError(ValueError):
    """Raised when a cross-lane public contract is invalid."""


_FORMAT_CHECKER = FormatChecker()


@_FORMAT_CHECKER.checks("date-time")
def _is_rfc3339_datetime(value: object) -> bool:
    """Accept only timezone-aware ISO/RFC3339 timestamps."""
    if not isinstance(value, str):
        return False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None


def _validate(schema_name: str, payload: dict[str, Any]) -> None:
    schema_path = resources.files(__package__).joinpath("schemas", schema_name)
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    errors = sorted(
        Draft202012Validator(schema, format_checker=_FORMAT_CHECKER).iter_errors(payload),
        key=lambda error: list(error.absolute_path),
    )
    if errors:
        error = errors[0]
        path = ".".join(str(part) for part in error.absolute_path) or "payload"
        raise ContractValidationError(f"{path}: {error.message}")


def validate_agent_experience(payload: dict[str, Any]) -> None:
    """Validate independent outcome evidence before an experience is stored."""
    _validate("agent-experience-v1.schema.json", payload)


def validate_cognitive_turn(payload: dict[str, Any]) -> None:
    """Validate the language-neutral fact-keyed receipt structure."""
    _validate("cognitive-turn-receipt-v1.schema.json", payload)


def validate_integration_certificate(payload: dict[str, Any]) -> None:
    """Validate a hash-bound host lifecycle certificate."""
    _validate("agent-integration-contract-v2.schema.json", payload)
