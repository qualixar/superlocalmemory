"""Held-out answer-probability calibration report tooling.

Raw retrieval scores are ranking utilities.  They are retained for distribution
diagnostics but are never passed to Brier, ECE, reliability, or selective-risk
calculations.  Those metrics require an explicitly supplied probability from a
separately fitted calibrator and a disjoint held-out evaluation artifact.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Sequence
from typing import Any

INPUT_SCHEMA = "superlocalmemory.calibration-input/v1"
REPORT_SCHEMA = "superlocalmemory.calibration-report/v1"
MIN_HELD_OUT_EXAMPLES = 100
DEFAULT_THRESHOLDS = (0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
_RAW_AS_PROBABILITY_METHODS = {
    "identity",
    "raw",
    "raw_score",
    "raw_retrieval_score",
    "none",
}


class CalibrationArtifactError(ValueError):
    """The input cannot support a truthful held-out calibration report."""


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _finite_number(value: object, *, field: str) -> float:
    if isinstance(value, bool):
        raise CalibrationArtifactError(f"{field} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise CalibrationArtifactError(f"{field} must be a finite number") from exc
    if not math.isfinite(number):
        raise CalibrationArtifactError(f"{field} must be a finite number")
    return number


def _validated_rows(examples: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    validated: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, example in enumerate(examples):
        if not isinstance(example, dict):
            raise CalibrationArtifactError(f"examples[{index}] must be an object")
        example_id = str(example.get("example_id", "")).strip()
        if not example_id:
            raise CalibrationArtifactError(f"examples[{index}].example_id is required")
        if example_id in seen_ids:
            raise CalibrationArtifactError("example_id values must be unique")
        seen_ids.add(example_id)

        if "predicted_probability" not in example:
            raise CalibrationArtifactError(
                "predicted_probability is required; raw retrieval scores are not probabilities"
            )
        probability = _finite_number(
            example["predicted_probability"], field="predicted_probability"
        )
        if probability < 0.0 or probability > 1.0:
            raise CalibrationArtifactError(
                "predicted_probability must be between 0 and 1"
            )

        label = example.get("label")
        if isinstance(label, bool):
            label = int(label)
        if label not in (0, 1):
            raise CalibrationArtifactError("label must be binary (0 or 1)")
        raw_score = _finite_number(
            example.get("raw_retrieval_score"), field="raw_retrieval_score"
        )
        validated.append(
            {
                "example_id": example_id,
                "raw_retrieval_score": raw_score,
                "predicted_probability": probability,
                "label": int(label),
            }
        )
    return validated


def _round(value: float) -> float:
    return round(value, 12)


def compute_calibration_metrics(
    examples: Sequence[dict[str, Any]],
    *,
    bin_count: int = 10,
    thresholds: Iterable[float] = DEFAULT_THRESHOLDS,
) -> dict[str, Any]:
    """Compute probability metrics while reporting raw scores separately."""
    if not isinstance(bin_count, int) or isinstance(bin_count, bool) or bin_count < 2:
        raise CalibrationArtifactError("bin_count must be an integer >= 2")
    rows = _validated_rows(examples)
    if not rows:
        raise CalibrationArtifactError("at least one example is required")
    total = len(rows)

    brier = sum(
        (row["predicted_probability"] - row["label"]) ** 2 for row in rows
    ) / total

    reliability: list[dict[str, Any]] = []
    ece = 0.0
    mce = 0.0
    for index in range(bin_count):
        lower = index / bin_count
        upper = (index + 1) / bin_count
        members = [
            row for row in rows
            if lower <= row["predicted_probability"]
            and (row["predicted_probability"] < upper or index == bin_count - 1)
        ]
        if members:
            mean_probability = sum(row["predicted_probability"] for row in members) / len(members)
            observed_accuracy = sum(row["label"] for row in members) / len(members)
            gap = abs(mean_probability - observed_accuracy)
            ece += len(members) / total * gap
            mce = max(mce, gap)
            mean_value: float | None = _round(mean_probability)
            accuracy_value: float | None = _round(observed_accuracy)
            gap_value: float | None = _round(gap)
        else:
            mean_value = accuracy_value = gap_value = None
        reliability.append(
            {
                "bin_index": index,
                "lower_inclusive": _round(lower),
                "upper_inclusive": index == bin_count - 1,
                "upper": _round(upper),
                "count": len(members),
                "mean_probability": mean_value,
                "observed_accuracy": accuracy_value,
                "absolute_gap": gap_value,
            }
        )

    selective: list[dict[str, Any]] = []
    previous = -1.0
    for raw_threshold in thresholds:
        threshold = _finite_number(raw_threshold, field="selective threshold")
        if threshold < 0.0 or threshold > 1.0:
            raise CalibrationArtifactError("selective threshold must be between 0 and 1")
        if threshold <= previous:
            raise CalibrationArtifactError("selective thresholds must be strictly increasing")
        previous = threshold
        retained = [row for row in rows if row["predicted_probability"] >= threshold]
        retained_count = len(retained)
        coverage = retained_count / total
        risk = (
            sum(1 - row["label"] for row in retained) / retained_count
            if retained_count
            else None
        )
        selective.append(
            {
                "threshold": _round(threshold),
                "retained": retained_count,
                "coverage": _round(coverage),
                "abstention_rate": _round(1.0 - coverage),
                "selective_risk": _round(risk) if risk is not None else None,
            }
        )

    raw_scores = [row["raw_retrieval_score"] for row in rows]
    probabilities = [row["predicted_probability"] for row in rows]
    return {
        "example_count": total,
        "brier_score": _round(brier),
        "expected_calibration_error": _round(ece),
        "maximum_calibration_error": _round(mce),
        "reliability_bins": reliability,
        "selective_risk": selective,
        "probability_summary": {
            "minimum": _round(min(probabilities)),
            "maximum": _round(max(probabilities)),
            "mean": _round(sum(probabilities) / total),
            "unique_count": len(set(probabilities)),
            "boundary_count": sum(value in (0.0, 1.0) for value in probabilities),
        },
        "raw_retrieval_score_summary": {
            "minimum": _round(min(raw_scores)),
            "maximum": _round(max(raw_scores)),
            "mean": _round(sum(raw_scores) / total),
            "unique_count": len(set(raw_scores)),
            "boundary_count": sum(value in (0.0, 1.0) for value in raw_scores),
            "used_in_probability_metrics": False,
        },
    }


def _require_sha256(value: object, *, field: str) -> str:
    text = str(value or "").lower()
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise CalibrationArtifactError(f"{field} must be a 64-character SHA-256")
    return text


def build_calibration_report(
    artifact: dict[str, Any],
    *,
    bin_count: int = 10,
    thresholds: Iterable[float] = DEFAULT_THRESHOLDS,
) -> dict[str, Any]:
    """Validate a disjoint held-out artifact and build a non-activating report."""
    if not isinstance(artifact, dict) or artifact.get("schema") != INPUT_SCHEMA:
        raise CalibrationArtifactError(f"schema must be {INPUT_SCHEMA}")
    calibrator = artifact.get("calibrator")
    evaluation = artifact.get("evaluation")
    examples = artifact.get("examples")
    if not isinstance(calibrator, dict) or not isinstance(evaluation, dict):
        raise CalibrationArtifactError("calibrator and evaluation metadata are required")
    if not isinstance(examples, list):
        raise CalibrationArtifactError("examples must be a list")

    calibration_id = str(calibrator.get("calibration_id", "")).strip()
    if not calibration_id:
        raise CalibrationArtifactError("calibration_id is required")
    method = str(calibrator.get("method", "")).strip().lower()
    if not method or method in _RAW_AS_PROBABILITY_METHODS:
        raise CalibrationArtifactError(
            "a fitted calibrator is required; a raw score identity mapping is forbidden"
        )
    fit_dataset = str(calibrator.get("fit_dataset_id", "")).strip()
    eval_dataset = str(evaluation.get("dataset_id", "")).strip()
    if not fit_dataset or not eval_dataset or fit_dataset == eval_dataset:
        raise CalibrationArtifactError(
            "calibrator fit and held-out evaluation datasets must be distinct"
        )
    fit_split = str(calibrator.get("fit_split", "")).strip()
    if not fit_split or fit_split == "held_out":
        raise CalibrationArtifactError("fit_split must be distinct from held_out")
    if evaluation.get("split") != "held_out":
        raise CalibrationArtifactError("evaluation split must be held_out")
    fit_artifact_hash = _require_sha256(
        calibrator.get("fit_artifact_sha256"), field="fit_artifact_sha256"
    )

    validated = _validated_rows(examples)
    if len(validated) < MIN_HELD_OUT_EXAMPLES:
        raise CalibrationArtifactError(
            f"held-out evaluation requires at least {MIN_HELD_OUT_EXAMPLES} examples"
        )
    expected_hash = _require_sha256(
        evaluation.get("examples_sha256"), field="examples_sha256"
    )
    actual_hash = _canonical_sha256(examples)
    if expected_hash != actual_hash:
        raise CalibrationArtifactError("examples hash does not match held-out artifact")

    metrics = compute_calibration_metrics(
        validated, bin_count=bin_count, thresholds=thresholds
    )
    return {
        "schema": REPORT_SCHEMA,
        "evidence_status": "held_out_report",
        "calibration_id": calibration_id,
        "calibrator": {
            "method": method,
            "fit_dataset_id": fit_dataset,
            "fit_split": fit_split,
            "fit_artifact_sha256": fit_artifact_hash,
        },
        "evaluation": {
            "dataset_id": eval_dataset,
            "split": "held_out",
            "examples_sha256": actual_hash,
        },
        "metrics": metrics,
        "separation": {
            "raw_score_field": "raw_retrieval_score",
            "probability_metric_input": "predicted_probability",
            "raw_retrieval_score_used_as_probability": False,
        },
        "runtime_calibration_status": "uncalibrated",
        "runtime_activation_allowed": False,
        "runtime_activation_reason": (
            "This artifact is evaluation evidence only; production activation "
            "requires a separately reviewed and frozen calibrator contract."
        ),
        "input_artifact_sha256": _canonical_sha256(artifact),
    }


__all__ = [
    "CalibrationArtifactError",
    "INPUT_SCHEMA",
    "MIN_HELD_OUT_EXAMPLES",
    "REPORT_SCHEMA",
    "build_calibration_report",
    "compute_calibration_metrics",
]
