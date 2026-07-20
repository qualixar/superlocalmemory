"""Held-out calibration reports must never promote raw retrieval scores."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from superlocalmemory.evaluation.calibration import (
    CalibrationArtifactError,
    build_calibration_report,
    compute_calibration_metrics,
)

ROOT = Path(__file__).resolve().parents[2]


def _sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _examples(count: int = 100) -> list[dict[str, object]]:
    return [
        {
            "example_id": f"q-{index:03d}",
            "raw_retrieval_score": (index % 17) / 16,
            "predicted_probability": 0.9 if index % 2 == 0 else 0.1,
            "label": 1 if index % 2 == 0 else 0,
        }
        for index in range(count)
    ]


def _artifact(*, examples: list[dict[str, object]] | None = None) -> dict:
    rows = examples or _examples()
    return {
        "schema": "superlocalmemory.calibration-input/v1",
        "calibrator": {
            "calibration_id": "local-isotonic-2026-07-15",
            "method": "isotonic_regression",
            "fit_dataset_id": "local-calibration-train-v1",
            "fit_split": "calibration_train",
            "fit_artifact_sha256": "a" * 64,
        },
        "evaluation": {
            "dataset_id": "local-held-out-v1",
            "split": "held_out",
            "examples_sha256": _sha256(rows),
        },
        "examples": rows,
    }


def test_metrics_keep_raw_scores_out_of_probability_metrics() -> None:
    rows = _examples()
    baseline = compute_calibration_metrics(rows, bin_count=10)

    changed_raw = [{**row, "raw_retrieval_score": 999.0} for row in rows]
    changed = compute_calibration_metrics(changed_raw, bin_count=10)

    assert changed["brier_score"] == baseline["brier_score"] == pytest.approx(0.01)
    assert changed["expected_calibration_error"] == baseline[
        "expected_calibration_error"
    ] == pytest.approx(0.1)
    assert changed["reliability_bins"] == baseline["reliability_bins"]
    assert changed["raw_retrieval_score_summary"] != baseline[
        "raw_retrieval_score_summary"
    ]


def test_report_contains_reliability_and_selective_risk_without_runtime_activation() -> None:
    report = build_calibration_report(_artifact(), bin_count=10)

    assert report["schema"] == "superlocalmemory.calibration-report/v1"
    assert report["evidence_status"] == "held_out_report"
    assert report["runtime_calibration_status"] == "uncalibrated"
    assert report["runtime_activation_allowed"] is False
    assert report["calibration_id"] == "local-isotonic-2026-07-15"
    assert len(report["metrics"]["reliability_bins"]) == 10
    assert report["metrics"]["selective_risk"]
    assert report["separation"]["probability_metric_input"] == "predicted_probability"
    assert report["separation"]["raw_retrieval_score_used_as_probability"] is False


def test_selective_risk_reports_coverage_and_abstention() -> None:
    rows = _examples()
    metrics = compute_calibration_metrics(rows, thresholds=(0.5, 0.95))

    half = metrics["selective_risk"][0]
    none = metrics["selective_risk"][1]
    assert half == {
        "threshold": 0.5,
        "retained": 50,
        "coverage": 0.5,
        "abstention_rate": 0.5,
        "selective_risk": 0.0,
    }
    assert none["retained"] == 0
    assert none["coverage"] == 0.0
    assert none["abstention_rate"] == 1.0
    assert none["selective_risk"] is None


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value["evaluation"].update(split="test"), "held_out"),
        (
            lambda value: value["evaluation"].update(
                dataset_id=value["calibrator"]["fit_dataset_id"]
            ),
            "distinct",
        ),
        (lambda value: value["calibrator"].update(method="identity"), "raw score"),
        (lambda value: value["evaluation"].update(examples_sha256="0" * 64), "hash"),
        (lambda value: value.update(examples=value["examples"][:20]), "100"),
        (
            lambda value: value["examples"][0].pop("predicted_probability"),
            "predicted_probability",
        ),
        (
            lambda value: value["examples"][0].update(predicted_probability=1.2),
            "between 0 and 1",
        ),
        (lambda value: value["examples"][0].update(label=2), "binary"),
        (
            lambda value: value["examples"][1].update(
                example_id=value["examples"][0]["example_id"]
            ),
            "unique",
        ),
    ],
)
def test_invalid_or_non_held_out_artifacts_fail_closed(mutate, message: str) -> None:
    value = _artifact()
    mutate(value)
    # Mutations unrelated to the hash should not be masked by stale hash errors.
    if message not in {"hash", "predicted_probability", "between 0 and 1", "binary", "unique"}:
        value["evaluation"]["examples_sha256"] = _sha256(value["examples"])

    with pytest.raises(CalibrationArtifactError, match=message):
        build_calibration_report(value)


def test_cli_writes_report_but_never_marks_runtime_calibrated(tmp_path: Path) -> None:
    input_path = tmp_path / "held-out.json"
    output_path = tmp_path / "report.json"
    input_path.write_text(json.dumps(_artifact()), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/calibration_report.py"),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["runtime_calibration_status"] == "uncalibrated"
    assert report["runtime_activation_allowed"] is False


def test_cli_refuses_to_emit_report_for_raw_scores_only(tmp_path: Path) -> None:
    artifact = _artifact()
    for row in artifact["examples"]:
        row.pop("predicted_probability")
    artifact["evaluation"]["examples_sha256"] = _sha256(artifact["examples"])
    input_path = tmp_path / "raw-only.json"
    input_path.write_text(json.dumps(artifact), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/calibration_report.py"), "--input", str(input_path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "predicted_probability" in result.stderr
