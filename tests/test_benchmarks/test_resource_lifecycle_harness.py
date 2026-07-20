"""Contracts for the isolated resource-lifecycle benchmark."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest

from benchmarks.resource_lifecycle.harness import (
    latency_summary,
    render_markdown,
    rss_analysis,
    run_mode_a_local,
)


def test_direct_benchmark_entrypoint_resolves_repository_package() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            "benchmarks/resource_lifecycle/run_benchmark.py",
            "--help",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
    assert "--ingests" in result.stdout


def test_metric_contract_reports_distribution_without_universal_budget() -> None:
    latency = latency_summary([1.0, 2.0, 3.0, 4.0, 100.0])
    rss = rss_analysis([100.0, 103.0, 103.5, 103.5, 103.5], calls_per_sample=10)

    assert latency == {
        "samples": 5,
        "min_ms": 1.0,
        "p50_ms": 3.0,
        "p95_ms": 100.0,
        "max_ms": 100.0,
        "mean_ms": 22.0,
    }
    assert rss["tail_plateau_observed"] is True
    assert rss["samples_mb"] == [100.0, 103.0, 103.5, 103.5, 103.5]


@pytest.mark.benchmark
def test_mode_a_resource_harness_smoke(tmp_path: Path) -> None:
    result = run_mode_a_local(
        warmup_iterations=2,
        ingest_iterations=4,
        idempotent_ingest_iterations=6,
        recall_iterations=6,
        sample_every=2,
    )

    assert result["protocol"]["mode"] == "A"
    assert result["latency"]["universal_budget_applied"] is False
    assert result["latency"]["ingest"]["samples"] == 4
    assert result["latency"]["idempotent_ingest"]["samples"] == 6
    assert result["latency"]["recall"]["samples"] == 6
    assert result["process_ownership"]["new_child_pids_after_close"] == []
    assert "Mode B Ollama" in render_markdown(result)
