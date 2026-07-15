# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Repeat-ingest/repeat-recall resource measurement for the local engine.

This harness uses the shipped Mode A engine, SQLite schema, canonical ingestion,
and retrieval pipeline with one deterministic in-process embedding boundary.
The embedding boundary prevents model downloads and worker processes from
turning a resource-lifecycle measurement into a provider/model benchmark.
"""

from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Sequence
from unittest.mock import patch

import numpy as np
import psutil


def nearest_rank(values: Sequence[float], percentile: float) -> float:
    """Return the deterministic nearest-rank percentile."""
    if not values:
        return 0.0
    if not 0 < percentile <= 1:
        raise ValueError("percentile must be in (0, 1]")
    ordered = sorted(float(value) for value in values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def latency_summary(samples_ms: Sequence[float]) -> dict[str, float | int]:
    """Summarize measured latency without inventing an environment-wide SLO."""
    if not samples_ms:
        return {"samples": 0, "min_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0,
                "max_ms": 0.0, "mean_ms": 0.0}
    values = [float(value) for value in samples_ms]
    return {
        "samples": len(values),
        "min_ms": round(min(values), 3),
        "p50_ms": round(nearest_rank(values, 0.50), 3),
        "p95_ms": round(nearest_rank(values, 0.95), 3),
        "max_ms": round(max(values), 3),
        "mean_ms": round(statistics.fmean(values), 3),
    }


def linear_slope(samples: Sequence[float]) -> float:
    """Return least-squares slope per sample index."""
    if len(samples) < 2:
        return 0.0
    ys = [float(value) for value in samples]
    xs = list(range(len(ys)))
    x_mean = statistics.fmean(xs)
    y_mean = statistics.fmean(ys)
    denominator = sum((x - x_mean) ** 2 for x in xs)
    if denominator == 0:
        return 0.0
    return sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denominator


def rss_analysis(samples_mb: Sequence[float], *, calls_per_sample: int) -> dict:
    """Describe RSS trend and a bounded tail-plateau observation.

    ``tail_plateau_observed`` is evidence for this run only. The 4 MiB tail
    span is an allocator/page-cache noise tolerance for this harness, not a
    universal product memory budget.
    """
    values = [float(value) for value in samples_mb]
    if not values:
        return {
            "samples_mb": [],
            "baseline_mb": 0.0,
            "final_mb": 0.0,
            "peak_mb": 0.0,
            "growth_mb": 0.0,
            "tail_span_mb": 0.0,
            "tail_slope_mb_per_100_calls": 0.0,
            "tail_plateau_observed": False,
        }
    tail = values[len(values) // 2:]
    slope_per_sample = linear_slope(tail)
    tail_span = max(tail) - min(tail)
    slope_per_100_calls = slope_per_sample * (100 / max(1, calls_per_sample))
    return {
        "samples_mb": [round(value, 3) for value in values],
        "baseline_mb": round(values[0], 3),
        "final_mb": round(values[-1], 3),
        "peak_mb": round(max(values), 3),
        "growth_mb": round(values[-1] - values[0], 3),
        "tail_span_mb": round(tail_span, 3),
        "tail_slope_mb_per_100_calls": round(slope_per_100_calls, 3),
        "tail_plateau_observed": bool(
            len(tail) >= 3 and tail_span <= 4.0 and slope_per_100_calls <= 4.0
        ),
    }


class DeterministicEmbedder:
    """Stable 768-dimensional local test double for the model boundary."""

    is_available = True
    model_name = "deterministic-sha256-numpy-768"
    dimension = 768

    @staticmethod
    def embed(text: str) -> list[float]:
        seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4], "big")
        rng = np.random.RandomState(seed)
        vector = rng.randn(768).astype(np.float32)
        vector /= np.linalg.norm(vector)
        return vector.tolist()

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]

    @staticmethod
    def compute_fisher_params(vector: Sequence[float]) -> tuple[list[float], list[float]]:
        return [0.0] * len(vector), [1.0] * len(vector)

    @staticmethod
    def close() -> None:
        return None


def _git_commit(repo_root: Path) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _rss_mb(process: psutil.Process) -> float:
    gc.collect()
    return process.memory_info().rss / (1024 * 1024)


def _measure(callable_) -> tuple[object, float]:
    started = time.perf_counter_ns()
    result = callable_()
    elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
    return result, elapsed_ms


def run_mode_a_local(
    *,
    warmup_iterations: int = 5,
    ingest_iterations: int = 50,
    idempotent_ingest_iterations: int = 100,
    recall_iterations: int = 100,
    sample_every: int = 10,
    repo_root: Path | None = None,
) -> dict:
    """Run one isolated Mode A local-engine resource measurement."""
    for value, name in (
        (warmup_iterations, "warmup_iterations"),
        (ingest_iterations, "ingest_iterations"),
        (idempotent_ingest_iterations, "idempotent_ingest_iterations"),
        (recall_iterations, "recall_iterations"),
        (sample_every, "sample_every"),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive")

    root = (repo_root or Path(__file__).resolve().parents[2]).resolve()
    process = psutil.Process()
    children_before = {child.pid for child in process.children(recursive=True)}

    with tempfile.TemporaryDirectory(prefix="slm-resource-benchmark-") as temp_dir:
        data_root = Path(temp_dir).resolve()
        old_environment = {
            key: os.environ.get(key)
            for key in ("SLM_DATA_DIR", "SL_MEMORY_PATH", "SLM_HOME")
        }
        os.environ["SLM_DATA_DIR"] = str(data_root)
        os.environ["SL_MEMORY_PATH"] = str(data_root)
        os.environ["SLM_HOME"] = str(data_root)

        from superlocalmemory.core.config import SLMConfig
        from superlocalmemory.core.engine import MemoryEngine
        from superlocalmemory.storage.models import Mode

        config = SLMConfig.for_mode(Mode.A, base_dir=data_root)
        config.retrieval.use_cross_encoder = False
        embedder = DeterministicEmbedder()

        engine = MemoryEngine(config)
        initialization_started = time.perf_counter_ns()
        with patch(
            "superlocalmemory.core.engine_wiring.init_embedder",
            return_value=embedder,
        ):
            engine.initialize()
        initialization_ms = (time.perf_counter_ns() - initialization_started) / 1_000_000

        ingest_latencies: list[float] = []
        idempotent_ingest_latencies: list[float] = []
        recall_latencies: list[float] = []
        ingest_rss: list[float] = []
        idempotent_ingest_rss: list[float] = []
        recall_rss: list[float] = []
        try:
            def ingest(index: int):
                content = (
                    f"Resource benchmark observation {index}: Project Atlas owner "
                    f"Alice approved incident recovery control {index} on 2026-07-15 "
                    "and the platform reliability team recorded the decision."
                )
                return engine.store(content)

            def recall(index: int):
                return engine.recall(
                    f"Who approved Project Atlas incident recovery control {index}?",
                    limit=5,
                )

            for index in range(warmup_iterations):
                fact_ids = ingest(index)
                if not fact_ids:
                    raise RuntimeError("warm-up ingestion produced no fact IDs")
                recall(index)

            ingest_rss.append(_rss_mb(process))
            for offset in range(ingest_iterations):
                index = warmup_iterations + offset
                fact_ids, elapsed_ms = _measure(lambda index=index: ingest(index))
                if not fact_ids:
                    raise RuntimeError(f"ingestion {index} produced no fact IDs")
                ingest_latencies.append(elapsed_ms)
                if (offset + 1) % sample_every == 0 or offset + 1 == ingest_iterations:
                    ingest_rss.append(_rss_mb(process))

            # Separate intended corpus/index growth from retry-path retention.
            # A stable idempotency key exercises canonical ingestion repeatedly
            # against fixed logical state, which is the leak-sensitive phase.
            from superlocalmemory.core.engine_ingestion import canonical_store

            retry_content = (
                "Resource benchmark stable retry: Project Atlas owner Alice "
                "approved the fixed recovery control and recorded the decision."
            )

            def idempotent_ingest():
                return canonical_store(
                    engine,
                    retry_content,
                    source_type="resource-benchmark",
                    trusted_actor_id="local-capability:resource-benchmark",
                    idempotency_key="resource-benchmark-stable-retry",
                )

            for _ in range(warmup_iterations):
                if not idempotent_ingest():
                    raise RuntimeError("idempotent warm-up produced no fact IDs")
            idempotent_ingest_rss.append(_rss_mb(process))
            for offset in range(idempotent_ingest_iterations):
                fact_ids, elapsed_ms = _measure(idempotent_ingest)
                if not fact_ids:
                    raise RuntimeError("idempotent ingestion produced no fact IDs")
                idempotent_ingest_latencies.append(elapsed_ms)
                if (
                    (offset + 1) % sample_every == 0
                    or offset + 1 == idempotent_ingest_iterations
                ):
                    idempotent_ingest_rss.append(_rss_mb(process))

            fixed_query_index = warmup_iterations + ingest_iterations - 1
            for _ in range(warmup_iterations):
                recall(fixed_query_index)
            recall_rss.append(_rss_mb(process))
            for offset in range(recall_iterations):
                response, elapsed_ms = _measure(
                    lambda: recall(fixed_query_index)
                )
                if not hasattr(response, "results"):
                    raise RuntimeError("recall did not return the public response contract")
                recall_latencies.append(elapsed_ms)
                if (offset + 1) % sample_every == 0 or offset + 1 == recall_iterations:
                    recall_rss.append(_rss_mb(process))
        finally:
            engine.close()
            for key, previous in old_environment.items():
                if previous is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = previous

    children_after = {child.pid for child in process.children(recursive=True)}
    virtual_memory = psutil.virtual_memory()
    recall_rss_result = rss_analysis(recall_rss, calls_per_sample=sample_every)
    result = {
        "schema_version": "1",
        "generated_at": datetime.now(UTC).isoformat(),
        "source_commit": _git_commit(root),
        "product_version": importlib.metadata.version("superlocalmemory"),
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor() or "not-reported",
            "logical_cpus": os.cpu_count(),
            "system_memory_mb": round(virtual_memory.total / (1024 * 1024), 1),
        },
        "protocol": {
            "mode": "A",
            "engine": "shipped MemoryEngine FULL + canonical ingestion + recall",
            "database": "isolated temporary SQLite",
            "embedding_boundary": embedder.model_name,
            "network": "disabled by construction; no provider calls",
            "warmup_iterations": warmup_iterations,
            "ingest_iterations": ingest_iterations,
            "idempotent_ingest_iterations": idempotent_ingest_iterations,
            "recall_iterations": recall_iterations,
            "rss_sample_every_calls": sample_every,
        },
        "initialization_ms": round(initialization_ms, 3),
        "latency": {
            "ingest": latency_summary(ingest_latencies),
            "idempotent_ingest": latency_summary(idempotent_ingest_latencies),
            "recall": latency_summary(recall_latencies),
            "universal_budget_applied": False,
        },
        "rss": {
            "corpus_build": rss_analysis(ingest_rss, calls_per_sample=sample_every),
            "repeat_idempotent_ingest": rss_analysis(
                idempotent_ingest_rss,
                calls_per_sample=sample_every,
            ),
            "repeat_recall": recall_rss_result,
            "bounded_window_conclusion": None,
            "scope_warning": (
                "Finite-run evidence only; this is not a proof for unlimited corpus size."
            ),
        },
        "process_ownership": {
            "child_pids_before": sorted(children_before),
            "child_pids_after": sorted(children_after),
            "new_child_pids_after_close": sorted(children_after - children_before),
        },
        "excluded": {
            "mode_b_ollama_e2e": "external model/host dependency; measure separately",
            "mode_c_provider_e2e": "external provider/network dependency; measure separately",
            "quality": "resource harness does not claim retrieval accuracy",
        },
    }
    return _finalize_conclusion(result)


def _finalize_conclusion(result: dict) -> dict:
    """Set the bounded conclusion only when both fixed-state tails plateau."""
    stable_ingest = result["rss"]["repeat_idempotent_ingest"][
        "tail_plateau_observed"
    ]
    stable_recall = result["rss"]["repeat_recall"]["tail_plateau_observed"]
    result["rss"]["bounded_window_conclusion"] = (
        "no_unbounded_growth_detected"
        if stable_ingest and stable_recall
        else "growth_signal_requires_investigation"
    )
    return result


def render_markdown(result: dict) -> str:
    """Render one evidence result without turning it into a universal claim."""
    ingest = result["latency"]["ingest"]
    idempotent_ingest = result["latency"]["idempotent_ingest"]
    recall = result["latency"]["recall"]
    ingest_rss = result["rss"]["corpus_build"]
    idempotent_ingest_rss = result["rss"]["repeat_idempotent_ingest"]
    recall_rss = result["rss"]["repeat_recall"]
    runtime = result["runtime"]
    protocol = result["protocol"]
    return "\n".join([
        "# Local resource-lifecycle evidence",
        "",
        f"Generated: `{result['generated_at']}`  ",
        f"Source: `{result['source_commit']}`  ",
        f"Package: `{result['product_version']}`",
        "",
        "## Protocol",
        "",
        f"- Mode A shipped local engine with `{protocol['embedding_boundary']}`.",
        f"- Warm-up: {protocol['warmup_iterations']} ingest+recall pairs.",
        f"- Measured: {protocol['ingest_iterations']} ingests and "
        f"{protocol['idempotent_ingest_iterations']} fixed-state retry ingests and "
        f"{protocol['recall_iterations']} fixed-corpus recalls.",
        "- Temporary SQLite namespace; no provider call or network dependency.",
        "",
        "## Machine",
        "",
        f"`{runtime['platform']}`, `{runtime['machine']}`, Python "
        f"`{runtime['python']}`, {runtime['logical_cpus']} logical CPUs, "
        f"{runtime['system_memory_mb']} MiB system memory.",
        "",
        "## Measured latency",
        "",
        "| path | samples | p50 | p95 | min | max |",
        "|---|---:|---:|---:|---:|---:|",
        f"| canonical ingest | {ingest['samples']} | {ingest['p50_ms']} ms | "
        f"{ingest['p95_ms']} ms | {ingest['min_ms']} ms | {ingest['max_ms']} ms |",
        f"| idempotent retry ingest | {idempotent_ingest['samples']} | "
        f"{idempotent_ingest['p50_ms']} ms | {idempotent_ingest['p95_ms']} ms | "
        f"{idempotent_ingest['min_ms']} ms | {idempotent_ingest['max_ms']} ms |",
        f"| repeat recall | {recall['samples']} | {recall['p50_ms']} ms | "
        f"{recall['p95_ms']} ms | {recall['min_ms']} ms | {recall['max_ms']} ms |",
        "",
        "No universal latency budget is applied. These distributions belong to the "
        "machine, corpus, source commit, and mock model boundary above.",
        "",
        "## RSS evidence",
        "",
        "| phase | baseline | final | peak | growth | tail span | tail slope / 100 calls |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| corpus build (expected index growth) | {ingest_rss['baseline_mb']} MiB | "
        f"{ingest_rss['final_mb']} MiB | {ingest_rss['peak_mb']} MiB | "
        f"{ingest_rss['growth_mb']} MiB | {ingest_rss['tail_span_mb']} MiB | "
        f"{ingest_rss['tail_slope_mb_per_100_calls']} MiB |",
        f"| fixed-state idempotent ingest | {idempotent_ingest_rss['baseline_mb']} MiB | "
        f"{idempotent_ingest_rss['final_mb']} MiB | "
        f"{idempotent_ingest_rss['peak_mb']} MiB | "
        f"{idempotent_ingest_rss['growth_mb']} MiB | "
        f"{idempotent_ingest_rss['tail_span_mb']} MiB | "
        f"{idempotent_ingest_rss['tail_slope_mb_per_100_calls']} MiB |",
        f"| fixed-corpus repeat recall | {recall_rss['baseline_mb']} MiB | "
        f"{recall_rss['final_mb']} MiB | {recall_rss['peak_mb']} MiB | "
        f"{recall_rss['growth_mb']} MiB | {recall_rss['tail_span_mb']} MiB | "
        f"{recall_rss['tail_slope_mb_per_100_calls']} MiB |",
        "",
        f"Bounded-window conclusion: **{result['rss']['bounded_window_conclusion']}**. "
        "This finite run cannot prove behavior at unlimited corpus size.",
        "",
        "## Explicit exclusions",
        "",
        "Mode B Ollama and Mode C cloud-provider end-to-end latency are external "
        "model, host, provider, and network measurements. They are not inferred "
        "from this local run. Retrieval quality is also outside this harness.",
        "",
    ])


def write_results(result: dict, output_dir: Path) -> tuple[Path, Path]:
    """Write raw JSON and the matching human-readable report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "RESULTS.json"
    markdown_path = output_dir / "RESULTS.md"
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(result), encoding="utf-8")
    return json_path, markdown_path
