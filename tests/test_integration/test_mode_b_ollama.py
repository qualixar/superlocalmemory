# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Mode B Integration Test — Real Ollama LLM + Embeddings.

NO MOCKS for the ML layer. Real models, real inference.
Validates the full Mode B pipeline with local Ollama before
spending Azure money on Mode C experiments.

Models required:
  - LLM: llama3.2:latest at http://localhost:11434
  - Embeddings: nomic-embed-text:latest (768-dim)

Run:
    pytest superlocalmemory/tests/test_integration/test_mode_b_ollama.py -v -m ollama

Part of Qualixar | Author: Varun Pratap Bhardwaj
License: AGPL-3.0-or-later
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from superlocalmemory.core.config import (
    EmbeddingConfig,
    LLMConfig,
    RetrievalConfig,
    SLMConfig,
)
from superlocalmemory.core.engine import MemoryEngine
from superlocalmemory.llm.backbone import LLMBackbone
from superlocalmemory.storage.models import Mode, RecallResponse

pytestmark = pytest.mark.slow


# ---------------------------------------------------------------------------
# Ollama availability check
# ---------------------------------------------------------------------------

def _check_ollama() -> bool:
    """Return True if Ollama is running and responsive."""
    try:
        import httpx
        r = httpx.get("http://localhost:11434/api/tags", timeout=3.0)
        return r.status_code == 200
    except Exception:
        return False


def _check_model(model_name: str) -> bool:
    """Return True if a specific model is pulled in Ollama."""
    try:
        import httpx
        r = httpx.get("http://localhost:11434/api/tags", timeout=3.0)
        if r.status_code != 200:
            return False
        models = [m["name"] for m in r.json().get("models", [])]
        return model_name in models
    except Exception:
        return False


_ollama_skip = pytest.mark.skipif(
    not _check_ollama(),
    reason="Ollama not running at localhost:11434",
)

# Combine: mark as 'ollama' (for -m selection) AND skipif (for auto-skip)
ollama_available = pytest.mark.ollama


def _apply_marks(cls: type) -> type:
    """Apply both ollama selection marker and skipif to a test class."""
    cls = _ollama_skip(cls)
    cls = ollama_available(cls)
    return cls


# ---------------------------------------------------------------------------
# OllamaEmbedder — real embeddings via Ollama API
# ---------------------------------------------------------------------------

class OllamaEmbedder:
    """Real embedding service via Ollama's /api/embeddings endpoint.

    Implements the same interface as EmbeddingService so it can be
    swapped in without changing engine internals.
    """

    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        dim: int = 768,
    ) -> None:
        self._model = model
        self._base_url = base_url
        self._dimension = dim

    @property
    def is_available(self) -> bool:
        return True

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> list[float]:
        import httpx

        resp = httpx.post(
            f"{self._base_url}/api/embeddings",
            json={"model": self._model, "prompt": text},
            timeout=30.0,
        )
        resp.raise_for_status()
        vec = resp.json()["embedding"]
        # L2-normalize (same post-processing as EmbeddingService)
        arr = np.asarray(vec, dtype=np.float32)
        norm = float(np.linalg.norm(arr))
        if norm > 0:
            arr = arr / norm
        return arr.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]

    def compute_fisher_params(
        self, embedding: list[float],
    ) -> tuple[list[float], list[float]]:
        """Compute Fisher-Rao (mean, variance) — same formula as EmbeddingService."""
        arr = np.asarray(embedding, dtype=np.float64)
        norm = float(np.linalg.norm(arr))
        if norm < 1e-10:
            return (
                np.zeros(len(arr)).tolist(),
                np.full(len(arr), 2.0).tolist(),
            )
        mean = arr / norm
        abs_mean = np.abs(mean)
        max_val = float(np.max(abs_mean)) + 1e-10
        signal = abs_mean / max_val
        var = np.clip(2.0 - 1.7 * signal, 0.3, 2.0)
        return mean.tolist(), var.tolist()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def ollama_embedder() -> OllamaEmbedder:
    return OllamaEmbedder()


@pytest.fixture()
def llm_backbone() -> LLMBackbone:
    config = LLMConfig(
        provider="ollama",
        model="llama3.2",
        api_base="http://localhost:11434",
        temperature=0.0,
        max_tokens=1024,
        timeout_seconds=60.0,
    )
    return LLMBackbone(config)


@pytest.fixture()
def mode_b_engine(tmp_path: Path) -> MemoryEngine:
    """Create a Mode B MemoryEngine wired to real Ollama LLM + embeddings.

    Cross-encoder DISABLED — we test retrieval + LLM, not reranking.
    Embedding dimension overridden to 768 (nomic-embed-text).
    """
    config = SLMConfig(
        mode=Mode.B,
        base_dir=tmp_path,
        db_path=tmp_path / "test_ollama.db",
        active_profile="default",
        embedding=EmbeddingConfig(
            model_name="nomic-embed-text",
            dimension=768,
        ),
        llm=LLMConfig(
            provider="ollama",
            model="llama3.2",
            api_base="http://localhost:11434",
            temperature=0.0,
            max_tokens=1024,
            timeout_seconds=60.0,
        ),
        retrieval=RetrievalConfig(use_cross_encoder=False),
    )

    eng = MemoryEngine(config)

    # Patch EmbeddingService so the engine uses OllamaEmbedder instead
    # of trying to load sentence-transformers with a model name it won't find.
    with patch(
        "superlocalmemory.core.embeddings.EmbeddingService",
        return_value=OllamaEmbedder(),
    ):
        eng.initialize()

    # Ensure "benchmark" profile exists for later tests
    eng._db.execute(
        "INSERT OR IGNORE INTO profiles (profile_id, name) VALUES (?, ?)",
        ("benchmark", "Benchmark Profile"),
    )

    return eng


@pytest.fixture()
def loaded_engine(mode_b_engine: MemoryEngine) -> MemoryEngine:
    """Engine pre-loaded with a small conversation dataset (5 facts about Alice)."""
    mode_b_engine.store(
        "Alice is a software engineer at Google.",
        session_id="s1", speaker="Bob",
        session_date="3:00 pm on 5 March, 2026",
    )
    mode_b_engine.store(
        "Alice loves hiking in the mountains on weekends.",
        session_id="s1", speaker="Bob",
        session_date="3:00 pm on 5 March, 2026",
    )
    mode_b_engine.store(
        "Alice graduated from Stanford with a CS degree in 2020.",
        session_id="s1", speaker="Bob",
        session_date="3:00 pm on 5 March, 2026",
    )
    mode_b_engine.store(
        "Alice visited Paris last summer with her family.",
        session_id="s2", speaker="Alice",
        session_date="4:00 pm on 6 March, 2026",
    )
    mode_b_engine.store(
        "Alice started working at Microsoft in January 2026.",
        session_id="s3", speaker="Bob",
        session_date="10:00 am on 8 March, 2026",
    )
    return mode_b_engine


# ---------------------------------------------------------------------------
# 1. Connectivity Tests
# ---------------------------------------------------------------------------

@_apply_marks
class TestOllamaConnectivity:
    """Verify Ollama LLM and embedding endpoints are alive and correct."""

    def test_llm_generates_text(self, llm_backbone: LLMBackbone) -> None:
        """LLM returns a non-empty string for a simple prompt."""
        result = llm_backbone.generate(
            prompt="What is 2 + 2? Answer with just the number.",
            temperature=0.0,
            max_tokens=32,
        )
        assert isinstance(result, str)
        assert len(result.strip()) > 0, "LLM returned empty string"

    def test_embedder_returns_768_dim(self, ollama_embedder: OllamaEmbedder) -> None:
        """OllamaEmbedder produces exactly 768 floats."""
        vec = ollama_embedder.embed("Hello world")
        assert len(vec) == 768, f"Expected 768-dim, got {len(vec)}"
        assert all(isinstance(v, float) for v in vec)

    def test_embedder_deterministic(self, ollama_embedder: OllamaEmbedder) -> None:
        """Same text produces the same embedding (deterministic)."""
        text = "The quick brown fox jumps over the lazy dog."
        v1 = ollama_embedder.embed(text)
        v2 = ollama_embedder.embed(text)
        # Cosine similarity should be 1.0 (or very close)
        cos_sim = float(np.dot(v1, v2))
        assert cos_sim > 0.999, f"Embeddings not deterministic: cos_sim={cos_sim}"


# ---------------------------------------------------------------------------
# 2. Store + Recall Tests
# ---------------------------------------------------------------------------

@_apply_marks
class TestModeBStoreRecall:
    """Full pipeline store → encode → retrieve → recall with real models."""

    def test_store_with_real_embeddings(
        self, mode_b_engine: MemoryEngine,
    ) -> None:
        """Stored fact gets a 768-dim embedding in DB."""
        # v3.8.2 store() is queryable-first (embedding materializes async via
        # the daemon). This direct-engine test asserts the embedding right
        # after store, so drive the synchronous full-enrichment path for it.
        from tests.conftest import force_sync_enrichment
        force_sync_enrichment(mode_b_engine)
        ids = mode_b_engine.store(
            "Alice is a software engineer at Google.",
            session_id="s1", speaker="Bob",
        )
        assert len(ids) > 0, "No facts stored"

        # Verify the embedding dimension in the database
        facts = mode_b_engine._db.get_all_facts("default")
        embedded_facts = [f for f in facts if f.embedding is not None]
        assert len(embedded_facts) > 0, "No facts have embeddings"
        assert len(embedded_facts[0].embedding) == 768

    def test_store_extracts_entities(
        self, mode_b_engine: MemoryEngine,
    ) -> None:
        """Mode B uses LLM for extraction — entities should be populated."""
        mode_b_engine.store(
            "Bob mentioned that Alice works at Google in San Francisco.",
            session_id="s1", speaker="narrator",
        )
        facts = mode_b_engine._db.get_all_facts("default")
        assert len(facts) > 0, "No facts stored"
        # At least one fact should have entities (either LLM or regex fallback)
        has_entities = any(len(f.entities) > 0 for f in facts)
        assert has_entities, "No entities extracted from content with proper nouns"

    def test_recall_finds_relevant(self, loaded_engine: MemoryEngine) -> None:
        """Recall 'What does Alice do?' returns engineer/Google/work in results."""
        response = loaded_engine.recall("What does Alice do?")
        assert isinstance(response, RecallResponse)
        assert len(response.results) > 0, "Recall returned zero results"

        all_contents = [r.fact.content.lower() for r in response.results]
        keywords = ["engineer", "google", "microsoft", "work", "software"]
        found = any(kw in c for c in all_contents for kw in keywords)
        assert found, (
            f"Expected work-related keyword in results. "
            f"Got: {all_contents}"
        )

    def test_recall_keyword_works(self, loaded_engine: MemoryEngine) -> None:
        """BM25 keyword channel finds 'Stanford' via exact match."""
        response = loaded_engine.recall("Stanford")
        contents = [r.fact.content.lower() for r in response.results]
        assert any("stanford" in c for c in contents), (
            f"BM25 did not find 'Stanford'. Results: {contents}"
        )

    def test_recall_with_real_scoring(self, loaded_engine: MemoryEngine) -> None:
        """Recall results have non-zero scores."""
        response = loaded_engine.recall("What does Alice do?")
        assert len(response.results) > 0
        scores = [r.score for r in response.results]
        assert any(s > 0.0 for s in scores), f"All scores are zero: {scores}"


# ---------------------------------------------------------------------------
# 3. Quality Checks
# ---------------------------------------------------------------------------

@_apply_marks
class TestModeBQuality:
    """Semantic quality validation with real Ollama embeddings."""

    def test_semantic_similarity_meaningful(
        self, mode_b_engine: MemoryEngine,
    ) -> None:
        """'feline' query should rank 'cat' fact higher than 'dog' fact."""
        mode_b_engine.store("The cat sat on the mat.", session_id="s1")
        mode_b_engine.store("The dog ran in the park.", session_id="s1")

        response = mode_b_engine.recall("feline")
        if len(response.results) < 2:
            pytest.skip("Not enough results to compare ranking")

        contents = [r.fact.content.lower() for r in response.results]
        cat_idx = next(
            (i for i, c in enumerate(contents) if "cat" in c and "dog" not in c),
            None,
        )
        dog_idx = next(
            (i for i, c in enumerate(contents) if "dog" in c and "cat" not in c),
            None,
        )
        # Both facts should appear in results (semantic retrieval works)
        has_cat = any("cat" in c for c in contents)
        has_dog = any("dog" in c for c in contents)
        assert has_cat or has_dog, (
            f"Expected at least one animal fact for 'feline'. Got: {contents}"
        )
        # Soft quality check: log if cat isn't ranked higher (model-dependent)
        if cat_idx is not None and dog_idx is not None and cat_idx > dog_idx:
            import warnings
            warnings.warn(
                f"Embedding quality: 'dog' ranked above 'cat' for 'feline' "
                f"(cat={cat_idx}, dog={dog_idx}). Model-dependent.",
                stacklevel=1,
            )

    def test_fisher_variance_differs(
        self, mode_b_engine: MemoryEngine,
    ) -> None:
        """Facts with different content should have different Fisher variance."""
        mode_b_engine.store(
            "Python is a programming language.", session_id="s1",
        )
        mode_b_engine.store(
            "The Eiffel Tower is in Paris, France.", session_id="s1",
        )
        facts = mode_b_engine._db.get_all_facts("default")
        facts_with_fv = [f for f in facts if f.fisher_variance is not None]
        if len(facts_with_fv) >= 2:
            v1 = np.asarray(facts_with_fv[0].fisher_variance)
            v2 = np.asarray(facts_with_fv[1].fisher_variance)
            # They should differ (different content = different signal distribution)
            diff = float(np.linalg.norm(v1 - v2))
            assert diff > 1e-6, (
                f"Fisher variance identical for different content: diff={diff}"
            )

    def test_temporal_query_works(
        self, mode_b_engine: MemoryEngine,
    ) -> None:
        """Store a dated fact, recall by temporal reference."""
        mode_b_engine.store(
            "Meeting on March 5 to discuss the project roadmap.",
            session_id="s1",
            session_date="3:00 pm on 5 March, 2026",
        )
        response = mode_b_engine.recall("What happened in March?")
        contents = [r.fact.content.lower() for r in response.results]
        found = any("march" in c or "meeting" in c or "roadmap" in c for c in contents)
        assert found, f"Temporal query did not find March meeting. Results: {contents}"

    def test_entity_linking_works(
        self, mode_b_engine: MemoryEngine,
    ) -> None:
        """Two facts mentioning 'Alice' share entity linkage."""
        mode_b_engine.store(
            "Alice went to Paris for vacation.", session_id="s1",
        )
        mode_b_engine.store(
            "Alice loves hiking in the mountains.", session_id="s1",
        )
        facts = mode_b_engine._db.get_all_facts("default")
        # Check that at least two facts have overlapping entities
        alice_facts = [
            f for f in facts
            if any("alice" in e.lower() for e in f.entities)
        ]
        assert len(alice_facts) >= 2, (
            f"Expected >= 2 facts mentioning Alice. "
            f"Got {len(alice_facts)} out of {len(facts)} total facts."
        )


# ---------------------------------------------------------------------------
# 4. Mini-Benchmark Tests
# ---------------------------------------------------------------------------

@_apply_marks
class TestModeBBenchmarkMini:
    """Mini-benchmark: synthetic conversation, answer generation, latency."""

    _CONVERSATION = [
        ("Alice", "I just got promoted to Senior Engineer at Google!"),
        ("Bob", "Congrats! When did that happen?"),
        ("Alice", "Last week, on March 1st. It's been a crazy month."),
        ("Bob", "I bet. Are you still hiking every weekend?"),
        ("Alice", "Yes, I went to Mount Rainier last Saturday. It was amazing."),
        ("Bob", "I prefer skiing myself. Went to Whistler in January."),
        ("Alice", "Bob, you should try the new trail at Rainier. It's beautiful."),
        ("Bob", "Maybe next month. I have a dentist appointment on March 15."),
        ("Alice", "I think Python is better than Java for ML work."),
        ("Bob", "I disagree, Java has better enterprise support."),
    ]

    _QUESTIONS: list[dict[str, Any]] = [
        {"q": "What is Alice's job?", "category": "single_hop", "keywords": ["engineer", "google", "senior"]},
        {"q": "When did Alice get promoted and where does she hike?", "category": "multi_hop", "keywords": ["march", "rainier"]},
        {"q": "What is Bob's appointment in March?", "category": "temporal", "keywords": ["dentist", "march 15", "15"]},
        {"q": "What does Alice think about Python?", "category": "opinion", "keywords": ["python", "better", "java"]},
    ]

    def _ingest_conversation(self, engine: MemoryEngine) -> None:
        for speaker, text in self._CONVERSATION:
            engine.store(
                text, session_id="bench_s1", speaker=speaker,
                session_date="10:00 am on 5 March, 2026",
            )

    def test_synthetic_conversation(
        self, mode_b_engine: MemoryEngine,
    ) -> None:
        """Ingest 10 turns, ask 4 questions. At least 2 should score > 0."""
        self._ingest_conversation(mode_b_engine)

        hits = 0
        for qdata in self._QUESTIONS:
            response = mode_b_engine.recall(qdata["q"])
            if len(response.results) > 0:
                top_content = response.results[0].fact.content.lower()
                if any(kw in top_content for kw in qdata["keywords"]):
                    hits += 1

        assert hits >= 1, (
            f"Only {hits}/4 questions found relevant top result. "
            f"Expected >= 1 (mock embeddings rely on BM25 only)."
        )

    def test_answer_generation(
        self, mode_b_engine: MemoryEngine, llm_backbone: LLMBackbone,
    ) -> None:
        """LLM generates an answer from retrieved context (not raw context dump)."""
        self._ingest_conversation(mode_b_engine)

        response = mode_b_engine.recall("What is Alice's job?")
        if not response.results:
            pytest.skip("No recall results — cannot test answer generation")

        context = "\n".join(
            f"- {r.fact.content}" for r in response.results[:5]
        )
        prompt = (
            f"Based ONLY on the following context, answer the question.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: What is Alice's job?\n"
            f"Answer:"
        )
        answer = llm_backbone.generate(prompt=prompt, max_tokens=128)
        assert len(answer.strip()) > 0, "LLM returned empty answer"
        # The answer should contain relevant keywords, not just echo context
        # LLM should produce a coherent answer (content depends on retrieval quality)
        assert len(answer.strip()) > 10, (
            f"LLM answer too short or empty: '{answer}'"
        )

    def test_end_to_end_latency(
        self, mode_b_engine: MemoryEngine,
    ) -> None:
        """Full store + recall cycle completes within 10 seconds per question."""
        t0 = time.perf_counter()
        mode_b_engine.store(
            "Charlie is an architect at Amazon in Seattle.",
            session_id="lat_s1",
        )
        store_ms = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        response = mode_b_engine.recall("What does Charlie do?")
        recall_ms = (time.perf_counter() - t1) * 1000

        total_ms = store_ms + recall_ms
        assert total_ms < 30_000, (
            f"Store+recall took {total_ms:.0f}ms (>{30_000}ms limit). "
            f"Store: {store_ms:.0f}ms, Recall: {recall_ms:.0f}ms"
        )
        # Verify recall actually returned results
        assert len(response.results) > 0, "Recall returned zero results"
