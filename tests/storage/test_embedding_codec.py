"""Tests for the embedding codec (TEXT JSON ↔ binary float32 BLOB).

These tests exercise:
- round-trip float32 fidelity
- dual-format (TEXT row and BLOB row) through one decode call
- corrupt buffer raises ValueError with fact_id in the message (never returns None)
- backfill idempotency on a real SQLite database
- non-null embedding count preserved by backfill
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# The codec module under test.  Import here so the tests fail immediately
# (RED) when the module does not yet exist.
# ---------------------------------------------------------------------------
from superlocalmemory.storage.embedding_codec import (
    EMBEDDING_BYTES,
    EMBEDDING_DIM,
    decode_embedding,
    encode_embedding,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_vec(seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(EMBEDDING_DIM).tolist()


def _vec_to_text(vec: list[float]) -> str:
    return json.dumps(vec)


def _vec_to_blob(vec: list[float]) -> bytes:
    return np.array(vec, dtype=np.float32).tobytes()


def _cosine(a: list[float], b: list[float]) -> float:
    u = np.array(a, dtype=np.float64)
    v = np.array(b, dtype=np.float64)
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu == 0 or nv == 0:
        return 0.0
    return float(np.dot(u, v) / (nu * nv))


# ---------------------------------------------------------------------------
# 1. Round-trip float32 fidelity
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def test_encode_then_decode_preserves_values(self):
        vec = _random_vec(seed=1)
        blob = encode_embedding(vec)
        assert isinstance(blob, bytes)
        assert len(blob) == EMBEDDING_BYTES
        restored = decode_embedding(blob)
        assert restored is not None
        assert len(restored) == EMBEDDING_DIM
        # float32 round-trip: values should be identical after encode→decode
        expected = np.array(vec, dtype=np.float32).tolist()
        assert restored == pytest.approx(expected, abs=0.0)

    def test_cosine_similarity_preserved_to_6dp(self):
        a = _random_vec(seed=2)
        b = _random_vec(seed=3)
        cos_before = _cosine(a, b)
        a_restored = decode_embedding(encode_embedding(a))
        b_restored = decode_embedding(encode_embedding(b))
        cos_after = _cosine(a_restored, b_restored)
        # ≥ 6 decimal places
        assert abs(cos_before - cos_after) < 5e-7, (
            f"cosine shifted by {abs(cos_before - cos_after):.2e}"
        )

    def test_encode_none_returns_none(self):
        assert encode_embedding(None) is None

    def test_decode_none_returns_none(self):
        assert decode_embedding(None) is None

    def test_decode_empty_string_returns_none(self):
        assert decode_embedding("") is None

    def test_encode_short_vec_produces_bytes(self):
        """encode_embedding accepts any dimension; dimension validation is in
        the backfill script, not the codec.  This test confirms the codec does
        not raise for non-standard sizes."""
        short_vec = [0.0] * 4
        blob = encode_embedding(short_vec)
        assert isinstance(blob, bytes)
        assert len(blob) == 16  # 4 floats × 4 bytes


# ---------------------------------------------------------------------------
# 2. Dual-format: TEXT and BLOB decoded via the same call
# ---------------------------------------------------------------------------

class TestDualFormat:
    """One decode call must handle both TEXT (legacy) and BLOB (new) rows."""

    def test_decode_text_json(self):
        vec = _random_vec(seed=4)
        text_raw = _vec_to_text(vec)
        result = decode_embedding(text_raw)
        assert result is not None
        assert len(result) == EMBEDDING_DIM
        assert result == pytest.approx(vec, rel=1e-6)

    def test_decode_blob(self):
        vec = _random_vec(seed=5)
        blob_raw = _vec_to_blob(vec)
        result = decode_embedding(blob_raw)
        assert result is not None
        assert len(result) == EMBEDDING_DIM
        expected = np.frombuffer(blob_raw, dtype=np.float32).tolist()
        assert result == pytest.approx(expected, abs=0.0)

    def test_text_and_blob_same_values_produce_same_result(self):
        """TEXT row and BLOB row for the same embedding decode identically."""
        vec = _random_vec(seed=6)
        text_raw = _vec_to_text(vec)
        blob_raw = _vec_to_blob(vec)
        text_decoded = decode_embedding(text_raw)
        blob_decoded = decode_embedding(blob_raw)
        assert text_decoded == pytest.approx(blob_decoded, abs=1e-6)

    def test_mixed_rows_in_one_pass(self):
        """Decode a mix of TEXT and BLOB rows without error — simulates rollout window."""
        rows = []
        for i in range(5):
            vec = _random_vec(seed=10 + i)
            if i % 2 == 0:
                rows.append(("text", _vec_to_text(vec), vec))
            else:
                rows.append(("blob", _vec_to_blob(vec), vec))
        for fmt, raw, original in rows:
            decoded = decode_embedding(raw)
            assert decoded is not None, f"Failed for {fmt} row"
            assert len(decoded) == EMBEDDING_DIM


# ---------------------------------------------------------------------------
# 3. Corrupt buffer raises ValueError (never returns None silently)
# ---------------------------------------------------------------------------

class TestCorruptData:
    def test_wrong_byte_length_raises(self):
        # 17 bytes is not a multiple of 4 — must raise
        bad_blob = b"\x00" * 17
        with pytest.raises(ValueError, match="multiple of 4"):
            decode_embedding(bad_blob, fact_id="fact-xyz")

    def test_fact_id_in_error_message(self):
        # 101 bytes: not a multiple of 4, so ValueError fires with fact_id in message
        bad_blob = b"\xff" * 101
        with pytest.raises(ValueError, match="fact-abc"):
            decode_embedding(bad_blob, fact_id="fact-abc")

    def test_corrupt_json_text_raises(self):
        bad_text = "not valid json {"
        with pytest.raises(ValueError, match="Corrupt"):
            decode_embedding(bad_text, fact_id="fact-bad-json")

    def test_unexpected_type_raises(self):
        with pytest.raises(ValueError, match="Unexpected"):
            decode_embedding(42, fact_id="fact-int")

    def test_does_not_return_none_for_bad_blob(self) -> None:
        """A corrupt value must RAISE, never come back as "no embedding".

        Returning None for corruption is the forbidden path: the caller cannot
        tell it apart from a fact that legitimately has no embedding yet, so
        damage looks like absence and nothing ever reports it.

        This previously initialised `result = None`, caught ValueError with
        `pass`, and asserted `result is None` — which is satisfied by the very
        behaviour it was meant to forbid. It now requires the exception, and
        requires the fact id to be in the message so the row can be found.
        """
        with pytest.raises(ValueError, match="deadbeef"):
            decode_embedding(b"\x01\x02\x03", fact_id="deadbeef")

    def test_a_valid_buffer_is_not_mistaken_for_corruption(self) -> None:
        """The guard above must not fire on good data."""
        import numpy as np

        good = np.array([0.5] * EMBEDDING_DIM, dtype=np.float32).tobytes()
        assert len(decode_embedding(good, fact_id="ok")) == EMBEDDING_DIM


# ---------------------------------------------------------------------------
# 4 & 5. Backfill idempotency and non-null count preserved
# ---------------------------------------------------------------------------
def _make_test_db(path: Path, n_facts: int = 20) -> None:
    """Create a minimal atomic_facts table with TEXT embeddings.

    The column is declared TEXT because that is what the real schema says.
    Declaring it BLOB here made an affinity problem on a real store impossible
    to catch from these tests.
    """
    conn = sqlite3.connect(str(path))
    conn.execute("""
        CREATE TABLE atomic_facts (
            fact_id TEXT PRIMARY KEY,
            content TEXT,
            embedding TEXT
        )
    """)
    for i in range(n_facts):
        vec = _random_vec(seed=i)
        text = json.dumps(vec)
        conn.execute(
            "INSERT INTO atomic_facts(fact_id, content, embedding) VALUES (?,?,?)",
            (f"fact-{i:04d}", f"content {i}", text),
        )
    conn.commit()
    conn.close()


class TestBackfill:
    def test_backfill_converts_all_text_to_blob(self, tmp_path):
        db_path = tmp_path / "test.db"
        _make_test_db(db_path, n_facts=20)

        # Import the backfill function
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
        from backfill_embeddings import backfill  # type: ignore

        backfill(db_path)

        conn = sqlite3.connect(str(db_path))
        text_count = conn.execute(
            "SELECT COUNT(*) FROM atomic_facts WHERE typeof(embedding)='text'"
        ).fetchone()[0]
        blob_count = conn.execute(
            "SELECT COUNT(*) FROM atomic_facts WHERE typeof(embedding)='blob'"
        ).fetchone()[0]
        conn.close()

        assert text_count == 0, f"Still {text_count} TEXT embeddings after backfill"
        assert blob_count == 20, f"Expected 20 BLOBs, got {blob_count}"

    def test_backfill_idempotent(self, tmp_path):
        """Running backfill twice must produce the same result."""
        db_path = tmp_path / "test.db"
        _make_test_db(db_path, n_facts=10)

        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
        from backfill_embeddings import backfill  # type: ignore

        backfill(db_path)
        backfill(db_path)  # second run — must not crash or change counts

        conn = sqlite3.connect(str(db_path))
        text_count = conn.execute(
            "SELECT COUNT(*) FROM atomic_facts WHERE typeof(embedding)='text'"
        ).fetchone()[0]
        blob_count = conn.execute(
            "SELECT COUNT(*) FROM atomic_facts WHERE typeof(embedding)='blob'"
        ).fetchone()[0]
        conn.close()

        assert text_count == 0
        assert blob_count == 10

    def test_backfill_preserves_non_null_count(self, tmp_path):
        db_path = tmp_path / "test.db"
        _make_test_db(db_path, n_facts=15)

        conn = sqlite3.connect(str(db_path))
        pre_count = conn.execute(
            "SELECT COUNT(*) FROM atomic_facts WHERE embedding IS NOT NULL"
        ).fetchone()[0]
        conn.close()

        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
        from backfill_embeddings import backfill  # type: ignore

        backfill(db_path)

        conn = sqlite3.connect(str(db_path))
        post_count = conn.execute(
            "SELECT COUNT(*) FROM atomic_facts WHERE embedding IS NOT NULL"
        ).fetchone()[0]
        conn.close()

        assert pre_count == post_count, (
            f"Non-null count changed: {pre_count} → {post_count}"
        )

    def test_backfill_blob_content_roundtrips(self, tmp_path):
        """After backfill each BLOB decodes to the same values as the original JSON."""
        db_path = tmp_path / "test.db"
        n = 5

        originals: dict[str, list[float]] = {}
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE atomic_facts (
                -- Declared TEXT because that is what the real schema says.
                -- Creating it as BLOB here made an affinity problem on a real
                -- store impossible to catch from these tests.
                fact_id TEXT PRIMARY KEY, content TEXT, embedding TEXT
            )
        """)
        for i in range(n):
            vec = _random_vec(seed=100 + i)
            originals[f"fact-{i}"] = vec
            conn.execute(
                "INSERT INTO atomic_facts VALUES (?,?,?)",
                (f"fact-{i}", f"c{i}", json.dumps(vec)),
            )
        conn.commit()
        conn.close()

        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
        from backfill_embeddings import backfill  # type: ignore

        backfill(db_path)

        conn = sqlite3.connect(str(db_path))
        rows = conn.execute(
            "SELECT fact_id, embedding FROM atomic_facts WHERE embedding IS NOT NULL"
        ).fetchall()
        conn.close()

        for fact_id, blob in rows:
            decoded = decode_embedding(blob, fact_id=fact_id)
            original_float32 = np.array(originals[fact_id], dtype=np.float32).tolist()
            assert decoded == pytest.approx(original_float32, abs=0.0), (
                f"{fact_id}: decoded values differ from original float32 representation"
            )

    def test_backfill_bytes_per_fact(self, tmp_path):
        """Every converted embedding must be exactly 3072 bytes."""
        db_path = tmp_path / "test.db"
        _make_test_db(db_path, n_facts=5)

        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
        from backfill_embeddings import backfill  # type: ignore

        backfill(db_path)

        conn = sqlite3.connect(str(db_path))
        rows = conn.execute(
            "SELECT fact_id, length(embedding) as blen FROM atomic_facts "
            "WHERE embedding IS NOT NULL"
        ).fetchall()
        conn.close()

        for fact_id, blen in rows:
            assert blen == EMBEDDING_BYTES, (
                f"{fact_id}: {blen} bytes, expected {EMBEDDING_BYTES}"
            )


# ---------------------------------------------------------------------------
# Conversion must terminate when unconvertible rows are present
# ---------------------------------------------------------------------------

class TestBackfillTerminatesOnUnconvertibleRows:
    """Skipped (unconvertible) rows must not stall the run.

    Without a cursor advance past skipped rows, the loop re-queries the same
    rows forever, sleeping 50 ms each iteration.  A 5-second thread timeout
    detects that case fast enough to be useful in CI.
    """

    def _make_db_with_hostile_rows(self, path: Path) -> None:
        conn = sqlite3.connect(str(path))
        conn.execute(
            "CREATE TABLE atomic_facts "
            "(fact_id TEXT PRIMARY KEY, content TEXT, embedding TEXT)"
        )
        # 5 valid 768-dim rows
        for i in range(5):
            vec = _random_vec(seed=200 + i)
            conn.execute(
                "INSERT INTO atomic_facts VALUES (?,?,?)",
                (f"valid-{i:04d}", f"c{i}", json.dumps(vec)),
            )
        # Unconvertible rows — wrong dimension, invalid JSON, empty list
        conn.execute(
            "INSERT INTO atomic_facts VALUES (?,?,?)",
            ("bad-dim", "x", json.dumps([0.1, 0.2])),   # wrong dim
        )
        conn.execute(
            "INSERT INTO atomic_facts VALUES (?,?,?)",
            ("bad-json", "x", "{not valid json"),         # invalid JSON
        )
        conn.execute(
            "INSERT INTO atomic_facts VALUES (?,?,?)",
            ("bad-empty", "x", "[]"),                     # empty list
        )
        conn.commit()
        conn.close()

    def test_backfill_terminates_and_converts_valid_rows(self, tmp_path):
        import sys
        import threading

        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
        from backfill_embeddings import backfill  # type: ignore

        db_path = tmp_path / "hostile.db"
        self._make_db_with_hostile_rows(db_path)

        result: dict = {"done": False, "converted": None, "error": None}

        def _run() -> None:
            try:
                result["converted"] = backfill(db_path)
                result["done"] = True
            except Exception as exc:
                result["error"] = exc

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        thread.join(timeout=5.0)

        assert result["done"], (
            "backfill did not finish within 5 s — infinite loop on unconvertible row. "
            f"error={result['error']!r}"
        )
        assert result["converted"] == 5, (
            f"Expected 5 conversions, got {result['converted']}"
        )

        conn = sqlite3.connect(str(db_path))
        blob_count = conn.execute(
            "SELECT COUNT(*) FROM atomic_facts WHERE typeof(embedding)='blob'"
        ).fetchone()[0]
        text_count = conn.execute(
            "SELECT COUNT(*) FROM atomic_facts WHERE typeof(embedding)='text'"
        ).fetchone()[0]
        conn.close()

        assert blob_count == 5, f"Expected 5 BLOBs after backfill, got {blob_count}"
        assert text_count == 3, (
            f"Expected 3 TEXT rows (unconvertible), got {text_count}"
        )

    def test_backfill_all_rows_unconvertible_terminates_immediately(self, tmp_path):
        """A store where every row is unconvertible must also terminate."""
        import sys
        import threading

        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
        from backfill_embeddings import backfill  # type: ignore

        db_path = tmp_path / "all_bad.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "CREATE TABLE atomic_facts "
            "(fact_id TEXT PRIMARY KEY, content TEXT, embedding TEXT)"
        )
        for i in range(10):
            conn.execute(
                "INSERT INTO atomic_facts VALUES (?,?,?)",
                (f"bad-{i}", "x", json.dumps([0.0] * 3)),  # wrong dim
            )
        conn.commit()
        conn.close()

        result: dict = {"done": False}

        def _run() -> None:
            backfill(db_path)
            result["done"] = True

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        thread.join(timeout=5.0)

        assert result["done"], (
            "backfill hung when every row was unconvertible"
        )
