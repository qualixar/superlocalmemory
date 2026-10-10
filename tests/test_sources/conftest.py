"""Shared fixtures: a fake memory writer, a throwaway data folder and a configured host."""

from __future__ import annotations

import types
from pathlib import Path

import pytest

from superlocalmemory import sources
from superlocalmemory.media import open_media_store
from superlocalmemory.sources.host import SourceHost


class FakeRuntime:
    ready = True

    def __init__(self) -> None:
        self.saved: list[dict] = []
        self.archived: list[str] = []
        self._by_key: dict[str, dict] = {}

    def remember(self, admission, actor, deadline_ms=0, accept_after_ms=0):
        key = admission.idempotency_key
        if key not in self._by_key:
            n = len(self.saved) + 1
            row = dict(content=admission.content, metadata=dict(admission.metadata), key=key,
                       mid=f"m{n}", fid=f"f{n}", profile=admission.profile_id,
                       source_type=admission.source_type, actor=admission.trusted_actor_id, scope=admission.scope,
                       session_date=admission.session_date)
            self.saved.append(row)
            self._by_key[key] = row
        row = self._by_key[key]
        return types.SimpleNamespace(payload={
            "status": "stored", "memory_id": row["mid"], "fact_ids": [row["fid"]],
            "operation_id": "op-" + row["mid"]})

    def archive_fact(self, profile_id, fact_id, *, idempotency_key=None):
        self.archived.append(fact_id)
        return {"ok": True}

    def contents(self) -> list[str]:
        return [r["content"] for r in self.saved]


class Env:
    def __init__(self, tmp_path: Path) -> None:
        self.data = tmp_path / "data"
        self.data.mkdir()
        self.root = tmp_path / "vault"
        self.root.mkdir()
        self.runtime = FakeRuntime()
        self.remote = False
        self.sleeps: list[float] = []
        self.erased: list[tuple] = []
        self.woken = 0
        self.host = SourceHost(
            remote_check=lambda: self.remote, runtime=lambda: self.runtime, config=lambda: None,
            eraser=self._erase, profile=lambda: "default", actor_id=lambda: "test-actor",
            data_root=self.data, sleep=self.sleeps.append, wake=self._wake)

    def _erase(self, profile_id, fact_ids, subject_id):
        self.erased.append((profile_id, tuple(fact_ids), subject_id))
        return {"erasure_complete": 1}

    def _wake(self):
        self.woken += 1

    def write(self, rel: str, text: str | bytes = "hello") -> Path:
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text if isinstance(text, bytes) else text.encode())
        return path

    def store(self):
        return open_media_store(data_root=self.data)

    def add_and_confirm(self) -> str:
        preview = sources.add_source(self.root, profile_id="default")
        sources.confirm_source(preview.source_id)
        return preview.source_id

    def scan(self, source_id: str):
        from superlocalmemory.sources.reconcile import scan_source
        from superlocalmemory.sources.store import SourceStore

        media = self.store()
        try:
            store = SourceStore(media)
            return scan_source(self.host, store, store.get_source(source_id))
        finally:
            media.close()

    def files(self, source_id: str) -> dict[str, dict]:
        from superlocalmemory.sources.store import SourceStore

        media = self.store()
        try:
            return {r["relpath"]: r for r in SourceStore(media).files(source_id)}
        finally:
            media.close()


@pytest.fixture
def env(tmp_path, monkeypatch):
    e = Env(tmp_path)
    monkeypatch.setenv("SLM_DATA_DIR", str(e.data))  # the data folder is the host's, not the whole temp folder
    sources.configure(e.host)
    yield e
    sources.configure(None)
