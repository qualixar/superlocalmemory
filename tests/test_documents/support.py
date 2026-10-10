"""Fakes shared by the document tests: a worker client, a writer, and fake parse scripts."""

from __future__ import annotations

import base64
import hashlib
import json
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

from superlocalmemory.documents.runner import DocumentJobService, Limits, RunnerDeps
from superlocalmemory.media.ingest import MediaInput
from tests.test_documents.pdfs import make_pdf

DIM = 8
KEY = "sk-abcdefghijklmnopqrstuvwxyz0123456789ABCD"


class FakeClient:
    """Answers the worker calls the pipeline makes; the page number is in the picture's bytes."""

    model_id, revision, dim = "fake:test", "r1", DIM

    def __init__(self, ocr=None, engine="rapidocr"):
        self.ocr, self.engine = ocr or {}, engine
        self.calls: list[tuple[str, int]] = []

    def _page(self, path) -> int:
        try:
            return int(Path(path).read_bytes().split(b"-")[-1])
        except ValueError:
            return 0

    def prepare_image(self, path, out_dir, *, wait_cold=True):
        self.calls.append(("prepare", self._page(path)))
        stored = Path(out_dir) / "stored.png"
        stored.write_bytes(b"STRIPPED")
        thumb = Path(out_dir) / "thumb.webp"
        thumb.write_bytes(b"RIFFthumb")
        return {"mime": "image/png", "width": 3, "height": 2, "exif": {}, "phash": hashlib.sha256(
            Path(path).read_bytes()).hexdigest()[:16], "stored_path": str(stored), "stored_ext": "png",
            "thumb_path": str(thumb)}

    def ocr_image(self, path, *, wait_cold=True):
        page = self._page(path)
        self.calls.append(("ocr", page))
        return {"engine": self.engine, "text": self.ocr.get(page, "")}

    def embed_images(self, paths, *, wait_cold=True):
        self.calls.append(("embed", self._page(paths[0])))
        return [[1.0] + [0.0] * (DIM - 1) for _ in paths]

    def kinds(self, kind):
        return [p for k, p in self.calls if k == kind]


class Runtime:
    """A writer that remembers by idempotency key, like the real one."""

    def __init__(self, on_remember=None):
        self.requests, self.archived, self.on_remember = [], [], on_remember
        self._by_key: dict[str, tuple[str, str]] = {}

    def remember(self, admission, actor, *, deadline_ms, accept_after_ms):
        if self.on_remember:
            self.on_remember(admission)
        key = admission.idempotency_key
        if key not in self._by_key:
            n = len(self._by_key) + 1
            self._by_key[key] = (f"mem{n}", f"fact{n}")
        self.requests.append(admission)
        memory_id, fact_id = self._by_key[key]
        return SimpleNamespace(payload={"status": "queryable", "operation_id": "op", "fact_ids": [fact_id],
                                        "memory_id": memory_id})

    def archive_fact(self, profile_id, fact_id, *, idempotency_key=None):
        self.archived.append((profile_id, fact_id))
        return {"ok": True, "archived_at": "now"}

    @property
    def keys(self):
        return [r.idempotency_key for r in self.requests]


_SCRIPT = textwrap.dedent('''
    import json, os, sys, time
    spec = json.loads({spec!r})
    req = json.loads(sys.stdin.readline())
    skip = set(req.get("skip") or [])
    def emit(o):
        sys.stdout.write(json.dumps(o) + "\\n"); sys.stdout.flush()
    if spec.get("error"):
        emit({{"error": spec["error"]}}); sys.exit(0)
    emit({{"opened": True, "page_count": len(spec["pages"])}})
    hold = []
    for n, page in enumerate(spec["pages"], 1):
        if n in skip:
            continue
        if spec.get("stall_on") == n:
            time.sleep(60)
        if spec.get("bloat_on") == n:
            hold.append(bytearray(250 * 1024 * 1024)); hold[0][::4096] = b"x" * len(hold[0][::4096]); time.sleep(60)
        path = os.path.join(req["out_dir"], "page-%d.png" % n)
        open(path, "wb").write(b"PNG-%d" % n)
        emit({{"page_no": n, "png_path": path, "text": page.get("text", ""),
              "has_text_layer": bool(page.get("text")), "width": 10, "height": 10}})
        if not sys.stdin.readline():
            sys.exit(0)
    emit({{"done": True, "page_count": len(spec["pages"]), "title": spec.get("title", ""),
          "created": spec.get("created", "")}})
''')


def fake_script(directory: Path, pages: list[str], **spec) -> Path:
    """A parse script that speaks the same lines as the real one; ``pages`` are text-layer texts."""
    spec["pages"] = [{"text": t} for t in pages]
    path = Path(directory) / "fake_parse.py"
    path.write_text(_SCRIPT.format(spec=json.dumps(spec)))
    return path


def pdf_input(pages=("one",), *, file_name="", **kw) -> MediaInput:
    return MediaInput(base64=base64.b64encode(make_pdf(list(pages), **kw)).decode(), file_name=file_name)


def make_service(store, runtime, client, script, *, config=None, enabled=True, deps=None, **limits) -> DocumentJobService:
    lim = dict(page_timeout_s=30.0, job_timeout_s=120.0, rss_limit_mb=0, max_pages=500, lease_s=60.0, poll_s=0.1, idle_s=0.1)
    lim.update(limits)
    deps_obj = RunnerDeps(
        store_factory=lambda: store, client_supplier=lambda: client, runtime_supplier=lambda: runtime,
        config_supplier=lambda: config or SimpleNamespace(pii_redaction=False),
        python_supplier=lambda: Path(sys.executable), enabled=lambda: enabled, script=script,
        limits=Limits(**lim), owns_store=False, **(deps or {}))
    return DocumentJobService(deps_obj)
