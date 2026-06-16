# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later
"""Stage-8 security regression: Vertex cache keys must isolate by GCP project + region.

Before this fix, the vertex build_key branch keyed only on the model name (from
the path) + body, dropping the project and location segments. Two different GCP
projects issuing an identical prompt collided to ONE cache key → User B could be
served User A's response from a project they cannot access (cross-tenant leak).
The fix folds project+region into model_id (key_builder whitelist-filters
raw_params, so model_id is the field that is actually hashed).
"""
import json
import tempfile
from pathlib import Path

from superlocalmemory.optimize.cache.manager import CacheManager
from superlocalmemory.optimize.proxy.lifecycle import ProxyRequest
from superlocalmemory.optimize.storage.db import CacheDB


def _cm():
    db = CacheDB(db_path=Path(tempfile.mkdtemp()) / "llmcache.db")
    return CacheManager(db=db)


def _req(project, location="us-central1", prompt="hi"):
    body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
    return ProxyRequest(
        provider="vertex",
        path=f"/v1/projects/{project}/locations/{location}/publishers/google/models/gemini-1.5-pro:generateContent",
        method="POST", headers={}, body=body,
        body_bytes=json.dumps(body).encode(), request_id="r1",
        has_tools=False, stream=False,
    )


def test_different_project_same_prompt_different_key():
    cm = _cm()
    assert cm.build_key(_req("internal-prod-12345"), "default") != \
        cm.build_key(_req("external-dev-99999"), "default")


def test_different_region_different_key():
    cm = _cm()
    assert cm.build_key(_req("p", location="us-central1"), "default") != \
        cm.build_key(_req("p", location="europe-west4"), "default")


def test_same_project_prompt_region_same_key_cache_still_works():
    cm = _cm()
    assert cm.build_key(_req("p"), "default") == cm.build_key(_req("p"), "default")


def test_different_prompt_different_key_no_regression():
    cm = _cm()
    assert cm.build_key(_req("p", prompt="a"), "default") != \
        cm.build_key(_req("p", prompt="b"), "default")
