import pytest

from superlocalmemory.cache import CacheKey

SHA = "a" * 64


@pytest.fixture(autouse=True)
def _fresh_cache_state(monkeypatch):
    from superlocalmemory.cache import factory

    monkeypatch.delenv("SLM_DERIVE_CACHE_MAX_MB", raising=False)
    monkeypatch.delenv("SLM_CACHE_BACKEND", raising=False)
    factory._reset_for_tests()
    yield
    factory._reset_for_tests()


def make_key(n=0, *, model="", deriver="pdf.render") -> CacheKey:
    return CacheKey(f"{n:064x}", deriver, "1", model, "")
