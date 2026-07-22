# Security regression guards for 3.7.9 findings H10-H14 (audit-01 H-02..H-06).
import inspect
import sys

from superlocalmemory.cli import setup_wizard
from superlocalmemory.core.embedding_worker import _trusts_remote_code
from superlocalmemory.server import write_identity


# --- H-02: trust_remote_code allowlist -------------------------------------

def test_trust_remote_code_only_for_pinned_models():
    assert _trusts_remote_code("nomic-ai/nomic-embed-text-v1.5") is True
    assert _trusts_remote_code("nomic-ai/nomic-embed-text-v1") is True


def test_trust_remote_code_denied_for_arbitrary_models():
    for name in ("evil/backdoor", "", "../../etc/passwd", "openai/whatever",
                 "nomic-ai/nomic-embed-text-v1.5-EVIL"):
        assert _trusts_remote_code(name) is False, name


# --- H-03: setup wizard passes model name as argv, never interpolated ------

def test_download_functions_pass_model_name_as_argv():
    for fn in (setup_wizard._download_model,
               setup_wizard._download_reranker,
               setup_wizard._download_compressor):
        src = inspect.getsource(fn)
        assert "sys.argv[1]" in src, f"{fn.__name__} must read model via sys.argv"
        assert "script, model_name" in src, (
            f"{fn.__name__} must pass model_name as a subprocess arg"
        )
        # the model name must NOT be interpolated into the executed source
        assert "'{model_name}'" not in src, (
            f"{fn.__name__} still interpolates model_name into executed code"
        )


# --- H-04: test-client auth bypass cannot activate in a real daemon --------

def test_test_isolation_guard_requires_pytest_present():
    # The guard is a conjunction: SLM_TEST_ISOLATION=1 AND pytest imported.
    # A production daemon process never imports pytest, so the bypass can never
    # activate there regardless of the env var.
    assert "pytest" in sys.modules
    assert isinstance(write_identity._TEST_ISOLATION_ALLOWED, bool)
    src = inspect.getsource(write_identity)
    assert '"pytest" in sys.modules' in src
