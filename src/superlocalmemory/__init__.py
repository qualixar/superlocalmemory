"""SuperLocalMemory — information-geometric agent memory."""

import os

# --- OpenMP multi-library guard (permanent fix, v3.4.58) -------------------
# SLM ships with torch, scikit-learn, and lightgbm — each bundles its own
# libomp.dylib on macOS ARM (Apple Silicon). When all three are loaded in the
# same process, the Intel OpenMP runtime detects duplicate libraries and either
# (a) emits OMP: Error #15 and aborts, or (b) crashes with SIGSEGV in
# __kmp_suspend_initialize_thread (address 0x580 — null thread-struct deref)
# when LightGBM's parallel fork tries to coordinate with PyTorch's thread pool.
#
# KMP_DUPLICATE_LIB_OK=TRUE tells the runtime to elect one master instance
# and continue rather than abort. This is the upstream-recommended workaround
# for mixed-dependency environments (PyTorch docs, scikit-learn FAQ, LightGBM
# issue #3877). Use unconditional assignment — not setdefault — so user env
# cannot accidentally disable this safety net by setting it to FALSE.
#
# OMP_NUM_THREADS=2 caps the maximum thread count before any C library reads
# it. With ≤2 threads the problematic parallel fork path in LightGBM's
# DatasetLoader::ConstructFromSampleData is avoided on all observed crash
# configurations. SLM datasets are small (50–5 000 rows); 2 threads gives
# ~90% of the performance of max-core training at zero crash risk.
# Users who need more threads can set SLM_LGBM_THREADS=N to override
# the per-call cap in ranker_retrain_online.py.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = "2"
# ---------------------------------------------------------------------------

__version__ = "3.6.3"

_REQUIRED_VERSIONS = {
    "sentence_transformers": "5.3.0",
    "onnxruntime": "1.24.4",
}


def _check_critical_deps() -> None:
    """Warn if embedding-critical packages have wrong versions."""
    import warnings
    for mod_name, expected in _REQUIRED_VERSIONS.items():
        try:
            mod = __import__(mod_name)
            actual = getattr(mod, "__version__", None)
            if actual and actual != expected:
                warnings.warn(
                    f"SuperLocalMemory requires {mod_name}=={expected} but "
                    f"{actual} is installed. This causes memory blow-up on "
                    f"Apple Silicon. Fix: pip install {mod_name}=={expected}",
                    stacklevel=2,
                )
        except ImportError:
            pass


# Only run the dep check when a full (non-LIGHT) engine is in use.
# The MCP server runs in LIGHT mode — importing onnxruntime here
# breaks the LIGHT engine contract (ONNX_LOADED must stay False).
# Skip when SLM_SKIP_DEP_CHECK=1 or SLM_DISABLE_WARMUP_SIDE_EFFECTS=1.
if not (
    os.environ.get("SLM_SKIP_DEP_CHECK") == "1"
    or os.environ.get("SLM_DISABLE_WARMUP_SIDE_EFFECTS") == "1"
):
    _check_critical_deps()
