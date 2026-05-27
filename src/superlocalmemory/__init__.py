"""SuperLocalMemory — information-geometric agent memory."""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

__version__ = "3.4.55"

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
