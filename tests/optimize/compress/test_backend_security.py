"""Known affected NLTK installations must not activate the optional backend."""

from importlib.metadata import PackageNotFoundError
from unittest.mock import MagicMock, patch

import pytest

from superlocalmemory.optimize.compress.prose_llmlingua import LLMLinguaCompressor


@pytest.mark.parametrize(
    "installed",
    [
        "3.9.4",
        "3.10.0",
        "3.10.3",
        "3.10.4rc1",
        "3.10.3.0",
        "3.10.3.1",
        "3.10.3.post1",
        "3.10.3+local",
        "1!3.10.3",
        "3.10.4",
        "unknown",
    ],
)
def test_affected_or_unknown_nltk_is_blocked_before_backend_import(installed: str) -> None:
    backend = MagicMock()
    with (
        patch(
            "superlocalmemory.optimize.compress.prose_llmlingua.version",
            return_value=installed,
            create=True,
        ),
        patch.dict("sys.modules", {"llmlingua": backend}),
    ):
        with pytest.raises(ImportError):
            LLMLinguaCompressor()
    backend.PromptCompressor.assert_not_called()


def test_missing_nltk_keeps_backend_unavailable() -> None:
    with (
        patch(
            "superlocalmemory.optimize.compress.prose_llmlingua.version",
            side_effect=PackageNotFoundError("nltk"),
            create=True,
        ),
        patch.dict("sys.modules", {"llmlingua": MagicMock()}),
    ):
        with pytest.raises(ImportError):
            LLMLinguaCompressor()


def test_readiness_does_not_recommend_installing_restricted_backend() -> None:
    from superlocalmemory.core.component_registry import probe_llmlingua

    result = probe_llmlingua()
    assert result.status != "ok"
    assert result.auto_fixable is False
    assert not result.fix_cmd
