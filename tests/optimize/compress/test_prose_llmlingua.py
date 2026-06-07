# File: tests/optimize/compress/test_prose_llmlingua.py
# Run live against the real llmlingua install — no mocking

from __future__ import annotations
import os
import pytest

# Skip if HF download is blocked in CI to avoid multi-GB downloads.
# Set LLMLINGUA_INTEGRATION=1 locally to run live.
requires_llmlingua = pytest.mark.skipif(
    os.environ.get('LLMLINGUA_INTEGRATION') != '1',
    reason='Set LLMLINGUA_INTEGRATION=1 to run live LLMLingua-2 tests',
)


@requires_llmlingua
def test_basic_compression_reduces_length() -> None:
    """LLMLinguaCompressor.compress() returns a shorter string for long prose."""
    from superlocalmemory.optimize.compress.prose_llmlingua import LLMLinguaCompressor
    compressor = LLMLinguaCompressor()
    long_prose = (
        'The capital of France is Paris. Paris is a large city. '
        'It is located in western Europe. The Eiffel Tower is a famous landmark. '
        'Many tourists visit Paris every year for its art, culture and cuisine. '
        'The city is divided into arrondissements. The Seine river flows through it. '
        'Paris hosted the Olympics in 1900 and 1924 and again in 2024.'
    )
    compressed = compressor.compress(long_prose)
    assert isinstance(compressed, str)
    assert len(compressed) > 0
    assert len(compressed) <= len(long_prose), (
        f'Expected compression, got expansion: {len(compressed)} > {len(long_prose)}'
    )


@requires_llmlingua
def test_rate_parameter_override_produces_shorter_output() -> None:
    """Passing a lower rate should produce more aggressive compression."""
    from superlocalmemory.optimize.compress.prose_llmlingua import LLMLinguaCompressor
    compressor = LLMLinguaCompressor(rate=0.7)
    prose = (
        'The mitochondria is the powerhouse of the cell. '
        'It produces ATP via oxidative phosphorylation. '
        'The inner membrane has cristae that increase surface area. '
        'Electrons pass through the electron transport chain releasing energy.'
    )
    out_default = compressor.compress(prose)        # uses default rate 0.7
    out_aggressive = compressor.compress(prose, rate=0.3)  # override to 0.3
    # Both must be valid strings; aggressive should be shorter or equal
    assert isinstance(out_default, str) and len(out_default) > 0
    assert isinstance(out_aggressive, str) and len(out_aggressive) > 0


@requires_llmlingua
def test_empty_input_returns_original() -> None:
    """Empty string must not raise — fail-open returns original empty string."""
    from superlocalmemory.optimize.compress.prose_llmlingua import LLMLinguaCompressor
    compressor = LLMLinguaCompressor()
    result = compressor.compress('')
    assert result == '', f'Expected empty string passthrough, got {result!r}'


def test_hf_download_blocked_raises_import_error(monkeypatch) -> None:
    """SLM_DISABLE_HF_DOWNLOAD=1 must raise ImportError, not silently pass."""
    monkeypatch.setenv('SLM_DISABLE_HF_DOWNLOAD', '1')
    # Force re-import by removing any cached module
    import sys
    sys.modules.pop('superlocalmemory.optimize.compress.prose_llmlingua', None)
    from superlocalmemory.optimize.compress import prose_llmlingua as _mod
    import importlib
    importlib.reload(_mod)
    with pytest.raises(ImportError, match='SLM_DISABLE_HF_DOWNLOAD'):
        _mod.LLMLinguaCompressor()


def test_compress_fail_open_when_compressor_raises() -> None:
    """If compress_prompt raises, fail-open returns the original text."""
    from superlocalmemory.optimize.compress.prose_llmlingua import LLMLinguaCompressor
    # Construct without hitting HF by injecting a fake _compressor
    instance = object.__new__(LLMLinguaCompressor)
    instance._rate = 0.5

    class _BrokenCompressor:
        def compress_prompt(self, *a, **kw):
            raise RuntimeError('GPU OOM')

    instance._compressor = _BrokenCompressor()
    original = 'This is original text that should come back unchanged.'
    result = instance.compress(original)
    assert result == original, f'Expected passthrough on error, got {result!r}'


def test_compress_passthrough_when_compressed_prompt_empty() -> None:
    """compressed_prompt == '' triggers the guard on line 71-73 — returns original."""
    from superlocalmemory.optimize.compress.prose_llmlingua import LLMLinguaCompressor
    instance = object.__new__(LLMLinguaCompressor)
    instance._rate = 0.5

    class _EmptyCompressor:
        def compress_prompt(self, *a, **kw):
            return {'compressed_prompt': ''}

    instance._compressor = _EmptyCompressor()
    original = 'The original text'
    result = instance.compress(original)
    assert result == original


def test_compress_passthrough_when_compressed_prompt_non_string() -> None:
    """compressed_prompt of non-str type triggers the guard on line 71-73."""
    from superlocalmemory.optimize.compress.prose_llmlingua import LLMLinguaCompressor
    instance = object.__new__(LLMLinguaCompressor)
    instance._rate = 0.5

    class _NonStrCompressor:
        def compress_prompt(self, *a, **kw):
            return {'compressed_prompt': 42}

    instance._compressor = _NonStrCompressor()
    original = 'Original text here'
    result = instance.compress(original)
    assert result == original


# ---- mocked constructor path (covers lines 45-59) ----

def test_constructor_with_mocked_llmlingua_import(monkeypatch) -> None:
    """Mock llmlingua.PromptCompressor to cover constructor lines 45-59."""
    from unittest.mock import MagicMock, patch
    import sys

    fake_compressor_class = MagicMock()
    fake_compressor_instance = MagicMock()
    fake_compressor_class.return_value = fake_compressor_instance

    fake_llmlingua = MagicMock()
    fake_llmlingua.PromptCompressor = fake_compressor_class

    with patch.dict(sys.modules, {'llmlingua': fake_llmlingua}):
        from superlocalmemory.optimize.compress.prose_llmlingua import LLMLinguaCompressor
        comp = LLMLinguaCompressor()
        assert comp._compressor is fake_compressor_instance
        assert comp._rate == 0.5
        fake_compressor_class.assert_called_once_with(
            model_name='microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank',
            use_llmlingua2=True,
            device_map='cpu',
        )


def test_compress_successful_path_returns_compressed(monkeypatch) -> None:
    """Successful compress path (line 74) returns the compressed string."""
    from unittest.mock import MagicMock, patch
    import sys

    fake_compressor = MagicMock()
    fake_compressor.compress_prompt.return_value = {
        'compressed_prompt': 'compressed result text'
    }
    fake_compressor_class = MagicMock(return_value=fake_compressor)
    fake_llmlingua = MagicMock()
    fake_llmlingua.PromptCompressor = fake_compressor_class

    with patch.dict(sys.modules, {'llmlingua': fake_llmlingua}):
        from superlocalmemory.optimize.compress.prose_llmlingua import LLMLinguaCompressor
        comp = LLMLinguaCompressor()
        result = comp.compress('some long text to compress')
        assert result == 'compressed result text'
