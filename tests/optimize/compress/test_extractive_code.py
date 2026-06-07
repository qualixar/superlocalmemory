"""Tests for extractive_code.py — CodeCompressor."""
from __future__ import annotations

from superlocalmemory.optimize.compress.extractive_code import (
    CodeCompressor,
    _apply_spans,
)
from superlocalmemory.optimize.compress.router import _detect_language


_PYTHON_SAMPLE = """import os
import sys
from pathlib import Path

def process_data(input_path: str, output_path: str) -> dict:
    \"\"\"Process data files from input to output.\"\"\"
    data = Path(input_path).read_text()
    result = {}
    for line in data.split("\\n"):
        if line.startswith("#"):
            continue
        key, _, value = line.partition("=")
        result[key.strip()] = value.strip()
    return result

def validate_input(data: str) -> bool:
    \"\"\"Validate the input format.\"\"\"
    if not data:
        raise ValueError("Empty data")
    if "=" not in data:
        return False
    return True

class DataHandler:
    \"\"\"Handles data processing pipeline.\"\"\"

    def __init__(self, config: dict):
        self.config = config
        self._cache = {}

    def run(self, path: str) -> dict:
        \"\"\"Run the full pipeline.\"\"\"
        raw = Path(path).read_text()
        if not validate_input(raw):
            raise ValueError("Invalid input")
        return process_data(path, self.config.get("output", "/dev/null"))
"""


def test_code_signatures_preserved_python() -> None:
    compressor = CodeCompressor()
    out = compressor.compress(_PYTHON_SAMPLE, language="python")
    assert "def process_data(" in out
    assert "def validate_input(" in out


def test_code_imports_preserved_python() -> None:
    compressor = CodeCompressor()
    code = "import os\nimport sys\nfrom pathlib import Path\n\ndef foo():\n    " + "pass\n" * 10
    out = compressor.compress(code, language="python")
    assert "import os" in out
    assert "import sys" in out
    assert "from pathlib import Path" in out


def test_code_body_stubbed() -> None:
    compressor = CodeCompressor()
    code = "def big_function(x):\n" + "    result = x * 2\n" * 10 + "    return result\n"
    out = compressor.compress(code, language="python")
    assert "# [slm:" in out
    assert "result = x * 2" not in out


def test_code_compressor_regex_path_same_language() -> None:
    import unittest.mock as mock
    with mock.patch(
        "superlocalmemory.optimize.compress.extractive_code._tree_sitter_available",
        return_value=False,
    ):
        compressor = CodeCompressor()
        code = "def foo(x):\n    " + "pass\n" * 10
        out = compressor.compress(code, language="python")
        assert isinstance(out, str) and len(out) > 0


def test_code_compressor_supported_languages() -> None:
    compressor = CodeCompressor()
    _LANG_SAMPLES = {
        "python": "def foo():\n    pass\n" * 5,
        "javascript": "function foo() {\n    return 1;\n}\n" * 5,
        "go": "func foo() {\n    return\n}\n" * 5,
        "rust": "fn foo() {\n    let x = 1;\n}\n" * 5,
        "java": "public void foo() {\n    int x = 1;\n}\n" * 5,
    }
    for lang, sample in _LANG_SAMPLES.items():
        out = compressor.compress(sample, language=lang)
        assert isinstance(out, str) and len(out) > 0


def test_code_compressor_never_raises() -> None:
    compressor = CodeCompressor()
    out = compressor.compress("", language="python")
    assert isinstance(out, str)
    out = compressor.compress("not code at all", language="go")
    assert isinstance(out, str)


def test_code_compressor_unknown_language_passthrough() -> None:
    compressor = CodeCompressor()
    code = "some code here"
    out = compressor.compress(code, language="haskell")
    assert out == code


def test_apply_spans_source_order_preserved() -> None:
    """B-04: Uncovered lines must appear in original position, not appended."""
    from superlocalmemory.optimize.compress.extractive_code import CodeSpan

    lines = [
        "import os",           # 0: structural
        "",                     # 1: blank (uncovered)
        "def foo():",           # 2: signature
        "    x = 1",            # 3: body
        "    y = 2",            # 4: body
        "    z = 3",            # 5: body
        "    return x + y + z", # 6: body
        "",                     # 7: blank (uncovered)
        "def bar():",           # 8: signature
    ]
    spans = [
        CodeSpan(0, 0, "import", True),
        CodeSpan(2, 2, "signature", True),
        CodeSpan(3, 6, "body", False),
        CodeSpan(8, 8, "signature", True),
    ]
    out = _apply_spans(lines, spans, "test-uuid", "python")
    out_lines = out.split("\n")

    assert "import os" in out_lines[0]
    assert "" in out_lines[1]
    assert "def foo():" in out_lines[2]
    assert "# [slm:" in out_lines[3]  # body stub
    assert "" in out_lines  # blank line at original position
    assert "def bar():" in out_lines  # second signature present somewhere


def test_detect_language_python() -> None:
    code = (
        "import os\n"
        "import sys\n"
        "from pathlib import Path\n"
        "\n"
        "def foo(x: int) -> int:\n"
        "    return x * 2\n"
    )
    result = _detect_language(code)
    assert result == "python"


def test_detect_language_false_positive_prose() -> None:
    """B-11: Prose discussing Python must NOT be detected as python."""
    prose = (
        "In Python, you can define functions with def. Classes are defined with class.\n"
        "You can import modules using import statements. This text discusses Python\n"
        "but it is NOT source code. Ordinary prose should not trigger the detector.\n"
        "Only actual source files with structural patterns should match.\n"
    )
    result = _detect_language(prose)
    assert result is None, f"Prose was incorrectly detected as {result}"


def test_detect_language_short_snippet_passthrough() -> None:
    """4-line function with only def + return → None from detect_language."""
    short = "def foo(x):\n    return x * 2\n"
    result = _detect_language(short)
    assert result is None


def test_code_stub_comment_syntax_by_language() -> None:
    """B-07: Python uses #, other languages use //. Tests stub generation."""
    compressor = CodeCompressor()
    # Python (should use #)
    py_code = "def test():\n" + "    x = 1\n" * 5 + "    return x\n"
    out = compressor.compress(py_code, language="python")
    assert "# [slm:" in out

    # Go (uses //)
    go_code = (
        'package main\n'
        'import "fmt"\n'
        '\n'
        'func main() {\n'
        + '    x := 1\n' * 5
        + '    fmt.Println(x)\n'
        + '}\n'
    )
    out_go = compressor.compress(go_code, language="go")
    assert "// [slm:" in out_go


def test_code_ccr_id_embedded_in_stub() -> None:
    """When ccr_id is provided, it appears in the stub comment."""
    compressor = CodeCompressor()
    code = "def foo():\n" + "    pass\n" * 6 + "    return 1\n"
    out = compressor.compress(code, language="python", ccr_id="test-ccr-id-1234")
    assert "test-ccr-id-1234" in out
