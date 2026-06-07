"""Tests for module-level helper functions in router.py."""
from __future__ import annotations

import json
import re

from superlocalmemory.optimize.compress.router import (
    _detect_language,
    _msg_has_tool_result,
    _set_tool_result_text,
    _token_estimate,
    _token_estimate_structured,
    _tool_result_text,
    CompressTextResult,
)


def test_token_estimate_prose() -> None:
    assert _token_estimate("hello world") == 2
    assert _token_estimate("") == 0


def test_token_estimate_structured() -> None:
    assert _token_estimate_structured("abcd" * 10) == 10  # 40 chars / 4
    assert _token_estimate_structured("") == 0


def test_msg_has_tool_result_true() -> None:
    msg = {"content": [{"type": "tool_result", "text": "x"}]}
    assert _msg_has_tool_result(msg) is True

    msg2 = {"content": [{"tool_use_id": "toolu_123", "text": "x"}]}
    assert _msg_has_tool_result(msg2) is True


def test_msg_has_tool_result_false() -> None:
    msg = {"content": [{"type": "text", "text": "hello"}]}
    assert _msg_has_tool_result(msg) is False

    msg2 = {"content": "plain string"}
    assert _msg_has_tool_result(msg2) is False


def test_tool_result_text_string() -> None:
    block = {"content": "plain text"}
    assert _tool_result_text(block) == "plain text"


def test_tool_result_text_list() -> None:
    block = {"content": [{"type": "text", "text": "hello"}, {"type": "text", "text": "world"}]}
    assert _tool_result_text(block) == "hello\nworld"


def test_tool_result_text_mixed() -> None:
    block = {"content": [{"type": "image_url", "url": "http://x"}, {"type": "text", "text": "hi"}]}
    assert _tool_result_text(block) == "hi"


def test_set_tool_result_text_string_content() -> None:
    block = {"content": "original", "other": "keep"}
    result = _set_tool_result_text(block, "new text")
    assert result["content"] == "new text"
    assert result["other"] == "keep"
    assert result is not block


def test_set_tool_result_text_list_content() -> None:
    block = {"content": [{"type": "text", "text": "original"}]}
    result = _set_tool_result_text(block, "new text")
    assert result is not block
    inner = result["content"]
    assert isinstance(inner, list)
    assert inner[0]["text"] == "new text"


def test_set_tool_result_text_list_with_multiple_blocks() -> None:
    block = {
        "content": [
            {"type": "image", "url": "x"},
            {"type": "text", "text": "old"},
            {"type": "text", "text": "keep_this"},
        ]
    }
    result = _set_tool_result_text(block, "new")
    texts_only = [b["text"] for b in result["content"] if b.get("type") == "text"]
    assert "new" in texts_only


def test_set_tool_result_text_unmodified() -> None:
    """Non-dict, non-str, non-list unchanged."""
    block = {"content": 42}
    result = _set_tool_result_text(block, "new")
    assert result == block


def test_detect_language_shebang_python() -> None:
    code = (
        "#!/usr/bin/env python\n"
        "import os\n"
        "import sys\n"
        "\n"
        "def main():\n"
        "    print(os.getcwd())\n"
        "    print(sys.argv)\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    main()\n"
    )
    result = _detect_language(code)
    assert result == "python"


def test_detect_language_shebang_node() -> None:
    code = (
        "#!/usr/bin/env node\n"
        "const fs = require('fs');\n"
        "const path = require('path');\n"
        "\n"
        "function main() {\n"
        "    console.log(fs.readdirSync('.'));\n"
        "}\n"
        "\n"
        "module.exports = { main };\n"
    )
    result = _detect_language(code)
    assert result == "javascript"


def test_detect_language_fenced_code_block() -> None:
    code = (
        "```python\n"
        "import os\n"
        "import sys\n"
        "\n"
        "def process(path):\n"
        "    return os.path.join(path, 'out')\n"
        "```\n"
    )
    result = _detect_language(code)
    assert result == "python"


def test_detect_language_fenced_js() -> None:
    code = (
        "```javascript\n"
        "import fs from 'fs';\n"
        "\n"
        "export function readFile(path) {\n"
        "    return fs.readFileSync(path, 'utf-8');\n"
        "}\n"
        "```\n"
    )
    result = _detect_language(code)
    assert result == "javascript"


def test_detect_language_fenced_ts() -> None:
    code = (
        "```typescript\n"
        "import { readFileSync } from 'fs';\n"
        "\n"
        "export function parse(path: string): string {\n"
        "    return readFileSync(path, 'utf-8');\n"
        "}\n"
        "```\n"
    )
    result = _detect_language(code)
    assert result == "javascript"


def test_detect_language_fenced_cpp() -> None:
    code = (
        "```c++\n"
        "#include <iostream>\n"
        "#include <string>\n"
        "\n"
        "int main() {\n"
        "    std::cout << \"hello\" << std::endl;\n"
        "    return 0;\n"
        "}\n"
        "```\n"
    )
    result = _detect_language(code)
    assert result == "cpp"


def test_detect_language_go() -> None:
    code = (
        "package main\n"
        'import "fmt"\n'
        'import "os"\n'
        "\n"
        "func main() {\n"
        "    fmt.Println(os.Args[0])\n"
        "}\n"
        "\n"
        "func helper(x int) int {\n"
        "    return x * 2\n"
        "}\n"
    )
    result = _detect_language(code)
    assert result == "go"


def test_detect_language_rust() -> None:
    code = (
        "use std::collections::HashMap;\n"
        "use std::fs;\n"
        "\n"
        "pub fn read_config(path: &str) -> HashMap<String, String> {\n"
        "    let data = fs::read_to_string(path).unwrap();\n"
        "    HashMap::new()\n"
        "}\n"
        "\n"
        "pub struct Config {\n"
        "    pub path: String,\n"
        "}\n"
        "\n"
        "impl Config {\n"
        "    pub fn new(p: &str) -> Self {\n"
        "        Config { path: p.to_string() }\n"
        "    }\n"
        "}\n"
    )
    result = _detect_language(code)
    assert result == "rust"


def test_detect_language_java() -> None:
    code = (
        "import java.util.List;\n"
        "import java.util.ArrayList;\n"
        "\n"
        "public class Processor {\n"
        "    private List<String> items;\n"
        "\n"
        "    public Processor() {\n"
        "        this.items = new ArrayList<>();\n"
        "    }\n"
        "\n"
        "    public void process(String input) {\n"
        "        items.add(input);\n"
        "    }\n"
        "}\n"
    )
    result = _detect_language(code)
    assert result == "java"


def test_detect_language_edge_cases() -> None:
    assert _detect_language("") is None
    assert _detect_language(None.__class__.__name__) is None
    assert _detect_language("a" * 30) is None


def test_compress_text_result_dataclass() -> None:
    r = CompressTextResult(
        compressed_text="hello",
        strategy="none",
        tokens_before=100,
        tokens_after=100,
    )
    assert r.compressed_text == "hello"
    assert r.strategy == "none"
    assert r.tokens_before == 100
    assert r.tokens_after == 100
