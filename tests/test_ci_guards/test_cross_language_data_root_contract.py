"""Shipped non-Python entry points must implement canonical alias precedence."""

from pathlib import Path


_ROOT = Path(__file__).resolve().parents[2]


def _text(relative: str) -> str:
    return (_ROOT / relative).read_text(encoding="utf-8")


def _assert_ordered_aliases(relative: str) -> None:
    text = _text(relative)
    positions = [text.find(name) for name in ("SLM_DATA_DIR", "SL_MEMORY_PATH", "SLM_HOME")]
    assert all(position >= 0 for position in positions), (relative, positions)
    assert positions == sorted(positions), (relative, positions)


def test_node_and_ide_hooks_share_alias_precedence() -> None:
    for relative in (
        "scripts/postinstall-interactive.js",
        "ide/hooks/context-hook.js",
        "ide/hooks/post-recall-hook.js",
    ):
        _assert_ordered_aliases(relative)


def test_npm_runtime_delegates_data_root_resolution_without_writing_it() -> None:
    for relative in ("bin/slm-npm", "scripts/postinstall.js"):
        text = _text(relative)
        assert "fs.mkdirSync" not in text
        assert "SLM_DATA_DIR" not in text or relative == "bin/slm-npm"


def test_generated_and_shell_hooks_share_alias_precedence() -> None:
    for relative in (
        "scripts/build_entry.py",
        "ide/hooks/tool-event-hook.sh",
    ):
        _assert_ordered_aliases(relative)


def test_framework_integrations_do_not_treat_data_root_as_install_root() -> None:
    for relative in (
        "ide/integrations/langchain/langchain_superlocalmemory/chat_message_history.py",
        "ide/integrations/llamaindex/llama_index/storage/chat_store/superlocalmemory/base.py",
        # v3.8.0 adapters — same data-root contract.
        "ide/integrations/crewai/crewai_superlocalmemory/backend.py",
        "ide/integrations/autogen/autogen_superlocalmemory/memory.py",
        "ide/integrations/google-adk/google_adk_superlocalmemory/memory_service.py",
        "ide/integrations/openai-agents/openai_agents_superlocalmemory/session.py",
    ):
        text = _text(relative)
        assert "SLM_INSTALL_DIR" in text
        assert "SLM_DATA_DIR" in text
        assert text.find("SLM_DATA_DIR") < text.find("SL_MEMORY_PATH") < text.find("SLM_HOME")
