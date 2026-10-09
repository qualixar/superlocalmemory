"""The docs say plainly where ChatGPT keeps the memory skill: only where the user
pastes it, never in a connected tool's skill library such as Composio's."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOSTS = ROOT / "docs" / "remote-access" / "hosts.md"
SETUP = ROOT / "docs" / "web-agents" / "setup-prompt.md"


def _chatgpt_section() -> str:
    match = re.search(r"## ChatGPT \(web, paid plan\)\n(.*?)(?=\n## )", HOSTS.read_text(), re.S)
    assert match, "ChatGPT section missing from hosts.md"
    return match.group(1)


def _chatgpt_row() -> str:
    rows = [r for r in SETUP.read_text().splitlines() if r.startswith("| ChatGPT |")]
    assert rows, "ChatGPT row missing from setup-prompt.md"
    return rows[0]


def _says_only_where_pasted(text: str) -> bool:
    return "paste it yourself" in text and "Composio" in text


def test_hosts_guide_says_chatgpt_keeps_the_skill_only_where_pasted():
    section = _chatgpt_section()
    assert "Three things to know" in section
    assert _says_only_where_pasted(section)


def test_setup_prompt_row_says_a_copy_saved_elsewhere_is_not_read():
    row = _chatgpt_row()
    assert _says_only_where_pasted(row)
    assert "does not read that copy" in row
