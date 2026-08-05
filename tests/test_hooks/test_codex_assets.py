from superlocalmemory.hooks import codex_assets


EXPECTED_SKILLS = {
    "slm-cache",
    "slm-compress",
    "slm-governance",
    "slm-graph",
    "slm-loop",
    "slm-mesh",
    "slm-profile",
    "slm-recall",
    "slm-remember",
    "slm-scope",
    "slm-session",
    "slm-status",
}


def test_install_and_remove_assets_are_scoped_to_slm_paths(tmp_path):
    result = codex_assets.install_assets(home=tmp_path)

    assert result["success"] is True
    assert set(result["skills"]) == EXPECTED_SKILLS
    assert {
        path.parent.name
        for path in (tmp_path / ".agents" / "skills").glob("*/SKILL.md")
    } == EXPECTED_SKILLS
    assert (tmp_path / ".agents" / "skills" / "slm-recall" / "SKILL.md").exists()
    assert (tmp_path / ".codex" / "agents" / "slm-memory-advisor.toml").exists()
    assert codex_assets.status_assets(home=tmp_path)["installed"] is True

    other = tmp_path / ".codex" / "agents" / "user-agent.toml"
    other.write_text("name = 'user-agent'\n")
    removed = codex_assets.remove_assets(home=tmp_path)

    assert removed["success"] is True
    assert other.exists()
    assert codex_assets.status_assets(home=tmp_path)["installed"] is False
