import tomllib
from pathlib import Path

from superlocalmemory.hooks import codex_assets


EXPECTED_SKILLS = {
    "slm-cache",
    "slm-compress",
    "slm-governance",
    "slm-graph",
    "slm-loop",
    "slm-media",
    "slm-mesh",
    "slm-profile",
    "slm-recall",
    "slm-remember",
    "slm-scope",
    "slm-session",
    "slm-status",
    "slm-web-access",
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
    other.write_text("name = 'user-agent'\n", encoding="utf-8")
    removed = codex_assets.remove_assets(home=tmp_path)

    assert removed["success"] is True
    assert other.exists()
    assert codex_assets.status_assets(home=tmp_path)["installed"] is False


def _write_foreign_agent(tmp_path):
    """An agent file this installer did not write, in the richer shape the
    hand-maintained advisors use."""
    agents = tmp_path / ".codex" / "agents"
    agents.mkdir(parents=True, exist_ok=True)
    target = agents / "slm-memory-advisor.toml"
    original = (
        'name = "slm-memory-advisor"\n'
        'description = "hand-maintained advisor"\n'
        'developer_instructions = """\n# Role\nDo not lose this body.\n"""\n'
    )
    target.write_text(original, encoding="utf-8")
    return target, original


def test_existing_foreign_agent_file_is_preserved_not_clobbered(tmp_path):
    """Regression: install once replaced two hand-maintained ~4.9KB advisors
    with one-line stubs, with no backup and no prompt."""
    target, original = _write_foreign_agent(tmp_path)

    result = codex_assets.install_assets(home=tmp_path)

    assert result["success"] is True
    assert target.read_text(encoding="utf-8") == original
    assert str(target) in result["agents_preserved"]
    assert str(target) not in result["agents_written"]
    assert "slm-memory-advisor.toml" not in result["agents"]
    # The other agent has no competing owner, so it is still installed.
    assert (tmp_path / ".codex" / "agents" / "slm-optimize-advisor.toml").exists()


def test_dry_run_reports_what_it_would_preserve(tmp_path):
    target, original = _write_foreign_agent(tmp_path)

    result = codex_assets.install_assets(home=tmp_path, dry_run=True)

    assert result["dry_run"] is True
    assert str(target) in result["agents_preserved"]
    assert target.read_text(encoding="utf-8") == original


def test_success_payload_names_only_paths_actually_written(tmp_path):
    """Regression: the payload once advertised a refresh of assets it had not
    touched, which the CLI reported to the operator as success."""
    target, _ = _write_foreign_agent(tmp_path)

    result = codex_assets.install_assets(home=tmp_path)

    for path in result["skills_written"] + result["agents_written"]:
        assert Path(path).exists(), f"payload names a path that was not written: {path}"
    assert str(target) not in result["skills_written"] + result["agents_written"]
    assert len(result["skills_written"]) == len(codex_assets.SKILLS)
    assert set(result["agents_written"]).isdisjoint(result["agents_preserved"])


def test_skill_paths_resolving_outside_the_written_location_are_reported(tmp_path):
    """Codex may read ~/.codex/skills/<name> pointed at a checkout, in which
    case writing ~/.agents/skills refreshes nothing it will load."""
    checkout = tmp_path / "checkout" / "slm-recall"
    checkout.mkdir(parents=True)
    (checkout / "SKILL.md").write_text("# served from a checkout\n", encoding="utf-8")
    codex_skills = tmp_path / ".codex" / "skills"
    codex_skills.mkdir(parents=True)
    (codex_skills / "slm-recall").symlink_to(checkout, target_is_directory=True)

    result = codex_assets.install_assets(home=tmp_path)

    assert str(codex_skills / "slm-recall") in result["skills_read_elsewhere"]
    assert (checkout / "SKILL.md").read_text(encoding="utf-8") == "# served from a checkout\n"


def test_unmodified_agent_file_is_still_refreshed_on_reinstall(tmp_path):
    """Preserving foreign files must not freeze legitimate upgrades."""
    first = codex_assets.install_assets(home=tmp_path)
    assert len(first["agents_written"]) == len(codex_assets.AGENTS)

    second = codex_assets.install_assets(home=tmp_path)

    assert len(second["agents_written"]) == len(codex_assets.AGENTS)
    assert second["agents_preserved"] == []


def test_force_overwrites_but_keeps_a_backup(tmp_path):
    target, original = _write_foreign_agent(tmp_path)

    result = codex_assets.install_assets(home=tmp_path, force=True)

    assert str(target) in result["agents_written"]
    assert result["agents_preserved"] == []
    assert target.with_suffix(".toml.bak").read_text(encoding="utf-8") == original
    assert target.read_text(encoding="utf-8") != original


def test_generated_agent_toml_parses_and_carries_the_full_advisor_body(tmp_path):
    """The advisors ship their real decision rules, not a summary of them."""
    codex_assets.install_assets(home=tmp_path)

    for filename in codex_assets.AGENTS:
        raw = (tmp_path / ".codex" / "agents" / filename).read_text(encoding="utf-8")
        data = tomllib.loads(raw)
        assert set(data) >= {"name", "description", "instructions"}
        assert data["name"] == filename.removesuffix(".toml")
        assert "---" not in data["instructions"].split("\n")[0]
        assert len(data["instructions"]) > 800, f"{filename} looks like a stub"
