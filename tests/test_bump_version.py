"""Release-version updater must not hide a mixed generated-source tree."""

from __future__ import annotations

from scripts import bump_version


def test_read_glob_reports_every_distinct_version(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(bump_version, "_ROOT", tmp_path)
    skills = tmp_path / "plugin-src" / "skills"
    skills.mkdir(parents=True)
    (skills / "a.md").write_text("SuperLocalMemory v4.1.10\n", encoding="utf-8")
    (skills / "b.md").write_text("SuperLocalMemory v4.1.9\n", encoding="utf-8")

    found = bump_version._read_glob(
        "plugin-src/**/*.md",
        r"SuperLocalMemory v([0-9]+\.[0-9]+\.[0-9]+)",
    )

    assert found == "<mixed: 4.1.10, 4.1.9>"


def _tree_version() -> str:
    import re
    text = (bump_version._ROOT / "pyproject.toml").read_text(encoding="utf-8")
    return re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE).group(1)


def test_check_is_green_on_the_tree_at_its_own_version(capsys) -> None:
    """Every source the script knows about must exist in the tree.

    A source whose pattern no longer matches (README stopped carrying the
    version) makes --check red forever and makes a real bump fail half way.
    """
    rc = bump_version.main([_tree_version(), "--check"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "<not found>" not in out


def test_a_missing_pattern_writes_nothing(tmp_path, monkeypatch, capsys) -> None:
    """All-or-nothing: one unmatched source must leave every file untouched."""
    monkeypatch.setattr(bump_version, "_ROOT", tmp_path)
    good = tmp_path / "good.txt"
    bad = tmp_path / "bad.txt"
    good.write_text('version = "1.0.0"\n', encoding="utf-8")
    bad.write_text("no version stamp here\n", encoding="utf-8")

    def plan(version):
        return [
            ("good", lambda: bump_version._read("good.txt", r'version = "([^"]+)"'),
             lambda: bump_version._sub_regex("good.txt", r'version = "([^"]+)"',
                                             'version = "{v}"', version)),
            ("bad", lambda: bump_version._read("bad.txt", r'stamp v([0-9.]+)'),
             lambda: bump_version._sub_regex("bad.txt", r'stamp v([0-9.]+)',
                                             "stamp v{v}", version)),
        ]

    monkeypatch.setattr(bump_version, "_plan", plan)
    rc = bump_version.main(["2.0.0"])
    out = capsys.readouterr().out

    assert rc != 0
    assert good.read_text(encoding="utf-8") == 'version = "1.0.0"\n'
    assert bad.read_text(encoding="utf-8") == "no version stamp here\n"
    assert "nothing was written" in out


def test_a_successful_bump_writes_every_source(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(bump_version, "_ROOT", tmp_path)
    (tmp_path / "a.json").write_text('{"version": "1.0.0"}\n', encoding="utf-8")
    (tmp_path / "b.txt").write_text("pkg==1.0.0\n", encoding="utf-8")

    def plan(version):
        return [
            ("a", lambda: bump_version._read_json("a.json"),
             lambda: bump_version._sub_json("a.json", version)),
            ("b", lambda: bump_version._read("b.txt", r"pkg==([^\s]+)"),
             lambda: bump_version._sub_regex("b.txt", r"pkg==([^\s]+)",
                                             "pkg=={v}", version)),
        ]

    monkeypatch.setattr(bump_version, "_plan", plan)
    assert bump_version.main(["2.0.0"]) == 0
    assert '"2.0.0"' in (tmp_path / "a.json").read_text(encoding="utf-8")
    assert (tmp_path / "b.txt").read_text(encoding="utf-8") == "pkg==2.0.0\n"


def test_date_released_may_be_one_day_ahead_of_utc(monkeypatch) -> None:
    """A release cut after midnight IST is still the previous day in UTC.

    4.2.31 was bumped at 01:00 IST on 20 Nov; the GitHub runner, on UTC, still
    read 19 Nov, called the date "in the future" and failed --check. The rule
    must match tests/release/test_v37_rc_candidate_contract.py: up to one day
    ahead of UTC today is a real release date, two days ahead is a typo.
    """
    from datetime import date

    monkeypatch.setattr(bump_version, "_utc_today", lambda: date(2026, 11, 19))
    assert bump_version._date_ok("2026-11-19")
    assert bump_version._date_ok("2026-11-20")
    assert not bump_version._date_ok("2026-11-21")
    assert not bump_version._date_ok("not-a-date")
