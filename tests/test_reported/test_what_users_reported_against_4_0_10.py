# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Four things people hit on 4.0.10 and told us about.

Each was reproduced against the shipped code before it was changed, and each
test below fails if the change is reverted. They are kept together because they
share one shape: in all four the software knew the right answer somewhere and
told the user a different one.

* An install instruction naming a command that does not exist.
* A documented package name that has never existed on npm.
* Two settings that saved to disk as nothing and loaded back as defaults.
* A health check reporting a failure without ever saying what failed.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import sqlite3

import pytest

import superlocalmemory


REPO = pathlib.Path(superlocalmemory.__file__).resolve().parents[2]


class TestTheInstallInstructionNamesRealCommands:
    """`slm connect claude-code` pointed at `slm plugin install`.

    There is no `slm plugin` subcommand and there never has been, so the one
    instruction the command printed ended at ``invalid choice: 'plugin'``. The
    working path was already in the setup wizard; nobody was told about it.
    """

    def test_it_does_not_name_a_subcommand_we_do_not_have(self) -> None:
        from superlocalmemory.hooks.portable_kit import CLAUDE_CODE_PLUGIN_POINTER

        assert "slm plugin" not in CLAUDE_CODE_PLUGIN_POINTER, (
            "the instruction names an slm subcommand that does not exist"
        )

    def test_it_names_the_commands_that_do_the_job(self) -> None:
        from superlocalmemory.hooks.portable_kit import CLAUDE_CODE_PLUGIN_POINTER

        assert "claude plugin install superlocalmemory@qualixar" in (
            CLAUDE_CODE_PLUGIN_POINTER
        )
        assert "claude plugin marketplace add qualixar/superlocalmemory" in (
            CLAUDE_CODE_PLUGIN_POINTER
        )

    def test_every_slm_command_it_mentions_is_a_real_one(self) -> None:
        """The general form of the bug, so the next edit cannot reintroduce it.

        The subcommand list is read from the CLI's own usage line rather than
        from a copy kept here — a copy would drift and start agreeing with a
        wrong instruction. There is no importable parser factory; the parser is
        built inside ``main()``, so this asks the CLI the way a user does.
        """
        import os
        import re
        import subprocess
        import sys

        from superlocalmemory.hooks.portable_kit import CLAUDE_CODE_PLUGIN_POINTER

        env = dict(os.environ, PYTHONPATH=str(REPO / "src"))
        proc = subprocess.run(
            [sys.executable, "-m", "superlocalmemory.cli", "--help"],
            capture_output=True, text=True, env=env, timeout=120,
        )
        usage = proc.stdout + proc.stderr
        match = re.search(r"\{([a-z0-9,\-_]+)\}", usage)
        assert match, f"could not read the subcommand list from:\n{usage[:400]}"
        known = set(match.group(1).split(","))

        mentioned = set(re.findall(r"\bslm ([a-z][a-z-]*)", CLAUDE_CODE_PLUGIN_POINTER))
        unknown = sorted(mentioned - known)
        assert not unknown, (
            f"the instruction tells users to run {unknown}, which slm does not "
            f"accept"
        )


class TestTheBridgeInstallNamesAPackageThatExists:
    """`docs/ide-setup.md` said to install `@modelcontextprotocol/client-cli`.

    That name 404s on npm and always has, while the config we write names the
    `mcp-remote` binary — which the `mcp-remote` package provides. So the
    documented path could not produce the binary the documented config needs.
    """

    DOC = REPO / "docs" / "ide-setup.md"

    @pytest.mark.skipif(
        not (REPO / "docs" / "ide-setup.md").is_file(),
        reason="docs not present in this layout",
    )
    def test_the_dead_package_name_is_gone(self) -> None:
        assert "client-cli" not in self.DOC.read_text(encoding="utf-8"), (
            "@modelcontextprotocol/client-cli does not exist on npm"
        )

    @pytest.mark.skipif(
        not (REPO / "docs" / "ide-setup.md").is_file(),
        reason="docs not present in this layout",
    )
    def test_the_documented_install_provides_the_binary_we_configure(self) -> None:
        """The two halves have to agree: whatever the docs tell you to install
        must be what supplies the command our connector writes."""
        from superlocalmemory.hooks.portable_kit import connect_ide  # noqa: F401

        text = self.DOC.read_text(encoding="utf-8")
        assert "npm install -g mcp-remote" in text
        # The connector writes "mcp-remote" as the command; that binary comes
        # from the package of the same name.
        assert "`mcp-remote`" in text

    @pytest.mark.skipif(
        not (REPO / "docs" / "ide-setup.md").is_file(),
        reason="docs not present in this layout",
    )
    def test_the_npx_alternative_changes_the_command_not_just_the_install(
        self,
    ) -> None:
        """`npx -y mcp-remote` leaves no `mcp-remote` binary on PATH.

        So offering npx as the no-global-install route is only correct if the
        config offered with it invokes npx as well. Swapping the install line
        alone would recreate the original defect in a new shape: a documented
        install that cannot produce the command the documented config names.
        """
        text = self.DOC.read_text(encoding="utf-8")
        if "npx" not in text:
            pytest.skip("docs do not offer the npx route")

        assert '"command": "npx"' in text or 'command = "npx"' in text, (
            "docs offer npx but every config still names the bare mcp-remote "
            "binary, which npx does not install"
        )
        # Every fenced block that runs mcp-remote through npx has to pass -y.
        # On a terminal, a cold cache makes npx stop on "Ok to proceed? (y)"
        # and wait, which is exactly what someone trying the command by hand
        # hits first. (Spawned by an IDE, with no tty, npm 11 installs without
        # asking - so -y is the guarantee, not the observed default.)
        blocks = text.split("```")[1::2]
        npx_blocks = [b for b in blocks if "npx" in b and "mcp-remote" in b]
        assert npx_blocks, "the npx route is described but never shown"
        for block in npx_blocks:
            assert "-y" in block, (
                f"an npx invocation of mcp-remote omits -y, so a first run on "
                f"a terminal stops on the install prompt:\n{block}"
            )


class TestTuningTwoSettingsSurvivesARestart:
    """`math` and `channel_weights` were never written and never restored.

    Both are read by engine wiring — the consistency threshold and the retrieval
    base weights. Setting either, by hand or by switching mode, lasted until the
    next restart and then reverted with nothing said.
    """

    def _config_in(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
        from superlocalmemory.core.config import SLMConfig

        return SLMConfig.load()

    def test_a_tuned_weight_is_still_there_after_a_reload(
        self, tmp_path, monkeypatch,
    ) -> None:
        from superlocalmemory.core.config import SLMConfig

        config = self._config_in(tmp_path, monkeypatch)
        assert config.channel_weights.semantic != 0.77, "pick a non-default value"

        dataclasses.replace(
            config,
            channel_weights=dataclasses.replace(
                config.channel_weights, semantic=0.77,
            ),
        ).save()

        assert SLMConfig.load().channel_weights.semantic == 0.77

    def test_a_tuned_threshold_is_still_there_after_a_reload(
        self, tmp_path, monkeypatch,
    ) -> None:
        from superlocalmemory.core.config import SLMConfig

        config = self._config_in(tmp_path, monkeypatch)
        assert config.math.sheaf_contradiction_threshold != 0.42

        dataclasses.replace(
            config,
            math=dataclasses.replace(
                config.math, sheaf_contradiction_threshold=0.42,
            ),
        ).save()

        assert SLMConfig.load().math.sheaf_contradiction_threshold == 0.42

    def test_a_pair_comes_back_a_pair_and_not_a_list(
        self, tmp_path, monkeypatch,
    ) -> None:
        """JSON has no tuple. Written as a list and restored as one, every
        reader that unpacks this gets a list, and the difference shows up
        somewhere a long way from the config file."""
        from superlocalmemory.core.config import SLMConfig

        config = self._config_in(tmp_path, monkeypatch)
        dataclasses.replace(
            config,
            math=dataclasses.replace(config.math, langevin_weight_range=(0.1, 0.9)),
        ).save()

        restored = SLMConfig.load().math.langevin_weight_range
        assert restored == (0.1, 0.9)
        assert isinstance(restored, tuple), f"came back a {type(restored).__name__}"

    def test_a_corrupt_section_falls_back_instead_of_bricking_slm(
        self, tmp_path, monkeypatch,
    ) -> None:
        """The same contract the neighbouring sections keep. A config file
        someone hand-edited badly must not make the tool unrunnable."""
        from superlocalmemory.core.config import SLMConfig

        self._config_in(tmp_path, monkeypatch).save()
        path = tmp_path / "config.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        data["math"] = "not a section"
        data["channel_weights"] = [1, 2, 3]
        path.write_text(json.dumps(data), encoding="utf-8")

        config = SLMConfig.load()
        assert config.math.sheaf_contradiction_threshold == 0.45
        assert config.channel_weights.semantic == 1.5


class TestAMigrationFailureSaysWhatFailed:
    """The health check named a migration and never said what was wrong.

    A migration recorded ``complete`` is re-checked by its own ``verify()`` on
    every start. When that check stops passing, the log still reads ``complete``
    and the health endpoint reports a failure — two surfaces answering different
    questions, with nothing anywhere saying so. The reporter had to guess, and
    guessed at a stale failure marker that does not exist.
    """

    def test_the_health_report_carries_the_reason_not_just_the_name(self) -> None:
        import inspect

        from superlocalmemory.server import unified_daemon

        source = inspect.getsource(unified_daemon)
        assert "migration_failure_reasons" in source
        # Both must be present, and the reasons must be built from the runner's
        # own details rather than composed here. Asserting they sit within N
        # characters of each other was brittle for no benefit -- inserting one
        # unrelated field between them broke it without changing any behaviour.
        assert "migration_details.get(name" in source, (
            "the reasons field must be built from the runner's own details, "
            "not invented here"
        )

    def test_status_flags_a_migration_whose_end_state_stopped_holding(
        self, tmp_path,
    ) -> None:
        """The store says complete; verification says otherwise. Both are true,
        and the command has to show that rather than pick one."""
        from superlocalmemory.cli.db_migrate import _end_state_disagreements

        memory_db = tmp_path / "memory.db"
        learning_db = tmp_path / "learning.db"
        for path in (memory_db, learning_db):
            conn = sqlite3.connect(path)
            conn.execute(
                "CREATE TABLE migration_log ("
                "  name TEXT PRIMARY KEY, ddl_hash TEXT, applied_at TEXT,"
                "  status TEXT)"
            )
            conn.commit()
            conn.close()

        conn = sqlite3.connect(memory_db)
        # Recorded as done, with nothing behind it: M043 needs atomic_facts to
        # carry a `quarantined` column, and this store has no such table at all.
        conn.execute(
            "INSERT INTO migration_log (name, ddl_hash, applied_at, status) "
            "VALUES ('M043_quarantine_display_summaries', 'x', 'now', 'complete')"
        )
        conn.execute("CREATE TABLE atomic_facts (fact_id TEXT PRIMARY KEY)")
        conn.commit()
        conn.close()

        found = _end_state_disagreements(learning_db, memory_db)

        assert "M043_quarantine_display_summaries" in found
        assert "no longer holds" in found["M043_quarantine_display_summaries"]

    def test_a_store_in_good_order_is_flagged_for_nothing(self, tmp_path) -> None:
        """The control. A diagnostic that fires on a healthy store is noise,
        and noise is how a real warning stops being read."""
        from superlocalmemory.cli.db_migrate import _end_state_disagreements

        memory_db = tmp_path / "memory.db"
        learning_db = tmp_path / "learning.db"
        for path in (memory_db, learning_db):
            conn = sqlite3.connect(path)
            conn.execute(
                "CREATE TABLE migration_log ("
                "  name TEXT PRIMARY KEY, ddl_hash TEXT, applied_at TEXT,"
                "  status TEXT)"
            )
            conn.commit()
            conn.close()

        assert _end_state_disagreements(learning_db, memory_db) == {}

    def test_it_never_breaks_the_command_it_prints_beside(self, tmp_path) -> None:
        """This is a diagnostic. It must never be why a status command fails."""
        from superlocalmemory.cli.db_migrate import _end_state_disagreements

        assert _end_state_disagreements(
            tmp_path / "nope.db", tmp_path / "also-nope.db",
        ) == {}
