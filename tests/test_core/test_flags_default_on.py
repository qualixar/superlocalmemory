"""Every 4.2.30 flag is on by default (Varun, 19 Nov 2026), and on without data it does no harm.

"Harmless" means: with no labels, no certificates and no anytime evidence, each gate acts exactly as it did
before 4.2.30 (no skip, no withhold, no retirement, nothing sent off the machine). The code path that
guarantees it is named in each test.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from superlocalmemory.core.feature_flags import (
    FLAGS,
    KILL_ENV,
    FeatureFlags,
    FlagState,
    env_name,
    set_kill_switch,
    write_flag,
)

ALL = sorted(FLAGS)


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / "slm-home"
    h.mkdir()
    monkeypatch.setenv("SLM_DATA_DIR", str(h))
    for name in list(os.environ):
        if name.startswith("SLM_FEATURE_") or name == KILL_ENV:
            monkeypatch.delenv(name)
    return h


# -- the registry -------------------------------------------------------------------------------

def test_there_are_exactly_the_flags_the_brief_lists_and_each_defaults_on():
    assert len(FLAGS) == 20  # truth.leases had no reader and was removed (F-09)
    assert all(f.default is FlagState.ON for f in FLAGS.values())


@pytest.mark.parametrize("flag_id", ALL)
def test_a_fresh_data_folder_reads_every_flag_on(home, flag_id):
    flags = FeatureFlags(home)
    assert flags.enabled(flag_id) is True and flags.state(flag_id) is FlagState.ON and not flags.killed


@pytest.mark.parametrize("flag_id", ALL)
def test_the_kill_switch_turns_each_one_off_by_file_and_by_environment(home, monkeypatch, flag_id):
    set_kill_switch(home, True)
    assert FeatureFlags(home).enabled(flag_id) is False
    set_kill_switch(home, False)
    assert FeatureFlags(home).enabled(flag_id) is True
    monkeypatch.setenv(KILL_ENV, "1")
    assert FeatureFlags(home).enabled(flag_id) is False


@pytest.mark.parametrize("flag_id", ALL)
def test_a_persons_explicit_off_wins_in_the_file_and_in_the_environment(home, monkeypatch, flag_id):
    write_flag(home, flag_id, FlagState.OFF)
    assert FeatureFlags(home).enabled(flag_id) is False
    write_flag(home, flag_id, FlagState.ON)
    monkeypatch.setenv(env_name(flag_id), "off")
    assert FeatureFlags(home).enabled(flag_id) is False


def test_an_unreadable_features_file_means_the_defaults_which_are_on(home):
    (home / "features.json").write_text("{not json")
    assert FeatureFlags(home).enabled("trust.secret_screen") is True


# -- delivery.need_gate (core/need_gate.py): no labels, no threshold -------------------------------------

def test_need_gate_without_labels_never_skips_by_score():
    """``calibrate([])`` is None (``needed_total`` is the unseen-turn weight alone, so the bound is above
    alpha at the first level) and ``Gate.decide`` passes ``threshold=None`` to ``decide``: the score
    skips nothing. Only the fixed rules (empty, acknowledgement, greeting, arithmetic, translation,
    slash command) skip, and the exploration draw recalls 5% of those anyway."""
    from superlocalmemory.core import need_gate

    assert need_gate.calibrate([]) is None
    for prompt in ("What did we decide about the database last time?", "refactor the parser", "why is it slow"):
        d = need_gate.decide(prompt, threshold=None, key=b"k" * 32)
        assert d.recall is True and d.rule is None and d.propensity == 1.0


def test_the_gate_object_on_a_store_without_turns_has_no_threshold_and_no_suspended_rule(home, tmp_path):
    from superlocalmemory.core import need_gate

    gate = need_gate.Gate(home, tmp_path / "missing.db")  # no database at all: fails to "no threshold"
    cal = gate.calibration("default")
    assert cal.threshold is None and cal.suspended == frozenset()
    decision = gate.decide("What did we decide about the database last time?", "default")
    assert decision is not None and decision.recall is True and decision.threshold is None


# -- delivery.harm_gate (core/harm_gate.py): shadow by construction ----------------------------------------

def test_harm_gate_never_acts_and_names_nothing_without_evidence(tmp_path):
    """``harm_gate.report`` returns ``acting: False`` unconditionally (it computes on read and writes nothing);
    with no randomised deliveries there is no test, so ``would_withhold`` is empty."""
    from superlocalmemory.core import harm_gate
    from tests.test_core.test_coverage_backfill import _db

    db = _db(tmp_path / "memory.db")
    with db.raw_connection() as conn:
        section = harm_gate.report(conn, "default", state="on")
    assert section["acting"] is False and section["would_withhold"] == [] and section["tests"] == 0


# -- judge.ladder (core/judge_tiers.py, D-20): nothing leaves without consent ---------------------------------

def test_the_hosted_judge_asks_nothing_without_a_consent_on_file(home):
    """``JevTier._permit`` returns None when ``consent_store.reference`` is None, before the judge is fetched,
    and ``build_ladder`` adds the hosted tier only while the flag is on (it is, now) - the consent is the gate."""
    from superlocalmemory.core import consent_store, judge_tiers
    from superlocalmemory.core.judge_questions import QUESTIONS

    assert FeatureFlags(home).enabled("judge.ladder") is True
    assert consent_store.reference(home, "judge_ladder_hosted") is None  # a fresh home has none

    def never():
        raise AssertionError("the hosted judge must not even be fetched")

    tier = judge_tiers.JevTier(never, consent_ref=lambda: consent_store.reference(home, "judge_ladder_hosted"),
                               live=lambda: FeatureFlags(home).enabled("judge.ladder"))
    for question in QUESTIONS.values():
        assert tier.available(question) is False
        assert tier.answer(question, []) is None


# -- trust.secret_screen: text with no credential is untouched ------------------------------------------------

def test_the_secret_screen_changes_nothing_in_text_without_a_credential():
    from superlocalmemory.trust import secret_screen

    text = "We decided to move the nightly backup to two in the morning, owner is the platform team."
    got = secret_screen.screen(text)
    assert got.text == text and got.total == 0


# -- compiler.wiki: the folder stays inside the data folder unless a person chose another ----------------------

def test_the_wiki_folder_is_inside_the_data_folder_by_default(home, monkeypatch):
    from superlocalmemory.compiler import wiki

    monkeypatch.delenv("SLM_WIKI_DIR", raising=False)
    root = wiki.wiki_root()
    assert Path(root).resolve() == (home / "wiki").resolve()


def test_a_chosen_wiki_folder_wins(home, monkeypatch, tmp_path):
    from superlocalmemory.compiler import wiki

    monkeypatch.setenv("SLM_WIKI_DIR", str(tmp_path / "mine"))
    assert wiki.wiki_root() == tmp_path / "mine"


# -- compiler.dream: an empty store proposes nothing; it never applies ---------------------------------------------

def test_the_dream_cycle_has_nothing_to_propose_on_a_fresh_store(tmp_path):
    from superlocalmemory.compiler import dream
    from tests.test_core.test_coverage_backfill import _db

    db = _db(tmp_path / "memory.db")
    assert dream.run_profile(db, "default")["skipped"] == "nothing to propose"


# -- judge certificates (compiler.auto_apply / truth.slots.auto_apply) --------------------------------------

def test_a_certificate_basis_needs_a_certificate_row_even_with_auto_apply_on(home):
    """``proposals.check_basis`` refuses a certificate that has no row: with no certificates nothing is applied
    on a certificate, whatever ``compiler.auto_apply`` says."""
    import sqlite3

    from superlocalmemory.compiler import proposals

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE judge_certificates (cert_id TEXT, profile_id TEXT, tier TEXT, question_id TEXT,"
                 " verdict TEXT, voided_at TEXT)")
    assert FeatureFlags(home).enabled("compiler.auto_apply") is True
    with pytest.raises(proposals.BasisRefused):
        proposals.check_basis(conn, "default", proposals.Basis.certificate("cert_that_does_not_exist"), ["merge"],
                              flags=FeatureFlags(home))


# -- portability.import / media.*: nothing runs by itself ------------------------------------------------------------

def test_the_media_parts_do_nothing_until_their_model_is_on_this_computer(home):
    from superlocalmemory.runtimes import media_packs

    assert media_packs.flag_enabled("media.captions", home) and media_packs.flag_enabled("media.document_pack", home)
    assert not media_packs.is_active("captions", home) and not media_packs.is_active("documents", home)
    assert media_packs.pack_client("captions", home) is None and media_packs.pack_client("documents", home) is None
    assert json.loads(json.dumps(FeatureFlags(home).as_dict()))["media.regions"] == "on"
