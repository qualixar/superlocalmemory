# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Generic secret assignments, and the prose that must survive next to them.

The strong (hosted) screen used to recognise a credential label only when the
credential word ended the name: ``DB_PASSWORD=…`` was caught,
``SLM_SECRET_KEY_2024=…`` and ``DB_PASS=…`` went out verbatim. "my password
is sunshine" went out too, because a plain word did not look like a secret.

Both directions are measured: every fake credential below is replaced, and
twenty ordinary sentences — several of them deliberately close to the new
rules — come back byte for byte.

Every value is an obvious placeholder assembled at run time.
"""

from __future__ import annotations

import pytest

from superlocalmemory.core.security_primitives import redact_secrets
from superlocalmemory.retrieval.hosted_redaction import (
    REDACTED_MARKER,
    redact_for_hosted_judge,
)

#: (shape, text, the value that must not survive)
GENERIC_ASSIGNMENTS: tuple[tuple[str, str, str], ...] = (
    ("credential word mid-name, year suffix",
     "SLM_SECRET_KEY_2024=abcd1234efgh5678", "abcd1234efgh5678"),
    ("credential word mid-name, env suffix",
     "export STRIPE_SECRET_PROD=sk9Xa8b7c6d5", "sk9Xa8b7c6d5"),
    ("token mid-name", "GITHUB_TOKEN_CI=abcdef123456ghijkl", "abcdef123456ghijkl"),
    ("secret mid-name", "MY_SECRET_VALUE=abcd1234efgh5678", "abcd1234efgh5678"),
    ("Rails secret_key_base", "SECRET_KEY_BASE=9f8e7d6c5b4a39281706f5e4d3c2b1a0",
     "9f8e7d6c5b4a39281706f5e4d3c2b1a0"),
    ("versioned token, quoted", "api_token_v2 = 'zz9yy8xx7ww6'", "zz9yy8xx7ww6"),
    ("PASS segment", "DB_PASS=Hunt3rTwo!x9", "Hunt3rTwo!x9"),
    ("YAML token mid-name", "X_API_TOKEN_PROD: FAKE0123fake4567", "FAKE0123fake4567"),
    ("YAML token at the end", "X_API_TOKEN: q8w7e6r5t4y3u2i1o0p9", "q8w7e6r5t4y3u2i1o0p9"),
    ("JSON key mid-name", '{"secret_key_2": "FAKEfake0000"}', "FAKEfake0000"),
    ("password is a plain word, end of text", "my password is sunshine", "sunshine"),
    ("password is a plain word, end of sentence", "The wifi password is coffeeshop.",
     "coffeeshop"),
    ("passphrase is a plain word", "Our passphrase is tangerine; do not share it.",
     "tangerine"),
    ("password was a plain word", "the old password was marmalade", "marmalade"),
    ("password is, with symbols", "password is Hunt3rTwo!x9", "Hunt3rTwo!x9"),
)

#: Twenty ordinary sentences that must leave unchanged.
ORDINARY_SENTENCES: tuple[str, ...] = (
    "The meeting with the platform team moved to Thursday at 3pm.",
    "We decided to use PostgreSQL 16 for the billing service.",
    "Remember to rotate the API keys every 90 days.",
    "The password is stored in the team vault, not in the repo.",
    "Your password is too short; use at least twelve characters.",
    "The password is correct, but the account is locked.",
    "The password is required.",
    "PASSWORD_MIN_LENGTH=12 and TOKEN_LIMIT=4096 in the config.",
    "SECRET_ROTATION_DAYS=90 is the policy for every service.",
    "token_type: bearer is what the OAuth server returns.",
    "Set max_tokens to 1024 for the summarizer.",
    "Key decisions: ship 4.1.19 on Friday, defer the UI rework.",
    "The secret to fast recall is a warm embedding model.",
    "Authentication uses OAuth 2.0 with PKCE for the dashboard.",
    "Varun prefers Sonnet 5 for routine work and Opus 5 for architecture.",
    "Use https://github.com/qualixar/superlocalmemory for the source.",
    "The record id 123e4567-e89b-12d3-a456-426614174000 belongs to Atlas.",
    "Secret Santa gift exchange is on December 18th.",
    "TOKEN_ENDPOINT_PATH=/oauth/token is configured on the gateway.",
    "The password is the same as the staging one, ask Priya.",
)


@pytest.mark.parametrize(
    ("shape", "text", "value"), GENERIC_ASSIGNMENTS, ids=[g[0] for g in GENERIC_ASSIGNMENTS],
)
def test_a_generic_secret_assignment_never_leaves_the_machine(
    shape: str, text: str, value: str,
) -> None:
    out = redact_for_hosted_judge(text)
    assert value not in out, f"{shape}: credential survived: {out!r}"
    assert REDACTED_MARKER in out


def test_the_label_survives_and_only_the_value_goes() -> None:
    assert redact_for_hosted_judge("SLM_SECRET_KEY_2024=abcd1234efgh5678") == (
        f"SLM_SECRET_KEY_2024={REDACTED_MARKER}"
    )
    assert redact_for_hosted_judge("my password is sunshine.") == (
        f"my password is {REDACTED_MARKER}."
    )


@pytest.mark.parametrize("text", ORDINARY_SENTENCES)
def test_an_ordinary_sentence_leaves_unchanged(text: str) -> None:
    assert redact_for_hosted_judge(text) == text


def test_there_are_twenty_ordinary_sentences() -> None:
    assert len(ORDINARY_SENTENCES) == 20


def test_storage_aggression_still_keeps_the_new_shapes() -> None:
    """Credentials stay in SLM: what is stored is screened at normal
    aggression, which must not start eating these assignments."""
    for _shape, text, _value in GENERIC_ASSIGNMENTS:
        assert redact_secrets(text) == text, text


#: A deliberate bound, in CPU seconds of this thread rather than wall-clock.
#: The linear scan of this 140 kB input costs ~0.27 s of CPU (doubling cleanly
#: with the input), but measured 0.65-0.76 s of wall-clock on a loaded host --
#: against the old 1.0 s wall-clock bound. CPU time does not grow while the
#: host deschedules the thread, so the same 1.0 s now has ~4x headroom, and a
#: rule that goes quadratic over even one 40 kB run of this input (~2 s of
#: CPU) still fails it.
_HOSTILE_CPU_CEILING_S = 1.0


def test_the_new_rules_stay_linear_on_hostile_input() -> None:
    import time

    hostile = ("A_" * 20000) + "=" + ("password is " * 5000) + ("x_" * 20000)
    started = time.thread_time()
    redact_for_hosted_judge(hostile)
    spent = time.thread_time() - started
    assert spent < _HOSTILE_CPU_CEILING_S, f"{spent:.2f}s of CPU on hostile input"
