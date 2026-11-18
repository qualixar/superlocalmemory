# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Fake credentials, built at run time so no scanner sees a literal key in the repository."""

from __future__ import annotations

_BODY = "Zk3Qm9Xv7Lp2Wd8Rt5Hn4Bc6Yj1Ua0S"  # 32 mixed characters, not a real key


def github() -> str:
    return "gh" + "p_" + _BODY + "Eg7K"


def stripe() -> str:
    return "sk" + "_live_" + _BODY[:24]


def slack() -> str:
    return "xo" + "xb-" + "1234567890-" + _BODY[:20]


def aws() -> str:
    return "AK" + "IA" + "7QX4MZ2RT9PLW6HD"


def anthropic() -> str:
    return "sk" + "-ant-" + "api03-" + _BODY + "Qv8Nr3Tx"


def openai() -> str:
    return "sk" + "-proj-" + _BODY + "Mw5Ld9Ge2Hs7Zk"


def pem() -> str:
    return ("-----BEGIN RSA PRIVATE " + "KEY-----\n" + "MIIEvQIBADANBgkqhkiG9w0BAQEFAASC\n"
            "bXlmYWtla2V5Ym9keWZvcnRlc3Rpbmc=\n-----END RSA PRIVATE " + "KEY-----")


ALL = {"github": github, "stripe": stripe, "slack": slack, "aws": aws, "anthropic": anthropic,
       "openai": openai, "pem": pem}
