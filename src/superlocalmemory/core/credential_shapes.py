# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Credential shapes for text that is about to leave the machine.

``security_primitives.redact_secrets(aggression="high")`` runs these on top of
its own patterns. They exist because the older scan missed shapes that real
keys and passwords take: a password inside a connection URL, ``label=value``
secrets, hex and UUID keys (hex can never clear the entropy threshold), Slack
user/app tokens, Stripe keys, short keys after a label, a PEM key's body.

Two rules keep ordinary information intact:

* A hex run or a UUID is redacted only after a key-like label — a commit id,
  a checksum in a changelog or a UUID used as a record id is not a secret.
* A label only makes its own value a secret; the label itself, the host of a
  URL and everything around them stay, so the memory still reads
  "db_password: [redacted]" rather than losing the sentence.

Normal aggression (what is scrubbed before a memory is stored, and most log
lines) runs only the vendor formats (``redact_token_shapes``), which nothing
ordinary resembles. The label, connection-string and prose rules are for text
leaving the machine only: a local memory such as a wifi password has to stay
answerable on this machine, and redacting at storage cannot be undone.

Every pattern is linear: bounded quantifiers, no nested repetition.
"""

from __future__ import annotations

import re
from collections.abc import Callable

#: Values shorter than this keep none of their characters in the local marker.
#: Four characters of a 40-character key say little; four of a 9-character
#: password are most of it.
_TAIL_MIN_LEN = 20
_SAFE_TAIL = re.compile(r"^[A-Za-z0-9_\-]{4}$")

#: Whole-token shapes. Replaced entirely; group 1 (when present) is kept.
_TOKEN_SHAPES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(
        r"-----BEGIN [A-Z0-9 ]{0,40}PRIVATE KEY[A-Z ]{0,10}-----"
        r"(?:[\s\S]{0,12000}?-----END [A-Z0-9 ]{0,40}PRIVATE KEY[A-Z ]{0,10}-----"
        r"|(?:[ \t]*\r?\n[ \t]*[A-Za-z0-9+/=:\-]{1,200}){0,400})"
    ), "PRIVATE_KEY"),
    (re.compile(r"\bxox[a-z]-[A-Za-z0-9\-]{10,400}"), "SLACK"),
    (re.compile(r"\bxapp-\d-[A-Za-z0-9\-]{10,400}"), "SLACK"),
    (re.compile(r"\b(?:sk|rk)_(?:live|test)_[A-Za-z0-9]{10,200}"), "STRIPE"),
    (re.compile(r"\bwhsec_[A-Za-z0-9]{16,200}"), "STRIPE"),
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,255}"), "GITHUB"),
    (re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,255}"), "GITHUB"),
    (re.compile(r"\bgl(?:pat|ptt|dt|rt|cbt)-[A-Za-z0-9_\-]{20,200}"), "GITLAB"),
    (re.compile(r"\bnpm_[A-Za-z0-9]{30,200}"), "NPM"),
    (re.compile(r"\bhf_[A-Za-z0-9]{30,200}"), "HUGGINGFACE"),
    (re.compile(r"\bpypi-[A-Za-z0-9_\-]{40,400}"), "PYPI"),
    (re.compile(r"\bSG\.[A-Za-z0-9_\-]{16,100}\.[A-Za-z0-9_\-]{16,100}"), "SENDGRID"),
    (re.compile(r"\bkey-[0-9a-f]{32}\b"), "MAILGUN"),
    (re.compile(r"\b\d{8,10}:AA[A-Za-z0-9_\-]{30,100}"), "TELEGRAM"),
    (re.compile(r"\bGOCSPX-[A-Za-z0-9_\-]{20,100}"), "GOOGLE_OAUTH"),
    (re.compile(r"\bya29\.[A-Za-z0-9_\-]{20,400}"), "GOOGLE_OAUTH"),
    (re.compile(r"\bAIza[0-9A-Za-z_\-]{35}"), "GOOGLE_API"),
    (re.compile(r"(https://hooks\.slack\.com/(?:services|workflows|triggers)/)"
                r"[A-Za-z0-9/_\-]{8,300}"), "SLACK_WEBHOOK"),
    (re.compile(r"(https://(?:discord|discordapp)\.com/api/webhooks/\d{1,30}/)"
                r"[A-Za-z0-9_\-]{8,300}"), "DISCORD_WEBHOOK"),
)

#: The vendor token shapes, for callers that only detect (no replacement).
TOKEN_SHAPES = _TOKEN_SHAPES

#: ``Authorization: <scheme> <credential>`` — keep the header and the scheme.
_AUTH_HEADER = re.compile(
    r"(?i)(\b(?:proxy-)?authorization[\"']?\s*[:=]\s*[\"']?\s*"
    r"(?:basic|bearer|token|digest|negotiate)\s+)([A-Za-z0-9+/=._~\-]{6,4096})"
)

#: ``scheme://user:password@host`` — keep scheme, user and host.
_URL_PASSWORD = re.compile(
    r"\b([a-z][a-z0-9+.\-]{1,20}://[^\s:/?#@\[\]]{0,256}:)([^\s/?#]{1,512})@"
)
#: ``scheme://<token>@host`` — a long token used as the user part.
_URL_TOKEN_USER = re.compile(r"\b([a-z][a-z0-9+.\-]{1,20}://)([A-Za-z0-9_\-]{16,256})@")

#: ``?token=…`` / ``&sig=…`` in a URL query.
_QUERY_SECRET = re.compile(
    r"(?i)([?&](?:access_token|refresh_token|id_token|token|api_key|apikey|key|sig|"
    r"signature|secret|client_secret|password|auth)=)([^&\s#\"'<>]{3,1024})"
)

#: ``--password X`` / ``--token=X`` on a command line.
_CLI_FLAG = re.compile(
    r"(?i)(--(?:password|passwd|token|api-key|apikey|secret|client-secret|"
    r"access-token|auth-token)(?:=|\s+))([^\s\-'\"][^\s'\"]{0,511})"
)

_STRONG_WORDS = (
    r"password|passwd|passwort|passphrase|passcode|pwd|secret|token|authtoken|"
    r"api[_\-]?key|access[_\-]?key|secret[_\-]?key|private[_\-]?key|signing[_\-]?key|"
    r"encryption[_\-]?key|master[_\-]?key|account[_\-]?key|client[_\-]?secret|"
    r"credentials?"
)
#: A label whose last word is a credential word: ``password``, ``DB_PASSWORD``,
#: ``PGPASSWORD``, ``authToken``, ``"api_key"``. ``max_tokens`` / ``tokenizer``
#: / ``token_count`` do not end in one, so they never match.
_LABELLED = re.compile(
    r"(?i)(?<![A-Za-z0-9])(?P<label>[A-Za-z0-9_.\-]{0,60}?(?:" + _STRONG_WORDS + r"))"
    r"(?![A-Za-z0-9])(?P<sep>[\"']?\s{0,4}(?:=>|[:=])\s{0,4})"
    r"(?P<value>\"[^\"\n]{1,512}\"|'[^'\n]{1,512}'|[^\s'\"`]{1,512})"
)
#: A label ending in "key" that is not a credential word by itself
#: (``STRIPE_KEY``, ``DD_APP_KEY``, ``key``): only a key-shaped value counts.
_WEAK_LABELLED = re.compile(
    r"(?i)(?<![A-Za-z0-9])(?P<label>[A-Za-z0-9_.\-]{0,60}?key)"
    r"(?![A-Za-z0-9])(?P<sep>[\"']?\s{0,4}(?:=>|[:=])\s{0,4})"
    r"(?P<value>\"[^\"\n]{1,512}\"|'[^'\n]{1,512}'|[^\s'\"`]{1,512})"
)
#: Prose: "my password is hunter2", "the API key 0123…".
_PROSE = re.compile(
    r"(?i)\b(?P<label>password|passphrase|passcode|pin|api[ _\-]?key|access[ _\-]?key|"
    r"secret[ _\-]?key|private[ _\-]?key|token|secret|key)"
    r"(?P<sep>\s{1,4}(?:is|was|=|:)\s{1,4}|\s{1,4})"
    r"(?P<value>\"[^\"\n]{1,512}\"|'[^'\n]{1,512}'|[^\s'\"`]{1,512})"
)

_PLACEHOLDER = re.compile(
    r"(?i)^(?:true|false|null|none|nil|yes|no|on|off|required|optional|redacted|"
    r"example|changeme|x{3,}|\*{1,}|%[A-Za-z_][A-Za-z0-9_]*%|"
    r"\[REDACTED[^\]]*\]|\[redacted\])$"
)
#: Syntax that names a value instead of holding one: a template slot, a shell
#: or environment expansion, a lookup in code. Anchored at the start.
_REFERENCE = re.compile(
    r"^(?:<|\$\{|\$\(|\{\{|%\(|process\.env\.|os\.getenv\(|os\.environ|"
    r"getenv\(|System\.getenv\(|ENV\[|env\()"
)
#: ``$API_KEY`` — an upper-case variable name. ``$uperSecret1`` is a password.
_SHELL_VAR = re.compile(r"^\$[A-Z_][A-Z0-9_]*$")
_HEX = re.compile(r"^[0-9a-fA-F]{16,}$")
_UUID = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
_KEYLIKE = re.compile(r"^[A-Za-z0-9+/=_\-]{16,}$")
_TRAILING = ".,;)]}>"
#: Prose labels too common to trust on their own ("Key is SHA-256
#: namespaced"): after these, only a key-shaped value counts.
_WEAK_PROSE = frozenset({"key", "secret"})
_SHELL_DIR_VARS = frozenset({"pwd", "oldpwd"})

#: An identifier with a credential word anywhere in it, not only at the end:
#: ``SLM_SECRET_KEY_2024``, ``DB_PASS``, ``api_token_v2``, ``STRIPE_SECRET_PROD``.
#: Bounded segments, and a start only where an identifier starts, keep it linear.
_SEGMENTED = re.compile(
    r"(?<![A-Za-z0-9_.\-])"
    r"(?P<label>[A-Za-z][A-Za-z0-9]{0,40}(?:[_.\-][A-Za-z0-9]{1,40}){1,8})"
    r"(?![A-Za-z0-9_.\-])(?P<sep>[\"']?\s{0,4}(?:=>|[:=])\s{0,4})"
    r"(?P<value>\"[^\"\n]{1,512}\"|'[^'\n]{1,512}'|[^\s'\"`]{1,512})"
)
_SEGMENT_SPLIT = re.compile(r"[_.\-]")
#: A segment equal to one of these makes the identifier a credential label.
#: Plurals and compounds (``tokens``, ``tokenizer``) are deliberately absent.
_SEGMENT_WORDS = frozenset({
    "password", "passwd", "passphrase", "passcode", "pass", "pwd", "secret",
    "secrets", "token", "apikey", "credential", "credentials", "creds", "privatekey",
})
#: Prose labels after which a plain word, ending its clause, is the password
#: itself: "my password is sunshine." The words below are what a sentence
#: says ABOUT a password ("the password is correct, but…"), not one.
_PLAIN_WORD_LABELS = ("password", "passwd", "passphrase", "passcode")
_STATE_WORDS = frozenset({
    "required", "optional", "needed", "necessary", "mandatory", "correct", "incorrect",
    "wrong", "right", "invalid", "valid", "expired", "expiring", "missing", "empty",
    "blank", "weak", "strong", "short", "long", "secure", "insecure", "set", "unset",
    "stored", "saved", "encrypted", "hashed", "salted", "protected", "changed", "reset",
    "updated", "rotated", "same", "different", "known", "unknown", "hidden", "visible",
    "disabled", "enabled", "locked", "unlocked", "compromised", "leaked", "shared",
    "private", "public", "good", "bad", "fine", "ok", "okay", "here", "there", "below",
    "above", "attached", "ready", "pending", "new", "old", "temporary", "permanent",
    "default", "generated", "random", "managed", "handled", "sensitive", "complex",
    "simple", "easy", "hard", "obvious", "unchanged", "lost", "forgotten", "accepted",
    "rejected", "revoked", "active", "inactive", "nothing", "unavailable", "available",
    "configured", "provided", "supplied", "specified", "listed", "given", "removed",
    "deleted", "gone", "secret", "confidential", "safe", "unsafe", "case-sensitive",
    "it", "this", "that", "mine", "yours", "his", "hers", "ours", "theirs",
})


def local_marker(label: str, value: str) -> str:
    """``[REDACTED:<LABEL>:<tail>]`` — the shape ``redact_secrets`` emits."""
    tail = value[-4:]
    if len(value) < _TAIL_MIN_LEN or not _SAFE_TAIL.match(tail):
        tail = "****"
    return f"[REDACTED:{label}:{tail}]"


def _is_placeholder(raw: str, value: str) -> bool:
    """A template, an expansion or an elided example — not a secret.

    Deliberately syntactic only. "your-key-here" reads like a placeholder, but
    so does a weak real password, and leaking one costs more than redacting a
    documentation example.
    """
    for candidate in (raw, raw.strip("\"'"), value):
        if (_PLACEHOLDER.match(candidate) or _REFERENCE.match(candidate)
                or _SHELL_VAR.match(candidate) or "..." in candidate or "…" in candidate):
            return True
    return False


def _is_key_shaped(value: str) -> bool:
    if _HEX.match(value) or _UUID.match(value):
        return True
    return bool(_KEYLIKE.match(value)) and any(c.isdigit() for c in value) and any(
        c.isalpha() for c in value
    )


def _looks_secret(value: str) -> bool:
    """For contexts where prose is likely (``:``, "is"): not a plain word."""
    if _is_key_shaped(value) or len(value) >= 16:
        return True
    if any(c.isdigit() for c in value) or any(not c.isalnum() for c in value):
        return True
    return any(c.isupper() for c in value[1:]) and any(c.islower() for c in value)


def _split_value(raw: str) -> tuple[str, str, str]:
    """(opening quote, the value, what follows it that is not part of it)."""
    if len(raw) >= 2 and raw[0] in "\"'" and raw[-1] == raw[0]:
        return raw[0], raw[1:-1], raw[0]
    core = raw.rstrip(_TRAILING)
    return "", core, raw[len(core):]


def _ends_clause(match: re.Match[str], rest: str) -> bool:
    """The value is the last word of its clause: punctuation, a line end or
    the end of the text follows it ("…is sunshine." / "…is sunshine")."""
    if rest:
        return True
    return match.string[match.end():match.end() + 1] in ("", "\n", "\r")


def _plain_password(label: str, value: str, rest: str, match: re.Match[str]) -> bool:
    """A plain word that IS the password: "my password is sunshine."

    Only after a password-type label, only when the word ends its clause, and
    never a word that describes a password ("is correct", "is stored in…").
    """
    return (label.lower().endswith(_PLAIN_WORD_LABELS)
            and value.replace("-", "").isalpha()
            and value.lower() not in _STATE_WORDS
            and _ends_clause(match, rest))


def _value_sub(*, strict: Callable[[str, str, bool], bool],
               plain_words: bool = False) -> Callable[[re.Match[str]], str]:
    def _sub(match: re.Match[str]) -> str:
        quote, value, rest = _split_value(match.group("value"))
        name = match.group("label")
        if (len(value) < 3 or _is_placeholder(match.group("value"), value) or "://" in value
                or value.startswith(("/", "~/", "./", "../"))
                or name.lower() in _SHELL_DIR_VARS):
            return match.group(0)
        if not (strict(match.group("sep"), value, bool(quote))
                or (plain_words and _plain_password(name, value, rest, match))):
            return match.group(0)
        return f"{name}{match.group('sep')}{quote}{local_marker('SECRET', value)}{rest}"
    return _sub


def _strong_ok(sep: str, value: str, quoted: bool) -> bool:
    # ``=`` is an assignment and a quoted value is a value. Only an unquoted
    # value after ``:`` might be a word in a sentence ("token: the next step").
    return "=" in sep or quoted or _looks_secret(value)


def _weak_ok(_sep: str, value: str, _quoted: bool) -> bool:
    # ``primary_key: id`` and ``sort_key: "created_at"`` are not keys.
    return _is_key_shaped(value)


_STRONG_SUB = _value_sub(strict=_strong_ok, plain_words=True)
_WEAK_SUB = _value_sub(strict=_weak_ok)


def _segment_sub(match: re.Match[str]) -> str:
    """``SLM_SECRET_KEY_2024=…``: a credential word anywhere in the name.

    Stricter about the value than an end-of-name label, because a word in the
    middle of a name is weaker evidence: ``PASSWORD_MIN_LENGTH=12``,
    ``TOKEN_LIMIT=4096`` and ``token_type: bearer`` are configuration.
    """
    label = match.group("label")
    segments = {part.lower() for part in _SEGMENT_SPLIT.split(label)}
    strong = not segments.isdisjoint(_SEGMENT_WORDS)
    if not strong and "key" not in segments:
        return match.group(0)
    raw = match.group("value")
    quote, value, rest = _split_value(raw)
    if (len(value) < 6 or value.isdigit() or _is_placeholder(raw, value)
            or "://" in value or value.startswith(("/", "~/", "./", "../"))):
        return match.group(0)
    sep = match.group("sep")
    if not strong or ("=" not in sep and not quote):
        ok = _is_key_shaped(value) or (
            len(value) >= 8 and any(c.isdigit() for c in value)
            and any(c.isalpha() for c in value))
    else:
        ok = _looks_secret(value)
    if not ok:
        return match.group(0)
    return f"{label}{sep}{quote}{local_marker('SECRET', value)}{rest}"


def _prose_sub(match: re.Match[str]) -> str:
    quote, value, rest = _split_value(match.group("value"))
    if len(value) < 3 or _is_placeholder(match.group("value"), value):
        return match.group(0)
    label = match.group("label")
    if not match.group("sep").strip() or label.lower() in _WEAK_PROSE:
        # "the api key 0123…": no "is" / ":" between, so only a key shape counts.
        ok = _is_key_shaped(value)
    else:
        ok = (bool(quote) or _looks_secret(value)
              or _plain_password(label, value, rest, match))
    if not ok:
        return match.group(0)
    start = match.start("value") - match.start(0) + len(quote)
    whole = match.group(0)
    return whole[:start] + local_marker("SECRET", value) + whole[start + len(value):]


def _keep_prefix_sub(label: str) -> Callable[[re.Match[str]], str]:
    def _sub(match: re.Match[str]) -> str:
        value = match.group(2)
        if _is_placeholder(value, value):
            return match.group(0)
        tail = match.group(0)[match.end(2) - match.start(0):]
        return f"{match.group(1)}{local_marker(label, value)}{tail}"
    return _sub


def _shape_sub(label: str) -> Callable[[re.Match[str]], str]:
    def _sub(match: re.Match[str]) -> str:
        whole = match.group(0)
        keep = match.group(1) if match.re.groups else ""
        return f"{keep}{local_marker(label, whole[len(keep):])}"
    return _sub


#: A marker an earlier pass emitted. Never scanned again: its label
#: (``URL_PASSWORD``, ``PRIVATE_KEY``) would otherwise read as a credential
#: label and swallow the text after it.
_MARKER = re.compile(r"\[REDACTED:[A-Z_]+:[^\]]*\]")


def _outside_markers(text: str, apply: Callable[[str], str]) -> str:
    parts: list[str] = []
    last = 0
    for marker in _MARKER.finditer(text):
        parts.append(apply(text[last:marker.start()]))
        parts.append(marker.group(0))
        last = marker.end()
    parts.append(apply(text[last:]))
    return "".join(parts)


def _token_shapes(text: str) -> str:
    out = text
    for pattern, label in _TOKEN_SHAPES:
        out = pattern.sub(_shape_sub(label), out)
    return out


def _connection_secrets(text: str) -> str:
    out = _AUTH_HEADER.sub(_keep_prefix_sub("AUTH_HEADER"), text)
    out = _URL_PASSWORD.sub(_keep_prefix_sub("URL_PASSWORD"), out)
    out = _URL_TOKEN_USER.sub(_url_user_sub, out)
    out = _QUERY_SECRET.sub(_keep_prefix_sub("URL_SECRET"), out)
    return _CLI_FLAG.sub(_keep_prefix_sub("SECRET"), out)


def _labelled_values(text: str) -> str:
    out = _LABELLED.sub(_STRONG_SUB, text)
    out = _WEAK_LABELLED.sub(_WEAK_SUB, out)
    out = _SEGMENTED.sub(_segment_sub, out)
    return _PROSE.sub(_prose_sub, out)


def _url_user_sub(match: re.Match[str]) -> str:
    user = match.group(2)
    if not (any(c.isdigit() for c in user) and any(c.isalpha() for c in user)):
        return match.group(0)
    return f"{match.group(1)}{local_marker('URL_TOKEN', user)}@"


def redact_token_shapes(text: str) -> str:
    """Known vendor credential formats, replaced whole. Both aggressions.

    Each has a prefix nothing ordinary carries (``xoxp-``, ``sk_live_``, a
    webhook path, a PEM private-key block), so it is as safe to apply before a
    memory is stored as the older ``sk-`` / ``ghp_`` / ``AKIA`` patterns are.
    """
    return _outside_markers(text, _token_shapes)


def redact_connection_secrets(text: str) -> str:
    """Credentials inside ordinary syntax: an ``Authorization`` header, a
    connection-URL password, ``?token=`` in a query, ``--password X``.
    High aggression only — text leaving the machine."""
    return _outside_markers(text, _connection_secrets)


def redact_labelled_values(text: str) -> str:
    """Values after credential labels. Runs after every token pattern, so a
    known token keeps its specific label and a marker is never re-wrapped."""
    return _outside_markers(text, _labelled_values)


__all__ = [
    "local_marker",
    "redact_connection_secrets",
    "redact_labelled_values",
    "redact_token_shapes",
]
