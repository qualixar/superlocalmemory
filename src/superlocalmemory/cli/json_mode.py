# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V4 | https://qualixar.com | https://varunpratap.com

"""JSON for every ``slm`` command: the switch and the one-document guarantee.

The contract an agent can rely on, whatever the command:

* ``slm --json <command> ...``, ``SLM_JSON=1 slm <command> ...`` and
  ``slm <command> ... --json`` are the same request.
* stdout carries exactly one JSON document, on one line: an object with a
  ``success`` key. Progress, banners, warnings and logs go to stderr.
* Errors carry ``{"code": "<snake_case>", "message": "<plain words>"}``.
  Exit 0 success, 1 the command ran and failed, 2 bad usage / could not run.
* Nothing prompts. A question a command would ask becomes an error envelope:
  ``confirmation_required`` (naming the flag that confirms) or
  ``input_required`` (naming the flags that answer).

``run_json`` holds the guarantee for every command, including ones written
before this module: it captures what the command prints, keeps a single JSON
object as it is (adding ``success``, ``command`` and ``version`` only when
missing), and wraps anything else in the standard envelope.

``SLM_JSON`` never applies to ``slm mcp``: an MCP server launched from an
environment that has it set must still start. ``slm mcp --json`` is explicit.
"""

from __future__ import annotations

import argparse
import builtins
import io
import json
import os
import sys
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, NoReturn

ENV_VAR = "SLM_JSON"
_TRUE = frozenset({"1", "true", "yes", "on"})
#: Commands the environment variable does not switch: their plain form is a protocol.
ENV_EXEMPT = frozenset({"mcp"})
#: Commands whose own JSON is passed through untouched when the caller asked for
#: pretty output (``slm tool NAME --pretty``); otherwise it is completed like any other.
PASS_THROUGH = frozenset({"tool"})
#: Flags that confirm without asking, in the order they are looked for.
CONFIRM_FLAGS = ("--yes", "--force", "--confirm", "--apply", "--execute")

EXIT_OK, EXIT_FAILED, EXIT_USAGE = 0, 1, 2


class PromptBlocked(Exception):
    """A command asked a question while JSON was on; nothing reads an answer."""

    def __init__(self, prompt: str, *, secret: bool = False) -> None:
        super().__init__(prompt)
        self.prompt = prompt
        self.secret = secret


# -- the switch ------------------------------------------------------------------


def env_requests_json(environ: Any = None) -> bool:
    """``SLM_JSON`` set to 1/true/yes/on (any case)."""
    value = (os.environ if environ is None else environ).get(ENV_VAR, "")
    return str(value).strip().lower() in _TRUE


def extract_global_json(argv: list[str]) -> tuple[list[str], bool]:
    """Remove ``--json`` placed before the command; say whether there was one.

    Only the options before the first positional token (the command) are
    looked at, so ``slm tool remember --json '{...}'`` keeps its own meaning.
    """
    out = [argv[0]] if argv else []
    found = False
    i = 1
    while i < len(argv):
        token = argv[i]
        if not token.startswith("-") or token == "-":
            out.extend(argv[i:])
            break
        if token == "--json":
            found = True
        else:
            out.append(token)
        i += 1
    return out, found


def command_word(argv: list[str]) -> str:
    """The first positional token of an argv (the command), or ''."""
    return next((t for t in argv[1:] if not t.startswith("-")), "")


def wants_json(argv: list[str], *, global_flag: bool, environ: Any = None) -> bool:
    """Whether this invocation is a JSON request, before argparse has run."""
    if global_flag:
        return True
    if env_requests_json(environ) and command_word(argv) not in ENV_EXEMPT:
        return True
    return "--json" in argv[1:]


def add_json_flag(parser: argparse.ArgumentParser, help_text: str, *,
                  inherit: bool = False) -> None:
    """Add ``--json`` to a leaf parser.

    ``inherit=True`` is for a leaf under a group that already has its own
    ``--json`` (``slm team --json status``): the leaf's default is
    ``SUPPRESS`` so an absent leaf flag does not reset the group's.
    """
    kwargs: dict[str, Any] = {"action": "store_true", "dest": "json", "help": help_text}
    if inherit:
        kwargs["default"] = argparse.SUPPRESS
    parser.add_argument("--json", **kwargs)


# -- usage errors ----------------------------------------------------------------


def install_json_usage_errors(parser: argparse.ArgumentParser) -> None:
    """Make every parser in the tree answer a usage error with a JSON envelope."""

    def _error(message: str, _p: argparse.ArgumentParser = parser) -> NoReturn:
        from superlocalmemory.cli.json_output import json_print

        command = _p.prog.removeprefix("slm").strip() or "slm"
        json_print(command, error={"code": "usage", "message": message,
                                   "usage": _p.format_usage().strip()})
        sys.exit(EXIT_USAGE)

    parser.error = _error  # type: ignore[method-assign]
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for child in {id(c): c for c in action.choices.values()}.values():
                install_json_usage_errors(child)


def command_path(parser: argparse.ArgumentParser, argv: list[str],
                 args: argparse.Namespace) -> str:
    """``db migrate``, ``remote keys add``...: the command words an argv chose."""
    from superlocalmemory.cli.help_tree import resolve_words

    return " ".join(resolve_words(parser, argv)) or str(getattr(args, "command", "") or "slm")


# -- no prompts ------------------------------------------------------------------


@contextmanager
def non_interactive() -> Iterator[None]:
    """``SLM_NON_INTERACTIVE=1`` for the duration (child processes inherit it)."""
    saved = os.environ.get("SLM_NON_INTERACTIVE")
    os.environ["SLM_NON_INTERACTIVE"] = "1"
    try:
        yield
    finally:
        if saved is None:
            os.environ.pop("SLM_NON_INTERACTIVE", None)
        else:
            os.environ["SLM_NON_INTERACTIVE"] = saved


@contextmanager
def no_prompts() -> Iterator[None]:
    """``input()`` and ``getpass`` raise :class:`PromptBlocked` instead of waiting.

    Reading piped data from ``sys.stdin`` (``slm observe``, ``slm tool NAME -``)
    still works; only the prompt functions are replaced.
    """
    import getpass

    real_input, real_getpass = builtins.input, getpass.getpass

    def _input(prompt: object = "") -> str:
        raise PromptBlocked(str(prompt))

    def _getpass(prompt: str = "Password: ", stream: Any = None) -> str:
        raise PromptBlocked(str(prompt), secret=True)

    builtins.input = _input  # type: ignore[assignment]
    getpass.getpass = _getpass  # type: ignore[assignment]
    try:
        yield
    finally:
        builtins.input = real_input  # type: ignore[assignment]
        getpass.getpass = real_getpass  # type: ignore[assignment]


def confirm_flag_of(parser: argparse.ArgumentParser | None) -> str | None:
    """The flag that answers a command's confirmation question, if it has one."""
    if parser is None:
        return None
    options = {o for a in parser._actions for o in a.option_strings}
    return next((f for f in CONFIRM_FLAGS if f in options), None)


def _looks_like_confirmation(prompt: str) -> bool:
    text = prompt.lower()
    return any(m in text for m in ("[y/n]", "y/n", "type yes", "to continue", "to go on",
                                   "proceed", "are you sure"))


def blocked_error(blocked: PromptBlocked, parser: argparse.ArgumentParser | None) -> dict:
    """The error body for a question nobody can answer."""
    prompt = " ".join(blocked.prompt.split())
    flag = confirm_flag_of(parser)
    if not blocked.secret and _looks_like_confirmation(prompt):
        message = (f"This command asks before it changes anything ({prompt!r}). "
                   + (f"Pass {flag} to confirm." if flag else
                      "It has no flag that confirms; run it in a terminal."))
        return {"code": "confirmation_required", "message": message, "flag": flag,
                "prompt": prompt}
    return {"code": "input_required", "message": (
        f"This command asked for input ({prompt!r}), and JSON mode never prompts. "
        "Pass the answer as an option; see --help."), "prompt": prompt,
        "missing_flags": []}


# -- capture and normalise ---------------------------------------------------------


@contextmanager
def _captured_stdout() -> Iterator[Callable[[], str]]:
    """Everything printed to stdout, by Python or by a child process.

    The file descriptor is redirected only when ``sys.stdout`` is the process's
    own (a test's capture object is left alone, and so is its fd).
    """
    real = sys.stdout
    buffer = io.StringIO()
    fd_level = real is sys.__stdout__ and _has_fd(real)
    saved_fd = tmp = None
    if fd_level:
        real.flush()
        tmp = tempfile.TemporaryFile(mode="w+b")
        saved_fd = os.dup(1)
        os.dup2(tmp.fileno(), 1)
    sys.stdout = buffer
    text: list[str] = []
    try:
        yield lambda: "".join(text)
    finally:
        sys.stdout = real
        fd_text = ""
        if fd_level and tmp is not None and saved_fd is not None:
            os.dup2(saved_fd, 1)
            os.close(saved_fd)
            tmp.seek(0)
            fd_text = tmp.read().decode("utf-8", errors="replace")
            tmp.close()
        text.append(fd_text + buffer.getvalue())


class _Tee(io.TextIOBase):
    """Writes go to stderr as always, and are kept for an error message."""

    def __init__(self, real: Any) -> None:
        super().__init__()
        self._real = real
        self.kept = io.StringIO()

    def write(self, text: str) -> int:
        self.kept.write(text)
        return self._real.write(text)

    def flush(self) -> None:
        self._real.flush()

    def isatty(self) -> bool:
        return bool(getattr(self._real, "isatty", lambda: False)())

    def fileno(self) -> int:
        return self._real.fileno()


@contextmanager
def _kept_stderr() -> Iterator[_Tee]:
    real = sys.stderr
    tee = _Tee(real)
    sys.stderr = tee
    try:
        yield tee
    finally:
        sys.stderr = real


def _has_fd(stream: Any) -> bool:
    try:
        stream.fileno()
        return True
    except (AttributeError, OSError, ValueError, io.UnsupportedOperation):
        return False


def _version() -> str:
    from superlocalmemory.cli.json_output import _get_version

    return _get_version()


def _emit(document: Any) -> None:
    sys.stdout.write(json.dumps(document, separators=(",", ":"), default=str,
                                ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _last_line(text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def _parse_single(text: str) -> tuple[bool, Any]:
    stripped = text.strip()
    if not stripped:
        return False, None
    try:
        return True, json.loads(stripped)
    except ValueError:
        return False, None


def _split_trailing_json(text: str) -> tuple[str, Any] | None:
    """Text lines followed by one JSON object on the last line(s)."""
    lines = text.rstrip().splitlines()
    for start in range(len(lines) - 1, 0, -1):
        if not lines[start].lstrip().startswith("{"):
            continue
        ok, value = _parse_single("\n".join(lines[start:]))
        if ok and isinstance(value, dict):
            return "\n".join(lines[:start]), value
    return None


def normalize(command: str, rc: int, text: str, stderr_text: str = "") -> Any:
    """The one document for what a command printed and how it exited."""
    ok, value = _parse_single(text)
    if not ok:
        split = _split_trailing_json(text)
        if split is not None:
            stray, value = split
            if stray.strip():
                print(stray, file=sys.stderr)
            ok = True
    if ok and isinstance(value, dict):
        return _complete(command, rc, value)
    envelope: dict[str, Any] = {"success": rc == 0, "command": command, "version": _version()}
    if ok:  # a JSON value that is not an object (a list, a number)
        if rc == 0:
            envelope["data"] = {"result": value}
        else:
            envelope["error"] = {"code": "command_failed", "message": f"exit code {rc}",
                                 "exit_code": rc, "result": value}
        return envelope
    if rc == 0:
        envelope["data"] = {"output": text.rstrip("\n")} if text.strip() else {}
    else:
        envelope["error"] = {"code": "command_failed" if rc != EXIT_USAGE else "usage",
                             "message": _last_line(text) or _last_line(stderr_text)
                             or f"The command exited with code {rc}.",
                             "exit_code": rc, "output": text.rstrip("\n")}
    return envelope


def _complete(command: str, rc: int, value: dict) -> dict:
    """Add what the contract needs to an existing JSON object, never renaming a key."""
    out = dict(value)
    out.setdefault("success", rc == 0)
    out.setdefault("command", command)
    out.setdefault("version", _version())
    error = out.get("error")
    if out.get("success") is False and isinstance(error, dict) and "code" not in error:
        out["error"] = {"code": "command_failed", **error}
    return out


def run_json(command: str, fn: Callable[[], Any],
             parser: argparse.ArgumentParser | None = None) -> None:
    """Run a command with JSON on and print exactly one document.

    Returns on success (exit 0, as a plain command returns); exits with the
    command's code otherwise.
    """
    rc = EXIT_OK
    failure: dict | None = None
    with _kept_stderr() as kept, _captured_stdout() as captured, no_prompts(), non_interactive():
        try:
            fn()
        except SystemExit as exc:
            rc = _exit_code(exc.code)
        except PromptBlocked as blocked:
            failure, rc = blocked_error(blocked, parser), EXIT_FAILED
        except KeyboardInterrupt:
            failure, rc = {"code": "interrupted", "message": "Interrupted."}, 130
        except Exception as exc:  # noqa: BLE001 - one JSON document whatever happens
            import traceback

            traceback.print_exc(file=sys.stderr)
            failure = {"code": "internal_error", "message": str(exc) or type(exc).__name__,
                       "type": type(exc).__name__}
            rc = EXIT_FAILED
    text = captured()
    if (command.split(" ")[0] in PASS_THROUGH and "--pretty" in sys.argv
            and failure is None and text.strip()):
        sys.stdout.write(text if text.endswith("\n") else text + "\n")
        sys.stdout.flush()
        if rc:
            sys.exit(rc)
        return
    if failure is not None:
        if text.strip():
            print(text.rstrip("\n"), file=sys.stderr)
        _emit({"success": False, "command": command, "version": _version(), "error": failure})
    else:
        _emit(normalize(command, rc, text, kept.kept.getvalue()))
    if rc:
        sys.exit(rc)


def _exit_code(code: Any) -> int:
    if code is None:
        return EXIT_OK
    if isinstance(code, int):
        return code
    print(code, file=sys.stderr)  # sys.exit("message") prints it and exits 1
    return EXIT_FAILED


def fail(command: str, code: str, message: str, *, exit_code: int = EXIT_FAILED,
         **fields: Any) -> NoReturn:
    """Print an error envelope and exit (for JSON branches of commands)."""
    from superlocalmemory.cli.json_output import json_print

    json_print(command, error={"code": code, "message": message, **fields})
    sys.exit(exit_code)


__all__ = ["ENV_VAR", "PromptBlocked", "add_json_flag", "command_path", "command_word",
           "env_requests_json", "extract_global_json", "fail", "install_json_usage_errors",
           "no_prompts", "normalize", "run_json", "wants_json"]
