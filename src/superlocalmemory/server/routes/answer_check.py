# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Answer-check settings — the local Laya judge, the hosted Jev judge, or off.

Over 75% of SLM's users are non-technical, so this is set up from the
dashboard only: there is no config file they are expected to hand-edit.
Exactly one backend runs at a time, by construction — ``sufficiency_judge``
is a single setting, so choosing one always replaces whatever was chosen
before. Jev is opt-in and needs both a stored key and explicit consent,
because choosing it sends the question and the top memories to a hosted
provider; nothing here ever resolves "auto" to "jev" on its own. "auto" is
the never-chosen starting state (the on-device check when a verified install
exists, otherwise off); the dashboard offers the three explicit choices.

Who may change it. Every mutation needs a credential the product issues —
the install token the dashboard sends on every change, the daemon
capability, or an API key — even from this machine: these settings decide
where memory text is sent and what is billed, so "a process that reached the
port" is not enough. On top of that, authority over the whole installation
(``_require_manage``: the machine owner or an administrator of every
workspace), since the consent sends memories of every workspace out.
GET is a plain read, open to anyone who can reach the dashboard, and it
changes nothing and starts nothing.

Where the settings live. One file for every mode
(core/answer_check_state.py), so a mode switch can neither restore a
withdrawn consent nor reset a choice, and these routes always read what they
wrote. A change that withdraws something returns success only after the
running check is verifiably holding to it (answer_check_support.enforce).

Nothing slow runs on the event loop. Every handler is a plain ``def`` (run on
a worker thread). Setting up the model and checking an existing install are
background jobs the dashboard polls; their results are saved and switched on
by the job itself, so it happens even when nobody is watching the page.

Failures are logged server-side and reported as a generic error — never a
stack trace, and never a secret.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StrictBool

from superlocalmemory.core import judge_keys, judge_selection, laya_runtime
from superlocalmemory.retrieval import jev_judge, sufficiency
from superlocalmemory.server.routes import answer_check_support as support
from superlocalmemory.server.routes.answer_calibration import register as _register_calibration

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v3/answer-check", tags=["answer-check"])

#: The dashboard's three choices. "auto" is the starting state, not a choice.
_MODE_PATTERN = "^(laya|jev|off)$"
_PROVIDER_PATTERN = "^(" + "|".join(judge_keys.PROVIDERS) + ")$"
_APPLIABLE_MODES = (judge_selection.MODE_LAYA, judge_selection.MODE_AUTO)

#: Each Jev connection test is one billed request — never more than one per
#: provider every ten seconds. Module-level because the daemon is one process;
#: a lock guards the read-then-write against concurrent requests.
_JEV_TEST_RATE_LIMIT_S = 10.0
_last_jev_test: dict[str, float] = {}
_jev_test_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Request models  (extra="forbid" -> 422 on an unknown key)
# ---------------------------------------------------------------------------


class ModeUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: str = Field(..., pattern=_MODE_PATTERN)


class LayaAdoptRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    python: str = Field(..., min_length=1, max_length=4096)
    hf_home: str = Field("", max_length=4096)
    model_path: str = Field("", max_length=4096)


class JevKeyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str = Field(..., pattern=_PROVIDER_PATTERN)
    key: str = Field(..., min_length=1, max_length=4096)


class JevKeyDelete(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str = Field(..., pattern=_PROVIDER_PATTERN)


class JevTestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str = Field(..., pattern=_PROVIDER_PATTERN)


class JevConsentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    accepted: StrictBool
    provider: str = Field(..., pattern=_PROVIDER_PATTERN)


class JevRerankRequest(BaseModel):
    """Turn "Also use Jev to reorder results" on or off.

    ``accepted`` is the person accepting its own notice, which names
    ``provider`` — so turning it on for a provider other than the one in use
    is refused rather than quietly sending the text somewhere else.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: StrictBool
    accepted: StrictBool
    provider: str = Field(..., pattern=_PROVIDER_PATTERN)


class JevLadderRequest(BaseModel):
    """Let the background judges ask Jev about pairs of memories (consent
    ``judge_ladder_hosted``). ``accepted`` is the person accepting the notice,
    which names ``provider``."""

    model_config = ConfigDict(extra="forbid")

    accepted: StrictBool
    provider: str = Field(..., pattern=_PROVIDER_PATTERN)


LADDER_PURPOSE = "judge_ladder_hosted"


# ---------------------------------------------------------------------------
# Gates and small helpers
# ---------------------------------------------------------------------------


def _internal_error() -> JSONResponse:
    """Log the full traceback server-side; never leak internals to the UI."""
    logger.exception("answer_check: request failed")
    return JSONResponse({"error": "Internal server error"}, status_code=500)


def _error(message: str, status_code: int = 400) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status_code)


def _not_confirmed() -> JSONResponse:
    return _error(support.NOT_CONFIRMED, 500)


def _require_credential(request: Request) -> None:
    """An issued credential, even from loopback (raises 403 without one)."""
    from superlocalmemory.server.write_identity import require_write_actor

    require_write_actor(request, getattr(request.app.state, "daemon_descriptor", None),
                        actor_kind="answer-check")


def _require_manage(request: Request) -> None:
    """One explicit authorization boundary, same pattern as other v3 routes."""
    from superlocalmemory.server.rbac_enforce import require_global_manage

    require_global_manage(request)


def _gate(request: Request) -> None:
    _require_credential(request)
    _require_manage(request)


def _test_jev_connection(provider: str) -> tuple[bool, str]:
    """Read the stored key internally, then one billed probe request.

    Loads nothing itself when there is no key to test — the route that calls
    this never touches the key directly, so it can never echo or log it.
    """
    try:
        key = judge_keys.JudgeKeyStore().load(provider)
    except judge_keys.JudgeKeyStoreError:
        return False, support.KEY_PROBLEM
    if not key:
        return False, "No key saved for this provider yet."
    return jev_judge.check_connection(provider, key, timeout_s=10.0)


def _job_hooks(app_state: Any):
    """(before_verify, on_done) for a background Laya job.

    before_verify stops the running on-device check, so the job's check never
    loads a second model next to it. on_done saves a passing install's paths,
    switches the check on when the mode wants it, and restores whatever was
    stopped — on the job's own thread, whether or not anyone is watching.
    """
    paused = {"laya": False}

    def before_verify() -> None:
        with support.MUTATION_LOCK:
            paused["laya"] = support.detach_laya(app_state)

    def on_done(status: Any) -> None:
        with support.MUTATION_LOCK:
            ready = status.state == laya_runtime.STATE_READY
            if ready:
                retrieval = support.save(lambda v: {**v, **support.laya_paths(status)})
            else:
                retrieval = support.effective_retrieval()
            wanted = ready and retrieval.sufficiency_judge in _APPLIABLE_MODES
            if wanted or paused["laya"]:
                support.attach(app_state, retrieval)
            support.enforce(app_state, retrieval)

    return before_verify, on_done


def _a_laya_job_is_running() -> bool:
    return (laya_runtime.LayaAdoptJob.instance().running
            or laya_runtime.LayaTestJob.instance().running
            or laya_runtime.LayaInstallJob.instance().running)


# ---------------------------------------------------------------------------
# GET  /api/v3/answer-check
# ---------------------------------------------------------------------------


@router.get("")
def get_answer_check(request: Request):
    try:
        return support.build_status(request.app.state)
    except Exception:
        return _internal_error()


# ---------------------------------------------------------------------------
# POST /api/v3/answer-check/mode
# ---------------------------------------------------------------------------


def _mode_refusal(mode: str, retrieval: Any) -> JSONResponse | None:
    if mode == "laya":
        if laya_runtime.detect(retrieval).state != laya_runtime.STATE_READY:
            return _error("Set up the on-device model first, in the Laya panel.")
    elif mode == "jev":
        if retrieval.sufficiency_jev_consent is not True:
            return _error("Accept the data-sharing notice before turning Jev on.")
        key = support.key_state(retrieval.sufficiency_jev_provider)
        if key["key_problem"]:
            return _error(key["key_problem"])
        if not key["has_key"]:
            return _error("Add a Jev key before turning it on.")
    return None


@router.post("/mode")
def post_mode(request: Request, body: ModeUpdate):
    _gate(request)
    try:
        app_state = request.app.state
        with support.MUTATION_LOCK:
            refusal = _mode_refusal(body.mode, support.effective_retrieval())
            if refusal is not None:
                return refusal
            mode = body.mode
            # Reordering is part of Jev: it runs only while Jev is chosen, and
            # the person's choice is remembered for when they come back to it.
            retrieval = support.save(lambda v: {**v, "sufficiency_judge": mode})
            support.attach(app_state, retrieval)
            if not support.enforce(app_state, retrieval):
                return _not_confirmed()
            status_body = support.build_status(app_state)
        status_body["message"] = support.switched_message(mode, retrieval)
        return status_body
    except Exception:
        return _internal_error()


# ---------------------------------------------------------------------------
# POST /api/v3/answer-check/laya/setup  (background job, polled)
# ---------------------------------------------------------------------------


@router.post("/laya/setup")
def post_laya_setup(request: Request):
    _gate(request)
    try:
        if not sufficiency.laya_supported():
            return _error("Needs a Mac with Apple Silicon.")
        with support.MUTATION_LOCK:
            if laya_runtime.LayaAdoptJob.instance().running:
                return _error("An existing install is being checked. Wait for it to finish.",
                              409)
            job = laya_runtime.LayaInstallJob.instance()
            before_verify, on_done = _job_hooks(request.app.state)
            if not job.start(on_done=on_done, before_verify=before_verify):
                return _error("A setup is already running.", 409)
            return job.status().to_dict()
    except Exception:
        return _internal_error()


# ---------------------------------------------------------------------------
# POST /api/v3/answer-check/laya/adopt  (background job, polled)
# ---------------------------------------------------------------------------


@router.post("/laya/adopt")
def post_laya_adopt(request: Request, body: LayaAdoptRequest):
    _gate(request)
    try:
        if not sufficiency.laya_supported():
            return _error("Needs a Mac with Apple Silicon.")
        refused = laya_runtime.check_interpreter(body.python)
        if refused:
            return _error(refused)
        with support.MUTATION_LOCK:
            if laya_runtime.LayaInstallJob.instance().status().state == \
                    laya_runtime.STATE_INSTALLING:
                return _error("A setup is already running. Wait for it to finish.", 409)
            job = laya_runtime.LayaAdoptJob.instance()
            before_verify, on_done = _job_hooks(request.app.state)
            started = job.start(body.python, body.hf_home, body.model_path,
                                on_done=on_done, before_verify=before_verify)
            if not started:
                return _error("That install is already being checked.", 409)
            return JSONResponse({"status": job.status().to_dict(), "applied": False,
                                 "running": True}, status_code=202)
    except Exception:
        return _internal_error()


# ---------------------------------------------------------------------------
# POST /api/v3/answer-check/laya/remove
# ---------------------------------------------------------------------------


@router.post("/laya/remove")
def post_laya_remove(request: Request):
    """Stop using the on-device check. SLM's own install (finished or not) is
    deleted; an install made elsewhere is only forgotten — its files stay."""
    _gate(request)
    try:
        app_state = request.app.state
        with support.MUTATION_LOCK:
            if _a_laya_job_is_running():
                return _error("Wait for the current setup or check to finish.", 409)
            before = laya_runtime.detect(support.effective_retrieval())
            stopped = support.detach_laya(app_state)  # before its files go
            result = laya_runtime.remove()
            if result.state != laya_runtime.STATE_NOT_INSTALLED and \
                    result.error != "No managed install to remove.":
                return _error(result.error)
            external = not before.managed and before.state != laya_runtime.STATE_NOT_INSTALLED
            if external:
                laya_runtime.forget_adopted()
            retrieval = support.save(
                lambda v: support.forget_managed_install(v, every_path=external))
            for job in (laya_runtime.LayaInstallJob, laya_runtime.LayaAdoptJob,
                        laya_runtime.LayaTestJob):
                forget = getattr(job.instance(), "forget", None)
                if callable(forget):
                    forget()
            support.forget_test("laya")
            if stopped and retrieval.sufficiency_judge != judge_selection.MODE_OFF:
                support.attach(app_state, retrieval)  # e.g. an adopted install kept
            support.enforce(app_state, retrieval)
            return laya_runtime.detect(retrieval).to_dict()
    except Exception:
        return _internal_error()


# ---------------------------------------------------------------------------
# POST / DELETE  /api/v3/answer-check/jev/key
# ---------------------------------------------------------------------------


@router.post("/jev/key")
def post_jev_key(request: Request, body: JevKeyRequest):
    _gate(request)
    store = judge_keys.JudgeKeyStore()
    try:
        store.set_key(body.provider, body.key)
    except ValueError as exc:
        return _error(str(exc))
    except judge_keys.JudgeKeyStoreError:
        return _error(support.KEY_PROBLEM)
    except Exception:
        # SEC: never interpolate the key itself into a log line.
        logger.exception("answer_check: saving the %s key failed", body.provider)
        return _internal_error()
    support.forget_test("jev")  # the last test was of another key
    try:
        with support.MUTATION_LOCK:
            # A key saved while Jev is the chosen check takes effect now, not
            # at the next restart.
            retrieval = support.effective_retrieval()
            if (retrieval.sufficiency_judge == judge_selection.MODE_JEV
                    and retrieval.sufficiency_jev_provider == body.provider):
                support.attach(request.app.state, retrieval)
            support.enforce(request.app.state, retrieval)
    except Exception:
        return _internal_error()
    key = support.key_state(body.provider)
    return {"has_key": key["has_key"], "key_hint": key["key_hint"]}


@router.delete("/jev/key")
def delete_jev_key(request: Request, body: JevKeyDelete):
    _gate(request)
    try:
        app_state = request.app.state
        with support.MUTATION_LOCK:
            judge_keys.JudgeKeyStore().clear(body.provider)
            was_jev = (support.effective_retrieval().sufficiency_judge == judge_selection.MODE_JEV
                       or judge_selection.backend_of(support.live_judge(app_state))
                       == judge_selection.MODE_JEV)
            retrieval = support.save(lambda v: {
                **support.without_rerank(v),
                **({"sufficiency_judge": judge_selection.MODE_OFF}
                   if v["sufficiency_judge"] == judge_selection.MODE_JEV else {})})
            if was_jev:
                support.attach(app_state, retrieval)
            if not support.enforce(app_state, retrieval):
                return _not_confirmed()
            return {"has_key": support.key_state(body.provider)["has_key"],
                    "mode": retrieval.sufficiency_judge}
    except Exception:
        return _internal_error()


# ---------------------------------------------------------------------------
# POST /api/v3/answer-check/jev/test
# ---------------------------------------------------------------------------


@router.post("/jev/test")
def post_jev_test(request: Request, body: JevTestRequest):
    _gate(request)
    provider = body.provider
    with _jev_test_lock:
        now = time.monotonic()
        last = _last_jev_test.get(provider, 0.0)
        if now - last < _JEV_TEST_RATE_LIMIT_S:
            return _error("Wait a few seconds before testing again.", 429)
        _last_jev_test[provider] = now
    try:
        started = time.monotonic()
        ok, message = _test_jev_connection(provider)
        seconds = time.monotonic() - started
        result = support.record_test("jev", ok, message, seconds)
        return {"ok": ok, "message": message, "result": result}
    except Exception:
        return _internal_error()


# ---------------------------------------------------------------------------
# POST /api/v3/answer-check/jev/consent
# ---------------------------------------------------------------------------


def _consent_change(accepted: bool, provider: str):
    def change(values: dict) -> dict:
        # The reordering notice named a provider; a withdrawal, or text now
        # going to another provider, ends it.
        out = dict(values)
        if not accepted or values.get("sufficiency_jev_provider") != provider:
            out = support.without_rerank(out)
        if not accepted and values.get("sufficiency_judge") == judge_selection.MODE_JEV:
            out["sufficiency_judge"] = judge_selection.MODE_OFF
        return {**out, "sufficiency_jev_consent": accepted,
                "sufficiency_jev_provider": provider}
    return change


@router.post("/jev/consent")
def post_jev_consent(request: Request, body: JevConsentRequest):
    _gate(request)
    try:
        app_state = request.app.state
        with support.MUTATION_LOCK:
            before = support.effective_retrieval()
            was_jev = (before.sufficiency_judge == judge_selection.MODE_JEV
                       or judge_selection.backend_of(support.live_judge(app_state))
                       == judge_selection.MODE_JEV)
            # A receipt (who, when, which notice) before the setting changes (OPS-19).
            from superlocalmemory.server.consent_receipts import record_change

            record_change(request, "answer_check", before=before,
                          accepted=bool(body.accepted), provider=body.provider)
            if not body.accepted and before.sufficiency_jev_rerank_consent is True:
                record_change(request, "rerank", before=before, accepted=False,
                              provider=body.provider)
            retrieval = support.save(_consent_change(bool(body.accepted), body.provider))
            _withdraw_ladder_unless(request, accepted=bool(body.accepted), provider=body.provider)
            if was_jev:
                # Rebuilt either way, so the running check uses the provider and
                # the reordering setting just saved — not the ones before.
                support.attach(app_state, retrieval)
            if not support.enforce(app_state, retrieval):
                return _not_confirmed()
            return support.build_status(app_state)
    except HTTPException:
        raise
    except Exception:
        return _internal_error()


# ---------------------------------------------------------------------------
# POST /api/v3/answer-check/jev/rerank
# ---------------------------------------------------------------------------


def _rerank_refusal(retrieval: Any, body: JevRerankRequest) -> JSONResponse | None:
    """Why reordering cannot be turned on now, or None when it can."""
    if not body.accepted:
        return _error("Accept the notice before turning reordering on.")
    if body.provider != retrieval.sufficiency_jev_provider:
        return _error("The provider changed. Read the notice again, then turn this on.", 409)
    if retrieval.sufficiency_judge != judge_selection.MODE_JEV:
        return _error("Turn on the online answer check with Jev first.")
    if retrieval.sufficiency_jev_consent is not True:
        return _error("Accept the data-sharing notice for Jev first.")
    key = support.key_state(retrieval.sufficiency_jev_provider)
    if key["key_problem"]:
        return _error(key["key_problem"])
    if not key["has_key"]:
        return _error("Add a Jev key first.")
    return None


def _rerank_change(enabled: bool, provider: str):
    def change(values: dict) -> dict:
        if not enabled:
            return support.without_rerank(values)
        # Checked again under the settings lock: a withdrawal or a switch that
        # landed after the checks above wins over this request.
        still_allowed = (values.get("sufficiency_judge") == judge_selection.MODE_JEV
                         and values.get("sufficiency_jev_consent") is True
                         and values.get("sufficiency_jev_provider") == provider)
        if not still_allowed:
            return values
        return {**values, "sufficiency_jev_rerank": True,
                "sufficiency_jev_rerank_consent": True}
    return change


@router.post("/jev/rerank")
def post_jev_rerank(request: Request, body: JevRerankRequest):
    _gate(request)
    try:
        app_state = request.app.state
        with support.MUTATION_LOCK:
            before = support.effective_retrieval()
            if body.enabled:
                refusal = _rerank_refusal(before, body)
                if refusal is not None:
                    return refusal
            from superlocalmemory.server.consent_receipts import record_change, rerank_count

            record_change(request, "rerank", before=before, accepted=bool(body.enabled),
                          provider=body.provider,
                          k=rerank_count(before))
            retrieval = support.save(_rerank_change(bool(body.enabled), body.provider))
            if body.enabled and judge_selection.jev_rerank_k(retrieval) <= 0:
                return _error("The answer check settings just changed. Try again.", 409)
            if (retrieval.sufficiency_judge == judge_selection.MODE_JEV
                    or judge_selection.backend_of(support.live_judge(app_state))
                    == judge_selection.MODE_JEV):
                support.attach(app_state, retrieval)
            if not support.enforce(app_state, retrieval):
                return _not_confirmed()
            status_body = support.build_status(app_state)
        status_body["message"] = (
            "Jev will also reorder results." if body.enabled
            else "Jev will no longer reorder results.")
        return status_body
    except HTTPException:
        raise
    except Exception:
        return _internal_error()

# ---------------------------------------------------------------------------
# POST /api/v3/answer-check/jev/ladder
# ---------------------------------------------------------------------------


def _ladder_home():
    from superlocalmemory.infra.data_root import canonical_data_root

    return canonical_data_root()


def _withdraw_ladder_unless(request: Request, *, accepted: bool, provider: str) -> None:
    """The background judges' consent follows the answer check's: withdrawn with
    it, and withdrawn when the answer check moves to another provider."""
    from superlocalmemory.core import consent_store
    from superlocalmemory.server.consent_receipts import _actor

    now = consent_store.state(_ladder_home(), LADDER_PURPOSE)
    if now["given"] and (not accepted or now["provider"] != provider):
        try:
            consent_store.change(_ladder_home(), LADDER_PURPOSE, False, via="dashboard",
                                 actor=_actor(request), provider=str(now["provider"] or provider))
        except consent_store.ConsentNotRecorded as exc:
            raise HTTPException(503, detail="The consent could not be recorded, so nothing "
                                            "was changed. Try again.") from exc


@router.post("/jev/ladder")
def post_jev_ladder(request: Request, body: JevLadderRequest):
    """Give or withdraw consent for the background judges to ask Jev (W2, D-20)."""
    _gate(request)
    from superlocalmemory.core import consent_store
    from superlocalmemory.server.consent_receipts import _actor

    try:
        app_state = request.app.state
        with support.MUTATION_LOCK:
            retrieval = support.effective_retrieval()
            if body.accepted:
                if body.provider != retrieval.sufficiency_jev_provider:
                    return _error("The provider changed. Read the notice again, then agree.", 409)
                if (retrieval.sufficiency_judge != judge_selection.MODE_JEV
                        or retrieval.sufficiency_jev_consent is not True):
                    return _error("Turn on the online answer check with Jev first: the "
                                  "background judges use the same connection.")
            try:
                consent_store.change(_ladder_home(), LADDER_PURPOSE, bool(body.accepted),
                                     via="dashboard", actor=_actor(request),
                                     provider=body.provider)
            except consent_store.ConsentNotRecorded:
                return _error("The consent could not be recorded, so nothing was changed. "
                              "Try again.", 503)
            status_body = support.build_status(app_state)
        status_body["message"] = (
            "The background judges may now ask Jev." if body.accepted
            else "The background judges will no longer ask Jev.")
        return status_body
    except HTTPException:
        raise
    except Exception:
        return _internal_error()


# Reuse the same credential + MANAGE boundary for store calibration.

_register_calibration(router, _gate)
