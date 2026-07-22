# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Skill Evolver — orchestrates the full evolution pipeline.

Pipeline: Trigger → Screen → LLM Confirm → Mutate → Blind Verify → Persist

Performance: NEVER runs on recall/remember hot path. Only during
consolidation (every 6h) or explicit trigger. Zero impact on latency.

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

import dataclasses
import difflib
import hashlib
import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from superlocalmemory.infra.data_root import canonical_data_root
from superlocalmemory.evolution.types import (
    EvolutionCandidate,
    EvolutionRecord,
    EvolutionStatus,
    EvolutionType,
    TriggerType,
)
from superlocalmemory.evolution.evolution_store import EvolutionStore
from superlocalmemory.evolution.triggers import (
    DegradationTrigger,
    HealthCheckTrigger,
    PostSessionTrigger,
)
from superlocalmemory.evolution import mutation_generator as mutgen
from superlocalmemory.evolution import blind_verifier as verifier
from superlocalmemory.evolution.budget import (
    BudgetExhausted,
    EvolutionBudget,
)
from superlocalmemory.evolution.llm_dispatch import _dispatch_llm
from superlocalmemory.evolution.model_selection import (
    _MODEL_ALIASES,
    _resolve_model_alias,
    resolve_evolution_models,
)

logger = logging.getLogger(__name__)

EVOLVED_SKILLS_DIR = Path.home() / ".claude" / "skills" / "evolved"

# Model aliasing + per-step selection live in ``model_selection`` — a single
# source of truth shared with the config layer. ``_MODEL_ALIASES`` and
# ``_resolve_model_alias`` are imported above and re-exported here for
# callers/tests that still reference them via ``skill_evolver``.
__all__ = ["SkillEvolver", "detect_backend", "_resolve_model_alias"]


def _ollama_running() -> bool:
    """Return True if a local Ollama daemon answers on the default port."""
    try:
        import urllib.request
        req = urllib.request.Request(
            "http://127.0.0.1:11434/api/tags", method="GET",
        )
        with urllib.request.urlopen(req, timeout=2):  # noqa: S310
            return True
    except Exception:
        return False


def detect_backend() -> str:
    """Auto-detect best available LLM backend.

    Priority: claude CLI → Ollama → API key → none
    """
    import shutil

    # 1. Claude CLI available?
    if shutil.which("claude"):
        return "claude"

    # 2. Ollama running?
    if _ollama_running():
        return "ollama"

    # 3. API key set?
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"

    return "none"


class SkillEvolver:
    """Main orchestrator for skill evolution.

    Call `run_consolidation_cycle()` from the consolidation pipeline.
    Call `run_post_session()` from the Stop hook.

    Respects EvolutionConfig.enabled — does nothing if disabled.
    Auto-detects LLM backend: claude CLI → Ollama → API → none.
    """

    def __init__(
        self,
        db_path: str | Path,
        config: object | None = None,
        *,
        profile_id: str = "default",
        budget: EvolutionBudget | None = None,
    ):
        self._db_path = str(db_path)
        self._store = EvolutionStore(db_path)
        self._degradation = DegradationTrigger(db_path)
        self._health = HealthCheckTrigger(db_path, profile_id=profile_id)
        self._config = config
        self._backend: str | None = None
        # ResolvedModels, resolved lazily & cached via _get_models().
        self._models = None
        self._profile_id = profile_id
        self._current_cycle_id: str | None = None

        # SB-3: SkillEvolver always holds a budget. Default one is rooted
        # at ~/.superlocalmemory so production callers pick it up
        # automatically. Tests inject their own.
        if budget is None:
            slm_home = canonical_data_root()
            budget = EvolutionBudget(
                profile_id=profile_id,
                learning_db=Path(self._db_path),
                lock_dir=slm_home,
            )
        self._budget = budget

    def _is_enabled(self) -> bool:
        """Check if evolution is enabled in config."""
        if self._config and hasattr(self._config, "evolution"):
            return self._config.evolution.enabled
        return False

    def _get_backend(self) -> str:
        """Get or detect the LLM backend."""
        if self._backend:
            return self._backend

        configured = "auto"
        if self._config and hasattr(self._config, "evolution"):
            configured = self._config.evolution.backend

        if configured == "auto":
            self._backend = detect_backend()
        else:
            self._backend = configured

        logger.info("Evolution backend: %s", self._backend)
        return self._backend

    def _get_models(self):
        """Resolve per-step evolution models (cached).

        Defaults to the lowest-cost model for the active backend and keeps
        the blind verifier independent of the generator where possible
        (see ``model_selection.resolve_evolution_models``). Falls back to
        stock ``EvolutionConfig`` defaults when no config is attached.
        """
        if self._models is not None:
            return self._models
        backend = self._get_backend()
        evo_cfg = getattr(self._config, "evolution", None)
        if evo_cfg is None:
            from superlocalmemory.core.config import EvolutionConfig
            evo_cfg = EvolutionConfig()
        self._models = resolve_evolution_models(
            evo_cfg, backend, ollama_available=_ollama_running(),
        )
        return self._models

    def run_consolidation_cycle(self, profile_id: str = "default") -> dict:
        """Run during consolidation. Checks triggers 2 and 3.

        Wrapped in ``self._budget.cycle()`` — honours the
        30min/10-LLM-calls/3-cycles-per-day caps (SB-3). If the budget
        is exhausted, returns ``{"aborted": True}`` rather than raising
        so the consolidation worker can continue cleanly.
        """
        if not self._is_enabled():
            return {"enabled": False, "message": "Evolution disabled. Enable via: slm config set evolution.enabled true"}

        backend = self._get_backend()
        if backend == "none":
            return {"enabled": True, "backend": "none",
                    "message": "No LLM backend available. Install Claude Code, Ollama, or set an API key."}

        try:
            with self._budget.cycle() as _b:
                self._current_cycle_id = None  # budget already recorded one
                return self._run_consolidation_body(profile_id, backend)
        except BudgetExhausted as exc:
            logger.warning(
                "evolution cycle aborted: budget exhausted [%s]",
                getattr(exc, "dimension", "?"),
            )
            return {
                "enabled": True,
                "backend": backend,
                "aborted": True,
                "budget_exhausted": True,
                "dimension": getattr(exc, "dimension", None),
                "candidates": 0, "evolved": 0, "rejected": 0, "skipped": 0,
            }
        finally:
            self._current_cycle_id = None

    def _run_consolidation_body(
        self, profile_id: str, backend: str,
    ) -> dict:
        """Inner consolidation loop — runs under an open budget cycle."""
        self._store.reset_cycle(profile_id)
        results = {"candidates": 0, "evolved": 0, "rejected": 0, "skipped": 0, "backend": backend}

        # Prune recovered skills from anti-loop tracking
        active_degraded = self._degradation.get_active_degraded(profile_id)
        self._store.prune_recovered(active_degraded)

        # Trigger 2: Degradation
        candidates = self._degradation.scan(profile_id)
        # Trigger 3: Health check (runs every Nth cycle)
        candidates.extend(self._health.scan(profile_id))

        results["candidates"] = len(candidates)

        for candidate in candidates:
            if not self._store.can_evolve(profile_id):
                results["skipped"] += len(candidates) - results["evolved"] - results["rejected"]
                break

            outcome = self._process_candidate(candidate, profile_id)
            if outcome == "evolved":
                results["evolved"] += 1
            elif outcome == "rejected":
                results["rejected"] += 1
            else:
                results["skipped"] += 1

        logger.info(
            "Evolution cycle: %d candidates, %d evolved, %d rejected, %d skipped",
            results["candidates"], results["evolved"],
            results["rejected"], results["skipped"],
        )
        return results

    def run_post_session(
        self, session_id: str, profile_id: str = "default",
    ) -> dict:
        """Run after a session ends. Checks trigger 1.

        Wrapped in a budget cycle (SB-3) so post-session evolution is
        subject to the same 10-LLM-call / 30-min wall-time cap.
        """
        if not self._is_enabled():
            return {"enabled": False, "candidates": 0, "evolved": 0, "rejected": 0}

        try:
            with self._budget.cycle() as _b:
                return self._run_post_session_body(session_id, profile_id)
        except BudgetExhausted as exc:
            logger.warning(
                "post-session evolution aborted: budget exhausted [%s]",
                getattr(exc, "dimension", "?"),
            )
            return {
                "enabled": True, "aborted": True, "budget_exhausted": True,
                "dimension": getattr(exc, "dimension", None),
                "candidates": 0, "evolved": 0, "rejected": 0,
            }
        finally:
            self._current_cycle_id = None

    def _run_post_session_body(
        self, session_id: str, profile_id: str,
    ) -> dict:
        results = {"candidates": 0, "evolved": 0, "rejected": 0}

        trigger = PostSessionTrigger(self._db_path)
        candidates = trigger.scan(session_id, profile_id)
        results["candidates"] = len(candidates)

        for candidate in candidates:
            if not self._store.can_evolve(profile_id):
                break
            outcome = self._process_candidate(candidate, profile_id)
            if outcome == "evolved":
                results["evolved"] += 1
            elif outcome == "rejected":
                results["rejected"] += 1

        return results

    def evolve_candidate(
        self, candidate: EvolutionCandidate, profile_id: str = "default",
    ) -> str:
        """Evolve a single candidate under a budget cycle (audit-10 fix).

        Public entry point for the ``evolve_skill`` MCP tool. Opening the
        budget cycle HERE is what makes the per-cycle LLM-call, wall-time and
        per-day caps apply to manual/MCP-triggered evolution, exactly as they
        do to consolidation. The MCP tool previously called
        ``_process_candidate`` directly, so budget charges happened outside a
        cycle, the guard raised RuntimeError, ``_llm_call`` swallowed it, and
        every cap was silently bypassed.
        """
        try:
            with self._budget.cycle():
                self._current_cycle_id = None
                return self._process_candidate(candidate, profile_id)
        except BudgetExhausted as exc:
            logger.info(
                "evolve_candidate skipped: budget exhausted [%s]",
                getattr(exc, "dimension", "?"),
            )
            return "skipped"

    def _process_candidate(
        self, candidate: EvolutionCandidate, profile_id: str,
    ) -> str:
        """Process a single evolution candidate through the full pipeline.

        Returns: "evolved", "rejected", or "skipped"
        """
        now = datetime.now(timezone.utc).isoformat()
        record_id = hashlib.sha256(
            f"{candidate.skill_name}:{candidate.trigger.value}:{now}".encode(),
        ).hexdigest()[:16]

        # Anti-loop: check if already addressed
        context_hash = hashlib.sha256(
            json.dumps(list(candidate.evidence)).encode(),
        ).hexdigest()[:12]

        if self._store.is_addressed(candidate.skill_name, context_hash):
            return "skipped"

        if self._store.has_exceeded_attempts(candidate.skill_name, profile_id):
            logger.info("Skill %s exceeded max attempts, flagging for review", candidate.skill_name)
            return "skipped"

        # Mark as addressed (even if we reject — prevents repeated checks)
        self._store.mark_addressed(candidate.skill_name, context_hash)

        # Step 1: Read original skill content
        original_content = self._read_skill_content(candidate.skill_name)

        # Create initial record
        record = EvolutionRecord(
            id=record_id,
            skill_name=candidate.skill_name,
            parent_skill_id=candidate.skill_name,
            evolution_type=candidate.evolution_type,
            trigger=candidate.trigger,
            status=EvolutionStatus.CANDIDATE,
            evidence=candidate.evidence,
            original_content=original_content[:2000],
            created_at=now,
        )
        self._store.save_record(record, profile_id)

        # Step 2: LLM confirmation gate (uses Haiku for cost)
        confirmed = self._llm_confirm(candidate, original_content)
        if not confirmed:
            record = dataclasses.replace(
                record,
                status=EvolutionStatus.REJECTED,
                rejection_reason="LLM confirmation gate rejected",
                completed_at=datetime.now(timezone.utc).isoformat(),
            )
            self._store.save_record(record, profile_id)
            return "rejected"

        # Step 3: Generate mutation (uses Sonnet for quality)
        prompt = mutgen.build_mutation_prompt(candidate, original_content)
        evolved_content = self._generate_mutation(prompt)
        if not evolved_content:
            record = dataclasses.replace(
                record,
                status=EvolutionStatus.FAILED,
                rejection_reason="Mutation generation failed",
                completed_at=datetime.now(timezone.utc).isoformat(),
            )
            self._store.save_record(record, profile_id)
            return "rejected"

        # Step 4: Blind verification (uses Haiku — different model from generator)
        description = self._extract_description(evolved_content)
        v_prompt = verifier.build_verification_prompt(
            candidate.skill_name, description, evolved_content,
        )
        v_result = self._blind_verify(v_prompt)
        if not v_result.passed:
            record = dataclasses.replace(
                record,
                status=EvolutionStatus.REJECTED,
                rejection_reason=f"Blind verification failed: {v_result.reasoning}",
                evolved_content=evolved_content[:2000],
                blind_verified=False,
                completed_at=datetime.now(timezone.utc).isoformat(),
            )
            self._store.save_record(record, profile_id)
            return "rejected"

        # Step 5: Persist evolved skill
        diff = self._compute_diff(original_content, evolved_content)
        skill_path = self._write_evolved_skill(candidate, evolved_content, record_id)

        # M-GENERATION: Compute generation from parent's history
        parent_history = self._store.get_skill_history(
            candidate.skill_name, profile_id, limit=1,
        )
        parent_gen = (
            parent_history[0].generation
            if parent_history and parent_history[0].status == EvolutionStatus.PROMOTED
            else 0
        )

        record = dataclasses.replace(
            record,
            status=EvolutionStatus.PROMOTED,
            evolved_content=evolved_content[:2000],
            content_diff=diff[:2000],
            mutation_summary=self._summarize_diff(diff),
            blind_verified=True,
            generation=parent_gen + 1,
            completed_at=datetime.now(timezone.utc).isoformat(),
        )
        self._store.save_record(record, profile_id)
        self._store.record_evolution_attempt(profile_id)

        logger.info(
            "Evolved skill: %s (%s via %s) → %s",
            candidate.skill_name, candidate.evolution_type.value,
            candidate.trigger.value, skill_path,
        )
        return "evolved"

    # ------------------------------------------------------------------
    # LLM calls — single-line funnel through evolution.llm_dispatch
    # ------------------------------------------------------------------
    #
    # SB-2: every LLM call goes through ``_dispatch_llm``. That function
    # owns model validation, redact_secrets, backend registry, and the
    # cost-log write. No other code in this module touches a backend.
    # SB-4: ``_dispatch_llm`` routes the claude-CLI path through
    # ``run_subprocess_safe`` — no bare ``subprocess.run`` here.

    def _llm_call(
        self, prompt: str, max_tokens: int = 500, model: str = "haiku",
    ) -> str:
        """Funnel a prompt through :func:`_dispatch_llm`.

        The ``model`` arg accepts short aliases (``haiku``/``sonnet``)
        for backward compatibility; they are resolved to the allow-listed
        model id before dispatch. Budget charge happens BEFORE dispatch;
        on ``BudgetExhausted`` we return an empty string so the caller
        treats this as "no evolution" (the usual fail-closed semantics).
        """
        backend = self._get_backend()
        if backend == "none":
            return ""

        # Charge the budget BEFORE dispatching so the cap is protective.
        # If budget is exhausted we degrade gracefully (empty string).
        try:
            self._budget.charge_llm_call()
            self._budget.check_time()
        except BudgetExhausted as exc:
            logger.info("evolution _llm_call skipped: %s", exc)
            return ""
        except RuntimeError:
            # charge_llm_call/check_time outside cycle() — shouldn't
            # happen in production wiring, but if a caller invokes
            # _llm_call without opening a cycle (e.g. tests), let it
            # pass so legacy behaviour is preserved.
            pass

        resolved_model = _resolve_model_alias(model)
        try:
            return _dispatch_llm(
                prompt,
                model=resolved_model,
                learning_db=Path(self._db_path),
                profile_id=self._profile_id,
                max_tokens=max_tokens,
                cycle_id=self._current_cycle_id,
            )
        except ValueError as exc:
            # A ValueError from _dispatch_llm is a CONTRACT breach, not a
            # runtime/transport failure: forbidden or unlisted model,
            # non-positive max_tokens, max_tokens over MAX_TOKENS_CAP, or
            # an empty profile_id. Every one is a misconfiguration in our
            # own wiring — e.g. a caller requesting more tokens than the
            # ceiling, which is exactly how mutation generation died
            # silently for releases. Log at ERROR so it surfaces in logs
            # and CI instead of masquerading as a normal "no LLM"
            # fail-closed. Behaviour is unchanged: we still return "" so a
            # consolidation cycle degrades gracefully rather than crashing.
            logger.error(
                "evolution dispatch misconfigured — no evolution this "
                "call (returning empty string): %s", exc,
            )
            return ""
        except Exception as exc:  # noqa: BLE001 — fail-closed
            logger.debug("evolution dispatch failed: %s", exc)
            return ""

    def _llm_confirm(self, candidate: EvolutionCandidate, original: str) -> bool:
        """LLM confirmation gate."""
        prompt = (
            f"A skill '{candidate.skill_name}' has effective score "
            f"{candidate.effective_score:.0%} over {candidate.invocation_count} invocations.\n"
            f"Evidence: {'; '.join(candidate.evidence)}\n\n"
            f"Should this skill be evolved ({candidate.evolution_type.value})? "
            f"Reply YES or NO with brief reason."
        )
        response = self._llm_call(
            prompt, max_tokens=100, model=self._get_models().confirm,
        )
        if not response:
            logger.warning("LLM confirmation gate: empty response, skipping evolution for %s", candidate.skill_name)
            return False  # Fail-closed: no LLM = no evolution
        return "yes" in response.lower()

    def _generate_mutation(self, prompt: str) -> Optional[str]:
        """Generate evolved SKILL.md via the configured mutation model.

        Defaults to the lowest-cost model for the backend (v3.7.9);
        users can opt up (e.g. to sonnet) via ``evolution.mutation_model``.
        """
        mutation_model = self._get_models().mutation
        for attempt in range(mutgen.MAX_APPLY_RETRIES):
            response = self._llm_call(
                prompt, max_tokens=4000, model=mutation_model,
            )
            if not response:
                return None
            content = mutgen.parse_mutation_output(response)
            if content:
                error = mutgen.validate_skill_content(content)
                if error is None:
                    return content
                prompt = mutgen.build_retry_prompt(prompt, error, attempt + 1)
            else:
                prompt = mutgen.build_retry_prompt(
                    prompt, "No valid SKILL.md content found in output", attempt + 1,
                )
        return None

    def _blind_verify(self, prompt: str) -> verifier.VerificationResult:
        """Blind verification — runs on a model independent of the generator.

        See ``model_selection.resolve_evolution_models``: the verify model
        is kept different from the mutation model where possible so the
        generator can't grade its own homework.
        """
        response = self._llm_call(
            prompt, max_tokens=500, model=self._get_models().verify,
        )
        if not response:
            return verifier.VerificationResult(
                passed=False, confidence=0.0, reasoning="No LLM response",
            )
        return verifier.parse_verification_response(response)

    # ------------------------------------------------------------------
    # Skill I/O
    # ------------------------------------------------------------------

    def _read_skill_content(self, skill_name: str) -> str:
        """Read a skill's SKILL.md content. Searches known skill directories."""
        search_dirs = [
            Path.home() / ".claude" / "skills",
            Path.home() / ".claude" / "plugins",
            EVOLVED_SKILLS_DIR,
        ]

        # Handle namespaced skills (e.g., "superpowers:brainstorming")
        if ":" in skill_name:
            parts = skill_name.split(":")
            search_patterns = [
                f"**/{parts[-1]}/SKILL.md",
                f"**/{skill_name.replace(':', '/')}/SKILL.md",
                f"**/skills/{parts[-1]}/SKILL.md",
            ]
        else:
            search_patterns = [f"**/{skill_name}/SKILL.md"]

        for search_dir in search_dirs:
            if not search_dir.exists():
                continue
            for pattern in search_patterns:
                matches = list(search_dir.glob(pattern))
                if matches:
                    try:
                        return matches[0].read_text(encoding="utf-8")
                    except OSError:
                        continue

        return ""

    def _write_evolved_skill(
        self,
        candidate: EvolutionCandidate,
        content: str,
        record_id: str,
    ) -> Path:
        """Write evolved SKILL.md to ~/.claude/skills/evolved/."""
        EVOLVED_SKILLS_DIR.mkdir(parents=True, exist_ok=True)

        # Build directory name. skill_name derives from a behavioral-assertion
        # trigger (any non-whitespace), so sanitize for ALL evolution types —
        # "../../.claude/settings" must not escape EVOLVED_SKILLS_DIR.
        base_name = re.sub(r"[^a-zA-Z0-9_-]", "-",
                           candidate.skill_name).lower()[:50] or "skill"
        if candidate.evolution_type == EvolutionType.FIX:
            dir_name = f"{base_name}-v{record_id[:6]}"
        elif candidate.evolution_type == EvolutionType.DERIVED:
            # Extract name from evolved content frontmatter
            name_match = re.search(r"name:\s*(.+)", content)
            dir_name = name_match.group(1).strip() if name_match else f"{base_name}-derived"
            dir_name = re.sub(r"[^a-zA-Z0-9_-]", "-", dir_name).lower()[:50]
        else:
            dir_name = base_name
        dir_name = dir_name or "skill"

        skill_dir = EVOLVED_SKILLS_DIR / dir_name
        # Defense in depth: never write outside the evolved-skills directory.
        if not str(skill_dir.resolve()).startswith(str(EVOLVED_SKILLS_DIR.resolve())):
            raise ValueError(f"evolved skill path escapes sandbox: {dir_name!r}")
        skill_dir.mkdir(parents=True, exist_ok=True)

        skill_path = skill_dir / "SKILL.md"
        skill_path.write_text(content, encoding="utf-8")

        # Write metadata sidecar
        meta = {
            "skill_id": dir_name,
            "parent_skill_id": candidate.skill_name,
            "evolution_type": candidate.evolution_type.value,
            "trigger": candidate.trigger.value,
            "record_id": record_id,
            "evidence": list(candidate.evidence),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        meta_path = skill_dir / ".skill_meta.json"
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        return skill_path

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _extract_description(self, content: str) -> str:
        """Extract description from SKILL.md frontmatter."""
        match = re.search(r"description:\s*(.+?)(?:\n|---)", content)
        return match.group(1).strip() if match else "Skill for AI agent tasks"

    def _compute_diff(self, original: str, evolved: str) -> str:
        if not original:
            return "(new skill — no original to diff)"
        diff = difflib.unified_diff(
            original.splitlines(keepends=True),
            evolved.splitlines(keepends=True),
            fromfile="original",
            tofile="evolved",
            n=3,
        )
        return "".join(diff)

    def _summarize_diff(self, diff: str) -> str:
        """Count additions and removals."""
        additions = sum(1 for line in diff.splitlines() if line.startswith("+") and not line.startswith("+++"))
        removals = sum(1 for line in diff.splitlines() if line.startswith("-") and not line.startswith("---"))
        return f"+{additions}/-{removals} lines"
