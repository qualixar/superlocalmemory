# ADR 0001: Treat recalled memory as untrusted evidence

- Status: Accepted for V3.7
- Date: 2026-07-15
- Owners: SuperLocalMemory maintainers
- Defect: `SLM-P0-004`

## Context

SuperLocalMemory stores text from conversations, tools, imports, hooks, and
other external sources. That text may contain secrets, prompt-injection text,
forged boundary markers, or instructions that were legitimate in their
original context but unsafe when recalled later.

Several product surfaces historically formatted recalled text independently.
Some IDE adapters also wrote dynamic recalled text into instruction files,
where the host can treat the entire file as trusted developer guidance. A
delimiter inside such a file does not change the host's trust level and is not
a sufficient security boundary.

No formatter can guarantee that every current or future language model will
ignore every adversarial string. The enforceable product contract is therefore
structural: minimize trusted instructions, sanitize retrieved data, preserve
provenance, bound its size, and prove that every supported injection surface
uses the same boundary.

## Considered alternatives

### 1. Keep per-surface formatters

This is the lowest-effort and most reversible option, but every new surface can
drift in redaction, token budgets, provenance, or delimiter handling. It makes
the security property depend on many implementations and was rejected.

### 2. Strip instruction-like sentences from stored memories

This can remove obvious attacks, but it also destroys valid technical evidence
such as runbooks, prompts, policies, and code. Attackers can paraphrase around a
keyword filter. The data loss is hard to reverse and the security benefit is
weak, so this was rejected.

### 3. Use one evidence renderer and keep IDE instruction files static

All runtime recall surfaces map results into one typed shape and one renderer.
The renderer redacts recognized secrets, neutralizes canonical boundary text,
attaches provenance, applies budgets, and wraps the result in an explicit
reference-only evidence block. Cursor, Copilot, and Antigravity instruction
files contain only product-authored protocol; they instruct the host to fetch
memory at runtime and do not persist recalled text. This has a moderate
migration cost but creates one auditable contract and is the selected design.

## Decision

V3.7 adopts alternative 3.

1. Dynamic recalled memory is untrusted evidence on every surface, regardless
   of whether its source is first-party or local.
2. Product injection surfaces must use `render_context(..., wrap=True)` or
   `render_untrusted_text(...)`. They must fail closed instead of falling back
   to an ad-hoc raw formatter.
3. The rendered block carries canonical begin/end markers, a reference-only
   policy, source/fact provenance where available, secret redaction, boundary
   neutralization, and configured size budgets.
4. Structured memory arrays returned beside rendered text are independently
   sanitized, bounded, and marked untrusted.
5. Trusted system and IDE instruction files contain only static,
   product-authored protocol. Dynamic topics, decisions, entities, patterns,
   and memories are never written into those files.
6. Stored text never gains authority to call tools, change roles, request
   secrets, or override product instructions merely because it was recalled.

## Consequences

Existing users may see different context formatting, and IDE sync no longer
creates a dynamic memory snapshot in a high-trust instruction file. Runtime MCP
recall is the supported freshness path. Secret redaction is defense in depth,
not a substitute for avoiding secret ingestion or for provider-side controls.

The `trust_first_party` configuration key remains readable for compatibility,
but it cannot weaken the mandatory boundary. A future removal requires a normal
deprecation cycle.

## Verification

The release gate must include:

- Adversarial tests for forged markers, instruction text, role changes, tool
  requests, and representative secret formats.
- Contract tests for hooks, MCP `session_init`, CLI session context, chat, and
  IDE adapters.
- A source guard rejecting product calls to `render_context(..., wrap=False)`
  and legacy ad-hoc memory boundary markers.
- Full default, slow, and warning-as-error test lanes on the frozen release
  candidate.

These tests prove structural containment and surface parity. They do not claim
universal behavioral immunity across arbitrary language models.
