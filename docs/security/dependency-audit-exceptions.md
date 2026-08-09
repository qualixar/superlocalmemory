# Dependency audit exceptions

The release dependency audit is fail-closed except for the exact advisories
listed below. Each exception must name the transitive dependency path, explain
why the vulnerable API is not reachable from untrusted input, and be removed
as soon as a stable patched release is available.

**Status as of 2026-08-09:** the NLTK exception is retired in 4.0.1. The two
remaining exceptions are actionable: stable releases beyond the affected
versions are available, so removal is blocked only on our scheduled, verified
native-stack upgrade. Their non-reachability arguments remain accurate and are
why these are exceptions rather than incidents.

## Retired: PYSEC-2026-597 — NLTK

SuperLocalMemory 4.0.1 pins `nltk==3.10.0`, above the affected 3.9.4 release.
The CI suppression was removed with that upgrade.
## PYSEC-2026-3447 — setuptools 81.0.0

- **Dependency path:** `superlocalmemory -> torch 2.11.0 -> setuptools <82`.
- **Exposure:** The advisory is a Unicode-normalization exclusion bypass while
  building an sdist on macOS. The installed runtime does not build sdists from
  user input. Release artifacts are built only from the controlled repository,
  and artifact tests reject unsafe archive member names and types.
- **Upstream state on 2026-08-08:** setuptools 83.0.0 is stable; PyTorch 2.13.0
  is stable. Both preconditions this entry depended on are satisfied.
- **Removal condition:** MET upstream. This exception is removed by the PyTorch
  upgrade below, whose own constraint permits a patched setuptools.
- **Review deadline:** 2026-09-05.

## GHSA-rrmf-rvhw-rf47 — PyTorch 2.11.0

- **Dependency path:** Direct runtime dependency used by the local embedding
  and reranking stack.
- **Exposure:** The advisory is memory corruption in `torch.jit.script()`.
  SuperLocalMemory does not call or expose TorchScript compilation on CLI,
  HTTP, dashboard, MCP, embedding, or reranking inputs.
- **Upstream state on 2026-08-08:** PyTorch 2.13.0 is stable on PyPI.
- **The previous deferral rationale is WITHDRAWN.** This entry previously stated
  that a combined native/ML stack upgrade "passed 446 focused tests but the
  6,000-test process segfaulted during garbage collection at 96%", and used that
  crash to justify staying on 2.11. That attribution was wrong. The crash has
  since been root-caused: the faulthandler C stack names `_pydantic_core`
  crashing while called by CPython's `gc_collect_main`. It is measured at ~22%
  of full-suite runs (2 crashes / 9 runs), reproduces on Python 3.12.13 **and**
  3.14.5, and occurs on the **current** stack with PyTorch 2.11 still pinned.
  It is an upstream pydantic-core defect, unrelated to the PyTorch upgrade, and
  it is not removed by staying on 2.11. Note also that `pydantic` 2.13.4 hard-pins
  `pydantic-core==2.46.4` and raises `SystemError` if that pin is changed, so
  there is currently no upgrade path out of it.
- **Removal condition:** Move to 2.13 or later. The previous condition ("after
  the native stack passes the complete matrix without a process crash") is not
  achievable while the pydantic-core defect persists, and gating a security
  upgrade on an unrelated upstream crash is the wrong trade. The correct gate is
  the focused native/ML suite plus the platform matrix, treating the
  pydantic-core GC crash as a known, separately-tracked upstream issue rather
  than as a signal about PyTorch.
- **Review deadline:** 2026-09-05.

---

## Why these stay published

Naming an advisory, proving the vulnerable API is unreachable from untrusted
input, and committing to a removal condition is more useful to a reader than
silence. Ignoring the advisories and saying nothing would be worse.

The obligation that comes with publishing is that the entries must stay true.
This revision exists because they had stopped being true: three "no patch
available upstream" statements had all been overtaken by stable releases, and
one deferral rationale blamed a crash that has since been root-caused to an
unrelated dependency.

## Retired exceptions

- `PYSEC-2026-597` was retired on 2026-08-08 after V4 pinned NLTK 3.10.0,
  verified LLMLingua compatibility, and removed the CI suppression.
