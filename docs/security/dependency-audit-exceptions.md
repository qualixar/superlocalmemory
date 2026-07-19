# Dependency audit exceptions

The release dependency audit is fail-closed except for the exact advisories
listed below. Each exception must name the transitive dependency path, explain
why the vulnerable API is not reachable from untrusted input, and be removed
as soon as a stable patched release is available.

## PYSEC-2026-597 — NLTK 3.9.4

- **Dependency path:** `superlocalmemory -> llmlingua 0.2.2 -> nltk 3.9.4`.
- **Exposure:** The advisory concerns percent-encoded path traversal supplied
  to `nltk.data.find()` or `nltk.data.load()`. SuperLocalMemory does not accept
  an NLTK resource path from CLI, HTTP, dashboard, or MCP input. Its optional
  LLMLingua compressor uses a fixed model identifier.
- **Upstream state on 2026-07-20:** pip-audit reports no stable fixed version.
  NLTK 3.10.0-rc1 is referenced upstream but is not available as an auditable
  stable PyPI dependency.
- **Removal condition:** Replace NLTK 3.9.4 as soon as a stable patched release
  is compatible with LLMLingua, then delete the exact CI ignore.
- **Review deadline:** 2026-08-20.

## PYSEC-2026-3447 — setuptools 81.0.0

- **Dependency path:** `superlocalmemory -> torch 2.11.0 -> setuptools <82`.
- **Exposure:** The advisory is a Unicode-normalization exclusion bypass while
  building an sdist on macOS. The installed runtime does not build sdists from
  user input. Release artifacts are built only from the controlled repository,
  and artifact tests reject unsafe archive member names and types.
- **Removal condition:** Removed with the PyTorch 2.13 upgrade because its
  runtime constraint permits patched setuptools.
- **Review deadline:** 2026-08-20.

## GHSA-rrmf-rvhw-rf47 — PyTorch 2.11.0

- **Dependency path:** Direct runtime dependency used by the local embedding
  and reranking stack.
- **Exposure:** The advisory is memory corruption in `torch.jit.script()`.
  SuperLocalMemory does not call or expose TorchScript compilation on CLI,
  HTTP, dashboard, MCP, embedding, or reranking inputs.
- **Why 2.13.0 is not accepted in 3.7.7:** A combined native/ML stack upgrade
  passed 446 focused tests but the 6,000-test process segfaulted during garbage
  collection at 96%. Restoring PyTorch alone did not remove the crash, so no
  single dependency is blamed; 2.11 remains the established runtime while the
  Transformers upgrade is narrowed independently.
- **Removal condition:** Move to 2.13 or later after the native stack passes the
  complete macOS, Linux, and Windows matrix without a process crash.
- **Review deadline:** 2026-08-20.
