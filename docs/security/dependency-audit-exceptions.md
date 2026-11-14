# Dependency audit exceptions

The release dependency audit is fail-closed except for the exact advisories
listed below. Each exception must name the transitive dependency path, explain
why the vulnerable API is not reachable from untrusted input, and be removed
as soon as a stable patched release is available.

**Status as of 2026-11-14:** no exceptions remain. The audit fails on any
known advisory.

## Closed exceptions

- **GHSA-8mgp-746c-j5xp (NLTK, via llmlingua 0.2.2).** Ignored from 2026-10-02
  because no patched release existed and SuperLocalMemory never reaches the
  affected model-artifact APIs (LLMLingua calls only `nltk.sent_tokenize`).
  NLTK 3.10.4 fixes it, together with eight further advisories published
  against 3.10.3. SuperLocalMemory 4.2.18 pins `nltk==3.10.4` and the
  `--ignore-vuln` line is removed from the audit.

---

## Why these stay published

Naming an advisory, proving the vulnerable API is unreachable from untrusted
input, and committing to a removal condition is more useful to a reader than
silence. Ignoring the advisories and saying nothing would be worse.

The obligation that comes with publishing is that the entries must stay true: each one has to be rechecked against upstream releases, and removed once a stable patched release exists.
