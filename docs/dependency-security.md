# Dependency security repair in source

The October 2026 repair updates affected runtime dependency floors and the lock
without adding a blanket audit exception. It covers sentence-transformers,
transformers, NLTK removal, HTTP clients, AnyIO, PyJWT, accelerate and urllib3.
The build-system setuptools floor is 83.0.0 or newer. This is a source repair;
a published package does not acquire these changes until a new release ships.

## Optional aggressive compression

NLTK through 3.10.3 has an unpatched High advisory,
[GHSA-8mgp-746c-j5xp](https://github.com/advisories/GHSA-8mgp-746c-j5xp).
LLMLingua 0.2.2 depends on NLTK. These packages are temporarily omitted from
shipped requirements rather than hidden by an audit ignore or moved into an
unsafe extra. The LLMLingua implementation and selected model remain in source.

An existing installation cannot activate this backend with a known-affected,
missing, malformed or prerelease NLTK version: the guard runs before importing
the backend. Setup uses the same guarded constructor. The router retains its
lossless fallback and records why the optional backend is unavailable.
Local memory, retrieval, cache, safe normalization and reversible storage are
separate capabilities. No new aggressive compression ratio is claimed.

## Native dependency and integration updates

Torch 2.13 removes the prior Low Torch advisory and permits a patched setuptools
runtime. Optional ingestion dependencies receive floors for httplib2, icalendar,
oauthlib and pyasn1, so an all-extras install follows the repaired graph too.
There is no blanket vulnerability ignore. Exact lock, runtime and all-extras
audit results must be recorded before release.

Before release, verify the exact wheel/lock, runtime and optional-backend behavior,
embedding/reranker compatibility, tests, and both filtered and unfiltered audits.
Do not present source-only changes as already fixed in the published PyPI/npm package.
