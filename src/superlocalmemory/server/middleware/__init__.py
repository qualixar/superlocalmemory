# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — LLD-04 §4.2

"""FastAPI middleware — strict security headers for the Brain UI (LLD-04 v2).

The existing ``server/security_middleware.py`` is kept for legacy routes
that still rely on permissive CSP (``'unsafe-inline'`` + CDNs). The
middleware in this subpackage enforces the v3.4.22 policy: no inline
scripts / styles, no nonces, no CDN sources.
"""
