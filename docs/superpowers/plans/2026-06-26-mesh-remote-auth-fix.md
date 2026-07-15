# SLM v3.6.20 — Mesh Remote Auth Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the mesh HTTP auth regression from v3.6.12 where `_get_broker` was made to require `X-Mesh-Secret` but all existing callers (the SLM client, the docs, and `_validate_remote_auth`) use `Authorization: Bearer`.

**Architecture:** Three-part fix: (1) make `_get_broker` accept BOTH `X-Mesh-Secret` (backwards compat) AND `Authorization: Bearer <secret>` (RFC 7617 — what remote_sync.py and the docs say), (2) delete the dead `_validate_remote_auth` function and its single call site on `/peers` (now redundant since `_get_broker` unifies auth), (3) update `distributed-deployment.md` mesh section to document the accepted auth headers. Version bump to 3.6.20 throughout.

**Tech Stack:** Python 3.12, FastAPI, pytest, httpx, hmac

---

## Root Cause Summary (read before touching any code)

`_get_broker()` in `mesh.py` was added in v3.6.12 with:
```python
presented = request.headers.get("x-mesh-secret", "")
if not hmac.compare_digest(presented, secret):
    raise HTTPException(401, detail="invalid or missing mesh secret")
```

But three independent sources all specify `Authorization: Bearer`:
- `docs/multi-machine.md` line 99: "The `/mesh/peers` endpoint requires `Authorization: Bearer <shared_secret>`"
- `docs/multi-machine.md` line 127: troubleshooting curl uses `-H 'Authorization: Bearer <secret>'`
- `remote_sync.py` lines 129, 179: `headers["Authorization"] = f"Bearer {self._shared_secret}"`
- `_validate_remote_auth()` in `mesh.py`: checks `Authorization: Bearer` — but is only wired to `/peers` and is never reached anyway (because `_get_broker` already rejects Bearer callers with `X-Mesh-Secret` check first)

The v3.6.12 test `test_secret_accepts_correct_header` validated `X-Mesh-Secret` as correct — but the test was written to match the new (broken) code, not the documented contract.

**Effect:** Any non-loopback caller using `Authorization: Bearer` (which is EVERY caller following the docs, every SLM `RemoteSyncClient`, and issue #60's user) gets `{"detail": "invalid or missing mesh secret"}` — even when the secret is correct.

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `src/superlocalmemory/server/routes/mesh.py` | **Modify** | Fix `_get_broker` auth, remove `_validate_remote_auth` |
| `tests/integration/test_mesh_http.py` | **Modify** | Add 7 new tests covering Bearer auth and all affected routes |
| `docs/distributed-deployment.md` | **Modify** | Update mesh table to document both accepted auth headers |
| `src/superlocalmemory/__init__.py` | **Modify** | Bump `__version__` to `3.6.20` |
| `pyproject.toml` | **Modify** | Bump `version` to `3.6.20` |
| `package.json` | **Modify** | Bump `version` to `3.6.20` |
| `plugin-src/manifest.json` | **Modify** | Bump `version` to `3.6.20` |
| `plugin-src/requirements.txt` | **Modify** | Bump pin to `superlocalmemory==3.6.20` |
| `tests/test_version_consistency.py` | **Modify** | Update `EXPECTED_VERSION` to `"3.6.20"` |

**Do NOT touch:**
- `src/superlocalmemory/hooks/claude_code_hooks.py` — `HOOKS_VERSION` stays `"3.6.18"` (hook behavior unchanged)
- `src/superlocalmemory/mesh/remote_sync.py` — already correct (sends `Authorization: Bearer`)
- `docs/multi-machine.md` — already correct (documents `Authorization: Bearer`)

---

## Task 1: Write the failing tests first (TDD)

**Files:**
- Modify: `tests/integration/test_mesh_http.py`

- [ ] **Step 1.1: Add 7 new test functions at the end of the file**

Open `tests/integration/test_mesh_http.py` and append these tests after the existing `test_no_secret_allows_all` function:

```python
def test_bearer_token_accepted_for_nonloopback() -> None:
    # Core regression: Authorization: Bearer must work (what remote_sync.py sends,
    # what the docs document). Before the fix this returns 401.
    app, _ = _app_with_broker(secret="topsecret")
    c = TestClient(app)
    r = c.post(
        "/mesh/register",
        json={"session_id": "s"},
        headers={"Authorization": "Bearer topsecret"},
    )
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"


def test_bearer_token_wrong_secret_rejected() -> None:
    # A wrong Bearer secret must still 401 — security is not weakened.
    app, _ = _app_with_broker(secret="topsecret")
    c = TestClient(app)
    r = c.post(
        "/mesh/register",
        json={"session_id": "s"},
        headers={"Authorization": "Bearer wrongsecret"},
    )
    assert r.status_code == 401


def test_xmesh_secret_still_works_backwards_compat() -> None:
    # X-Mesh-Secret must still work — do NOT break callers that discovered this
    # undocumented header and started using it.
    app, _ = _app_with_broker(secret="topsecret")
    c = TestClient(app)
    r = c.post(
        "/mesh/register",
        json={"session_id": "s"},
        headers={"X-Mesh-Secret": "topsecret"},
    )
    assert r.status_code == 200


def test_status_endpoint_bearer_auth() -> None:
    # /mesh/status was the endpoint reported in issue #60. Must accept Bearer.
    app, _ = _app_with_broker(secret="topsecret")
    c = TestClient(app)
    r = c.get("/mesh/status", headers={"Authorization": "Bearer topsecret"})
    assert r.status_code == 200
    body = r.json()
    assert body.get("broker_up") is True


def test_status_endpoint_blocked_without_auth() -> None:
    # /mesh/status must 401 when secret is configured and no auth header is given.
    app, _ = _app_with_broker(secret="topsecret")
    c = TestClient(app)
    r = c.get("/mesh/status")
    assert r.status_code == 401


def test_peers_endpoint_bearer_auth() -> None:
    # /mesh/peers must also work with Bearer (remote_sync.py calls this endpoint).
    app, _ = _app_with_broker(secret="topsecret")
    c = TestClient(app)
    r = c.get("/mesh/peers", headers={"Authorization": "Bearer topsecret"})
    assert r.status_code == 200
    assert "peers" in r.json()


def test_send_endpoint_bearer_auth() -> None:
    # /mesh/send must also work with Bearer (remote_sync.py calls this endpoint).
    app, broker = _app_with_broker(secret="topsecret")
    # Register a recipient peer without auth (no secret check for loopback in real
    # server; TestClient uses "testclient" host so we need the header here too).
    c = TestClient(app)
    reg = c.post(
        "/mesh/register",
        json={"session_id": "receiver"},
        headers={"Authorization": "Bearer topsecret"},
    )
    assert reg.status_code == 200
    peer_id = reg.json()["peer_id"]

    r = c.post(
        "/mesh/send",
        json={"from_peer": "sender", "to_peer": peer_id, "content": "hello", "type": "text"},
        headers={"Authorization": "Bearer topsecret"},
    )
    assert r.status_code == 200
    assert r.json().get("ok") is True
```

- [ ] **Step 1.2: Run the new tests — confirm they ALL fail (red)**

```bash
cd /Users/varunpratapbhardwaj/Documents/work/varun-world/Agentic_official/superlocalmemory
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  KMP_DUPLICATE_LIB_OK=TRUE PYTHONMALLOC=malloc \
  PYTHONPATH=src .venv/bin/python -m pytest tests/integration/test_mesh_http.py -v \
  -p no:cacheprovider -p no:subtests -o faulthandler_timeout=0
```

Expected: The 4 original tests PASS. The 7 new tests FAIL with 401 status codes (Bearer is rejected, status is unprotected).

Specifically:
- `test_bearer_token_accepted_for_nonloopback` → FAIL (401 → expected 200)
- `test_bearer_token_wrong_secret_rejected` → PASS (already correct behaviour by accident — wrong secret is rejected regardless)
- `test_xmesh_secret_still_works_backwards_compat` → PASS (X-Mesh-Secret already works)
- `test_status_endpoint_bearer_auth` → FAIL (401 → expected 200)
- `test_status_endpoint_blocked_without_auth` → PASS (no auth → 401 already correct)
- `test_peers_endpoint_bearer_auth` → FAIL (401 → expected 200)
- `test_send_endpoint_bearer_auth` → FAIL (401 → expected 200)

If fewer than 4 tests fail, re-read the root cause section above before continuing.

---

## Task 2: Fix `_get_broker` to accept both auth headers

**Files:**
- Modify: `src/superlocalmemory/server/routes/mesh.py` (the `_get_broker` function)

- [ ] **Step 2.1: Locate the exact block to change in `_get_broker`**

In `mesh.py`, find this block inside `_get_broker`:
```python
    secret = getattr(broker, "_shared_secret", None)
    if secret:
        client_host = request.client.host if request.client else ""
        if client_host not in ("127.0.0.1", "::1", "localhost"):
            import hmac
            presented = request.headers.get("x-mesh-secret", "")
            if not hmac.compare_digest(presented, secret):
                raise HTTPException(401, detail="invalid or missing mesh secret")
    return broker
```

- [ ] **Step 2.2: Replace that block with the dual-header version**

Replace with:
```python
    secret = getattr(broker, "_shared_secret", None)
    if secret:
        client_host = request.client.host if request.client else ""
        if client_host not in ("127.0.0.1", "::1", "localhost"):
            import hmac
            # Accept X-Mesh-Secret (legacy undocumented header, backwards compat) OR
            # Authorization: Bearer <secret> (RFC 7617 — what remote_sync.py sends and
            # what the docs specify). The v3.6.12 gate accidentally used X-Mesh-Secret
            # only, breaking all documented callers. Both are now valid; Bearer is canonical.
            presented = (
                request.headers.get("x-mesh-secret")
                or request.headers.get("authorization", "").removeprefix("Bearer ").strip()
            )
            if not presented or not hmac.compare_digest(presented, secret):
                raise HTTPException(401, detail="invalid or missing mesh secret")
    return broker
```

---

## Task 3: Remove the dead `_validate_remote_auth` function

**Files:**
- Modify: `src/superlocalmemory/server/routes/mesh.py`

`_validate_remote_auth` is now dead code — `_get_broker` handles all remote auth. It also has a bug: it requires Bearer from ALL callers when `broker._is_remote` (i.e., `SLM_MESH_HOST=0.0.0.0`), which would incorrectly block loopback callers. Removing it is strictly correct.

- [ ] **Step 3.1: Delete the `_validate_remote_auth` function**

Find and remove the entire function (it's between `_get_broker` and `# -- Routes --`):
```python
def _validate_remote_auth(request: Request, broker) -> None:
    """Validate bearer token for cross-machine requests."""
    if not broker._is_remote:
        return  # local mode — no auth needed
    secret = broker._shared_secret
    if not secret:
        return
    auth = request.headers.get("Authorization", "")
    if auth != f"Bearer {secret}":
        raise HTTPException(401, detail="Unauthorized")
```

- [ ] **Step 3.2: Remove the `_validate_remote_auth` call from the `/peers` route**

Find:
```python
@router.get("/peers")
async def peers(request: Request):
    broker = _get_broker(request)
    _validate_remote_auth(request, broker)
    return {"peers": broker.list_all_peers()}
```

Replace with:
```python
@router.get("/peers")
async def peers(request: Request):
    broker = _get_broker(request)
    return {"peers": broker.list_all_peers()}
```

---

## Task 4: Run the full test suite — confirm green

- [ ] **Step 4.1: Run only the mesh HTTP tests first**

```bash
cd /Users/varunpratapbhardwaj/Documents/work/varun-world/Agentic_official/superlocalmemory
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  KMP_DUPLICATE_LIB_OK=TRUE PYTHONMALLOC=malloc \
  PYTHONPATH=src .venv/bin/python -m pytest tests/integration/test_mesh_http.py -v \
  -p no:cacheprovider -p no:subtests -o faulthandler_timeout=0
```

Expected: **11 passed, 0 failed** (4 original + 7 new).

If any test fails, do NOT proceed to the next step. Read the error, fix the code in `mesh.py`, and re-run.

- [ ] **Step 4.2: Run the full suite**

```bash
cd /Users/varunpratapbhardwaj/Documents/work/varun-world/Agentic_official/superlocalmemory
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  KMP_DUPLICATE_LIB_OK=TRUE PYTHONMALLOC=malloc \
  PYTHONPATH=src .venv/bin/python -m pytest -p no:cacheprovider -p no:subtests -q \
  -o faulthandler_timeout=0
```

Expected: **6003+ passed, 0 failed** (the 7 new tests bring total to 6010).

If anything fails that was passing before, that is a regression — stop and fix before continuing.

- [ ] **Step 4.3: Commit the fix**

```bash
cd /Users/varunpratapbhardwaj/Documents/work/varun-world/Agentic_official/superlocalmemory
git add src/superlocalmemory/server/routes/mesh.py tests/integration/test_mesh_http.py
git commit -m "fix(mesh): accept Authorization: Bearer in _get_broker, remove dead _validate_remote_auth

_get_broker (added v3.6.12) required X-Mesh-Secret but every caller
uses Authorization: Bearer — remote_sync.py, multi-machine.md, and
_validate_remote_auth all specified Bearer. Now accepts both headers
(X-Mesh-Secret for backwards compat, Bearer as canonical per RFC 7617).
Removes dead _validate_remote_auth that was wired only to /peers and
had a secondary bug (blocking loopback callers when SLM_MESH_HOST=0.0.0.0).

Fixes: issue #60 (SLM_MESH_HOST=0.0.0.0 + Bearer rejected)
Closes: mesh-1 auth regression from v3.6.12"
```

---

## Task 5: Update `distributed-deployment.md` mesh section

**Files:**
- Modify: `docs/distributed-deployment.md`

The Mesh table (around line 201-211) only documents `SLM_MESH_SHARED_SECRET` as "Required when `SLM_DAEMON_HOST != 127.0.0.1`" without explaining what header to use. Add that information.

- [ ] **Step 5.1: Find the Mesh table in `distributed-deployment.md`**

Locate this row in the `### Mesh` table:
```
| `SLM_MESH_SHARED_SECRET` | Required when `SLM_DAEMON_HOST != 127.0.0.1`; authenticates mesh peers | — |
```

- [ ] **Step 5.2: Update the row and add an auth note below the table**

Change the row to:
```
| `SLM_MESH_SHARED_SECRET` | Auth secret for the mesh HTTP API. Required when `SLM_MESH_HOST` is not localhost. Send as `Authorization: Bearer <secret>` (canonical) or `X-Mesh-Secret: <secret>` (legacy). | — |
```

Then directly below the Mesh table, add:

```markdown
> **Mesh API auth:** When `SLM_MESH_SHARED_SECRET` is set, non-loopback callers must authenticate every request to `/mesh/*`. The canonical header is `Authorization: Bearer <your-secret>` — this is what `RemoteSyncClient` sends automatically and what the troubleshooting curl examples use. The legacy `X-Mesh-Secret: <your-secret>` header is also accepted for backwards compatibility.
>
> Example: `curl http://192.168.50.144:8765/mesh/status -H "Authorization: Bearer <your-secret>"`
```

- [ ] **Step 5.3: Commit the docs**

```bash
cd /Users/varunpratapbhardwaj/Documents/work/varun-world/Agentic_official/superlocalmemory
git add docs/distributed-deployment.md
git commit -m "docs(mesh): document Authorization: Bearer as canonical auth header for mesh API

Fixes the missing auth documentation that led to issue #60. The mesh
table previously only mentioned the secret is required but not which
HTTP header to send. Canonical header is Authorization: Bearer (RFC
7617), X-Mesh-Secret accepted for backwards compat."
```

---

## Task 6: Bump version to 3.6.20 across all 5 sources

**Files (ALL must be updated atomically — test_version_consistency.py enforces this):**
- `src/superlocalmemory/__init__.py`
- `pyproject.toml`
- `package.json`
- `plugin-src/manifest.json`
- `plugin-src/requirements.txt`
- `tests/test_version_consistency.py` (update `EXPECTED_VERSION`)

- [ ] **Step 6.1: Bump `src/superlocalmemory/__init__.py`**

Find: `__version__ = "3.6.19"`
Replace with: `__version__ = "3.6.20"`

- [ ] **Step 6.2: Bump `pyproject.toml`**

Find: `version = "3.6.19"`
Replace with: `version = "3.6.20"`

- [ ] **Step 6.3: Bump `package.json`**

Find: `"version": "3.6.19",`
Replace with: `"version": "3.6.20",`

- [ ] **Step 6.4: Bump `plugin-src/manifest.json`**

Find: `"version": "3.6.19"` (or similar — confirm the exact field in the file)
Replace with: `"version": "3.6.20"`

- [ ] **Step 6.5: Bump `plugin-src/requirements.txt`**

Find: `superlocalmemory==3.6.19`
Replace with: `superlocalmemory==3.6.20`

- [ ] **Step 6.6: Update `tests/test_version_consistency.py`**

Find: `EXPECTED_VERSION = "3.6.19"`
Replace with: `EXPECTED_VERSION = "3.6.20"`

- [ ] **Step 6.7: Run version consistency test — confirm green**

```bash
cd /Users/varunpratapbhardwaj/Documents/work/varun-world/Agentic_official/superlocalmemory
env PYTHONPATH=src .venv/bin/python -m pytest tests/test_version_consistency.py -v \
  -p no:cacheprovider -p no:subtests -o faulthandler_timeout=0
```

Expected: **6 passed, 0 failed** — all 5 sources agree on `3.6.20`.

If any test fails, check which source was missed and fix it before continuing.

- [ ] **Step 6.8: Commit the version bump**

```bash
cd /Users/varunpratapbhardwaj/Documents/work/varun-world/Agentic_official/superlocalmemory
git add src/superlocalmemory/__init__.py pyproject.toml package.json \
  plugin-src/manifest.json plugin-src/requirements.txt \
  tests/test_version_consistency.py
git commit -m "chore(release): 3.6.20 — mesh remote auth fix (issue #60)"
```

---

## Task 7: Final full suite run + release verification

- [ ] **Step 7.1: Run the full test suite one last time**

```bash
cd /Users/varunpratapbhardwaj/Documents/work/varun-world/Agentic_official/superlocalmemory
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  KMP_DUPLICATE_LIB_OK=TRUE PYTHONMALLOC=malloc \
  PYTHONPATH=src .venv/bin/python -m pytest -p no:cacheprovider -p no:subtests -q \
  -o faulthandler_timeout=0 2>&1 | tail -5
```

Expected: **6010 passed, 0 failed** (6003 baseline + 7 new mesh tests).

Do NOT proceed to publish if there are any failures.

- [ ] **Step 7.2: Verify slm --version**

```bash
PYTHONPATH=src .venv/bin/python -c "import superlocalmemory; print(superlocalmemory.__version__)"
```

Expected: `3.6.20`

- [ ] **Step 7.3: Build the Python package**

```bash
cd /Users/varunpratapbhardwaj/Documents/work/varun-world/Agentic_official/superlocalmemory
rm -rf dist/
.venv/bin/python -m build
ls dist/
```

Expected: `superlocalmemory-3.6.20.tar.gz` and `superlocalmemory-3.6.20-py3-none-any.whl`

- [ ] **Step 7.4: Publish to PyPI**

```bash
source ~/.claude-secrets.env
TWINE_USERNAME=__token__ TWINE_PASSWORD=$PYPI_TOKEN \
  .venv/bin/python -m twine upload dist/superlocalmemory-3.6.20*
```

- [ ] **Step 7.5: Publish to npm**

```bash
source ~/.claude-secrets.env
NPMRC_TMP=$(mktemp)
echo "//registry.npmjs.org/:_authToken=$NPM_TOKEN" > "$NPMRC_TMP"
npm publish --userconfig "$NPMRC_TMP"
rm -f "$NPMRC_TMP"
```

- [ ] **Step 7.6: Push the git tag**

```bash
cd /Users/varunpratapbhardwaj/Documents/work/varun-world/Agentic_official/superlocalmemory
git tag v3.6.20
git push origin main --tags
```

- [ ] **Step 7.7: Verify live on PyPI**

```bash
pip index versions superlocalmemory 2>/dev/null | head -3
```

Expected: `3.6.20` appears in the list.

---

## Task 8: Reply to issue #60

- [ ] **Step 8.1: Post reply to GitHub issue #60**

```bash
gh issue comment 60 --repo qualixar/superlocalmemory --body "$(cat <<'EOF'
Thank you for the detailed report and the clear reproduction table — this was very helpful.

**Root cause confirmed:** The v3.6.12 `_get_broker` security gate introduced `X-Mesh-Secret` as the required header for remote mesh API calls. However, both the documentation and `RemoteSyncClient` (SLM's own mesh client) use `Authorization: Bearer <secret>`. The two implementations were inconsistent, causing the exact failure you observed.

**Fix in v3.6.20 (just released):**
- `_get_broker` now accepts **both** `Authorization: Bearer <secret>` (canonical, per RFC 7617) and `X-Mesh-Secret: <secret>` (legacy, backwards compat).
- Removed a dead secondary auth function (`_validate_remote_auth`) that had a related bug where it would incorrectly block loopback callers when `SLM_MESH_HOST=0.0.0.0`.
- Updated `docs/distributed-deployment.md` to clearly document which header to use.

**To fix your LXC setup:**

```bash
pip install --upgrade superlocalmemory==3.6.20
```

Your existing env config (`SLM_MESH_HOST=0.0.0.0`, `SLM_MESH_SHARED_SECRET=...`, `SLM_REMOTE=1`, `SLM_MCP_ALLOWED_HOSTS=*`) is correct. After upgrading, this will work:

```bash
curl -H "Authorization: Bearer VIZO6eZQY4xwZDnAaQ99frSdNf4z" http://192.168.50.144:8765/mesh/status
```

We've added 7 new tests that cover the Bearer token path on all major mesh endpoints so this cannot regress silently again. Thank you again for the LXC reproduction — we had no cross-machine test coverage before this issue.
EOF
)"
```

---

## Self-Review Checklist

**Spec coverage:**
- [x] Issue #60: `Authorization: Bearer` rejected → fixed in Task 2
- [x] `/mesh/status` not working → now covered by test in Task 1 + fixed by Task 2
- [x] `/mesh/peers` not working for remote → fixed + test in Task 1
- [x] `/mesh/send` not working for remote → fixed + test in Task 1
- [x] Backwards compat for `X-Mesh-Secret` → tested in Task 1 Step 1.1
- [x] Dead code `_validate_remote_auth` removed → Task 3
- [x] Docs updated → Task 5
- [x] Version bump all 5 sources → Task 6
- [x] Full suite green before publish → Task 7

**Placeholder scan:** None found.

**Type consistency:** No new types introduced — all changes are within existing functions. `hmac.compare_digest(str, str)` → `bool`. `str.removeprefix(str)` → `str`. No changes to function signatures.

**Potential regressions:**
- Existing `test_secret_accepts_correct_header` (X-Mesh-Secret) → still passes because the new code accepts X-Mesh-Secret first (`or` short-circuits)
- Existing `test_secret_required_for_nonloopback` → still passes because no auth = `presented = ""` = `not presented` guard raises 401
- No other auth code in the router was touched

**Security:**
- `hmac.compare_digest` is timing-safe — maintained in the fix
- Bearer token in `Authorization` header is no less secure than `X-Mesh-Secret` — same value, same constant-time comparison
- Loopback exemption is unchanged — local callers still skip auth
- The `not presented` guard prevents empty-string timing oracle on the `compare_digest` call
