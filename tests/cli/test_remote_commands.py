# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""``slm remote``: certificates that verify under strict defaults, keys shown once."""

from __future__ import annotations

import json
import os
import socket
import ssl
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from tests._portable import child_env_base

from superlocalmemory.cli import remote_commands

REPO = Path(__file__).resolve().parents[2]
#: What ssl.create_default_context() adds on Python 3.13 and later.
STRICT_FLAGS = ssl.VERIFY_X509_STRICT | ssl.VERIFY_X509_PARTIAL_CHAIN


def _slm(tmp_path: Path, *argv: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": str(REPO / "src"), "SLM_DATA_DIR": str(tmp_path),
           **child_env_base(tmp_path / "home"), "SLM_SKIP_FIRST_USE": "1"}
    return subprocess.run([sys.executable, "-m", "superlocalmemory.cli.main", *argv],
                          env=env, capture_output=True, text=True, timeout=120)


def _serve_tls(cert: str, key: str):
    import uvicorn

    async def app(scope, receive, send):
        if scope["type"] != "http":
            return
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    sock = socket.socket()
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, lifespan="off",
                                           ssl_certfile=cert, ssl_keyfile=key,
                                           log_level="warning"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    deadline = time.monotonic() + 15
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.05)
    return server, thread, port


def test_tls_init_produces_a_chain_that_verifies_under_strict_flags() -> None:
    info = remote_commands.tls_init(["localhost"], ["127.0.0.1"], 30, force=False)
    server, thread, port = _serve_tls(info["server_cert"], info["server_key"])
    try:
        context = ssl.create_default_context(cafile=info["ca"])
        # Python 3.13+ clients verify strictly by default; 3.12 does not, so
        # ask for the same checks explicitly and test every version alike.
        context.verify_flags |= STRICT_FLAGS
        assert context.verify_flags & STRICT_FLAGS == STRICT_FLAGS
        for host in ("localhost", "127.0.0.1"):
            with socket.create_connection(("127.0.0.1", port), timeout=5) as raw:
                with context.wrap_socket(raw, server_hostname=host) as tls:
                    assert tls.version() in ("TLSv1.2", "TLSv1.3")
        # Without the SLM CA the system trust store refuses it.
        with socket.create_connection(("127.0.0.1", port), timeout=5) as raw:
            with pytest.raises(ssl.SSLCertVerificationError):
                ssl.create_default_context().wrap_socket(raw, server_hostname="localhost")
        # A name not on the certificate is refused.
        with socket.create_connection(("127.0.0.1", port), timeout=5) as raw:
            with pytest.raises(ssl.SSLCertVerificationError):
                ssl.create_default_context(cafile=info["ca"]).wrap_socket(
                    raw, server_hostname="evil.example")
    finally:
        server.should_exit = True
        thread.join(10)


def test_ca_name_constraints_reject_a_leaf_for_another_name(tmp_path) -> None:
    """A stolen SLM CA key cannot mint a certificate for another site."""
    from cryptography.hazmat.primitives import serialization

    info = remote_commands.tls_init(["localhost"], ["127.0.0.1"], 30, force=False)
    ca_key = serialization.load_pem_private_key(
        (Path(info["ca"]).parent / "ca.key").read_bytes(), password=None)
    from cryptography import x509

    ca_cert = x509.load_pem_x509_certificate(Path(info["ca"]).read_bytes())
    evil_cert, evil_key = remote_commands.make_leaf(ca_cert, ca_key, ["evil.example"], [], 30)
    cert_path, key_path = tmp_path / "evil.pem", tmp_path / "evil.key"
    cert_path.write_bytes(evil_cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(evil_key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption()))
    server, thread, port = _serve_tls(str(cert_path), str(key_path))
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=5) as raw:
            with pytest.raises(ssl.SSLCertVerificationError):
                ssl.create_default_context(cafile=info["ca"]).wrap_socket(
                    raw, server_hostname="evil.example")
    finally:
        server.should_exit = True
        thread.join(10)


def test_tls_init_refuses_to_overwrite_without_force_and_keeps_the_ca() -> None:
    first = remote_commands.tls_init(["localhost"], [], 30, force=False)
    with pytest.raises(ValueError):
        remote_commands.tls_init(["localhost"], [], 30, force=False)
    second = remote_commands.tls_init(["localhost"], [], 30, force=True)
    assert second["ca_sha256"] == first["ca_sha256"] and not second["created_ca"]
    if os.name == "posix":
        tls = Path(first["ca"]).parent
        for private in ("ca.key", "server.key"):
            assert (tls / private).stat().st_mode & 0o077 == 0


@pytest.mark.parametrize("names,ips,days", [([], [], 30), (["x"], ["not-an-ip"], 30),
                                            (["x"], [], 0), (["x"], [], 398)])
def test_tls_init_rejects_bad_input(names, ips, days) -> None:
    with pytest.raises(ValueError):
        remote_commands.tls_init(names, ips, days, force=False)


def test_keys_add_prints_the_secret_once_and_list_never_shows_it(tmp_path) -> None:
    added = _slm(tmp_path, "remote", "keys", "add", "hermes-laptop")
    assert added.returncode == 0, added.stderr[-2000:]
    secret = next(tok for tok in added.stdout.split() if tok.startswith("slmr_"))
    assert "Bearer ${SLM_REMOTE_KEY}" in added.stdout
    listed = _slm(tmp_path, "remote", "keys", "list", "--json")
    assert listed.returncode == 0, listed.stderr[-2000:]
    rows = json.loads(listed.stdout)["data"]["keys"]
    assert rows[0]["name"] == "hermes-laptop" and rows[0]["scope"] == "write"
    assert secret not in listed.stdout and "digest" not in listed.stdout
    stored = (tmp_path / "remote_keys.json").read_text(encoding="utf-8")
    assert secret not in stored
    ro = _slm(tmp_path, "remote", "keys", "add", "viewer", "--read-only")
    assert ro.returncode == 0
    revoked = _slm(tmp_path, "remote", "keys", "revoke", "viewer")
    assert revoked.returncode == 0 and "Revoked 'viewer'" in revoked.stdout
    again = _slm(tmp_path, "remote", "keys", "revoke", "viewer")
    assert again.returncode == 2


def test_check_exit_code_is_nonzero_on_a_failure(tmp_path) -> None:
    off = _slm(tmp_path, "remote", "check", "--json")
    assert off.returncode == 0, off.stdout + off.stderr[-1000:]
    enabled = _slm(tmp_path, "remote", "enable", "--listen", "127.0.0.1:18443")
    assert enabled.returncode == 0 and "slm restart" in enabled.stdout
    broken = _slm(tmp_path, "remote", "check", "--json")
    assert broken.returncode == 1
    failures = [r for r in json.loads(broken.stdout)["data"]["results"] if r["status"] == "FAIL"]
    assert any("tls_missing" in r["detail"] for r in failures)
    disabled = _slm(tmp_path, "remote", "disable")
    assert disabled.returncode == 0


def test_check_flags_a_world_readable_key_store(tmp_path) -> None:
    if os.name != "posix":
        pytest.skip("POSIX permissions")
    assert _slm(tmp_path, "remote", "keys", "add", "a").returncode == 0
    os.chmod(tmp_path / "remote_keys.json", 0o644)
    result = _slm(tmp_path, "remote", "check", "--json")
    assert result.returncode == 1
    assert any(r["check"] == "key store" and r["status"] == "FAIL"
               for r in json.loads(result.stdout)["data"]["results"])


# -- keys are bound to one profile (audit 4.1.20 L2 F2) ----------------------------------


def test_keys_add_binds_the_active_profile_and_warns_about_memory_text(tmp_path) -> None:
    added = _slm(tmp_path, "remote", "keys", "add", "hermes")
    assert added.returncode == 0, added.stderr[-2000:]
    assert "profile 'default' only" in added.stdout
    assert "full text of every memory" in added.stdout
    row = json.loads(_slm(tmp_path, "remote", "keys", "list", "--json").stdout)["data"]["keys"][0]
    assert row["profile"] == "default" and row["profile_source"] == "active-at-creation"
    listed = _slm(tmp_path, "remote", "keys", "list")
    assert "profile default" in listed.stdout


def test_keys_add_refuses_a_profile_that_does_not_exist(tmp_path) -> None:
    bad = _slm(tmp_path, "remote", "keys", "add", "ghost", "--profile", "no-such-profile")
    assert bad.returncode == 2 and "does not exist" in bad.stderr
    assert "slmr_" not in bad.stdout
    assert not (tmp_path / "remote_keys.json").exists()


def test_keys_add_profile_flag_is_documented(tmp_path) -> None:
    helped = _slm(tmp_path, "remote", "keys", "add", "--help")
    assert "--profile" in helped.stdout and "full text of every" in helped.stdout


def test_keys_list_binds_a_pre_4_1_20_key_and_tells_the_user(tmp_path) -> None:
    from superlocalmemory.server.remote_keys import KEY_PREFIX, digest_secret

    (tmp_path / "config.json").write_text(json.dumps({"active_profile": "work"}), encoding="utf-8")
    store = tmp_path / "remote_keys.json"
    store.write_text(json.dumps({"version": 1, "keys": [{
        "key_id": "rk_0000abcd", "name": "old-hermes", "scope": "write",
        "digest": digest_secret(KEY_PREFIX + "A" * 43),
        "created_at": "2026-09-01T00:00:00+00:00", "revoked_at": None}]}), encoding="utf-8")
    os.chmod(store, 0o600)
    listed = _slm(tmp_path, "remote", "keys", "list")
    assert listed.returncode == 0, listed.stderr[-2000:]
    assert "old-hermes" in listed.stderr and "bound to profile 'work'" in listed.stderr
    assert "work (bound on upgrade)" in listed.stdout
    again = _slm(tmp_path, "remote", "keys", "list")
    assert "bound to profile" not in again.stderr  # told once, recorded for good
    assert json.loads(store.read_text(encoding="utf-8"))["keys"][0]["profile"] == "work"


def test_keys_allow_and_disallow_mesh_and_media(tmp_path) -> None:
    assert _slm(tmp_path, "remote", "keys", "add", "web-x").returncode == 0
    allowed = _slm(tmp_path, "remote", "keys", "allow", "web-x", "mesh")
    assert allowed.returncode == 0, allowed.stderr
    again = _slm(tmp_path, "remote", "keys", "allow", "web-x", "media", "--json")
    assert again.returncode == 0, again.stderr
    assert json.loads(again.stdout)["data"]["key"]["extras"] == ["media", "mesh"]
    rows = json.loads(_slm(tmp_path, "remote", "keys", "list", "--json").stdout)["data"]["keys"]
    assert rows[0]["extras"] == ["media", "mesh"]
    gone = _slm(tmp_path, "remote", "keys", "disallow", "web-x", "mesh")
    assert gone.returncode == 0, gone.stderr
    rows = json.loads(_slm(tmp_path, "remote", "keys", "list", "--json").stdout)["data"]["keys"]
    assert rows[0]["extras"] == ["media"]


def test_keys_allow_rejects_unknown_extra_and_unknown_key(tmp_path) -> None:
    assert _slm(tmp_path, "remote", "keys", "add", "web-x").returncode == 0
    assert _slm(tmp_path, "remote", "keys", "allow", "web-x", "admin").returncode != 0
    missing = _slm(tmp_path, "remote", "keys", "allow", "nope", "mesh")
    assert missing.returncode != 0 and "nope" in (missing.stderr + missing.stdout)
