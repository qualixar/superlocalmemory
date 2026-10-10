# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V4 | https://qualixar.com | https://varunpratap.com

"""``slm remote`` - let AI tools on other computers use this SuperLocalMemory.

    slm remote tls init --name <dns>... --ip <addr>... [--days 397] [--force]
    slm remote enable --listen HOST:PORT
    slm remote disable
    slm remote keys add <name> [--read-only] [--profile <profile>]
    slm remote keys list [--json]
    slm remote keys allow|disallow <name|key_id> mesh|media
    slm remote keys revoke <name|key_id>
    slm remote check [--json]

Nothing here starts a network listener. ``enable`` writes the setting; the
daemon opens the TLS listener on its next start (``slm restart``).
"""

from __future__ import annotations

import ipaddress
import json
import os
import socket
import ssl
import sys
from argparse import Namespace
from datetime import datetime, timedelta, timezone
from pathlib import Path

CA_DAYS = 825
MAX_LEAF_DAYS = 397


def _tls_dir() -> Path:
    from superlocalmemory.server.remote_listener import TLS_DIR, data_path

    return data_path(*TLS_DIR)


def _write_private(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    if tmp.exists():
        tmp.unlink()
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        # 0600 means nothing on Windows; owner-only there too, before any byte.
        from superlocalmemory.infra.owner_only_acl import restrict_to_owner

        restrict_to_owner(tmp)
    except BaseException:
        os.close(fd)
        tmp.unlink(missing_ok=True)
        raise
    with os.fdopen(fd, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _write_public(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_bytes(data)
    os.chmod(tmp, 0o644)
    os.replace(tmp, path)


# -- certificates --------------------------------------------------------------------


def _subject_alt_names(names: list[str], ips: list[str]):
    from cryptography import x509

    entries = [x509.DNSName(n.lower().rstrip(".")) for n in names]
    entries += [x509.IPAddress(ipaddress.ip_address(ip)) for ip in ips]
    return entries


def _name_constraints(names: list[str], ips: list[str]):
    from cryptography import x509

    permitted = [x509.DNSName(n.lower().rstrip(".")) for n in names]
    for ip in ips:
        addr = ipaddress.ip_address(ip)
        permitted.append(x509.IPAddress(ipaddress.ip_network(
            f"{addr}/{32 if addr.version == 4 else 128}")))
    return x509.NameConstraints(permitted_subtrees=permitted, excluded_subtrees=None)


def make_ca(names: list[str], ips: list[str]):
    """A CA that can only vouch for the given names and addresses."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    key = ec.generate_private_key(ec.SECP256R1())
    now = datetime.now(timezone.utc)
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME,
                                            "SuperLocalMemory remote access CA")])
    ski = x509.SubjectKeyIdentifier.from_public_key(key.public_key())
    cert = (
        x509.CertificateBuilder().subject_name(subject).issuer_name(subject)
        .public_key(key.public_key()).serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=5))
        .not_valid_after(now + timedelta(days=CA_DAYS))
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(x509.KeyUsage(
            digital_signature=False, content_commitment=False, key_encipherment=False,
            data_encipherment=False, key_agreement=False, key_cert_sign=True,
            crl_sign=True, encipher_only=False, decipher_only=False), critical=True)
        .add_extension(ski, critical=False)
        .add_extension(_name_constraints(names, ips), critical=True)
        .sign(key, hashes.SHA256())
    )
    return cert, key


def make_leaf(ca_cert, ca_key, names: list[str], ips: list[str], days: int):
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

    key = ec.generate_private_key(ec.SECP256R1())
    now = datetime.now(timezone.utc)
    common = (names or ips)[0]
    cert = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, common)]))
        .issuer_name(ca_cert.subject).public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=5))
        .not_valid_after(min(now + timedelta(days=days), ca_cert.not_valid_after_utc))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(x509.KeyUsage(
            digital_signature=True, content_commitment=False, key_encipherment=False,
            data_encipherment=False, key_agreement=False, key_cert_sign=False,
            crl_sign=False, encipher_only=False, decipher_only=False), critical=True)
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]),
                       critical=False)
        .add_extension(x509.SubjectAlternativeName(_subject_alt_names(names, ips)),
                       critical=False)
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()),
                       critical=False)
        .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(
            ca_key.public_key()), critical=False)
        .sign(ca_key, hashes.SHA256())
    )
    return cert, key


def _pem_cert(cert) -> bytes:
    from cryptography.hazmat.primitives import serialization

    return cert.public_bytes(serialization.Encoding.PEM)


def _pem_key(key) -> bytes:
    from cryptography.hazmat.primitives import serialization

    return key.private_bytes(serialization.Encoding.PEM,
                             serialization.PrivateFormat.PKCS8,
                             serialization.NoEncryption())


def fingerprint(cert) -> str:
    from cryptography.hazmat.primitives import hashes

    return cert.fingerprint(hashes.SHA256()).hex(":").upper()


def tls_init(names: list[str], ips: list[str], days: int, force: bool) -> dict:
    from cryptography import x509
    from cryptography.hazmat.primitives import serialization

    names = [n.strip() for n in names if n and n.strip()]
    ips = [i.strip() for i in ips if i and i.strip()]
    if not names and not ips:
        raise ValueError("Give at least one --name or --ip that the other computers will use.")
    for ip in ips:
        ipaddress.ip_address(ip)
    if not 1 <= days <= MAX_LEAF_DAYS:
        raise ValueError(f"--days must be 1 to {MAX_LEAF_DAYS}.")
    folder = _tls_dir()
    ca_pem, ca_key_path = folder / "ca.pem", folder / "ca.key"
    leaf_pem, leaf_key = folder / "server.pem", folder / "server.key"
    if leaf_pem.exists() and not force:
        raise ValueError(f"{leaf_pem} already exists. Use --force to issue a new server "
                         "certificate with the existing CA.")
    if ca_pem.exists() and ca_key_path.exists():
        ca_cert = x509.load_pem_x509_certificate(ca_pem.read_bytes())
        ca_key = serialization.load_pem_private_key(ca_key_path.read_bytes(), password=None)
        created_ca = False
    else:
        ca_cert, ca_key = make_ca(names, ips)
        _write_private(ca_key_path, _pem_key(ca_key))
        _write_public(ca_pem, _pem_cert(ca_cert))
        created_ca = True
    cert, key = make_leaf(ca_cert, ca_key, names, ips, days)
    _write_private(leaf_key, _pem_key(key))
    _write_public(leaf_pem, _pem_cert(cert))
    return {"ca": str(ca_pem), "ca_sha256": fingerprint(ca_cert), "created_ca": created_ca,
            "server_cert": str(leaf_pem), "server_key": str(leaf_key),
            "names": names, "ips": ips,
            "expires": cert.not_valid_after_utc.isoformat()}


# -- settings ------------------------------------------------------------------------


def _settings_path() -> Path:
    from superlocalmemory.server.remote_listener import CONFIG_FILE, data_path

    return data_path(*CONFIG_FILE)


def enable(listen: str) -> dict:
    from superlocalmemory.server.remote_listener import parse_listen

    host, port = parse_listen(listen)
    settings = {"listen": f"[{host}]:{port}" if ":" in host else f"{host}:{port}"}
    _write_private(_settings_path(), (json.dumps(settings, indent=2) + "\n").encode())
    return settings


def disable() -> bool:
    path = _settings_path()
    if not path.exists():
        return False
    path.unlink()
    return True


# -- check -----------------------------------------------------------------------------


def _probe(cfg, ca_path: Path) -> tuple[str, str]:
    host = "127.0.0.1" if cfg.host in ("0.0.0.0", "::") else cfg.host
    server_name = cfg.server_names[0].strip("[]")
    context = ssl.create_default_context(cafile=str(ca_path))
    try:
        with socket.create_connection((host, cfg.port), timeout=3) as raw:
            with context.wrap_socket(raw, server_hostname=server_name):
                return "PASS", f"TLS handshake to {host}:{cfg.port} verified with ca.pem"
    except ConnectionRefusedError:
        return "WARN", f"nothing is listening on {host}:{cfg.port} (run: slm restart)"
    except ssl.SSLError as exc:
        return "FAIL", f"TLS handshake failed: {exc.reason or exc}"
    except OSError as exc:
        return "WARN", f"could not reach {host}:{cfg.port}: {exc}"


def run_checks(probe: bool = True) -> list[dict]:
    from superlocalmemory.server.remote_access import PLAINTEXT_ENV, plaintext_allowed
    from superlocalmemory.server.remote_keys import default_store, store_problem
    from superlocalmemory.server.remote_listener import RemoteListenerError

    results: list[dict] = []

    def add(status: str, check: str, detail: str) -> None:
        results.append({"status": status, "check": check, "detail": detail})

    from superlocalmemory.server.remote_listener import load_remote_listener_config

    main_port = int(os.environ.get("SLM_DAEMON_PORT", "") or 8765)
    cfg = None
    try:
        cfg = load_remote_listener_config(main_port=main_port)
    except RemoteListenerError as exc:
        add("FAIL", "listener", f"{exc.code}: {exc}")
    if cfg is None and not results:
        add("INFO", "listener",
            "remote access is off (enable: slm remote enable --listen HOST:PORT)")
    if cfg is not None:
        add("PASS", "listener", f"https://{cfg.host}:{cfg.port} for {', '.join(cfg.server_names)}")
        days = (cfg.not_after - datetime.now(timezone.utc)).days
        add("WARN" if days < 30 else "PASS", "certificate", f"expires in {days} days")
    store = default_store()
    problem = store_problem(store.path)
    if problem:
        add("FAIL", "key store", problem + " - every remote key is refused")
    else:
        active = [k for k in store.list() if k.active]
        add("PASS" if active else "WARN", "keys",
            f"{len(active)} active key(s)" if active
            else "no active keys (slm remote keys add <name>)")
        unbound = [k.name for k in active if k.profile is None]
        if unbound:
            add("WARN", "key profiles",
                f"{len(unbound)} key(s) not bound to a profile yet and refused: "
                f"{', '.join(unbound)} (run: slm remote keys list)")
    try:
        from superlocalmemory.server.remote_access import company_mode_active

        if company_mode_active(None):
            add("FAIL", "company mode", "company mode refuses remote keys")
    except Exception as exc:  # noqa: BLE001
        add("FAIL", "company mode", f"cannot read deployment mode: {exc}")
    bind = os.environ.get("SLM_DAEMON_HOST") or os.environ.get("SLM_HOST") or "127.0.0.1"
    if bind not in ("127.0.0.1", "::1", "localhost"):
        add("WARN", "main listener", f"SLM_DAEMON_HOST={bind} serves plain HTTP to the network; "
            "use the remote listener instead")
    if plaintext_allowed():
        add("WARN", "plaintext", f"{PLAINTEXT_ENV}=1 accepts remote MCP over plain HTTP")
    if cfg is not None and probe:
        ca = _tls_dir() / "ca.pem"
        if ca.exists():
            status, detail = _probe(cfg, ca)
            add(status, "tls probe", detail)
        else:
            add("WARN", "tls probe", "no ca.pem here; probe skipped (own certificate in use)")
    return results


# -- profiles ------------------------------------------------------------------------


def active_profile() -> str:
    """The profile this computer is using, read without side effects."""
    from superlocalmemory.infra.data_root import state_path

    for name, field in (("config.json", "active_profile"), ("profiles.json", "active_profile")):
        try:
            value = json.loads(Path(state_path(name)).read_text(encoding="utf-8")).get(field)
        except (OSError, ValueError, AttributeError):
            continue
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "default"


def profile_exists(profile: str) -> bool:
    """Whether ``profile`` exists in the memory store (read-only check)."""
    import sqlite3

    from superlocalmemory.infra.data_root import state_path

    if profile == "default":
        return True
    db = Path(state_path("memory.db"))
    if not db.exists():
        return False
    try:
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=10)
        try:
            row = conn.execute("SELECT 1 FROM profiles WHERE profile_id = ?",
                               (profile,)).fetchone()
        finally:
            conn.close()
    except sqlite3.Error as exc:
        raise ValueError(f"Cannot read the profile list ({exc}).") from exc
    return row is not None


def _bind_unbound_keys(store) -> None:
    """Bind keys made before 4.1.20 to the active profile, and say so."""
    bound = store.bind_unbound(active_profile())
    for key in bound:
        print(f"Note: remote key '{key.name}' was made before keys were tied to one profile. "
              f"It is now bound to profile '{key.profile}' (the active profile) and can "
              "reach only that profile. To use another profile, revoke it and run "
              f"'slm remote keys add {key.name} --profile <profile>'.", file=sys.stderr)


# -- CLI glue ------------------------------------------------------------------------


def _print_key_snippet(name: str, secret: str, profile: str, scope: str) -> None:
    print(f"Remote key '{name}' created. This is the only time it is shown:\n")
    print(f"  {secret}\n")
    reach = "read" if scope == "read" else "read and save to"
    print(f"It can {reach} profile '{profile}' only. Whoever holds it can read the full "
          "text of every memory in that profile, including any paths, names or other "
          "details written into those memories. Give it only to tools you trust with "
          "that profile.\n")
    print("Hermes (~/.hermes/config.yaml), with the key in your Hermes secrets as SLM_REMOTE_KEY:")
    print("  mcp_servers:\n    superlocalmemory:\n      url: \"https://<slm-host>:<port>/mcp/hermes\"")
    print("      headers:\n        Authorization: \"Bearer ${SLM_REMOTE_KEY}\"")
    print("      ssl_verify: \"/path/to/slm-ca.pem\"")
    print("\nNever put the key in a URL. Revoke it any time: slm remote keys revoke " + name)


def cmd_remote(args: Namespace) -> None:
    action = getattr(args, "remote_command", None)
    try:
        if action == "tls":
            if getattr(args, "tls_command", None) != "init":
                print("Usage: slm remote tls init --name <host> --ip <address>")
                sys.exit(2)
            info = tls_init(args.name or [], args.ip or [], args.days, args.force)
            print(f"Server certificate: {info['server_cert']} (expires {info['expires']})")
            print(f"CA certificate:     {info['ca']}")
            print(f"CA SHA-256:         {info['ca_sha256']}")
            print("Copy ca.pem to each computer that connects. It is public, not a secret.")
            return
        if action == "enable":
            settings = enable(args.listen)
            print(f"Remote access set to {settings['listen']}. Run 'slm restart' to apply, "
                  "then 'slm remote check'.")
            return
        if action == "disable":
            print("Remote access disabled. Run 'slm restart' to apply." if disable()
                  else "Remote access was not enabled.")
            return
        if action == "keys":
            _cmd_keys(args)
            return
        if action == "check":
            results = run_checks()
            if getattr(args, "json", False):
                # The standard CLI envelope (success/command/version/data),
                # like `slm doctor --json`; exit status still says pass/fail.
                from superlocalmemory.cli.json_output import json_print

                summary = {status.lower(): sum(r["status"] == status for r in results)
                           for status in ("PASS", "WARN", "FAIL", "INFO")}
                json_print("remote check", data={"results": results, "summary": summary})
            else:
                for r in results:
                    print(f"[{r['status']:<4}] {r['check']}: {r['detail']}")
            if any(r["status"] == "FAIL" for r in results):
                sys.exit(1)
            return
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(2)
    print("Usage: slm remote {tls,enable,disable,keys,check} ...")
    sys.exit(2)


def _cmd_keys(args: Namespace) -> None:
    from superlocalmemory.server.remote_keys import default_store

    store = default_store()
    sub = getattr(args, "keys_command", None)
    if sub in ("add", "list", "revoke", "allow", "disallow"):
        _bind_unbound_keys(store)
    if sub == "add":
        chosen = getattr(args, "profile", None)
        profile = chosen.strip() if isinstance(chosen, str) and chosen.strip() else active_profile()
        if not profile_exists(profile):
            raise ValueError(f"Profile '{profile}' does not exist. Create it first: "
                             f"slm profile create {profile}")
        scope = "read" if args.read_only else "write"
        record, secret = store.add(args.key_name, scope, profile=profile,
                                   profile_source="chosen" if chosen else "active-at-creation")
        _print_key_snippet(record.name, secret, profile, scope)
        return
    if sub == "list":
        rows = [k.public() for k in store.list()]
        if getattr(args, "json", False):
            from superlocalmemory.cli.json_output import json_print

            json_print("remote keys list", data={"keys": rows})
            return
        if not rows:
            print("No remote keys. Create one: slm remote keys add <name>")
        for row in rows:
            state = f"revoked {row['revoked_at']}" if row["revoked_at"] else "active"
            profile = row["profile"] or "(unbound - refused)"
            if row["profile_source"] == "bound-on-upgrade":
                profile += " (bound on upgrade)"
            print(f"{row['name']:<24} {row['key_id']:<12} {row['scope']:<5} "
                  f"profile {profile:<28} created {row['created_at']}  {state}")
        return
    if sub in ("allow", "disallow"):
        _change_extra(store, args, allow=sub == "allow")
        return
    if sub == "revoke":
        record = store.revoke(args.key_ref)
        print(f"Revoked '{record.name}' ({record.key_id}). It stops working on its next request.")
        return
    print("Usage: slm remote keys {add,list,allow,disallow,revoke}")
    sys.exit(2)


def _change_extra(store, args: Namespace, *, allow: bool) -> None:
    """Opt a key in to (or out of) mesh or media, keeping its other opt-ins."""
    current = next((k for k in store.list()
                    if k.active and args.key_ref in (k.name, k.key_id)), None)
    if current is None:
        raise ValueError(f"No active remote key is named '{args.key_ref}'.")
    extras = set(current.extras)
    (extras.add if allow else extras.discard)(args.extra)
    record = store.set_extras(args.key_ref, extras)
    if getattr(args, "json", False):
        from superlocalmemory.cli.json_output import json_print

        json_print("remote keys " + ("allow" if allow else "disallow"),
                   data={"key": record.public()})
        return
    verb = "may now use" if allow else "may no longer use"
    print(f"Remote key '{record.name}' {verb} {args.extra}. Takes effect on its next request.")


def add_parser(sub) -> None:
    """Register ``slm remote`` on the top-level subparsers."""
    remote = sub.add_parser(
        "remote", help="Let AI tools on other computers use this SLM (TLS, keys)")
    rsub = remote.add_subparsers(dest="remote_command")
    tls = rsub.add_parser("tls", help="Create the TLS certificate for remote access")
    tsub = tls.add_subparsers(dest="tls_command")
    init = tsub.add_parser("init", help="Create a CA and a server certificate")
    init.add_argument("--name", action="append", help="Host name other computers use (repeat)")
    init.add_argument("--ip", action="append", help="IP address other computers use (repeat)")
    init.add_argument("--days", type=int, default=MAX_LEAF_DAYS,
                      help=f"Server certificate lifetime (max {MAX_LEAF_DAYS})")
    init.add_argument("--force", action="store_true",
                      help="Issue a new server certificate with the existing CA")
    en = rsub.add_parser("enable", help="Turn remote access on (applies on restart)")
    en.add_argument("--listen", required=True, help="HOST:PORT for the TLS listener")
    rsub.add_parser("disable", help="Turn remote access off (applies on restart)")
    keys = rsub.add_parser("keys", help="Create, list and revoke remote keys")
    ksub = keys.add_subparsers(dest="keys_command")
    add = ksub.add_parser(
        "add", help="Create a key (shown once), bound to one profile",
        description="Create a remote key. The key reaches exactly one profile: --profile, "
                    "or the profile active now. Its holder can read the full text of every "
                    "memory in that profile.")
    add.add_argument("key_name", help="A name for the tool or computer, e.g. hermes-laptop")
    add.add_argument("--read-only", action="store_true", help="Recall only, no saves")
    add.add_argument("--profile", default=None,
                     help="The one profile this key may reach (default: the active profile)")
    lst = ksub.add_parser("list", help="List keys (never shows secrets)")
    lst.add_argument("--json", action="store_true")
    for verb, text in (("allow", "Let a key use mesh (other bots) or media (images and documents)"),
                       ("disallow", "Stop a key using mesh or media")):
        extra = ksub.add_parser(verb, help=text)
        extra.add_argument("key_ref", help="Key name or key id")
        extra.add_argument("extra", choices=("mesh", "media"))
        extra.add_argument("--json", action="store_true")
    rev = ksub.add_parser("revoke", help="Revoke a key; effective on its next request")
    rev.add_argument("key_ref", help="Key name or key id")
    chk = rsub.add_parser("check", help="Check remote access; exit 1 on any failure")
    chk.add_argument("--json", action="store_true")


__all__ = ["active_profile", "add_parser", "cmd_remote", "enable", "disable", "make_ca",
           "make_leaf", "profile_exists", "run_checks", "tls_init"]
