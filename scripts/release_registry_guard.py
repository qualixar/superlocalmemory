#!/usr/bin/env python3
"""Fail-closed registry state checks for a coordinated PyPI/npm release.

Registry publication cannot be one distributed transaction.  The release
workflow therefore builds both artifacts first, records which immutable
versions already exist, skips those versions on recovery, and finishes only
after both registries report the requested version.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


PACKAGE_NAME = "superlocalmemory"
PYPI_URL = "https://pypi.org/pypi/{name}/{version}/json"
NPM_URL = "https://registry.npmjs.org/{name}/{version}"
USER_AGENT = "superlocalmemory-release-guard/1"


@dataclass(frozen=True)
class RegistryState:
    pypi_exists: bool
    npm_exists: bool


def _registry_payload(url: str, *, timeout: float = 15.0) -> dict | None:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise RuntimeError(f"registry returned HTTP {exc.code} for {url}") from exc
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"could not verify registry state for {url}: {exc}") from exc

    if not isinstance(payload, dict):
        raise RuntimeError(f"registry returned a non-object payload for {url}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha512_integrity(path: Path) -> str:
    digest = hashlib.sha512()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha512-" + base64.b64encode(digest.digest()).decode("ascii")


def _assert_pypi_identity(payload: dict, version: str, dist_dir: Path) -> None:
    published = payload.get("info", {}).get("version")
    if published != version:
        raise RuntimeError(
            f"PyPI version mismatch: expected {version!r}, got {published!r}"
        )
    local = {
        path.name: _sha256(path)
        for path in sorted(dist_dir.iterdir())
        if path.is_file()
    }
    remote = {
        item.get("filename"): item.get("digests", {}).get("sha256")
        for item in payload.get("urls", [])
    }
    if not local or local != remote:
        raise RuntimeError(
            f"PyPI artifact identity mismatch: local={sorted(local)}, remote={sorted(remote)}"
        )


def _assert_npm_identity(payload: dict, version: str, tarball: Path) -> None:
    published = payload.get("version")
    if published != version:
        raise RuntimeError(
            f"npm version mismatch: expected {version!r}, got {published!r}"
        )
    expected = _sha512_integrity(tarball)
    actual = payload.get("dist", {}).get("integrity")
    if actual != expected:
        raise RuntimeError(
            f"npm artifact identity mismatch: expected {expected!r}, got {actual!r}"
        )


def registry_state(
    version: str,
    *,
    pypi_dist: Path,
    npm_tarball: Path,
    fetcher: Callable[[str], dict | None] = _registry_payload,
) -> RegistryState:
    encoded_name = urllib.parse.quote(PACKAGE_NAME, safe="")
    encoded_version = urllib.parse.quote(version, safe="")
    pypi_payload = fetcher(
        PYPI_URL.format(name=encoded_name, version=encoded_version)
    )
    npm_payload = fetcher(NPM_URL.format(name=encoded_name, version=encoded_version))
    if pypi_payload is not None:
        _assert_pypi_identity(pypi_payload, version, pypi_dist)
    if npm_payload is not None:
        _assert_npm_identity(npm_payload, version, npm_tarball)
    return RegistryState(pypi_payload is not None, npm_payload is not None)


def _write_github_output(path: Path, state: RegistryState) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(f"pypi_exists={str(state.pypi_exists).lower()}\n")
        stream.write(f"npm_exists={str(state.npm_exists).lower()}\n")


def _verify_with_retries(
    version: str,
    attempts: int,
    interval: float,
    *,
    pypi_dist: Path,
    npm_tarball: Path,
) -> RegistryState:
    state = RegistryState(False, False)
    for attempt in range(1, attempts + 1):
        state = registry_state(
            version, pypi_dist=pypi_dist, npm_tarball=npm_tarball
        )
        if state.pypi_exists and state.npm_exists:
            return state
        if attempt < attempts:
            time.sleep(interval)
    raise RuntimeError(
        f"registry parity failed for {version}: "
        f"pypi={state.pypi_exists}, npm={state.npm_exists}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode", choices=("check-release-registries", "verify-release-registries")
    )
    parser.add_argument("version")
    parser.add_argument("--python-dist", type=Path, required=True)
    parser.add_argument("--npm-tarball", type=Path, required=True)
    parser.add_argument("--github-output", type=Path)
    parser.add_argument("--attempts", type=int, default=1)
    parser.add_argument("--interval", type=float, default=0.0)
    args = parser.parse_args(argv)

    if args.attempts < 1 or args.interval < 0:
        parser.error("attempts must be positive and interval must be non-negative")

    if args.mode == "check-release-registries":
        state = registry_state(
            args.version,
            pypi_dist=args.python_dist,
            npm_tarball=args.npm_tarball,
        )
        output = args.github_output or (
            Path(os.environ["GITHUB_OUTPUT"]) if os.environ.get("GITHUB_OUTPUT") else None
        )
        if output is None:
            parser.error("check mode requires --github-output or GITHUB_OUTPUT")
        _write_github_output(output, state)
    else:
        state = _verify_with_retries(
            args.version,
            args.attempts,
            args.interval,
            pypi_dist=args.python_dist,
            npm_tarball=args.npm_tarball,
        )

    print(
        json.dumps(
            {
                "version": args.version,
                "pypi_exists": state.pypi_exists,
                "npm_exists": state.npm_exists,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
