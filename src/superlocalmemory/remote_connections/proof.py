"""Native installation proof with cryptography/PyJWT, compatible with JOSE."""

from __future__ import annotations

import base64
import hashlib
import json
import time
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec


def _base64url(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


class DeviceSigner:
    def __init__(self, private_pem: str):
        try:
            if not isinstance(private_pem, str) or len(private_pem) > 2048:
                raise ValueError("invalid")
            key = serialization.load_pem_private_key(private_pem.encode("ascii"), password=None)
            if not isinstance(key, ec.EllipticCurvePrivateKey) or not isinstance(
                key.curve, ec.SECP256R1
            ):
                raise ValueError("invalid")
            self._key = key
        except Exception:
            raise ValueError("invalid_device_key") from None

    def __repr__(self) -> str:
        return "DeviceSigner(P-256, private_key=<protected>)"

    @classmethod
    def generate(cls) -> "DeviceSigner":
        key = ec.generate_private_key(ec.SECP256R1())
        return cls(
            key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            ).decode("ascii")
        )

    @property
    def private_pem(self) -> str:
        """For the OS secure store only; never a public API response."""
        return self._key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        ).decode("ascii")

    @property
    def public_jwk(self) -> dict[str, str]:
        numbers = self._key.public_key().public_numbers()
        return {
            "kty": "EC",
            "crv": "P-256",
            "x": _base64url(numbers.x.to_bytes(32, "big")),
            "y": _base64url(numbers.y.to_bytes(32, "big")),
        }

    @property
    def thumbprint(self) -> str:
        canonical = json.dumps(self.public_jwk, sort_keys=True, separators=(",", ":")).encode(
            "ascii"
        )
        return _base64url(hashlib.sha256(canonical).digest())

    def proof(
        self, method: str, url: str, *, token: str | None = None, nonce: str | None = None
    ) -> str:
        target = urlsplit(url)
        if (
            method not in {"GET", "POST", "DELETE", "PUT", "PATCH"}
            or target.scheme != "https"
            or not target.hostname
            or target.username
            or target.password
            or target.fragment
        ):
            raise ValueError("invalid_proof_target")
        uri = urlunsplit((target.scheme, target.netloc, target.path or "/", "", ""))
        claims = {"jti": uuid4().hex, "iat": int(time.time()), "htm": method, "htu": uri}
        if token is not None:
            if not isinstance(token, str) or not token or len(token) > 4096:
                raise ValueError("invalid_proof_token")
            claims["ath"] = _base64url(hashlib.sha256(token.encode("utf-8")).digest())
        if nonce is not None:
            if not isinstance(nonce, str) or not 1 <= len(nonce) <= 256:
                raise ValueError("invalid_proof_nonce")
            claims["nonce"] = nonce
        return jwt.encode(
            claims,
            self._key,
            algorithm="ES256",
            headers={"typ": "dpop+jwt", "jwk": self.public_jwk},
        )
