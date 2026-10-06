"""Optional local enrollment orchestration; cloud identity remains separate."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Protocol
from urllib.parse import parse_qsl, urlsplit

from superlocalmemory.remote_connections.journal import (
    HOSTS, EnrollmentJournal, Enrollment, JournalConflict,
)


@dataclass(frozen=True)
class GatewayReceipt:
    reference: str
    authorization_url: str | None = None


class GatewayProvider(Protocol):
    async def enroll(self, *, installation_id: str, connection_id: str,
                     owner: str, profile: str, intent: dict) -> GatewayReceipt: ...


def validate_sign_in(url: str, connection_id: str) -> str:
    """Only an owner-login URL for this exact locally journaled connection."""
    if not isinstance(url, str) or len(url) > 2048:
        raise ValueError("invalid_authorization_url")
    parsed = urlsplit(url)
    if (parsed.scheme != "https" or parsed.netloc != "auth.superlocalmemory.com"
            or parsed.path != "/owner-login" or parsed.username or parsed.password or parsed.fragment
            or parse_qsl(parsed.query, keep_blank_values=True) != [("connection_id", connection_id)]):
        raise ValueError("invalid_authorization_url")
    return url


class RemoteConnectionService:
    """Configured only after operator provisioning; absent means local-only.

    Dispatch leases prevent concurrent duplicate enrollments. The gateway must
    also use the stable connection ID as durable idempotency identity after
    network uncertainty or process restart. No connected proof is minted here.
    """

    def __init__(self, journal: EnrollmentJournal, provider: GatewayProvider, *, hosts: tuple[str, ...]):
        if not hosts or len(set(hosts)) != len(hosts) or any(host not in HOSTS for host in hosts):
            raise ValueError("invalid_host_catalog")
        self.journal, self.provider, self.hosts = journal, provider, tuple(hosts)

    @staticmethod
    def public(record: Enrollment) -> dict:
        return {"connection_id": record.connection_id, "host": record.host,
                "state": record.state, "version": record.version, "verified": False,
                "cleanup_pending": record.cleanup_pending}

    def status(self, owner: str, profile: str) -> dict:
        return {"available": True, "installation_id": self.journal.installation_id,
                "current_profile": profile, "hosts": list(self.hosts),
                "connections": [self.public(row) for row in self.journal.list(owner, profile)]}

    async def initiate(self, owner: str, profile: str, key: str, intent: dict) -> dict:
        if intent["host"] not in self.hosts:
            raise JournalConflict("host_unavailable")
        row = await asyncio.to_thread(self.journal.begin, owner, profile, key, intent)
        if row.state != "pending":
            raise JournalConflict("intent_cancelled")
        lease = await asyncio.to_thread(self.journal.claim, owner, profile, row.connection_id)
        if lease is not None:
            receipt = await asyncio.wait_for(self.provider.enroll(
                installation_id=row.installation_id, connection_id=row.connection_id,
                owner=owner, profile=profile, intent=row.intent), timeout=10)
            if not isinstance(receipt, GatewayReceipt):
                raise ValueError("invalid_gateway_receipt")
            if receipt.authorization_url is not None:
                validate_sign_in(receipt.authorization_url, row.connection_id)
            accepted = await asyncio.to_thread(self.journal.acknowledge, owner, profile,
                row.connection_id, lease.token, lease.version, receipt.reference, receipt.authorization_url)
            if not accepted:
                raise JournalConflict("stale_enrollment_receipt")
        # Reread durable truth after an async callback; never manufacture the
        # latest version for a stale callback or publish a cancelled receipt.
        current = await asyncio.to_thread(self.journal.get, owner, profile, row.connection_id)
        if current.state != "pending":
            raise JournalConflict("intent_cancelled")
        response = {"connection_id": current.connection_id, "state": "pending"}
        if current.requested and current.authorization_url:
            response["authorization_url"] = current.authorization_url
        return response
