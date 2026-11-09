"""Sample Python module for testing."""

import os
from pathlib import Path
from typing import Optional

from utils.helpers import validate_token


class AuthService:
    """Authentication service."""

    def __init__(self, config: dict):
        self.config = config

    def authenticate(self, token: str) -> Optional[dict]:
        """Authenticate a user by token."""
        if not validate_token(token):
            return None
        return self._lookup_user(token)

    def _lookup_user(self, token: str) -> dict:
        """Internal: look up user from token."""
        return {"user": "test"}


def create_auth_service(config: dict) -> AuthService:
    """Factory function."""
    return AuthService(config)


class AdminService(AuthService):
    """Admin service extending AuthService."""

    def promote_user(self, user_id: str) -> bool:
        result = self.authenticate("admin_token")
        return result is not None
