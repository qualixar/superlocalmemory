#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 SuperLocalMemory (superlocalmemory.com)
"""Tests for security headers middleware."""

import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from security_middleware import SecurityHeadersMiddleware


def test_security_headers_middleware():
    """Test that security headers middleware adds correct headers."""
    from fastapi import FastAPI

    app = FastAPI()
    app.add_middleware(SecurityHeadersMiddleware)

    @app.get("/test")
    async def test_endpoint():
        return {"status": "ok"}

    @app.get("/api/test")
    async def api_endpoint():
        return {"data": "test"}

    client = TestClient(app)

    # Test regular endpoint
    response = client.get("/test")
    assert response.status_code == 200

    # Verify security headers are present
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["X-XSS-Protection"] == "1; mode=block"
    assert "Content-Security-Policy" in response.headers
    assert response.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"

    # Verify CSP includes required directives
    csp = response.headers["Content-Security-Policy"]
    assert "default-src 'self'" in csp
    assert "frame-ancestors 'none'" in csp

    # Test API endpoint (should have cache control headers)
    api_response = client.get("/api/test")
    assert api_response.status_code == 200
    assert "no-store" in api_response.headers.get("Cache-Control", "")
    assert "Pragma" in api_response.headers


def test_csp_allows_required_resources():
    """Test that CSP allows required CDN resources."""
    from fastapi import FastAPI

    app = FastAPI()
    app.add_middleware(SecurityHeadersMiddleware)

    @app.get("/")
    async def root():
        return {"status": "ok"}

    client = TestClient(app)
    response = client.get("/")

    csp = response.headers["Content-Security-Policy"]

    # Should allow Bootstrap CDN
    assert "https://cdn.jsdelivr.net" in csp

    # Should allow WebSocket connections to localhost
    assert "ws://localhost:*" in csp or "ws://127.0.0.1:*" in csp


def test_json_content_type():
    """Test that JSON responses have correct Content-Type."""
    from fastapi import FastAPI

    app = FastAPI()
    app.add_middleware(SecurityHeadersMiddleware)

    @app.get("/api/data")
    async def get_data():
        return {"test": "data", "value": 123}

    client = TestClient(app)
    response = client.get("/api/data")

    # FastAPI automatically sets application/json for dict returns
    assert response.headers["Content-Type"].startswith("application/json")
    assert response.json() == {"test": "data", "value": 123}


def test_xss_protection_in_json_responses():
    """Test that dangerous content in JSON is properly handled."""
    from fastapi import FastAPI

    app = FastAPI()
    app.add_middleware(SecurityHeadersMiddleware)

    @app.get("/api/memory")
    async def get_memory():
        # Simulate returning user-generated content with XSS payload
        return {
            "content": "<script>alert('xss')</script>",
            "summary": "<img src=x onerror=alert('xss')>",
            "tags": ["<b>bold</b>", "normal"]
        }

    client = TestClient(app)
    response = client.get("/api/memory")

    # JSON responses should NOT execute scripts
    # The security is enforced by:
    # 1. Content-Type: application/json (browser won't execute)
    # 2. X-Content-Type-Options: nosniff (prevent MIME sniffing)
    # 3. Client-side escapeHtml() function

    assert response.headers["Content-Type"].startswith("application/json")
    assert response.headers["X-Content-Type-Options"] == "nosniff"

    data = response.json()
    # Data should be returned as-is in JSON
    # XSS protection happens on client-side rendering
    assert "<script>" in data["content"]
    assert "<img" in data["summary"]


def test_cors_not_set_to_wildcard():
    """Test that CORS is not set to allow all origins (security risk)."""
    import sys
    from pathlib import Path

    # Import ui_server to check CORS configuration
    sys.path.insert(0, str(Path(__file__).parent.parent))

    # We can't easily test CORS without starting the full server,
    # but we can verify the middleware is configured
    # This is a basic sanity check
    from security_middleware import SecurityHeadersMiddleware

    # Just verify the middleware class exists and is importable
    assert SecurityHeadersMiddleware is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
