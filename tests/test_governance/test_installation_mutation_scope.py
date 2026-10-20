"""Installation mutations require authority over every workspace."""

# ruff: noqa: F811
import pytest

from tests.test_governance.test_role_surface_matrix_rest import _company, matrix  # noqa: F401
from tests.test_server.test_rbac_company_mode_credential import app_and_engine  # noqa: F401


@pytest.mark.parametrize(
    "path,body",
    [
        ("/api/integrity/repair", {"root": "/sandbox"}),
        ("/api/integrity/repair-now", {}),
        ("/api/integrity/repair-entities", {"root": "/sandbox"}),
        ("/maintenance/run", {}),
        ("/maintenance/compact", {}),
        ("/maintenance/regraph", {}),
    ],
)
def test_scoped_admin_cannot_mutate_installation(matrix, path, body):
    app, client, who, _ = matrix
    _company(app)
    response = client.post(path, json=body, headers=who["admin_b"])
    assert response.status_code == 403, response.text
    assert "every workspace" in response.text or "installation" in response.text
