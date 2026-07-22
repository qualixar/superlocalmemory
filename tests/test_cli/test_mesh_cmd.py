# M-03 (3.7.9): `slm mesh {status,peers}` terminal inspection.
from argparse import Namespace
from unittest.mock import patch

from superlocalmemory.cli.mesh_cmd import cmd_mesh


def test_mesh_status_prints_broker_state(capsys):
    with patch("superlocalmemory.cli.daemon.daemon_request",
               return_value={"broker_up": True, "peer_count": 2}):
        rc = cmd_mesh(Namespace(mesh_action="status"))
    assert rc == 0
    assert "broker_up" in capsys.readouterr().out


def test_mesh_peers_routes_to_peers_endpoint():
    with patch("superlocalmemory.cli.daemon.daemon_request",
               return_value={"peers": []}) as m:
        rc = cmd_mesh(Namespace(mesh_action="peers"))
    assert rc == 0
    m.assert_called_once_with("GET", "/mesh/peers")


def test_mesh_status_routes_to_status_endpoint():
    with patch("superlocalmemory.cli.daemon.daemon_request",
               return_value={"broker_up": True}) as m:
        cmd_mesh(Namespace(mesh_action="status"))
    m.assert_called_once_with("GET", "/mesh/status")


def test_mesh_daemon_unreachable_returns_1(capsys):
    with patch("superlocalmemory.cli.daemon.daemon_request", return_value=None):
        rc = cmd_mesh(Namespace(mesh_action="status"))
    assert rc == 1
    assert "daemon" in capsys.readouterr().out.lower()


def test_mesh_unknown_action_returns_2(capsys):
    rc = cmd_mesh(Namespace(mesh_action="bogus"))
    assert rc == 2
    assert "Usage" in capsys.readouterr().out
