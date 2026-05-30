"""
End-to-end server tests using FastAPI's TestClient.

These pin three fixes from the v2.1.0 review:

  * /export is no longer broken (it called a non-existent method before).
  * Anonymous requests don't share a mutable PhysicsCompiler.
  * The session store evicts old entries instead of growing forever.

The whole module is skipped if FastAPI isn't installed - the server is
shipped behind the optional ``[server]`` extra.
"""

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from mechanics_dsl.server.app import create_app  # noqa: E402
from mechanics_dsl.server import routes as routes_mod  # noqa: E402


PENDULUM_DSL = (
    r"\system{pendulum}"
    r"\defvar{theta}{Angle}{rad}"
    r"\parameter{m}{1.0}{kg}\parameter{l}{1.0}{m}\parameter{g}{9.81}{m/s^2}"
    r"\lagrangian{0.5*m*l^2*\dot{theta}^2 - m*g*l*(1 - \cos{theta})}"
)


@pytest.fixture
def client():
    """A fresh app + cleared session store for each test."""
    app = create_app()
    routes_mod._sessions.clear()
    with TestClient(app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json()["status"] == "healthy"


def test_compile_returns_system_info(client):
    r = client.post("/api/compile", json={"code": PENDULUM_DSL})
    assert r.status_code == 200
    body = r.json()
    assert body["success"]
    assert body["system_name"] == "pendulum"
    assert body["coordinates"] == ["theta"]


def test_simulate_returns_trajectory(client):
    r = client.post(
        "/api/simulate",
        json={"code": PENDULUM_DSL, "t_start": 0, "t_end": 1.0, "num_points": 50},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["success"], body
    assert len(body["t"]) == 50
    assert len(body["y"]) == 2  # theta and theta_dot
    assert body["coordinates"] == ["theta"]


def test_export_returns_generated_code(client):
    """/export used to call compiler.export(), which didn't exist."""
    r = client.post(
        "/api/export",
        json={"code": PENDULUM_DSL, "target": "python"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["success"], body
    assert body["language"] == "python"
    assert body["code"] and len(body["code"]) > 100


def test_export_rejects_unknown_target(client):
    r = client.post(
        "/api/export",
        json={"code": PENDULUM_DSL, "target": "brainfuck"},
    )
    assert r.status_code == 200
    body = r.json()
    assert not body["success"]
    assert "Invalid target" in body["error"]


def test_anonymous_requests_do_not_share_state(client):
    """Two anonymous /compile calls must not share a PhysicsCompiler."""
    one_dof = (
        r"\system{a}\defvar{x}{Position}{m}\parameter{m}{1.0}{kg}"
        r"\lagrangian{0.5*m*\dot{x}^2}"
    )
    other_dof = (
        r"\system{b}\defvar{y}{Position}{m}\parameter{m}{1.0}{kg}"
        r"\lagrangian{0.5*m*\dot{y}^2}"
    )
    a = client.post("/api/compile", json={"code": one_dof}).json()
    b = client.post("/api/compile", json={"code": other_dof}).json()

    assert a["coordinates"] == ["x"]
    assert b["coordinates"] == ["y"]  # would be ["x", "y"] under the old bug
    # No sessions were keyed, so nothing was cached.
    assert len(routes_mod._sessions) == 0


def test_session_keyed_requests_share_compiler(client):
    r1 = client.post(
        "/api/compile?session_id=alice",
        json={"code": PENDULUM_DSL},
    )
    assert r1.json()["success"]
    assert "alice" in routes_mod._sessions
    same = routes_mod._sessions["alice"]

    r2 = client.post(
        "/api/compile?session_id=alice",
        json={"code": PENDULUM_DSL},
    )
    assert r2.json()["success"]
    # Same PhysicsCompiler instance reused under the same session id.
    assert routes_mod._sessions["alice"] is same


def test_session_store_evicts_oldest_when_full(client, monkeypatch):
    """The bounded LRU must drop the oldest session when the cap is hit."""
    monkeypatch.setattr(routes_mod, "MAX_SESSIONS", 3)
    for sid in ("s1", "s2", "s3", "s4"):
        r = client.post(
            f"/api/compile?session_id={sid}",
            json={"code": PENDULUM_DSL},
        )
        assert r.json()["success"]

    assert len(routes_mod._sessions) == 3
    assert "s1" not in routes_mod._sessions  # oldest evicted
    assert set(routes_mod._sessions) == {"s2", "s3", "s4"}


def test_clear_session_removes_entry(client):
    client.post("/api/compile?session_id=bob", json={"code": PENDULUM_DSL})
    assert "bob" in routes_mod._sessions

    r = client.delete("/api/session/bob")
    assert r.json()["cleared"] is True
    assert "bob" not in routes_mod._sessions


def test_generators_listed(client):
    r = client.get("/api/generators")
    body = r.json()
    assert "cpp" in body["generators"]
    assert "python" in body["generators"]
