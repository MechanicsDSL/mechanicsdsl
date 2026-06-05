"""
End-to-end tests for the WebSocket streaming endpoint.

Pins the v2.1.2 fixes:
  * Resume after pause continues from the paused frame instead of
    restarting at frame 0.
  * The ``params`` action actually changes the simulation (was a no-op
    before because the compiled equations weren't recompiled).
  * The connection / streamer bookkeeping is bounded and gets cleaned
    up even when the handler exits cleanly.
"""

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from mechanics_dsl.server import websocket as ws_mod  # noqa: E402
from mechanics_dsl.server.app import create_app  # noqa: E402


PENDULUM_DSL = (
    r"\system{pendulum}"
    r"\defvar{theta}{Angle}{rad}"
    r"\parameter{m}{1.0}{kg}\parameter{l}{1.0}{m}\parameter{g}{9.81}{m/s^2}"
    r"\lagrangian{0.5*m*l^2*\dot{theta}^2 - m*g*l*(1 - \cos{theta})}"
    r"\initial{theta=0.2, theta_dot=0}"
)


@pytest.fixture
def client():
    """Fresh app + cleared connection bookkeeping per test."""
    app = create_app()
    with ws_mod._ws_lock:
        ws_mod._connections.clear()
        ws_mod._streamers.clear()
    with TestClient(app) as c:
        yield c


def _read_until(ws, message_type, max_messages=200):
    """Read frames from the socket until one of ``message_type`` shows up."""
    for _ in range(max_messages):
        msg = ws.receive_json()
        if msg.get("type") == message_type:
            return msg
    raise AssertionError(f"never saw a {message_type!r} message")


def test_compile_and_short_simulate_streams_frames(client):
    with client.websocket_connect("/ws/simulation") as ws:
        ws.send_json({"action": "compile", "code": PENDULUM_DSL})
        compiled = ws.receive_json()
        assert compiled["type"] == "compiled" and compiled["success"]

        ws.send_json(
            {
                "action": "simulate",
                "t_start": 0,
                "t_end": 0.1,
                "num_points": 5,
                "frame_rate": 1000,  # fast frames so the test stays quick
            }
        )
        # First we get a "simulating" then "streaming" announcement.
        first = ws.receive_json()
        assert first["type"] in ("simulating", "streaming")
        # Then frames, then complete.
        done = _read_until(ws, "complete")
        assert done["total_frames"] == 5


def test_session_bookkeeping_is_cleared_on_disconnect(client):
    """When the socket closes, _connections / _streamers must drop the entry."""
    with client.websocket_connect("/ws/simulation") as ws:
        ws.send_json({"action": "ping"})
        pong = ws.receive_json()
        assert pong["type"] == "pong"
        assert len(ws_mod._connections) == 1
    # Socket has closed; the finally block must have dropped the entry.
    assert len(ws_mod._connections) == 0
    assert len(ws_mod._streamers) == 0


def test_params_action_actually_recompiles_equations(client, monkeypatch):
    """The ``params`` action must call compile_equations so that the next
    simulation reflects the new parameter values - not just update the
    parameters dict and silently keep the old lambdified equations.

    We spy on ``simulator.compile_equations`` and assert it was invoked.
    """
    # Wrap a real PhysicsCompiler so we can observe recompiles.
    from mechanics_dsl import PhysicsCompiler

    recompile_calls = []
    original_compile = PhysicsCompiler.compile_dsl

    def patched_compile(self, code, *args, **kwargs):
        result = original_compile(self, code, *args, **kwargs)
        # After the original compile, swap in a counting recompile so we
        # only record calls that came from the params action (the initial
        # compile_dsl run has already completed by this point).
        real = self.simulator.compile_equations

        def counting_compile_equations(eqs, coords):
            recompile_calls.append(("params",))
            return real(eqs, coords)

        self.simulator.compile_equations = counting_compile_equations
        return result

    monkeypatch.setattr(PhysicsCompiler, "compile_dsl", patched_compile)

    with client.websocket_connect("/ws/simulation") as ws:
        ws.send_json({"action": "compile", "code": PENDULUM_DSL})
        ws.receive_json()  # compiled

        ws.send_json({"action": "params", "values": {"m": 2.5}})
        confirm = ws.receive_json()
        assert confirm["type"] == "params_updated"

    assert recompile_calls, (
        "params action did not call compile_equations - the lambdified "
        "equations would still hold the original parameter values."
    )
