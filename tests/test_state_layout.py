"""
test_state_layout.py
--------------------
The simulator returns ``solution["y"]`` interleaved by coordinate,
``[q1, q1_dot, q2, q2_dot, ...]``, with ``solution["state_vars"]`` naming the
rows. These tests pin that contract and the exporters that label it, because
reading the rows positionally as ``[q..., q_dot...]`` silently mislabels every
multi-DOF result (it caused wrong energy, plot and dataset conclusions in the
satellite packages before they were fixed).
"""
import csv
import json
from argparse import Namespace

import numpy as np
import pytest

from mechanics_dsl import PhysicsCompiler, get_preset
from mechanics_dsl.cli import cmd_export


@pytest.fixture(scope="module")
def double_pendulum():
    c = PhysicsCompiler()
    assert c.compile_dsl(get_preset("double_pendulum"))["success"]
    return c, c.simulate(t_span=(0.0, 1.0), num_points=50)


def test_state_vars_are_interleaved(double_pendulum):
    _, sol = double_pendulum
    assert list(sol["state_vars"]) == ["theta1", "theta1_dot", "theta2", "theta2_dot"]
    assert sol["y"].shape[0] == len(sol["state_vars"])


def test_rows_match_initial_conditions_by_name(double_pendulum):
    c, sol = double_pendulum
    rows = dict(zip(sol["state_vars"], sol["y"]))
    for name in sol["state_vars"]:
        assert rows[name][0] == pytest.approx(c.initial_conditions.get(name, 0.0))


def test_energy_is_conserved_when_read_by_name(double_pendulum):
    c, _ = double_pendulum
    sol = c.simulate(t_span=(0.0, 5.0), num_points=2001, method="DOP853",
                     rtol=1e-12, atol=1e-14)
    r = dict(zip(sol["state_vars"], sol["y"]))
    p = {k: float(v["value"]) for k, v in c.parameters_def.items()}
    th1, th2, w1, w2 = r["theta1"], r["theta2"], r["theta1_dot"], r["theta2_dot"]
    T = (0.5 * (p["m1"] + p["m2"]) * p["l1"] ** 2 * w1 ** 2
         + 0.5 * p["m2"] * p["l2"] ** 2 * w2 ** 2
         + p["m2"] * p["l1"] * p["l2"] * w1 * w2 * np.cos(th1 - th2))
    V = (-(p["m1"] + p["m2"]) * p["g"] * p["l1"] * np.cos(th1)
         - p["m2"] * p["g"] * p["l2"] * np.cos(th2))
    E = T + V
    assert np.max(np.abs(E - E[0])) / abs(E[0]) < 1e-6


def _export(tmp_path, fmt):
    spec = tmp_path / "dp.mdsl"
    spec.write_text(get_preset("double_pendulum"), encoding="utf-8")
    out = tmp_path / "dp_out"
    rc = cmd_export(Namespace(input=str(spec), output=str(out), format=fmt,
                              t_span="0,1", points=20))
    assert rc in (0, None)
    return out


def test_cli_csv_header_labels_every_row(tmp_path):
    out = _export(tmp_path, "csv").with_suffix(".csv")
    rows = list(csv.reader(out.open()))
    assert rows[0] == ["t", "theta1", "theta1_dot", "theta2", "theta2_dot"]
    assert all(len(r) == len(rows[0]) for r in rows[1:])
    assert float(rows[1][3]) == pytest.approx(2.0)  # theta2(0) from the preset
    assert float(rows[1][2]) == pytest.approx(0.0)  # theta1_dot(0)


def test_cli_json_includes_state_vars(tmp_path):
    out = _export(tmp_path, "json").with_suffix(".json")
    data = json.loads(out.read_text())
    assert data["state_vars"] == ["theta1", "theta1_dot", "theta2", "theta2_dot"]
    assert len(data["y"]) == 4
