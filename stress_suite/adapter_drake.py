"""
Drake engine adapter -- the third column of the cross-engine comparison.

EXECUTION BOUNDARY
------------------
Drake installs on Ubuntu, not native Windows, so this module runs inside WSL
while the rest of the harness runs on the Windows side. It is therefore written
to stand alone: it imports only `numpy`, `pydrake`, and `reference`, takes no
argument from the Windows harness, and writes its results to JSON for the
comparison step to merge.

    wsl -d Ubuntu -e bash -lc "~/drake-venv/bin/python \\
        /mnt/c/.../stress_suite/adapter_drake.py --json out/drake.json"

`reference.py` is numpy-only in its core path, so it imports cleanly on both
sides of the boundary and adjudicates Drake exactly as it adjudicates the other
two engines.

MODELLING
---------
The suite's chain is N point masses on massless rods. In Drake this is built as
N rigid bodies joined by revolute joints:

  * Body i's frame origin sits AT joint i.
  * Its point mass sits at (0, 0, -l) in that frame, i.e. at the far end of
    link i, expressed via SpatialInertia.MakeFromCentralInertia with zero
    rotational inertia about the mass.
  * Joint i+1 is located at (0, 0, -l) in body i's frame -- the position of
    mass i -- so the next link pivots about the previous mass.
  * Revolute axis is +y; gravity is -z. At theta = 0 a link hangs along -z, so
    angles are measured from the downward vertical, matching `systems.py`.

Drake 1.56 removed `SpatialInertia.PointMass`; `MakeFromCentralInertia` with a
zero RotationalInertia is the current equivalent.

WHAT IS AND IS NOT PORTABLE
---------------------------
Only the `dof` axis is modelled here. See `portability_note()` for why the
other two portable-in-principle axes are not straightforward in a rigid-body
engine, which is a scope question for the study rather than a limitation of
this code.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import numpy as np

try:
    from pydrake.all import (MultibodyPlant, SpatialInertia, RotationalInertia,
                             RevoluteJoint, RigidTransform,
                             FixedOffsetFrame)
    HAVE_DRAKE = True
except Exception as _e:                                    # pragma: no cover
    HAVE_DRAKE = False
    _DRAKE_ERR = _e

import reference

K_PROBE = 16
TOL = 1e-8
SEED = 20260822


class DrakeChain:
    """Planar N-link pendulum as a Drake MultibodyPlant."""

    def __init__(self, N: int, m: float = 1.0, l: float = 1.0,
                 g: float = 9.81) -> None:
        if not HAVE_DRAKE:
            raise RuntimeError(f"pydrake unavailable: {_DRAKE_ERR}")
        self.N, self.m, self.l, self.g = int(N), float(m), float(l), float(g)

        plant = MultibodyPlant(0.0)
        zero_I = RotationalInertia(0.0, 0.0, 0.0)
        com = np.array([0.0, 0.0, -self.l])          # mass at the far end
        axis = np.array([0.0, 1.0, 0.0])

        bodies = []
        for i in range(self.N):
            si = SpatialInertia.MakeFromCentralInertia(self.m, com, zero_I)
            bodies.append(plant.AddRigidBody(f"m{i}", si))

        # RevoluteJoint requires its two frames to be coincident, so the pivot
        # offset cannot be passed to the constructor. Each joint past the first
        # gets an explicit FixedOffsetFrame on the parent, positioned at the
        # parent's point mass -- that is where the next link pivots.
        for i, b in enumerate(bodies):
            if i == 0:
                parent_frame = plant.world_frame()
            else:
                parent_frame = plant.AddFrame(FixedOffsetFrame(
                    f"pivot{i}", bodies[i - 1].body_frame(),
                    RigidTransform(com)))
            plant.AddJoint(RevoluteJoint(
                f"j{i}", parent_frame, b.body_frame(), axis))

        plant.mutable_gravity_field().set_gravity_vector([0.0, 0.0, -self.g])
        plant.Finalize()
        self.plant = plant
        self.ctx = plant.CreateDefaultContext()
        if plant.num_positions() != self.N:
            raise RuntimeError(f"expected {self.N} positions, "
                               f"got {plant.num_positions()}")

    # -- ANGLE CONVENTION ---------------------------------------------------
    #
    # The suite's Lagrangian uses ABSOLUTE angles measured from the downward
    # vertical: its kinetic coupling is cos(theta_j - theta_k), which is only
    # meaningful for absolute angles. A chain of Drake revolute joints reports
    # RELATIVE angles -- joint i rotates its body with respect to its PARENT,
    # not with respect to the world.
    #
    # The two coincide only for the first link, so a naive comparison passes at
    # N=1 and fails from N=2 onward with an error that grows with N. This is
    # the same class of mistake as the (q,p) mismatch in TR-2026-01 section 5:
    # a convention difference masked by a degenerate case.
    #
    # The map is linear and lower-triangular:
    #     q_0 = theta_0,   q_i = theta_i - theta_{i-1}
    # so theta = cumsum(q), and the same relation holds for the velocities and
    # accelerations because the map is constant.

    @staticmethod
    def _abs_to_rel(x: np.ndarray) -> np.ndarray:
        """Absolute angles (or their derivatives) -> Drake joint coordinates."""
        out = np.asarray(x, dtype=float).copy()
        out[1:] -= np.asarray(x, dtype=float)[:-1]
        return out

    @staticmethod
    def _rel_to_abs(x: np.ndarray) -> np.ndarray:
        """Drake joint coordinates (or their derivatives) -> absolute angles."""
        return np.cumsum(np.asarray(x, dtype=float))

    def accel(self, state: np.ndarray) -> np.ndarray:
        """Absolute angular accelerations at interleaved [th0, w0, th1, w1, ...].

        Drake exposes M(q), the bias term C(q,v)v, and generalised gravity
        separately, so the equation solved here is
            M(q) a = tau_g(q) - C(q,v) v
        which is the same content as the reference's M a = -(C + G). The state
        is converted into Drake's relative coordinates on the way in and the
        accelerations converted back on the way out.
        """
        s = np.asarray(state, dtype=float)
        self.plant.SetPositions(self.ctx, self._abs_to_rel(s[0::2]))
        self.plant.SetVelocities(self.ctx, self._abs_to_rel(s[1::2]))
        M = self.plant.CalcMassMatrix(self.ctx)
        Cv = self.plant.CalcBiasTerm(self.ctx)
        tau_g = self.plant.CalcGravityGeneralizedForces(self.ctx)
        a_rel = np.linalg.solve(M, tau_g - Cv)
        return self._rel_to_abs(a_rel)

    def rhs(self, _t: float, state: np.ndarray) -> np.ndarray:
        s = np.asarray(state, dtype=float)
        out = np.empty_like(s)
        out[0::2] = s[1::2]
        out[1::2] = self.accel(s)
        return out


def portability_note() -> str:
    return (
        "Only the `dof` axis is modelled in Drake.\n"
        "\n"
        "  mass_ratio    Two masses joined by springs. Expressible as prismatic\n"
        "                joints plus force elements, but it is a spring network\n"
        "                rather than a mechanism, so the translation is a\n"
        "                modelling choice rather than a transcription.\n"
        "\n"
        "  near_singular Requires the mass matrix [[m, cm], [cm, m]], i.e. a\n"
        "                kinetic cross-term c*m*xdot*ydot. A rigid-body engine\n"
        "                produces a diagonal mass matrix for two independent\n"
        "                prismatic degrees of freedom; the cross-term can only\n"
        "                be induced geometrically, by placing two prismatic\n"
        "                axes at angle arccos(c) with a MASSLESS intermediate\n"
        "                body. That is expressible but contrived, and a\n"
        "                massless body is itself the kind of degeneracy the\n"
        "                study is measuring -- so the construction would change\n"
        "                what the axis tests.\n"
        "\n"
        "This is a scope question, not a defect: the `dof` axis is portable to\n"
        "all three engines by direct transcription, and it is the only one of\n"
        "the three that is."
    )


def validate(max_n: int = 5) -> dict:
    """Compare Drake against the library-independent reference."""
    results = []
    for N in range(1, max_n + 1):
        row = {"N": N}
        t0 = time.time()
        try:
            d = DrakeChain(N)
            ref = reference.NLinkChain(N)
            rng = np.random.default_rng(SEED)
            worst = 0.0
            for _ in range(K_PROBE):
                st = rng.uniform(-0.5, 0.5, size=2 * N)
                a_d = d.accel(st)
                a_r = ref.accel(st)
                worst = max(worst, float(np.max(np.abs(a_d - a_r)
                                                / np.maximum(np.abs(a_r), 1.0))))
            row.update(status="pass" if worst <= TOL else "WRONG",
                       worst=worst, seconds=time.time() - t0)
        except Exception as e:
            row.update(status="refused", worst=None,
                       seconds=time.time() - t0,
                       error=f"{type(e).__name__}: {e}"[:120])
        results.append(row)
    return {"engine": "drake", "cases": results}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None, help="write results here")
    ap.add_argument("--max-n", type=int, default=5)
    args = ap.parse_args()

    if not HAVE_DRAKE:
        print(f"pydrake unavailable: {_DRAKE_ERR}")
        return 2

    import importlib.metadata as md
    print("Drake adapter -- N-link chain vs the independent reference")
    print(f"  drake   : {md.version('drake')}")
    print(f"  probes  : {K_PROBE} states/case, tolerance {TOL:g} relative\n")

    out = validate(args.max_n)
    print(f"{'N':>2}  {'status':<9}{'vs reference':>14}{'seconds':>10}")
    print("-" * 38)
    bad = 0
    for r in out["cases"]:
        w = "     --      " if r["worst"] is None else f"{r['worst']:14.3e}"
        print(f"{r['N']:>2}  {r['status']:<9}{w}{r['seconds']:10.2f}")
        if r["status"] != "pass":
            bad += 1
            if r.get("error"):
                print(f"      {r['error']}")

    print()
    print(portability_note())

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"\nwrote {args.json}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
