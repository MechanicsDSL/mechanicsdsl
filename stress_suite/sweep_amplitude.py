"""
Three-engine sweep of the amplitude axis.

For each amplitude: build the 3-link chain in all three engines, check each
engine's equations of motion against the library-independent reference, then
integrate all of them with one pinned integrator and compare energy drift.

Run inside WSL (all three engines import in one process there):

    PYTHONPATH=<repo>/src ~/drake-venv/bin/python sweep_amplitude.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import logging
logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")

import numpy as np
from scipy.integrate import solve_ivp

import axis_amplitude as AX
import reference

K_PROBE = 16
TOL = 1e-8
SEED = 20260822
RTOL, ATOL = 1e-10, 1e-12
METHOD = "DOP853"
T_END = 10.0
N_POINTS = 1500
N = AX.N_LINKS


def engine_rhs(name, amp):
    """Return (rhs, accel) for one engine at one amplitude, or raise."""
    if name == "MechanicsDSL":
        import worker
        from mechanics_dsl import PhysicsCompiler
        c = PhysicsCompiler()
        res = c.compile_dsl(AX.amplitude_dsl(amp), use_hamiltonian=False,
                            use_constraints=False)
        if not res.get("success"):
            raise RuntimeError("compile_success=False")
        fn, route = worker._engine_accel_fn(c, [f"theta{i}" for i in range(N)])
        if fn is None:
            raise RuntimeError(f"no_accel_route:{route}")
    elif name == "SymPy":
        import adapter_sympy as A
        eng = A.SymPyEngine(A.build_chain(N), mode="idiomatic")
        o = eng.compile()
        if not o.compiled:
            raise RuntimeError(o.error or "compile_failed")
        fn = eng.accel
    elif name == "Drake":
        import adapter_drake as D
        fn = D.DrakeChain(N).accel
    else:
        raise ValueError(name)

    def rhs(_t, y):
        out = np.empty_like(y)
        out[0::2] = y[1::2]
        out[1::2] = fn(y)
        return out
    return rhs, fn


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="out/amplitude.json")
    args = ap.parse_args()

    t_eval = np.linspace(0.0, T_END, N_POINTS)
    ref = reference.NLinkChain(N)
    engines = ["MechanicsDSL", "SymPy", "Drake"]

    print(f"Amplitude sweep -- {N}-link chain, three engines")
    print(f"  integrator : {METHOD} pinned, rtol={RTOL:g} atol={ATOL:g}")
    print(f"  referee    : reference.py (numpy closed form)")
    print(f"  probes     : {K_PROBE} states/case for the EOM check\n")

    hdr = (f"{'amp(rad)':>9} {'growth':>9} {'regime':<9}"
           f"{'engine':<14}{'EOM vs ref':>12}{'energy drift':>14}  status")
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for case in AX.all_cases():
        amp = case["knob"]
        y0 = AX.initial_state(amp)

        # Regime: how much a 1e-10 perturbation grows over the horizon.
        yb = y0.copy(); yb[0] += 1e-10
        sa = solve_ivp(ref.rhs, (0, T_END), y0, t_eval=t_eval,
                       method=METHOD, rtol=1e-12, atol=1e-14)
        sb = solve_ivp(ref.rhs, (0, T_END), yb, t_eval=t_eval,
                       method=METHOD, rtol=1e-12, atol=1e-14)
        growth = float(np.linalg.norm(sa.y - sb.y, axis=0)[-1] / 1e-10)
        reg = "REGULAR" if growth < 1e3 else "CHAOTIC"

        # Reference's own motion, for the equilibrium check.
        ref_range = float(np.max(np.abs(sa.y[0::2] - sa.y[0::2][:, [0]])))

        first = True
        for name in engines:
            row = {"amp": amp, "engine": name, "growth": growth,
                   "regime": reg, "expected_moving": case["expected_moving"]}
            try:
                rhs, accel = engine_rhs(name, amp)

                rng = np.random.default_rng(SEED)
                worst = 0.0
                for _ in range(K_PROBE):
                    st = rng.uniform(-0.5, 0.5, size=2 * N)
                    a, r = np.asarray(accel(st), float), ref.accel(st)
                    worst = max(worst, float(np.max(np.abs(a - r)
                                / np.maximum(np.abs(r), 1.0))))

                sol = solve_ivp(rhs, (0, T_END), y0, t_eval=t_eval,
                                method=METHOD, rtol=RTOL, atol=ATOL)
                if not sol.success:
                    raise RuntimeError(sol.message[:32])
                E = np.array([ref.energy(sol.y[:, i])
                              for i in range(sol.y.shape[1])])
                drift = float(np.max(np.abs(E - E[0])) / max(abs(E[0]), 1e-30))
                moved = float(np.max(np.abs(sol.y[0::2] - sol.y[0::2][:, [0]])))

                status = "ok"
                if worst > TOL:
                    status = "WRONG-EOM"
                elif case["expected_moving"] and moved < 1e-8:
                    status = "FROZEN"
                elif (not case["expected_moving"]) and moved > 1e-6:
                    # NOT an engine defect. The equilibrium at theta = pi is
                    # unstable and sin(pi) = 1.22e-16 in float64, so a residual
                    # acceleration of ~1.2e-15 is amplified into full-scale
                    # motion. The independent reference does the same. What is
                    # notable is that no engine WARNS about it.
                    status = "UNWARNED-ROUNDOFF"

                row.update(ok=True, eom=worst, drift=drift, moved=moved,
                           status=status)
                pre = (f"{amp:9.4f} {growth:9.1e} {reg:<9}" if first
                       else " " * 29)
                print(f"{pre}{name:<14}{worst:12.2e}{drift:14.3e}  {status}")
            except Exception as e:
                row.update(ok=False,
                           status=f"refused:{type(e).__name__}",
                           error=str(e)[:60])
                pre = (f"{amp:9.4f} {growth:9.1e} {reg:<9}" if first
                       else " " * 29)
                print(f"{pre}{name:<14}{'--':>12}{'--':>14}  {row['status']}")
            rows.append(row)
            first = False
        print(f"{'':>29}{'(reference)':<14}{'':>12}"
              f"{'':>14}  motion={ref_range:.3e}")
        print()

    # -- summary -------------------------------------------------------------
    print("=" * len(hdr))
    bad = [r for r in rows if r.get("status") not in ("ok",)
           and r.get("ok") is not False]
    refused = [r for r in rows if r.get("ok") is False]
    print(f"\n  engine-amplitude pairs run : {len(rows)}")
    print(f"  wrong equations or motion  : {len(bad)}")
    print(f"  refused to compile         : {len(refused)}")
    for r in bad + refused:
        print(f"    amp={r['amp']:.3f} {r['engine']}: {r['status']}")

    worst_drift = {}
    for r in rows:
        if r.get("ok"):
            worst_drift.setdefault(r["engine"], []).append(r["drift"])
    print("\n  worst energy drift across the axis:")
    for e, ds in worst_drift.items():
        print(f"    {e:<14}{max(ds):.3e}")

    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"\n  wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
