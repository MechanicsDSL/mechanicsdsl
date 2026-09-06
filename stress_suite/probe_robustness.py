"""
Is the agreement an artefact of how probe states were sampled?

WHY THIS EXISTS
---------------
Adjudication throughout this study compares each engine's accelerations with the
reference's at 16 probe states per case, drawn uniformly from a fixed box under
one seed. That is a defensible sample but a thin one, and it invites a fair
question: would adversarially chosen states have found a disagreement that
uniform sampling missed?

This answers it directly. The same comparison is repeated with five seeds and
four sampling regimes chosen to stress different failure mechanisms:

  uniform   the study's own regime, moderate angles and velocities
  wide      velocities an order of magnitude larger, where Coriolis terms
            dominate and any error in the C(q, qdot) qdot bracket is amplified
  near_pi   angles clustered near pi, the region where the study's one shared
            failure lives and where gravity terms nearly cancel
  aligned   angles nearly equal to one another, where the mass matrix of a
            serial chain is worst-conditioned and a linear solve is most
            fragile

Every engine is driven along the best path found for it in TR-2026-06, so a
disagreement could not be dismissed as an artefact of poor driving.

Run in WSL, where all three engines import in one process:

    PYTHONPATH=../src ~/drake-venv/bin/python probe_robustness.py
"""

from __future__ import annotations

import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "src"))
for p in (HERE, SRC):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

SEEDS = (20260905, 1, 7919, 20260101, 999983)
PROBES = 64
LADDER = (2, 3, 5, 8, 12)
REGIMES = ("uniform", "wide", "near_pi", "aligned")
TOL = 1e-8


def draw(regime: str, N: int, rng) -> np.ndarray:
    st = np.zeros(2 * N)
    if regime == "uniform":
        st[0::2] = rng.uniform(-1.0, 1.0, N)
        st[1::2] = rng.uniform(-0.8, 0.8, N)
    elif regime == "wide":
        st[0::2] = rng.uniform(-math.pi, math.pi, N)
        st[1::2] = rng.uniform(-8.0, 8.0, N)
    elif regime == "near_pi":
        st[0::2] = math.pi + rng.normal(0.0, 1e-3, N)
        st[1::2] = rng.uniform(-0.5, 0.5, N)
    elif regime == "aligned":
        base = rng.uniform(-math.pi, math.pi)
        st[0::2] = base + rng.normal(0.0, 1e-4, N)
        st[1::2] = rng.uniform(-2.0, 2.0, N)
    else:
        raise ValueError(regime)
    return st


def builders():
    import systems
    import worker
    from mechanics_dsl import PhysicsCompiler
    from sweep_sympy_variants import build_kane_cse

    def mdsl(N):
        try:                                    # the fix, when present
            from mechanics_dsl.utils import config
            config.enable_simplification = False
        except (ImportError, AttributeError):
            pass                                # frozen engine: no such switch
        c = PhysicsCompiler()
        r = c.compile_dsl(systems.n_pendulum_dsl(N), use_hamiltonian=False,
                          use_constraints=False)
        if not r.get("success"):
            raise RuntimeError("compile failed")
        fn, _ = worker._engine_accel_fn(c, [f"theta{i}" for i in range(N)])
        return fn

    def drake(N):
        from adapter_drake import DrakeChain
        ch = DrakeChain(N)
        return lambda s: np.asarray(ch.accel(np.asarray(s, float)),
                                    float).reshape(N)

    return {"MechanicsDSL": mdsl, "SymPy": build_kane_cse, "Drake": drake}


def main() -> int:
    import reference

    B = builders()
    print("Probe-state robustness: 5 seeds x 64 probes x 4 regimes\n")
    print(f"{'N':>3}  {'engine':<14}" + "".join(f"{r:>12}" for r in REGIMES))
    print("-" * (19 + 12 * len(REGIMES)))

    rows, worst_overall, disagreements = [], 0.0, 0
    for N in LADDER:
        ref = reference.NLinkChain(N)
        for name, build in B.items():
            try:
                fn = build(N)
            except Exception as e:
                print(f"{N:>3}  {name:<14}  build failed: "
                      f"{type(e).__name__}: {str(e)[:40]}")
                continue
            cells = []
            for regime in REGIMES:
                worst = 0.0
                for seed in SEEDS:
                    rng = np.random.default_rng(seed)
                    for _ in range(PROBES):
                        st = draw(regime, N, rng)
                        a = np.asarray(fn(st), float).reshape(N)
                        r = np.asarray(ref.accel(st), float).reshape(N)
                        worst = max(worst, float(np.max(
                            np.abs(a - r) / np.maximum(np.abs(r), 1.0))))
                rows.append({"N": N, "engine": name, "regime": regime,
                             "worst": worst, "agree": worst <= TOL})
                if worst > TOL:
                    disagreements += 1
                worst_overall = max(worst_overall, worst)
                cells.append(f"{worst:>12.2e}")
            print(f"{N:>3}  {name:<14}" + "".join(cells), flush=True)

    total = len(SEEDS) * PROBES * len(REGIMES) * len(LADDER) * len(B)
    print(f"\n  comparisons          : {total}")
    print(f"  worst disagreement   : {worst_overall:.3e}")
    print(f"  cells above {TOL:g} : {disagreements}")
    print("  VERDICT: " + ("no disagreement under any regime or seed"
                           if disagreements == 0 else "DISAGREEMENT FOUND"))

    out = os.path.join(HERE, "out", "probe_robustness.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"seeds": list(SEEDS), "probes_per_seed": PROBES,
                   "regimes": list(REGIMES), "comparisons": total,
                   "worst": worst_overall, "disagreements": disagreements,
                   "rows": rows}, fh, indent=1)
    print(f"  written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
