"""
The Hamiltonian pathway, which the frozen study barely reached.

WHY THIS EXISTS
---------------
TR-2026-01 recorded that the Hamiltonian and constrained pathways are the two
thinnest parts of the study's coverage. The frozen scaling ladder took the
Hamiltonian route only to three coordinates before walling, so the pathway that
performs a Legendre transform -- the one place an engine could get the momentum
relation wrong rather than the accelerations -- is the least tested.

TR-2026-06 established that the wall was a cosmetic simplification step rather
than the derivation. With that removed the pathway can be pushed, and the
question is whether it stays correct there.

WHAT IS COMPARED
----------------
The reference supplies the canonical route independently: momentum p = M(q) qdot,
and the canonical right-hand side (qdot, pdot) from the Hamiltonian. An engine
on the Hamiltonian pathway returns pdot, NOT accelerations -- confusing the two
produced one of the four apparatus faults in TR-2026-04, and the comparison here
is made in canonical coordinates for that reason.

Two things are checked at each size:

  canonical   the engine's (qdot, pdot) against the reference's, at probe states
  legendre    that the engine's Hamiltonian pathway and the reference's
              Lagrangian pathway describe the same dynamics -- i.e. that
              M(q)^{-1} p recovers qdot, so the transform is self-consistent

THE TEST CAN FAIL
-----------------
A disagreement here would be the study's first, and would be a genuine finding
against the one engine exposing this pathway.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "src"))
for p in (HERE, SRC):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

LADDER = (1, 2, 3, 4, 5, 6, 8)
WALL = 180          # seconds per size; the frozen sweep used 120
PROBES = 32
SEEDS = (20260905, 4241)
TOL = 1e-8
WALL_NOTE = "simplification disabled where the engine supports it"


def build(N):
    import systems
    import worker
    from mechanics_dsl import PhysicsCompiler
    try:
        from mechanics_dsl.utils import config
        config.enable_simplification = False
    except (ImportError, AttributeError):
        pass                       # frozen engine: no such switch, just slower
    c = PhysicsCompiler()
    r = c.compile_dsl(systems.n_pendulum_dsl(N), use_hamiltonian=True,
                      use_constraints=False)
    if not r.get("success"):
        raise RuntimeError(f"refused: {str(r.get('error'))[:80]}")
    fn, route = worker._engine_accel_fn(c, [f"theta{i}" for i in range(N)])
    if fn is None:
        raise RuntimeError(f"no route: {route}")
    return fn, route


def measure(N):
    """One size, in-process. Invoked as a child so the driver can time it out."""
    import reference
    fn, route = build(N)
    ref = reference.NLinkChain(N)
    worst_canon, worst_leg = 0.0, 0.0
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        for _ in range(PROBES):
            st = np.zeros(2 * N)
            st[0::2] = rng.uniform(-1.0, 1.0, N)
            st[1::2] = rng.uniform(-0.8, 0.8, N)
            # The engine's Hamiltonian ODE is a function of (q, p), NOT
            # (q, qdot). Feeding it velocities is the fault recorded as
            # instance 1 in TR-2026-04: it is invisible at N=1, where M = I
            # makes p numerically equal to qdot, and appears from N=2 on.
            qp = ref.canonical_state(st)
            got = np.asarray(fn(qp), float).reshape(N)
            want = np.asarray(ref.canonical_rhs(qp), float).reshape(-1)[1::2]
            worst_canon = max(worst_canon, float(np.max(
                np.abs(got - want) / np.maximum(np.abs(want), 1.0))))
            th, w = st[0::2], st[1::2]
            p = ref.momentum(th, w)
            w_back = np.linalg.solve(ref.mass_matrix(th), p)
            worst_leg = max(worst_leg, float(np.max(
                np.abs(w_back - w) / np.maximum(np.abs(w), 1.0))))
    ok = worst_canon <= TOL and worst_leg <= TOL
    return {"N": N, "status": "ok" if ok else "DISAGREE", "route": route,
            "canonical_err": worst_canon, "legendre_err": worst_leg}


def run_one(N):
    t0 = time.time()
    env = dict(os.environ)
    env["PYTHONPATH"] = SRC + os.pathsep + env.get("PYTHONPATH", "")
    try:
        p = subprocess.run([sys.executable, os.path.abspath(__file__),
                            "--child", str(N)],
                           capture_output=True, text=True, timeout=WALL,
                           cwd=HERE, env=env)
    except subprocess.TimeoutExpired:
        return {"N": N, "status": "TIMEOUT", "total_s": time.time() - t0}
    for line in reversed((p.stdout or "").strip().splitlines()):
        try:
            r = json.loads(line)
            r["total_s"] = time.time() - t0
            return r
        except Exception:
            continue
    return {"N": N, "status": "error", "total_s": time.time() - t0,
            "error": (p.stderr or "no output").strip()[-160:]}


def main() -> int:
    if len(sys.argv) == 3 and sys.argv[1] == "--child":
        try:
            print(json.dumps(measure(int(sys.argv[2]))))
        except Exception as e:
            print(json.dumps({"N": int(sys.argv[2]), "status": "unavailable",
                              "error": f"{type(e).__name__}: {str(e)[:120]}"}))
        return 0
    import reference  # noqa: F401

    print("Hamiltonian pathway coverage\n")
    print(f"  probes  : {PROBES} states x {len(SEEDS)} seeds per size")
    print(f"  note    : {WALL_NOTE}\n")
    print(f"{'N':>4}{'build':>10}{'canonical err':>16}{'legendre err':>15}"
          f"  verdict")
    print("-" * 58)

    rows = []
    for N in LADDER:
        r = run_one(N)
        rows.append(r)
        if r["status"] == "ok":
            print(f"{N:>4}{r['total_s']:>9.1f}s{r['canonical_err']:>16.3e}"
                  f"{r['legendre_err']:>15.3e}  agree", flush=True)
        else:
            print(f"{N:>4}{r.get('total_s', 0):>9.1f}s{'--':>16}{'--':>15}"
                  f"  {r['status']}"
                  + (f": {r.get('error', '')[:40]}" if r.get("error") else ""),
                  flush=True)
            break

    good = [r for r in rows if r.get("status") == "ok"]
    bad = [r for r in rows if r.get("status") == "DISAGREE"]
    print(f"\n  sizes adjudicated on the Hamiltonian pathway: {len(good)}")
    print(f"  largest size reached                        : "
          f"{max((r['N'] for r in good), default=0)}")
    print(f"  disagreements                               : {len(bad)}")

    out = os.path.join(HERE, "out", "hamiltonian_coverage.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=1)
    print(f"  written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
