"""
The same question, the same budget, to all three engines.

WHY THIS EXISTS
---------------
TR-2026-06 established that the frozen ladder's tiers (5 / 12 / 30) measured
driving style rather than capability, and then pushed MechanicsDSL to 100
coordinates. That left a new asymmetry, and it favours the author's engine:
only MechanicsDSL was ever asked for more than 30. SymPy and Drake stopped at
30 because 30 was the top of the frozen ladder, not because either failed
there. A table reporting "MechanicsDSL 100, SymPy 30, Drake 30" would repeat
exactly the error TR-2026-06 exists to correct.

This sweep removes the asymmetry. Every engine is asked for the same systems,
each driven along the best path found for it, under the same wall clock, in one
process on one machine, adjudicated against the same closed-form reference.

  MechanicsDSL  simplification disabled (config.enable_simplification = False),
                deep-recursion retry enabled -- the best supported path
  SymPy         KanesMethod + numeric solve + lambdify(cse=True), the fastest
                of the four documented paths measured in TR-2026-06
  Drake         MultibodyPlant, as in adapter_drake.py; there is only one path

WHY WSL
-------
Drake has no native Windows build, so a cross-platform comparison would confound
engine with operating system. All three import in one WSL process, so the
comparison is made there:

    wsl bash -lc 'cd /mnt/c/.../mechanicsdsl-main && PYTHONPATH=./src \\
        ~/drake-venv/bin/python stress_suite/sweep_equal_budget.py'

PROVENANCE
----------
Every row here comes from ONE engine state in ONE session. The ladder reported
in TR-2026-06 for MechanicsDSL did not: its rows were taken across three code
states as the fixes were developed. Those numbers are consistent and each was
validated, but they are not a clean basis for a published table, and this sweep
supersedes them for that purpose.

The study's frozen matrix is untouched. This measures reach, not correctness,
and no frozen figure is restated.
"""

from __future__ import annotations

import argparse
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

ENGINES = ("Drake", "SymPy", "MechanicsDSL")
LADDER = (40, 60, 80, 100)
PROBES = 6
SEED = 20260905
TOL = 1e-8


def build_mechanicsdsl(N):
    import systems
    import worker
    from mechanics_dsl import PhysicsCompiler
    from mechanics_dsl.utils import config
    config.enable_simplification = False      # the supported switch
    config.deep_recursion = True              # the supported retry
    c = PhysicsCompiler()
    r = c.compile_dsl(systems.n_pendulum_dsl(N), use_hamiltonian=False,
                      use_constraints=False)
    if not r.get("success"):
        raise RuntimeError(f"refused: {str(r.get('error'))[:120]}")
    fn, route = worker._engine_accel_fn(c, [f"theta{i}" for i in range(N)])
    if fn is None:
        raise RuntimeError(f"no accel route: {route}")
    return fn


def build_sympy(N):
    from sweep_sympy_variants import build_kane_cse
    return build_kane_cse(N)


def build_drake(N):
    import numpy as np
    from adapter_drake import DrakeChain
    chain = DrakeChain(N)
    return lambda state: np.asarray(chain.accel(np.asarray(state, float)),
                                    float).reshape(N)


BUILDERS = {"MechanicsDSL": build_mechanicsdsl, "SymPy": build_sympy,
            "Drake": build_drake}


def measure(engine, N):
    import numpy as np
    import reference

    t0 = time.time()
    fn = BUILDERS[engine](N)
    build_s = time.time() - t0

    ref = reference.NLinkChain(N)
    rng = np.random.default_rng(SEED)
    worst, t1 = 0.0, time.time()
    for _ in range(PROBES):
        st = np.zeros(2 * N)
        st[0::2] = rng.uniform(-1.0, 1.0, N)
        st[1::2] = rng.uniform(-0.8, 0.8, N)
        a = np.asarray(fn(st), float).reshape(N)
        r = np.asarray(ref.accel(st), float).reshape(N)
        worst = max(worst, float(np.max(np.abs(a - r)
                                        / np.maximum(np.abs(r), 1.0))))
    return {"engine": engine, "N": N,
            "status": "ok" if worst <= TOL else "DISAGREE",
            "build_s": build_s, "eval_s": (time.time() - t1) / PROBES,
            "err": worst, "total_s": time.time() - t0}


def run_one(engine, N, wall):
    t0 = time.time()
    env = dict(os.environ)
    env["PYTHONPATH"] = SRC + os.pathsep + env.get("PYTHONPATH", "")
    try:
        p = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--child", engine,
             str(N)],
            capture_output=True, text=True, timeout=wall, cwd=HERE, env=env)
    except subprocess.TimeoutExpired:
        return {"engine": engine, "N": N, "status": "TIMEOUT",
                "total_s": time.time() - t0}
    for line in reversed((p.stdout or "").strip().splitlines()):
        try:
            return json.loads(line)
        except Exception:
            continue
    return {"engine": engine, "N": N, "status": "error",
            "total_s": time.time() - t0,
            "error": ((p.stderr or "no output").strip()[-200:])}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", nargs=2, metavar=("ENGINE", "N"))
    ap.add_argument("--wall", type=int, default=7200)
    ap.add_argument("--ladder", default="")
    ap.add_argument("--json", default="out/equal_budget.json")
    args = ap.parse_args()

    if args.child:
        engine, N = args.child[0], int(args.child[1])
        try:
            print(json.dumps(measure(engine, N)))
        except Exception as e:
            print(json.dumps({"engine": engine, "N": N, "status": "error",
                              "error": (type(e).__name__ + ": "
                                        + str(e))[:200]}))
        return 0

    ladder = (tuple(int(x) for x in args.ladder.split(","))
              if args.ladder else LADDER)

    print("Equal-budget reach: the same systems, the same clock, one session")
    print(f"  wall clock : {args.wall}s per (engine, N)")
    print(f"  validation : {PROBES} probes against reference.NLinkChain, "
          f"tolerance {TOL:g}\n")
    print(f"{'N':>4}  " + "".join(f"{e:>18}" for e in ENGINES))
    print("-" * (6 + 18 * len(ENGINES)))

    rows, walled = [], set()
    for N in ladder:
        cells = []
        for e in ENGINES:
            if e in walled:
                cells.append(f"{'(walled)':>18}")
                continue
            r = run_one(e, N, args.wall)
            rows.append(r)
            if r["status"] == "ok":
                cells.append(f"{r['build_s']:>14.1f}s   ")
            else:
                cells.append(f"{r['status']:>18}")
                walled.add(e)
        print(f"{N:>4}  " + "".join(cells), flush=True)
        path = os.path.join(HERE, args.json)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=1)
        if len(walled) == len(ENGINES):
            break

    print("\nLargest N answered correctly:")
    for e in ENGINES:
        ok = [r["N"] for r in rows if r["engine"] == e and r["status"] == "ok"]
        print(f"  {e:<14} {max(ok) if ok else 0}")
    print(f"\n  disagreements: "
          f"{sum(1 for r in rows if r['status'] == 'DISAGREE')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
