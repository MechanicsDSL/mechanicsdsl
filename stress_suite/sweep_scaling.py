"""
Where the three engines are NOT equivalent: the scaling walls.

Everything measured so far found agreement. This looks at the one place
disagreement is already known to exist -- the point at which each engine stops
returning an answer at all.

A timeout is not a wrong answer, and the study's scoring keeps the two apart on
purpose. But it IS a disagreement: at a given system size one engine hands you
equations and another hands you nothing. For a study about what an engine tells
its user, "it never came back" is a distinct and honest outcome, and it is the
only outcome on which these three engines are known to differ.

METHOD
------
Each (engine, N) pair runs in its own SUBPROCESS under a wall clock, so a hang
is recorded rather than taking down the sweep -- the same isolation the main
harness uses. Measured is the time to build the equations of motion and
evaluate the accelerations once. Correctness is checked against the
library-independent reference wherever an answer comes back.

Run inside WSL:
    PYTHONPATH=<repo>/src ~/drake-venv/bin/python sweep_scaling.py
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

LADDER = [1, 2, 3, 5, 8, 12, 20, 30]     # 30 links is SCOPE.md's stated target
ENGINES = ["MechanicsDSL", "SymPy", "Drake"]
PATHWAYS = ["lagrangian", "hamiltonian"]


# --------------------------------------------------------------------------
# Child process: build one (engine, pathway, N) and report via stdout JSON
# --------------------------------------------------------------------------
CHILD = r'''
import json, os, sys, time, logging, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.disable(logging.CRITICAL); warnings.filterwarnings("ignore")
import numpy as np
import reference

engine, pathway, N = sys.argv[1], sys.argv[2], int(sys.argv[3])
t0 = time.time()
out = {"engine": engine, "pathway": pathway, "N": N}
try:
    if engine == "MechanicsDSL":
        import systems, worker
        from mechanics_dsl import PhysicsCompiler
        c = PhysicsCompiler()
        r = c.compile_dsl(systems.n_pendulum_dsl(N),
                          use_hamiltonian=(pathway == "hamiltonian"),
                          use_constraints=False)
        if not r.get("success"):
            raise RuntimeError("compile_success=False")
        fn, route = worker._engine_accel_fn(c, [f"theta{i}" for i in range(N)])
        if fn is None:
            raise RuntimeError("no_accel_route:" + str(route))
    elif engine == "SymPy":
        if pathway == "hamiltonian":
            print(json.dumps({**out, "status": "n/a"})); raise SystemExit(0)
        import adapter_sympy as A
        e = A.SymPyEngine(A.build_chain(N), mode="idiomatic")
        o = e.compile()
        if not o.compiled:
            raise RuntimeError(o.error or "compile_failed")
        fn = e.accel
    else:
        if pathway == "hamiltonian":
            print(json.dumps({**out, "status": "n/a"})); raise SystemExit(0)
        import adapter_drake as D
        fn = D.DrakeChain(N).accel

    build = time.time() - t0
    ref = reference.NLinkChain(N)
    rng = np.random.default_rng(20260822)
    st = rng.uniform(-0.5, 0.5, size=2 * N)
    if pathway == "hamiltonian":
        # Engine integrates (q,p); compare full RHS on its own terms.
        qp = ref.canonical_state(st)
        a = np.asarray(fn(qp), dtype=float)
        r_ = ref.canonical_rhs(qp)[1::2]
    else:
        a = np.asarray(fn(st), dtype=float)
        r_ = ref.accel(st)
    err = float(np.max(np.abs(a - r_) / np.maximum(np.abs(r_), 1.0)))
    out.update(status="ok", build_s=build, total_s=time.time() - t0, err=err)
except Exception as e:
    out.update(status="error", total_s=time.time() - t0,
               error=(type(e).__name__ + ": " + str(e))[:90])
print(json.dumps(out))
'''


def run_one(engine, pathway, N, wall, env):
    child = os.path.join(HERE, "_scaling_child.py")
    t0 = time.time()
    try:
        p = subprocess.run([sys.executable, child, engine, pathway, str(N)],
                           capture_output=True, text=True, timeout=wall,
                           cwd=HERE, env=env)
    except subprocess.TimeoutExpired:
        return {"engine": engine, "pathway": pathway, "N": N,
                "status": "TIMEOUT", "total_s": time.time() - t0}
    for line in reversed((p.stdout or "").strip().splitlines()):
        try:
            return json.loads(line)
        except Exception:
            continue
    return {"engine": engine, "pathway": pathway, "N": N, "status": "error",
            "total_s": time.time() - t0,
            "error": (p.stderr or "no output")[-90:]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wall", type=int, default=120)
    ap.add_argument("--json", default="out/scaling.json")
    args = ap.parse_args()

    with open(os.path.join(HERE, "_scaling_child.py"), "w",
              encoding="utf-8", newline="\n") as f:
        f.write(CHILD)

    env = dict(os.environ)
    print("Scaling walls -- where the engines stop being equivalent")
    print(f"  wall clock : {args.wall}s per (engine, pathway, N)")
    print(f"  ladder     : N = {LADDER}")
    print("  reference  : closed form, O(N^2), never fails\n")

    rows = []
    # Once an engine has walled out, stop climbing: it will not recover.
    walled = set()
    for pathway in PATHWAYS:
        print(f"--- {pathway} pathway ---")
        print(f"{'N':>3}  " + "".join(f"{e:<24}" for e in ENGINES))
        for N in LADDER:
            cells = []
            for engine in ENGINES:
                key = (engine, pathway)
                if key in walled:
                    cells.append(f"{'(walled)':<24}")
                    continue
                r = run_one(engine, pathway, N, args.wall, env)
                rows.append(r)
                if r["status"] == "ok":
                    cells.append(f"{r['total_s']:7.2f}s err={r['err']:.0e}   ")
                elif r["status"] == "n/a":
                    cells.append(f"{'n/a':<24}")
                else:
                    walled.add(key)
                    tag = "TIMEOUT" if r["status"] == "TIMEOUT" else "ERROR"
                    cells.append(f"{tag:<8}{r.get('total_s', 0):6.1f}s      ")
            print(f"{N:>3}  " + "".join(cells))
        print()

    # -- summary -------------------------------------------------------------
    print("=" * 70)
    print("\n  Largest N returning an answer:\n")
    print(f"  {'engine':<16}{'lagrangian':>12}{'hamiltonian':>14}")
    for engine in ENGINES:
        line = f"  {engine:<16}"
        for pathway in PATHWAYS:
            ok = [r["N"] for r in rows if r["engine"] == engine
                  and r["pathway"] == pathway and r["status"] == "ok"]
            na = any(r["status"] == "n/a" for r in rows
                     if r["engine"] == engine and r["pathway"] == pathway)
            line += f"{('n/a' if na else (str(max(ok)) if ok else 'none')):>12}  "
        print(line)
    print(f"\n  reference       {'>=30':>12}{'>=30':>14}   (closed form)")

    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"\n  wrote {args.json}")
    try:
        os.remove(os.path.join(HERE, "_scaling_child.py"))
    except OSError:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
