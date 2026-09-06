"""Where does MechanicsDSL's compile time actually go on the N-link chain?

CSE at lambdify time gave no gain (it cost 24% at N=8), which only makes sense
if lambdify is not where the time is. This locates the cost before proposing
another lever, so the next attempt is aimed rather than guessed.
"""
from __future__ import annotations

import cProfile
import os
import pstats
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "src"))
for p in (HERE, SRC):
    if p not in sys.path:
        sys.path.insert(0, p)

import systems  # noqa: E402
from mechanics_dsl import PhysicsCompiler  # noqa: E402


def compile_chain(N: int):
    c = PhysicsCompiler()
    r = c.compile_dsl(systems.n_pendulum_dsl(N), use_hamiltonian=False,
                      use_constraints=False)
    assert r.get("success"), "compile failed"
    return c


def main() -> int:
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    print(f"Profiling MechanicsDSL compile of the {N}-link chain\n")

    t0 = time.time()
    pr = cProfile.Profile()
    pr.enable()
    compile_chain(N)
    pr.disable()
    total = time.time() - t0
    print(f"total compile: {total:.2f}s\n")

    st = pstats.Stats(pr)
    st.sort_stats("cumulative")

    print("Top 22 by cumulative time:")
    print(f"{'cum_s':>9} {'tot_s':>9} {'calls':>9}  function")
    rows = []
    for func, (cc, nc, tt, ct, _callers) in st.stats.items():
        rows.append((ct, tt, nc, func))
    rows.sort(reverse=True)
    for ct, tt, nc, func in rows[:22]:
        fn = f"{os.path.basename(func[0])}:{func[1]}({func[2]})"
        print(f"{ct:9.2f} {tt:9.2f} {nc:9d}  {fn}")

    print("\nShare of total by module:")
    by_mod = {}
    for ct, tt, nc, func in rows:
        mod = os.path.basename(func[0])
        by_mod[mod] = by_mod.get(mod, 0.0) + tt      # tottime: no double count
    for mod, tt in sorted(by_mod.items(), key=lambda kv: -kv[1])[:12]:
        print(f"  {tt:8.2f}s  {100*tt/max(total,1e-9):5.1f}%  {mod}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
