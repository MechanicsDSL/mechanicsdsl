"""
Three-way cross-validation of the N-link chain derivation.

WHAT THIS PROVES
----------------
Three derivations of the same system, sharing as little as possible:

  1. reference.py    -- closed form, numpy only, no symbolic algebra
  2. groundtruth.py  -- sympy.physics.mechanics.LagrangesMethod
  3. MechanicsDSL    -- the engine under test, at the frozen pin

If all three agree at random states, each one corroborates the others. That is
worth more than any pairwise check: (1) and (2) share no library, so their
agreement validates the closed-form algebra independently of sympy; and (3)
agreeing with both is the strongest statement the suite can make about the
engine on this family.

It also answers the question that motivated writing (1): can a
library-independent reference adjudicate the HAMILTONIAN pathway? The symbolic
oracle cannot -- it needs symbolic equations to read, and the Hamiltonian route
does not expose them in the required form, which is why 19 Hamiltonian cases
currently carry no independent derivation check. The reference works from the
state alone, so it should not care which pathway produced the accelerations.

Run:
    python validate_reference.py
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)
# Repo source must precede any installed copy. A stale site-packages build
# would otherwise be validated instead of the pinned engine.
sys.path.insert(0, SRC)

import logging
logging.disable(logging.CRITICAL)
import warnings
warnings.filterwarnings("ignore")

import numpy as np

import reference
import systems
import worker


K_PROBE = 24
RTOL = 1e-8
SEED = 20260822


def engine_accel(dsl: str, coords, tool: str):
    """Compile with MechanicsDSL and return (accel_fn, route) or (None, why)."""
    from mechanics_dsl import PhysicsCompiler
    c = PhysicsCompiler()
    res = c.compile_dsl(dsl, use_hamiltonian=(tool == "hamiltonian"),
                        use_constraints=False)
    if not res.get("success"):
        return None, f"compile_failed", None
    fn, route = worker._engine_accel_fn(c, coords)
    return fn, route, c


def sympy_accel(dsl: str, coords, compiler):
    """Build the sympy.physics.mechanics oracle for the same system."""
    import groundtruth as gt
    L = compiler.symbolic.ast_to_sympy(compiler.lagrangian)
    truth = gt.build_truth(L, coords, compiler.simulator.parameters, None)
    return truth.accel


def compare(N: int, tool: str):
    """Return a dict of pairwise worst relative mismatches for one case."""
    dsl = systems.n_pendulum_dsl(N)
    coords = [f"theta{i}" for i in range(N)]

    ref = reference.NLinkChain(N)
    eng_fn, route, compiler = engine_accel(dsl, coords, tool)

    row = {"N": N, "tool": tool, "route": route,
           "ref_vs_engine": None, "ref_vs_sympy": None,
           "sympy_vs_engine": None, "note": ""}

    if eng_fn is None:
        row["note"] = route or "engine_unavailable"
        return row

    # The sympy oracle speaks accelerations only, so it contributes nothing to
    # a Hamiltonian row beyond what the Lagrangian row at the same N already
    # establishes -- and building it via LagrangesMethod is the dominant cost
    # at N>=3. Skip it there rather than spend the wall clock twice.
    if tool == "hamiltonian":
        sym_fn = None
        row["note"] = "engine compared in (q,p); sympy oracle skipped (accel-only)"
    else:
        try:
            sym_fn = sympy_accel(dsl, coords, compiler)
        except Exception as e:
            sym_fn = None
            row["note"] = f"sympy_oracle_unavailable:{type(e).__name__}"

    # ---- STATE CONVENTION -------------------------------------------------
    # The Lagrangian pathway integrates (q, q_dot) and returns (q_dot, q_ddot).
    # The Hamiltonian pathway integrates (q, p) and returns (q_dot, p_dot).
    #
    # Taking dydt[1::2] and calling it "acceleration" is therefore correct on
    # one pathway and a category error on the other. The error is invisible at
    # N=1, where M = m l^2 = 1 makes p numerically equal to q_dot, and shows up
    # as a mismatch of order 1 from N=2 onward. Comparing p_dot against q_ddot
    # on dof_N2 yields ~4.5 relative, which looks exactly like a silent failure
    # and is not one.
    #
    # So each pathway is compared on its own terms: full right-hand side
    # against full right-hand side, in the coordinates that pathway uses.
    ham = (tool == "hamiltonian")
    eom = compiler.simulator.equations_of_motion

    rng = np.random.default_rng(SEED)
    w_re = w_rs = w_se = 0.0
    for _ in range(K_PROBE):
        st = rng.uniform(-0.5, 0.5, size=2 * N)      # always (q, q_dot)

        if ham:
            probe = ref.canonical_state(st)           # -> (q, p)
            dy_ref = ref.canonical_rhs(probe)         # -> (q_dot, p_dot)
        else:
            probe = st
            dy_ref = np.empty(2 * N)
            dy_ref[0::2] = st[1::2]
            dy_ref[1::2] = ref.accel(st)

        dy_eng = np.asarray(eom(0.0, probe), dtype=float)
        denom = np.maximum(np.abs(dy_ref), 1.0)
        w_re = max(w_re, float(np.max(np.abs(dy_eng - dy_ref) / denom)))

        # The sympy oracle only speaks accelerations, so it is compared in the
        # Lagrangian coordinates regardless of which pathway the engine used.
        if sym_fn is not None:
            a_ref = ref.accel(st)
            a_sym = np.asarray(sym_fn(st), dtype=float)
            d2 = np.maximum(np.abs(a_ref), 1.0)
            w_rs = max(w_rs, float(np.max(np.abs(a_sym - a_ref) / d2)))
            if not ham:
                w_se = max(w_se, float(np.max(np.abs(dy_eng[1::2] - a_sym) / d2)))

    row["ref_vs_engine"] = w_re
    if sym_fn is not None:
        row["ref_vs_sympy"] = w_rs
        if not ham:
            row["sympy_vs_engine"] = w_se
        else:
            row["note"] = "engine compared in (q,p); sympy oracle is (q,qdot)-only"
    return row


def fmt(x):
    return "     --    " if x is None else f"{x:11.3e}"


def main() -> int:
    import mechanics_dsl
    print("Three-way cross-validation of the N-link chain\n")
    print(f"  engine     : mechanics_dsl {mechanics_dsl.__version__}")
    print(f"  loaded from: {mechanics_dsl.__file__}")
    print(f"  probes     : {K_PROBE} random states per case, seed {SEED}")
    print(f"  tolerance  : {RTOL:g} relative\n")

    if not mechanics_dsl.__file__.startswith(SRC):
        print("  !! engine did NOT load from the repo src/ -- aborting")
        return 2

    hdr = (f"{'N':>2} {'pathway':<12} {'ref~engine':>12} {'ref~sympy':>12} "
           f"{'sympy~engine':>12}  note")
    print(hdr)
    print("-" * len(hdr))

    # N=4,5 on the Hamiltonian pathway exceed the suite's 180 s wall and are
    # recorded as timeouts in the baseline. Attempting them here would hang
    # without a wall clock, so the ladder stops where the engine does.
    LADDER = {"lagrangian": (1, 2, 3, 4, 5), "hamiltonian": (1, 2, 3)}

    rows = []
    for tool in ("lagrangian", "hamiltonian"):
        for N in LADDER[tool]:
            try:
                r = compare(N, tool)
            except Exception as e:
                r = {"N": N, "tool": tool, "ref_vs_engine": None,
                     "ref_vs_sympy": None, "sympy_vs_engine": None,
                     "note": f"EXC:{type(e).__name__}: {str(e)[:40]}"}
            rows.append(r)
            print(f"{r['N']:>2} {r['tool']:<12} {fmt(r['ref_vs_engine'])} "
                  f"{fmt(r['ref_vs_sympy'])} {fmt(r['sympy_vs_engine'])}  "
                  f"{r.get('note','')}")

    print()
    # -- headline conclusions ------------------------------------------------
    lag = [r for r in rows if r["tool"] == "lagrangian"]
    ham = [r for r in rows if r["tool"] == "hamiltonian"]

    def summarise(label, group):
        ok = [r for r in group if r["ref_vs_engine"] is not None]
        if not ok:
            print(f"  {label}: reference adjudicated 0 of {len(group)} cases")
            return
        worst = max(r["ref_vs_engine"] for r in ok)
        verdict = "AGREE" if worst < RTOL else "DISAGREE"
        print(f"  {label}: reference adjudicated {len(ok)} of {len(group)} "
              f"cases, worst mismatch {worst:.3e} -> {verdict}")

    summarise("Lagrangian ", lag)
    summarise("Hamiltonian", ham)

    ham_ok = sum(1 for r in ham if r["ref_vs_engine"] is not None)
    sym_ham = sum(1 for r in ham if r["sympy_vs_engine"] is not None)
    print()
    print(f"  Coverage gain on the Hamiltonian pathway: "
          f"sympy oracle adjudicates {sym_ham}/{len(ham)}, "
          f"reference adjudicates {ham_ok}/{len(ham)}")

    worst_rs = [r["ref_vs_sympy"] for r in rows if r["ref_vs_sympy"] is not None]
    if worst_rs:
        print(f"  Closed form vs sympy derivation: worst {max(worst_rs):.3e} "
              f"across {len(worst_rs)} cases (shares no library)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
