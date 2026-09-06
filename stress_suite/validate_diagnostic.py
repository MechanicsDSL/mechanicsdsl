"""
Does the convention-mismatch diagnostic actually discriminate?

WHY THIS EXISTS
---------------
The paper offers a transferable rule: a discrepancy that is constant across
cases with different parameters indicates a convention mismatch in the
comparison rather than a defect in the implementation, and two degenerate cases
produce characteristic values -- a term absent from one side gives relative
error exactly 1, a term with the wrong sign gives exactly 2.

That rule was inferred from three incidents. It has never been tested against
faults of known type, which is the only way to learn whether it discriminates or
merely describes the cases that produced it.

METHOD
------
Known faults are injected into a copy of the reference and scored by the same
relative-error metric used throughout the study, over states AND over systems of
different size and mass. Two statistics matter:

  value  -- how close the mean relative error sits to a clean 1 or 2
  spread -- the coefficient of variation across cases. The diagnostic claims
            convention faults give a constant error; if a convention fault
            varies as much as a numerical one, the rule does not discriminate.

Faults of convention:  sign flip, whole term absent, coordinate permutation,
                       relative-vs-absolute angles
Faults of arithmetic:  mass-matrix entry perturbed, Coriolis term dropped,
                       gravity slightly wrong

THE TEST CAN FAIL
-----------------
If a convention fault shows a large spread, or an arithmetic fault produces a
clean constant, the rule as stated in the paper is wrong and must be narrowed.
"""

from __future__ import annotations

import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

SIZES = (2, 3, 4, 5, 6)
MASSES = (0.5, 1.0, 3.0)
PROBES = 24
SEED = 20260905


def faults():
    """name -> (kind, fn(chain, state, true_accel) -> perturbed accel)."""
    def sign_flip(ch, st, a):
        return -a

    def absent(ch, st, a):
        return np.zeros_like(a)

    def permute(ch, st, a):
        return a[::-1].copy()

    def relative_angles(ch, st, a):
        # The engine reports angles relative to the previous link; the
        # reference uses absolute. Evaluate the reference at the wrong angles.
        th = st[0::2].copy()
        rel = np.diff(np.concatenate(([0.0], th)))
        s2 = st.copy()
        s2[0::2] = rel
        return ch.accel(s2)

    def mass_perturbed(ch, st, a):
        th, w = st[0::2], st[1::2]
        M = ch.mass_matrix(th).copy()
        M[0, 0] *= 1.0 + 1e-6
        return np.linalg.solve(M, -ch.coriolis(th, w) - ch.gravity(th))

    def no_coriolis(ch, st, a):
        th, w = st[0::2], st[1::2]
        return np.linalg.solve(ch.mass_matrix(th), -ch.gravity(th))

    def gravity_off(ch, st, a):
        th, w = st[0::2], st[1::2]
        return np.linalg.solve(ch.mass_matrix(th),
                               -ch.coriolis(th, w) - 1.01 * ch.gravity(th))

    return {
        "sign flip":            ("convention", sign_flip),
        "term absent":          ("convention", absent),
        "coordinates permuted": ("convention", permute),
        "relative vs absolute": ("convention", relative_angles),
        "mass entry +1e-6":     ("arithmetic", mass_perturbed),
        "Coriolis dropped":     ("arithmetic", no_coriolis),
        "gravity 1% wrong":     ("arithmetic", gravity_off),
    }


def main() -> int:
    import reference

    print("Does the convention-mismatch diagnostic discriminate?\n")
    print(f"  cases  : {len(SIZES)} sizes x {len(MASSES)} masses, "
          f"{PROBES} probes each")
    print("  metric : mean relative error per case, then spread across cases\n")
    print(f"{'fault':<22}{'kind':<12}{'mean err':>12}{'spread (CoV)':>15}"
          f"{'clean?':>9}")
    print("-" * 70)

    rows = []
    for name, (kind, fn) in faults().items():
        per_case = []
        for N in SIZES:
            for m in MASSES:
                ch = reference.NLinkChain(N, m=m)
                rng = np.random.default_rng(SEED)
                errs = []
                for _ in range(PROBES):
                    st = np.zeros(2 * N)
                    st[0::2] = rng.uniform(-1.0, 1.0, N)
                    st[1::2] = rng.uniform(-0.8, 0.8, N)
                    a = np.asarray(ch.accel(st), float)
                    b = np.asarray(fn(ch, st, a), float).reshape(N)
                    denom = np.maximum(np.abs(a), 1e-12)
                    errs.append(float(np.max(np.abs(b - a) / denom)))
                per_case.append(statistics.fmean(errs))
        mean = statistics.fmean(per_case)
        cov = (statistics.pstdev(per_case) / mean) if mean else 0.0
        clean = min(abs(mean - 1.0), abs(mean - 2.0)) < 1e-9
        rows.append({"fault": name, "kind": kind, "mean": mean,
                     "cov": cov, "clean": clean, "cases": len(per_case)})
        print(f"{name:<22}{kind:<12}{mean:>12.4f}{cov:>15.2e}"
              f"{('yes' if clean else 'no'):>9}")

    conv = [r for r in rows if r["kind"] == "convention"]
    arith = [r for r in rows if r["kind"] == "arithmetic"]
    print(f"\n  convention faults with a clean 1 or 2 : "
          f"{sum(1 for r in conv if r['clean'])}/{len(conv)}")
    print(f"  convention faults that are constant   : "
          f"{sum(1 for r in conv if r['cov'] < 1e-9)}/{len(conv)}")
    print(f"  arithmetic faults that are constant   : "
          f"{sum(1 for r in arith if r['cov'] < 1e-9)}/{len(arith)}")
    print(f"  arithmetic faults mistaken as clean   : "
          f"{sum(1 for r in arith if r['clean'])}/{len(arith)}")

    out = os.path.join(HERE, "out", "diagnostic_validation.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=1)
    print(f"\n  written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
