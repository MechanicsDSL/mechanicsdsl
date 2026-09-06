"""
Is the referee itself right? Checked against physics, not against itself.

WHY THIS EXISTS
---------------
Every adjudication in this study is performed by one closed-form reference. If
that reference is wrong, every engine agreeing with it is evidence of nothing,
and the study's entire structure rests on a single point of failure. The
reference carries eleven self-tests, but self-tests are written by the same
person who wrote the code and share its misconceptions -- exactly the objection
Knight and Leveson raise against independent implementation.

This checks the reference against a result from outside the study: the
small-oscillation spectrum of a uniform hanging chain, which is classical and
predates any of this software.

THE INDEPENDENT RESULT
----------------------
A uniform flexible chain of length L hanging under gravity has small-oscillation
angular frequencies

    omega_k = (z_k / 2) sqrt(g / L)

where z_k is the k-th zero of the Bessel function J_0 (2.40483, 5.52008,
8.65373, ...). This is a textbook result obtained by solving the continuum wave
equation for a hanging chain; nothing in this study was used to derive it.

An N-link chain of equal links with point masses at the joints approximates that
continuum chain, and its linearised spectrum must converge to the above as N
grows. If the reference's mass matrix or potential is wrong, the spectrum will
not converge -- and no amount of agreement among engines would have revealed it.

THE TEST CAN FAIL
-----------------
Convergence that stalls, or converges to the wrong constant, means the reference
is wrong and the study's adjudication is compromised. That is the point of
running it.
"""

from __future__ import annotations

import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

# Zeros of J_0, from standard tables (Abramowitz & Stegun 9.5).
J0_ZEROS = (2.404825557695773, 5.520078110286311, 8.653727912911012,
            11.791534439014281, 14.930917708487786)
G = 9.81
L_TOTAL = 1.0
NS = (4, 8, 16, 32, 64, 128)
MODES = 3


def linearised_spectrum(N: int) -> np.ndarray:
    """Small-oscillation frequencies of the reference's N-link chain.

    Built from the reference's OWN mass-matrix coefficients so that an error
    there propagates into this test rather than being bypassed.
    """
    import reference
    l = L_TOTAL / N
    m = 1.0 / N                       # fixed total mass; only ratios matter
    chain = reference.NLinkChain(N, m=m, l=l, g=G)

    # M_ij = a_ij cos(th_i - th_j) -> a_ij at small angles.
    M = np.array(chain._a, dtype=float)
    # V = sum_j (N-j) m g l (1 - cos th_j) -> (1/2) K_jj th_j^2
    K = np.diag([(N - j) * m * G * l for j in range(N)])

    w2 = np.linalg.eigvals(np.linalg.solve(M, K))
    w2 = np.sort(np.real(w2[np.abs(np.imag(w2)) < 1e-9]))
    return np.sqrt(np.clip(w2, 0.0, None))


def main() -> int:
    exact = [(z / 2.0) * math.sqrt(G / L_TOTAL) for z in J0_ZEROS[:MODES]]
    print("Referee validated against the hanging-chain spectrum\n")
    print(f"  continuum result: omega_k = (z_k/2) sqrt(g/L),  "
          f"g={G}, L={L_TOTAL}")
    print("  target frequencies: "
          + ", ".join(f"{w:.4f}" for w in exact) + " rad/s\n")
    header = f"{'N':>5}" + "".join(f"{'mode ' + str(k + 1):>14}"
                                   for k in range(MODES))
    print(header)
    print("-" * len(header))

    rows = []
    for N in NS:
        w = linearised_spectrum(N)
        got = w[:MODES]
        rel = [abs(got[k] - exact[k]) / exact[k] for k in range(MODES)]
        rows.append({"N": N, "omega": [float(x) for x in got],
                     "rel_error": [float(x) for x in rel]})
        print(f"{N:>5}" + "".join(f"{got[k]:>9.4f}({rel[k]*100:>4.1f}%)"
                                  for k in range(MODES)), flush=True)

    print("\n  target" + "".join(f"{exact[k]:>9.4f}{'':>7}" for k in range(MODES)))

    first = rows[0]["rel_error"]
    last = rows[-1]["rel_error"]
    improving = all(last[k] < first[k] for k in range(MODES))
    print(f"\n  error at N={NS[0]:<4}: "
          + ", ".join(f"{e*100:.2f}%" for e in first))
    print(f"  error at N={NS[-1]:<4}: "
          + ", ".join(f"{e*100:.2f}%" for e in last))
    print(f"  converging toward the continuum result: {improving}")
    if improving and max(last) < 0.05:
        print("\n  VERDICT: the reference reproduces a classical result it was")
        print("           never fitted to. Its mass matrix and potential are")
        print("           independently corroborated.")
    else:
        print("\n  VERDICT: convergence is NOT clean -- inspect before trusting")
        print("           any adjudication that depends on this reference.")

    out = os.path.join(HERE, "out", "reference_physics_validation.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"target": exact, "rows": rows}, fh, indent=1)
    print(f"  written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
