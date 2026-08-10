# Baseline results — 10 August 2026

Measured at commit `e43def8`. Engine changes in `b582e4f`, suite fix in `e43def8`.
This is the frozen baseline. Per `SCOPE.md`, MechanicsDSL does not change again
until the study is finished.

## The number

**One silent failure in 45 adjudicated cases. About 2%.**

A silent failure is the engine handing you a wrong answer while reporting
success. It's the only failure mode you can't defend against as a user, so it's
the number the whole study is built around.

## The tally

55 cases run. 43 passed. 1 silent. 1 loud error. 10 never finished.

The 10 that never finished are not counted as failures. They're missing answers,
not wrong ones, and how many there are depends entirely on the time limit you
picked — which is a fact about the test harness, not about the engine.

## The one silent failure

A two-coordinate system whose mass matrix is almost singular (ε = 1e-8).

The equations MechanicsDSL derived are **correct** — they match an independently
written derivation to 7 parts in 10¹⁵. What goes wrong is the integration. The
mass matrix has a condition number around 2×10⁸, which costs about eight of the
sixteen decimal digits double precision gives you. Energy drifts 3.7% over the
run, and the engine still reports success.

It does now warn that the matrix is ill-conditioned. The warning is advisory and
doesn't block the result, so it still counts as silent. That's the honest call:
the user gets a wrong trajectory and a green light.

## What got fixed since July

All four engine fixes from the old master report are in.

| Fix | Effect |
|---|---|
| Degenerate solves now fail | Exact singularity went from a silent wrong answer to a clean compile failure |
| Momentum relations inverted together | The two-link Hamiltonian went from frozen-and-silent to correct |
| Ill-conditioning warning | The near-singular case now says something instead of nothing |
| Numeric mass-matrix path | **The Lagrangian scaling wall is gone** |

That last one is the biggest change in the run. Four and five coupled pendulum
links used to hit the 180-second limit and die. They now finish in 6.8 and 11.2
seconds, correctly. Zero timeouts on that axis, down from 40%.

## Where it still stops

- Hamiltonian path: walls at 3 linked pendulums.
- Closed loops: wall at 4 nodes.
- Nested-function potentials: wall at depth 16.

Not failures. It just stops answering, and says so by not returning.

## What we still don't know

- **Whether anything hides past the time limit.** Every timeout was at 180
  seconds. If a pathway would return a *wrong* answer at 400 seconds, this run
  can't see it. Three cases are worth probing individually: nested depth 16,
  ε = 1e-11, and the 3-link Hamiltonian.
- **Whether it stays correct.** Nothing here runs automatically. Correctness is
  a thing that was true on 10 August, not a thing that's enforced. The Legendre
  fix shipped with no tests.
- **A speedup that's sitting in the timings.** N=3 takes 53s while N=4 takes 6.8s
  and N=5 takes 11.2s. The bigger systems are faster because they cross the
  threshold onto the numeric path and N=3 doesn't. Lowering that threshold is a
  one-constant change. Do it *after* the study — the engine is frozen.

## Provenance

The raw run reported 39 pass and 5 silent. Four of those five were a bug in the
suite, not the engine: the classifier treated *any* warning on a successful
result as proof of wrong physics, which was right when the only warnings meant
broken math, but wrong once the engine started emitting advisory
ill-conditioning notices. Those four match the independent derivation to between
1e-25 and 1e-33 and conserve energy to 5e-8. They are correct results that got
flagged for being warned about.

Fixed in `e43def8` and each of the five re-verified individually. The figures
above are the corrected ones.
