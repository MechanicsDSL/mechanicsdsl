# Baseline results — 10 August 2026

> **Pending re-verification.** The engine pin moved to `a8dc2b2` (`v2.1.3`) on
> 22 August — see `FREEZE.md`. Every number below was measured at the previous
> pin and must not be cited until the sweep is repeated at the current one.

Measured at commit `d04207c`.

Supersedes the pre-fix run of 28 July, kept for comparison in
`MASTER_REPORT.md` and `out/report_legacy_3way.md`. What changed in between is
documented case by case in `FIXES.md`.

## The number

**Zero silent failures in 45 adjudicated cases.**

A silent failure is the engine handing you a wrong answer while reporting
success. It's the only failure mode you can't defend against as a user, so it's
the number the whole study is built around.

## The tally

55 cases. 44 passed. 1 refused. 10 never finished.

Of the 45 cases where MechanicsDSL gave an answer, 44 were correct and 1 was an
explicit refusal. It produced no wrong answers presented as right.

The 10 that never finished are not counted as failures — they're missing
answers, and how many there are depends on the time limit chosen, which is a
fact about the harness rather than the engine.

**The check actually ran.** Every case where the independent ground-truth
oracle was applicable was compared against it. This matters more than the zero:
a silent-failure rate is worth exactly as much as the checks behind it, and an
earlier version of this suite reported passes on cases where the strongest check
had quietly failed to execute.

## The one refusal

`nearsing_e0` — a mass matrix that is exactly singular. The system is genuinely
degenerate: the coordinates stop being independent, and there is no motion to
compute. The engine fails at compile time and says so.

That is the correct outcome. Refusing an impossible problem is good behaviour;
the failure would be answering it anyway.

## What changed since July

Three silent failures existed on 28 July. All three are closed, each with an
identified cause:

| Was | Cause | Now |
|---|---|---|
| Two-link Hamiltonian froze, reported success | Velocities solved one at a time; breaks down when momenta couple | pass, 37s |
| Exact singularity returned zero accelerations as success | Success not gated on degenerate solves | error — honest refusal |
| ε=1e-8 drifted 3.7% in energy, no warning | Integration tolerance ignored the mass matrix conditioning | pass, drift 1.7e-6 |

Four more cases were reported silent on the intermediate run and were never real
— the suite was counting advisory warnings as evidence of wrong physics. Their
equations matched the independent derivation to between 1e-25 and 1e-33.

The scaling wall also moved: four and five coupled pendulum links used to exceed
180 seconds and get killed. They now finish in 4.6 and 13.8 seconds, correctly
and verified.

## Accuracy

Two separate things, both measured.

**The equations it derives**, against an independently written SymPy derivation
sampled at random states: agreement at or below 1e-15, with several cases exact
to zero. Indistinguishable from a separate implementation of the same physics.

**The trajectories it simulates**, by energy conservation over the full run:
between roughly 1e-6 and 1e-10. The worst case in the suite loses about two
ten-thousandths of a percent of its energy over ten thousand oscillations.

## Where it still stops

| Pathway | Wall |
|---|---|
| Hamiltonian | 3 linked pendulums |
| Closed loops | 4 nodes |
| Nested-function potentials | depth 16 |
| Near-singular | ε = 1e-11 |

Not failures. It stops answering, and the absence of an answer is itself honest.

## What this does and does not claim

It claims: across 55 cases spanning six stress axes, at commit `d04207c`, there
is **no known case where MechanicsDSL returns a wrong answer while reporting
success.**

It does not claim the engine is correct in general. Fifty-five cases is not a
proof, and correctness here is established against one independent
implementation plus conservation laws — if both implementations shared a
conceptual error, neither would catch it. "No known silent failures" is the
strongest claim this method supports, and it is the honest one.

Two further limits worth stating plainly:

- The stiffness-detection fix (`FIXES.md` #7) is not exercised by any case in
  this suite. It was fixed by inspection and remains untested.
- Nothing here runs automatically. This is a measurement taken on 10 August, not
  a property that is enforced going forward. Wiring a fast subset into CI is what
  would turn it into one.
