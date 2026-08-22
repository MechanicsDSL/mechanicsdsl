# Engine freeze — 22 August 2026

The measuring instrument is pinned here. This file records *what* is frozen and
*why the pin moved*. `SCOPE.md` remains the plan; this is the plan's anchor.

## The pin

| | |
|---|---|
| **Commit** | `a8dc2b27ab1110d1f08dd24d0d6a6bdd994b4599` (`a8dc2b2`) |
| **Tag** | `v2.1.3` (annotated, points at the above) |
| **Authored** | 2026-08-15 11:44:07 -0600 |
| **Subject** | release: v2.1.3 - correctness release for ARM, Hamiltonian, and degenerate solves |
| **Frozen on** | 2026-08-22 |
| **Working tree** | clean except untracked `RELEASE_NOTES_v2.1.3.md`, which is not engine code |

Reproduce the frozen engine with:

    git checkout v2.1.3

## Why the pin moved from `d04207c`

The previous pin was `d04207c` (10 August). Seven commits landed after it. Per
`SCOPE.md`, any engine change voids the baseline — so the pin was moved
deliberately rather than allowed to drift, and the baseline re-run below is the
price of moving it.

What changed in `src/` across `d04207c..a8dc2b2`:

| File | Change | Touched by the suite? |
|---|---|---|
| `mechanics_dsl/__init__.py` | `__version__` 2.1.2 -> 2.1.3 | string only |
| `mechanics_dsl/codegen/arm.py` | ARM embedded-derivative and integer-power fixes | no |
| `mechanics_dsl/cli.py` | target count, export repair | no |
| `mechanics_dsl/server/routes.py` | export repair | no |
| `mechanics_dsl/integrations/modelica.py` | export repair | no |

The rest of the diff is examples, docs, `CHANGELOG`, `CITATION.cff`, and one
ARM codegen test.

No symbolic or dynamics module changed. The suite reaches the engine through
`from mechanics_dsl import PhysicsCompiler` inside `worker.py` and never
invokes the CLI or the REST layer, so no changed file lies on a path the sweep
executes. The v2.1.3 release title mentions Hamiltonian and degenerate solves,
but those fixes predate `d04207c` and are bundled into the release notes, not
into these seven commits.

**This reasoning is not a substitute for the re-run.** It is the reason to
expect the re-run to reproduce, and if it does not, the discrepancy is a finding
about the suite rather than about the seven commits.

## Baseline status

`RESULTS.md` reports zero silent failures in 45 adjudicated cases, measured at
`d04207c`. Those numbers are **pending re-verification at `a8dc2b2`** and must
not be cited as current until the sweep is repeated and this section says so.

Re-run with:

    cd stress_suite && python run.py --timeout 180

The baseline is confirmed only if the four-box tally reproduces — in particular
the silent count and the count of cases where the ground-truth oracle actually
executed. A reproduced tally supersedes nothing; it re-anchors the same numbers
to the new pin. A changed tally voids `RESULTS.md` outright.
