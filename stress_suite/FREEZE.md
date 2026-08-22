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

## Baseline status — CONFIRMED at this pin, 22 August 2026

The sweep was repeated at `a8dc2b2` (180 s per-case timeout, 55 cases). It
reproduced the `d04207c` baseline exactly.

| Quantity | `d04207c` | `a8dc2b2` |
|---|---|---|
| pass / silent / error / timeout | 44 / 0 / 1 / 10 | 44 / 0 / 1 / 10 |
| adjudicated | 45 | 45 |
| oracle applicable / ran | 22 / 22 | 22 / 22 |
| per-case status changes | — | **0 of 55** |
| max change, any recorded metric | — | **0.0 (exact)** |

**The baseline is zero silent failures in 45 adjudicated cases at `a8dc2b2`.**
It may be cited at this pin. Not one case changed status and every recorded
number — energy drift, EOM mismatch, constraint residual, position range — is
bit-identical to the earlier run, so the two sweeps are the same measurement
rather than merely compatible ones.

Re-run with:

    cd stress_suite && python run.py --timeout 180

If the engine moves off `a8dc2b2`, this section is void again and the sweep has
to be repeated before any number is cited.

## Known limit on the number

The independent ground-truth oracle applies to the unconstrained Lagrangian
pathway only. Of the 45 adjudicated cases it covered 22; the other 23 —
every Hamiltonian and every constrained case — rest on energy conservation,
constraint residual, and structural checks alone.

The oracle ran on every case where it was applicable, which is the property
the harness was rebuilt to guarantee. But "zero silent failures in 45 cases"
should be stated pathway-by-pathway rather than pooled. Extending the oracle to
the Hamiltonian pathway is the first thing to do after the study closes — not
during it, because changing the instrument mid-measurement is the same error as
changing the engine.
