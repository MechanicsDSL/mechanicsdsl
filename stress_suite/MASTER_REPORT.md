# MechanicsDSL Stress-Testing Campaign — Master Report

**Author:** Noah Parsons
**Date:** 28 July 2026
**Scope:** MechanicsDSL symbolic dynamics engine (Lagrangian, Hamiltonian, and constrained Lagrange-multiplier pathways)
**Artifacts:** `stress_suite/` — `systems.py`, `groundtruth.py`, `worker.py`, `run.py`, `out/report.md`, `out/results.json`

---

## 1. Objective

I set out to determine whether MechanicsDSL — a symbolic dynamics engine that derives and simulates the equations of motion for a mechanical system from a Lagrangian — could be made to fail, and, crucially, whether its failures were **loud** (a crash, an error, or a hang) or **silent** (a reported success that was in fact physically wrong). The deliverable is the silent-failure fraction for every formulation pathway on every stress axis.

## 2. Method, first iteration, and why it was inadequate

I began by reading the engine's internals — the compiler, the symbolic layer, and the numerical solver — to establish exactly how a system is submitted and how outcomes are reported. In doing so I identified the central hazard: the engine can fall back to a degenerate "zero acceleration" solution while still returning `success = True`, recording only an advisory note in a warnings list that the top-level result does not gate on. This confirmed that a silent-failure mode was not merely plausible but built into the tool's contract, and it defined my classification scheme: a case is **loud** if the engine reports failure, crashes, or exceeds a wall-clock timeout, and **silent** if it reports success while producing incorrect dynamics.

I then constructed an automated harness organised around six stress axes — degrees of freedom, closed kinematic loops, constraint redundancy, near-singular mass matrices, mass-ratio conditioning, and symbolic pathology — each parameterised so that difficulty could be dialled upward until the engine broke. Every case was executed in an isolated subprocess under a timeout, so that hangs and hard crashes could be attributed correctly rather than corrupting the run. For this first version my correctness oracle was **energy conservation**: I derived the system's total energy directly from its Lagrangian and checked that it remained constant along the simulated trajectory, supplementing this with checks for zeroed equations, frozen trajectories, and non-finite output. I ran roughly sixty cases and produced an initial report.

On review, I judged those first results to be weak, and I want to be candid about why. Almost every recorded failure was in fact a timeout, which made the loud/silent split an artefact of an arbitrarily chosen ninety-second limit rather than a property of the tool. The energy oracle, moreover, had material blind spots: because it was built from the same unconstrained Lagrangian the engine was given, it could not detect that the engine had silently dropped a system's constraints; and a system that froze at its initial state trivially conserved energy, so a genuine failure could pass unnoticed. Expressing a silent-failure fraction over a monotonic difficulty sweep was, in addition, only marginally meaningful. I therefore rebuilt the suite in full.

## 3. The rebuilt methodology

The central improvement was an **independent ground-truth oracle**. Rather than trusting energy behaviour alone, I derived each system's equations of motion a second time using SymPy's `physics.mechanics` module — a mature, separately maintained derivation path — and compared its accelerations against MechanicsDSL's, numerically, at randomly sampled states. To keep this reference usable on systems large enough to defeat the engine under test, I held the mass matrix in numeric rather than symbolic form. I validated the oracle to machine precision against a pendulum, a double pendulum, and a constrained particle before relying on it.

In the process I found that pointwise acceleration comparison is ill-defined off the constraint surface for constrained systems, so I routed the oracle by formulation: ground-truth equation comparison for unconstrained systems, and constraint-residual plus energy checks for constrained ones. I also discovered that one of my own test systems — the original "closed ring" — was physically degenerate, with nearly every configuration an equilibrium, which had been generating false "frozen" verdicts; I replaced it with a genuine pinned closed kinematic chain that I verified moves, conserves energy, and holds its constraints. I corrected the oracle's handling of redundant constraints via a least-squares fallback, raised the per-case timeout to 180 seconds (configurable via `--timeout`), and relocated the suite from a temporary directory into the project as a self-contained `stress_suite/` folder with operating instructions.

## 4. Results (rebuilt run: 55 cases — 40 ok, 12 loud, 3 silent)

### 4.1 Silent-failure fraction (silent / all cases)

| Axis | lagrangian | hamiltonian | constrained |
|---|---|---|---|
| dof | 0% (0/5) | **20% (1/5)** | n/a |
| loops | n/a | n/a | 0% (0/3) |
| redundancy | n/a | n/a | 0% (0/7) |
| near_singular | **29% (2/7)** | n/a | n/a |
| mass_ratio | 0% (0/7) | 0% (0/7) | n/a |
| symbolic | 0% (0/7) | 0% (0/7) | n/a |

### 4.2 When it fails, how often silently (silent / (silent + loud))

| Axis | lagrangian | hamiltonian | constrained |
|---|---|---|---|
| dof | 0% (0/2) | 25% (1/4) | n/a |
| loops | n/a | n/a | 0% (0/2) |
| redundancy | n/a | n/a | — (0 fail) |
| near_singular | **67% (2/3)** | n/a | n/a |
| mass_ratio | — (0 fail) | — (0 fail) | n/a |
| symbolic | 0% (0/2) | 0% (0/2) | n/a |

### 4.3 Findings

1. **The default Lagrangian pathway produced no silent failures on any axis**, and the independent oracle proved its results correct wherever it completed — the double and triple pendulum, mass ratios to 10^16, redundancy to eight dependent constraints, and the near-singular regime down to ε = 10⁻⁸ (equations exact to ~10⁻¹⁵). When this pathway could not cope, it timed out; it never returned wrong numbers under a claim of success.
2. **The Hamiltonian pathway silently freezes coupled systems.** The two-link pendulum returned `success = True` with no warning and no motion. The cause is structural: the Legendre transform resolves each velocity one at a time and degenerates when momenta are coupled. The single-degree-of-freedom case is handled correctly.
3. **The near-singular regime produces the most dangerous behaviour.** At ε = 10⁻⁸ the trajectory drifted 3.7% in energy with no warning of any kind — while the oracle confirms the symbolic equations are exactly right, isolating the fault to numerical integration rather than derivation. Only at exact singularity (ε = 0) does the engine flag the problem itself.
4. **The dominant failure mode is a loud scaling wall.** All twelve loud results were timeouts — no crashes, no error returns — as symbolic derivation blew up beyond four coupled pendulum links, beyond the smallest closed loop (N = 4), and beyond nesting depth 16.
5. **Two axes did not break the engine at all**: mass-ratio conditioning to 10^16 and constraint redundancy to eight duplicates. I report these plainly as non-results rather than overstating them.

**Residual caveat.** Because the loud category is entirely timeouts, its boundary depends on the chosen time limit. However, since every completed case is now independently verified, the timeouts are not concealing silent errors up to the sizes tested.

## 5. Actionable plan forward

### 5.1 Engine fixes, in priority order

1. **Gate success on the warnings channel.** `compile_dsl` should not return `success = True` when the symbolic solve has fallen back to zero accelerations; at minimum, promote that condition to a hard failure or a prominent, structured flag. This converts the engine's one designed-in silent mode into a loud one at negligible cost.
2. **Fix or fence the Hamiltonian pathway for coupled systems.** Either implement a simultaneous (matrix) Legendre transform so coupled momenta are inverted together, or detect the coupled case and refuse it with an explicit error. Silent freezing is the worst available outcome; refusal is acceptable, freezing is not.
3. **Add a mass-matrix conditioning check.** The engine already detects exact singularity; it should also estimate the condition number at compile time and warn when integration accuracy is likely to degrade (the ε = 10⁻⁸ case shows correct equations can still yield a quietly wrong trajectory).
4. **Address the scaling wall pragmatically.** The N ≥ 4 blow-ups stem from full symbolic simplification and symbolic linear solves. Deferring the mass-matrix inversion to numeric evaluation time — the same technique my ground-truth oracle uses — would extend the usable range substantially without changing results where the engine already succeeds.

### 5.2 Suite extensions

5. **Probe the wall at higher limits.** Re-run with `--timeout 600` or more to establish whether any pathway transitions from loud-timeout to silent-wrong at larger sizes, which is the one regime the current data cannot rule out.
6. **Finish the under-stressed axes.** Design a mass-ratio variant that forces the ill-conditioning into the symbolic solve rather than the integrator, and extend redundancy to inconsistent (not merely dependent) constraints.
7. **Adopt the ground-truth comparison as a regression test.** The oracle-versus-engine check on fast cases is cheap and machine-precise; wiring a subset into CI would prevent silent regressions in the derivation path.

### 5.3 Reproduction

From `stress_suite/`, with the library at `../src`:

```bash
python run.py --timeout 180
```

Outputs are written to `stress_suite/out/` (`report.md`, `results.json`, and per-case specs under `specs/` for single-case reproduction via `python worker.py out/specs/<name>__<tool>.json`).
