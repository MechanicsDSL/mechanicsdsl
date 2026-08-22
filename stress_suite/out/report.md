# MechanicsDSL symbolic-dynamics stress report

Per-case wall-clock timeout: **180s**. Tools are MechanicsDSL formulation pathways (Lagrangian / Hamiltonian / constrained Lagrange-multiplier).

## Status taxonomy

| Status | Meaning |
|---|---|
| **pass** | Ran and produced correct dynamics on every applicable check. |
| **silent** | Returned `success=True` but the physics is wrong. |
| **error** | Reported `success=False`, raised, or crashed — a *loud*, attributable failure. |
| **timeout** | Exceeded the wall clock. Never adjudicated. |
| **skipped** | Known to exceed even a 600s budget; not run. Never adjudicated. |

Correctness fractions below use **adjudicated** cases (pass + silent + error) as the denominator. Timeouts and skips are absence of evidence, not evidence of failure — folding them in would make every rate a function of `--timeout`, which is a property of this harness rather than of the engine. The timeout rate is reported separately, as the scaling wall.

Correctness is judged by an *independent* SymPy-mechanics ground-truth derivation of the equations of motion (compared numerically at random states, unconstrained pathway), plus energy-conservation, constraint-residual, frozen-trajectory, NaN/Inf, all-zero-EOM, and solve-fallback-warning checks.

Totals across 55 cases: **44 pass**, **1 error**, **10 timeout**. Adjudicated: **45**.

> Every case where the independent ground-truth oracle was applicable was checked against it.

## 1. Silent-failure rate (silent / adjudicated)

The headline number: of the cases the engine actually returned a verdict on, how often was that verdict wrong *and* claimed to be successful.

| Axis | lagrangian | hamiltonian | constrained |
|---|---|---|---|
| dof | 0% (0/5) | 0% (0/2) | n/a |
| loops | n/a | n/a | 0% (0/1) |
| redundancy | n/a | n/a | 0% (0/7) |
| near_singular | 0% (0/6) | n/a | n/a |
| mass_ratio | 0% (0/7) | 0% (0/7) | n/a |
| symbolic | 0% (0/5) | 0% (0/5) | n/a |

## 2. Of adjudicated failures, how many were silent (silent / (silent + error))

| Axis | lagrangian | hamiltonian | constrained |
|---|---|---|---|
| dof | — (0 fail) | — (0 fail) | n/a |
| loops | n/a | n/a | — (0 fail) |
| redundancy | n/a | n/a | — (0 fail) |
| near_singular | 0% (0/1) | n/a | n/a |
| mass_ratio | — (0 fail) | — (0 fail) | n/a |
| symbolic | — (0 fail) | — (0 fail) | n/a |

## 3. Scaling wall (timeout+skipped / all cases)

Reported separately from correctness. The parenthesised value is the smallest knob setting at which the pathway stopped returning.

| Axis | lagrangian | hamiltonian | constrained |
|---|---|---|---|
| dof | 0% (0/5) | 60% (3/5) @ 3 | n/a |
| loops | n/a | n/a | 67% (2/3) @ 4 |
| redundancy | n/a | n/a | 0% (0/7) |
| near_singular | 14% (1/7) @ 1e-11 | n/a | n/a |
| mass_ratio | 0% (0/7) | 0% (0/7) | n/a |
| symbolic | 29% (2/7) @ 16 | 29% (2/7) @ 16 | n/a |

## 4. Per-axis detail

### dof

| tool | knob | status | reason | warned | time |
|---|---|---|---|---|---|
| hamiltonian | 1 | pass |  |  | 4.1s |
| hamiltonian | 2 | pass |  |  | 63.5s |
| hamiltonian | 3 | timeout | timeout>180s |  | 180.1s |
| hamiltonian | 4 | timeout | timeout>180s |  | 180.0s |
| hamiltonian | 5 | timeout | timeout>180s |  | 180.0s |
| lagrangian | 1 | pass |  |  | 4.0s |
| lagrangian | 2 | pass |  |  | 7.0s |
| lagrangian | 3 | pass |  |  | 56.0s |
| lagrangian | 4 | pass |  |  | 17.9s |
| lagrangian | 5 | pass |  |  | 20.9s |

### loops

| tool | knob | status | reason | warned | time |
|---|---|---|---|---|---|
| constrained | 3 | pass |  |  | 7.5s |
| constrained | 4 | timeout | timeout>180s |  | 180.1s |
| constrained | 5 | timeout | timeout>180s |  | 180.1s |

### redundancy

| tool | knob | status | reason | warned | time |
|---|---|---|---|---|---|
| constrained | 0 | pass |  |  | 4.1s |
| constrained | 1 | pass |  |  | 4.0s |
| constrained | 2 | pass |  |  | 4.2s |
| constrained | 3 | pass |  |  | 3.9s |
| constrained | 4 | pass |  |  | 4.3s |
| constrained | 6 | pass |  |  | 4.5s |
| constrained | 8 | pass |  |  | 4.5s |

### near_singular

| tool | knob | status | reason | warned | time |
|---|---|---|---|---|---|
| lagrangian | 0.1 | pass |  |  | 4.2s |
| lagrangian | 0.01 | pass |  |  | 4.2s |
| lagrangian | 0.001 | pass |  |  | 4.1s |
| lagrangian | 1e-05 | pass |  |  | 6.5s |
| lagrangian | 1e-08 | pass |  | y | 83.1s |
| lagrangian | 1e-11 | timeout | timeout>180s |  | 180.0s |
| lagrangian | 0 | error | compile_failed | y | 3.4s |

### mass_ratio

| tool | knob | status | reason | warned | time |
|---|---|---|---|---|---|
| hamiltonian | 1 | pass |  |  | 4.1s |
| hamiltonian | 1000 | pass |  |  | 2.9s |
| hamiltonian | 1e+06 | pass |  |  | 4.0s |
| hamiltonian | 1e+09 | pass |  |  | 4.1s |
| hamiltonian | 1e+12 | pass |  |  | 4.1s |
| hamiltonian | 1e+14 | pass |  |  | 4.2s |
| hamiltonian | 1e+16 | pass |  |  | 4.1s |
| lagrangian | 1 | pass |  |  | 4.2s |
| lagrangian | 1000 | pass |  |  | 3.7s |
| lagrangian | 1e+06 | pass |  |  | 4.1s |
| lagrangian | 1e+09 | pass |  | y | 4.2s |
| lagrangian | 1e+12 | pass |  | y | 4.3s |
| lagrangian | 1e+14 | pass |  | y | 4.3s |
| lagrangian | 1e+16 | pass |  | y | 4.0s |

### symbolic

| tool | knob | status | reason | warned | time |
|---|---|---|---|---|---|
| hamiltonian | 1 | pass |  |  | 4.2s |
| hamiltonian | 2 | pass |  |  | 4.2s |
| hamiltonian | 4 | pass |  |  | 3.9s |
| hamiltonian | 8 | pass |  |  | 5.7s |
| hamiltonian | 12 | pass |  |  | 30.6s |
| hamiltonian | 16 | timeout | timeout>180s |  | 180.1s |
| hamiltonian | 24 | timeout | timeout>180s |  | 180.0s |
| lagrangian | 1 | pass |  |  | 4.0s |
| lagrangian | 2 | pass |  |  | 4.1s |
| lagrangian | 4 | pass |  |  | 4.5s |
| lagrangian | 8 | pass |  |  | 7.8s |
| lagrangian | 12 | pass |  |  | 85.1s |
| lagrangian | 16 | timeout | timeout>180s |  | 180.1s |
| lagrangian | 24 | timeout | timeout>180s |  | 180.0s |
