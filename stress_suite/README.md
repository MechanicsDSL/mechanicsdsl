# MechanicsDSL symbolic-dynamics stress suite

A test suite that dials mechanical systems up along six stress axes until
MechanicsDSL fails, and records whether it failed **loudly** (crash, timeout,
reported error) or **silently** (ran fine, produced wrong physics). It reports
the silent-failure fraction per stress axis per formulation pathway.

This suite lives entirely outside `src/` and never imports or modifies library
internals beyond the public `PhysicsCompiler` API, so it does not touch
MechanicsDSL itself.

## What it measures

Six axes (`systems.py`):

| Axis | System | Knob dialed |
|------|--------|-------------|
| `dof` | serial N-pendulum (dense coupled mass matrix) | N = 1…5 |
| `loops` | pinned closed kinematic chain (rod-length constraints) | N = 3…5 |
| `redundancy` | particle on a circle + R dependent duplicate constraints | R = 0…8 |
| `near_singular` | 2-DOF system, mass-matrix det → 0 | ε = 1e-1…0 |
| `mass_ratio` | two masses + spring, m₂ = 10ᵏ | k = 0…16 |
| `symbolic` | pendulum with V = 1−cos(cos(…cos θ…)) nested to depth D | D = 1…24 |

Three "tools" = MechanicsDSL formulation pathways: **lagrangian**,
**hamiltonian**, **constrained** (Lagrange-multiplier). Each axis tests only the
pathways that are meaningful for it.

## How a case is judged

Each `(system, tool)` case runs in an isolated subprocess (`worker.py`) with a
wall-clock timeout, so hangs and hard crashes are caught by the parent.

- **loud** — `compile_dsl`/`simulate` returned `success=False`, the subprocess
  crashed, or it exceeded the timeout.
- **silent** — returned `success=True` but the physics is wrong, detected by:
  - **ground-truth EOM mismatch** (unconstrained pathways): an *independent*
    derivation via `sympy.physics.mechanics.LagrangesMethod` (`groundtruth.py`),
    compared to MechanicsDSL's accelerations numerically at 12 random states;
  - **energy drift** > 1% relative along the trajectory (energy derived from the
    same Lagrangian/Hamiltonian);
  - **constraint-residual drift** (constrained pathway): declared constraints
    stop being satisfied along the trajectory;
  - a **NaN/Inf** trajectory reported as successful, an **all-zero EOM**, a
    **frozen** trajectory when the system was displaced, or a **solve-fallback
    warning** riding on a `success=True` result.
- **ok** — ran and produced correct dynamics on every applicable check.

The two oracles are complementary: EOM-mismatch catches a wrong *derivation*;
energy/constraint drift catches a wrong *integration* even when the symbolic
equations are exactly right.

## Running it

Requires the same environment MechanicsDSL runs in (Python 3.9+, numpy, scipy,
sympy). From this directory:

```bash
python run.py
```

By default it finds the library at `../src`. Override if needed, and raise the
per-case timeout to probe deeper into the scaling wall:

```bash
python run.py --timeout 600 --src /path/to/mechanicsdsl/src
```

On Windows PowerShell:

```powershell
python run.py --timeout 600
```

If `mechanics_dsl` is installed as a package (e.g. `pip install
mechanicsdsl-core`) rather than sitting in `../src`, pass `--src .` — the suite
prepends `--src` to `PYTHONPATH`, and the installed package is then imported
normally.

## Outputs

Written to `stress_suite/out/`:

- `report.md` — the two headline tables (silent/all, silent/failures) plus a
  per-axis breakdown with the knob value at which behavior changes.
- `results.json` — every case verdict with full diagnostics (energy drift, EOM
  mismatch, constraint residual, timings, warnings).
- `specs/` — the exact spec handed to each worker (for reproducing one case:
  `python worker.py out/specs/<name>__<tool>.json`).

## Files

- `systems.py` — the six axis generators (edit sweeps/knobs here).
- `groundtruth.py` — independent EOM oracle via SymPy mechanics.
- `worker.py` — runs and classifies one case.
- `run.py` — orchestrates, times out, aggregates, writes the report.
