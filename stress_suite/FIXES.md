# Every fix, and why — 10 August 2026

Two kinds of change happened. The engine got fixed, and the thing measuring the
engine got fixed. Both moved the numbers, and it matters which is which.

---

## Engine fixes

### 1. A degenerate solve no longer reports success

When the symbolic solve gave up, it filled in zero accelerations and returned
`success = True` with a note buried in a warnings list nothing checked. You got
a placeholder and a green light.

Now it fails. Exception: a declared variable with no kinetic energy term isn't
a real coordinate, and zero acceleration is the correct answer for it, so that
case still passes.

### 2. Coupled Hamiltonian systems no longer freeze

The Legendre transform solved for one velocity at a time. When momenta are
coupled that breaks down, and a two-link pendulum would compile, simulate, and
sit perfectly still while reporting success.

Now the whole momentum system is inverted at once, and relations that can't be
inverted raise an error instead of producing a half-built Hamiltonian.

*Evidence:* the two-link pendulum went from frozen to an 0.84 rad swing
conserving energy to 2.4e-7.

### 3. Large systems stopped hitting a wall

Inverting the mass matrix symbolically blows up as systems grow. Past a size
threshold the engine now keeps the matrix and solves `M q̈ = F` numerically at
each step instead.

*Evidence:* four and five coupled pendulum links used to run past 180 seconds
and get killed. They now finish in 6.8 and 11.2 seconds, correctly.

### 4. Ill-conditioning is now detected

The engine estimates how badly conditioned the mass matrix is at the starting
state and warns when it's bad — long before it becomes exactly singular.

### 5. Integration accuracy now matches the problem *(this is the one that got to 0%)*

An ill-conditioned mass matrix means the system has one vibration mode far
faster than the others. At ε=1e-8 that mode runs at 10,000 rad/s — about
16,000 oscillations in a ten-second run. Integrating that at the default
accuracy setting accumulates a little error per oscillation, and it adds up.

The engine was already computing the number that predicts this (fix 4) and then
integrating at a setting meant for easy problems anyway.

Now the accuracy setting scales with the conditioning, with a floor at 1e-13 —
past that you're chasing round-off and the system needs rescaling, not smaller
steps.

*Evidence:* energy drift on that case went from **3.7% to 0.00017%** — a factor
of 21,000 — for 2.1× the computation. Other cases got more accurate too, at no
measurable cost.

### 6. A singular mass matrix mid-run is no longer papered over

If the matrix went singular partway through a simulation, the solver quietly
substituted a least-squares answer. That's a mathematically well-defined
solution to a question the model doesn't actually answer — a plausible
trajectory that isn't the system's motion. It was mentioned only in a log
nobody reads.

Now those events are counted and fail the simulation, with a message pointing
at the likely causes.

### 7. Stiffness detection actually does something

It ran a test integration, concluded "this system is stiff," logged *"consider
using LSODA or Radau"* — and then integrated with the original method anyway.
It was also gated so that it usually never ran at all.

Now it runs for any explicit method and switches.

**Untested.** No case in this suite triggers it. It's a fix by inspection.

### 8. Generated Rust compiles

Parameter constants were being upper-cased to match Rust style while the
generated equations still referred to the original lower-case name, so the
output didn't build. Unrelated to the physics.

---

## Test suite fixes

These changed the reported numbers **without changing the engine at all**, which
is exactly why they're listed separately.

### 9. Timeouts are no longer counted as failures

Statuses used to be pass / silent / loud, with timeouts lumped in with real
failures. That made the headline rate depend on the time limit you picked —
a fact about the test harness, not the engine.

Now: **pass**, **silent**, **error**, **timeout**. A timeout is a missing
answer, not a wrong one, so it sits outside the correctness denominator and
gets reported separately as the scaling wall.

### 10. A warning is no longer treated as proof of wrong physics

The classifier flagged *any* warning on a successful result as a silent
failure. That was right when the only warnings meant broken math — but once the
engine started emitting advisory ill-conditioning notices (fix 4), the suite
punished it for communicating.

*Impact:* four mass-ratio cases matching the independent derivation to 1e-25 and
conserving energy to 5e-8 were being scored as silent failures purely for having
been warned about. Only genuine degeneracy markers count now.

### 11. The independent check now actually runs

The strongest oracle — comparing against a separately written derivation —
worked by reading the engine's symbolic equations. On the numeric path (fix 3)
there are no symbolic equations to read, so it errored, returned nothing, and
the classifier read *nothing* as *nothing wrong*.

The two cases fix 3 had just rescued were reporting verified passes that were
backed by energy conservation alone.

It now probes the actual equations of motion the integrator runs, which works
on both paths and tests what the engine *does* rather than what it derived.

*Evidence:* those two cases now check out at 0.0 and 1.8e-15. And where both
methods apply they agree exactly, which is how we know the new probe is sound.

### 12. A check that couldn't run is now visible

Following from 11: if the independent oracle was applicable but failed, the case
is flagged `unverified` and called out in the report. A silent-failure rate is
only worth the checks that actually ran, and this makes a hollow 0% impossible
to report by accident.

---

## The short version

Three silent failures existed. All three are gone, and we know why each one
happened:

| Silent failure | Cause | Fixed by |
|---|---|---|
| Two-link Hamiltonian froze | Velocities solved one at a time | 2 |
| Exact singularity returned zeros | Success not gated on degeneracy | 1 |
| Near-singular drifted 3.7% | Accuracy setting ignored the conditioning | 5 |

Four more were never real — the suite was miscounting warnings (10).

Two latent problems were found by reading the code rather than by testing, and
fixed before they ever bit: the least-squares substitution (6) and the inert
stiffness detection (7).
