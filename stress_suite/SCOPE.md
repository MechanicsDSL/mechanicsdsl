# Study scope — fixed 10 August 2026

If the plan seems to have moved, this file is the plan. Changing it is a
deliberate act, not something that happens across a conversation.

## The question

**Does a physics engine tell you when it's wrong, or does it hand you a wrong
answer and claim it worked?**

A wrong answer that announces itself is survivable — you go fix it. A wrong
answer that reports success is not, because you have no reason to doubt it.
This study measures how often engines do the second thing.

## What gets tested

Three engines:

1. **MechanicsDSL** — already wired up.
2. **sympy.physics.mechanics** — pip, already installed, takes Lagrangians directly.
3. **Drake** — chosen 22 August over Pinocchio. Install in Ubuntu WSL, not
   native Windows. It is the heavier install of the two, so do it in the first
   days of week 1 rather than discovering it on day four.

Three is enough. One of them is done.

## The test problem

One family: **a chain of pendulum links.** Every engine can express it, which is
the whole reason it was chosen.

Four knobs, dialed up until things break:

- **Mass ratio between links** — one link a trillion times heavier than its neighbour.
- **Near-zero link mass or inertia** — makes the system genuinely degenerate.
- **Degenerate starting positions** — links perfectly aligned or fully extended.
- **Chain length** — thirty links.

These four were picked because they are *both* portable and nasty. They are just
numbers in a model file, so translating them across engines costs nothing, and
every one of them is a known way to wreck a solver.

Chain length alone is not enough. That mostly measures symbolic scaling, which
only MechanicsDSL suffers from, and "my tool is slower than Drake" is not a paper.

## How a run is scored

Four boxes:

| Box | Meaning |
|---|---|
| **pass** | Right answer. |
| **silent** | Wrong answer, reported as success. This is the one that matters. |
| **error** | Wrong or impossible, and said so. Acceptable behaviour. |
| **timeout** | Never finished. Not counted as a failure — it's a missing answer, not a wrong one. |

Wrongness is established by running all three engines on the same problem and
comparing them. If they disagree, at least one is wrong, and the only question
left is which ones admitted it. Energy conservation is the second check; it works
on any engine without needing to understand its internals.

No single engine is the referee. That was the old design and it created a
conflict, because the referee (SymPy) is also a contestant.

## The claim the paper defends

> Dynamics engines disagree on the same mechanical system, and some of them
> don't warn you.

Narrow, true, and defensible. This is a **short or workshop paper**. It is not a
registered report, and pitching it as one invites rejection on thin evidence.

## Two weeks

- **Week 1 — adapters.** The same pendulum chain expressed three ways, all three
  agreeing on an easy case that can be checked by hand. This is the hard part and
  where all the risk lives. Budget the whole week.
- **Week 2 — sweep, table, write.**

## Out of scope — do not reopen

- The other four axes from the old suite (nested cosines, hand-built near-singular
  mass matrices). They only work on engines that eat raw Lagrangians, so they
  cannot support a cross-engine claim.
- Fixing MechanicsDSL bugs once measurement starts. Fixing the tool mid-study is
  what makes the numbers meaningless.
- The ESEM study on test assertions. **Separate project.** Not this.
- Speed comparisons.

## Verify before quoting any number

- **The engine is frozen at `a8dc2b2`, tagged `v2.1.3`** — pinned 22 August,
  recorded in `FREEZE.md`. That file is the anchor; check it before quoting
  anything. If the engine moves again, the baseline is void and the suite has to
  be re-run before any number is cited.
- **The baseline in `RESULTS.md` — zero silent failures in 45 adjudicated cases
  — was measured at the previous pin `d04207c` and is pending re-verification.**
  Do not cite it until the sweep has been repeated at `a8dc2b2` and `FREEZE.md`
  records the result. `MASTER_REPORT.md` describes the pre-fix engine and must
  not be quoted as current; `FIXES.md` explains every difference between them.
- You are MechanicsDSL's maintainer and it is one of the engines under
  measurement. Say so plainly in the paper — a maintainer measuring their own
  tool is normal and defensible when disclosed, and indefensible when not.
- Disclose AI assistance. It is permitted everywhere and concealing it is not.
