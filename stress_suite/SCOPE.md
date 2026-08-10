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
3. **Pinocchio or Drake** — pick one. Install in Ubuntu WSL, not native Windows.

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

- `out/results.json` is stale. It predates the uncommitted fixes in `src/`, which
  already repaired the Hamiltonian freeze. Re-run before citing anything from
  `MASTER_REPORT.md`.
- MechanicsDSL has an internal 5-second timeout that raises rather than returning
  a failure — `dof_N3__hamiltonian` dies on it in ~5s, not the 180s the old run
  recorded. Check whether it is silently truncating other cases.
- You wrote MechanicsDSL. Say so plainly in the paper.
