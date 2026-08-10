"""
Orchestrate the MechanicsDSL stress suite.

For every (system, tool) case: write a spec, run worker.py as a subprocess with
a wall-clock timeout, and classify the outcome into one of four statuses:

    pass     -- ran and produced physically correct dynamics
    silent   -- reported success=True but the physics is wrong
    error    -- reported failure, raised, or crashed (attributable)
    timeout  -- exceeded the wall clock; never returned a verdict

plus `skipped` for cases excluded up front via --skip-known-slow.

`timeout` is deliberately NOT folded in with `error`. An errored case is one
where the engine reached a conclusion and that conclusion was failure; a timed
-out case was never adjudicated at all. Mixing them makes the silent-failure
rate a function of the chosen --timeout, which is a property of the harness
rather than of the tool. Correctness fractions therefore use ADJUDICATED cases
(pass + silent + error) as the denominator, and the timeout rate is reported
separately as the scaling wall.

Usage:
    python run.py [--timeout SECONDS] [--src PATH] [--skip-known-slow]
    python run.py --report-only            # rebuild report.md from results.json

Outputs (in stress_suite/out/):
    results.json  -- every case verdict with full diagnostics
    report.md     -- correctness fractions and the scaling wall, per tool/axis
"""

import argparse
import json
import os
import subprocess
import sys
import time
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SRC = os.path.abspath(os.path.join(HERE, "..", "src"))

sys.path.insert(0, HERE)
import systems  # noqa: E402

STATUSES = ["pass", "silent", "error", "timeout", "skipped"]

# Statuses that represent a case the engine actually adjudicated. Timeouts and
# skips are excluded: they are absence of evidence, not evidence of failure.
ADJUDICATED = ("pass", "silent", "error")


def _normalize(rec):
    """Map the pre-2026-08 {ok, loud} statuses onto the four-way taxonomy so an
    older results.json can still be re-reported via --report-only."""
    st = rec.get("status")
    if st == "ok":
        rec["status"] = "pass"
    elif st == "loud":
        rec["status"] = "timeout" if rec.get("reason") == "timeout" else "error"
    return rec


def run_one(case, tool, specdir, timeout, src):
    spec = dict(case)
    spec["tool"] = tool
    spec_path = os.path.join(specdir, f"{case['name']}__{tool}.json")
    with open(spec_path, "w", encoding="utf-8") as f:
        json.dump(spec, f)

    env = dict(os.environ)
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")

    t0 = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, os.path.join(HERE, "worker.py"), spec_path],
            capture_output=True, text=True, timeout=timeout, env=env, cwd=HERE)
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "reason": f"timeout>{timeout}s", "warned": False,
                "detail": {"tool": tool}, "elapsed": time.time() - t0}

    elapsed = time.time() - t0
    out = proc.stdout or ""
    idx = out.rfind("VERDICT_JSON:")
    if idx == -1:
        return {"status": "error", "reason": f"crash(exit={proc.returncode})",
                "warned": False,
                "detail": {"tool": tool, "stderr_tail": (proc.stderr or "")[-400:]},
                "elapsed": elapsed}
    verdict = json.loads(out[idx + len("VERDICT_JSON:"):].splitlines()[0])
    verdict["elapsed"] = elapsed
    return verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=int, default=180,
                    help="wall-clock seconds per (system,tool) case (default 180)")
    ap.add_argument("--src", default=DEFAULT_SRC,
                    help="path to MechanicsDSL src/ (default: ../src)")
    ap.add_argument("--skip-known-slow", action="store_true",
                    help="do not spend the wall clock on the (case,tool) pairs in "
                         "systems.KNOWN_SLOW; record them as 'skipped'")
    ap.add_argument("--report-only", action="store_true",
                    help="rebuild report.md from the existing out/results.json "
                         "without re-running any case")
    args = ap.parse_args()

    outdir = os.path.join(HERE, "out")
    specdir = os.path.join(outdir, "specs")
    os.makedirs(specdir, exist_ok=True)

    if args.report_only:
        path = os.path.join(outdir, "results.json")
        with open(path, "r", encoding="utf-8") as f:
            results = [_normalize(r) for r in json.load(f)]
        write_report(results, outdir, args.timeout)
        print(f"Rebuilt {os.path.join(outdir, 'report.md')} from {len(results)} "
              f"cached cases (no runs performed)")
        return

    if not os.path.isdir(os.path.join(args.src, "mechanics_dsl")):
        print(f"ERROR: no mechanics_dsl package under --src={args.src}", file=sys.stderr)
        sys.exit(2)

    cases = systems.all_cases()
    total = sum(len(c["tools"]) for c in cases)
    print(f"MechanicsDSL stress suite: {len(cases)} systems, {total} "
          f"(system,tool) cases, timeout={args.timeout}s each")
    if args.skip_known_slow:
        print(f"Skipping {len(systems.KNOWN_SLOW)} known-slow cases "
              f"(excluded from all denominators)")
    print()

    results, done = [], 0
    for case in cases:
        for tool in case["tools"]:
            done += 1
            if args.skip_known_slow and (case["name"], tool) in systems.KNOWN_SLOW:
                v = {"status": "skipped", "reason": "known_slow", "warned": False,
                     "detail": {"tool": tool}, "elapsed": 0.0}
            else:
                v = run_one(case, tool, specdir, args.timeout, args.src)
            rec = {"axis": case["axis"], "level": case["level"], "knob": case["knob"],
                   "name": case["name"], "tool": tool, "formulation": case["formulation"],
                   "status": v["status"], "reason": v.get("reason", ""),
                   "warned": v.get("warned", False),
                   "elapsed": round(v.get("elapsed", 0.0), 2),
                   "detail": v.get("detail", {})}
            results.append(rec)
            flag = {"pass": " pass ", "silent": "SILENT", "error": "ERROR ",
                    "timeout": "t/out ", "skipped": " skip "}[v["status"]]
            print(f"[{done:3d}/{total}] {flag} {case['name']:<18} {tool:<12} "
                  f"{rec['reason'][:44]:<44} {rec['elapsed']:7.1f}s")

    with open(os.path.join(outdir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    write_report(results, outdir, args.timeout)
    print(f"\nWrote {os.path.join(outdir, 'results.json')}")
    print(f"Wrote {os.path.join(outdir, 'report.md')}")


def _frac(num, den, na="—"):
    if den == 0:
        return na
    return f"{num / den * 100:.0f}% ({num}/{den})"


def write_report(results, outdir, timeout):
    axes, tools = systems.AXES, systems.TOOLS
    cells = defaultdict(list)
    for r in results:
        cells[(r["axis"], r["tool"])].append(r)

    def stat(axis, tool):
        rs = cells.get((axis, tool))
        if not rs:
            return None
        c = {s: sum(1 for r in rs if r["status"] == s) for s in STATUSES}
        c["n"] = len(rs)
        c["adj"] = sum(c[s] for s in ADJUDICATED)
        c["fail"] = c["silent"] + c["error"]
        # smallest knob at which the pathway stopped returning
        walls = [r for r in rs if r["status"] in ("timeout", "skipped")]
        c["wall"] = min(walls, key=lambda r: r["level"])["knob"] if walls else None
        return c

    def knobfmt(k):
        if k is None:
            return "—"
        return f"{k:g}" if isinstance(k, float) else str(k)

    L = []
    L.append("# MechanicsDSL symbolic-dynamics stress report\n")
    L.append(f"Per-case wall-clock timeout: **{timeout}s**. Tools are MechanicsDSL "
             "formulation pathways (Lagrangian / Hamiltonian / constrained "
             "Lagrange-multiplier).\n")
    L.append("## Status taxonomy\n")
    L.append("| Status | Meaning |")
    L.append("|---|---|")
    L.append("| **pass** | Ran and produced correct dynamics on every applicable check. |")
    L.append("| **silent** | Returned `success=True` but the physics is wrong. |")
    L.append("| **error** | Reported `success=False`, raised, or crashed — a *loud*, "
             "attributable failure. |")
    L.append("| **timeout** | Exceeded the wall clock. Never adjudicated. |")
    L.append("| **skipped** | Known to exceed even a 600s budget; not run. Never adjudicated. |")
    L.append("")
    L.append("Correctness fractions below use **adjudicated** cases "
             "(pass + silent + error) as the denominator. Timeouts and skips are "
             "absence of evidence, not evidence of failure — folding them in would "
             "make every rate a function of `--timeout`, which is a property of this "
             "harness rather than of the engine. The timeout rate is reported "
             "separately, as the scaling wall.\n")
    L.append("Correctness is judged by an *independent* SymPy-mechanics ground-truth "
             "derivation of the equations of motion (compared numerically at random "
             "states, unconstrained pathway), plus energy-conservation, "
             "constraint-residual, frozen-trajectory, NaN/Inf, all-zero-EOM, and "
             "solve-fallback-warning checks.\n")

    counts = defaultdict(int)
    for r in results:
        counts[r["status"]] += 1
    adj_total = sum(counts[s] for s in ADJUDICATED)
    L.append(f"Totals across {len(results)} cases: "
             + ", ".join(f"**{counts[s]} {s}**" for s in STATUSES if counts[s])
             + f". Adjudicated: **{adj_total}**.\n")

    header = "| Axis | " + " | ".join(tools) + " |"
    sep = "|" + "---|" * (len(tools) + 1)

    L.append("## 1. Silent-failure rate (silent / adjudicated)\n")
    L.append("The headline number: of the cases the engine actually returned a "
             "verdict on, how often was that verdict wrong *and* claimed to be "
             "successful.\n")
    L += [header, sep]
    for axis in axes:
        row = [axis]
        for tool in tools:
            s = stat(axis, tool)
            row.append("n/a" if s is None else _frac(s["silent"], s["adj"], "— (0 adj)"))
        L.append("| " + " | ".join(row) + " |")
    L.append("")

    L.append("## 2. Of adjudicated failures, how many were silent "
             "(silent / (silent + error))\n")
    L += [header, sep]
    for axis in axes:
        row = [axis]
        for tool in tools:
            s = stat(axis, tool)
            row.append("n/a" if s is None else _frac(s["silent"], s["fail"], "— (0 fail)"))
        L.append("| " + " | ".join(row) + " |")
    L.append("")

    L.append("## 3. Scaling wall (timeout+skipped / all cases)\n")
    L.append("Reported separately from correctness. The parenthesised value is the "
             "smallest knob setting at which the pathway stopped returning.\n")
    L += [header, sep]
    for axis in axes:
        row = [axis]
        for tool in tools:
            s = stat(axis, tool)
            if s is None:
                row.append("n/a")
                continue
            nw = s["timeout"] + s["skipped"]
            if nw == 0:
                row.append(f"0% (0/{s['n']})")
            else:
                row.append(f"{nw / s['n'] * 100:.0f}% ({nw}/{s['n']}) "
                           f"@ {knobfmt(s['wall'])}")
        L.append("| " + " | ".join(row) + " |")
    L.append("")

    L.append("## 4. Per-axis detail\n")
    for axis in axes:
        L.append(f"### {axis}\n")
        L.append("| tool | knob | status | reason | warned | time |")
        L.append("|---|---|---|---|---|---|")
        for r in sorted([x for x in results if x["axis"] == axis],
                        key=lambda x: (x["tool"], x["level"])):
            L.append(f"| {r['tool']} | {knobfmt(r['knob'])} | {r['status']} | "
                     f"{r['reason'][:42]} | {'y' if r['warned'] else ''} | "
                     f"{r['elapsed']:.1f}s |")
        L.append("")

    with open(os.path.join(outdir, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))


if __name__ == "__main__":
    main()
