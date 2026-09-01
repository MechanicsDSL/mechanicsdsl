"""Adjudicate the constrained pathway against an independent derivation.

These 8 cases have never had one: the frozen baseline scored them on energy
conservation and constraint residual alone.
"""
import logging, warnings, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")))
logging.disable(logging.CRITICAL); warnings.filterwarnings("ignore")
import numpy as np, systems, worker, reference_constrained as RC
from mechanics_dsl import PhysicsCompiler

K, TOL, SEED = 16, 1e-8, 20260822
print("Constrained pathway vs the independent KKT reference")
print(f"  probes: {K} random states/case, tolerance {TOL:g} relative\n")
print(f"{'case':<14}{'coords':>7}{'cons':>6}{'worst err':>13}  verdict")
print("-"*52)
bad = 0
for case in systems.all_cases():
    # loops_N4 and loops_N5 are baseline TIMEOUTS, not adjudicated
    # cases; the blind spot is the 8 cases that were scored.
    if case["axis"] not in ("loops", "redundancy"):
        continue
    if case["name"] in ("loops_N4", "loops_N5"):
        continue
    ref = RC.constrained_reference_for_case(case)
    n = ref.n
    coords = ([f"{a}{i}" for i in range(1, case['knob']) for a in ("x","y")]
              if case["axis"] == "loops" else ["x","y"])
    try:
        c = PhysicsCompiler()
        r = c.compile_dsl(case["dsl"], use_hamiltonian=False, use_constraints=True)
        if not r.get("success"):
            print(f"{case['name']:<14}{n:>7}{len(ref.jac_gamma(np.zeros(n),np.zeros(n))[1]):>6}"
                  f"{'--':>13}  refused"); bad += 1; continue
        fn, route = worker._engine_accel_fn(c, coords)
        if fn is None:
            print(f"{case['name']:<14}{n:>7}{'':>6}{'--':>13}  no route: {route}"); bad += 1; continue
        rng = np.random.default_rng(SEED)
        y0 = ref.initial_state()
        worst = 0.0
        for _ in range(K):
            # Perturb around the manifold so states stay physically meaningful.
            st = y0 + rng.uniform(-0.05, 0.05, size=2*n)
            a_e = np.asarray(fn(st), dtype=float)
            a_r = ref.accel(st)
            worst = max(worst, float(np.max(np.abs(a_e-a_r)/np.maximum(np.abs(a_r),1.0))))
        nc = len(ref.jac_gamma(y0[0::2], y0[1::2])[1])
        ok = worst <= TOL
        if not ok: bad += 1
        print(f"{case['name']:<14}{n:>7}{nc:>6}{worst:13.3e}  {'AGREE' if ok else 'DISAGREE'}")
    except Exception as e:
        print(f"{case['name']:<14}{n:>7}{'':>6}{'--':>13}  error: {type(e).__name__}: {str(e)[:28]}")
        bad += 1
print()
print(f"  cases disagreeing or unavailable: {bad}")
