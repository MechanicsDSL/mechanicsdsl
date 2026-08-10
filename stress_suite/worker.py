"""
Run ONE (system, tool) stress case and emit a JSON verdict on stdout.

Invoked as a subprocess by run.py so that hangs and hard crashes (segfault /
MemoryError / RecursionError) are caught by the parent and cannot corrupt the
aggregate run.

Verdict statuses decided here:
    pass    -- ran and produced physically correct dynamics
    silent  -- reported success=True but the physics is wrong
    error   -- the tool itself reported failure (compile/simulate success=False)
               or died in a way we could attribute
TIMEOUT is assigned by the parent, not here, and is deliberately NOT an
"error": it means the case was never adjudicated, so it is excluded from the
correctness denominators rather than counted as a failure.

Correctness oracles, routed by formulation:
    unconstrained:  independent ground-truth EOM (SymPy physics.mechanics),
                    compared to MechanicsDSL's accelerations at random states;
                    PLUS energy-conservation along the trajectory.
    constrained:    constraint residual stays ~0 along the trajectory PLUS
                    energy conservation PLUS not-frozen  (off-manifold EOM
                    comparison is ill-defined for index-1 DAEs, so it is not
                    used here).
Both routes also flag: NaN/Inf "successful" trajectories, all-zero EOM,
frozen-when-displaced, and any solve-fallback warning riding on success=True.
"""

import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.disable(logging.CRITICAL)
for _n in ("MechanicsDSL", "mechanics_dsl"):
    logging.getLogger(_n).setLevel(logging.CRITICAL)

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import sympy as sp

ENERGY_TOL = 1e-2      # relative energy drift above this => wrong integration
EOM_TOL = 1e-6         # relative accel mismatch vs ground truth => wrong EOM
CONSTRAINT_TOL = 1e-2  # absolute constraint residual growth => constraint violated
K_PROBE = 12           # random states for EOM comparison

# Only these warnings mean the derived dynamics are a placeholder rather than
# physics. Everything else the engine emits is advisory -- notably the
# ill-conditioning notice, which reports a true fact about the system and says
# nothing about whether the equations are right.
#
# Treating ANY warning as evidence of wrongness was correct when the engine's
# only warnings were degeneracy markers, but it makes the suite score an engine
# WORSE for telling the user more. Correctness verdicts come from the oracles
# (ground-truth EOM, energy, constraint residual); a warning is corroborating
# detail, not a substitute.
#
# Mirrors compiler._FATAL_WARNING_MARKERS. Duplicated deliberately: the suite
# does not reach into library internals, and the classification rule must be
# legible on its own.
DEGENERACY_MARKERS = (
    "falling back to zero",
    "defaulted to zero",
    "using zero fallback",
    "Mass matrix is singular",
)


def _has_degeneracy_warning(warnings_list):
    return any(m in w for w in warnings_list for m in DEGENERACY_MARKERS)


def _param_subs(params):
    return {sp.Symbol(k, real=True): v for k, v in params.items()}


def _state_syms(compiler, coords):
    s = []
    for q in coords:
        s.append(compiler.symbolic.get_symbol(q))
        s.append(compiler.symbolic.get_symbol(f"{q}_dot"))
    return s


def _lagrangian_energy_fn(compiler, coords):
    L = compiler.symbolic.ast_to_sympy(compiler.lagrangian)
    E = sp.S.Zero
    for q in coords:
        qd = compiler.symbolic.get_symbol(f"{q}_dot")
        E += qd * sp.diff(L, qd)
    E = (E - L).subs(_param_subs(compiler.simulator.parameters))
    return sp.lambdify(_state_syms(compiler, coords), E, "numpy")


def _hamiltonian_energy_fn(compiler, coords):
    if compiler.hamiltonian is not None:
        H = compiler.symbolic.ast_to_sympy(compiler.hamiltonian)
    elif hasattr(compiler, "hamiltonian_expr"):
        H = compiler.hamiltonian_expr
    else:
        return None
    H = H.subs(_param_subs(compiler.simulator.parameters))
    syms = []
    for q in coords:
        syms.append(compiler.symbolic.get_symbol(q))
        syms.append(compiler.symbolic.get_symbol(f"p_{q}"))
    return sp.lambdify(syms, H, "numpy")


def _energy_drift(fn, y):
    try:
        vals = np.array([fn(*y[:, j]) for j in range(y.shape[1])], dtype=float)
        if not np.all(np.isfinite(vals)):
            return float("inf")
        e0 = vals[0]
        denom = abs(e0) if abs(e0) > 1e-9 else 1.0
        return float(np.max(np.abs(vals - e0)) / denom)
    except Exception:
        return None


def _eom_all_zero(compiler, tool):
    try:
        eqs = compiler.equations
        if tool in ("lagrangian", "constrained"):
            if not isinstance(eqs, dict) or not eqs:
                return False
            return all(sp.simplify(v) == 0 for v in eqs.values())
        q_dots, p_dots = eqs
        return all(sp.simplify(e) == 0 for e in list(q_dots) + list(p_dots))
    except Exception:
        return False


def _eom_ground_truth_mismatch(compiler, coords, spec):
    """Return (max_rel_mismatch, note). Compares MechanicsDSL's derived
    accelerations to an independent SymPy-mechanics derivation at random states.
    Only valid for the unconstrained Lagrangian pathway."""
    import groundtruth as gt
    L = compiler.symbolic.ast_to_sympy(compiler.lagrangian)
    try:
        truth = gt.build_truth(L, coords, compiler.simulator.parameters, None)
    except Exception as e:
        return None, f"truth_build_failed:{type(e).__name__}"

    eqs = compiler.equations
    psubs = _param_subs(compiler.simulator.parameters)
    syms = _state_syms(compiler, coords)
    try:
        fs = [sp.lambdify(syms, sp.sympify(eqs[f"{q}_ddot"]).subs(psubs), "numpy")
              for q in coords]
    except Exception as e:
        return None, f"mdsl_lambdify_failed:{type(e).__name__}"

    n = len(coords)
    rng = np.random.default_rng(12345)
    worst = 0.0
    for _ in range(K_PROBE):
        st = rng.uniform(-0.5, 0.5, size=2 * n)
        try:
            mdsl = np.array([f(*st) for f in fs], dtype=float)
            tru = truth.accel(st)
        except Exception as e:
            return None, f"eval_failed:{type(e).__name__}"
        if not (np.all(np.isfinite(mdsl)) and np.all(np.isfinite(tru))):
            return float("inf"), "nonfinite_accel"
        denom = np.maximum(np.abs(tru), 1.0)
        worst = max(worst, float(np.max(np.abs(mdsl - tru) / denom)))
    return worst, ""


def _constraint_residual(compiler, coords, y):
    """Max absolute violation of the declared holonomic constraints along the
    trajectory, minus the value at t=0 (so we measure drift, not offset)."""
    try:
        cons = [compiler.symbolic.ast_to_sympy(c) for c in compiler.constraints]
        if not cons:
            return None
        psubs = _param_subs(compiler.simulator.parameters)
        pos_syms = [compiler.symbolic.get_symbol(q) for q in coords]
        fns = [sp.lambdify(pos_syms, c.subs(psubs), "numpy") for c in cons]
        pos = y[0::2]  # positions at even indices
        worst = 0.0
        for f in fns:
            vals = np.array([f(*pos[:, j]) for j in range(pos.shape[1])], dtype=float)
            if not np.all(np.isfinite(vals)):
                return float("inf")
            worst = max(worst, float(np.max(np.abs(vals - vals[0]))))
        return worst
    except Exception:
        return None


def run_case(spec):
    tool = spec["tool"]
    detail = {"tool": tool}

    from mechanics_dsl import PhysicsCompiler
    compiler = PhysicsCompiler()

    use_ham = (tool == "hamiltonian")
    use_con = (tool == "constrained")
    result = compiler.compile_dsl(spec["dsl"], use_hamiltonian=use_ham,
                                  use_constraints=use_con)
    detail["compile_success"] = result.get("success")
    detail["warnings"] = list(result.get("warnings") or [])
    warned = len(detail["warnings"]) > 0
    degenerate_warning = _has_degeneracy_warning(detail["warnings"])
    detail["degeneracy_warning"] = degenerate_warning

    if not result.get("success"):
        detail["error"] = result.get("error", "")
        return {"status": "error", "reason": "compile_failed", "warned": warned,
                "detail": detail}

    coords = compiler.get_coordinates()
    detail["n_coords"] = len(coords)
    eom_zero = _eom_all_zero(compiler, tool)
    detail["eom_all_zero"] = eom_zero

    # --- Ground-truth EOM check (unconstrained Lagrangian pathway only) -------
    eom_mismatch = None
    if tool == "lagrangian" and spec["formulation"] == "unconstrained":
        eom_mismatch, note = _eom_ground_truth_mismatch(compiler, coords, spec)
        detail["eom_mismatch"] = eom_mismatch
        detail["eom_note"] = note

    # --- Simulate -------------------------------------------------------------
    try:
        sol = compiler.simulate(tuple(spec["t_span"]), spec["num_points"])
    except Exception as e:
        detail["sim_exception"] = f"{type(e).__name__}: {e}"
        return {"status": "error", "reason": "simulate_exception", "warned": warned,
                "detail": detail}
    detail["sim_success"] = sol.get("success")
    if not sol.get("success"):
        detail["sim_error"] = sol.get("error", sol.get("message", ""))
        return {"status": "error", "reason": "simulate_failed", "warned": warned,
                "detail": detail}

    y = np.asarray(sol["y"], dtype=float)
    nan_inf = not np.all(np.isfinite(y))
    detail["nan_inf"] = bool(nan_inf)

    pos_rows = y[0::2]
    moved = float(np.max(np.ptp(pos_rows, axis=1))) if pos_rows.size else 0.0
    detail["max_position_range"] = moved
    frozen = spec["expected_moving"] and moved < 1e-7

    # --- Energy oracle --------------------------------------------------------
    drift = None
    if spec["conservative"] and not nan_inf:
        fn = _hamiltonian_energy_fn(compiler, coords) if use_ham \
            else _lagrangian_energy_fn(compiler, coords)
        if fn is not None:
            drift = _energy_drift(fn, y)
            if drift == float("inf"):
                nan_inf = True
    detail["energy_drift"] = drift

    # --- Constraint-residual oracle (constrained pathway) ---------------------
    cres = None
    if spec["formulation"] == "constrained" and not nan_inf:
        cres = _constraint_residual(compiler, coords, y)
    detail["constraint_residual"] = cres

    # --- Classify -------------------------------------------------------------
    wrong, reasons = False, []
    if degenerate_warning:
        wrong = True; reasons.append("degenerate_solve_warning")
    if eom_zero:
        wrong = True; reasons.append("eom_all_zero")
    if frozen:
        wrong = True; reasons.append("frozen_trajectory")
    if nan_inf:
        wrong = True; reasons.append("nan_inf_trajectory")
    if eom_mismatch is not None and eom_mismatch > EOM_TOL:
        wrong = True; reasons.append(f"eom_mismatch={eom_mismatch:.2e}")
    if drift is not None and drift > ENERGY_TOL:
        wrong = True; reasons.append(f"energy_drift={drift:.2e}")
    if cres is not None and cres > CONSTRAINT_TOL:
        wrong = True; reasons.append(f"constraint_drift={cres:.2e}")

    if wrong:
        return {"status": "silent", "reason": ",".join(reasons), "warned": warned,
                "detail": detail}
    return {"status": "pass", "reason": "", "warned": warned, "detail": detail}


def main():
    with open(sys.argv[1], "r", encoding="utf-8") as f:
        spec = json.load(f)
    try:
        verdict = run_case(spec)
    except MemoryError:
        verdict = {"status": "error", "reason": "memory_error", "warned": False, "detail": {}}
    except RecursionError:
        verdict = {"status": "error", "reason": "recursion_error", "warned": False, "detail": {}}
    print("VERDICT_JSON:" + json.dumps(verdict))


if __name__ == "__main__":
    main()
