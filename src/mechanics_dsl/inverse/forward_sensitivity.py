"""
Analytic forward sensitivities for MechanicsDSL systems.

For a compiled Lagrangian system with state ``x = (q_1, q_1_dot, ..., q_n, q_n_dot)``
and parameters ``theta``, the equations of motion are ``dx/dt = f(t, x, theta)``.
The forward sensitivity ``S = dx/dtheta`` obeys the linear variational system

    dS/dt = (df/dx) S + df/dtheta,     S(0) = 0   (parameters)
                                       S(0) = e_j (initial conditions)

Both Jacobians are obtained by exact symbolic differentiation of the equations
derived by the compiler, so the sensitivities carry only integration error,
not finite-difference truncation error. The state and sensitivity equations
are integrated together.

This module complements :class:`~mechanics_dsl.inverse.SensitivityAnalyzer`,
whose ``local_sensitivity`` uses finite differences of repeated simulations.

Example::

    from mechanics_dsl import PhysicsCompiler
    from mechanics_dsl.inverse.forward_sensitivity import (
        ForwardSensitivity, pool_rms, uncertainty_weighted_ranking)

    compiler = PhysicsCompiler()
    compiler.compile_dsl(source)

    fs = ForwardSensitivity(compiler, params=["m", "d", "I_c", "beta"])
    res = fs.solve(t_eval=t_measured)
    err = res.rms_error(phi_measured, state="phi")
    ranking = uncertainty_weighted_ranking(err.gradient, sigmas)

Limitations: standard Lagrangian formulation only. Systems compiled with the
Hamiltonian formulation, or with enough coordinates that the compiler defers
the mass-matrix solve to run time, raise ``NotImplementedError``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp

__all__ = [
    "ForwardSensitivity",
    "ForwardSensitivityResult",
    "RMSErrorGradient",
    "pool_rms",
    "uncertainty_weighted_ranking",
]


@dataclass
class RMSErrorGradient:
    """Root-mean-square error and its gradient, kept in poolable form.

    The sum of squared residuals ``sse``, the sample count ``n`` and the
    gradient of ``sse`` are stored so that several runs (release angles,
    added-mass configurations) can be pooled exactly with :func:`pool_rms`.
    """

    sse: float
    n: int
    dsse: dict[str, float]

    @property
    def rms(self) -> float:
        """E = sqrt(sse / n)."""
        return float(np.sqrt(self.sse / self.n))

    @property
    def gradient(self) -> dict[str, float]:
        """dE/dtheta_i = (d sse/dtheta_i) / (2 n E)."""
        e = self.rms
        if e == 0.0:
            return {k: 0.0 for k in self.dsse}
        return {k: float(v / (2.0 * self.n * e)) for k, v in self.dsse.items()}


def pool_rms(parts: Iterable[RMSErrorGradient]) -> RMSErrorGradient:
    """Pool RMS errors from several runs into one RMS over all samples."""
    parts = list(parts)
    if not parts:
        raise ValueError("pool_rms needs at least one RMSErrorGradient")
    keys = list(parts[0].dsse)
    for p in parts[1:]:
        if list(p.dsse) != keys:
            raise ValueError("all parts must be differentiated with respect to the same parameters")
    return RMSErrorGradient(
        sse=float(sum(p.sse for p in parts)),
        n=int(sum(p.n for p in parts)),
        dsse={k: float(sum(p.dsse[k] for p in parts)) for k in keys},
    )


def uncertainty_weighted_ranking(
    gradient: dict[str, float], sigmas: dict[str, float]
) -> list[tuple[str, float]]:
    """Rank parameters by s_i = sigma_i * |dE/dtheta_i|, largest first.

    Args:
        gradient: dE/dtheta_i for each parameter.
        sigmas: plausible uncertainty sigma_i of each parameter, in the
            parameter's own units. Every parameter in ``gradient`` needs one.

    Returns:
        List of (parameter, s_i) sorted by decreasing s_i.
    """
    missing = [k for k in gradient if k not in sigmas]
    if missing:
        raise ValueError(f"no sigma given for: {missing}")
    for k in gradient:
        if not np.isfinite(sigmas[k]) or sigmas[k] < 0:
            raise ValueError(f"sigma for {k!r} must be finite and non-negative")
    s = [(k, float(sigmas[k] * abs(g))) for k, g in gradient.items()]
    return sorted(s, key=lambda kv: kv[1], reverse=True)


@dataclass
class ForwardSensitivityResult:
    """Trajectory and forward sensitivities on a common time grid.

    Attributes:
        t: times, shape (nt,)
        y: states, shape (n_state, nt), rows ordered as ``state_vars``
        S: sensitivities dy/dtheta, shape (n_state, n_wrt, nt), columns
           ordered as ``wrt``
        state_vars: names of the state rows, e.g. ["phi", "phi_dot"]
        wrt: names of the sensitivity columns. Model parameters appear by
           name; initial conditions appear as "x0:<state>".
        param_values: parameter values used for this solution
    """

    t: np.ndarray
    y: np.ndarray
    S: np.ndarray
    state_vars: list[str]
    wrt: list[str]
    param_values: dict[str, float] = field(default_factory=dict)
    message: str = ""

    def state(self, name: str) -> np.ndarray:
        """Trajectory of one state variable."""
        return self.y[self._row(name)]

    def sensitivity(self, state: str, wrt: str) -> np.ndarray:
        """d(state)/d(wrt) over time."""
        if wrt not in self.wrt:
            raise KeyError(f"{wrt!r} is not one of {self.wrt}")
        return self.S[self._row(state), self.wrt.index(wrt)]

    def rms_error(self, measured: Sequence[float], state: str) -> RMSErrorGradient:
        """RMS error of one state against measurements on ``self.t``.

        ``measured`` must be sampled at the same times as this solution
        (pass the measurement times as ``t_eval`` to :meth:`ForwardSensitivity.solve`).
        """
        meas = np.asarray(measured, dtype=float)
        pred = self.state(state)
        if meas.shape != pred.shape:
            raise ValueError(
                f"measured has shape {meas.shape}; prediction has shape {pred.shape}. "
                "Solve with t_eval equal to the measurement times."
            )
        r = pred - meas
        row = self._row(state)
        dsse = {w: float(2.0 * np.dot(r, self.S[row, j])) for j, w in enumerate(self.wrt)}
        return RMSErrorGradient(sse=float(np.dot(r, r)), n=int(r.size), dsse=dsse)

    def _row(self, name: str) -> int:
        if name not in self.state_vars:
            raise KeyError(f"{name!r} is not one of {self.state_vars}")
        return self.state_vars.index(name)


class ForwardSensitivity:
    """Symbolic forward-sensitivity equations for a compiled system.

    Args:
        compiler: a ``PhysicsCompiler`` on which ``compile_dsl`` has succeeded.
        params: names of model parameters to differentiate with respect to.
        initial_conditions: optional names of state variables (e.g. "phi",
            "phi_dot") whose initial values should also be differentiated;
            they appear in results as "x0:<name>".
    """

    def __init__(
        self,
        compiler: Any,
        params: Sequence[str],
        initial_conditions: Sequence[str] = (),
    ):
        if getattr(compiler, "equations", None) is None:
            raise ValueError("compiler has no equations; call compile_dsl first")
        if getattr(compiler, "use_hamiltonian_formulation", False):
            raise NotImplementedError(
                "forward sensitivities support the Lagrangian formulation only"
            )
        eqs = compiler.equations
        if not isinstance(eqs, dict):
            raise NotImplementedError("unsupported equation form for forward sensitivities")
        if "__mass_matrix__" in eqs:
            raise NotImplementedError(
                "this system defers the mass-matrix solve to run time; "
                "forward sensitivities are not yet available for it"
            )

        self.compiler = compiler
        self.symbolic = compiler.symbolic
        self.coordinates: list[str] = list(compiler.get_coordinates())
        self.state_vars: list[str] = []
        for q in self.coordinates:
            self.state_vars.extend([q, f"{q}_dot"])

        self.params: list[str] = list(params)
        self.ic_names: list[str] = list(initial_conditions)
        if len(set(self.params)) != len(self.params):
            raise ValueError("duplicate parameter names")
        bad_ic = [s for s in self.ic_names if s not in self.state_vars]
        if bad_ic:
            raise ValueError(f"unknown state variables {bad_ic}; known: {self.state_vars}")

        self.base_params: dict[str, float] = dict(compiler.simulator.parameters)
        unknown = [p for p in self.params if p not in self.base_params]
        if unknown:
            raise ValueError(f"unknown parameters {unknown}; known: {sorted(self.base_params)}")

        get = self.symbolic.get_symbol
        t_sym = self.symbolic.time_symbol
        x_syms = [get(s) for s in self.state_vars]
        all_param_names = sorted(self.base_params)
        p_syms = [get(p) for p in all_param_names]
        wrt_syms = [get(p) for p in self.params]

        # Right-hand side f(t, x, p): (q_dot, q_ddot) for each coordinate.
        rhs = []
        for q in self.coordinates:
            key = f"{q}_ddot"
            if key not in eqs:
                raise ValueError(f"no equation for {key}")
            acc = compiler.simulator._replace_derivatives(sp.sympify(eqs[key]), self.coordinates)
            rhs.extend([get(f"{q}_dot"), acc])
        f = sp.Matrix(rhs)

        allowed = set(x_syms) | set(p_syms) | {t_sym}
        stray = sorted(str(s) for s in f.free_symbols if s not in allowed)
        if stray:
            raise ValueError(
                f"equations contain symbols with no value: {stray}. "
                "Declare them with \\parameter."
            )

        self.rhs = f
        self.jac_x = f.jacobian(x_syms)
        self.jac_p = f.jacobian(wrt_syms) if wrt_syms else sp.zeros(len(rhs), 0)

        self._all_param_names = all_param_names
        args = (t_sym, x_syms, p_syms)
        self._f = sp.lambdify(args, f, modules="numpy", cse=True)
        self._jx = sp.lambdify(args, self.jac_x, modules="numpy", cse=True)
        self._jp = sp.lambdify(args, self.jac_p, modules="numpy", cse=True)

    @property
    def wrt(self) -> list[str]:
        return self.params + [f"x0:{s}" for s in self.ic_names]

    def solve(
        self,
        t_eval: Sequence[float],
        param_values: dict[str, float] | None = None,
        initial_conditions: dict[str, float] | None = None,
        method: str = "DOP853",
        rtol: float = 1e-10,
        atol: float = 1e-12,
    ) -> ForwardSensitivityResult:
        """Integrate the state and sensitivity equations together.

        Args:
            t_eval: output times; the integration runs from t_eval[0] to t_eval[-1].
            param_values: overrides for parameter values (others keep the
                values from the compiled source).
            initial_conditions: overrides for initial state values.
            method, rtol, atol: passed to ``scipy.integrate.solve_ivp``.
        """
        times = np.asarray(t_eval, dtype=float)
        if times.ndim != 1 or times.size < 2:
            raise ValueError("t_eval must be a 1-D array with at least two times")
        if np.any(np.diff(times) <= 0):
            raise ValueError("t_eval must be strictly increasing")

        pv = dict(self.base_params)
        if param_values:
            unknown = [k for k in param_values if k not in pv]
            if unknown:
                raise ValueError(f"unknown parameters {unknown}")
            pv.update({k: float(v) for k, v in param_values.items()})
        p = [pv[k] for k in self._all_param_names]

        ic = dict(self.compiler.simulator.initial_conditions)
        if initial_conditions:
            ic.update({k: float(v) for k, v in initial_conditions.items()})
        x0 = np.array([float(ic.get(s, 0.0)) for s in self.state_vars])

        n = len(self.state_vars)
        n_p = len(self.params)
        n_w = n_p + len(self.ic_names)
        S0 = np.zeros((n, n_w))
        for j, s in enumerate(self.ic_names):
            S0[self.state_vars.index(s), n_p + j] = 1.0

        f_fn, jx_fn, jp_fn = self._f, self._jx, self._jp

        def rhs(t, z):
            x = z[:n]
            S = z[n:].reshape(n, n_w)
            dx = np.asarray(f_fn(t, x, p), dtype=float).reshape(n)
            Jx = np.asarray(jx_fn(t, x, p), dtype=float).reshape(n, n)
            dS = Jx @ S
            if n_p:
                dS[:, :n_p] += np.asarray(jp_fn(t, x, p), dtype=float).reshape(n, n_p)
            return np.concatenate([dx, dS.ravel()])

        sol = solve_ivp(
            rhs,
            (times[0], times[-1]),
            np.concatenate([x0, S0.ravel()]),
            method=method,
            t_eval=times,
            rtol=rtol,
            atol=atol,
        )
        if not sol.success:
            raise RuntimeError(f"integration failed: {sol.message}")

        y = sol.y[:n]
        S = sol.y[n:].reshape(n, n_w, -1)
        return ForwardSensitivityResult(
            t=sol.t,
            y=y,
            S=S,
            state_vars=list(self.state_vars),
            wrt=self.wrt,
            param_values=pv,
            message=str(sol.message),
        )
