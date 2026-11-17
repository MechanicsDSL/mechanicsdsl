"""
Tests for analytic forward sensitivities (mechanics_dsl.inverse.forward_sensitivity).

Every reference below is written directly in NumPy/SciPy from the closed-form
equation of motion and shares no code with MechanicsDSL. Sensitivities are
checked against exact solutions where they exist and against central finite
differences of the independent reference otherwise.
"""

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from mechanics_dsl import PhysicsCompiler
from mechanics_dsl.inverse.forward_sensitivity import (
    ForwardSensitivity,
    RMSErrorGradient,
    pool_rms,
    uncertainty_weighted_ranking,
)

# ---------------------------------------------------------------------------
# Systems
# ---------------------------------------------------------------------------

OSCILLATOR = r"""
\system{oscillator}
\defvar{x}{Position}{m}
\parameter{m}{2.0}{kg}
\parameter{k}{5.0}{N/m}
\lagrangian{\frac{1}{2} * m * \dot{x}^2 - \frac{1}{2} * k * x^2}
\initial{x=0.3, x_dot=0.0}
"""

ELBOW = r"""
\system{elbow_pendulum}
\defvar{phi}{Angle}{rad}
\parameter{m}{0.40}{kg}
\parameter{d}{0.080}{m}
\parameter{I_c}{0.0012}{kg*m^2}
\parameter{beta}{0.10}{rad}
\parameter{c}{0.0020}{kg*m^2/s}
\parameter{g}{9.81}{m/s^2}
\lagrangian{\frac{1}{2} * (I_c + m * d^2) * \dot{phi}^2 + m * g * d * \cos{beta} * \cos{phi}}
\force{-c * \dot{phi}}
\initial{phi=0.6, phi_dot=0.0}
"""

DOUBLE = r"""
\system{double_pendulum}
\defvar{theta1}{Angle}{rad}
\defvar{theta2}{Angle}{rad}
\parameter{m1}{1.0}{kg}
\parameter{m2}{0.7}{kg}
\parameter{l1}{1.0}{m}
\parameter{l2}{0.8}{m}
\parameter{g}{9.81}{m/s^2}
\lagrangian{\frac{1}{2} * (m1 + m2) * l1^2 * \dot{theta1}^2 + \frac{1}{2} * m2 * l2^2 * \dot{theta2}^2 + m2 * l1 * l2 * \dot{theta1} * \dot{theta2} * \cos{theta1 - theta2} + (m1 + m2) * g * l1 * \cos{theta1} + m2 * g * l2 * \cos{theta2}}
\initial{theta1=0.4, theta1_dot=0.0, theta2=-0.2, theta2_dot=0.0}
"""

ELBOW_P = {"m": 0.40, "d": 0.080, "I_c": 0.0012, "beta": 0.10, "c": 0.0020, "g": 9.81}
DOUBLE_P = {"m1": 1.0, "m2": 0.7, "l1": 1.0, "l2": 0.8, "g": 9.81}


def _compile(src):
    c = PhysicsCompiler()
    r = c.compile_dsl(src)
    assert r["success"], r.get("error")
    return c


# ---------------------------------------------------------------------------
# Independent NumPy references (no MechanicsDSL code)
# ---------------------------------------------------------------------------


def ref_elbow(t, p, phi0=0.6, w0=0.0):
    def f(_, z):
        phi, w = z
        I_o = p["I_c"] + p["m"] * p["d"] ** 2
        return [w, -(p["c"] * w + p["m"] * p["g"] * p["d"] * np.cos(p["beta"]) * np.sin(phi)) / I_o]

    s = solve_ivp(f, (t[0], t[-1]), [phi0, w0], t_eval=t, method="DOP853", rtol=1e-12, atol=1e-14)
    return s.y[0]


def ref_double(t, p, z0=(0.4, -0.2, 0.0, 0.0)):
    # Point-mass double pendulum, angles from the downward vertical.
    def f(_, z):
        a1, a2, w1, w2 = z
        m1, m2, l1, l2, g = p["m1"], p["m2"], p["l1"], p["l2"], p["g"]
        dlt = a1 - a2
        den = 2 * m1 + m2 - m2 * np.cos(2 * dlt)
        dd1 = (
            -g * (2 * m1 + m2) * np.sin(a1)
            - m2 * g * np.sin(a1 - 2 * a2)
            - 2 * np.sin(dlt) * m2 * (w2**2 * l2 + w1**2 * l1 * np.cos(dlt))
        ) / (l1 * den)
        dd2 = (
            2
            * np.sin(dlt)
            * (w1**2 * l1 * (m1 + m2) + g * (m1 + m2) * np.cos(a1) + w2**2 * l2 * m2 * np.cos(dlt))
        ) / (l2 * den)
        return [w1, w2, dd1, dd2]

    s = solve_ivp(f, (t[0], t[-1]), list(z0), t_eval=t, method="DOP853", rtol=1e-12, atol=1e-14)
    return s.y[0], s.y[1]


def central_fd(fun, p, name, rel=1e-4):
    # Step chosen so truncation (~rel^2) and integrator rounding (~1e-12/rel)
    # both stay well below the 1e-6 tolerance.
    h = rel * abs(p[name])
    up, dn = dict(p), dict(p)
    up[name] += h
    dn[name] -= h
    return (np.asarray(fun(up)) - np.asarray(fun(dn))) / (2 * h)


# ---------------------------------------------------------------------------
# Exact solutions
# ---------------------------------------------------------------------------


class TestOscillatorExact:
    """x'' = -(k/m) x, x(0) = x0, v(0) = 0 has x = x0 cos(w t), w = sqrt(k/m)."""

    def setup_method(self):
        self.c = _compile(OSCILLATOR)
        self.fs = ForwardSensitivity(self.c, params=["k", "m"], initial_conditions=["x"])
        self.t = np.linspace(0.0, 10.0, 401)
        self.res = self.fs.solve(self.t)
        self.m, self.k, self.x0 = 2.0, 5.0, 0.3
        self.w = np.sqrt(self.k / self.m)

    def test_trajectory(self):
        np.testing.assert_allclose(
            self.res.state("x"), self.x0 * np.cos(self.w * self.t), atol=1e-9
        )

    def test_dx_dk(self):
        dw_dk = 1.0 / (2.0 * np.sqrt(self.k * self.m))
        exact = -self.x0 * self.t * np.sin(self.w * self.t) * dw_dk
        np.testing.assert_allclose(self.res.sensitivity("x", "k"), exact, atol=1e-9)

    def test_dx_dm(self):
        dw_dm = -np.sqrt(self.k) / (2.0 * self.m**1.5)
        exact = -self.x0 * self.t * np.sin(self.w * self.t) * dw_dm
        np.testing.assert_allclose(self.res.sensitivity("x", "m"), exact, atol=1e-9)

    def test_initial_condition_sensitivity(self):
        np.testing.assert_allclose(
            self.res.sensitivity("x", "x0:x"), np.cos(self.w * self.t), atol=1e-9
        )
        np.testing.assert_allclose(
            self.res.sensitivity("x_dot", "x0:x"), -self.w * np.sin(self.w * self.t), atol=1e-9
        )

    def test_scaling_symmetry(self):
        # x depends on k/m only, so k dx/dk + m dx/dm = 0.
        s = self.k * self.res.sensitivity("x", "k") + self.m * self.res.sensitivity("x", "m")
        np.testing.assert_allclose(s, 0.0, atol=1e-9)


# ---------------------------------------------------------------------------
# Finite differences of independent references
# ---------------------------------------------------------------------------


class TestElbowPendulum:
    def setup_method(self):
        self.c = _compile(ELBOW)
        self.names = ["m", "d", "I_c", "beta", "c"]
        self.fs = ForwardSensitivity(self.c, params=self.names)
        self.t = np.linspace(0.0, 3.0, 301)
        self.res = self.fs.solve(self.t)

    def test_trajectory_matches_reference(self):
        np.testing.assert_allclose(self.res.state("phi"), ref_elbow(self.t, ELBOW_P), atol=1e-9)

    @pytest.mark.parametrize("name", ["m", "d", "I_c", "beta", "c"])
    def test_sensitivity_matches_central_fd(self, name):
        fd = central_fd(lambda p: ref_elbow(self.t, p), ELBOW_P, name)
        an = self.res.sensitivity("phi", name)
        scale = np.max(np.abs(fd))
        assert scale > 0
        assert np.max(np.abs(an - fd)) / scale < 1e-6

    def test_parameter_override(self):
        p = dict(ELBOW_P, m=0.55, d=0.095)
        res = self.fs.solve(self.t, param_values={"m": 0.55, "d": 0.095})
        np.testing.assert_allclose(res.state("phi"), ref_elbow(self.t, p), atol=1e-9)
        fd = central_fd(lambda q: ref_elbow(self.t, q), p, "I_c")
        an = res.sensitivity("phi", "I_c")
        assert np.max(np.abs(an - fd)) / np.max(np.abs(fd)) < 1e-6

    def test_initial_condition_override(self):
        res = self.fs.solve(self.t, initial_conditions={"phi": 0.9})
        np.testing.assert_allclose(
            res.state("phi"), ref_elbow(self.t, ELBOW_P, phi0=0.9), atol=1e-9
        )

    def test_identifiable_combinations(self):
        # The motion depends on m, d, I_c only through a = m*d and
        # b = I_c + m*d^2. The chain rule then gives
        # d*dphi/dd = m*dphi/dm + m*d^2*dphi/dI_c.
        S = self.res.sensitivity
        lhs = ELBOW_P["d"] * S("phi", "d")
        rhs = ELBOW_P["m"] * S("phi", "m") + ELBOW_P["m"] * ELBOW_P["d"] ** 2 * S("phi", "I_c")
        np.testing.assert_allclose(lhs, rhs, atol=1e-9)


class TestDoublePendulum:
    def setup_method(self):
        self.c = _compile(DOUBLE)
        self.names = ["m1", "m2", "l1", "l2"]
        self.fs = ForwardSensitivity(self.c, params=self.names)
        self.t = np.linspace(0.0, 2.0, 201)
        self.res = self.fs.solve(self.t)

    def test_trajectory_matches_reference(self):
        a1, a2 = ref_double(self.t, DOUBLE_P)
        np.testing.assert_allclose(self.res.state("theta1"), a1, atol=1e-8)
        np.testing.assert_allclose(self.res.state("theta2"), a2, atol=1e-8)

    @pytest.mark.parametrize("name", ["m1", "m2", "l1", "l2"])
    def test_sensitivity_matches_central_fd(self, name):
        for k, coord in enumerate(["theta1", "theta2"]):
            fd = central_fd(lambda p, k=k: ref_double(self.t, p)[k], DOUBLE_P, name)
            an = self.res.sensitivity(coord, name)
            assert np.max(np.abs(an - fd)) / np.max(np.abs(fd)) < 1e-5


# ---------------------------------------------------------------------------
# Error metric, pooling and ranking
# ---------------------------------------------------------------------------


class TestRMSGradient:
    def setup_method(self):
        self.c = _compile(ELBOW)
        self.names = ["m", "d", "I_c", "beta"]
        self.fs = ForwardSensitivity(self.c, params=self.names)
        self.t = np.linspace(0.0, 2.5, 251)
        true = dict(ELBOW_P, m=0.46, d=0.072, I_c=0.0015, beta=0.18)
        rng = np.random.default_rng(20261116)
        self.meas = ref_elbow(self.t, true) + rng.normal(0.0, 1e-3, self.t.size)

    def _E(self, p, meas=None, phi0=0.6):
        meas = self.meas if meas is None else meas
        r = ref_elbow(self.t, p, phi0=phi0) - meas
        return np.sqrt(np.mean(r**2))

    def test_rms_value(self):
        err = self.fs.solve(self.t).rms_error(self.meas, "phi")
        assert err.n == self.t.size
        assert err.rms == pytest.approx(self._E(ELBOW_P), rel=1e-8)

    @pytest.mark.parametrize("name", ["m", "d", "I_c", "beta"])
    def test_gradient_matches_central_fd(self, name):
        g = self.fs.solve(self.t).rms_error(self.meas, "phi").gradient[name]
        fd = central_fd(lambda p: self._E(p), ELBOW_P, name)
        assert g == pytest.approx(float(fd), rel=1e-5)

    def test_pooled_gradient(self):
        meas2 = ref_elbow(self.t, dict(ELBOW_P, m=0.5), phi0=0.9)
        e1 = self.fs.solve(self.t).rms_error(self.meas, "phi")
        e2 = self.fs.solve(self.t, initial_conditions={"phi": 0.9}).rms_error(meas2, "phi")
        pooled = pool_rms([e1, e2])

        def E_pool(p):
            r1 = ref_elbow(self.t, p) - self.meas
            r2 = ref_elbow(self.t, p, phi0=0.9) - meas2
            return np.sqrt((np.sum(r1**2) + np.sum(r2**2)) / (r1.size + r2.size))

        assert pooled.rms == pytest.approx(E_pool(ELBOW_P), rel=1e-8)
        for name in self.names:
            fd = central_fd(E_pool, ELBOW_P, name)
            assert pooled.gradient[name] == pytest.approx(float(fd), rel=1e-5)

    def test_shape_mismatch(self):
        res = self.fs.solve(self.t)
        with pytest.raises(ValueError, match="t_eval"):
            res.rms_error(self.meas[:-1], "phi")

    def test_zero_error_gradient(self):
        res = self.fs.solve(self.t)
        err = res.rms_error(res.state("phi"), "phi")
        assert err.rms == 0.0
        assert all(v == 0.0 for v in err.gradient.values())


class TestRanking:
    def test_order_and_values(self):
        r = uncertainty_weighted_ranking(
            {"a": -2.0, "b": 0.5, "c": 10.0}, {"a": 1.0, "b": 10.0, "c": 0.1}
        )
        assert [k for k, _ in r] == ["b", "a", "c"]
        assert dict(r) == pytest.approx({"a": 2.0, "b": 5.0, "c": 1.0})

    def test_missing_sigma(self):
        with pytest.raises(ValueError, match="no sigma"):
            uncertainty_weighted_ranking({"a": 1.0, "b": 1.0}, {"a": 1.0})

    def test_negative_sigma(self):
        with pytest.raises(ValueError):
            uncertainty_weighted_ranking({"a": 1.0}, {"a": -1.0})

    def test_pool_rejects_mismatched_parameters(self):
        a = RMSErrorGradient(1.0, 2, {"m": 1.0})
        b = RMSErrorGradient(1.0, 2, {"d": 1.0})
        with pytest.raises(ValueError):
            pool_rms([a, b])
        with pytest.raises(ValueError):
            pool_rms([])


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_uncompiled(self):
        with pytest.raises(ValueError, match="compile_dsl"):
            ForwardSensitivity(PhysicsCompiler(), params=["m"])

    def test_unknown_parameter(self):
        with pytest.raises(ValueError, match="unknown parameters"):
            ForwardSensitivity(_compile(ELBOW), params=["mass"])

    def test_unknown_initial_condition(self):
        with pytest.raises(ValueError, match="unknown state"):
            ForwardSensitivity(_compile(ELBOW), params=["m"], initial_conditions=["theta"])

    def test_bad_time_grid(self):
        fs = ForwardSensitivity(_compile(ELBOW), params=["m"])
        with pytest.raises(ValueError):
            fs.solve([0.0])
        with pytest.raises(ValueError):
            fs.solve([0.0, 1.0, 0.5])

    def test_unknown_override(self):
        fs = ForwardSensitivity(_compile(ELBOW), params=["m"])
        with pytest.raises(ValueError, match="unknown parameters"):
            fs.solve([0.0, 1.0], param_values={"mass": 1.0})

    def test_parameter_absent_from_equations_has_zero_sensitivity(self):
        # g enters; add a parameter that does not appear in the Lagrangian.
        src = ELBOW.replace(r"\parameter{g}", r"\parameter{unused}{3.0}{m}" "\n" r"\parameter{g}")
        fs = ForwardSensitivity(_compile(src), params=["unused", "m"])
        res = fs.solve(np.linspace(0, 1, 11))
        assert np.all(res.sensitivity("phi", "unused") == 0.0)
