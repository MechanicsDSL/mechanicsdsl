"""
Reference Solution Validation Tests

These tests validate MechanicsDSL against known analytical solutions
to ensure physics correctness. This is critical for "best in class" status.

Systems with exact solutions:
- Simple harmonic oscillator
- Simple pendulum (small angle)
- Kepler orbits
- Free fall
- Projectile motion
"""

import math

import numpy as np
import pytest
import sympy as sp

from mechanics_dsl.solver import NumericalSimulator
from mechanics_dsl.symbolic import SymbolicEngine


class TestHarmonicOscillator:
    """
    Simple harmonic oscillator: exact solution known.

    x(t) = A cos(ωt + φ)
    v(t) = -Aω sin(ωt + φ)

    where ω = √(k/m)
    """

    def test_oscillator_position_accuracy(self):
        """Test position matches analytical solution."""
        # Parameters
        k = 10.0  # Spring constant
        m = 1.0  # Mass
        x0 = 1.0  # Initial displacement
        v0 = 0.0  # Initial velocity

        # Analytical solution
        omega = math.sqrt(k / m)
        A = x0  # Amplitude (starting from rest)
        phi = 0  # Phase (starting at max displacement)

        # Create MechanicsDSL system
        engine = SymbolicEngine()
        x = engine.get_symbol("x")
        x_dot = engine.get_symbol("x_dot")
        m_sym = engine.get_symbol("m")
        k_sym = engine.get_symbol("k")

        # Lagrangian: L = T - V = (1/2)mv² - (1/2)kx²
        T = sp.Rational(1, 2) * m_sym * x_dot**2
        V = sp.Rational(1, 2) * k_sym * x**2
        L = T - V

        # Derive equations
        eqs = engine.derive_equations_of_motion(L, ["x"])
        accels = engine.solve_for_accelerations(eqs, ["x"])

        # Simulate
        sim = NumericalSimulator(engine)
        sim.set_parameters({"k": k, "m": m})
        sim.set_initial_conditions({"x": x0, "x_dot": v0})
        sim.compile_equations(accels, ["x"])

        result = sim.simulate((0, 2 * math.pi / omega), num_points=1000)

        assert result["success"], f"Simulation failed: {result.get('message')}"

        # Compare with analytical
        t = result["t"]
        x_numerical = result["y"][0, :]
        x_analytical = A * np.cos(omega * t + phi)

        # Relative error should be < 1%
        max_error = np.max(np.abs(x_numerical - x_analytical)) / A
        assert max_error < 0.01, f"Max position error {max_error:.4f} exceeds 1%"

    def test_oscillator_energy_conservation(self):
        """Test energy is conserved."""
        k, m = 10.0, 1.0
        x0, v0 = 1.0, 0.0
        omega = math.sqrt(k / m)

        engine = SymbolicEngine()
        x_dot = engine.get_symbol("x_dot")
        m_sym, k_sym = engine.get_symbol("m"), engine.get_symbol("k")

        L = (
            sp.Rational(1, 2) * m_sym * x_dot**2
            - sp.Rational(1, 2) * k_sym * engine.get_symbol("x") ** 2
        )
        eqs = engine.derive_equations_of_motion(L, ["x"])
        accels = engine.solve_for_accelerations(eqs, ["x"])

        sim = NumericalSimulator(engine)
        sim.set_parameters({"k": k, "m": m})
        sim.set_initial_conditions({"x": x0, "x_dot": v0})
        sim.compile_equations(accels, ["x"])

        result = sim.simulate((0, 10 * 2 * math.pi / omega), num_points=5000)

        # Compute energy at each timestep
        x = result["y"][0, :]
        v = result["y"][1, :]
        E = 0.5 * m * v**2 + 0.5 * k * x**2

        E0 = E[0]
        max_drift = np.max(np.abs(E - E0)) / E0

        # Energy drift should be < 0.1% over 10 periods
        assert max_drift < 0.001, f"Energy drift {max_drift:.6f} exceeds 0.1%"

    def test_oscillator_period(self):
        """Test period matches theory T = 2π/ω."""
        k, m = 4.0, 1.0  # ω = 2, T = π
        x0, v0 = 1.0, 0.0
        T_theory = 2 * math.pi / math.sqrt(k / m)

        engine = SymbolicEngine()
        x_dot = engine.get_symbol("x_dot")

        L = (
            sp.Rational(1, 2) * engine.get_symbol("m") * x_dot**2
            - sp.Rational(1, 2) * engine.get_symbol("k") * engine.get_symbol("x") ** 2
        )
        eqs = engine.derive_equations_of_motion(L, ["x"])
        accels = engine.solve_for_accelerations(eqs, ["x"])

        sim = NumericalSimulator(engine)
        sim.set_parameters({"k": k, "m": m})
        sim.set_initial_conditions({"x": x0, "x_dot": v0})
        sim.compile_equations(accels, ["x"])

        result = sim.simulate((0, 3 * T_theory), num_points=3000)

        x = result["y"][0, :]
        t = result["t"]

        # Find zero crossings (going negative to positive)
        crossings = []
        for i in range(1, len(x)):
            if x[i - 1] < 0 and x[i] >= 0:
                # Linear interpolation for more precision
                t_cross = t[i - 1] + (0 - x[i - 1]) / (x[i] - x[i - 1]) * (t[i] - t[i - 1])
                crossings.append(t_cross)

        if len(crossings) >= 2:
            T_measured = crossings[1] - crossings[0]
            period_error = abs(T_measured - T_theory) / T_theory
            assert period_error < 0.01, f"Period error {period_error:.4f} exceeds 1%"


class TestSimplePendulum:
    """
    Simple pendulum in small angle approximation.

    θ(t) = θ₀ cos(ωt) for small θ₀
    where ω = √(g/L)
    """

    def test_small_angle_oscillation(self):
        """Test small angle pendulum matches linearized solution."""
        g = 9.81
        L = 1.0
        theta0 = 0.1  # Small angle in radians
        omega = math.sqrt(g / L)

        engine = SymbolicEngine()
        theta = engine.get_symbol("theta")
        theta_dot = engine.get_symbol("theta_dot")

        # Full nonlinear Lagrangian
        L_sym = engine.get_symbol("L")
        g_sym = engine.get_symbol("g")
        m_sym = engine.get_symbol("m")

        # L = (1/2)mL²θ̇² - mgL(1 - cos(θ))
        kinetic = sp.Rational(1, 2) * m_sym * L_sym**2 * theta_dot**2
        potential = m_sym * g_sym * L_sym * (1 - sp.cos(theta))
        lagrangian = kinetic - potential

        eqs = engine.derive_equations_of_motion(lagrangian, ["theta"])
        accels = engine.solve_for_accelerations(eqs, ["theta"])

        sim = NumericalSimulator(engine)
        sim.set_parameters({"g": g, "L": L, "m": 1.0})
        sim.set_initial_conditions({"theta": theta0, "theta_dot": 0.0})
        sim.compile_equations(accels, ["theta"])

        T = 2 * math.pi / omega
        result = sim.simulate((0, 2 * T), num_points=1000)

        t = result["t"]
        theta_num = result["y"][0, :]
        theta_analytical = theta0 * np.cos(omega * t)

        # For small angles, error should be < 1%
        max_error = np.max(np.abs(theta_num - theta_analytical)) / theta0
        assert max_error < 0.01, f"Small angle error {max_error:.4f} exceeds 1%"

    def test_pendulum_energy_conservation(self):
        """Test energy conservation for nonlinear pendulum."""
        g, L, m = 9.81, 1.0, 1.0
        theta0 = 0.5  # Moderate angle

        engine = SymbolicEngine()
        theta = engine.get_symbol("theta")
        theta_dot = engine.get_symbol("theta_dot")

        lagrangian = sp.Rational(1, 2) * engine.get_symbol("m") * engine.get_symbol(
            "L"
        ) ** 2 * theta_dot**2 - engine.get_symbol("m") * engine.get_symbol(
            "g"
        ) * engine.get_symbol(
            "L"
        ) * (
            1 - sp.cos(theta)
        )

        eqs = engine.derive_equations_of_motion(lagrangian, ["theta"])
        accels = engine.solve_for_accelerations(eqs, ["theta"])

        sim = NumericalSimulator(engine)
        sim.set_parameters({"g": g, "L": L, "m": m})
        sim.set_initial_conditions({"theta": theta0, "theta_dot": 0.0})
        sim.compile_equations(accels, ["theta"])

        omega0 = math.sqrt(g / L)
        result = sim.simulate((0, 10 * 2 * math.pi / omega0), num_points=5000)

        theta_vals = result["y"][0, :]
        omega_vals = result["y"][1, :]

        # Total energy: E = (1/2)mL²ω² + mgL(1 - cos θ)
        E = 0.5 * m * L**2 * omega_vals**2 + m * g * L * (1 - np.cos(theta_vals))

        E0 = E[0]
        max_drift = np.max(np.abs(E - E0)) / E0

        assert max_drift < 0.001, f"Energy drift {max_drift:.6f} exceeds 0.1%"


class TestProjectileMotion:
    """
    Projectile motion in uniform gravity.

    x(t) = x₀ + v₀ₓt
    y(t) = y₀ + v₀ᵧt - (1/2)gt²
    """

    def test_projectile_trajectory(self):
        """Test projectile follows parabolic path."""
        g = 9.81
        v0 = 20.0  # Initial speed
        angle = math.pi / 4  # 45 degrees

        v0x = v0 * math.cos(angle)
        v0y = v0 * math.sin(angle)

        # Time of flight
        t_flight = 2 * v0y / g

        engine = SymbolicEngine()
        x = engine.get_symbol("x")  # noqa: F841
        y = engine.get_symbol("y")
        x_dot = engine.get_symbol("x_dot")
        y_dot = engine.get_symbol("y_dot")

        # Lagrangian: L = (1/2)m(ẋ² + ẏ²) - mgy
        m_sym = engine.get_symbol("m")
        g_sym = engine.get_symbol("g")

        lagrangian = sp.Rational(1, 2) * m_sym * (x_dot**2 + y_dot**2) - m_sym * g_sym * y

        eqs = engine.derive_equations_of_motion(lagrangian, ["x", "y"])
        accels = engine.solve_for_accelerations(eqs, ["x", "y"])

        sim = NumericalSimulator(engine)
        sim.set_parameters({"m": 1.0, "g": g})
        sim.set_initial_conditions({"x": 0.0, "x_dot": v0x, "y": 0.0, "y_dot": v0y})
        sim.compile_equations(accels, ["x", "y"])

        result = sim.simulate((0, t_flight), num_points=500)

        t = result["t"]
        x_num = result["y"][0, :]
        y_num = result["y"][2, :]

        x_analytical = v0x * t
        y_analytical = v0y * t - 0.5 * g * t**2

        max_x_error = np.max(np.abs(x_num - x_analytical))
        max_y_error = np.max(np.abs(y_num - y_analytical))

        # Errors should be < 1 cm for accurate numerical integration
        assert max_x_error < 0.01, f"X error {max_x_error:.4f}m exceeds 1cm"
        assert max_y_error < 0.01, f"Y error {max_y_error:.4f}m exceeds 1cm"

    def test_range_formula(self):
        """Test range matches R = v₀²sin(2θ)/g."""
        g = 9.81
        v0 = 30.0
        angle = math.pi / 6  # 30 degrees

        R_theory = v0**2 * math.sin(2 * angle) / g

        engine = SymbolicEngine()
        x_dot = engine.get_symbol("x_dot")
        y_dot = engine.get_symbol("y_dot")

        lagrangian = sp.Rational(1, 2) * engine.get_symbol("m") * (
            x_dot**2 + y_dot**2
        ) - engine.get_symbol("m") * engine.get_symbol("g") * engine.get_symbol("y")

        eqs = engine.derive_equations_of_motion(lagrangian, ["x", "y"])
        accels = engine.solve_for_accelerations(eqs, ["x", "y"])

        sim = NumericalSimulator(engine)
        sim.set_parameters({"m": 1.0, "g": g})
        sim.set_initial_conditions(
            {"x": 0.0, "x_dot": v0 * math.cos(angle), "y": 0.0, "y_dot": v0 * math.sin(angle)}
        )
        sim.compile_equations(accels, ["x", "y"])

        t_flight = 2 * v0 * math.sin(angle) / g
        result = sim.simulate((0, t_flight * 1.1), num_points=500)

        x = result["y"][0, :]
        y = result["y"][2, :]

        # Find where y crosses zero (landing)
        for i in range(1, len(y)):
            if y[i - 1] > 0 and y[i] <= 0:
                # Interpolate
                frac = y[i - 1] / (y[i - 1] - y[i])
                R_measured = x[i - 1] + frac * (x[i] - x[i - 1])
                break
        else:
            pytest.fail("Projectile never landed")

        range_error = abs(R_measured - R_theory) / R_theory
        assert range_error < 0.01, f"Range error {range_error:.4f} exceeds 1%"


class TestKeplerOrbit:
    """
    Kepler orbit: test conservation laws.

    For elliptical orbits:
    - Energy E = -GMm/(2a) is constant
    - Angular momentum L = m√(GMa(1-e²)) is constant
    """

    def test_circular_orbit_angular_momentum(self):
        """Test angular momentum conservation in circular orbit."""
        # Circular orbit: r = a (constant), v = √(GM/a)
        G = 1.0  # Normalized
        M = 1.0
        a = 1.0  # Orbital radius

        v_circular = math.sqrt(G * M / a)

        engine = SymbolicEngine()
        r = engine.get_symbol("r")
        phi = engine.get_symbol("phi")  # noqa: F841
        r_dot = engine.get_symbol("r_dot")
        phi_dot = engine.get_symbol("phi_dot")

        # Lagrangian in polar coordinates
        m_sym = engine.get_symbol("m")
        GM_sym = engine.get_symbol("GM")

        # L = (1/2)m(ṙ² + r²φ̇²) + GMm/r
        lagrangian = sp.Rational(1, 2) * m_sym * (r_dot**2 + r**2 * phi_dot**2) + GM_sym * m_sym / r

        eqs = engine.derive_equations_of_motion(lagrangian, ["r", "phi"])
        accels = engine.solve_for_accelerations(eqs, ["r", "phi"])

        sim = NumericalSimulator(engine)
        sim.set_parameters({"m": 1.0, "GM": G * M})
        sim.set_initial_conditions({"r": a, "r_dot": 0.0, "phi": 0.0, "phi_dot": v_circular / a})
        sim.compile_equations(accels, ["r", "phi"])

        T_orbit = 2 * math.pi * a / v_circular
        result = sim.simulate((0, 3 * T_orbit), num_points=2000)

        r_vals = result["y"][0, :]
        phi_dot_vals = result["y"][3, :]

        # Angular momentum L = mr²φ̇
        L = r_vals**2 * phi_dot_vals
        L0 = L[0]

        max_L_drift = np.max(np.abs(L - L0)) / L0
        assert max_L_drift < 0.001, f"Angular momentum drift {max_L_drift:.6f} exceeds 0.1%"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
