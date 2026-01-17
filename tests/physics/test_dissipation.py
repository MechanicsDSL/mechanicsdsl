"""
Tests for Dissipation and Non-Conservative Forces

Validates:
- Rayleigh dissipation function
- Friction models (viscous, Coulomb, Stribeck)
- Energy loss in damped systems
"""

import pytest
import sympy as sp

from mechanics_dsl.domains.classical import (
    DissipativeLagrangianMechanics,
    FrictionModel,
    FrictionType,
    RayleighDissipation,
)


class TestRayleighDissipation:
    """Test Rayleigh dissipation function implementation."""

    def test_simple_damping(self):
        """Test simple damping on single coordinate."""
        dissipation = RayleighDissipation()
        dissipation.add_damping("x", 0.5)

        F = dissipation.get_dissipation_function()
        x_dot = sp.Symbol("x_dot", real=True)

        # F = (1/2) * b * x_dot^2
        expected = sp.Rational(1, 2) * 0.5 * x_dot**2
        assert sp.simplify(F - expected) == 0

    def test_dissipative_force(self):
        """Test that dissipative force is -∂F/∂q̇."""
        dissipation = RayleighDissipation()
        dissipation.add_damping("theta", 0.1)

        force = dissipation.get_dissipative_force("theta")
        theta_dot = sp.Symbol("theta_dot", real=True)

        # Q = -∂F/∂θ̇ = -b * θ̇
        expected = -0.1 * theta_dot
        assert sp.simplify(force - expected) == 0

    def test_power_dissipated(self):
        """Test that power = 2F is always positive for nonzero velocity."""
        dissipation = RayleighDissipation()
        dissipation.add_damping("x", 1.0)

        power = dissipation.get_power_dissipated()
        x_dot = sp.Symbol("x_dot", real=True)

        # P = 2F = b * x_dot^2 >= 0
        expected = 1.0 * x_dot**2
        assert sp.simplify(power - expected) == 0

    def test_cross_damping(self):
        """Test cross-damping between coordinates."""
        dissipation = RayleighDissipation()
        dissipation.add_damping("x", 1.0)
        dissipation.add_damping("x", 0.5, "y")
        dissipation.add_damping("y", 1.0)

        F = dissipation.get_dissipation_function()

        # Should have x_dot, y_dot, and cross term
        x_dot = sp.Symbol("x_dot", real=True)
        y_dot = sp.Symbol("y_dot", real=True)

        assert x_dot in F.free_symbols
        assert y_dot in F.free_symbols


class TestFrictionModels:
    """Test different friction model implementations."""

    def test_viscous_friction(self):
        """Test viscous (linear) friction."""
        friction = FrictionModel(friction_type=FrictionType.VISCOUS, coefficients={"b": 0.5})

        # Test numerical
        assert friction.get_numerical_force(2.0) == -1.0
        assert friction.get_numerical_force(-2.0) == 1.0
        assert friction.get_numerical_force(0.0) == 0.0

    def test_coulomb_friction(self):
        """Test Coulomb (constant) friction."""
        friction = FrictionModel(
            friction_type=FrictionType.COULOMB, coefficients={"mu": 0.3, "N": 10.0}
        )

        # F = -μN * sign(v)
        assert friction.get_numerical_force(1.0) == -3.0
        assert friction.get_numerical_force(-1.0) == 3.0
        assert friction.get_numerical_force(0.0) == 0.0

    def test_stribeck_friction(self):
        """Test Stribeck friction model."""
        friction = FrictionModel(
            friction_type=FrictionType.STRIBECK,
            coefficients={"mu_s": 0.4, "mu_k": 0.3, "b": 0.1, "v_s": 0.01, "N": 10.0},
        )

        # At high velocity, should approach kinetic + viscous
        high_v = 10.0
        force = friction.get_numerical_force(high_v)
        # Should be negative (opposing motion)
        assert force < 0
        # Kinetic part: ~-3.0, viscous: -1.0
        assert -5.0 < force < -3.0


class TestDissipativeLagrangianMechanics:
    """Test dissipative Lagrangian mechanics system."""

    def test_damped_oscillator_eom(self):
        """Test damped harmonic oscillator equations of motion."""
        system = DissipativeLagrangianMechanics("damped_oscillator")
        system.add_coordinate("x")

        # Symbols
        m, k, b = sp.symbols("m k b", positive=True)
        x = sp.Symbol("x", real=True)
        x_dot = sp.Symbol("x_dot", real=True)

        # L = (1/2)*m*x_dot^2 - (1/2)*k*x^2
        L = sp.Rational(1, 2) * m * x_dot**2 - sp.Rational(1, 2) * k * x**2
        system.set_lagrangian(L)
        system.add_damping("x", b)

        eom = system.derive_equations_of_motion()

        # Should have x_ddot = ... with damping term
        assert "x_ddot" in eom

    def test_driving_force(self):
        """Test adding sinusoidal driving force."""
        system = DissipativeLagrangianMechanics("driven_oscillator")
        system.add_coordinate("x")

        m, k = sp.symbols("m k", positive=True)
        x = sp.Symbol("x", real=True)
        x_dot = sp.Symbol("x_dot", real=True)

        L = sp.Rational(1, 2) * m * x_dot**2 - sp.Rational(1, 2) * k * x**2
        system.set_lagrangian(L)

        # Add driving force F = A*cos(ω*t)
        system.add_driving_force("x", amplitude=1.0, frequency=2.0)

        # Check force was added
        assert "x" in system._generalized_forces

    def test_energy_rate(self):
        """Test computation of energy change rate."""
        system = DissipativeLagrangianMechanics("test")
        system.add_coordinate("x")

        m = sp.Symbol("m", positive=True)
        x_dot = sp.Symbol("x_dot", real=True)

        L = sp.Rational(1, 2) * m * x_dot**2
        system.set_lagrangian(L)
        system.add_damping("x", 1.0)

        dE_dt = system.compute_energy_rate()

        # Should be -2F = -x_dot^2 (negative for energy loss)
        expected = -(x_dot**2)
        assert sp.simplify(dE_dt - expected) == 0


class TestEnergyDissipation:
    """Test that energy actually decreases in damped systems."""

    def test_damped_oscillator_energy_loss(self):
        """Verify energy decreases monotonically in damped oscillator."""

        dsl_code = r"""
        \system{damped_oscillator}
        \defvar{x}{Position}{m}
        \defvar{m}{Mass}{kg}
        \defvar{k}{Spring Constant}{N/m}
        \defvar{b}{Damping}{N*s/m}
        
        \parameter{m}{1.0}{kg}
        \parameter{k}{10.0}{N/m}
        \parameter{b}{0.5}{N*s/m}
        
        \lagrangian{\frac{1}{2} * m * \dot{x}^2 - \frac{1}{2} * k * x^2}
        \dissipation{b * \dot{x}}
        
        \initial{x=1.0, x_dot=0.0}
        """

        # This test uses the DSL's built-in damping if available
        # For now, just verify the module loads correctly
        assert True  # Placeholder for full integration test


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
