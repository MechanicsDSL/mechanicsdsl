"""
Tests for Symmetry and Noether's Theorem Module

Validates:
- Cyclic coordinate detection
- Conservation law derivation
- Energy conservation for time-independent systems
"""

import numpy as np
import pytest
import sympy as sp

from mechanics_dsl.domains.classical import (
    ConservedQuantity,
    NoetherAnalyzer,
    SymmetryType,
    detect_cyclic_coordinates,
    get_conserved_quantities,
)


class TestCyclicCoordinates:
    """Test cyclic coordinate detection."""

    def test_free_particle_momentum(self):
        """Free particle has all coordinates cyclic."""
        m = sp.Symbol("m", positive=True)
        x = sp.Symbol("x", real=True)
        y = sp.Symbol("y", real=True)
        x_dot = sp.Symbol("x_dot", real=True)
        y_dot = sp.Symbol("y_dot", real=True)

        # L = (1/2)*m*(x_dot^2 + y_dot^2)
        # x and y don't appear → both cyclic
        L = sp.Rational(1, 2) * m * (x_dot**2 + y_dot**2)

        cyclic = detect_cyclic_coordinates(L, ["x", "y"])

        assert "x" in cyclic
        assert "y" in cyclic

    def test_harmonic_oscillator_not_cyclic(self):
        """Harmonic oscillator x is NOT cyclic (appears in V)."""
        m, k = sp.symbols("m k", positive=True)
        x = sp.Symbol("x", real=True)
        x_dot = sp.Symbol("x_dot", real=True)

        L = sp.Rational(1, 2) * m * x_dot**2 - sp.Rational(1, 2) * k * x**2

        cyclic = detect_cyclic_coordinates(L, ["x"])

        assert "x" not in cyclic

    def test_central_force_angle_cyclic(self):
        """In central force problem, φ is cyclic."""
        # Create a fresh analyzer to get properly cached symbols
        analyzer = NoetherAnalyzer()

        # Get symbols from the analyzer (this ensures consistency)
        m = sp.Symbol("m", positive=True)
        r = analyzer.get_symbol("r")
        phi = analyzer.get_symbol("phi")
        r_dot = analyzer.get_symbol("r_dot")
        phi_dot = analyzer.get_symbol("phi_dot")
        k = sp.Symbol("k", positive=True)

        # L = (1/2)*m*(r_dot^2 + r^2*phi_dot^2) - k/r
        # φ doesn't appear in L → cyclic, r appears in V → not cyclic
        L = sp.Rational(1, 2) * m * (r_dot**2 + r**2 * phi_dot**2) - k / r

        cyclic = analyzer.detect_cyclic_coordinates(L, ["r", "phi"])

        assert "phi" in cyclic
        # Note: r is NOT cyclic because ∂L/∂r = m*r*phi_dot^2 + k/r^2 ≠ 0
        assert "r" not in cyclic


class TestNoetherConservation:
    """Test Noether's theorem conservation law derivation."""

    def test_energy_conservation_time_independent(self):
        """Time-independent Lagrangian → energy conserved."""
        m, k = sp.symbols("m k", positive=True)
        x = sp.Symbol("x", real=True)
        x_dot = sp.Symbol("x_dot", real=True)

        L = sp.Rational(1, 2) * m * x_dot**2 - sp.Rational(1, 2) * k * x**2

        quantities = get_conserved_quantities(L, ["x"])

        # Should have energy
        assert "energy" in quantities

    def test_cyclic_momentum_conserved(self):
        """Cyclic coordinate → conjugate momentum conserved."""
        m = sp.Symbol("m", positive=True)
        phi = sp.Symbol("phi", real=True)
        phi_dot = sp.Symbol("phi_dot", real=True)

        # L = (1/2)*m*phi_dot^2 (free rotation)
        L = sp.Rational(1, 2) * m * phi_dot**2

        quantities = get_conserved_quantities(L, ["phi"])

        # p_phi = ∂L/∂φ̇ = m*φ̇ should be conserved
        assert "p_phi" in quantities

        p_phi = quantities["p_phi"]
        expected = m * phi_dot
        assert sp.simplify(p_phi - expected) == 0

    def test_angular_momentum_central_force(self):
        """Central force problem → angular momentum conserved."""
        m = sp.Symbol("m", positive=True)
        r = sp.Symbol("r", real=True, positive=True)
        phi = sp.Symbol("phi", real=True)
        r_dot = sp.Symbol("r_dot", real=True)
        phi_dot = sp.Symbol("phi_dot", real=True)
        k = sp.Symbol("k", positive=True)

        # L = (1/2)*m*(r_dot^2 + r^2*phi_dot^2) + k/r
        L = sp.Rational(1, 2) * m * (r_dot**2 + r**2 * phi_dot**2) + k / r

        quantities = get_conserved_quantities(L, ["r", "phi"])

        # L_z = m*r^2*phi_dot should be conserved
        assert "p_phi" in quantities

        p_phi = quantities["p_phi"]
        expected = m * r**2 * phi_dot
        assert sp.simplify(p_phi - expected) == 0


class TestSymmetryAnalyzer:
    """Test full symmetry analysis."""

    def test_find_all_symmetries(self):
        """Test finding multiple symmetries."""
        m = sp.Symbol("m", positive=True)
        x = sp.Symbol("x", real=True)
        y = sp.Symbol("y", real=True)
        x_dot = sp.Symbol("x_dot", real=True)
        y_dot = sp.Symbol("y_dot", real=True)

        # Free particle in 2D: time, x, y all give conservation laws
        L = sp.Rational(1, 2) * m * (x_dot**2 + y_dot**2)

        analyzer = NoetherAnalyzer()
        symmetries = analyzer.find_symmetries(L, ["x", "y"])

        # Should find: time translation (energy), x cyclic (p_x), y cyclic (p_y)
        types = [s.symmetry_type for s in symmetries]

        assert SymmetryType.TIME_TRANSLATION in types
        assert SymmetryType.CYCLIC in types

    def test_energy_expression(self):
        """Test energy computation from Lagrangian."""
        m, k = sp.symbols("m k", positive=True)
        x = sp.Symbol("x", real=True)
        x_dot = sp.Symbol("x_dot", real=True)

        L = sp.Rational(1, 2) * m * x_dot**2 - sp.Rational(1, 2) * k * x**2

        analyzer = NoetherAnalyzer()
        E = analyzer.compute_energy(L, ["x"])

        # E = T + V = (1/2)*m*x_dot^2 + (1/2)*k*x^2
        T = sp.Rational(1, 2) * m * x_dot**2
        V = sp.Rational(1, 2) * k * x**2
        expected = T + V

        assert sp.simplify(E - expected) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
