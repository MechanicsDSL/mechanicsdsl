"""
Tests for Normal Modes and Small Oscillations Module

Validates:
- Mass matrix extraction
- Stiffness matrix extraction
- Normal mode computation
- Modal decomposition
"""

import numpy as np
import pytest
import sympy as sp

from mechanics_dsl.domains.classical import (
    CoupledOscillatorSystem,
    ModalAnalysisResult,
    NormalMode,
    NormalModeAnalyzer,
    compute_normal_modes,
    extract_mass_matrix,
    extract_stiffness_matrix,
)


class TestMatrixExtraction:
    """Test mass and stiffness matrix extraction."""

    def test_single_oscillator_mass(self):
        """Test mass matrix for single oscillator."""
        m = sp.Symbol("m", positive=True)
        x = sp.Symbol("x", real=True)
        x_dot = sp.Symbol("x_dot", real=True)
        k = sp.Symbol("k", positive=True)

        L = sp.Rational(1, 2) * m * x_dot**2 - sp.Rational(1, 2) * k * x**2

        M = extract_mass_matrix(L, ["x"])

        assert M.shape == (1, 1)
        assert M[0, 0] == m

    def test_single_oscillator_stiffness(self):
        """Test stiffness matrix for single oscillator."""
        m = sp.Symbol("m", positive=True)
        x = sp.Symbol("x", real=True)
        x_dot = sp.Symbol("x_dot", real=True)
        k = sp.Symbol("k", positive=True)

        L = sp.Rational(1, 2) * m * x_dot**2 - sp.Rational(1, 2) * k * x**2

        K = extract_stiffness_matrix(L, ["x"])

        assert K.shape == (1, 1)
        assert K[0, 0] == k

    def test_coupled_oscillators_coupling(self):
        """Test stiffness matrix with coupling term."""
        m = sp.Symbol("m", positive=True)
        k1, k2, kc = sp.symbols("k1 k2 kc", positive=True)
        x1 = sp.Symbol("x1", real=True)
        x2 = sp.Symbol("x2", real=True)
        x1_dot = sp.Symbol("x1_dot", real=True)
        x2_dot = sp.Symbol("x2_dot", real=True)

        # L = T - V
        T = sp.Rational(1, 2) * m * (x1_dot**2 + x2_dot**2)
        V = (
            sp.Rational(1, 2) * k1 * x1**2
            + sp.Rational(1, 2) * k2 * x2**2
            + sp.Rational(1, 2) * kc * (x1 - x2) ** 2
        )
        L = T - V

        K = extract_stiffness_matrix(L, ["x1", "x2"])

        # K[0,1] should be -kc (coupling)
        assert K[0, 1] == -kc
        # K[0,0] should be k1 + kc
        assert K[0, 0] == k1 + kc


class TestNormalModes:
    """Test normal mode computation."""

    def test_single_oscillator_frequency(self):
        """Test ω = √(k/m) for single oscillator."""
        m_val, k_val = 1.0, 4.0  # ω = 2

        M = np.array([[m_val]])
        K = np.array([[k_val]])

        analyzer = NormalModeAnalyzer()
        modes = analyzer.compute_normal_modes(M, K)

        assert len(modes) == 1
        assert abs(modes[0].frequency - 2.0) < 1e-10

    def test_coupled_oscillators_two_modes(self):
        """Test two coupled oscillators have two distinct modes."""
        # Two identical masses, three identical springs
        m = 1.0
        k = 10.0

        M = np.array([[m, 0], [0, m]])
        K = np.array([[2 * k, -k], [-k, 2 * k]])

        analyzer = NormalModeAnalyzer()
        modes = analyzer.compute_normal_modes(M, K)

        assert len(modes) == 2

        # Frequencies: ω₁ = √(k/m), ω₂ = √(3k/m)
        freqs = sorted([m.frequency for m in modes])
        expected = sorted([np.sqrt(k / m), np.sqrt(3 * k / m)])

        np.testing.assert_array_almost_equal(freqs, expected, decimal=5)

    def test_mode_shapes_orthogonal(self):
        """Test that mode shapes are M-orthogonal."""
        m = 1.0
        k = 10.0

        M = np.array([[m, 0], [0, m]])
        K = np.array([[2 * k, -k], [-k, 2 * k]])

        analyzer = NormalModeAnalyzer()
        modes = analyzer.compute_normal_modes(M, K)

        phi1 = modes[0].mode_shape
        phi2 = modes[1].mode_shape

        # Should be M-orthogonal: φ₁ᵀ M φ₂ = 0
        orthogonality = phi1 @ M @ phi2
        assert abs(orthogonality) < 1e-10


class TestModalAnalysis:
    """Test full modal analysis from Lagrangian."""

    def test_full_analysis(self):
        """Test complete modal analysis with numerical values."""
        x = sp.Symbol("x", real=True)
        x_dot = sp.Symbol("x_dot", real=True)

        # Use numerical values directly in Lagrangian
        m_val, k_val = 1.0, 4.0
        L = sp.Rational(1, 2) * m_val * x_dot**2 - sp.Rational(1, 2) * k_val * x**2

        # No parameters needed when values are in Lagrangian
        result = compute_normal_modes(L, ["x"])

        assert isinstance(result, ModalAnalysisResult)
        assert len(result.modes) == 1
        # Check frequency is approximately sqrt(k/m) = 2.0
        assert abs(result.modes[0].frequency - 2.0) < 0.01

    def test_modal_analysis_result_properties(self):
        """Test ModalAnalysisResult helper methods."""
        M = np.array([[1.0]])
        K = np.array([[4.0]])

        analyzer = NormalModeAnalyzer()
        modes = analyzer.compute_normal_modes(M, K)

        result = ModalAnalysisResult(
            modes=modes,
            mass_matrix=M,
            stiffness_matrix=K,
            modal_matrix=np.array([[1.0]]),
            coordinates=["x"],
        )

        freqs = result.get_frequencies()
        periods = result.get_periods()

        assert len(freqs) == 1
        assert abs(freqs[0] - 2.0) < 1e-10
        assert abs(periods[0] - np.pi) < 1e-10


class TestCoupledOscillatorSystem:
    """Test the coupled oscillator system builder."""

    def test_chain_of_masses(self):
        """Test building a chain of coupled masses."""
        system = CoupledOscillatorSystem()

        system.add_mass("m1", 1.0)
        system.add_mass("m2", 1.0)
        system.add_spring("wall", "m1", k=10.0)
        system.add_spring("m1", "m2", k=5.0)
        system.add_spring("m2", "wall", k=10.0)

        L = system.build_lagrangian()

        # Should have kinetic and potential terms
        m1_dot = sp.Symbol("m1_dot", real=True)
        m2_dot = sp.Symbol("m2_dot", real=True)

        assert m1_dot in L.free_symbols
        assert m2_dot in L.free_symbols

    def test_analyze_chain(self):
        """Test modal analysis of spring chain."""
        system = CoupledOscillatorSystem()

        system.add_mass("m1", 1.0)
        system.add_mass("m2", 1.0)
        system.add_spring("wall", "m1", k=10.0)
        system.add_spring("m1", "m2", k=5.0)
        system.add_spring("m2", "wall", k=10.0)

        result = system.analyze()

        assert len(result.modes) == 2
        assert result.modes[0].frequency < result.modes[1].frequency


class TestModalDecomposition:
    """Test modal decomposition for forced response."""

    def test_resonance_detection(self):
        """Test that resonance gives large response."""
        M = np.array([[1.0]])
        K = np.array([[4.0]])  # ω = 2
        F = np.array([1.0])

        analyzer = NormalModeAnalyzer()

        # Off-resonance (ω_drive = 1)
        q_off = analyzer.modal_decomposition(M, K, F, omega_drive=1.0)

        # Near-resonance (ω_drive = 1.9)
        q_near = analyzer.modal_decomposition(M, K, F, omega_drive=1.9)

        # Near-resonance should have larger response
        assert abs(q_near[0]) > abs(q_off[0])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
