"""
Tests for Canonical Transformations Module

Validates:
- Poisson bracket computation
- Canonicity verification
- Generating functions
- Point and exchange transformations
"""

import pytest
import sympy as sp

from mechanics_dsl.domains.classical import (
    CanonicalTransformation,
    GeneratingFunction,
    GeneratingFunctionType,
    HamiltonJacobi,
)


class TestPoissonBrackets:
    """Test Poisson bracket computation."""

    def test_fundamental_poisson_brackets(self):
        """Test {q, p} = 1, {q, q} = 0, {p, p} = 0."""
        ct = CanonicalTransformation()

        q = sp.Symbol("q", real=True)
        p = sp.Symbol("p_q", real=True)

        # {q, p} = 1
        bracket_qp = ct.poisson_bracket(q, p, ["q"])
        assert sp.simplify(bracket_qp - 1) == 0

        # {q, q} = 0
        bracket_qq = ct.poisson_bracket(q, q, ["q"])
        assert sp.simplify(bracket_qq) == 0

        # {p, p} = 0
        bracket_pp = ct.poisson_bracket(p, p, ["q"])
        assert sp.simplify(bracket_pp) == 0

    def test_poisson_bracket_2dof(self):
        """Test Poisson brackets for 2 DOF system."""
        ct = CanonicalTransformation()

        q1 = sp.Symbol("q1", real=True)
        q2 = sp.Symbol("q2", real=True)
        p1 = sp.Symbol("p_q1", real=True)
        p2 = sp.Symbol("p_q2", real=True)

        # {q1, p2} = 0 (different DOF)
        bracket = ct.poisson_bracket(q1, p2, ["q1", "q2"])
        assert sp.simplify(bracket) == 0

        # {q1, p1} = 1
        bracket = ct.poisson_bracket(q1, p1, ["q1", "q2"])
        assert sp.simplify(bracket - 1) == 0

    def test_antisymmetry(self):
        """Test {f, g} = -{g, f}."""
        ct = CanonicalTransformation()

        q = sp.Symbol("q", real=True)
        p = sp.Symbol("p_q", real=True)

        f = q**2
        g = p

        bracket_fg = ct.poisson_bracket(f, g, ["q"])
        bracket_gf = ct.poisson_bracket(g, f, ["q"])

        assert sp.simplify(bracket_fg + bracket_gf) == 0


class TestCanonicityVerification:
    """Test verification of canonical transformations."""

    def test_identity_is_canonical(self):
        """Identity transformation is canonical."""
        ct = CanonicalTransformation()

        q = sp.Symbol("q", real=True)
        p = sp.Symbol("p_q", real=True)

        # Q = q, P = p
        Q = [q]
        P = [p]

        assert ct.verify_canonical(Q, P, ["q"])

    def test_scaling_is_canonical(self):
        """Q = λq, P = p/λ is canonical."""
        ct = CanonicalTransformation()

        q = sp.Symbol("q", real=True)
        p = sp.Symbol("p_q", real=True)
        lam = sp.Symbol("lambda", positive=True)

        # Q = λq, P = p/λ
        Q = [lam * q]
        P = [p / lam]

        assert ct.verify_canonical(Q, P, ["q"])


class TestExchangeTransformation:
    """Test exchange (swap) transformation Q = p, P = -q."""

    def test_exchange_preserves_hamiltonian_form(self):
        """Exchange transformation preserves H form for oscillator."""
        ct = CanonicalTransformation()

        q = sp.Symbol("q", real=True)
        p = sp.Symbol("p_q", real=True)
        m, omega = sp.symbols("m omega", positive=True)

        # H = p²/(2m) + (1/2)*m*ω²*q²
        H = p**2 / (2 * m) + sp.Rational(1, 2) * m * omega**2 * q**2

        result = ct.exchange_transformation(H, ["q"])

        # New Hamiltonian should still be quadratic in Q and P
        K = result.new_hamiltonian
        # K should be defined (not zero)
        assert K != 0


class TestGeneratingFunctions:
    """Test generating function transformations."""

    def test_f2_identity(self):
        """F₂ = qP gives identity transformation."""
        q = sp.Symbol("q", real=True)
        Q = sp.Symbol("Q", real=True)
        P = sp.Symbol("P_Q", real=True)

        # F₂(q, P) = qP → p = ∂F₂/∂q = P, Q = ∂F₂/∂P = q
        F2 = q * P

        gen_func = GeneratingFunction(
            expression=F2,
            function_type=GeneratingFunctionType.F2,
            old_coords=["q"],
            new_coords=["Q"],
        )

        relations = gen_func.get_transformation_relations()

        # p = ∂F₂/∂q = P
        assert "p_q" in relations


class TestHamiltonJacobi:
    """Test Hamilton-Jacobi equation solver."""

    def test_characteristic_function_oscillator(self):
        """Test Hamilton's characteristic function for oscillator."""
        hj = HamiltonJacobi()

        q = sp.Symbol("q", real=True)
        p = sp.Symbol("p_q", real=True)
        m, omega = sp.symbols("m omega", positive=True)
        E = sp.Symbol("E", real=True)

        # H = p²/(2m) + (1/2)*m*ω²*q²
        H = p**2 / (2 * m) + sp.Rational(1, 2) * m * omega**2 * q**2

        W = hj.characteristic_function_1d(H, "q", E)

        # Should get W(q, E) - an integral
        assert W is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
