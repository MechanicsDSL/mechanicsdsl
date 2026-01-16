"""
Tests for symplectic and variational integrators
"""

from unittest.mock import Mock

import numpy as np
import pytest


class TestSymplecticIntegrators:
    """Tests for symplectic integrator implementations"""

    def test_integrator_imports(self):
        """Test symplectic integrators can be imported"""
        from mechanics_dsl.solver.symplectic import (
            Leapfrog,
            McLachlan4,
            Ruth3,
            StormerVerlet,
            SymplecticIntegrator,
            Yoshida4,
            get_symplectic_integrator,
        )

        assert StormerVerlet is not None
        assert Leapfrog is not None
        assert Yoshida4 is not None
        assert Ruth3 is not None
        assert McLachlan4 is not None

    def test_verlet_properties(self):
        """Test Verlet integrator properties"""
        from mechanics_dsl.solver.symplectic import StormerVerlet

        verlet = StormerVerlet()
        assert verlet.order == 2
        assert verlet.name == "Störmer-Verlet"

    def test_leapfrog_properties(self):
        """Test Leapfrog integrator properties"""
        from mechanics_dsl.solver.symplectic import Leapfrog

        leapfrog = Leapfrog()
        assert leapfrog.order == 2
        assert leapfrog.name == "Leapfrog"

    def test_yoshida4_properties(self):
        """Test Yoshida4 integrator properties"""
        from mechanics_dsl.solver.symplectic import Yoshida4

        yoshida = Yoshida4()
        assert yoshida.order == 4
        assert yoshida.name == "Yoshida-4"

    def test_ruth3_properties(self):
        """Test Ruth3 integrator properties"""
        from mechanics_dsl.solver.symplectic import Ruth3

        ruth = Ruth3()
        assert ruth.order == 3
        assert ruth.name == "Ruth-3"

    def test_mclachlan4_properties(self):
        """Test McLachlan4 integrator properties"""
        from mechanics_dsl.solver.symplectic import McLachlan4

        mclachlan = McLachlan4()
        assert mclachlan.order == 4
        assert mclachlan.name == "McLachlan-4"

    def test_verlet_step(self):
        """Test Verlet single step"""
        from mechanics_dsl.solver.symplectic import StormerVerlet

        verlet = StormerVerlet()

        def grad_T(p):
            return p / 1.0

        def grad_V(q):
            return 4.0 * q

        q = np.array([1.0])
        p = np.array([0.0])

        q_new, p_new = verlet.step(0, q, p, 0.01, grad_T, grad_V)
        assert q_new.shape == q.shape
        assert p_new.shape == p.shape

    def test_leapfrog_step(self):
        """Test Leapfrog single step"""
        from mechanics_dsl.solver.symplectic import Leapfrog

        leapfrog = Leapfrog()

        def grad_T(p):
            return p

        def grad_V(q):
            return q

        q = np.array([1.0])
        p = np.array([0.5])

        q_new, p_new = leapfrog.step(0, q, p, 0.01, grad_T, grad_V)
        assert np.all(np.isfinite(q_new))
        assert np.all(np.isfinite(p_new))

    def test_yoshida4_step(self):
        """Test Yoshida4 single step"""
        from mechanics_dsl.solver.symplectic import Yoshida4

        yoshida = Yoshida4()

        def grad_T(p):
            return p

        def grad_V(q):
            return q

        q = np.array([1.0])
        p = np.array([0.0])

        q_new, p_new = yoshida.step(0, q, p, 0.01, grad_T, grad_V)
        assert np.all(np.isfinite(q_new))
        assert np.all(np.isfinite(p_new))

    def test_ruth3_step(self):
        """Test Ruth3 single step"""
        from mechanics_dsl.solver.symplectic import Ruth3

        ruth = Ruth3()

        def grad_T(p):
            return p

        def grad_V(q):
            return q

        q = np.array([1.0])
        p = np.array([0.0])

        q_new, p_new = ruth.step(0, q, p, 0.01, grad_T, grad_V)
        assert np.all(np.isfinite(q_new))

    def test_mclachlan4_step(self):
        """Test McLachlan4 single step"""
        from mechanics_dsl.solver.symplectic import McLachlan4

        mclachlan = McLachlan4()

        def grad_T(p):
            return p

        def grad_V(q):
            return q

        q = np.array([1.0])
        p = np.array([0.0])

        q_new, p_new = mclachlan.step(0, q, p, 0.01, grad_T, grad_V)
        assert np.all(np.isfinite(q_new))

    def test_verlet_integrate(self):
        """Test Verlet full integration"""
        from mechanics_dsl.solver.symplectic import StormerVerlet

        verlet = StormerVerlet()

        def grad_T(p):
            return p / 1.0

        def grad_V(q):
            return 4.0 * q

        result = verlet.integrate(
            t_span=(0, 1),
            q0=np.array([1.0]),
            p0=np.array([0.0]),
            h=0.01,
            grad_T=grad_T,
            grad_V=grad_V,
        )

        assert result["success"]
        assert "q" in result
        assert "p" in result
        assert "t" in result

    def test_get_symplectic_integrator(self):
        """Test factory function"""
        from mechanics_dsl.solver.symplectic import get_symplectic_integrator

        verlet = get_symplectic_integrator("verlet")
        assert verlet.order == 2

        yoshida = get_symplectic_integrator("yoshida4")
        assert yoshida.order == 4

    def test_get_unknown_integrator(self):
        """Test factory with unknown integrator"""
        from mechanics_dsl.solver.symplectic import get_symplectic_integrator

        with pytest.raises(ValueError):
            get_symplectic_integrator("nonexistent")

    def test_integrate_with_t_eval(self):
        """Test integration with specific evaluation times"""
        from mechanics_dsl.solver.symplectic import StormerVerlet

        verlet = StormerVerlet()

        def grad_T(p):
            return p

        def grad_V(q):
            return q

        t_eval = np.linspace(0, 1, 11)
        result = verlet.integrate(
            t_span=(0, 1),
            q0=np.array([1.0]),
            p0=np.array([0.0]),
            h=0.01,
            grad_T=grad_T,
            grad_V=grad_V,
            t_eval=t_eval,
        )

        assert len(result["t"]) == len(t_eval)


class TestVariationalIntegrators:
    """Tests for variational integrator implementations"""

    def test_integrator_imports(self):
        """Test variational integrators can be imported"""
        from mechanics_dsl.solver.variational import (
            GalerkinVariational,
            MidpointVariational,
            TrapezoidalVariational,
            VariationalIntegrator,
            get_variational_integrator,
        )

        assert MidpointVariational is not None
        assert TrapezoidalVariational is not None
        assert GalerkinVariational is not None

    def test_midpoint_properties(self):
        """Test Midpoint integrator properties"""
        from mechanics_dsl.solver.variational import MidpointVariational

        midpoint = MidpointVariational()
        assert midpoint.order == 2
        assert midpoint.name == "Midpoint-Variational"

    def test_trapezoidal_properties(self):
        """Test Trapezoidal integrator properties"""
        from mechanics_dsl.solver.variational import TrapezoidalVariational

        trap = TrapezoidalVariational()
        assert trap.order == 2
        assert trap.name == "Trapezoidal-Variational"

    def test_galerkin_properties(self):
        """Test Galerkin integrator properties"""
        from mechanics_dsl.solver.variational import GalerkinVariational

        galerkin = GalerkinVariational(num_points=2)
        assert galerkin.order == 4  # 2*num_points
        assert "Galerkin" in galerkin.name

    def test_galerkin_different_points(self):
        """Test Galerkin with different quadrature points"""
        from mechanics_dsl.solver.variational import GalerkinVariational

        g1 = GalerkinVariational(num_points=1)
        assert g1.order == 2

        g2 = GalerkinVariational(num_points=2)
        assert g2.order == 4

        g3 = GalerkinVariational(num_points=3)
        assert g3.order == 6

    def test_discrete_lagrangian(self):
        """Test discrete Lagrangian computation"""
        from mechanics_dsl.solver.variational import MidpointVariational

        midpoint = MidpointVariational()

        def lagrangian(q, q_dot):
            return 0.5 * np.dot(q_dot, q_dot) - 0.5 * np.dot(q, q)

        q0 = np.array([1.0])
        q1 = np.array([1.1])
        h = 0.1

        Ld = midpoint.discrete_lagrangian(q0, q1, h, lagrangian)
        assert np.isfinite(Ld)

    def test_trapezoidal_discrete_lagrangian(self):
        """Test trapezoidal discrete Lagrangian"""
        from mechanics_dsl.solver.variational import TrapezoidalVariational

        trap = TrapezoidalVariational()

        def lagrangian(q, q_dot):
            return 0.5 * np.dot(q_dot, q_dot) - 0.5 * np.dot(q, q)

        q0 = np.array([1.0])
        q1 = np.array([1.1])
        h = 0.1

        Ld = trap.discrete_lagrangian(q0, q1, h, lagrangian)
        assert np.isfinite(Ld)

    def test_galerkin_discrete_lagrangian(self):
        """Test Galerkin discrete Lagrangian"""
        from mechanics_dsl.solver.variational import GalerkinVariational

        galerkin = GalerkinVariational(num_points=2)

        def lagrangian(q, q_dot):
            return 0.5 * np.dot(q_dot, q_dot) - 0.5 * np.dot(q, q)

        q0 = np.array([1.0])
        q1 = np.array([1.1])
        h = 0.1

        Ld = galerkin.discrete_lagrangian(q0, q1, h, lagrangian)
        assert np.isfinite(Ld)

    def test_midpoint_integrate(self):
        """Test Midpoint full integration"""
        from mechanics_dsl.solver.variational import MidpointVariational

        midpoint = MidpointVariational()

        def lagrangian(q, q_dot):
            return 0.5 * np.dot(q_dot, q_dot) - 0.5 * 4.0 * np.dot(q, q)

        result = midpoint.integrate(
            t_span=(0, 1),
            q0=np.array([1.0]),
            q_dot0=np.array([0.0]),
            h=0.05,
            lagrangian=lagrangian,
            max_steps=100,
        )

        assert result["success"]
        assert "q" in result
        assert "q_dot" in result

    def test_get_variational_integrator(self):
        """Test factory function"""
        from mechanics_dsl.solver.variational import get_variational_integrator

        midpoint = get_variational_integrator("midpoint")
        assert midpoint.order == 2

        trap = get_variational_integrator("trapezoidal")
        assert trap.order == 2

    def test_get_unknown_variational(self):
        """Test factory with unknown integrator"""
        from mechanics_dsl.solver.variational import get_variational_integrator

        with pytest.raises(ValueError):
            get_variational_integrator("nonexistent")

    def test_variational_config(self):
        """Test variational config"""
        from mechanics_dsl.solver.variational import VariationalConfig

        config = VariationalConfig(newton_tol=1e-8, newton_max_iter=100)
        assert config.newton_tol == 1e-8
        assert config.newton_max_iter == 100
