"""
Conservation Law Verification Tests

Tests that verify fundamental physics conservation laws:
- Energy conservation in conservative systems
- Linear momentum conservation when net external force is zero
- Angular momentum conservation when net external torque is zero
- Noether's theorem verification

These tests ensure MechanicsDSL produces physically correct results.
"""

import math

import numpy as np
import pytest
import sympy as sp

from mechanics_dsl.symbolic import SymbolicEngine
from mechanics_dsl.solver import NumericalSimulator


class TestEnergyConservation:
    """
    Test energy conservation for conservative systems.
    
    For Hamiltonian systems with time-independent H:
    dH/dt = ∂H/∂t = 0 (energy is conserved)
    """
    
    def test_simple_pendulum_energy(self):
        """Pendulum: E = T + V = const."""
        g, L, m = 9.81, 1.0, 1.0
        theta0 = 0.3  # Initial angle
        
        engine = SymbolicEngine()
        theta = engine.get_symbol("theta")
        theta_dot = engine.get_symbol("theta_dot")
        
        # L = T - V = (1/2)mL²θ̇² - mgL(1 - cos θ)
        lagrangian = (sp.Rational(1, 2) * engine.get_symbol("m") * 
                     engine.get_symbol("L")**2 * theta_dot**2 - 
                     engine.get_symbol("m") * engine.get_symbol("g") * 
                     engine.get_symbol("L") * (1 - sp.cos(theta)))
        
        eqs = engine.derive_equations_of_motion(lagrangian, ["theta"])
        accels = engine.solve_for_accelerations(eqs, ["theta"])
        
        sim = NumericalSimulator(engine)
        sim.set_parameters({"g": g, "L": L, "m": m})
        sim.set_initial_conditions({"theta": theta0, "theta_dot": 0.0})
        sim.compile_equations(accels, ["theta"])
        
        omega0 = math.sqrt(g / L)
        result = sim.simulate((0, 20 * 2 * math.pi / omega0), num_points=10000)
        
        theta_arr = result["y"][0, :]
        omega_arr = result["y"][1, :]
        
        # E = T + V = (1/2)mL²ω² + mgL(1 - cos θ)
        T = 0.5 * m * L**2 * omega_arr**2
        V = m * g * L * (1 - np.cos(theta_arr))
        E_total = T + V
        
        E0 = E_total[0]
        relative_drift = np.max(np.abs(E_total - E0)) / E0
        
        # Energy should be conserved to < 0.01% over 20 periods
        assert relative_drift < 0.0001, f"Energy drift {relative_drift*100:.4f}% exceeds 0.01%"
    
    def test_double_pendulum_energy(self):
        """Double pendulum (chaotic but energy conserved)."""
        g = 9.81
        m1, m2, L1, L2 = 1.0, 1.0, 1.0, 1.0
        theta1_0, theta2_0 = 0.5, 0.3
        
        engine = SymbolicEngine()
        theta1 = engine.get_symbol("theta1")
        theta2 = engine.get_symbol("theta2")
        theta1_dot = engine.get_symbol("theta1_dot")
        theta2_dot = engine.get_symbol("theta2_dot")
        
        m1_s, m2_s = engine.get_symbol("m1"), engine.get_symbol("m2")
        L1_s, L2_s = engine.get_symbol("L1"), engine.get_symbol("L2")
        g_s = engine.get_symbol("g")
        
        # Double pendulum Lagrangian (standard form)
        T = (sp.Rational(1, 2) * (m1_s + m2_s) * L1_s**2 * theta1_dot**2 +
             sp.Rational(1, 2) * m2_s * L2_s**2 * theta2_dot**2 +
             m2_s * L1_s * L2_s * theta1_dot * theta2_dot * sp.cos(theta1 - theta2))
        
        V = (-(m1_s + m2_s) * g_s * L1_s * sp.cos(theta1) - 
             m2_s * g_s * L2_s * sp.cos(theta2))
        
        lagrangian = T - V
        
        eqs = engine.derive_equations_of_motion(lagrangian, ["theta1", "theta2"])
        accels = engine.solve_for_accelerations(eqs, ["theta1", "theta2"])
        
        sim = NumericalSimulator(engine)
        sim.set_parameters({"m1": m1, "m2": m2, "L1": L1, "L2": L2, "g": g})
        sim.set_initial_conditions({
            "theta1": theta1_0, "theta1_dot": 0.0,
            "theta2": theta2_0, "theta2_dot": 0.0
        })
        sim.compile_equations(accels, ["theta1", "theta2"])
        
        result = sim.simulate((0, 10), num_points=5000)
        
        th1 = result["y"][0, :]
        w1 = result["y"][1, :]
        th2 = result["y"][2, :]
        w2 = result["y"][3, :]
        
        # Compute total energy
        T_val = (0.5 * (m1 + m2) * L1**2 * w1**2 +
                 0.5 * m2 * L2**2 * w2**2 +
                 m2 * L1 * L2 * w1 * w2 * np.cos(th1 - th2))
        V_val = (-(m1 + m2) * g * L1 * np.cos(th1) - 
                 m2 * g * L2 * np.cos(th2))
        E = T_val + V_val
        
        E0 = E[0]
        max_drift = np.max(np.abs(E - E0)) / abs(E0)
        
        # Even chaotic system should conserve energy to < 1%
        assert max_drift < 0.01, f"Energy drift {max_drift*100:.2f}% exceeds 1%"
    
    def test_coupled_oscillators_energy(self):
        """Coupled harmonic oscillators."""
        k1, k2, kc = 10.0, 10.0, 2.0  # Springs
        m1, m2 = 1.0, 1.0
        
        engine = SymbolicEngine()
        x1, x2 = engine.get_symbol("x1"), engine.get_symbol("x2")
        v1, v2 = engine.get_symbol("x1_dot"), engine.get_symbol("x2_dot")
        
        # L = (1/2)m1ẋ1² + (1/2)m2ẋ2² - (1/2)k1x1² - (1/2)k2x2² - (1/2)kc(x2-x1)²
        T = sp.Rational(1, 2) * engine.get_symbol("m1") * v1**2 + sp.Rational(1, 2) * engine.get_symbol("m2") * v2**2
        V = (sp.Rational(1, 2) * engine.get_symbol("k1") * x1**2 + 
             sp.Rational(1, 2) * engine.get_symbol("k2") * x2**2 +
             sp.Rational(1, 2) * engine.get_symbol("kc") * (x2 - x1)**2)
        
        lagrangian = T - V
        
        eqs = engine.derive_equations_of_motion(lagrangian, ["x1", "x2"])
        accels = engine.solve_for_accelerations(eqs, ["x1", "x2"])
        
        sim = NumericalSimulator(engine)
        sim.set_parameters({"m1": m1, "m2": m2, "k1": k1, "k2": k2, "kc": kc})
        sim.set_initial_conditions({"x1": 1.0, "x1_dot": 0.0, "x2": 0.0, "x2_dot": 0.0})
        sim.compile_equations(accels, ["x1", "x2"])
        
        result = sim.simulate((0, 20), num_points=5000)
        
        x1_arr, v1_arr = result["y"][0, :], result["y"][1, :]
        x2_arr, v2_arr = result["y"][2, :], result["y"][3, :]
        
        T_val = 0.5 * m1 * v1_arr**2 + 0.5 * m2 * v2_arr**2
        V_val = 0.5 * k1 * x1_arr**2 + 0.5 * k2 * x2_arr**2 + 0.5 * kc * (x2_arr - x1_arr)**2
        E = T_val + V_val
        
        drift = np.max(np.abs(E - E[0])) / E[0]
        assert drift < 0.0001, f"Energy drift {drift*100:.4f}% exceeds 0.01%"


class TestMomentumConservation:
    """
    Test momentum conservation when appropriate symmetries exist.
    
    Linear momentum is conserved when there's no net external force.
    Angular momentum is conserved when there's no net external torque.
    """
    
    def test_two_body_momentum_conservation(self):
        """Center of mass momentum conserved in isolated two-body system."""
        m1, m2 = 2.0, 3.0
        
        engine = SymbolicEngine()
        x1 = engine.get_symbol("x1")
        x2 = engine.get_symbol("x2")
        v1 = engine.get_symbol("x1_dot")
        v2 = engine.get_symbol("x2_dot")
        
        # Spring connecting two masses (no external forces)
        k = engine.get_symbol("k")
        m1_s, m2_s = engine.get_symbol("m1"), engine.get_symbol("m2")
        
        lagrangian = (sp.Rational(1, 2) * m1_s * v1**2 + 
                     sp.Rational(1, 2) * m2_s * v2**2 -
                     sp.Rational(1, 2) * k * (x2 - x1)**2)
        
        eqs = engine.derive_equations_of_motion(lagrangian, ["x1", "x2"])
        accels = engine.solve_for_accelerations(eqs, ["x1", "x2"])
        
        sim = NumericalSimulator(engine)
        sim.set_parameters({"m1": m1, "m2": m2, "k": 5.0})
        sim.set_initial_conditions({
            "x1": 0.0, "x1_dot": 1.0,
            "x2": 2.0, "x2_dot": -0.5
        })
        sim.compile_equations(accels, ["x1", "x2"])
        
        result = sim.simulate((0, 10), num_points=2000)
        
        v1_arr = result["y"][1, :]
        v2_arr = result["y"][3, :]
        
        # Total momentum p = m1*v1 + m2*v2 should be constant
        p_total = m1 * v1_arr + m2 * v2_arr
        p0 = p_total[0]
        
        max_deviation = np.max(np.abs(p_total - p0))
        assert max_deviation < 1e-10, f"Momentum deviation {max_deviation} exceeds tolerance"
    
    def test_central_force_angular_momentum(self):
        """Angular momentum conserved in central force problem."""
        # Using 2D polar coordinates for central force
        engine = SymbolicEngine()
        r = engine.get_symbol("r")
        phi = engine.get_symbol("phi")
        r_dot = engine.get_symbol("r_dot")
        phi_dot = engine.get_symbol("phi_dot")
        
        m = engine.get_symbol("m")
        k = engine.get_symbol("k")
        
        # L = (1/2)m(ṙ² + r²φ̇²) - (1/2)kr² (harmonic oscillator in 2D polar)
        lagrangian = (sp.Rational(1, 2) * m * (r_dot**2 + r**2 * phi_dot**2) -
                     sp.Rational(1, 2) * k * r**2)
        
        eqs = engine.derive_equations_of_motion(lagrangian, ["r", "phi"])
        accels = engine.solve_for_accelerations(eqs, ["r", "phi"])
        
        sim = NumericalSimulator(engine)
        sim.set_parameters({"m": 1.0, "k": 4.0})
        sim.set_initial_conditions({
            "r": 1.0, "r_dot": 0.0,
            "phi": 0.0, "phi_dot": 2.0
        })
        sim.compile_equations(accels, ["r", "phi"])
        
        result = sim.simulate((0, 10), num_points=2000)
        
        r_arr = result["y"][0, :]
        phi_dot_arr = result["y"][3, :]
        
        # Angular momentum L = mr²φ̇
        L = 1.0 * r_arr**2 * phi_dot_arr
        L0 = L[0]
        
        max_deviation = np.max(np.abs(L - L0))
        assert max_deviation < 1e-8, f"Angular momentum deviation {max_deviation} exceeds tolerance"


class TestNoetherSymmetries:
    """
    Test Noether's theorem explicitly.
    
    For every continuous symmetry, there's a conserved quantity.
    """
    
    def test_time_translation_energy(self):
        """Time translation invariance => Energy conservation."""
        # Already tested above, this is explicit Noether verification
        engine = SymbolicEngine()
        x = engine.get_symbol("x")
        v = engine.get_symbol("x_dot")
        m, k = engine.get_symbol("m"), engine.get_symbol("k")
        
        # Lagrangian has no explicit time dependence
        L = sp.Rational(1, 2) * m * v**2 - sp.Rational(1, 2) * k * x**2
        
        # Verify ∂L/∂t = 0
        t = engine.time_symbol
        dL_dt = sp.diff(L, t)
        
        assert dL_dt == 0, f"Lagrangian should not depend on time: ∂L/∂t = {dL_dt}"
    
    def test_space_translation_momentum(self):
        """Space translation invariance => Momentum conservation."""
        engine = SymbolicEngine()
        x1 = engine.get_symbol("x1")
        x2 = engine.get_symbol("x2")
        v1 = engine.get_symbol("x1_dot")
        v2 = engine.get_symbol("x2_dot")
        m1, m2 = engine.get_symbol("m1"), engine.get_symbol("m2")
        k = engine.get_symbol("k")
        
        # This Lagrangian depends only on (x2 - x1), not on absolute position
        L = (sp.Rational(1, 2) * m1 * v1**2 + 
             sp.Rational(1, 2) * m2 * v2**2 -
             sp.Rational(1, 2) * k * (x2 - x1)**2)
        
        # Under translation x1 -> x1 + ε, x2 -> x2 + ε, L is unchanged
        # This means total momentum p1 + p2 is conserved
        
        # Verify: ∂L/∂x1 + ∂L/∂x2 = 0 (generator of translations)
        dL_dx1 = sp.diff(L, x1)
        dL_dx2 = sp.diff(L, x2)
        
        total_force = dL_dx1 + dL_dx2
        total_force_simplified = sp.simplify(total_force)
        
        assert total_force_simplified == 0, f"Total external force should be zero: {total_force_simplified}"
    
    def test_rotation_angular_momentum(self):
        """Rotational invariance => Angular momentum conservation."""
        engine = SymbolicEngine()
        r = engine.get_symbol("r")
        theta = engine.get_symbol("theta")
        r_dot = engine.get_symbol("r_dot")
        theta_dot = engine.get_symbol("theta_dot")
        m = engine.get_symbol("m")
        
        # V(r) - potential depends only on r, not θ
        V = engine.get_function("V")
        
        L = sp.Rational(1, 2) * m * (r_dot**2 + r**2 * theta_dot**2) - V(r)
        
        # ∂L/∂θ = 0 means θ is cyclic => p_θ = ∂L/∂θ̇ = mr²θ̇ is conserved
        dL_dtheta = sp.diff(L, theta)
        
        assert dL_dtheta == 0, f"θ should be cyclic (∂L/∂θ = 0): got {dL_dtheta}"
        
        # The conserved quantity is p_θ = ∂L/∂θ̇
        p_theta = sp.diff(L, theta_dot)
        p_theta_simplified = sp.simplify(p_theta)
        
        expected = m * r**2 * theta_dot
        assert sp.simplify(p_theta_simplified - expected) == 0, \
            f"Angular momentum should be mr²θ̇: got {p_theta_simplified}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
