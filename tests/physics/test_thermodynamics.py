"""
Comprehensive tests for Thermodynamics domain.
"""
import pytest
import numpy as np


class TestCarnotEngine:
    """Tests for Carnot engine."""
    
    def test_carnot_efficiency_formula(self):
        """η = 1 - T_cold/T_hot."""
        from mechanics_dsl.domains.thermodynamics import CarnotEngine
        
        engine = CarnotEngine(T_hot=500, T_cold=300)
        eta = engine.efficiency()
        
        expected = 1 - 300/500
        
        assert np.isclose(eta, expected)
    
    def test_efficiency_less_than_one(self):
        """Efficiency < 1 always."""
        from mechanics_dsl.domains.thermodynamics import CarnotEngine
        
        # Even with very large temperature difference
        engine = CarnotEngine(T_hot=10000, T_cold=1)
        
        assert engine.efficiency() < 1.0
    
    def test_work_plus_heat_rejected(self):
        """W + Q_cold = Q_hot (energy conservation)."""
        from mechanics_dsl.domains.thermodynamics import CarnotEngine
        
        engine = CarnotEngine(T_hot=600, T_cold=300)
        Q_hot = 1000.0
        
        W = engine.work_output(Q_hot)
        Q_cold = engine.heat_rejected(Q_hot)
        
        assert np.isclose(W + Q_cold, Q_hot)
    
    def test_cop_refrigerator(self):
        """COP_ref = T_cold/(T_hot - T_cold)."""
        from mechanics_dsl.domains.thermodynamics import CarnotEngine
        
        engine = CarnotEngine(T_hot=300, T_cold=250)
        cop = engine.cop_refrigerator()
        
        expected = 250 / (300 - 250)
        
        assert np.isclose(cop, expected)


class TestOttoCycle:
    """Tests for Otto cycle."""
    
    def test_otto_efficiency_increases_with_compression(self):
        """Higher compression ratio = higher efficiency."""
        from mechanics_dsl.domains.thermodynamics import OttoCycle
        
        otto_low = OttoCycle(compression_ratio=8)
        otto_high = OttoCycle(compression_ratio=12)
        
        assert otto_high.efficiency() > otto_low.efficiency()
    
    def test_otto_vs_carnot(self):
        """Otto efficiency ≤ Carnot for same compression ratio."""
        from mechanics_dsl.domains.thermodynamics import OttoCycle, CarnotEngine
        
        otto = OttoCycle(compression_ratio=10, gamma=1.4)
        
        # For Otto, max T ratio is r^(γ-1)
        T_ratio = 10**(1.4 - 1)
        
        # Otto at same conditions equals the Carnot efficiency
        # (this is the theoretical limit)
        carnot = CarnotEngine(T_hot=300 * T_ratio, T_cold=300)
        
        # Should be equal (Otto is ideal cycle)
        assert np.isclose(otto.efficiency(), carnot.efficiency(), rtol=0.01)



class TestDieselCycle:
    """Tests for Diesel cycle."""
    
    def test_diesel_efficiency(self):
        """Diesel cycle gives reasonable efficiency."""
        from mechanics_dsl.domains.thermodynamics import DieselCycle
        
        diesel = DieselCycle(compression_ratio=18, cutoff_ratio=2.0)
        eta = diesel.efficiency()
        
        # Typical diesel: 40-50%
        assert 0.3 < eta < 0.7


class TestVanDerWaalsGas:
    """Tests for van der Waals equation."""
    
    def test_ideal_gas_limit(self):
        """a=b=0 gives ideal gas."""
        from mechanics_dsl.domains.thermodynamics import VanDerWaalsGas, R_GAS
        
        # Very small a and b
        vdw = VanDerWaalsGas(a=0.0, b=0.0)
        
        V = 0.0224  # 22.4 L
        T = 273.15
        P = vdw.pressure(V, T, n=1)
        
        # Should match ideal gas
        P_ideal = R_GAS * T / V
        
        assert np.isclose(P, P_ideal, rtol=0.01)
    
    def test_critical_point_exists(self):
        """Critical point exists for nonzero a, b."""
        from mechanics_dsl.domains.thermodynamics import VanDerWaalsGas
        
        # CO2 parameters
        vdw = VanDerWaalsGas(a=0.364, b=4.27e-5)
        P_c, V_c, T_c = vdw.critical_point()
        
        assert P_c > 0 and V_c > 0 and T_c > 0
    
    def test_compressibility_deviation(self):
        """Z deviates from 1 for real gas."""
        from mechanics_dsl.domains.thermodynamics import VanDerWaalsGas
        
        vdw = VanDerWaalsGas(a=0.364, b=4.27e-5)
        
        Z = vdw.compressibility_factor(V=0.001, T=300)
        
        # Should be close to but not exactly 1
        assert Z != 1.0


class TestPhaseTransition:
    """Tests for phase transitions."""
    
    def test_clausius_clapeyron(self):
        """dP/dT = L/(T ΔV)."""
        from mechanics_dsl.domains.thermodynamics import PhaseTransition
        
        # Water at 100°C
        transition = PhaseTransition(
            T_transition=373.15,
            P_transition=101325,
            latent_heat=40650,  # J/mol
            delta_V=0.03  # m³/mol (steam volume >> water)
        )
        
        slope = transition.clausius_clapeyron_slope()
        
        expected = 40650 / (373.15 * 0.03)
        
        assert np.isclose(slope, expected)
    
    def test_entropy_change(self):
        """ΔS = L/T."""
        from mechanics_dsl.domains.thermodynamics import PhaseTransition
        
        transition = PhaseTransition(
            T_transition=373.15,
            P_transition=101325,
            latent_heat=40650,
            delta_V=0.03
        )
        
        delta_S = transition.entropy_change()
        expected = 40650 / 373.15
        
        assert np.isclose(delta_S, expected)


class TestHeatCapacity:
    """Tests for heat capacity models."""
    
    def test_debye_high_t_limit(self):
        """At T >> θ_D, C → 3nR."""
        from mechanics_dsl.domains.thermodynamics import HeatCapacity, R_GAS
        
        C = HeatCapacity.debye_heat_capacity(T=1000, theta_D=300, n=1)
        
        assert np.isclose(C, 3 * R_GAS, rtol=0.01)
    
    def test_einstein_high_t_limit(self):
        """At T >> θ_E, C → 3nR."""
        from mechanics_dsl.domains.thermodynamics import HeatCapacity, R_GAS
        
        C = HeatCapacity.einstein_heat_capacity(T=1000, theta_E=200, n=1)
        
        assert np.isclose(C, 3 * R_GAS, rtol=0.01)
