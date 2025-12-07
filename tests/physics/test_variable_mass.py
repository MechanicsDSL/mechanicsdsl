"""
Tests for Variable Mass Systems Module
"""
import pytest
import numpy as np

from mechanics_dsl.domains.classical import (
    RocketEquation,
    RocketParameters,
    RocketState,
    VariableMassSystem,
    SymbolicVariableMass,
    tsiolkovsky_delta_v,
    required_mass_ratio,
    specific_impulse_to_exhaust_velocity
)


class TestRocketParameters:
    """Test RocketParameters class."""
    
    def test_create_parameters(self):
        """Test creating rocket parameters."""
        params = RocketParameters(
            initial_mass=1000.0,
            fuel_mass=800.0,
            exhaust_velocity=3000.0,
            mass_flow_rate=10.0
        )
        
        assert params.initial_mass == 1000.0
        assert params.fuel_mass == 800.0
    
    def test_dry_mass(self):
        """Test dry mass calculation."""
        params = RocketParameters(
            initial_mass=1000.0,
            fuel_mass=800.0,
            exhaust_velocity=3000.0,
            mass_flow_rate=10.0
        )
        
        assert params.dry_mass == 200.0
    
    def test_mass_ratio(self):
        """Test mass ratio."""
        params = RocketParameters(
            initial_mass=1000.0,
            fuel_mass=800.0,
            exhaust_velocity=3000.0,
            mass_flow_rate=10.0
        )
        
        # 1000 / 200 = 5
        assert params.mass_ratio == 5.0


class TestRocketEquation:
    """Test RocketEquation class."""
    
    def test_ideal_delta_v(self):
        """Test Tsiolkovsky ideal delta-v."""
        rocket = RocketEquation()
        
        params = RocketParameters(
            initial_mass=1000.0,
            fuel_mass=800.0,
            exhaust_velocity=3000.0,
            mass_flow_rate=10.0
        )
        
        dv = rocket.ideal_delta_v(params)
        
        # Δv = v_e * ln(m_0/m_f) = 3000 * ln(5) ≈ 4828
        expected = 3000.0 * np.log(5.0)
        assert np.isclose(dv, expected, rtol=1e-10)
    
    def test_burn_time(self):
        """Test burn time calculation."""
        rocket = RocketEquation()
        
        params = RocketParameters(
            initial_mass=1000.0,
            fuel_mass=800.0,
            exhaust_velocity=3000.0,
            mass_flow_rate=10.0
        )
        
        t_burn = rocket.burn_time(params)
        
        # 800 / 10 = 80 seconds
        assert t_burn == 80.0
    
    def test_delta_v_with_gravity(self):
        """Test delta-v with gravity loss."""
        rocket = RocketEquation()
        
        params = RocketParameters(
            initial_mass=1000.0,
            fuel_mass=800.0,
            exhaust_velocity=3000.0,
            mass_flow_rate=10.0
        )
        
        dv_gravity = rocket.delta_v_with_gravity(params, g=9.81)
        dv_ideal = rocket.ideal_delta_v(params)
        
        # Gravity loss reduces delta-v
        assert dv_gravity < dv_ideal
    
    def test_specific_impulse(self):
        """Test specific impulse calculation."""
        rocket = RocketEquation()
        
        params = RocketParameters(
            initial_mass=1000.0,
            fuel_mass=800.0,
            exhaust_velocity=3000.0,
            mass_flow_rate=10.0
        )
        
        isp = rocket.specific_impulse(params)
        
        # I_sp = v_e / g_0 = 3000 / 9.81 ≈ 305.8 seconds
        assert np.isclose(isp, 3000.0 / 9.81, rtol=1e-10)
    
    def test_simulate(self):
        """Test rocket simulation."""
        rocket = RocketEquation()
        
        params = RocketParameters(
            initial_mass=1000.0,
            fuel_mass=800.0,
            exhaust_velocity=3000.0,
            mass_flow_rate=10.0
        )
        
        result = rocket.simulate(params, t_span=(0, 100), g=0.0, num_points=50)
        
        assert 'time' in result
        assert 'velocity' in result
        assert 'mass' in result
        assert len(result['time']) == 50


class TestVariableMassSystem:
    """Test VariableMassSystem class."""
    
    def test_conveyor_belt_force(self):
        """Test conveyor belt force calculation."""
        system = VariableMassSystem()
        
        force = system.conveyor_belt_force(belt_velocity=2.0, mass_rate=5.0)
        
        # F = v * dm/dt = 2 * 5 = 10
        assert force == 10.0
    
    def test_falling_chain(self):
        """Test falling chain force."""
        system = VariableMassSystem()
        
        # Chain falling from height onto table
        force = system.falling_chain(
            chain_length=2.0,
            linear_density=1.0,
            height=1.0,
            g=10.0
        )
        
        # Should be positive (weight + impact)
        assert force > 0


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_tsiolkovsky_delta_v(self):
        """Test tsiolkovsky_delta_v function."""
        dv = tsiolkovsky_delta_v(1000.0, 200.0, 3000.0)
        
        expected = 3000.0 * np.log(5.0)
        assert np.isclose(dv, expected)
    
    def test_required_mass_ratio(self):
        """Test required_mass_ratio function."""
        ratio = required_mass_ratio(delta_v=3000.0, exhaust_velocity=3000.0)
        
        # exp(1) ≈ 2.718
        assert np.isclose(ratio, np.e)
    
    def test_specific_impulse_to_exhaust_velocity(self):
        """Test specific_impulse_to_exhaust_velocity function."""
        v_e = specific_impulse_to_exhaust_velocity(isp=300.0)
        
        # v_e = I_sp * g_0 = 300 * 9.81 ≈ 2943
        assert np.isclose(v_e, 300.0 * 9.81)


class TestSymbolicVariableMass:
    """Test symbolic variable mass analysis."""
    
    def test_rocket_equation_symbolic(self):
        """Test symbolic rocket equation derivation."""
        symbolic = SymbolicVariableMass()
        
        result = symbolic.rocket_equation_symbolic()
        
        assert 'delta_v_ideal' in result
        assert 'burn_time' in result
        assert 'specific_impulse' in result
    
    def test_multistage_rocket(self):
        """Test multistage rocket delta-v."""
        symbolic = SymbolicVariableMass()
        
        total_dv = symbolic.multistage_rocket(stages=2)
        
        # Should have terms from both stages
        import sympy as sp
        assert total_dv.has(sp.Symbol('v_e1', positive=True))
        assert total_dv.has(sp.Symbol('v_e2', positive=True))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
