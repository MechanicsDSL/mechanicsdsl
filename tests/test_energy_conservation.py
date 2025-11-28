"""
Energy conservation tests - Verify energy is conserved in conservative systems
Tests both new package structure and original core.py
"""
import pytest
import numpy as np
import sys
import os
from pathlib import Path

# Detect CI environment and adjust tolerances
IS_CI = os.getenv('CI') == 'true' or os.getenv('GITHUB_ACTIONS') == 'true'
# Use more lenient tolerances in CI due to numerical differences
ENERGY_TOL_MULTIPLIER = 3.0 if IS_CI else 1.0

try:
    from mechanics_dsl import PhysicsCompiler
    from mechanics_dsl.energy import PotentialEnergyCalculator
    NEW_PACKAGE = True
except ImportError:
    NEW_PACKAGE = False

try:
    sys.path.insert(0, str(Path(__file__).parent.parent / 'reference'))
    from core import PhysicsCompiler as CorePhysicsCompiler, PotentialEnergyCalculator as CorePotentialEnergyCalculator
    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False


def get_compiler():
    """Get compiler instance"""
    if NEW_PACKAGE:
        return PhysicsCompiler()
    elif CORE_AVAILABLE:
        return CorePhysicsCompiler()
    else:
        pytest.skip("Neither new package nor core.py available")


def get_energy_calculator():
    """Get energy calculator"""
    if NEW_PACKAGE:
        return PotentialEnergyCalculator
    elif CORE_AVAILABLE:
        return CorePotentialEnergyCalculator
    else:
        pytest.skip("Energy calculator not available")


class TestEnergyConservation:
    """Test energy conservation in various systems"""
    
    def test_harmonic_oscillator_energy(self):
        """Test energy conservation in harmonic oscillator"""
        dsl_code = r"""
        \system{oscillator_energy}
        \defvar{x}{Position}{m}
        \defvar{m}{Mass}{kg}
        \defvar{k}{Spring Constant}{N/m}
        
        \parameter{m}{1.0}{kg}
        \parameter{k}{10.0}{N/m}
        
        \lagrangian{\frac{1}{2} * m * \dot{x}^2 - \frac{1}{2} * k * x^2}
        
        \initial{x=1.0, x_dot=0.0}
        """
        
        compiler = get_compiler()
        result = compiler.compile_dsl(dsl_code)
        assert result['success']
        
        solution = compiler.simulate(t_span=(0, 10), num_points=1000)
        assert solution['success']
        
        EnergyCalc = get_energy_calculator()
        params = compiler.simulator.parameters
        KE = EnergyCalc.compute_kinetic_energy(solution, params)
        PE = EnergyCalc.compute_potential_energy(solution, params, 'oscillator_energy')
        E_total = KE + PE
        
        # Energy should be conserved to within tolerance (more lenient in CI)
        if E_total[0] != 0:
            energy_error = np.abs((E_total - E_total[0]) / E_total[0])
            max_error = np.max(energy_error)
            tolerance = 0.01 * ENERGY_TOL_MULTIPLIER
            assert max_error < tolerance, f"Energy conservation violated: max error {max_error:.6f} (tolerance: {tolerance:.6f})"
            
            # Mean error should be even smaller
            mean_error = np.mean(energy_error)
            mean_tolerance = 0.005 * ENERGY_TOL_MULTIPLIER
            assert mean_error < mean_tolerance, f"Mean energy error too large: {mean_error:.6f} (tolerance: {mean_tolerance:.6f})"
    
    def test_pendulum_energy(self):
        """Test energy conservation in simple pendulum"""
        dsl_code = r"""
        \system{pendulum_energy}
        \defvar{theta}{Angle}{rad}
        \defvar{m}{Mass}{kg}
        \defvar{l}{Constant}{m}
        \defvar{g}{Acceleration}{m/s^2}
        
        \parameter{m}{1.0}{kg}
        \parameter{l}{1.0}{m}
        \parameter{g}{9.81}{m/s^2}
        
        \lagrangian{\frac{1}{2} * m * l^2 * \dot{theta}^2 - m * g * l * (1 - \cos{theta})}
        
        \initial{theta=0.5, theta_dot=0.0}
        """
        
        compiler = get_compiler()
        result = compiler.compile_dsl(dsl_code)
        assert result['success']
        
        solution = compiler.simulate(t_span=(0, 10), num_points=1000, method='RK45')
        assert solution['success']
        
        EnergyCalc = get_energy_calculator()
        params = compiler.simulator.parameters
        KE = EnergyCalc.compute_kinetic_energy(solution, params)
        PE = EnergyCalc.compute_potential_energy(solution, params, 'pendulum_energy')
        E_total = KE + PE
        
        # Energy should be conserved (more lenient in CI)
        if E_total[0] != 0:
            energy_error = np.abs((E_total - E_total[0]) / E_total[0])
            tolerance = 0.05 * ENERGY_TOL_MULTIPLIER
            assert np.max(energy_error) < tolerance, f"Energy error: {np.max(energy_error):.6f} (tolerance: {tolerance:.6f})"
    
    def test_coupled_oscillators_energy(self):
        """Test energy conservation in coupled oscillators"""
        dsl_code = r"""
        \system{coupled_energy}
        \defvar{x1}{Position}{m}
        \defvar{x2}{Position}{m}
        \defvar{m}{Mass}{kg}
        \defvar{k}{Spring Constant}{N/m}
        \defvar{k_c}{Coupling Constant}{N/m}
        
        \parameter{m}{1.0}{kg}
        \parameter{k}{10.0}{N/m}
        \parameter{k_c}{2.0}{N/m}
        
        \lagrangian{
            \frac{1}{2} * m * \dot{x1}^2 
            + \frac{1}{2} * m * \dot{x2}^2 
            - \frac{1}{2} * k * x1^2 
            - \frac{1}{2} * k * x2^2 
            - \frac{1}{2} * k_c * (x1 - x2)^2
        }
        
        \initial{x1=1.0, x1_dot=0.0, x2=0.0, x2_dot=0.0}
        """
        
        compiler = get_compiler()
        result = compiler.compile_dsl(dsl_code)
        assert result['success']
        
        solution = compiler.simulate(t_span=(0, 10), num_points=1000)
        assert solution['success']
        
        EnergyCalc = get_energy_calculator()
        params = compiler.simulator.parameters
        KE = EnergyCalc.compute_kinetic_energy(solution, params)
        PE = EnergyCalc.compute_potential_energy(solution, params, 'coupled_energy')
        E_total = KE + PE
        
        if E_total[0] != 0:
            energy_error = np.abs((E_total - E_total[0]) / E_total[0])
            tolerance = 0.05 * ENERGY_TOL_MULTIPLIER
            assert np.max(energy_error) < tolerance, f"Energy error: {np.max(energy_error):.6f} (tolerance: {tolerance:.6f})"
    
    def test_kepler_problem_energy(self):
        """Test energy conservation in Kepler problem"""
        dsl_code = r"""
        \system{kepler_energy}
        \defvar{r}{Length}{m}
        \defvar{phi}{Angle}{rad}
        \defvar{m}{Mass}{kg}
        \defvar{M}{Central Mass}{kg}
        \defvar{G}{Gravitational Constant}{m^3/(kg*s^2)}
        
        \parameter{m}{1.0}{kg}
        \parameter{M}{1000.0}{kg}
        \parameter{G}{6.674e-11}{m^3/(kg*s^2)}
        
        \lagrangian{
            \frac{1}{2} * m * (\dot{r}^2 + r^2 * \dot{phi}^2) 
            + G * M * m / r
        }
        
        \initial{r=10.0, r_dot=0.0, phi=0.0, phi_dot=0.1}
        """
        
        compiler = get_compiler()
        result = compiler.compile_dsl(dsl_code)
        assert result['success']
        
        solution = compiler.simulate(t_span=(0, 50), num_points=2000, method='LSODA')
        assert solution['success']
        
        EnergyCalc = get_energy_calculator()
        params = compiler.simulator.parameters
        KE = EnergyCalc.compute_kinetic_energy(solution, params)
        PE = EnergyCalc.compute_potential_energy(solution, params, 'kepler_energy')
        E_total = KE + PE
        
        if E_total[0] != 0:
            energy_error = np.abs((E_total - E_total[0]) / E_total[0])
            # Kepler problem may have larger errors due to long integration
            tolerance = 0.1 * ENERGY_TOL_MULTIPLIER
            assert np.max(energy_error) < tolerance, f"Energy error: {np.max(energy_error):.6f} (tolerance: {tolerance:.6f})"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

