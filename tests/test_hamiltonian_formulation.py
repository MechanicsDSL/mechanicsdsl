"""
Hamiltonian formulation tests
Tests both new package structure and original core.py
"""
import pytest
import numpy as np
import sys
from pathlib import Path

try:
    from mechanics_dsl import PhysicsCompiler
    NEW_PACKAGE = True
except ImportError:
    NEW_PACKAGE = False

try:
    sys.path.insert(0, str(Path(__file__).parent.parent / 'reference'))
    from core import PhysicsCompiler as CorePhysicsCompiler
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


class TestHamiltonianOscillator:
    """Test harmonic oscillator using Hamiltonian formulation"""
    
    def test_hamiltonian_oscillator(self):
        """Test harmonic oscillator with Hamiltonian formulation"""
        dsl_code = r"""
        \system{hamiltonian_oscillator}
        \defvar{x}{Position}{m}
        \defvar{m}{Mass}{kg}
        \defvar{k}{Spring Constant}{N/m}
        
        \parameter{m}{1.0}{kg}
        \parameter{k}{10.0}{N/m}
        
        \lagrangian{\frac{1}{2} * m * \dot{x}^2 - \frac{1}{2} * k * x^2}
        
        \initial{x=1.0, x_dot=0.0}
        """
        
        compiler = get_compiler()
        result = compiler.compile_dsl(dsl_code, use_hamiltonian=True)
        
        assert result['success']
        assert result['formulation'] == 'Hamiltonian'
        assert compiler.use_hamiltonian_formulation
        
        solution = compiler.simulate(t_span=(0, 10), num_points=500)
        
        assert solution['success']
        assert solution['use_hamiltonian']
        
        # Should show oscillatory motion
        x = solution['y'][0]
        assert np.max(np.abs(x)) > 0.5
        
        # Energy should be conserved
        from mechanics_dsl.energy import PotentialEnergyCalculator
        params = compiler.simulator.parameters
        KE = PotentialEnergyCalculator.compute_kinetic_energy(solution, params)
        PE = PotentialEnergyCalculator.compute_potential_energy(solution, params, 'hamiltonian_oscillator')
        E_total = KE + PE
        
        if E_total[0] != 0:
            energy_error = np.abs((E_total - E_total[0]) / E_total[0])
            assert np.max(energy_error) < 0.05, f"Energy error: {np.max(energy_error)}"
    
    def test_hamiltonian_pendulum(self):
        """Test pendulum with Hamiltonian formulation"""
        dsl_code = r"""
        \system{hamiltonian_pendulum}
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
        result = compiler.compile_dsl(dsl_code, use_hamiltonian=True)
        
        assert result['success']
        assert result['formulation'] == 'Hamiltonian'
        
        solution = compiler.simulate(t_span=(0, 5), num_points=500)
        
        assert solution['success']
        
        # Should show periodic motion
        theta = solution['y'][0]
        assert np.max(np.abs(theta)) > 0.1


class TestExplicitHamiltonian:
    """Test systems with explicitly defined Hamiltonian"""
    
    def test_explicit_hamiltonian(self):
        """Test system with explicit Hamiltonian definition"""
        dsl_code = r"""
        \system{explicit_hamiltonian}
        \defvar{x}{Position}{m}
        \defvar{p}{Momentum}{kg*m/s}
        \defvar{m}{Mass}{kg}
        \defvar{k}{Spring Constant}{N/m}
        
        \parameter{m}{1.0}{kg}
        \parameter{k}{10.0}{N/m}
        
        \hamiltonian{\frac{p^2}{2 * m} + \frac{1}{2} * k * x^2}
        
        \initial{x=1.0, p=0.0}
        """
        
        compiler = get_compiler()
        result = compiler.compile_dsl(dsl_code)
        
        assert result['success']
        assert result['formulation'] == 'Hamiltonian'
        
        solution = compiler.simulate(t_span=(0, 10), num_points=500)
        
        assert solution['success']
        assert solution['use_hamiltonian']
        
        # Position should oscillate
        x = solution['y'][0]
        assert np.max(np.abs(x)) > 0.5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

