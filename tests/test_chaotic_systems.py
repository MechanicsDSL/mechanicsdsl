"""
Chaotic system tests - Lorenz, Rössler, Hénon-Heiles, etc.
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


class TestLorenzSystem:
    """Test Lorenz attractor (chaotic)"""
    
    def test_lorenz_system(self):
        """Test Lorenz system compilation and simulation"""
        dsl_code = r"""
        \system{lorenz_system}
        \defvar{x}{Position}{m}
        \defvar{y}{Position}{m}
        \defvar{z}{Position}{m}
        \defvar{sigma}{Parameter}{1}
        \defvar{rho}{Parameter}{1}
        \defvar{beta}{Parameter}{1}
        
        \parameter{sigma}{10.0}{1}
        \parameter{rho}{28.0}{1}
        \parameter{beta}{2.666}{1}
        
        \lagrangian{
            \frac{1}{2} * (\dot{x}^2 + \dot{y}^2 + \dot{z}^2) 
            - \frac{1}{2} * sigma * x^2 
            - \frac{1}{2} * beta * z^2
        }
        
        \force{sigma * (y - x)}
        \force{rho * x - y - x * z}
        \force{x * y - beta * z}
        
        \initial{x=1.0, x_dot=0.0, y=1.0, y_dot=0.0, z=1.0, z_dot=0.0}
        """
        
        compiler = get_compiler()
        result = compiler.compile_dsl(dsl_code)
        
        assert result['success']
        assert len(result['coordinates']) == 3
        
        solution = compiler.simulate(t_span=(0, 10), num_points=1000, method='RK45')
        
        assert solution['success']
        assert solution['y'].shape[0] == 6  # x, x_dot, y, y_dot, z, z_dot
        
        # Check for chaotic behavior - values should vary significantly
        x = solution['y'][0]
        y = solution['y'][2]
        z = solution['y'][4]
        
        assert np.max(np.abs(x)) > 5.0  # Lorenz attractor has large excursions
        assert np.max(np.abs(y)) > 5.0
        assert np.max(np.abs(z)) > 5.0
        
        # Check that system doesn't blow up
        assert np.all(np.isfinite(x))
        assert np.all(np.isfinite(y))
        assert np.all(np.isfinite(z))


class TestRosslerAttractor:
    """Test Rössler attractor"""
    
    def test_rossler_attractor(self):
        """Test Rössler attractor system"""
        dsl_code = r"""
        \system{rossler_attractor}
        \defvar{x}{Position}{m}
        \defvar{y}{Position}{m}
        \defvar{z}{Position}{m}
        \defvar{a}{Parameter}{1}
        \defvar{b}{Parameter}{1}
        \defvar{c}{Parameter}{1}
        
        \parameter{a}{0.2}{1}
        \parameter{b}{0.2}{1}
        \parameter{c}{5.7}{1}
        
        \lagrangian{
            \frac{1}{2} * (\dot{x}^2 + \dot{y}^2 + \dot{z}^2) 
            - \frac{1}{2} * x^2 
            - \frac{1}{2} * y^2
        }
        
        \force{-y - z}
        \force{x + a * y}
        \force{b + z * (x - c)}
        
        \initial{x=1.0, x_dot=0.0, y=1.0, y_dot=0.0, z=0.0, z_dot=0.0}
        """
        
        compiler = get_compiler()
        result = compiler.compile_dsl(dsl_code)
        
        assert result['success']
        
        solution = compiler.simulate(t_span=(0, 50), num_points=2000, method='RK45')
        
        assert solution['success']
        
        # Rössler attractor should show bounded chaotic motion
        x = solution['y'][0]
        assert np.max(np.abs(x)) < 50.0  # Should be bounded
        assert np.all(np.isfinite(x))


class TestHenonHeiles:
    """Test Hénon-Heiles system"""
    
    def test_henon_heiles(self):
        """Test Hénon-Heiles system with energy conservation"""
        dsl_code = r"""
        \system{henon_heiles}
        \defvar{x}{Position}{m}
        \defvar{y}{Position}{m}
        \defvar{m}{Mass}{kg}
        
        \parameter{m}{1.0}{kg}
        
        \lagrangian{
            \frac{1}{2} * m * (\dot{x}^2 + \dot{y}^2) 
            - \frac{1}{2} * (x^2 + y^2) 
            - x^2 * y + \frac{1}{3} * y^3
        }
        
        \initial{x=0.1, x_dot=0.0, y=0.0, y_dot=0.0}
        """
        
        compiler = get_compiler()
        result = compiler.compile_dsl(dsl_code)
        
        assert result['success']
        
        solution = compiler.simulate(t_span=(0, 20), num_points=1000, method='RK45')
        
        assert solution['success']
        
        # Check energy conservation
        from mechanics_dsl.energy import PotentialEnergyCalculator
        params = compiler.simulator.parameters
        KE = PotentialEnergyCalculator.compute_kinetic_energy(solution, params)
        PE = PotentialEnergyCalculator.compute_potential_energy(solution, params, 'henon_heiles')
        E_total = KE + PE
        
        if E_total[0] != 0:
            energy_error = np.abs((E_total - E_total[0]) / E_total[0])
            assert np.max(energy_error) < 0.15, f"Energy error too large: {np.max(energy_error)}"


class TestVanDerPol:
    """Test Van der Pol oscillator"""
    
    def test_van_der_pol(self):
        """Test Van der Pol oscillator (limit cycle)"""
        dsl_code = r"""
        \system{van_der_pol}
        \defvar{x}{Position}{m}
        \defvar{mu}{Damping Parameter}{1}
        \defvar{omega}{Natural Frequency}{rad/s}
        
        \parameter{mu}{1.0}{1}
        \parameter{omega}{1.0}{rad/s}
        
        \lagrangian{
            \frac{1}{2} * \dot{x}^2 
            - \frac{1}{2} * omega^2 * x^2
        }
        
        \force{-mu * (x^2 - 1) * x_dot}
        
        \initial{x=2.0, x_dot=0.0}
        """
        
        compiler = get_compiler()
        result = compiler.compile_dsl(dsl_code)
        
        assert result['success']
        
        solution = compiler.simulate(t_span=(0, 20), num_points=1000)
        
        assert solution['success']
        
        # Van der Pol should show limit cycle behavior
        x = solution['y'][0]
        assert np.max(np.abs(x)) > 1.0
        assert np.all(np.isfinite(x))


class TestDuffingOscillator:
    """Test Duffing oscillator (nonlinear)"""
    
    def test_duffing_oscillator(self):
        """Test Duffing oscillator with nonlinear spring"""
        dsl_code = r"""
        \system{duffing_oscillator}
        \defvar{x}{Position}{m}
        \defvar{m}{Mass}{kg}
        \defvar{k}{Spring Constant}{N/m}
        \defvar{alpha}{Nonlinear Coeff}{N/m^3}
        \defvar{b}{Damping Coeff}{N*s/m}
        
        \parameter{m}{1.0}{kg}
        \parameter{k}{10.0}{N/m}
        \parameter{alpha}{-1.0}{N/m^3}
        \parameter{b}{0.1}{N*s/m}
        
        \lagrangian{
            \frac{1}{2} * m * \dot{x}^2 
            - \frac{1}{2} * k * x^2 
            - \frac{1}{4} * alpha * x^4
        }
        
        \force{-b * x_dot}
        
        \initial{x=1.0, x_dot=0.0}
        """
        
        compiler = get_compiler()
        result = compiler.compile_dsl(dsl_code)
        
        assert result['success']
        
        solution = compiler.simulate(t_span=(0, 10), num_points=500)
        
        assert solution['success']
        
        # Should show nonlinear oscillation
        x = solution['y'][0]
        assert np.max(np.abs(x)) > 0.1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

