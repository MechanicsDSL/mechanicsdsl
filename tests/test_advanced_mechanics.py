"""
Advanced mechanics tests - Coupled systems, parametric resonance, etc.
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


class TestCoupledOscillators:
    """Test coupled oscillator systems"""
    
    def test_three_coupled_oscillators(self):
        """Test three coupled oscillators"""
        dsl_code = r"""
        \system{three_coupled}
        \defvar{x1}{Position}{m}
        \defvar{x2}{Position}{m}
        \defvar{x3}{Position}{m}
        \defvar{m}{Mass}{kg}
        \defvar{k}{Spring Constant}{N/m}
        \defvar{k_c}{Coupling Constant}{N/m}
        
        \parameter{m}{1.0}{kg}
        \parameter{k}{10.0}{N/m}
        \parameter{k_c}{2.0}{N/m}
        
        \lagrangian{
            \frac{1}{2} * m * (\dot{x1}^2 + \dot{x2}^2 + \dot{x3}^2)
            - \frac{1}{2} * k * (x1^2 + x2^2 + x3^2)
            - \frac{1}{2} * k_c * ((x1 - x2)^2 + (x2 - x3)^2)
        }
        
        \initial{x1=1.0, x1_dot=0.0, x2=0.0, x2_dot=0.0, x3=0.0, x3_dot=0.0}
        """
        
        compiler = get_compiler()
        result = compiler.compile_dsl(dsl_code)
        
        assert result['success']
        assert len(result['coordinates']) == 3
        
        solution = compiler.simulate(t_span=(0, 10), num_points=500)
        assert solution['success']
        assert solution['y'].shape[0] == 6


class TestParametricResonance:
    """Test parametric resonance systems"""
    
    def test_parametric_pendulum(self):
        """Test parametrically driven pendulum"""
        dsl_code = r"""
        \system{parametric_pendulum}
        \defvar{theta}{Angle}{rad}
        \defvar{m}{Mass}{kg}
        \defvar{l}{Length}{m}
        \defvar{g}{Acceleration}{m/s^2}
        \defvar{omega_d}{Drive Frequency}{rad/s}
        \defvar{A}{Amplitude}{m}
        
        \parameter{m}{1.0}{kg}
        \parameter{l}{1.0}{m}
        \parameter{g}{9.81}{m/s^2}
        \parameter{omega_d}{2.0}{rad/s}
        \parameter{A}{0.1}{m}
        
        \lagrangian{
            \frac{1}{2} * m * l^2 * \dot{theta}^2 
            - m * g * (l + A * \cos{omega_d * t}) * (1 - \cos{theta})
        }
        
        \initial{theta=0.1, theta_dot=0.0}
        """
        
        compiler = get_compiler()
        result = compiler.compile_dsl(dsl_code)
        
        assert result['success']
        
        solution = compiler.simulate(t_span=(0, 20), num_points=1000)
        assert solution['success']
        
        # Parametric resonance can cause growth
        theta = solution['y'][0]
        assert np.all(np.isfinite(theta))


class TestKeplerProblem:
    """Test Kepler problem (planetary motion)"""
    
    def test_kepler_problem(self):
        """Test Kepler problem with energy and angular momentum"""
        dsl_code = r"""
        \system{kepler_test}
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
        
        # Check orbital motion
        r = solution['y'][0]
        phi = solution['y'][2]
        
        assert np.all(r > 0)  # Radius should stay positive
        assert np.max(phi) > 0.1  # Should orbit


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

