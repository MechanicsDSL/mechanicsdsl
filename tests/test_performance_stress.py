"""
Performance and stress tests - Long simulations, many coordinates, etc.
Tests both new package structure and original core.py
"""
import pytest
import numpy as np
import sys
import time
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


class TestLongSimulations:
    """Test long-duration simulations"""
    
    def test_long_pendulum_simulation(self):
        """Test pendulum simulation over long time span"""
        dsl_code = r"""
        \system{long_pendulum}
        \defvar{theta}{Angle}{rad}
        \defvar{m}{Mass}{kg}
        \defvar{l}{Constant}{m}
        \defvar{g}{Acceleration}{m/s^2}
        
        \parameter{m}{1.0}{kg}
        \parameter{l}{1.0}{m}
        \parameter{g}{9.81}{m/s^2}
        
        \lagrangian{\frac{1}{2} * m * l^2 * \dot{theta}^2 - m * g * l * (1 - \cos{theta})}
        
        \initial{theta=0.1, theta_dot=0.0}
        """
        
        compiler = get_compiler()
        result = compiler.compile_dsl(dsl_code)
        assert result['success']
        
        # Long time span
        start_time = time.time()
        solution = compiler.simulate(t_span=(0, 100), num_points=5000, method='LSODA')
        elapsed = time.time() - start_time
        
        assert solution['success']
        assert len(solution['t']) == 5000
        assert elapsed < 30.0  # Should complete in reasonable time
        
        # Should show many oscillations
        theta = solution['y'][0]
        # Count zero crossings
        zero_crossings = np.sum(np.diff(np.sign(theta)) != 0)
        assert zero_crossings > 10  # Should have many oscillations
    
    def test_high_resolution_simulation(self):
        """Test simulation with very high resolution"""
        dsl_code = r"""
        \system{high_res}
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
        
        # Very high resolution
        solution = compiler.simulate(t_span=(0, 10), num_points=10000, method='RK45')
        
        assert solution['success']
        assert len(solution['t']) == 10000
        
        # Check solution quality
        x = solution['y'][0]
        assert np.all(np.isfinite(x))
        
        # Energy should be well conserved with high resolution
        from mechanics_dsl.energy import PotentialEnergyCalculator
        params = compiler.simulator.parameters
        KE = PotentialEnergyCalculator.compute_kinetic_energy(solution, params)
        PE = PotentialEnergyCalculator.compute_potential_energy(solution, params, 'high_res')
        E_total = KE + PE
        
        if E_total[0] != 0:
            energy_error = np.abs((E_total - E_total[0]) / E_total[0])
            assert np.max(energy_error) < 0.001  # Very high resolution should give better energy conservation


class TestManyCoordinates:
    """Test systems with many degrees of freedom"""
    
    def test_six_coordinate_system(self):
        """Test system with 6 coordinates"""
        dsl_code = r"""
        \system{six_dof}
        \defvar{x1}{Position}{m}
        \defvar{x2}{Position}{m}
        \defvar{x3}{Position}{m}
        \defvar{y1}{Position}{m}
        \defvar{y2}{Position}{m}
        \defvar{y3}{Position}{m}
        \defvar{m}{Mass}{kg}
        \defvar{k}{Spring Constant}{N/m}
        
        \parameter{m}{1.0}{kg}
        \parameter{k}{10.0}{N/m}
        
        \lagrangian{
            \frac{1}{2} * m * (\dot{x1}^2 + \dot{x2}^2 + \dot{x3}^2 + \dot{y1}^2 + \dot{y2}^2 + \dot{y3}^2)
            - \frac{1}{2} * k * (x1^2 + x2^2 + x3^2 + y1^2 + y2^2 + y3^2)
        }
        
        \initial{x1=1.0, x1_dot=0.0, x2=0.0, x2_dot=0.0, x3=0.0, x3_dot=0.0, y1=0.0, y1_dot=0.0, y2=0.0, y2_dot=0.0, y3=0.0, y3_dot=0.0}
        """
        
        compiler = get_compiler()
        result = compiler.compile_dsl(dsl_code)
        
        assert result['success']
        assert len(result['coordinates']) == 6
        
        solution = compiler.simulate(t_span=(0, 5), num_points=500, method='LSODA')
        
        assert solution['success']
        assert solution['y'].shape[0] == 12  # 6 coordinates * 2 states


class TestSolverMethods:
    """Test different solver methods"""
    
    def test_rk45_solver(self):
        """Test RK45 solver"""
        dsl_code = r"""
        \system{rk45_test}
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
        
        solution = compiler.simulate(t_span=(0, 10), num_points=500, method='RK45')
        assert solution['success']
        assert solution['method_used'] == 'RK45'
    
    def test_lsoda_solver(self):
        """Test LSODA solver"""
        dsl_code = r"""
        \system{lsoda_test}
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
        
        solution = compiler.simulate(t_span=(0, 10), num_points=500, method='LSODA')
        assert solution['success']
        assert solution['method_used'] == 'LSODA'
    
    def test_radau_solver(self):
        """Test Radau solver for stiff systems"""
        dsl_code = r"""
        \system{radau_test}
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
        
        solution = compiler.simulate(t_span=(0, 10), num_points=500, method='Radau')
        assert solution['success']
        assert solution['method_used'] == 'Radau'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

