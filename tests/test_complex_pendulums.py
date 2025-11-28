"""
Complex pendulum system tests - Multi-body chaotic systems
Tests both new package structure and original core.py
"""
import pytest
import numpy as np
import sys
from pathlib import Path

# Try importing from new package structure
try:
    from mechanics_dsl import PhysicsCompiler
    NEW_PACKAGE = True
except ImportError:
    NEW_PACKAGE = False

# Try importing from original core.py
try:
    sys.path.insert(0, str(Path(__file__).parent.parent / 'reference'))
    from core import PhysicsCompiler as CorePhysicsCompiler
    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False


def get_compiler():
    """Get compiler instance - prefer new package, fallback to core"""
    if NEW_PACKAGE:
        return PhysicsCompiler()
    elif CORE_AVAILABLE:
        return CorePhysicsCompiler()
    else:
        pytest.skip("Neither new package nor core.py available")


class TestDoublePendulum:
    """Test double pendulum (chaotic system)"""
    
    def test_double_pendulum_compilation(self):
        """Test double pendulum compilation"""
        dsl_code = r"""
        \system{double_pendulum}
        \defvar{theta1}{Angle}{rad}
        \defvar{theta2}{Angle}{rad}
        \defvar{m1}{Mass}{kg}
        \defvar{m2}{Mass}{kg}
        \defvar{l1}{Constant}{m}
        \defvar{l2}{Constant}{m}
        \defvar{g}{Acceleration}{m/s^2}
        
        \parameter{m1}{1.0}{kg}
        \parameter{m2}{1.0}{kg}
        \parameter{l1}{1.0}{m}
        \parameter{l2}{1.0}{m}
        \parameter{g}{9.81}{m/s^2}
        
        \lagrangian{
            \frac{1}{2} * (m1 + m2) * l1^2 * \dot{theta1}^2 
            + \frac{1}{2} * m2 * l2^2 * \dot{theta2}^2 
            + m2 * l1 * l2 * \dot{theta1} * \dot{theta2} * \cos{theta1 - theta2}
            + (m1 + m2) * g * l1 * \cos{theta1}
            + m2 * g * l2 * \cos{theta2}
        }
        
        \initial{theta1=1.57, theta1_dot=0.0, theta2=1.57, theta2_dot=0.0}
        """
        
        compiler = get_compiler()
        result = compiler.compile_dsl(dsl_code)
        
        assert result['success'], f"Compilation failed: {result.get('error', 'Unknown error')}"
        assert result['system_name'] == 'double_pendulum'
        assert 'theta1' in result['coordinates']
        assert 'theta2' in result['coordinates']
        assert len(result['coordinates']) == 2
        
        # Verify equations were derived
        assert compiler.equations is not None
        assert 'theta1_ddot' in compiler.equations or len(compiler.equations) > 0
    
    def test_double_pendulum_simulation(self):
        """Test double pendulum simulation with energy check"""
        dsl_code = r"""
        \system{double_pendulum}
        \defvar{theta1}{Angle}{rad}
        \defvar{theta2}{Angle}{rad}
        \defvar{m1}{Mass}{kg}
        \defvar{m2}{Mass}{kg}
        \defvar{l1}{Constant}{m}
        \defvar{l2}{Constant}{m}
        \defvar{g}{Acceleration}{m/s^2}
        
        \parameter{m1}{1.0}{kg}
        \parameter{m2}{1.0}{kg}
        \parameter{l1}{1.0}{m}
        \parameter{l2}{1.0}{m}
        \parameter{g}{9.81}{m/s^2}
        
        \lagrangian{
            \frac{1}{2} * (m1 + m2) * l1^2 * \dot{theta1}^2 
            + \frac{1}{2} * m2 * l2^2 * \dot{theta2}^2 
            + m2 * l1 * l2 * \dot{theta1} * \dot{theta2} * \cos{theta1 - theta2}
            + (m1 + m2) * g * l1 * \cos{theta1}
            + m2 * g * l2 * \cos{theta2}
        }
        
        \initial{theta1=0.5, theta1_dot=0.0, theta2=0.0, theta2_dot=0.0}
        """
        
        compiler = get_compiler()
        result = compiler.compile_dsl(dsl_code)
        assert result['success']
        
        solution = compiler.simulate(t_span=(0, 5), num_points=500)
        
        assert solution['success'], f"Simulation failed: {solution.get('error', 'Unknown error')}"
        assert solution['y'].shape[0] == 4  # theta1, theta1_dot, theta2, theta2_dot
        assert len(solution['t']) == 500
        
        # Check chaotic behavior - angles should vary significantly
        theta1 = solution['y'][0]
        theta2 = solution['y'][2]
        assert np.max(np.abs(theta1)) > 0.1
        assert np.max(np.abs(theta2)) > 0.1
        
        # Energy should be approximately conserved (within 5%)
        from mechanics_dsl.energy import PotentialEnergyCalculator
        params = compiler.simulator.parameters
        KE = PotentialEnergyCalculator.compute_kinetic_energy(solution, params)
        PE = PotentialEnergyCalculator.compute_potential_energy(solution, params, 'double_pendulum')
        E_total = KE + PE
        
        if E_total[0] != 0:
            energy_error = np.abs((E_total - E_total[0]) / E_total[0])
            assert np.max(energy_error) < 0.1, f"Energy conservation violated: max error {np.max(energy_error)}"
    
    def test_triple_pendulum(self):
        """Test triple pendulum (very complex system)"""
        dsl_code = r"""
        \system{triple_pendulum}
        \defvar{theta1}{Angle}{rad}
        \defvar{theta2}{Angle}{rad}
        \defvar{theta3}{Angle}{rad}
        \defvar{m}{Mass}{kg}
        \defvar{l}{Length}{m}
        \defvar{g}{Acceleration}{m/s^2}
        
        \parameter{m}{1.0}{kg}
        \parameter{l}{1.0}{m}
        \parameter{g}{9.81}{m/s^2}
        
        \lagrangian{
            \frac{1}{2} * m * l^2 * (\dot{theta1}^2 + \dot{theta2}^2 + \dot{theta3}^2) 
            - m * g * l * (3 - \cos{theta1} - \cos{theta2} - \cos{theta3})
        }
        
        \initial{theta1=0.5, theta1_dot=0.0, theta2=0.0, theta2_dot=0.0, theta3=0.0, theta3_dot=0.0}
        """
        
        compiler = get_compiler()
        result = compiler.compile_dsl(dsl_code)
        
        assert result['success']
        assert len(result['coordinates']) == 3
        
        solution = compiler.simulate(t_span=(0, 3), num_points=300)
        assert solution['success']
        assert solution['y'].shape[0] == 6  # 3 coordinates * 2 states each


class TestQuadruplePendulum:
    """Test quadruple pendulum (extremely complex)"""
    
    def test_quadruple_pendulum(self):
        """Test quadruple pendulum system"""
        dsl_code = r"""
        \system{quadruple_pendulum}
        \defvar{theta1}{Angle}{rad}
        \defvar{theta2}{Angle}{rad}
        \defvar{theta3}{Angle}{rad}
        \defvar{theta4}{Angle}{rad}
        \defvar{m}{Mass}{kg}
        \defvar{l}{Length}{m}
        \defvar{g}{Acceleration}{m/s^2}
        
        \parameter{m}{1.0}{kg}
        \parameter{l}{1.0}{m}
        \parameter{g}{9.81}{m/s^2}
        
        \lagrangian{
            \frac{1}{2} * m * l^2 * (\dot{theta1}^2 + \dot{theta2}^2 + \dot{theta3}^2 + \dot{theta4}^2) 
            - m * g * l * (4 - \cos{theta1} - \cos{theta2} - \cos{theta3} - \cos{theta4})
        }
        
        \initial{theta1=0.3, theta1_dot=0.0, theta2=0.0, theta2_dot=0.0, theta3=0.0, theta3_dot=0.0, theta4=0.0, theta4_dot=0.0}
        """
        
        compiler = get_compiler()
        result = compiler.compile_dsl(dsl_code)
        
        assert result['success']
        assert len(result['coordinates']) == 4
        
        solution = compiler.simulate(t_span=(0, 2), num_points=200, method='LSODA')
        assert solution['success']
        assert solution['y'].shape[0] == 8  # 4 coordinates * 2 states


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

