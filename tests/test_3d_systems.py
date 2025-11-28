"""
3D system tests - Gyroscope, rigid body, spherical pendulum, 3D elastic pendulum
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


class TestGyroscope:
    """Test gyroscope (3D rotation with Euler angles)"""
    
    def test_gyroscope(self):
        """Test gyroscope with Euler angles"""
        dsl_code = r"""
        \system{gyroscope}
        \defvar{theta}{Angle}{rad}
        \defvar{phi}{Angle}{rad}
        \defvar{psi}{Angle}{rad}
        \defvar{I1}{Moment of Inertia 1}{kg*m^2}
        \defvar{I3}{Moment of Inertia 3}{kg*m^2}
        \defvar{omega}{Spin Rate}{rad/s}
        
        \parameter{I1}{1.0}{kg*m^2}
        \parameter{I3}{0.5}{kg*m^2}
        \parameter{omega}{10.0}{rad/s}
        
        \lagrangian{
            \frac{1}{2} * I1 * (\dot{theta}^2 + \sin{theta}^2 * \dot{phi}^2) 
            + \frac{1}{2} * I3 * (\dot{psi} + \cos{theta} * \dot{phi})^2
        }
        
        \initial{theta=0.1, theta_dot=0.0, phi=0.0, phi_dot=0.0, psi=0.0, psi_dot=omega}
        """
        
        compiler = get_compiler()
        result = compiler.compile_dsl(dsl_code)
        
        assert result['success']
        assert len(result['coordinates']) == 3
        assert 'theta' in result['coordinates']
        assert 'phi' in result['coordinates']
        assert 'psi' in result['coordinates']
        
        solution = compiler.simulate(t_span=(0, 5), num_points=500)
        
        assert solution['success']
        assert solution['y'].shape[0] == 6  # 3 angles * 2 states
        
        # Gyroscope should show precession
        theta = solution['y'][0]
        phi = solution['y'][2]
        psi = solution['y'][4]
        
        assert np.all(np.isfinite(theta))
        assert np.all(np.isfinite(phi))
        assert np.all(np.isfinite(psi))


class TestRigidBody3D:
    """Test rigid body 3D rotation"""
    
    def test_rigid_body_3d(self):
        """Test rigid body with three different moments of inertia"""
        dsl_code = r"""
        \system{rigid_body_3d}
        \defvar{theta}{Angle}{rad}
        \defvar{phi}{Angle}{rad}
        \defvar{psi}{Angle}{rad}
        \defvar{I1}{Moment of Inertia 1}{kg*m^2}
        \defvar{I2}{Moment of Inertia 2}{kg*m^2}
        \defvar{I3}{Moment of Inertia 3}{kg*m^2}
        
        \parameter{I1}{1.0}{kg*m^2}
        \parameter{I2}{0.8}{kg*m^2}
        \parameter{I3}{0.5}{kg*m^2}
        
        \lagrangian{
            \frac{1}{2} * I1 * (\dot{theta}^2 + \sin{theta}^2 * \dot{phi}^2) 
            + \frac{1}{2} * I2 * (\dot{psi}^2 + \cos{theta}^2 * \dot{phi}^2)
            + \frac{1}{2} * I3 * (\dot{phi} + \dot{psi} * \cos{theta})^2
        }
        
        \initial{theta=0.1, theta_dot=0.0, phi=0.0, phi_dot=1.0, psi=0.0, psi_dot=0.0}
        """
        
        compiler = get_compiler()
        result = compiler.compile_dsl(dsl_code)
        
        assert result['success']
        
        solution = compiler.simulate(t_span=(0, 5), num_points=500)
        
        assert solution['success']
        
        # All angles should evolve
        theta = solution['y'][0]
        phi = solution['y'][2]
        psi = solution['y'][4]
        
        assert np.max(np.abs(theta)) > 0.01
        assert np.max(np.abs(phi)) > 0.01


class TestSphericalPendulum:
    """Test spherical pendulum (2D motion in 3D space)"""
    
    def test_spherical_pendulum(self):
        """Test spherical pendulum with two angular coordinates"""
        dsl_code = r"""
        \system{spherical_pendulum}
        \defvar{theta}{Angle}{rad}
        \defvar{phi}{Angle}{rad}
        \defvar{m}{Mass}{kg}
        \defvar{l}{Length}{m}
        \defvar{g}{Acceleration}{m/s^2}
        
        \parameter{m}{1.0}{kg}
        \parameter{l}{1.0}{m}
        \parameter{g}{9.81}{m/s^2}
        
        \lagrangian{
            \frac{1}{2} * m * l^2 * (\dot{theta}^2 + \sin{theta}^2 * \dot{phi}^2) 
            - m * g * l * (1 - \cos{theta})
        }
        
        \initial{theta=0.3, theta_dot=0.0, phi=0.0, phi_dot=0.5}
        """
        
        compiler = get_compiler()
        result = compiler.compile_dsl(dsl_code)
        
        assert result['success']
        assert len(result['coordinates']) == 2
        
        solution = compiler.simulate(t_span=(0, 10), num_points=1000)
        
        assert solution['success']
        
        # Both angles should evolve
        theta = solution['y'][0]
        phi = solution['y'][2]
        
        assert np.max(np.abs(theta)) > 0.01
        assert np.max(np.abs(phi)) > 0.01
        
        # Check energy conservation
        from mechanics_dsl.energy import PotentialEnergyCalculator
        params = compiler.simulator.parameters
        KE = PotentialEnergyCalculator.compute_kinetic_energy(solution, params)
        PE = PotentialEnergyCalculator.compute_potential_energy(solution, params, 'spherical_pendulum')
        E_total = KE + PE
        
        if E_total[0] != 0:
            energy_error = np.abs((E_total - E_total[0]) / E_total[0])
            assert np.max(energy_error) < 0.1


class TestElasticPendulum3D:
    """Test 3D elastic pendulum"""
    
    def test_elastic_pendulum_3d(self):
        """Test elastic pendulum with radial and angular motion"""
        dsl_code = r"""
        \system{elastic_pendulum_3d}
        \defvar{r}{Length}{m}
        \defvar{theta}{Angle}{rad}
        \defvar{phi}{Angle}{rad}
        \defvar{m}{Mass}{kg}
        \defvar{k}{Spring Constant}{N/m}
        \defvar{g}{Acceleration}{m/s^2}
        
        \parameter{m}{1.0}{kg}
        \parameter{k}{10.0}{N/m}
        \parameter{g}{9.81}{m/s^2}
        
        \lagrangian{
            \frac{1}{2} * m * (\dot{r}^2 + r^2 * \dot{theta}^2 + r^2 * \sin{theta}^2 * \dot{phi}^2) 
            - \frac{1}{2} * k * (r - 1.0)^2 
            - m * g * r * \cos{theta}
        }
        
        \initial{r=1.5, r_dot=0.0, theta=0.3, theta_dot=0.0, phi=0.0, phi_dot=0.5}
        """
        
        compiler = get_compiler()
        result = compiler.compile_dsl(dsl_code)
        
        assert result['success']
        assert len(result['coordinates']) == 3
        
        solution = compiler.simulate(t_span=(0, 5), num_points=500)
        
        assert solution['success']
        assert solution['y'].shape[0] == 6  # 3 coordinates * 2 states
        
        # All coordinates should evolve
        r = solution['y'][0]
        theta = solution['y'][2]
        phi = solution['y'][4]
        
        assert np.all(r > 0)  # Radius should stay positive
        assert np.max(np.abs(theta)) > 0.01
        assert np.max(np.abs(phi)) > 0.01


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

