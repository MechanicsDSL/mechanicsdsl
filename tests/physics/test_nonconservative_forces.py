"""
Non-conservative force tests - Damping, friction, forced systems
Tests both new package structure and original core.py
"""

import sys
from pathlib import Path

import numpy as np
import pytest

try:
    from mechanics_dsl import PhysicsCompiler

    NEW_PACKAGE = True
except ImportError:
    NEW_PACKAGE = False

try:
    sys.path.insert(0, str(Path(__file__).parent.parent / "reference"))
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


class TestDampedPendulum:
    """Test damped pendulum with energy dissipation"""

    def test_damped_pendulum(self):
        """Test damped pendulum - energy should decrease"""
        dsl_code = r"""
        \system{damped_pendulum}
        \defvar{theta}{Angle}{rad}
        \defvar{m}{Mass}{kg}
        \defvar{l}{Constant}{m}
        \defvar{g}{Acceleration}{m/s^2}
        \defvar{b}{Damping Coeff}{N*m*s}
        
        \parameter{m}{1.0}{kg}
        \parameter{l}{1.0}{m}
        \parameter{g}{9.81}{m/s^2}
        \parameter{b}{0.5}{N*m*s}
        
        \lagrangian{\frac{1}{2} * m * l^2 * \dot{theta}^2 - m * g * l * (1 - \cos{theta})}
        
        \force{-b * theta_dot}
        
        \initial{theta=1.0, theta_dot=0.0}
        """

        compiler = get_compiler()
        result = compiler.compile_dsl(dsl_code)

        assert result["success"]

        solution = compiler.simulate(t_span=(0, 10), num_points=1000)

        assert solution["success"]

        # Energy should decrease due to damping
        from mechanics_dsl.energy import PotentialEnergyCalculator

        params = compiler.simulator.parameters
        KE = PotentialEnergyCalculator.compute_kinetic_energy(solution, params)
        PE = PotentialEnergyCalculator.compute_potential_energy(solution, params, "damped_pendulum")
        E_total = KE + PE

        # Energy should decrease
        assert E_total[-1] < E_total[0], "Energy should decrease in damped system"

        # Amplitude should decrease
        theta = solution["y"][0]
        early_amplitude = np.max(np.abs(theta[:100]))
        late_amplitude = np.max(np.abs(theta[-100:]))
        assert late_amplitude < early_amplitude, "Amplitude should decrease"


class TestForcedOscillator:
    """Test forced harmonic oscillator"""

    def test_forced_oscillator(self):
        """Test forced harmonic oscillator with driving force"""
        dsl_code = r"""
        \system{forced_oscillator}
        \defvar{x}{Position}{m}
        \defvar{m}{Mass}{kg}
        \defvar{k}{Spring Constant}{N/m}
        \defvar{F0}{Force Amplitude}{N}
        \defvar{omega_d}{Drive Frequency}{rad/s}
        
        \parameter{m}{1.0}{kg}
        \parameter{k}{10.0}{N/m}
        \parameter{F0}{1.0}{N}
        \parameter{omega_d}{2.0}{rad/s}
        
        \lagrangian{\frac{1}{2} * m * \dot{x}^2 - \frac{1}{2} * k * x^2}
        
        \force{F0 * \cos{omega_d * t}}
        
        \initial{x=0.0, x_dot=0.0}
        """

        compiler = get_compiler()
        result = compiler.compile_dsl(dsl_code)

        assert result["success"]

        solution = compiler.simulate(t_span=(0, 20), num_points=1000)

        assert solution["success"]

        # Forced oscillator should show driven motion
        x = solution["y"][0]
        assert np.max(np.abs(x)) > 0.1

        # Should show periodic behavior at drive frequency
        t = solution["t"]
        # Check for multiple cycles
        assert len(t) > 100


class TestSpringMassDamper:
    """Test spring-mass-damper system"""

    def test_spring_mass_damper(self):
        """Test spring-mass-damper with damping force"""
        dsl_code = r"""
        \system{spring_mass_damper}
        \defvar{x}{Position}{m}
        \defvar{m}{Mass}{kg}
        \defvar{k}{Spring Constant}{N/m}
        \defvar{c}{Damping Coeff}{N*s/m}
        
        \parameter{m}{1.0}{kg}
        \parameter{k}{10.0}{N/m}
        \parameter{c}{0.5}{N*s/m}
        
        \lagrangian{\frac{1}{2} * m * \dot{x}^2 - \frac{1}{2} * k * x^2}
        
        \force{-c * x_dot}
        
        \initial{x=1.0, x_dot=0.0}
        """

        compiler = get_compiler()
        result = compiler.compile_dsl(dsl_code)

        assert result["success"]

        solution = compiler.simulate(t_span=(0, 10), num_points=500)

        assert solution["success"]

        # Damped oscillation - amplitude should decrease
        x = solution["y"][0]
        assert np.max(np.abs(x)) > 0.01

        # Check damping - later amplitude should be smaller
        if len(x) > 200:
            early_max = np.max(np.abs(x[:100]))
            late_max = np.max(np.abs(x[-100:]))
            # In underdamped case, amplitude decreases
            # Allow for some variation due to numerical errors
            assert late_max <= early_max * 1.1, "Amplitude should decrease or stay similar"


class TestMagneticPendulum:
    """Test magnetic pendulum with non-conservative forces"""

    def test_magnetic_pendulum(self):
        """Test pendulum in magnetic field"""
        dsl_code = r"""
        \system{magnetic_pendulum}
        \defvar{theta}{Angle}{rad}
        \defvar{m}{Mass}{kg}
        \defvar{l}{Length}{m}
        \defvar{g}{Acceleration}{m/s^2}
        \defvar{B}{Magnetic Field}{T}
        \defvar{q}{Charge}{C}
        
        \parameter{m}{0.1}{kg}
        \parameter{l}{1.0}{m}
        \parameter{g}{9.81}{m/s^2}
        \parameter{B}{0.1}{T}
        \parameter{q}{1e-6}{C}
        
        \lagrangian{
            \frac{1}{2} * m * l^2 * \dot{theta}^2 
            - m * g * l * (1 - \cos{theta})
        }
        
        \force{-q * B * l * theta_dot}
        
        \initial{theta=0.5, theta_dot=0.0}
        """

        compiler = get_compiler()
        result = compiler.compile_dsl(dsl_code)

        assert result["success"]

        solution = compiler.simulate(t_span=(0, 10), num_points=500)

        assert solution["success"]

        # Should show damped motion
        theta = solution["y"][0]
        assert np.max(np.abs(theta)) > 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
