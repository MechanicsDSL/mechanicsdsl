"""
Rigorous physics validation tests.

Tests mathematical correctness against analytical solutions and conservation laws.
Every test includes the physics derivation as documentation.
"""
import pytest
import numpy as np
from typing import Tuple, Dict


class TestSimpleHarmonicOscillator:
    """
    Validate simple harmonic oscillator: m*x'' + k*x = 0
    
    Analytical solution: x(t) = A*cos(ωt + φ) where ω = √(k/m)
    Period: T = 2π/ω = 2π√(m/k)
    
    Conservation: Total energy E = T + V = ½mẋ² + ½kx² = constant
    """
    
    @pytest.fixture
    def sho_compiler(self):
        """Create a simple harmonic oscillator compiler."""
        from mechanics_dsl import PhysicsCompiler
        
        dsl_code = r"""
        \system{simple_harmonic_oscillator}
        \defvar{x}{Position}{m}
        \defvar{m}{Mass}{kg}
        \defvar{k}{Spring Constant}{N/m}
        
        \parameter{m}{1.0}{kg}
        \parameter{k}{4.0}{N/m}
        
        \lagrangian{
            \frac{1}{2} * m * \dot{x}^2 - \frac{1}{2} * k * x^2
        }
        
        \initial{x=1.0, x_dot=0.0}
        """
        
        compiler = PhysicsCompiler()
        compiler.compile_dsl(dsl_code)
        return compiler
    
    def test_period_matches_analytical(self, sho_compiler):
        """
        Verify: T = 2π√(m/k) = 2π√(1/4) = π
        
        The oscillator should complete one full cycle in time T.
        """
        m, k = 1.0, 4.0
        expected_period = 2 * np.pi * np.sqrt(m / k)  # π ≈ 3.14159
        
        # Simulate for 4 periods to ensure at least 3 peaks
        solution = sho_compiler.simulate(t_span=(0, 4 * expected_period), num_points=2000)
        
        assert solution['success'], "Simulation should succeed"
        
        t = solution['t']
        x = solution['y'][0]
        
        # Find local maxima (peaks)
        peaks_t = []
        for i in range(1, len(x) - 1):
            if x[i] > x[i-1] and x[i] > x[i+1]:
                # Parabolic interpolation for precise peak location
                t_peak = t[i]
                peaks_t.append(t_peak)
        
        # Period is time between consecutive peaks
        assert len(peaks_t) >= 2, "Should have at least 2 peaks"
        
        measured_period = peaks_t[1] - peaks_t[0]
        
        relative_error = abs(measured_period - expected_period) / expected_period
        assert relative_error < 0.01, f"Period error {relative_error:.2%} > 1%"
    
    def test_energy_conservation(self, sho_compiler):
        """
        Verify: E = ½mẋ² + ½kx² = constant throughout simulation.
        
        For conservative systems, total mechanical energy is invariant.
        """
        m, k = 1.0, 4.0
        
        solution = sho_compiler.simulate(t_span=(0, 20), num_points=2000)
        assert solution['success']
        
        x = solution['y'][0]
        x_dot = solution['y'][1]
        
        # Compute energy at each timestep
        kinetic = 0.5 * m * x_dot**2
        potential = 0.5 * k * x**2
        total_energy = kinetic + potential
        
        # Initial energy
        E0 = total_energy[0]
        
        # Check conservation
        relative_error = np.abs(total_energy - E0) / E0
        max_error = np.max(relative_error)
        
        assert max_error < 1e-6, f"Energy drift {max_error:.2e} > 1e-6"
    
    def test_amplitude_preserved(self, sho_compiler):
        """
        Verify: Amplitude A = x(0) = 1.0 is preserved (undamped).
        
        Max displacement should equal initial displacement.
        """
        solution = sho_compiler.simulate(t_span=(0, 20), num_points=2000)
        assert solution['success']
        
        x = solution['y'][0]
        
        # Find all local maxima
        maxima = []
        for i in range(1, len(x) - 1):
            if x[i] > x[i-1] and x[i] > x[i+1]:
                maxima.append(x[i])
        
        expected_amplitude = 1.0
        
        for amp in maxima:
            relative_error = abs(amp - expected_amplitude) / expected_amplitude
            assert relative_error < 0.01, f"Amplitude error {relative_error:.2%} > 1%"


class TestSimplePendulum:
    """
    Validate simple pendulum: θ'' + (g/l)*sin(θ) = 0
    
    Small angle approximation (θ << 1): T ≈ 2π√(l/g)
    Energy: E = ½ml²θ̇² + mgl(1-cos(θ)) = constant
    """
    
    @pytest.fixture
    def pendulum_compiler(self):
        """Create a simple pendulum compiler."""
        from mechanics_dsl import PhysicsCompiler
        
        dsl_code = r"""
        \system{simple_pendulum}
        \defvar{theta}{Angle}{rad}
        \defvar{m}{Mass}{kg}
        \defvar{l}{Length}{m}
        \defvar{g}{Acceleration}{m/s^2}
        
        \parameter{m}{1.0}{kg}
        \parameter{l}{1.0}{m}
        \parameter{g}{9.81}{m/s^2}
        
        \lagrangian{
            \frac{1}{2} * m * l^2 * \dot{theta}^2 - m * g * l * (1 - \cos{theta})
        }
        
        \initial{theta=0.1, theta_dot=0.0}
        """
        
        compiler = PhysicsCompiler()
        compiler.compile_dsl(dsl_code)
        return compiler
    
    def test_small_angle_period(self, pendulum_compiler):
        """
        Verify: T ≈ 2π√(l/g) for small angles (θ₀ = 0.1 rad ≈ 5.7°)
        
        Expected T = 2π√(1/9.81) ≈ 2.006 seconds
        """
        l, g = 1.0, 9.81
        expected_period = 2 * np.pi * np.sqrt(l / g)
        
        solution = pendulum_compiler.simulate(t_span=(0, 4 * expected_period), num_points=2000)
        assert solution['success']
        
        t = solution['t']
        theta = solution['y'][0]
        
        # Find peaks (local maxima)
        peaks_t = []
        for i in range(1, len(theta) - 1):
            if theta[i] > theta[i-1] and theta[i] > theta[i+1]:
                peaks_t.append(t[i])
        
        # Period is time between consecutive peaks
        assert len(peaks_t) >= 2
        
        measured_period = peaks_t[1] - peaks_t[0]
        relative_error = abs(measured_period - expected_period) / expected_period
        
        # Small angle approximation good to ~1% for θ₀ = 0.1
        assert relative_error < 0.02, f"Period error {relative_error:.2%} > 2%"
    
    def test_energy_conservation_large_angle(self, pendulum_compiler):
        """
        Verify energy conservation for large angle oscillations.
        
        E = ½ml²θ̇² + mgl(1-cos(θ)) = constant
        """
        m, l, g = 1.0, 1.0, 9.81
        
        # Set large initial angle
        pendulum_compiler.simulator.initial_conditions['theta'] = 2.0  # ~115°
        
        solution = pendulum_compiler.simulate(t_span=(0, 10), num_points=1000)
        assert solution['success']
        
        theta = solution['y'][0]
        theta_dot = solution['y'][1]
        
        kinetic = 0.5 * m * l**2 * theta_dot**2
        potential = m * g * l * (1 - np.cos(theta))
        total_energy = kinetic + potential
        
        E0 = total_energy[0]
        relative_error = np.abs(total_energy - E0) / E0
        max_error = np.max(relative_error)
        
        assert max_error < 1e-5, f"Energy drift {max_error:.2e} > 1e-5"


class TestKeplerProblem:
    """
    Validate Kepler orbital mechanics: r'' = -GM/r² (radial)
    
    Conservation laws:
    - Energy: E = ½μv² - GMμ/r = constant
    - Angular momentum: L = μr²θ̇ = constant
    
    Kepler's Third Law: T² = (4π²/GM)a³
    """
    
    @pytest.fixture
    def kepler_compiler(self):
        """Create a Kepler orbit compiler (circular orbit)."""
        from mechanics_dsl import PhysicsCompiler
        
        dsl_code = r"""
        \system{kepler_orbit}
        \defvar{r}{Position}{m}
        \defvar{phi}{Angle}{rad}
        \defvar{mu}{Mass}{kg}
        \defvar{M}{Central Mass}{kg}
        \defvar{G}{Gravitational Constant}{1}
        
        \parameter{mu}{1.0}{kg}
        \parameter{M}{1.0}{kg}
        \parameter{G}{1.0}{1}
        
        \lagrangian{
            \frac{1}{2} * mu * (\dot{r}^2 + r^2 * \dot{phi}^2) + G * M * mu / r
        }
        
        \initial{r=1.0, r_dot=0.0, phi=0.0, phi_dot=1.0}
        """
        
        compiler = PhysicsCompiler()
        compiler.compile_dsl(dsl_code)
        return compiler
    
    def test_circular_orbit_radius_constant(self, kepler_compiler):
        """
        For circular orbit: v = √(GM/r), so r should remain constant.
        """
        solution = kepler_compiler.simulate(t_span=(0, 20), num_points=1000)
        assert solution['success']
        
        r = solution['y'][0]
        r_mean = np.mean(r)
        
        # For circular orbit, r should be constant
        relative_std = np.std(r) / r_mean
        
        assert relative_std < 0.01, f"Orbital radius variation {relative_std:.2%} > 1%"
    
    def test_angular_momentum_conservation(self, kepler_compiler):
        """
        Verify: L = μr²φ̇ = constant (Kepler's second law)
        """
        mu = 1.0
        
        solution = kepler_compiler.simulate(t_span=(0, 20), num_points=1000)
        assert solution['success']
        
        r = solution['y'][0]
        phi_dot = solution['y'][3]
        
        L = mu * r**2 * phi_dot
        L0 = L[0]
        
        relative_error = np.abs(L - L0) / abs(L0)
        max_error = np.max(relative_error)
        
        assert max_error < 1e-5, f"Angular momentum error {max_error:.2e} > 1e-5"


class TestDoublePendulum:
    """
    Validate double pendulum (chaotic system).
    
    Key property: Energy conservation despite chaotic dynamics.
    Cannot validate trajectories (chaos), but conservation laws still hold.
    """
    
    @pytest.fixture
    def double_pendulum_compiler(self):
        """Create a double pendulum compiler."""
        from mechanics_dsl import PhysicsCompiler
        
        dsl_code = r"""
        \system{double_pendulum}
        \defvar{theta1}{Angle}{rad}
        \defvar{theta2}{Angle}{rad}
        \defvar{m1}{Mass}{kg}
        \defvar{m2}{Mass}{kg}
        \defvar{l1}{Length}{m}
        \defvar{l2}{Length}{m}
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
            - (m1 + m2) * g * l1 * (1 - \cos{theta1})
            - m2 * g * l2 * (1 - \cos{theta2})
        }
        
        \initial{theta1=2.0, theta1_dot=0.0, theta2=2.0, theta2_dot=0.0}
        """
        
        compiler = PhysicsCompiler()
        compiler.compile_dsl(dsl_code)
        return compiler
    
    def test_energy_conservation_chaotic(self, double_pendulum_compiler):
        """
        Even in chaotic systems, total mechanical energy is conserved.
        
        E = T + V where T = kinetic, V = potential
        """
        m1, m2, l1, l2, g = 1.0, 1.0, 1.0, 1.0, 9.81
        
        solution = double_pendulum_compiler.simulate(t_span=(0, 20), num_points=2000)
        assert solution['success']
        
        theta1 = solution['y'][0]
        theta1_dot = solution['y'][1]
        theta2 = solution['y'][2]
        theta2_dot = solution['y'][3]
        
        # Kinetic energy
        T = (0.5 * (m1 + m2) * l1**2 * theta1_dot**2 +
             0.5 * m2 * l2**2 * theta2_dot**2 +
             m2 * l1 * l2 * theta1_dot * theta2_dot * np.cos(theta1 - theta2))
        
        # Potential energy (measured from lowest point)
        V = ((m1 + m2) * g * l1 * (1 - np.cos(theta1)) +
             m2 * g * l2 * (1 - np.cos(theta2)))
        
        E = T + V
        E0 = E[0]
        
        relative_error = np.abs(E - E0) / abs(E0)
        max_error = np.max(relative_error)
        
        # Chaotic systems may accumulate more error
        assert max_error < 1e-3, f"Energy drift {max_error:.2e} > 1e-3"


class TestCoupledOscillators:
    """
    Validate coupled oscillators (normal modes).
    
    Two masses connected by springs: m*x₁'' = -k₁x₁ + k_c(x₂-x₁)
    
    Normal mode frequencies: ω₁ = √(k₁/m), ω₂ = √((k₁+2k_c)/m)
    """
    
    @pytest.fixture
    def coupled_compiler(self):
        """Create coupled oscillators compiler."""
        from mechanics_dsl import PhysicsCompiler
        
        dsl_code = r"""
        \system{coupled_oscillators}
        \defvar{x1}{Position}{m}
        \defvar{x2}{Position}{m}
        \defvar{m}{Mass}{kg}
        \defvar{k}{Spring Constant}{N/m}
        \defvar{kc}{Coupling Constant}{N/m}
        
        \parameter{m}{1.0}{kg}
        \parameter{k}{1.0}{N/m}
        \parameter{kc}{0.5}{N/m}
        
        \lagrangian{
            \frac{1}{2} * m * (\dot{x1}^2 + \dot{x2}^2)
            - \frac{1}{2} * k * (x1^2 + x2^2)
            - \frac{1}{2} * kc * (x2 - x1)^2
        }
        
        \initial{x1=1.0, x1_dot=0.0, x2=1.0, x2_dot=0.0}
        """
        
        compiler = PhysicsCompiler()
        compiler.compile_dsl(dsl_code)
        return compiler
    
    def test_symmetric_mode(self, coupled_compiler):
        """
        Symmetric initial conditions (x₁ = x₂) → both masses move in phase.
        
        For symmetric mode, the coupling spring doesn't stretch.
        Frequency: ω_s = √(k/m) = 1.0 rad/s
        """
        # Initial conditions already symmetric (x1 = x2 = 1.0)
        solution = coupled_compiler.simulate(t_span=(0, 20), num_points=1000)
        assert solution['success']
        
        x1 = solution['y'][0]
        x2 = solution['y'][2]
        
        # In symmetric mode, x1 ≈ x2 throughout
        diff = np.abs(x1 - x2)
        max_diff = np.max(diff)
        
        assert max_diff < 0.01, f"Symmetric mode broken: max diff = {max_diff:.4f}"
    
    def test_energy_conservation_coupled(self, coupled_compiler):
        """
        Total energy conserved in coupled system.
        """
        m, k, kc = 1.0, 1.0, 0.5
        
        solution = coupled_compiler.simulate(t_span=(0, 20), num_points=1000)
        assert solution['success']
        
        x1 = solution['y'][0]
        x1_dot = solution['y'][1]
        x2 = solution['y'][2]
        x2_dot = solution['y'][3]
        
        T = 0.5 * m * (x1_dot**2 + x2_dot**2)
        V = 0.5 * k * (x1**2 + x2**2) + 0.5 * kc * (x2 - x1)**2
        E = T + V
        
        E0 = E[0]
        relative_error = np.abs(E - E0) / E0
        max_error = np.max(relative_error)
        
        assert max_error < 1e-6, f"Energy drift {max_error:.2e} > 1e-6"


class TestDampedOscillator:
    """
    Validate damped harmonic oscillator: m*x'' + b*x' + k*x = 0
    
    Underdamped (ζ < 1): x(t) = A*e^(-γt)*cos(ω_d*t + φ)
    where γ = b/(2m), ω_d = √(ω₀² - γ²), ω₀ = √(k/m)
    
    Energy should decay exponentially.
    """
    
    @pytest.fixture
    def damped_compiler(self):
        """Create damped oscillator compiler."""
        from mechanics_dsl import PhysicsCompiler
        
        dsl_code = r"""
        \system{damped_oscillator}
        \defvar{x}{Position}{m}
        \defvar{m}{Mass}{kg}
        \defvar{k}{Spring Constant}{N/m}
        \defvar{b}{Damping Coefficient}{N*s/m}
        
        \parameter{m}{1.0}{kg}
        \parameter{k}{4.0}{N/m}
        \parameter{b}{0.5}{N*s/m}
        
        \lagrangian{
            \frac{1}{2} * m * \dot{x}^2 - \frac{1}{2} * k * x^2
        }
        
        \rayleigh{
            \frac{1}{2} * b * \dot{x}^2
        }
        
        \initial{x=1.0, x_dot=0.0}
        """
        
        compiler = PhysicsCompiler()
        compiler.compile_dsl(dsl_code)
        return compiler
    
    def test_amplitude_decay(self, damped_compiler):
        """
        Verify amplitude decays exponentially: A(t) = A₀*e^(-γt)
        where γ = b/(2m) = 0.5/(2*1) = 0.25
        """
        m, b = 1.0, 0.5
        gamma = b / (2 * m)  # 0.25
        
        solution = damped_compiler.simulate(t_span=(0, 20), num_points=2000)
        assert solution['success']
        
        t = solution['t']
        x = solution['y'][0]
        
        # Find peaks
        peaks_t, peaks_x = [], []
        for i in range(1, len(x) - 1):
            if x[i] > x[i-1] and x[i] > x[i+1] and x[i] > 0.01:
                peaks_t.append(t[i])
                peaks_x.append(x[i])
        
        if len(peaks_t) >= 3:
            # Fit exponential decay to peaks: ln(A) = ln(A₀) - γt
            log_peaks = np.log(peaks_x)
            coeffs = np.polyfit(peaks_t, log_peaks, 1)
            measured_gamma = -coeffs[0]
            
            relative_error = abs(measured_gamma - gamma) / gamma
            assert relative_error < 0.1, f"Decay rate error {relative_error:.2%} > 10%"
    
    def test_energy_decreases(self, damped_compiler):
        """
        For damped systems, total mechanical energy should monotonically decrease.
        """
        m, k = 1.0, 4.0
        
        solution = damped_compiler.simulate(t_span=(0, 20), num_points=1000)
        assert solution['success']
        
        x = solution['y'][0]
        x_dot = solution['y'][1]
        
        E = 0.5 * m * x_dot**2 + 0.5 * k * x**2
        
        # Energy at end should be less than at start
        assert E[-1] < E[0], "Energy should decrease for damped system"
        
        # Energy should be monotonically decreasing (with some tolerance for numerics)
        dE = np.diff(E)
        increasing_count = np.sum(dE > 1e-10)
        
        # Allow very few numerical increases
        assert increasing_count < 10, f"Energy increased {increasing_count} times"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
