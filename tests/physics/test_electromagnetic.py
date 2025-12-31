"""
Tests for electromagnetic physics domain.
"""
import pytest
import numpy as np
import sympy as sp


class TestChargedParticle:
    """Test ChargedParticle dynamics."""
    
    def test_creation(self):
        """Test basic creation."""
        from mechanics_dsl.domains.electromagnetic import ChargedParticle
        
        particle = ChargedParticle(mass=9.1e-31, charge=1.6e-19)
        assert particle.mass == 9.1e-31
        assert particle.charge == 1.6e-19
        assert particle.coordinates == ['x', 'y', 'z']
    
    def test_cyclotron_frequency(self):
        """Test cyclotron frequency calculation."""
        from mechanics_dsl.domains.electromagnetic import ChargedParticle
        
        particle = ChargedParticle(mass=1.0, charge=1.0)
        omega_c = particle.cyclotron_frequency(B_magnitude=2.0)
        assert omega_c == 2.0  # ωc = qB/m = 1*2/1
    
    def test_larmor_radius(self):
        """Test Larmor radius calculation."""
        from mechanics_dsl.domains.electromagnetic import ChargedParticle
        
        particle = ChargedParticle(mass=1.0, charge=1.0)
        r_L = particle.larmor_radius(v_perp=1.0, B_magnitude=2.0)
        assert r_L == 0.5  # rL = mv/(qB) = 1*1/(1*2)
    
    def test_uniform_magnetic_field(self):
        """Test equations of motion in uniform B field."""
        from mechanics_dsl.domains.electromagnetic import ChargedParticle
        
        particle = ChargedParticle(mass=1.0, charge=1.0)
        particle.set_uniform_magnetic_field(Bz=1.0)
        
        eom = particle.derive_equations_of_motion()
        
        # Should have x, y, z accelerations
        assert 'x_ddot' in eom
        assert 'y_ddot' in eom
        assert 'z_ddot' in eom
    
    def test_lagrangian(self):
        """Test Lagrangian construction."""
        from mechanics_dsl.domains.electromagnetic import ChargedParticle
        
        particle = ChargedParticle()
        L = particle.define_lagrangian()
        
        assert L is not None
        assert isinstance(L, sp.Expr)


class TestCyclotronMotion:
    """Test cyclotron motion analysis."""
    
    def test_exact_trajectory(self):
        """Test exact cyclotron trajectory calculation."""
        from mechanics_dsl.domains.electromagnetic import ChargedParticle, CyclotronMotion
        
        particle = ChargedParticle(mass=1.0, charge=1.0)
        cyclotron = CyclotronMotion(particle)
        
        t = np.linspace(0, 10, 100)
        trajectory = cyclotron.exact_trajectory(
            v0=(1.0, 0.0, 0.0),
            r0=(0.0, 0.0, 0.0),
            B=1.0,
            t_array=t
        )
        
        assert 'x' in trajectory
        assert 'y' in trajectory
        assert len(trajectory['x']) == 100
        
        # Trajectory should be circular in x-y plane
        r = np.sqrt(trajectory['x']**2 + trajectory['y']**2)
        # Allow small numerical variation
        assert np.std(r) < 0.1  # Approximately constant radius


class TestDipoleTrap:
    """Test magnetic dipole trap."""
    
    def test_mirror_ratio(self):
        """Test mirror ratio calculation."""
        from mechanics_dsl.domains.electromagnetic import DipoleTrap
        
        trap = DipoleTrap(B0=1.0, L=1.0)
        R = trap.mirror_ratio(z_mirror=1.0)
        
        # B(1) = B0(1 + 1) = 2, so R = 2
        assert R == 2.0
    
    def test_loss_cone(self):
        """Test loss cone angle."""
        from mechanics_dsl.domains.electromagnetic import DipoleTrap
        
        trap = DipoleTrap(B0=1.0, L=1.0)
        theta_loss = trap.loss_cone_angle(z_mirror=1.0)
        
        # sin²(θ) = 1/R = 0.5, so sin(θ) = 1/√2
        expected = np.arcsin(1.0 / np.sqrt(2))
        assert np.isclose(theta_loss, expected)


class TestConvenienceFunctions:
    """Test module-level convenience functions."""
    
    def test_crossed_fields(self):
        """Test crossed fields configuration."""
        from mechanics_dsl.domains.electromagnetic import uniform_crossed_fields
        
        particle = uniform_crossed_fields(E=100.0, B=1.0)
        assert particle is not None
    
    def test_drift_velocity(self):
        """Test E×B drift velocity."""
        from mechanics_dsl.domains.electromagnetic import calculate_drift_velocity
        
        v_drift = calculate_drift_velocity(E=100.0, B=1.0)
        assert v_drift == 100.0
