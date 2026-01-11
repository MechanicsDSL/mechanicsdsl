"""
General Relativity Domain for MechanicsDSL

Provides tools for general relativistic calculations, including:
- Schwarzschild spacetime (non-rotating black holes)
- Kerr spacetime (rotating black holes)
- Geodesic equations
- Gravitational lensing
- Cosmological models (FLRW)
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Callable, Union
import numpy as np
from scipy.integrate import solve_ivp
import sympy as sp


# Physical constants
SPEED_OF_LIGHT = 299792458.0  # m/s
GRAVITATIONAL_CONSTANT = 6.674e-11  # m³/(kg·s²)
SOLAR_MASS = 1.989e30  # kg


@dataclass
class SchwarzschildMetric:
    """
    Schwarzschild metric for non-rotating, spherically symmetric black holes.
    
    ds² = -(1 - rs/r)c²dt² + (1 - rs/r)⁻¹dr² + r²dΩ²
    
    where rs = 2GM/c² is the Schwarzschild radius (event horizon).
    
    Example:
        >>> bh = SchwarzschildMetric(mass=10 * SOLAR_MASS)
        >>> rs = bh.schwarzschild_radius()  # ~30 km
        >>> g_tt = bh.metric_component('tt', r=100e3)
    """
    
    mass: float  # Black hole mass (kg)
    c: float = SPEED_OF_LIGHT
    G: float = GRAVITATIONAL_CONSTANT
    
    def schwarzschild_radius(self) -> float:
        """
        Event horizon radius rs = 2GM/c².
        
        Returns:
            Schwarzschild radius (m)
        """
        return 2 * self.G * self.mass / self.c**2
    
    def metric_component(self, component: str, r: float, theta: float = np.pi/2) -> float:
        """
        Get metric tensor component g_μν at given coordinates.
        
        Args:
            component: 'tt', 'rr', 'theta_theta', 'phi_phi'
            r: Radial coordinate (m)
            theta: Polar angle (rad)
            
        Returns:
            Metric component value
        """
        rs = self.schwarzschild_radius()
        
        if r <= rs:
            raise ValueError(f"r={r} is inside event horizon rs={rs}")
        
        f = 1 - rs/r
        
        if component == 'tt':
            return -f * self.c**2
        elif component == 'rr':
            return 1 / f
        elif component == 'theta_theta':
            return r**2
        elif component == 'phi_phi':
            return r**2 * np.sin(theta)**2
        else:
            return 0.0  # Off-diagonal components are zero
    
    def proper_time_factor(self, r: float) -> float:
        """
        Gravitational time dilation factor.
        
        dτ/dt = √(1 - rs/r)
        
        Time runs slower closer to the black hole.
        """
        rs = self.schwarzschild_radius()
        if r <= rs:
            return 0.0
        return np.sqrt(1 - rs/r)
    
    def gravitational_redshift(self, r_emit: float, r_obs: float = np.inf) -> float:
        """
        Gravitational redshift z for light emitted at r_emit.
        
        1 + z = √(g_tt(r_obs)/g_tt(r_emit))
        
        For distant observer (r_obs → ∞):
        1 + z = 1/√(1 - rs/r_emit)
        """
        rs = self.schwarzschild_radius()
        
        if r_emit <= rs:
            return np.inf
        
        if r_obs == np.inf:
            return 1/np.sqrt(1 - rs/r_emit) - 1
        else:
            return np.sqrt((1 - rs/r_obs)/(1 - rs/r_emit)) - 1
    
    def photon_sphere_radius(self) -> float:
        """
        Photon sphere radius where light can orbit the black hole.
        
        r_photon = (3/2) rs = 3GM/c²
        """
        return 1.5 * self.schwarzschild_radius()
    
    def isco_radius(self) -> float:
        """
        Innermost Stable Circular Orbit (ISCO) radius.
        
        r_ISCO = 3 rs = 6GM/c²
        
        Particles inside this radius spiral into the black hole.
        """
        return 3 * self.schwarzschild_radius()
    
    def escape_velocity(self, r: float) -> float:
        """
        Escape velocity at radius r.
        
        v_esc = c √(rs/r)
        
        At the event horizon, v_esc = c.
        """
        rs = self.schwarzschild_radius()
        if r <= rs:
            return self.c
        return self.c * np.sqrt(rs / r)
    
    def orbital_velocity(self, r: float) -> float:
        """
        Circular orbital velocity at radius r.
        
        v_orb = c √(rs / (2r))
        """
        rs = self.schwarzschild_radius()
        if r <= self.isco_radius():
            raise ValueError(f"No stable orbit at r={r}, ISCO={self.isco_radius()}")
        return self.c * np.sqrt(rs / (2 * r))
    
    def orbital_period(self, r: float) -> float:
        """
        Orbital period at radius r (coordinate time).
        
        T = 2π √(r³/(GM))
        """
        return 2 * np.pi * np.sqrt(r**3 / (self.G * self.mass))
    
    def tidal_acceleration(self, r: float, delta_r: float = 1.0) -> float:
        """
        Tidal acceleration (spaghettification) between two points.
        
        Δa ≈ 2GM·Δr/r³
        
        Args:
            r: Radial distance from singularity
            delta_r: Separation between points (default 1m)
        """
        return 2 * self.G * self.mass * delta_r / r**3
    
    def surface_gravity(self) -> float:
        """
        Surface gravity at the event horizon (for Hawking radiation).
        
        κ = c⁴/(4GM) = c²/(2rs)
        """
        return self.c**4 / (4 * self.G * self.mass)
    
    def hawking_temperature(self) -> float:
        """
        Hawking temperature of the black hole.
        
        T_H = ℏc³/(8πGMk_B)
        
        Returns:
            Temperature in Kelvin
        """
        hbar = 1.055e-34  # Reduced Planck constant
        k_B = 1.381e-23   # Boltzmann constant
        
        return hbar * self.c**3 / (8 * np.pi * self.G * self.mass * k_B)


class GeodesicSolver:
    """
    Solver for geodesic equations in curved spacetime.
    
    Geodesics are the paths of freely falling particles (including light).
    
    For Schwarzschild spacetime, uses effective potential method
    for radial motion and full numerical integration for general orbits.
    
    Example:
        >>> solver = GeodesicSolver(SchwarzschildMetric(mass=1e30))
        >>> orbit = solver.solve_orbit(r0=1e7, phi0=0, E=0.95, L=4e6)
    """
    
    def __init__(self, metric: SchwarzschildMetric):
        self.metric = metric
        self.rs = metric.schwarzschild_radius()
        self.c = metric.c
    
    def effective_potential(self, r: float, L: float, 
                           is_massless: bool = False) -> float:
        """
        Effective potential for radial motion.
        
        For massive particles:
        V_eff = -GM/r + L²/(2r²) - GML²/(r³c²)
        
        For photons (L is impact parameter b):
        V_eff = (1 - rs/r)/b²
        
        Args:
            r: Radial coordinate
            L: Angular momentum (or impact parameter for photons)
            is_massless: True for photon geodesics
        """
        rs = self.rs
        
        if is_massless:
            # Photon effective potential
            return (1 - rs/r) / L**2
        else:
            # Massive particle
            GM = self.metric.G * self.metric.mass
            return -GM/r + L**2/(2*r**2) - GM*L**2/(r**3 * self.c**2)
    
    def solve_orbit(self, r0: float, phi0: float, 
                    E: float, L: float,
                    tau_span: Tuple[float, float] = (0, 1e6),
                    is_massless: bool = False) -> Dict:
        """
        Solve geodesic equations numerically.
        
        Args:
            r0: Initial radial coordinate
            phi0: Initial azimuthal angle
            E: Energy parameter (E/mc² for massive, 1 for photons)
            L: Angular momentum
            tau_span: Proper time range
            is_massless: True for null geodesics (light)
            
        Returns:
            Dictionary with tau, r, phi, t arrays
        """
        rs = self.rs
        c = self.c
        
        def equations(tau, y):
            r, phi, t, dr_dtau = y
            
            if r <= rs * 1.01:  # Near horizon
                return [0, 0, 0, 0]
            
            f = 1 - rs/r
            
            # dphi/dτ = L/r²
            dphi_dtau = L / r**2
            
            # dt/dτ = E/f
            dt_dtau = E * c / f
            
            # d²r/dτ² from geodesic equation
            if is_massless:
                d2r_dtau2 = -rs * c**2 / (2 * r**2) + L**2 * (r - 1.5*rs) / r**4
            else:
                d2r_dtau2 = (-rs * c**2 * f / (2 * r**2) + 
                            L**2 * (r - 1.5*rs) / r**4)
            
            return [dr_dtau, dphi_dtau, dt_dtau, d2r_dtau2]
        
        # Initial dr/dτ from energy conservation
        f0 = 1 - rs/r0
        if is_massless:
            dr_dtau0 = c * np.sqrt(max(0, E**2 - L**2 * f0 / r0**2))
        else:
            dr_dtau0 = c * np.sqrt(max(0, E**2 - f0 * (1 + L**2/(r0**2 * c**2))))
        
        y0 = [r0, phi0, 0, dr_dtau0]
        
        # Event to stop at horizon
        def horizon_event(tau, y):
            return y[0] - rs * 1.05
        horizon_event.terminal = True
        
        sol = solve_ivp(equations, tau_span, y0, 
                       method='RK45', 
                       events=horizon_event,
                       max_step=tau_span[1]/1000)
        
        return {
            'tau': sol.t,
            'r': sol.y[0],
            'phi': sol.y[1],
            't': sol.y[2],
            'dr_dtau': sol.y[3]
        }


class KerrMetric:
    """
    Kerr metric for rotating black holes.
    
    Characterized by mass M and angular momentum J (or spin parameter a = J/(Mc)).
    
    Key features:
    - Ergosphere: region where spacetime is dragged
    - Frame dragging: everything rotates with the black hole
    - Two horizons (outer and inner for a < M)
    
    Example:
        >>> kerr = KerrMetric(mass=10*SOLAR_MASS, spin_parameter=0.9)
        >>> r_outer = kerr.outer_horizon()
    """
    
    def __init__(self, mass: float, spin_parameter: float = 0.0,
                 c: float = SPEED_OF_LIGHT, G: float = GRAVITATIONAL_CONSTANT):
        """
        Initialize Kerr black hole.
        
        Args:
            mass: Black hole mass
            spin_parameter: a = J/(Mc), must satisfy 0 ≤ a ≤ M (in geometric units)
        """
        self.mass = mass
        self.c = c
        self.G = G
        
        # Convert to geometric units: M = GM/c²
        self.M_geom = G * mass / c**2
        
        # Spin parameter (dimensionless 0 to 1)
        self.a = spin_parameter * self.M_geom
        
        if abs(self.a) > self.M_geom:
            raise ValueError("Spin parameter too large (naked singularity)")
    
    def outer_horizon(self) -> float:
        """
        Outer event horizon radius.
        
        r+ = M + √(M² - a²)
        """
        M = self.M_geom
        return M + np.sqrt(M**2 - self.a**2)
    
    def inner_horizon(self) -> float:
        """
        Inner (Cauchy) horizon radius.
        
        r- = M - √(M² - a²)
        """
        M = self.M_geom
        return M - np.sqrt(M**2 - self.a**2)
    
    def ergosphere_radius(self, theta: float = np.pi/2) -> float:
        """
        Ergosphere (static limit) radius.
        
        r_ergo = M + √(M² - a²cos²θ)
        
        Inside this radius, nothing can remain stationary.
        """
        M = self.M_geom
        return M + np.sqrt(M**2 - self.a**2 * np.cos(theta)**2)
    
    def frame_dragging_rate(self, r: float, theta: float = np.pi/2) -> float:
        """
        Frame dragging angular velocity.
        
        ω = 2Mar / (Σ² + 2Mr·a²sin²θ)
        
        where Σ² = r² + a²cos²θ
        """
        M = self.M_geom
        a = self.a
        
        Sigma2 = r**2 + a**2 * np.cos(theta)**2
        Delta = r**2 - 2*M*r + a**2
        
        return 2 * M * a * r / (Sigma2**2 + 2*M*r * a**2 * np.sin(theta)**2) * self.c
    
    def isco_radius(self, prograde: bool = True) -> float:
        """
        ISCO radius for Kerr black hole.
        
        Depends on direction of orbit (prograde/retrograde).
        
        For extremal Kerr (a = M): r_ISCO = M (prograde), 9M (retrograde)
        """
        M = self.M_geom
        a = self.a
        
        Z1 = 1 + (1 - a**2/M**2)**(1/3) * ((1 + a/M)**(1/3) + (1 - a/M)**(1/3))
        Z2 = np.sqrt(3 * a**2/M**2 + Z1**2)
        
        if prograde:
            return M * (3 + Z2 - np.sqrt((3 - Z1)*(3 + Z1 + 2*Z2)))
        else:
            return M * (3 + Z2 + np.sqrt((3 - Z1)*(3 + Z1 + 2*Z2)))
    
    def angular_momentum(self) -> float:
        """Total angular momentum J = Mac."""
        return self.a * self.mass * self.c


class GravitationalLensing:
    """
    Gravitational lensing calculations.
    
    Light bending by massive objects:
    - Weak lensing (far from source)
    - Strong lensing (Einstein rings, multiple images)
    - Microlensing
    
    Example:
        >>> lens = GravitationalLensing(mass=SOLAR_MASS)
        >>> angle = lens.deflection_angle(impact_parameter=7e8)  # Sun's radius
    """
    
    def __init__(self, mass: float, c: float = SPEED_OF_LIGHT,
                 G: float = GRAVITATIONAL_CONSTANT):
        self.mass = mass
        self.c = c
        self.G = G
        self.rs = 2 * G * mass / c**2
    
    def deflection_angle(self, impact_parameter: float) -> float:
        """
        Light deflection angle (weak field approximation).
        
        α = 4GM/(c²b) = 2rs/b
        
        For the Sun (b = R_sun): α ≈ 1.75 arcseconds
        
        Args:
            impact_parameter: Closest approach distance (m)
            
        Returns:
            Deflection angle (radians)
        """
        return 4 * self.G * self.mass / (self.c**2 * impact_parameter)
    
    def einstein_radius(self, D_lens: float, D_source: float) -> float:
        """
        Einstein radius for perfect alignment.
        
        θ_E = √(4GM D_ls / (c² D_l D_s))
        
        Args:
            D_lens: Distance to lens
            D_source: Distance to source
            
        Returns:
            Einstein radius (radians)
        """
        D_ls = D_source - D_lens
        return np.sqrt(4 * self.G * self.mass * D_ls / 
                      (self.c**2 * D_lens * D_source))
    
    def magnification(self, u: float) -> float:
        """
        Magnification factor for point-mass lens.
        
        μ = (u² + 2) / (u√(u² + 4))
        
        where u is the source-lens separation in units of Einstein radius.
        
        Args:
            u: Normalized separation
            
        Returns:
            Magnification factor
        """
        return (u**2 + 2) / (u * np.sqrt(u**2 + 4))
    
    def time_delay(self, theta_1: float, theta_2: float, 
                   D_lens: float, D_source: float) -> float:
        """
        Time delay between two lensed images.
        
        For cosmological lensing (Shapiro delay + geometric delay).
        """
        D_ls = D_source - D_lens
        # Simplified formula
        return (1 + D_lens/D_ls) * D_lens * D_source / (self.c * D_ls) * \
               (0.5 * (theta_1**2 - theta_2**2) - 
                self.rs/2 * np.log(abs(theta_1/theta_2)))


class FLRWCosmology:
    """
    Friedmann-Lemaître-Robertson-Walker cosmological model.
    
    Describes a homogeneous, isotropic expanding universe.
    
    ds² = -c²dt² + a(t)²(dr²/(1-kr²) + r²dΩ²)
    
    where a(t) is the scale factor and k is the curvature.
    
    Example:
        >>> cosmos = FLRWCosmology(H0=70, Omega_m=0.3, Omega_Lambda=0.7)
        >>> age = cosmos.age()  # Age of universe
    """
    
    def __init__(self, H0: float = 70.0, 
                 Omega_m: float = 0.3,
                 Omega_Lambda: float = 0.7,
                 Omega_r: float = 0.0,
                 c: float = SPEED_OF_LIGHT):
        """
        Initialize cosmological model.
        
        Args:
            H0: Hubble constant (km/s/Mpc)
            Omega_m: Matter density parameter
            Omega_Lambda: Dark energy density parameter
            Omega_r: Radiation density parameter
            c: Speed of light
        """
        self.H0 = H0 * 1000 / 3.086e22  # Convert to 1/s
        self.Omega_m = Omega_m
        self.Omega_Lambda = Omega_Lambda
        self.Omega_r = Omega_r
        self.Omega_k = 1 - Omega_m - Omega_Lambda - Omega_r
        self.c = c
    
    def hubble_parameter(self, z: float) -> float:
        """
        Hubble parameter at redshift z.
        
        H(z) = H₀ √(Ωᵣ(1+z)⁴ + Ωₘ(1+z)³ + Ωₖ(1+z)² + Ω_Λ)
        """
        return self.H0 * np.sqrt(
            self.Omega_r * (1+z)**4 +
            self.Omega_m * (1+z)**3 +
            self.Omega_k * (1+z)**2 +
            self.Omega_Lambda
        )
    
    def hubble_time(self) -> float:
        """Hubble time t_H = 1/H₀ (seconds)."""
        return 1 / self.H0
    
    def hubble_distance(self) -> float:
        """Hubble distance d_H = c/H₀ (meters)."""
        return self.c / self.H0
    
    def age(self) -> float:
        """
        Age of the universe (approximate).
        
        For flat ΛCDM: t ≈ (2/3H₀) × (1/√Ω_Λ) × arcsinh(√(Ω_Λ/Ωₘ))
        
        Returns:
            Age in seconds
        """
        if self.Omega_Lambda > 0:
            x = np.sqrt(self.Omega_Lambda / self.Omega_m)
            return (2/(3*self.H0)) * (1/np.sqrt(self.Omega_Lambda)) * np.arcsinh(x)
        else:
            return 2 / (3 * self.H0)
    
    def comoving_distance(self, z: float, n_steps: int = 1000) -> float:
        """
        Comoving distance to redshift z.
        
        d_C = c ∫₀ᶻ dz'/H(z')
        """
        from scipy.integrate import quad
        
        def integrand(z_prime):
            return 1 / self.hubble_parameter(z_prime)
        
        result, _ = quad(integrand, 0, z)
        return self.c * result
    
    def luminosity_distance(self, z: float) -> float:
        """
        Luminosity distance d_L = (1+z) d_C.
        """
        return (1 + z) * self.comoving_distance(z)
    
    def angular_diameter_distance(self, z: float) -> float:
        """
        Angular diameter distance d_A = d_C / (1+z).
        """
        return self.comoving_distance(z) / (1 + z)
    
    def lookback_time(self, z: float) -> float:
        """
        Lookback time to redshift z.
        
        t_L = ∫₀ᶻ dz' / ((1+z')H(z'))
        """
        from scipy.integrate import quad
        
        def integrand(z_prime):
            return 1 / ((1 + z_prime) * self.hubble_parameter(z_prime))
        
        result, _ = quad(integrand, 0, z)
        return result
    
    def critical_density(self) -> float:
        """
        Critical density ρ_c = 3H₀²/(8πG).
        """
        return 3 * self.H0**2 / (8 * np.pi * GRAVITATIONAL_CONSTANT)
    
    def deceleration_parameter(self, z: float = 0) -> float:
        """
        Deceleration parameter q.
        
        q = (Ωₘ/2 + Ωᵣ - Ω_Λ) at z=0
        
        Negative q means accelerating expansion.
        """
        return 0.5 * self.Omega_m + self.Omega_r - self.Omega_Lambda


__all__ = [
    'SPEED_OF_LIGHT',
    'GRAVITATIONAL_CONSTANT', 
    'SOLAR_MASS',
    'SchwarzschildMetric',
    'GeodesicSolver',
    'KerrMetric',
    'GravitationalLensing',
    'FLRWCosmology',
]
