"""
Smoothed Particle Hydrodynamics (SPH) Implementation

Provides particle-based fluid simulation using SPH methods.
"""
from typing import Dict, List, Tuple, Optional
import numpy as np

from ...utils import logger


class SPHFluid:
    """
    Smoothed Particle Hydrodynamics fluid simulation.
    
    Uses particle-based methods to simulate incompressible fluids
    with pressure, viscosity, and gravity forces.
    
    Attributes:
        particles: List of particle dictionaries with position, velocity, mass
        smoothing_length: Kernel smoothing length 'h'
        rest_density: Rest density for equation of state
        gas_constant: Stiffness parameter for pressure
        viscosity: Dynamic viscosity coefficient
    """
    
    def __init__(self, 
                 smoothing_length: float = 0.1,
                 rest_density: float = 1000.0,
                 gas_constant: float = 2000.0,
                 viscosity: float = 1.0):
        self.particles: List[Dict] = []
        self.boundary_particles: List[Dict] = []
        self.smoothing_length = smoothing_length
        self.rest_density = rest_density
        self.gas_constant = gas_constant
        self.viscosity = viscosity
        self.gravity = np.array([0.0, -9.81])
        
    def add_particle(self, x: float, y: float, mass: float = 1.0,
                    vx: float = 0.0, vy: float = 0.0,
                    particle_type: str = 'fluid') -> None:
        """Add a particle to the simulation."""
        particle = {
            'x': x, 'y': y,
            'vx': vx, 'vy': vy,
            'mass': mass,
            'density': self.rest_density,
            'pressure': 0.0,
            'type': particle_type
        }
        if particle_type == 'boundary':
            self.boundary_particles.append(particle)
        else:
            self.particles.append(particle)
    
    def kernel_poly6(self, r: float, h: float) -> float:
        """Poly6 smoothing kernel."""
        if r > h:
            return 0.0
        factor = 315.0 / (64.0 * np.pi * h**9)
        return factor * (h**2 - r**2)**3
    
    def kernel_spiky_grad(self, r_vec: np.ndarray, h: float) -> np.ndarray:
        """Gradient of spiky kernel for pressure forces."""
        r = np.linalg.norm(r_vec)
        if r > h or r < 1e-8:
            return np.zeros(2)
        factor = -45.0 / (np.pi * h**6)
        return factor * (h - r)**2 * (r_vec / r)
    
    def kernel_viscosity_laplacian(self, r: float, h: float) -> float:
        """Laplacian of viscosity kernel."""
        if r > h:
            return 0.0
        return 45.0 / (np.pi * h**6) * (h - r)
    
    def compute_density_pressure(self) -> None:
        """Compute density and pressure for all particles."""
        h = self.smoothing_length
        all_particles = self.particles + self.boundary_particles
        
        for p in self.particles:
            density = 0.0
            pos_i = np.array([p['x'], p['y']])
            
            for q in all_particles:
                pos_j = np.array([q['x'], q['y']])
                r = np.linalg.norm(pos_i - pos_j)
                density += q['mass'] * self.kernel_poly6(r, h)
            
            p['density'] = max(density, self.rest_density * 0.1)
            # Tait equation of state
            p['pressure'] = self.gas_constant * (p['density'] - self.rest_density)
    
    def compute_forces(self) -> List[np.ndarray]:
        """Compute pressure, viscosity, and gravity forces."""
        h = self.smoothing_length
        forces = []
        all_particles = self.particles + self.boundary_particles
        
        for p in self.particles:
            force = np.zeros(2)
            pos_i = np.array([p['x'], p['y']])
            vel_i = np.array([p['vx'], p['vy']])
            
            # Pressure and viscosity from neighbors
            for q in all_particles:
                if p is q:
                    continue
                    
                pos_j = np.array([q['x'], q['y']])
                r_vec = pos_i - pos_j
                r = np.linalg.norm(r_vec)
                
                if r < h and r > 1e-8:
                    vel_j = np.array([q['vx'], q['vy']])
                    
                    # Pressure force (symmetric)
                    pressure_term = (p['pressure'] + q['pressure']) / (2 * q['density'])
                    force -= q['mass'] * pressure_term * self.kernel_spiky_grad(r_vec, h)
                    
                    # Viscosity force
                    visc_term = self.viscosity * q['mass'] * (vel_j - vel_i) / q['density']
                    force += visc_term * self.kernel_viscosity_laplacian(r, h)
            
            # Gravity
            force += p['mass'] * self.gravity
            
            forces.append(force)
        
        return forces
    
    def step(self, dt: float) -> None:
        """Advance simulation by one timestep."""
        self.compute_density_pressure()
        forces = self.compute_forces()
        
        # Semi-implicit Euler integration
        for i, p in enumerate(self.particles):
            acc = forces[i] / p['mass']
            p['vx'] += acc[0] * dt
            p['vy'] += acc[1] * dt
            p['x'] += p['vx'] * dt
            p['y'] += p['vy'] * dt
    
    def get_positions(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get all particle positions as arrays."""
        x = np.array([p['x'] for p in self.particles])
        y = np.array([p['y'] for p in self.particles])
        return x, y
