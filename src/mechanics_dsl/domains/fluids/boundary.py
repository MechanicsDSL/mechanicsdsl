"""
Boundary Conditions for Fluid Simulations

Provides various boundary condition implementations for SPH and grid-based
fluid simulations.
"""
from typing import List, Dict, Tuple
import numpy as np


class BoundaryConditions:
    """
    Boundary condition handler for fluid simulations.
    
    Supports:
    - Solid wall (no-slip, free-slip)
    - Periodic boundaries
    - Open boundaries
    """
    
    def __init__(self, domain_min: Tuple[float, float] = (0.0, 0.0),
                 domain_max: Tuple[float, float] = (1.0, 1.0)):
        self.domain_min = np.array(domain_min)
        self.domain_max = np.array(domain_max)
        self.walls: List[Dict] = []
    
    def add_wall(self, x1: float, y1: float, x2: float, y2: float,
                 wall_type: str = 'no_slip') -> None:
        """
        Add a wall boundary segment.
        
        Args:
            x1, y1, x2, y2: Wall endpoints
            wall_type: 'no_slip', 'free_slip', or 'open'
        """
        self.walls.append({
            'start': np.array([x1, y1]),
            'end': np.array([x2, y2]),
            'type': wall_type,
            'normal': self._compute_normal(x1, y1, x2, y2)
        })
    
    def _compute_normal(self, x1: float, y1: float, 
                        x2: float, y2: float) -> np.ndarray:
        """Compute outward normal of wall segment."""
        tangent = np.array([x2 - x1, y2 - y1])
        tangent = tangent / np.linalg.norm(tangent)
        # Rotate 90 degrees for normal
        return np.array([-tangent[1], tangent[0]])
    
    def enforce_box_boundary(self, position: np.ndarray, velocity: np.ndarray,
                            restitution: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Enforce box domain boundaries with reflection.
        
        Args:
            position: Particle position [x, y]
            velocity: Particle velocity [vx, vy]
            restitution: Coefficient of restitution (0-1)
            
        Returns:
            Updated (position, velocity)
        """
        pos = position.copy()
        vel = velocity.copy()
        
        # Check each dimension
        for d in range(2):
            if pos[d] < self.domain_min[d]:
                pos[d] = self.domain_min[d]
                vel[d] = -restitution * vel[d]
            elif pos[d] > self.domain_max[d]:
                pos[d] = self.domain_max[d]
                vel[d] = -restitution * vel[d]
        
        return pos, vel
    
    def enforce_periodic(self, position: np.ndarray) -> np.ndarray:
        """
        Enforce periodic boundary conditions.
        
        Args:
            position: Particle position
            
        Returns:
            Wrapped position
        """
        pos = position.copy()
        domain_size = self.domain_max - self.domain_min
        
        for d in range(2):
            while pos[d] < self.domain_min[d]:
                pos[d] += domain_size[d]
            while pos[d] > self.domain_max[d]:
                pos[d] -= domain_size[d]
        
        return pos
    
    def generate_boundary_particles(self, spacing: float) -> List[Dict]:
        """
        Generate boundary particles along walls.
        
        Args:
            spacing: Distance between boundary particles
            
        Returns:
            List of boundary particle dictionaries
        """
        particles = []
        
        for wall in self.walls:
            start = wall['start']
            end = wall['end']
            length = np.linalg.norm(end - start)
            n_particles = int(length / spacing) + 1
            
            for i in range(n_particles):
                t = i / max(n_particles - 1, 1)
                pos = start + t * (end - start)
                particles.append({
                    'x': pos[0], 'y': pos[1],
                    'vx': 0.0, 'vy': 0.0,
                    'mass': 1000.0,  # Large mass for boundary
                    'type': 'boundary'
                })
        
        return particles
