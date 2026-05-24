#!/usr/bin/env python3
"""
THE THREE-BODY PROBLEM: Exploring an Unsolved Problem in Physics

The three-body problem has NO general closed-form solution - proven impossible
by Henri Poincare in 1890. This is one of the oldest unsolved problems in physics.

What makes it "unsolved":
1. No formula exists to predict positions for arbitrary initial conditions
2. The system is chaotic - tiny changes cause exponentially diverging trajectories
3. We can only solve NUMERICALLY or for special cases

What this script does:
1. Demonstrates the CHAOS - shows how microscopic differences explode
2. Searches for PERIODIC ORBITS - rare stable solutions that do exist
3. Computes LYAPUNOV EXPONENTS - quantifies the unpredictability
4. Visualizes the beautiful complexity of gravitational three-body dynamics

This is REAL research-grade physics - astronomers use similar methods to study
stellar systems, galaxy formation, and the long-term stability of our solar system.

Author: MechanicsDSL Research Examples
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dataclasses import dataclass
from typing import List, Tuple, Optional
import warnings

# =============================================================================
# CONSTANTS
# =============================================================================

G = 1.0  # Gravitational constant (normalized units)


# =============================================================================
# THREE-BODY PHYSICS ENGINE
# =============================================================================

@dataclass
class Body:
    """A gravitating body with mass, position, and velocity."""
    mass: float
    x: float
    y: float
    vx: float
    vy: float
    name: str = "body"


class ThreeBodySystem:
    """
    Gravitational three-body problem solver.
    
    The equations of motion are:
        m_i * d^2(r_i)/dt^2 = -G * sum_{j != i} m_i * m_j * (r_i - r_j) / |r_i - r_j|^3
    
    This is a 12-dimensional ODE system (3 bodies x 2D x (position + velocity)).
    """
    
    def __init__(self, bodies: List[Body], G: float = 1.0):
        self.bodies = bodies
        self.G = G
        self.masses = np.array([b.mass for b in bodies])
        
    def get_state_vector(self) -> np.ndarray:
        """Pack body states into a single vector: [x1,y1,x2,y2,x3,y3,vx1,vy1,vx2,vy2,vx3,vy3]"""
        state = []
        for b in self.bodies:
            state.extend([b.x, b.y])
        for b in self.bodies:
            state.extend([b.vx, b.vy])
        return np.array(state)
    
    def equations_of_motion(self, t: float, state: np.ndarray) -> np.ndarray:
        """
        Compute derivatives for the ODE system.
        
        Newton's law of gravitation between each pair.
        """
        # Unpack positions
        x1, y1, x2, y2, x3, y3 = state[:6]
        vx1, vy1, vx2, vy2, vx3, vy3 = state[6:]
        
        m1, m2, m3 = self.masses
        
        # Compute distances (with softening to avoid singularities)
        eps = 1e-10
        
        r12 = np.sqrt((x2-x1)**2 + (y2-y1)**2 + eps)
        r13 = np.sqrt((x3-x1)**2 + (y3-y1)**2 + eps)
        r23 = np.sqrt((x3-x2)**2 + (y3-y2)**2 + eps)
        
        # Accelerations from Newton's law
        # Body 1
        ax1 = self.G * m2 * (x2-x1) / r12**3 + self.G * m3 * (x3-x1) / r13**3
        ay1 = self.G * m2 * (y2-y1) / r12**3 + self.G * m3 * (y3-y1) / r13**3
        
        # Body 2
        ax2 = self.G * m1 * (x1-x2) / r12**3 + self.G * m3 * (x3-x2) / r23**3
        ay2 = self.G * m1 * (y1-y2) / r12**3 + self.G * m3 * (y3-y2) / r23**3
        
        # Body 3
        ax3 = self.G * m1 * (x1-x3) / r13**3 + self.G * m2 * (x2-x3) / r23**3
        ay3 = self.G * m1 * (y1-y3) / r13**3 + self.G * m2 * (y2-y3) / r23**3
        
        return np.array([vx1, vy1, vx2, vy2, vx3, vy3, ax1, ay1, ax2, ay2, ax3, ay3])
    
    def compute_energy(self, state: np.ndarray) -> float:
        """Compute total energy (should be conserved)."""
        x1, y1, x2, y2, x3, y3 = state[:6]
        vx1, vy1, vx2, vy2, vx3, vy3 = state[6:]
        m1, m2, m3 = self.masses
        
        # Kinetic energy
        T = 0.5 * (m1*(vx1**2+vy1**2) + m2*(vx2**2+vy2**2) + m3*(vx3**2+vy3**2))
        
        # Potential energy
        r12 = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        r13 = np.sqrt((x3-x1)**2 + (y3-y1)**2)
        r23 = np.sqrt((x3-x2)**2 + (y3-y2)**2)
        
        V = -self.G * (m1*m2/r12 + m1*m3/r13 + m2*m3/r23)
        
        return T + V
    
    def compute_angular_momentum(self, state: np.ndarray) -> float:
        """Compute total angular momentum (should be conserved)."""
        x1, y1, x2, y2, x3, y3 = state[:6]
        vx1, vy1, vx2, vy2, vx3, vy3 = state[6:]
        m1, m2, m3 = self.masses
        
        L = m1*(x1*vy1 - y1*vx1) + m2*(x2*vy2 - y2*vx2) + m3*(x3*vy3 - y3*vx3)
        return L
    
    def simulate(self, t_end: float, n_points: int = 10000, 
                 method: str = 'DOP853') -> dict:
        """
        Integrate the equations of motion.
        
        Uses high-order adaptive method for accuracy.
        """
        y0 = self.get_state_vector()
        t_span = (0, t_end)
        t_eval = np.linspace(0, t_end, n_points)
        
        sol = solve_ivp(
            self.equations_of_motion,
            t_span,
            y0,
            method=method,
            t_eval=t_eval,
            rtol=1e-12,
            atol=1e-14
        )
        
        return {
            't': sol.t,
            'states': sol.y,
            'success': sol.success
        }


# =============================================================================
# FAMOUS PERIODIC ORBITS
# =============================================================================

def get_figure8_initial_conditions() -> List[Body]:
    """
    The famous Figure-8 orbit discovered by Moore (1993) and proven by Chenciner & Montgomery (2000).
    
    This is one of the most beautiful solutions to the three-body problem:
    - Three equal masses chase each other around a figure-8 path
    - It's mathematically proven to be periodic
    - It's also proven to be UNSTABLE (tiny perturbations destroy it)
    """
    # Precise initial conditions from numerical optimization
    x1 = 0.97000436
    y1 = -0.24308753
    vx3 = -0.93240737
    vy3 = -0.86473146
    
    return [
        Body(mass=1.0, x=x1, y=y1, vx=-vx3/2, vy=-vy3/2, name="Body 1"),
        Body(mass=1.0, x=-x1, y=-y1, vx=-vx3/2, vy=-vy3/2, name="Body 2"),
        Body(mass=1.0, x=0, y=0, vx=vx3, vy=vy3, name="Body 3"),
    ]


def get_lagrange_triangle_conditions() -> List[Body]:
    """
    Lagrange's equilateral triangle solution (1772).
    
    Three bodies at vertices of an equilateral triangle, rotating rigidly.
    This was one of the first exact solutions found.
    """
    # Equilateral triangle with radius 1
    angle1, angle2, angle3 = 0, 2*np.pi/3, 4*np.pi/3
    r = 1.0
    omega = 0.5  # Angular velocity
    
    return [
        Body(mass=1.0, x=r*np.cos(angle1), y=r*np.sin(angle1),
             vx=-omega*r*np.sin(angle1), vy=omega*r*np.cos(angle1), name="Body 1"),
        Body(mass=1.0, x=r*np.cos(angle2), y=r*np.sin(angle2),
             vx=-omega*r*np.sin(angle2), vy=omega*r*np.cos(angle2), name="Body 2"),
        Body(mass=1.0, x=r*np.cos(angle3), y=r*np.sin(angle3),
             vx=-omega*r*np.sin(angle3), vy=omega*r*np.cos(angle3), name="Body 3"),
    ]


def get_chaotic_conditions() -> List[Body]:
    """
    Generic initial conditions that lead to chaotic behavior.
    
    This demonstrates WHY the three-body problem is unsolved:
    no formula can predict where these bodies will be in the future.
    """
    return [
        Body(mass=1.0, x=-1.0, y=0.0, vx=0.0, vy=0.5, name="Body 1"),
        Body(mass=1.0, x=1.0, y=0.0, vx=0.0, vy=-0.5, name="Body 2"),
        Body(mass=1.0, x=0.0, y=1.5, vx=0.0, vy=0.0, name="Body 3"),
    ]


# =============================================================================
# CHAOS ANALYSIS: LYAPUNOV EXPONENT
# =============================================================================

def compute_lyapunov_exponent(system: ThreeBodySystem, 
                               perturbation: float = 1e-9,
                               t_end: float = 50,
                               n_samples: int = 100) -> Tuple[float, np.ndarray]:
    """
    Compute the largest Lyapunov exponent - the measure of chaos.
    
    A POSITIVE Lyapunov exponent means the system is chaotic.
    It tells us how fast nearby trajectories diverge:
        separation(t) ~ separation(0) * exp(lambda * t)
    
    For the three-body problem, this is typically positive,
    meaning prediction is fundamentally limited.
    """
    y0 = system.get_state_vector()
    
    # Create perturbed initial conditions
    y0_perturbed = y0.copy()
    y0_perturbed[0] += perturbation  # Tiny change to x1
    
    t_eval = np.linspace(0, t_end, n_samples)
    
    # Simulate both trajectories
    sol1 = solve_ivp(system.equations_of_motion, (0, t_end), y0,
                     method='DOP853', t_eval=t_eval, rtol=1e-12, atol=1e-14)
    sol2 = solve_ivp(system.equations_of_motion, (0, t_end), y0_perturbed,
                     method='DOP853', t_eval=t_eval, rtol=1e-12, atol=1e-14)
    
    # Compute separation over time
    separation = np.sqrt(np.sum((sol1.y - sol2.y)**2, axis=0))
    
    # Lyapunov exponent from exponential fit
    # log(separation) ~ log(eps) + lambda * t
    valid = separation > 0
    if np.sum(valid) > 10:
        t_valid = sol1.t[valid]
        log_sep = np.log(separation[valid])
        # Linear regression
        coeffs = np.polyfit(t_valid[:len(t_valid)//2], log_sep[:len(t_valid)//2], 1)
        lyapunov = coeffs[0]
    else:
        lyapunov = np.nan
    
    return lyapunov, separation


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_trajectories(result: dict, title: str = "Three-Body Trajectories"):
    """Plot the paths of all three bodies."""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    states = result['states']
    x1, y1 = states[0], states[1]
    x2, y2 = states[2], states[3]
    x3, y3 = states[4], states[5]
    
    ax.plot(x1, y1, 'b-', linewidth=0.5, alpha=0.7, label='Body 1')
    ax.plot(x2, y2, 'r-', linewidth=0.5, alpha=0.7, label='Body 2')
    ax.plot(x3, y3, 'g-', linewidth=0.5, alpha=0.7, label='Body 3')
    
    # Mark starting positions
    ax.scatter([x1[0]], [y1[0]], c='blue', s=100, marker='o', zorder=5)
    ax.scatter([x2[0]], [y2[0]], c='red', s=100, marker='o', zorder=5)
    ax.scatter([x3[0]], [y3[0]], c='green', s=100, marker='o', zorder=5)
    
    # Mark ending positions
    ax.scatter([x1[-1]], [y1[-1]], c='blue', s=100, marker='x', zorder=5)
    ax.scatter([x2[-1]], [y2[-1]], c='red', s=100, marker='x', zorder=5)
    ax.scatter([x3[-1]], [y3[-1]], c='green', s=100, marker='x', zorder=5)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    return fig


def plot_chaos_demonstration(system1: ThreeBodySystem, 
                             system2: ThreeBodySystem,
                             t_end: float = 30):
    """
    Show how tiny differences explode over time.
    
    This is the ESSENCE of why the three-body problem is unsolved:
    you cannot predict the future without infinite precision.
    """
    result1 = system1.simulate(t_end, n_points=5000)
    result2 = system2.simulate(t_end, n_points=5000)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Trajectory comparison
    ax1 = axes[0, 0]
    s1, s2 = result1['states'], result2['states']
    
    ax1.plot(s1[0], s1[1], 'b-', linewidth=0.5, alpha=0.7, label='Original')
    ax1.plot(s2[0], s2[1], 'b--', linewidth=0.5, alpha=0.7, label='Perturbed')
    ax1.set_title('Body 1 Trajectory: Original vs Perturbed (1e-10 change)')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.legend()
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    
    # Separation over time
    ax2 = axes[0, 1]
    separation = np.sqrt(np.sum((s1 - s2)**2, axis=0))
    ax2.semilogy(result1['t'], separation, 'k-', linewidth=1)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Separation (log scale)')
    ax2.set_title('Exponential Divergence of Nearby Trajectories')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=1.0, color='r', linestyle='--', label='Macroscopic separation')
    ax2.legend()
    
    # Energy conservation check
    ax3 = axes[1, 0]
    E1 = [system1.compute_energy(s1[:, i]) for i in range(len(result1['t']))]
    E2 = [system2.compute_energy(s2[:, i]) for i in range(len(result2['t']))]
    ax3.plot(result1['t'], np.array(E1) - E1[0], 'b-', label='Original')
    ax3.plot(result2['t'], np.array(E2) - E2[0], 'r--', label='Perturbed')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Energy - E(0)')
    ax3.set_title('Energy Conservation (Should Stay Near Zero)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Position difference for one body
    ax4 = axes[1, 1]
    dx = s1[0] - s2[0]
    dy = s1[1] - s2[1]
    ax4.plot(result1['t'], dx, 'b-', linewidth=0.8, label='dx (Body 1)')
    ax4.plot(result1['t'], dy, 'r-', linewidth=0.8, label='dy (Body 1)')
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Position Difference')
    ax4.set_title('How a 1e-10 Difference Grows')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


# =============================================================================
# MAIN: EXPLORE THE THREE-BODY PROBLEM
# =============================================================================

def main():
    print("=" * 70)
    print("   THE THREE-BODY PROBLEM: Exploring an Unsolved Physics Problem")
    print("=" * 70)
    print()
    print("The three-body problem has NO general analytical solution.")
    print("Henri Poincare proved in 1890 that no closed-form formula exists.")
    print("This script demonstrates WHY it's fundamentally unpredictable.")
    print()
    
    # =========================================================================
    # PART 1: The Famous Figure-8 Orbit
    # =========================================================================
    print("-" * 70)
    print("PART 1: THE FIGURE-8 ORBIT (One of Few Known Periodic Solutions)")
    print("-" * 70)
    print()
    print("Discovered: 1993 (Cris Moore, numerical)")
    print("Proven: 2000 (Chenciner & Montgomery, variational methods)")
    print("Significance: Shows periodic solutions exist, but are RARE and UNSTABLE")
    print()
    
    figure8_bodies = get_figure8_initial_conditions()
    figure8_system = ThreeBodySystem(figure8_bodies)
    
    # Period is approximately 6.3259
    T = 6.3259
    result_figure8 = figure8_system.simulate(2*T, n_points=5000)
    
    # Check periodicity
    initial_state = result_figure8['states'][:, 0]
    final_state = result_figure8['states'][:, -1]
    periodicity_error = np.linalg.norm(initial_state - final_state)
    
    print(f"Period: T = {T:.4f}")
    print(f"Simulated: 2 periods")
    print(f"Return error after 2T: {periodicity_error:.2e}")
    print(f"Energy drift: {abs(figure8_system.compute_energy(final_state) - figure8_system.compute_energy(initial_state)):.2e}")
    print()
    
    fig1 = plot_trajectories(result_figure8, "Figure-8 Orbit: A Rare Periodic Solution")
    plt.savefig('three_body_figure8.png', dpi=150, bbox_inches='tight')
    print("Saved: three_body_figure8.png")
    
    # =========================================================================
    # PART 2: Demonstrating Chaos
    # =========================================================================
    print()
    print("-" * 70)
    print("PART 2: CHAOS - Why the Problem is UNSOLVABLE")
    print("-" * 70)
    print()
    print("We'll show that a 1e-10 change in position leads to")
    print("completely different outcomes. This is DETERMINISTIC chaos.")
    print()
    
    # Create two almost-identical systems
    chaotic_bodies1 = get_chaotic_conditions()
    chaotic_bodies2 = get_chaotic_conditions()
    chaotic_bodies2[0].x += 1e-10  # Tiny, unmeasurable difference
    
    system1 = ThreeBodySystem(chaotic_bodies1)
    system2 = ThreeBodySystem(chaotic_bodies2)
    
    print("Initial difference: 1e-10 (smaller than an atom)")
    print("Simulating both systems...")
    
    fig2 = plot_chaos_demonstration(system1, system2, t_end=40)
    plt.savefig('three_body_chaos.png', dpi=150, bbox_inches='tight')
    print("Saved: three_body_chaos.png")
    
    # =========================================================================
    # PART 3: Lyapunov Exponent - Quantifying Unpredictability
    # =========================================================================
    print()
    print("-" * 70)
    print("PART 3: LYAPUNOV EXPONENT - Measuring the Chaos")
    print("-" * 70)
    print()
    
    lyapunov, separation = compute_lyapunov_exponent(system1, t_end=30)
    
    print(f"Largest Lyapunov exponent: lambda = {lyapunov:.4f}")
    print()
    if lyapunov > 0:
        print("RESULT: lambda > 0 means the system is CHAOTIC")
        print(f"Prediction horizon: errors double every {0.693/lyapunov:.2f} time units")
        print()
        print("This means: even with perfect equations, prediction fails")
        print("because initial conditions can never be measured exactly.")
    else:
        print("System appears regular (may need longer simulation)")
    
    # =========================================================================
    # PART 4: Statistical Outcomes
    # =========================================================================
    print()
    print("-" * 70)
    print("PART 4: STATISTICAL OUTCOMES - What Usually Happens")
    print("-" * 70)
    print()
    print("Running 20 simulations with slightly different initial conditions...")
    print()
    
    outcomes = {'ejection': 0, 'bound': 0, 'collision': 0}
    
    np.random.seed(42)
    for i in range(20):
        bodies = get_chaotic_conditions()
        # Random perturbation
        bodies[0].x += np.random.uniform(-0.01, 0.01)
        bodies[0].vy += np.random.uniform(-0.01, 0.01)
        
        system = ThreeBodySystem(bodies)
        result = system.simulate(100, n_points=1000)
        
        # Check final state
        final = result['states'][:, -1]
        x1, y1, x2, y2, x3, y3 = final[:6]
        
        # Check for ejection (one body far from others)
        r12 = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        r13 = np.sqrt((x3-x1)**2 + (y3-y1)**2)
        r23 = np.sqrt((x3-x2)**2 + (y3-y2)**2)
        
        max_dist = max(r12, r13, r23)
        min_dist = min(r12, r13, r23)
        
        if max_dist > 10:
            outcomes['ejection'] += 1
        elif min_dist < 0.01:
            outcomes['collision'] += 1
        else:
            outcomes['bound'] += 1
    
    print("Outcome statistics:")
    print(f"  Ejections (one body escapes): {outcomes['ejection']}/20 = {outcomes['ejection']*5}%")
    print(f"  Near-collisions: {outcomes['collision']}/20 = {outcomes['collision']*5}%")
    print(f"  Remained bound: {outcomes['bound']}/20 = {outcomes['bound']*5}%")
    print()
    
    # =========================================================================
    # CONCLUSION
    # =========================================================================
    print("=" * 70)
    print("CONCLUSION: Why the Three-Body Problem Remains 'Unsolved'")
    print("=" * 70)
    print()
    print("1. NO FORMULA EXISTS for general initial conditions")
    print("   - Unlike the two-body problem (Kepler's laws)")
    print("   - Proven impossible by Poincare (1890)")
    print()
    print("2. CHAOS makes prediction fundamentally limited")
    print(f"   - Lyapunov exponent ~ {lyapunov:.3f} means exponential error growth")
    print("   - Tiny measurement errors destroy predictability")
    print()
    print("3. ONLY SPECIAL SOLUTIONS are known:")
    print("   - Lagrange points (1772)")
    print("   - Figure-8 orbit (1993)")
    print("   - ~1000 other periodic families discovered numerically")
    print()
    print("4. APPLICATIONS remain vital:")
    print("   - Spacecraft trajectory planning")
    print("   - Star cluster dynamics")
    print("   - Galaxy formation simulations")
    print("   - Solar system stability (is our system stable for 5 billion years?)")
    print()
    print("The three-body problem is a window into DETERMINISTIC CHAOS -")
    print("systems that are fully determined by physics, yet fundamentally")
    print("unpredictable without infinite precision.")
    print()
    
    plt.show()
    
    return result_figure8, lyapunov


if __name__ == "__main__":
    result, lyapunov = main()
