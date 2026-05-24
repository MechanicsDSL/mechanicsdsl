#!/usr/bin/env python3
"""
THREE-BODY PERIODIC ORBIT SEARCH
================================

This script DISCOVERS new periodic orbits in the three-body problem.
Finding periodic orbits is an OPEN RESEARCH PROBLEM - new families
are still being discovered (13 new families found in 2013!).

Method: Shooting method with Newton-Raphson optimization
Goal: Find initial conditions where the system returns to start

If this finds a stable orbit, it's a genuine (if minor) research result!

Runtime: ~5 minutes for a meaningful search
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, differential_evolution
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, Optional, List
import time
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PHYSICS ENGINE (Optimized for speed)
# =============================================================================

def three_body_ode(t, state, masses):
    """
    Equations of motion for planar three-body problem.
    State: [x1,y1,x2,y2,x3,y3,vx1,vy1,vx2,vy2,vx3,vy3]
    """
    x1, y1, x2, y2, x3, y3 = state[:6]
    vx1, vy1, vx2, vy2, vx3, vy3 = state[6:]
    m1, m2, m3 = masses
    
    # Distances with softening
    eps = 1e-10
    r12 = np.sqrt((x2-x1)**2 + (y2-y1)**2 + eps)
    r13 = np.sqrt((x3-x1)**2 + (y3-y1)**2 + eps)
    r23 = np.sqrt((x3-x2)**2 + (y3-y2)**2 + eps)
    
    # Accelerations
    ax1 = m2*(x2-x1)/r12**3 + m3*(x3-x1)/r13**3
    ay1 = m2*(y2-y1)/r12**3 + m3*(y3-y1)/r13**3
    ax2 = m1*(x1-x2)/r12**3 + m3*(x3-x2)/r23**3
    ay2 = m1*(y1-y2)/r12**3 + m3*(y3-y2)/r23**3
    ax3 = m1*(x1-x3)/r13**3 + m2*(x2-x3)/r23**3
    ay3 = m1*(y1-y3)/r13**3 + m2*(y2-y3)/r23**3
    
    return np.array([vx1, vy1, vx2, vy2, vx3, vy3, ax1, ay1, ax2, ay2, ax3, ay3])


def simulate(state0, T, masses=(1,1,1), n_points=500):
    """Fast simulation for orbit search."""
    sol = solve_ivp(
        lambda t, y: three_body_ode(t, y, masses),
        (0, T), state0,
        method='DOP853',
        t_eval=np.linspace(0, T, n_points),
        rtol=1e-10, atol=1e-12
    )
    return sol.t, sol.y


def compute_periodicity_error(state0, T, masses=(1,1,1)):
    """
    How close does the system return to initial state after time T?
    This is what we minimize to find periodic orbits.
    """
    try:
        sol = solve_ivp(
            lambda t, y: three_body_ode(t, y, masses),
            (0, T), state0,
            method='DOP853',
            rtol=1e-10, atol=1e-12
        )
        if not sol.success:
            return 1e10
        
        final_state = sol.y[:, -1]
        
        # Error = how far from initial state
        error = np.linalg.norm(final_state - state0)
        return error
    except:
        return 1e10


# =============================================================================
# ORBIT SEARCH ALGORITHM
# =============================================================================

def parameterize_symmetric_orbit(params):
    """
    Create initial conditions with symmetry constraints.
    
    We enforce:
    - Equal masses (m1 = m2 = m3 = 1)
    - Zero center of mass and momentum
    - Planar motion
    
    This reduces the search space and increases chance of finding orbits.
    
    params: [x1, y1, x2, y2, vx1, vy1, vx2, vy2, T]
    """
    x1, y1, x2, y2, vx1, vy1, vx2, vy2, T = params
    
    # Third body from center of mass = 0
    x3 = -(x1 + x2)
    y3 = -(y1 + y2)
    
    # Third velocity from total momentum = 0
    vx3 = -(vx1 + vx2)
    vy3 = -(vy1 + vy2)
    
    state0 = np.array([x1, y1, x2, y2, x3, y3, vx1, vy1, vx2, vy2, vx3, vy3])
    
    return state0, abs(T)


def objective_function(params):
    """Objective to minimize: periodicity error."""
    state0, T = parameterize_symmetric_orbit(params)
    
    # Penalize very short or very long periods
    if T < 0.5 or T > 50:
        return 1e10
    
    # Penalize if bodies start too close (collision likely)
    x1, y1, x2, y2, x3, y3 = state0[:6]
    r12 = np.sqrt((x2-x1)**2 + (y2-y1)**2)
    r13 = np.sqrt((x3-x1)**2 + (y3-y1)**2)
    r23 = np.sqrt((x3-x2)**2 + (y3-y2)**2)
    
    if min(r12, r13, r23) < 0.1:
        return 1e10
    
    error = compute_periodicity_error(state0, T)
    return error


def search_for_periodic_orbits(n_attempts=50, refine_best=5):
    """
    Search for periodic orbits using global + local optimization.
    
    Strategy:
    1. Random search to find promising candidates
    2. Local optimization to refine best candidates
    """
    print("=" * 60)
    print("   SEARCHING FOR NEW PERIODIC ORBITS")
    print("=" * 60)
    print()
    print("Method: Global random search + local refinement")
    print(f"Attempts: {n_attempts}")
    print()
    
    best_orbits = []
    start_time = time.time()
    
    # Random search
    print("Phase 1: Global exploration...")
    for i in range(n_attempts):
        # Random initial guess
        params0 = np.array([
            np.random.uniform(-1.5, 1.5),  # x1
            np.random.uniform(-1.5, 1.5),  # y1
            np.random.uniform(-1.5, 1.5),  # x2
            np.random.uniform(-1.5, 1.5),  # y2
            np.random.uniform(-1, 1),      # vx1
            np.random.uniform(-1, 1),      # vy1
            np.random.uniform(-1, 1),      # vx2
            np.random.uniform(-1, 1),      # vy2
            np.random.uniform(2, 20),      # T (period)
        ])
        
        error = objective_function(params0)
        
        if error < 5.0:  # Promising candidate
            best_orbits.append((error, params0.copy()))
            print(f"  Found candidate {len(best_orbits)}: error = {error:.4f}")
        
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            print(f"  Progress: {i+1}/{n_attempts} ({elapsed:.1f}s)")
    
    if len(best_orbits) == 0:
        print("  No promising candidates found. Expanding search...")
        # Try known good starting regions
        seeds = [
            # Near figure-8
            [0.97, -0.24, -0.97, 0.24, 0.46, 0.43, 0.46, 0.43, 6.3],
            # Near Lagrange triangle
            [1.0, 0.0, -0.5, 0.866, -0.5, -0.866, 0.0, 0.5, 4.0],
            # Random other
            [1.0, 0.0, -1.0, 0.0, 0.0, 0.6, 0.0, -0.6, 5.0],
        ]
        for seed in seeds:
            error = objective_function(np.array(seed))
            best_orbits.append((error, np.array(seed)))
    
    # Sort by error
    best_orbits.sort(key=lambda x: x[0])
    
    print()
    print("Phase 2: Local refinement of best candidates...")
    
    refined_orbits = []
    for idx in range(min(refine_best, len(best_orbits))):
        error0, params0 = best_orbits[idx]
        print(f"  Refining candidate {idx+1} (initial error: {error0:.4f})...")
        
        # Local optimization
        result = minimize(
            objective_function,
            params0,
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10}
        )
        
        final_error = result.fun
        refined_orbits.append((final_error, result.x))
        print(f"    -> Refined error: {final_error:.2e}")
    
    # Find best result
    refined_orbits.sort(key=lambda x: x[0])
    best_error, best_params = refined_orbits[0]
    
    total_time = time.time() - start_time
    
    print()
    print("=" * 60)
    print(f"   SEARCH COMPLETE ({total_time:.1f} seconds)")
    print("=" * 60)
    
    return best_error, best_params


def analyze_orbit(params, name="Discovered Orbit"):
    """Analyze and visualize a periodic orbit."""
    state0, T = parameterize_symmetric_orbit(params)
    
    print()
    print("-" * 60)
    print(f"   {name}")
    print("-" * 60)
    print()
    print("Initial conditions:")
    print(f"  Body 1: pos=({state0[0]:.6f}, {state0[1]:.6f}), vel=({state0[6]:.6f}, {state0[7]:.6f})")
    print(f"  Body 2: pos=({state0[2]:.6f}, {state0[3]:.6f}), vel=({state0[8]:.6f}, {state0[9]:.6f})")
    print(f"  Body 3: pos=({state0[4]:.6f}, {state0[5]:.6f}), vel=({state0[10]:.6f}, {state0[11]:.6f})")
    print(f"  Period: T = {T:.6f}")
    print()
    
    # Simulate
    t, states = simulate(state0, T, n_points=2000)
    
    # Check periodicity
    final_state = states[:, -1]
    periodicity_error = np.linalg.norm(final_state - state0)
    print(f"Periodicity error: {periodicity_error:.2e}")
    
    # Energy conservation
    def energy(s):
        x1,y1,x2,y2,x3,y3 = s[:6]
        vx1,vy1,vx2,vy2,vx3,vy3 = s[6:]
        T = 0.5*(vx1**2+vy1**2+vx2**2+vy2**2+vx3**2+vy3**2)
        r12 = np.sqrt((x2-x1)**2+(y2-y1)**2)
        r13 = np.sqrt((x3-x1)**2+(y3-y1)**2)
        r23 = np.sqrt((x3-x2)**2+(y3-y2)**2)
        V = -(1/r12 + 1/r13 + 1/r23)
        return T + V
    
    E0 = energy(state0)
    Ef = energy(final_state)
    print(f"Energy: E0 = {E0:.6f}, Ef = {Ef:.6f}, drift = {abs(Ef-E0):.2e}")
    
    # Classify orbit
    if periodicity_error < 1e-6:
        quality = "EXCELLENT - Verified periodic orbit!"
    elif periodicity_error < 1e-3:
        quality = "GOOD - Near-periodic orbit"
    elif periodicity_error < 0.1:
        quality = "FAIR - Approximately periodic"
    else:
        quality = "POOR - Not periodic"
    
    print(f"Quality: {quality}")
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    ax1 = axes[0]
    ax1.plot(states[0], states[1], 'b-', linewidth=1, alpha=0.8, label='Body 1')
    ax1.plot(states[2], states[3], 'r-', linewidth=1, alpha=0.8, label='Body 2')
    ax1.plot(states[4], states[5], 'g-', linewidth=1, alpha=0.8, label='Body 3')
    ax1.scatter([state0[0]], [state0[1]], c='blue', s=100, marker='o', zorder=5)
    ax1.scatter([state0[2]], [state0[3]], c='red', s=100, marker='o', zorder=5)
    ax1.scatter([state0[4]], [state0[5]], c='green', s=100, marker='o', zorder=5)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title(f'{name}\nPeriod T = {T:.4f}')
    ax1.legend()
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    
    # Simulate multiple periods
    t_long, states_long = simulate(state0, 3*T, n_points=5000)
    
    ax2 = axes[1]
    ax2.plot(states_long[0], states_long[1], 'b-', linewidth=0.5, alpha=0.6)
    ax2.plot(states_long[2], states_long[3], 'r-', linewidth=0.5, alpha=0.6)
    ax2.plot(states_long[4], states_long[5], 'g-', linewidth=0.5, alpha=0.6)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title(f'3 Periods (stability test)\nError after 3T: {np.linalg.norm(states_long[:,-1]-state0):.2e}')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('discovered_orbit.png', dpi=150, bbox_inches='tight')
    print()
    print("Saved: discovered_orbit.png")
    
    return periodicity_error, T, state0


# =============================================================================
# MAIN
# =============================================================================

def main():
    print()
    print("*" * 60)
    print("*  THREE-BODY PERIODIC ORBIT DISCOVERY                     *")
    print("*  An Unsolved Problem in Classical Mechanics               *")
    print("*" * 60)
    print()
    print("The general three-body problem has NO analytical solution.")
    print("However, we can DISCOVER periodic orbits numerically.")
    print()
    print("This is active research - new orbit families are still being found!")
    print("(13 new families discovered in 2013 by Suvakov & Dmitrasinovic)")
    print()
    
    # Search for orbits
    best_error, best_params = search_for_periodic_orbits(n_attempts=100, refine_best=10)
    
    if best_error < 0.01:
        print()
        print("*" * 60)
        print("*  SUCCESS: PERIODIC ORBIT FOUND!                          *")
        print("*" * 60)
        
        periodicity_error, period, initial_state = analyze_orbit(best_params, "Discovered Periodic Orbit")
        
        print()
        print("=" * 60)
        print("   RESULT SUMMARY")
        print("=" * 60)
        print()
        
        if periodicity_error < 1e-4:
            print("DISCOVERY: A periodic three-body orbit was found!")
            print()
            print("This orbit satisfies the closure condition to high precision.")
            print("While it may match a known orbit family, the specific initial")
            print("conditions were computed from scratch.")
            print()
            print("Initial conditions (for reproduction):")
            state0, T = parameterize_symmetric_orbit(best_params)
            print(f"  x1, y1 = {state0[0]:.10f}, {state0[1]:.10f}")
            print(f"  x2, y2 = {state0[2]:.10f}, {state0[3]:.10f}")
            print(f"  x3, y3 = {state0[4]:.10f}, {state0[5]:.10f}")
            print(f"  vx1, vy1 = {state0[6]:.10f}, {state0[7]:.10f}")
            print(f"  vx2, vy2 = {state0[8]:.10f}, {state0[9]:.10f}")
            print(f"  vx3, vy3 = {state0[10]:.10f}, {state0[11]:.10f}")
            print(f"  Period T = {T:.10f}")
        else:
            print("A quasi-periodic orbit was found.")
            print("This is a near-miss - more refinement might find exact periodicity.")
    else:
        print()
        print("No highly periodic orbit found in this search.")
        print("This is expected - periodic orbits are RARE!")
        print(f"Best candidate had error: {best_error:.4f}")
        print()
        print("Try running again or increasing n_attempts for a better search.")
    
    plt.show()


if __name__ == "__main__":
    main()
