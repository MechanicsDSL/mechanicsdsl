import pytest
import numpy as np
from mechanics_dsl.compiler import PhysicsCompiler
from mechanics_dsl.energy import PotentialEnergyCalculator

def test_harmonic_oscillator():
    """
    Test Case 1: Simple Harmonic Oscillator
    Verifies that the compiler can handle basic Lagrangian mechanics
    and that energy is conserved for a conservative system.
    """
    code = r"""
    \system{oscillator}
    \defvar{x}{Position}{m} 
    \defvar{m}{Mass}{kg} 
    \defvar{k}{K}{N/m}
    
    \parameter{m}{1.0}{kg} 
    \parameter{k}{10.0}{N/m}
    
    \lagrangian{0.5 * m * \dot{x}^2 - 0.5 * k * x^2}
    
    \initial{x=1.0, x_dot=0.0}
    \solve{RK45}
    """
    
    # 1. Compile
    compiler = PhysicsCompiler()
    res = compiler.compile_dsl(code)
    assert res['success'] is True, f"Compilation failed: {res.get('error')}"
    
    # 2. Simulate
    sol = compiler.simulate((0, 10))
    assert sol['success'] is True, f"Simulation failed: {sol.get('error')}"
    
    # 3. Verify Energy Conservation
    # For a harmonic oscillator, Total Energy E = T + V = constant
    params = compiler.simulator.parameters
    ke = PotentialEnergyCalculator.compute_kinetic_energy(sol, params)
    pe = PotentialEnergyCalculator.compute_potential_energy(sol, params, "oscillator")
    total_energy = ke + pe
    
    # Calculate drift: |(E_final - E_initial) / E_initial|
    # Note: E_initial = 0.5 * k * x^2 = 0.5 * 10 * 1^2 = 5.0
    drift = np.max(np.abs(total_energy - total_energy[0]))
    
    # RK45 should be very accurate for this smooth system
    assert drift < 1e-3, f"Energy drift too high: {drift}"

def test_figure8_stability():
    """
    Test Case 2: The 'Grandmaster' Figure-8 Orbit
    Verifies N-Body dynamics, complex potential parsing, and LSODA solver stability.
    """
    code = r"""
    \system{figure8}
    \defvar{x1}{X}{m} \defvar{y1}{Y}{m}
    \defvar{x2}{X}{m} \defvar{y2}{Y}{m}
    \defvar{x3}{X}{m} \defvar{y3}{Y}{m}
    \parameter{m}{1.0}{kg} \parameter{G}{1.0}{1}
    
    \lagrangian{ 
        0.5 * m * (\dot{x1}^2 + \dot{y1}^2 + \dot{x2}^2 + \dot{y2}^2 + \dot{x3}^2 + \dot{y3}^2) 
        + G*m^2/((x1-x2)^2 + (y1-y2)^2)^0.5 
        + G*m^2/((x2-x3)^2 + (y2-y3)^2)^0.5 
        + G*m^2/((x1-x3)^2 + (y1-y3)^2)^0.5
    }
    
    \initial{
        x1=0.97000436, y1=-0.24308753, x1_dot=0.4662036850, y1_dot=0.4323657300,
        x2=-0.97000436, y2=0.24308753, x2_dot=0.4662036850, y2_dot=0.4323657300,
        x3=0.0, y3=0.0, x3_dot=-0.93240737, y3_dot=-0.86473146
    }
    \solve{LSODA}
    """
    
    compiler = PhysicsCompiler()
    res = compiler.compile_dsl(code)
    assert res['success'] is True
    
    # Simulate for exactly one period (T approx 6.3259)
    # We use tight tolerance (1e-9) to ensure the orbit closes
    sol = compiler.simulate((0, 6.3259), rtol=1e-9, atol=1e-9)
    assert sol['success'] is True
    
    # Check Periodicity: Bodies should return to starting positions
    # We check body 1 (x1, y1)
    y = sol['y']
    start_pos = np.array([y[0][0], y[2][0]])  # x1_0, y1_0
    end_pos = np.array([y[0][-1], y[2][-1]])  # x1_T, y1_T
    
    dist = np.linalg.norm(start_pos - end_pos)
    
    # If the physics engine is sound, the bodies should be very close to where they started
    assert dist < 0.1, f"Orbit drifted by {dist}, physics engine may be unstable"

def test_parser_error_recovery():
    """
    Test Case 3: Error Handling
    Verifies that the compiler fails gracefully on bad input instead of crashing.
    """
    bad_code = r"""
    \system{broken}
    \defvar{x}{Position}{m}
    \lagrangian{ 0.5 * m * \dot{x}^2 } % Missing potential and parameters
    \initial{x=0}
    """
    
    compiler = PhysicsCompiler()
    # This should NOT raise a Python exception (crash), but return success=False
    res = compiler.compile_dsl(bad_code)
    
    assert res['success'] is False
    assert 'error' in res
