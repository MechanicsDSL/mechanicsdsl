"""
Pytest tests for MechanicsDSL
"""
import pytest
import numpy as np
from mechanics_dsl import PhysicsCompiler


def test_harmonic_oscillator():
    """Test harmonic oscillator system"""
    dsl_code = """
    \\system{oscillator}
    \\defvar{x}{position}{m}
    \\parameter{k}{1.0}{N/m}
    \\parameter{m}{1.0}{kg}
    \\lagrangian{0.5 * m * \\dot{x}^2 - 0.5 * k * x^2}
    \\initial{x=1.0, x_dot=0}
    """
    
    compiler = PhysicsCompiler()
    result = compiler.compile_dsl(dsl_code)
    
    assert result['success'], f"Compilation failed: {result.get('error', 'Unknown error')}"
    assert result['system_name'] == 'oscillator'
    assert 'x' in result['coordinates']
    
    # Run simulation
    solution = compiler.simulate(t_span=(0, 10), num_points=100)
    
    assert solution['success'], f"Simulation failed: {solution.get('error', 'Unknown error')}"
    assert 't' in solution
    assert 'y' in solution
    assert len(solution['t']) == 100
    assert solution['y'].shape[1] == 100
    
    # Check that position oscillates
    x = solution['y'][0]
    assert np.max(x) > 0.5  # Should reach at least 0.5
    assert np.min(x) < -0.5  # Should go negative


def test_figure8_orbit():
    """Test figure-8 orbit system (three-body problem)"""
    # Simplified figure-8: two masses in orbit
    dsl_code = """
    \\system{figure8}
    \\defvar{x}{position}{m}
    \\defvar{y}{position}{m}
    \\parameter{m}{1.0}{kg}
    \\parameter{G}{1.0}{N*m^2/kg^2}
    \\lagrangian{0.5 * m * (\\dot{x}^2 + \\dot{y}^2) + G * m^2 / \\sqrt{x^2 + y^2}}
    \\initial{x=1.0, y=0, x_dot=0, y_dot=0.5}
    """
    
    compiler = PhysicsCompiler()
    result = compiler.compile_dsl(dsl_code)
    
    assert result['success'], f"Compilation failed: {result.get('error', 'Unknown error')}"
    assert result['system_name'] == 'figure8'
    assert 'x' in result['coordinates']
    assert 'y' in result['coordinates']
    
    # Run simulation
    solution = compiler.simulate(t_span=(0, 10), num_points=100)
    
    assert solution['success'], f"Simulation failed: {solution.get('error', 'Unknown error')}"
    assert 't' in solution
    assert 'y' in solution
    assert solution['y'].shape[0] == 4  # x, x_dot, y, y_dot
    
    # Check that motion occurs
    x = solution['y'][0]
    y = solution['y'][2]
    assert np.max(np.abs(x)) > 0.1
    assert np.max(np.abs(y)) > 0.1


def test_simple_pendulum():
    """Test simple pendulum system"""
    dsl_code = """
    \\system{pendulum}
    \\defvar{theta}{angle}{rad}
    \\parameter{m}{1.0}{kg}
    \\parameter{l}{1.0}{m}
    \\parameter{g}{9.81}{m/s^2}
    \\lagrangian{0.5 * m * l^2 * \\dot{theta}^2 - m * g * l * (1 - \\cos{theta})}
    \\initial{theta=0.1, theta_dot=0}
    """
    
    compiler = PhysicsCompiler()
    result = compiler.compile_dsl(dsl_code)
    
    assert result['success'], f"Compilation failed: {result.get('error', 'Unknown error')}"
    assert result['system_name'] == 'pendulum'
    assert 'theta' in result['coordinates']
    
    # Run simulation
    solution = compiler.simulator.simulate(t_span=(0, 5), num_points=100)
    
    assert solution['success'], f"Simulation failed: {solution.get('error', 'Unknown error')}"
    assert 't' in solution
    assert 'y' in solution
    
    # Check that angle oscillates
    theta = solution['y'][0]
    assert np.max(theta) > 0.05
    assert np.min(theta) < -0.05


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

