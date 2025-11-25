import pytest
import numpy as np
import matplotlib
import os
from mechanics_dsl import PhysicsCompiler, SystemValidator

# Force matplotlib to not use a GUI (prevents crashes in Codespaces)
matplotlib.use('Agg')

# -----------------------------------------------------------------------------
# 1. PARSER STABILITY
# -----------------------------------------------------------------------------

def test_parser_valid_syntax():
    """Ensure valid DSL code compiles without error."""
    dsl = r"""
    \system{test_sys}
    \defvar{x}{Position}{m}
    \defvar{m}{Mass}{kg}
    \parameter{m}{1.0}{kg}
    \lagrangian{ 0.5 * m * \dot{x}^2 }
    \initial{x=0.0, x_dot=1.0}
    \solve{RK45}
    """
    compiler = PhysicsCompiler()
    result = compiler.compile_dsl(dsl)
    assert result['success'] is True
    assert result['system_name'] == 'test_sys'

# -----------------------------------------------------------------------------
# 2. PHYSICS ACCURACY
# -----------------------------------------------------------------------------

def test_harmonic_oscillator_frequency():
    """
    Critical Stability Check:
    Does the math match analytical solutions? (w = sqrt(k/m))
    """
    dsl = r"""
    \system{oscillator}
    \defvar{x}{Position}{m}
    \defvar{m}{Mass}{kg}
    \defvar{k}{Constant}{N/m}
    \parameter{m}{1.0}{kg}
    \parameter{k}{16.0}{N/m} % w = 4 rad/s
    \lagrangian{ 0.5 * m * \dot{x}^2 - 0.5 * k * x^2 }
    \initial{x=1.0, x_dot=0.0}
    \solve{RK45}
    """
    compiler = PhysicsCompiler()
    res = compiler.compile_dsl(dsl)
    # Simulate for a short time
    sol = compiler.simulate(t_span=(0, 2.0))
    
    # Validate against analytical solution x(t) = cos(4t)
    t = sol['t']
    x_sim = sol['y'][0]
    x_analytical = np.cos(4 * t)
    
    # Check max error is small (< 0.1)
    error = np.max(np.abs(x_sim - x_analytical))
    assert error < 0.1

# -----------------------------------------------------------------------------
# 3. SYSTEM I/O STABILITY
# -----------------------------------------------------------------------------

def test_json_export_import(tmp_path):
    """Ensure saving/loading systems works."""
    dsl = r"""
    \system{io_test}
    \defvar{q}{Generic}{unit}
    \lagrangian{ \dot{q}^2 }
    """
    compiler = PhysicsCompiler()
    compiler.compile_dsl(dsl)
    
    # Test Export
    filepath = tmp_path / "test_system.json"
    success = compiler.export_system(str(filepath))
    assert success
    assert os.path.exists(filepath)
