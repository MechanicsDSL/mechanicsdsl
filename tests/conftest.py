"""
Pytest configuration and shared fixtures for MechanicsDSL tests
"""
import pytest
import sys
from pathlib import Path

# Add src to path for testing
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from mechanics_dsl import PhysicsCompiler


@pytest.fixture
def compiler():
    """Create a fresh PhysicsCompiler instance for each test"""
    return PhysicsCompiler()


@pytest.fixture
def simple_dsl():
    """Simple harmonic oscillator DSL for testing"""
    return """
\\system{test_oscillator}

\\var{x}{Position}{m}

\\parameter{m}{1.0}{kg}
\\parameter{k}{10.0}{N/m}

\\lagrangian{\\frac{1}{2} * m * \\dot{x}^2 - \\frac{1}{2} * k * x^2}

\\initial{x=1.0, x_dot=0.0}
"""


@pytest.fixture
def pendulum_dsl():
    """Simple pendulum DSL for testing"""
    return """
\\system{test_pendulum}

\\var{theta}{Angle}{rad}

\\parameter{m}{1.0}{kg}
\\parameter{L}{1.0}{m}
\\parameter{g}{9.81}{m/s^2}

\\lagrangian{
    \\frac{1}{2} * m * L^2 * \\dot{theta}^2 - m * g * L * (1 - \\cos{theta})
}

\\initial{theta=0.1, theta_dot=0.0}
"""

