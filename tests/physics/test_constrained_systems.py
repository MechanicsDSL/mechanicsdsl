"""
Constrained system tests - Holonomic and non-holonomic constraints
Tests both new package structure and original core.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

# Detect CI environment and adjust tolerances
IS_CI = os.getenv("CI") == "true" or os.getenv("GITHUB_ACTIONS") == "true"
CONSTRAINT_TOL_MULTIPLIER = 2.0 if IS_CI else 1.0

try:
    from mechanics_dsl import PhysicsCompiler

    NEW_PACKAGE = True
except ImportError:
    NEW_PACKAGE = False

try:
    sys.path.insert(0, str(Path(__file__).parent.parent / "reference"))
    from core import PhysicsCompiler as CorePhysicsCompiler

    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False


def get_compiler():
    """Get compiler instance"""
    if NEW_PACKAGE:
        return PhysicsCompiler()
    elif CORE_AVAILABLE:
        return CorePhysicsCompiler()
    else:
        pytest.skip("Neither new package nor core.py available")


class TestRollingBall:
    """Test rolling ball with constraint"""

    def test_rolling_ball_constraint(self):
        """Test ball rolling down incline with rolling constraint"""
        dsl_code = r"""
        \system{rolling_ball}
        \defvar{x}{Position}{m}
        \defvar{theta}{Angle}{rad}
        \defvar{m}{Mass}{kg}
        \defvar{R}{Radius}{m}
        \defvar{g}{Acceleration}{m/s^2}
        \defvar{alpha}{Incline Angle}{rad}

        \parameter{m}{1.0}{kg}
        \parameter{R}{0.1}{m}
        \parameter{g}{9.81}{m/s^2}
        \parameter{alpha}{0.3}{rad}

        \lagrangian{
            \frac{1}{2} * m * \dot{x}^2
            + \frac{1}{2} * \frac{2}{5} * m * R^2 * \dot{theta}^2
            - m * g * x * \sin{alpha}
        }

        \constraint{x - R * theta}

        \initial{x=0.0, x_dot=0.0}
        """

        compiler = get_compiler()
        result = compiler.compile_dsl(dsl_code, use_constraints=True)

        assert result["success"]
        assert len(result["coordinates"]) >= 1

        solution = compiler.simulate(t_span=(0, 5), num_points=500)

        assert solution["success"]

        # Behavioral check: a uniform solid sphere (I = 2/5 m R^2) rolling
        # without slipping on an incline has acceleration of magnitude
        # a = (5/7) g sin(alpha) along the slope, independent of m and R.
        # With this Lagrangian's sign convention the coordinate x decreases,
        # so we test the magnitude and the rolling constraint x = R*theta,
        # not a hard-coded direction.
        t = solution["t"]
        x = solution["y"][0]
        theta = solution["y"][2]

        g, alpha, R = 9.81, 0.3, 0.1
        a_expected = (5.0 / 7.0) * g * np.sin(alpha)

        # Constant-acceleration trajectory from rest: |x(t) - x0| = 1/2 a t^2.
        displacement = np.abs(x - x[0])
        predicted = 0.5 * a_expected * t**2
        # Relative agreement over the whole trajectory (skip t=0 where both ~0).
        mask = t > 0.5
        rel_err = np.abs(displacement[mask] - predicted[mask]) / predicted[mask]
        tol = 0.05 * CONSTRAINT_TOL_MULTIPLIER
        assert np.max(rel_err) < tol, (
            f"Rolling acceleration magnitude wrong: max rel err {np.max(rel_err):.4f} "
            f"(expected a = {a_expected:.4f} m/s^2)"
        )

        # Rolling constraint x = R*theta must hold to tight tolerance.
        constraint_resid = np.max(np.abs(x - R * theta))
        assert constraint_resid < 1e-6, (
            f"Rolling constraint x = R*theta violated: max resid {constraint_resid:.2e}"
        )

        # And it genuinely moved (guards against a frozen/degenerate solve).
        assert np.abs(x[-1] - x[0]) > 1.0, f"Ball barely moved: {x[-1] - x[0]:.6f}"


class TestAtwoodMachine:
    """Test Atwood machine with constraint"""

    def test_atwood_machine(self):
        """Test Atwood machine (two masses connected by string)"""
        dsl_code = r"""
        \system{atwood_machine}
        \defvar{x1}{Position}{m}
        \defvar{x2}{Position}{m}
        \defvar{m1}{Mass}{kg}
        \defvar{m2}{Mass}{kg}
        \defvar{g}{Acceleration}{m/s^2}
        \defvar{l}{Constant}{m}

        \parameter{m1}{2.0}{kg}
        \parameter{m2}{1.0}{kg}
        \parameter{g}{9.81}{m/s^2}
        \parameter{l}{5.0}{m}

        \lagrangian{
            \frac{1}{2} * m1 * \dot{x1}^2
            + \frac{1}{2} * m2 * \dot{x2}^2
            + m1 * g * x1
            + m2 * g * x2
        }

        \constraint{x1 + x2 - l}

        \initial{x1=2.0, x1_dot=0.0, x2=3.0, x2_dot=0.0}
        """

        compiler = get_compiler()
        result = compiler.compile_dsl(dsl_code, use_constraints=True)
        assert result["success"], result

        solution = compiler.simulate(t_span=(0, 3), num_points=300)
        assert solution["success"]

        # Behavioral check: an Atwood machine has constant acceleration of
        # magnitude a = (m1 - m2)/(m1 + m2) * g for the coordinate x1. With
        # m1=2, m2=1: a = g/3. The inextensible-string constraint x1 + x2 = l
        # must also hold throughout.
        t = solution["t"]
        x1 = solution["y"][0]
        x2 = solution["y"][2]

        m1, m2, g, length = 2.0, 1.0, 9.81, 5.0
        a_expected = (m1 - m2) / (m1 + m2) * g

        displacement = np.abs(x1 - x1[0])
        predicted = 0.5 * a_expected * t**2
        mask = t > 0.5
        rel_err = np.abs(displacement[mask] - predicted[mask]) / predicted[mask]
        tol = 0.05 * CONSTRAINT_TOL_MULTIPLIER
        assert np.max(rel_err) < tol, (
            f"Atwood acceleration wrong: max rel err {np.max(rel_err):.4f} "
            f"(expected a = {a_expected:.4f} m/s^2)"
        )

        # String constraint x1 + x2 = l must be preserved.
        constraint_resid = np.max(np.abs(x1 + x2 - length))
        assert constraint_resid < 1e-6, (
            f"String constraint x1 + x2 = l violated: max resid {constraint_resid:.2e}"
        )


class TestPendulumWithConstraint:
    """Test pendulum with additional constraints"""

    def test_constrained_pendulum(self):
        """Test pendulum with length constraint"""
        dsl_code = r"""
        \system{constrained_pendulum}
        \defvar{x}{Position}{m}
        \defvar{y}{Position}{m}
        \defvar{m}{Mass}{kg}
        \defvar{g}{Acceleration}{m/s^2}
        \defvar{l}{Constant}{m}

        \parameter{m}{1.0}{kg}
        \parameter{g}{9.81}{m/s^2}
        \parameter{l}{1.0}{m}

        \lagrangian{
            \frac{1}{2} * m * (\dot{x}^2 + \dot{y}^2)
            - m * g * y
        }

        \constraint{x^2 + y^2 - l^2}

        \initial{x=0.0, x_dot=1.5, y=-1.0, y_dot=0.0}
        """

        compiler = get_compiler()
        result = compiler.compile_dsl(dsl_code, use_constraints=True)
        assert result["success"], result

        solution = compiler.simulate(t_span=(0, 5), num_points=500)
        assert solution["success"]

        # The bob starts at the bottom (0, -1) with a tangential push, so it
        # must actually swing (not sit frozen at a degenerate/NaN solve).
        x = solution["y"][0]
        y = solution["y"][2] if solution["y"].shape[0] > 2 else solution["y"][1]
        assert np.ptp(x) > 0.2, f"Pendulum did not swing (x range {np.ptp(x):.4f})"

        # Behavioral check: the holonomic constraint x^2 + y^2 = l^2 must hold
        # throughout the trajectory. The acceleration-level formulation keeps
        # this tight over this horizon.
        r = np.sqrt(x**2 + y**2)
        constraint_error = np.max(np.abs(r - 1.0))
        tolerance = 1e-2 * CONSTRAINT_TOL_MULTIPLIER
        assert constraint_error < tolerance, (
            f"Constraint x^2+y^2=l^2 violated: max |r - 1| = {constraint_error:.6f}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
