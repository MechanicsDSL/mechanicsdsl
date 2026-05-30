"""
End-to-end checks for the README's headline optional features.

Each test verifies one of the high-level capabilities the project advertises
actually works on a real example, not just at the import level:

  * plugin loading
  * inverse parameter estimation
  * JAX backend (skipped when jax is not installed)
"""

import os
import tempfile
import textwrap

import numpy as np
import pytest

from mechanics_dsl import PhysicsCompiler


# ---------------------------------------------------------------------------
# Plugin system - load a user-supplied plugin from a file
# ---------------------------------------------------------------------------


PLUGIN_SOURCE = textwrap.dedent(
    '''
    """Trivial domain plugin used by the v2.1.x regression suite."""

    from mechanics_dsl.plugins import PhysicsDomainPlugin
    from mechanics_dsl.plugins.base import PluginMetadata


    class _AcousticsDomain:
        """Stand-in 'domain class' the plugin advertises."""

        name = "acoustics"

        def speed_of_sound(self, gamma: float, R: float, T: float) -> float:
            return (gamma * R * T) ** 0.5


    class AcousticsPlugin(PhysicsDomainPlugin):
        @property
        def metadata(self):
            return PluginMetadata(
                name="acoustics_test_plugin",
                version="0.1.0",
                description="Test plugin for the v2.1.x suite",
            )

        def get_domain_class(self):
            return _AcousticsDomain

        def get_domain_name(self):
            return "acoustics"
    '''
).lstrip()


def test_plugin_can_be_loaded_from_a_file_and_invoked():
    """Drop a plugin .py into a temp dir, load it, instantiate, use it."""
    from mechanics_dsl.plugins import PluginLoader, PluginRegistry
    from mechanics_dsl.plugins.registry import PluginType

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(PLUGIN_SOURCE)
        plugin_path = f.name

    try:
        # Use an isolated registry so we don't pollute the global one.
        registry = PluginRegistry()
        loader = PluginLoader(registry=registry)
        loaded = loader.load_file(plugin_path)

        domain_names = loaded[PluginType.DOMAIN]
        assert domain_names, f"plugin did not register: {loaded}"
        assert "acoustics_test_plugin" in domain_names

        # `registry.get` returns an instance, not the class.
        instance = registry.get(PluginType.DOMAIN, "acoustics_test_plugin")
        assert instance is not None
        domain = instance.get_domain_class()()
        # gamma=1.4, R=287, T=293 K -> sound speed in dry air ~ 343 m/s
        c = domain.speed_of_sound(1.4, 287.0, 293.0)
        assert 300 < c < 360
    finally:
        try:
            os.unlink(plugin_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Inverse problems: parameter estimation recovers ground-truth params
# ---------------------------------------------------------------------------


def test_parameter_estimator_recovers_pendulum_length():
    """Generate a synthetic pendulum trajectory with l=1.5, perturb the
    initial guess to l=1.0, and fit; the estimator must land near 1.5."""
    from mechanics_dsl.inverse import ParameterEstimator

    true_length = 1.5
    g = 9.81

    # Build the ground-truth system and generate observations.
    truth = PhysicsCompiler()
    truth.compile_dsl(
        r"\system{ptrue}"
        r"\defvar{theta}{Angle}{rad}"
        rf"\parameter{{m}}{{1.0}}{{kg}}\parameter{{l}}{{{true_length}}}{{m}}"
        rf"\parameter{{g}}{{{g}}}{{m/s^2}}"
        r"\lagrangian{0.5*m*l^2*\dot{theta}^2 - m*g*l*(1 - \cos{theta})}"
        r"\initial{theta=0.1, theta_dot=0}"
    )
    truth_sol = truth.simulate(t_span=(0, 5), num_points=200)
    assert truth_sol["success"]
    observations = truth_sol["y"][0]  # theta(t)
    t_obs = truth_sol["t"]

    # Build a fitting system with the wrong initial length.
    fit_compiler = PhysicsCompiler()
    fit_compiler.compile_dsl(
        r"\system{pfit}"
        r"\defvar{theta}{Angle}{rad}"
        rf"\parameter{{m}}{{1.0}}{{kg}}\parameter{{l}}{{1.0}}{{m}}"
        rf"\parameter{{g}}{{{g}}}{{m/s^2}}"
        r"\lagrangian{0.5*m*l^2*\dot{theta}^2 - m*g*l*(1 - \cos{theta})}"
        r"\initial{theta=0.1, theta_dot=0}"
    )

    estimator = ParameterEstimator(fit_compiler)
    result = estimator.fit(
        observations=observations,
        t_obs=t_obs,
        params_to_fit=["l"],
        bounds={"l": (0.1, 5.0)},
        method="L-BFGS-B",
    )

    assert result.success, getattr(result, "message", "")
    fitted_length = result.parameters["l"]
    assert abs(fitted_length - true_length) / true_length < 0.05, (
        f"Estimator landed at l={fitted_length:.3f}, expected {true_length}"
    )


# ---------------------------------------------------------------------------
# JAX backend: skipped when jax isn't installed
# ---------------------------------------------------------------------------


def test_jax_backend_runs_pendulum_simulation():
    """If jax/diffrax are installed, the JAX backend must run a pendulum."""
    pytest.importorskip("jax")
    pytest.importorskip("diffrax")

    from mechanics_dsl.backends.jax_backend import JAXBackend

    compiler = PhysicsCompiler()
    compiler.compile_dsl(
        r"\system{pjax}"
        r"\defvar{theta}{Angle}{rad}"
        r"\parameter{m}{1.0}{kg}\parameter{l}{1.0}{m}\parameter{g}{9.81}{m/s^2}"
        r"\lagrangian{0.5*m*l^2*\dot{theta}^2 - m*g*l*(1 - \cos{theta})}"
        r"\initial{theta=0.1, theta_dot=0}"
    )
    backend = JAXBackend()
    sol = backend.simulate(compiler, t_span=(0, 1.0), num_points=20)
    assert sol is not None
    # Some shape with 20 time points.
    arr = np.asarray(sol["y"] if isinstance(sol, dict) else sol)
    assert arr.shape[-1] in (20, 2)
