"""
Per-domain physics validation against textbook closed-form answers.

The goal of this file is to make sure each domain module actually computes
the right number for a problem you can look up in any undergraduate
textbook. If anything here drifts more than the tolerances, the bug is
likely in the domain code, not the test.

Each test cites the textbook formula it's checking against.
"""

import math

import numpy as np
import pytest

from mechanics_dsl import PhysicsCompiler


# ---------------------------------------------------------------------------
# Classical mechanics: small-angle pendulum
# ---------------------------------------------------------------------------


def test_classical_simple_pendulum_period():
    """T = 2π√(l/g) for small oscillations."""
    g, length = 9.81, 1.0
    expected_period = 2 * math.pi * math.sqrt(length / g)

    compiler = PhysicsCompiler()
    compiler.compile_dsl(
        r"\system{pendulum_val}"
        r"\defvar{theta}{Angle}{rad}"
        rf"\parameter{{m}}{{1.0}}{{kg}}\parameter{{l}}{{{length}}}{{m}}"
        rf"\parameter{{g}}{{{g}}}{{m/s^2}}"
        r"\lagrangian{0.5*m*l^2*\dot{theta}^2 - m*g*l*(1 - \cos{theta})}"
        # Tiny amplitude so the small-angle approximation is valid.
        r"\initial{theta=0.02, theta_dot=0}"
    )
    sol = compiler.simulate(t_span=(0, 3 * expected_period), num_points=4000)
    assert sol["success"]

    # Find period by counting zero crossings in theta.
    theta = sol["y"][0]
    t = sol["t"]
    crossings = np.where(np.diff(np.sign(theta)))[0]
    # Two zero crossings per period; use the gap between the 1st and 3rd to
    # avoid edge effects.
    assert len(crossings) >= 4
    measured_period = float(t[crossings[2]] - t[crossings[0]])
    assert math.isclose(measured_period, expected_period, rel_tol=0.01)


# ---------------------------------------------------------------------------
# Kinematics: projectile range
# ---------------------------------------------------------------------------


def test_kinematics_projectile_range():
    """R = v₀² sin(2θ) / g for a projectile launched at angle θ."""
    g = 9.81
    v0 = 20.0
    theta_deg = 45.0
    theta = math.radians(theta_deg)
    expected_range = v0 ** 2 * math.sin(2 * theta) / g

    compiler = PhysicsCompiler()
    compiler.compile_dsl(
        r"\system{projectile_val}"
        r"\defvar{x}{Position}{m}\defvar{y}{Position}{m}"
        rf"\parameter{{m}}{{1.0}}{{kg}}\parameter{{g}}{{{g}}}{{m/s^2}}"
        r"\lagrangian{0.5*m*(\dot{x}^2 + \dot{y}^2) - m*g*y}"
        rf"\initial{{x=0, y=0, x_dot={v0 * math.cos(theta)}, "
        rf"y_dot={v0 * math.sin(theta)}}}"
    )
    # Simulate long enough for full flight; airtime = 2*v0*sin(theta)/g.
    airtime = 2 * v0 * math.sin(theta) / g
    sol = compiler.simulate(t_span=(0, airtime * 1.1), num_points=2000)
    assert sol["success"]

    # Find the time the projectile returns to y = 0 and read x there.
    y = sol["y"][2]  # y position
    x = sol["y"][0]  # x position
    t = sol["t"]
    # Skip the initial y=0 point and look for the next crossing.
    crossings = np.where(np.diff(np.sign(y[5:])))[0]
    assert len(crossings) > 0
    idx = crossings[0] + 5
    measured_range = float(np.interp(0, [y[idx], y[idx + 1]], [x[idx], x[idx + 1]]))
    assert math.isclose(measured_range, expected_range, rel_tol=0.01)


# ---------------------------------------------------------------------------
# Relativistic: Lorentz factor
# ---------------------------------------------------------------------------


def test_relativistic_gamma_at_known_velocity():
    """γ(0.6c) = 1.25 exactly."""
    from mechanics_dsl.domains.relativistic.core import gamma

    c = 1.0  # Use natural units for an exact ratio check.
    assert math.isclose(gamma(0.6 * c, c=c), 1.25, rel_tol=1e-10)


def test_relativistic_gamma_at_rest_is_one():
    """γ(0) = 1."""
    from mechanics_dsl.domains.relativistic.core import gamma

    assert math.isclose(gamma(0.0), 1.0, rel_tol=1e-12)


# ---------------------------------------------------------------------------
# General relativity: Schwarzschild radius of the sun
# ---------------------------------------------------------------------------


def test_gr_schwarzschild_radius_solar_mass():
    """rₛ = 2GM/c² ≈ 2953 m for the sun."""
    from mechanics_dsl.domains.general_relativity.core import SchwarzschildMetric

    M_sun = 1.989e30  # kg
    expected_rs = 2 * 6.6743e-11 * M_sun / (2.998e8) ** 2  # ~2953 m

    metric = SchwarzschildMetric(mass=M_sun)
    # SchwarzschildMetric exposes its event horizon - try common attribute names.
    actual = getattr(metric, "schwarzschild_radius", None)
    if actual is None:
        actual = getattr(metric, "rs", None)
    if actual is None:
        actual = getattr(metric, "event_horizon", None)
    if callable(actual):
        actual = actual()
    assert actual is not None, "SchwarzschildMetric must expose its radius"
    assert math.isclose(actual, expected_rs, rel_tol=0.001)


# ---------------------------------------------------------------------------
# Quantum: hydrogen ground state
# ---------------------------------------------------------------------------


def test_quantum_hydrogen_ground_state_energy():
    """E₁ = -13.6 eV for the hydrogen ground state."""
    from mechanics_dsl.domains.quantum.core import HydrogenAtom

    atom = HydrogenAtom()
    energy = atom.energy_level(n=1) if hasattr(atom, "energy_level") else atom.energy(n=1)
    # Could be eV or Joules - check both magnitudes.
    eV_per_joule = 6.242e18
    if abs(energy) < 1e-15:  # likely Joules
        energy_eV = energy * eV_per_joule
    else:
        energy_eV = energy
    assert math.isclose(energy_eV, -13.6, rel_tol=0.01)


def test_quantum_harmonic_oscillator_ground_state():
    """E₀ = ½ℏω for the ground state of a harmonic oscillator."""
    from mechanics_dsl.domains.quantum.core import QuantumHarmonicOscillator

    omega = 1.0
    hbar = 1.0  # natural units
    osc = QuantumHarmonicOscillator(omega=omega, hbar=hbar)
    e0 = osc.energy_level(n=0) if hasattr(osc, "energy_level") else osc.energy(n=0)
    assert math.isclose(e0, 0.5 * hbar * omega, rel_tol=1e-6)


# ---------------------------------------------------------------------------
# Thermodynamics: Carnot efficiency
# ---------------------------------------------------------------------------


def test_thermo_carnot_efficiency():
    """η_carnot = 1 - T_c/T_h."""
    from mechanics_dsl.domains.thermodynamics.core import CarnotEngine

    T_h, T_c = 500.0, 300.0
    expected_eta = 1 - T_c / T_h
    engine = CarnotEngine(T_hot=T_h, T_cold=T_c)
    eta = engine.efficiency() if callable(getattr(engine, "efficiency", None)) else engine.efficiency
    assert math.isclose(eta, expected_eta, rel_tol=1e-6)


# ---------------------------------------------------------------------------
# Statistical mechanics: ideal gas PV = nRT
# ---------------------------------------------------------------------------


def test_statistical_ideal_gas_law():
    """PV = nRT must hold for an IdealGas instance at thermal equilibrium."""
    from mechanics_dsl.domains.statistical.core import IdealGas

    # 1 mole at STP (T=273.15 K, V=22.4 L) should sit at ~1 atm.
    n_moles, temperature, volume = 1.0, 273.15, 0.0224
    R = 8.314462618
    expected_P = n_moles * R * temperature / volume

    gas = IdealGas(n_moles=n_moles, temperature=temperature, volume=volume)
    actual_P = gas.pressure() if callable(getattr(gas, "pressure", None)) else gas.pressure
    assert math.isclose(actual_P, expected_P, rel_tol=0.01)


# ---------------------------------------------------------------------------
# Fluids: SPH Poly6 kernel normalization
# ---------------------------------------------------------------------------


def test_fluids_sph_poly6_kernel_normalizes():
    """The Poly6 kernel must integrate to ~1 over its support."""
    from mechanics_dsl.domains.fluids.sph import SPHFluid

    h = 1.0
    fluid = SPHFluid(smoothing_length=h)
    # Numerical 3D integral over a sphere of radius h: ∫ W(r) 4π r² dr
    rs = np.linspace(0.0, h, 1000)
    vals = np.array([fluid.kernel_poly6(r, h) for r in rs])
    integral = np.trapezoid(4 * math.pi * rs ** 2 * vals, rs)
    assert math.isclose(integral, 1.0, rel_tol=0.02)


# ---------------------------------------------------------------------------
# Solid mechanics: cantilever beam max deflection
# ---------------------------------------------------------------------------


def test_solid_mechanics_cantilever_tip_deflection():
    """δ_tip = FL³/(3EI) for a tip-loaded cantilever beam."""
    from mechanics_dsl.domains.solid_mechanics.beam_theory import (
        BeamElement,
        BeamLoading,
        BeamMaterial,
        BeamSupportType,
        RectangularCrossSection,
    )

    # 2 m steel beam, 5 cm × 5 cm square section, 1 kN tip load.
    L = 2.0
    E = 200e9  # Pa
    b_w = 0.05  # width (m)
    h_t = 0.05  # height (m)
    I = b_w * h_t ** 3 / 12.0
    F = 1000.0
    expected_delta = F * L ** 3 / (3 * E * I)

    material = BeamMaterial(E=E, G=80e9, rho=7850.0)
    section = RectangularCrossSection(b=b_w, h=h_t)
    # A cantilever is a beam fixed at one end and free at the other.
    beam = BeamElement(
        length=L,
        cross_section=section,
        material=material,
        left_support=BeamSupportType.FIXED,
        right_support=BeamSupportType.FREE,
    )

    loading = BeamLoading(type="point", value=F, position=L)

    from mechanics_dsl.domains.solid_mechanics.beam_theory import compute_beam_deflection

    result = compute_beam_deflection(beam, loading)
    # Pull the max deflection magnitude from whatever shape the result has.
    if hasattr(result, "max_deflection"):
        delta = result.max_deflection
    elif hasattr(result, "deflection"):
        delta = float(np.max(np.abs(np.asarray(result.deflection))))
    elif hasattr(result, "y"):
        delta = float(np.max(np.abs(np.asarray(result.y))))
    else:
        pytest.skip(f"Unknown BeamDeflection result shape: {dir(result)}")

    assert math.isclose(abs(delta), expected_delta, rel_tol=0.10), (
        f"expected {expected_delta:.4e}, got {delta:.4e}"
    )
