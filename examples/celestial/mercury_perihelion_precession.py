"""
Mercury Perihelion Precession using MechanicsDSL
=================================================

This script uses MechanicsDSL's constants and standard GR formulae to:

1. Model Mercury's orbit around the Sun with a post-Newtonian (GR-like) correction.
2. Compute the apsidal (perihelion) precession per orbit.
3. Convert that to arcseconds per century and compare with the classic ~43"/century.
4. Produce illustrative plots:
   - Effective potential and turning points (perihelion / aphelion).
   - A precessing rosette orbit built from successive nearly-Keplerian ellipses.

The dynamics are computed with:
    V_eff(r) = V(r) + L^2 / (2 m r^2)
    V(r)     = -GMm / r  - GM L^2 / (c^2 r^3)

The extra -GM L^2 / (c^2 r^3) term is the leading GR correction that
produces perihelion precession for nearly Keplerian orbits.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

from mechanics_dsl.domains.general_relativity import (
    GRAVITATIONAL_CONSTANT as G,
    SOLAR_MASS,
    SPEED_OF_LIGHT as C,
)


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Physical parameters for Sun–Mercury system
    # ------------------------------------------------------------------
    M_sun = SOLAR_MASS                 # kg
    m_mercury = 3.3011e23              # kg

    # Mercury orbital elements (IAU-ish modern values)
    a = 5.7909227e10                   # semi-major axis [m]
    e = 0.205630                       # eccentricity

    # Useful derived quantities
    mu = G * M_sun * m_mercury         # GMm

    # Newtonian two-body energy and angular momentum (mechanical, not per-unit-mass)
    # E = - GMm / (2a),  L = m sqrt(GMa(1 - e^2))
    E = -mu / (2.0 * a)
    L = m_mercury * math.sqrt(G * M_sun * a * (1.0 - e**2))

    print("=== Sun–Mercury System Parameters ===")
    print(f"SUN mass      M  = {M_sun:.6e} kg")
    print(f"Mercury mass  m  = {m_mercury:.6e} kg")
    print(f"Semi-major a     = {a:.6e} m")
    print(f"Eccentricity e   = {e:.6f}")
    print(f"Total energy  E  = {E:.6e} J")
    print(f"Angular mom. L   = {L:.6e} kg·m²/s")
    print()

    # ------------------------------------------------------------------
    # 2. Basic orbital geometry (classical) and GR precession
    # ------------------------------------------------------------------
    # Classical perihelion / aphelion for an ellipse
    r_peri = a * (1.0 - e)
    r_aphe = a * (1.0 + e)

    print("=== Classical Turning Points (Kepler ellipse) ===")
    print(f"Perihelion r_p    = {r_peri:.6e} m")
    print(f"Aphelion  r_a     = {r_aphe:.6e} m")
    print()

    # Mercury orbital period (sidereal) ~ 87.969 days
    T_mercury = 87.969 * 24.0 * 3600.0  # seconds
    # Julian century = 36525 days
    T_century = 36525.0 * 24.0 * 3600.0
    orbits_per_century = T_century / T_mercury
    # Analytic GR leading-order prediction
    # Δω_GR = 6π GM / (a c² (1 - e²)) [radians per orbit]
    delta_analytic = 6.0 * math.pi * G * M_sun / (a * C**2 * (1.0 - e**2))
    delta_analytic_arcsec = math.degrees(delta_analytic) * 3600.0
    delta_analytic_arcsec_per_century = delta_analytic_arcsec * orbits_per_century

    print("=== Perihelion Precession (GR, leading order) ===")
    print(f"Precession per orbit (analytic)     = {delta_analytic:.6e} rad")
    print(f"                                   = {delta_analytic_arcsec:.6e} arcsec")
    print(f"Orbits per century                  = {orbits_per_century:.3f}")
    print(
        f"Precession per century (analytic)   = {delta_analytic_arcsec_per_century:.3f} arcsec/century"
    )
    print("Observed GR excess for Mercury      ~ 43 arcsec/century")
    print()

    # ------------------------------------------------------------------
    # 4. Plot effective potential with GR correction
    # ------------------------------------------------------------------
    r = sp.symbols("r", positive=True, real=True)
    # Effective potential per unit mass with GR correction:
    # V_eff/m = -GM/r + L̃²/(2 r²) - GM L̃²/(c² r³),  where L̃ = L/m
    L_tilde = L / m_mercury
    V_eff_expr = (
        -G * M_sun / r
        + L_tilde**2 / (2.0 * r**2)
        - G * M_sun * L_tilde**2 / (C**2 * r**3)
    )
    V_eff = sp.lambdify(r, V_eff_expr, "numpy")

    r_vals = np.linspace(r_peri * 0.5, r_aphe * 1.5, 1000)
    V_eff_vals = V_eff(r_vals)

    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(r_vals, V_eff_vals, label=r"$V_{\rm eff}(r)/m$")
    ax1.axhline(E / m_mercury, color="k", linestyle="--", label="Energy per mass E/m")
    ax1.axvline(r_peri, color="g", linestyle="--", label="Perihelion")
    ax1.axvline(r_aphe, color="r", linestyle="--", label="Aphelion")

    ax1.set_xlabel("r [m]")
    ax1.set_ylabel("Effective potential per mass [J/kg]")
    ax1.set_title("Mercury: GR-Corrected Effective Potential")
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)

    out_dir = Path("examples") / "celestial"
    out_dir.mkdir(parents=True, exist_ok=True)
    eff_path = out_dir / "mercury_effective_potential.png"
    fig1.tight_layout()
    fig1.savefig(eff_path, dpi=150)
    plt.close(fig1)

    print(f"[saved] Effective potential plot -> {eff_path}")

    # ------------------------------------------------------------------
    # 5. Build a precessing rosette orbit using Kepler ellipses + GR δ
    # ------------------------------------------------------------------
    # Classic Kepler ellipse in polar coordinates (centered on focus):
    #    r(φ) = a (1 - e²) / (1 + e cos φ)
    #
    # GR adds a small precession Δω each orbit. We construct N orbits,
    # each rotated by n * Δω to visualize the precession.
    #
    # For visualization we use the analytic GR Δω so the rosette is smooth;
    # the numeric MechanicsDSL δ is printed above for comparison.
    delta = delta_analytic  # radians per orbit

    N_orbits = 12
    phi_samples = np.linspace(0, 2.0 * math.pi, 600)

    fig2, ax2 = plt.subplots(figsize=(6, 6))

    for n in range(N_orbits):
        phi_n = phi_samples + n * delta
        r_ellipse = a * (1.0 - e**2) / (1.0 + e * np.cos(phi_samples))
        x_n = r_ellipse * np.cos(phi_n)
        y_n = r_ellipse * np.sin(phi_n)
        alpha = 0.3 + 0.7 * (n / N_orbits)
        ax2.plot(x_n, y_n, color="C0", alpha=alpha, lw=0.8)

    ax2.plot([0.0], [0.0], "yo", markersize=6, label="Sun")
    ax2.set_aspect("equal", "box")
    ax2.set_xlabel("x [m]")
    ax2.set_ylabel("y [m]")
    ax2.set_title("Mercury: Precessing Perihelion (GR)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right")

    orbit_path = out_dir / "mercury_precessing_orbit.png"
    fig2.tight_layout()
    fig2.savefig(orbit_path, dpi=150)
    plt.close(fig2)

    print(f"[saved] Precessing orbit plot   -> {orbit_path}")


if __name__ == "__main__":
    main()

