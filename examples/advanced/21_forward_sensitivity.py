"""
Example 21: Analytic forward sensitivities and an uncertainty-weighted ranking

A rigid segment swings freely about a hinge under gravity (for instance a
forelimb segment about the elbow). Its equation of motion is

    (I_c + m d^2) phi'' + c phi' + m g d cos(beta) sin(phi) = 0

MechanicsDSL derives this from the Lagrangian, and ForwardSensitivity then
differentiates it symbolically to obtain d(phi)/d(theta) for each parameter.

The example builds a "generic" model and a synthetic "specimen" whose true
parameter values differ from the generic ones. It then

  1. computes the RMS angle error E of the generic model against the
     specimen's swings from three release angles, and its exact gradient;
  2. weights each gradient by a plausible parameter uncertainty sigma_i to
     give s_i = sigma_i |dE/dtheta_i| and ranks the parameters;
  3. replaces each generic value with the specimen's value one at a time and
     reports the actual reduction in E, for comparison with the ranking.

The specimen data here are simulated, so this demonstrates the computation,
not an experimental result.
"""

import sys as _sys

try:
    _sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np

from mechanics_dsl import PhysicsCompiler
from mechanics_dsl.inverse.forward_sensitivity import (
    ForwardSensitivity,
    pool_rms,
    uncertainty_weighted_ranking,
)

SOURCE = r"""
\system{hinged_segment}
\defvar{phi}{Angle}{rad}
\parameter{m}{0.40}{kg}
\parameter{d}{0.080}{m}
\parameter{I_c}{0.0012}{kg*m^2}
\parameter{beta}{0.10}{rad}
\parameter{c}{0.0020}{kg*m^2/s}
\parameter{g}{9.81}{m/s^2}
\lagrangian{\frac{1}{2} * (I_c + m * d^2) * \dot{phi}^2 + m * g * d * \cos{beta} * \cos{phi}}
\force{-c * \dot{phi}}
\initial{phi=0.6, phi_dot=0.0}
"""

PARAMS = ["m", "d", "I_c", "beta"]
GENERIC = {"m": 0.40, "d": 0.080, "I_c": 0.0012, "beta": 0.10}
SPECIMEN = {"m": 0.44, "d": 0.074, "I_c": 0.0010, "beta": 0.04}  # simulated
SIGMA = {"m": 0.04, "d": 0.006, "I_c": 0.0002, "beta": 0.05}  # illustrative
RELEASE_ANGLES = [0.4, 0.6, 0.8]  # rad
NOISE_SD = 0.002  # rad, simulated motion-capture noise


def pooled_error(fs, t, measured, values):
    parts = []
    for phi0, meas in zip(RELEASE_ANGLES, measured):
        res = fs.solve(t, param_values=values, initial_conditions={"phi": phi0, "phi_dot": 0.0})
        parts.append(res.rms_error(meas, "phi"))
    return pool_rms(parts)


def main():
    compiler = PhysicsCompiler()
    result = compiler.compile_dsl(SOURCE)
    if not result["success"]:
        raise SystemExit(result.get("error"))

    print("Equation of motion derived by MechanicsDSL:")
    print("  phi_ddot =", compiler.equations["phi_ddot"])

    fs = ForwardSensitivity(compiler, params=PARAMS)
    print("\nSymbolic d(phi_ddot)/d(theta):")
    for j, name in enumerate(PARAMS):
        print(f"  {name:5s}: {fs.jac_p[1, j]}")

    # Simulated specimen swings, two seconds each.
    t = np.linspace(0.0, 2.0, 401)
    rng = np.random.default_rng(1)
    measured = []
    for phi0 in RELEASE_ANGLES:
        res = fs.solve(t, param_values=SPECIMEN, initial_conditions={"phi": phi0, "phi_dot": 0.0})
        measured.append(res.state("phi") + rng.normal(0.0, NOISE_SD, t.size))

    # 1-2. Prospective ranking at the generic values.
    err = pooled_error(fs, t, measured, GENERIC)
    ranking = uncertainty_weighted_ranking(err.gradient, SIGMA)
    print(f"\nGeneric model: E = {np.degrees(err.rms):.3f} deg")
    print("\nPredicted ranking, s_i = sigma_i |dE/dtheta_i| (deg):")
    for name, s in ranking:
        print(f"  {name:5s}  {np.degrees(s):8.4f}")

    # 3. Actual error reduction from substituting each specimen value alone.
    print("\nActual reduction in E from substituting one specimen value (deg):")
    reductions = {}
    for name in PARAMS:
        values = dict(GENERIC, **{name: SPECIMEN[name]})
        reductions[name] = err.rms - pooled_error(fs, t, measured, values).rms
    for name, dE in sorted(reductions.items(), key=lambda kv: kv[1], reverse=True):
        print(f"  {name:5s}  {np.degrees(dE):8.4f}")

    all_values = dict(GENERIC, **SPECIMEN)
    print(
        "\nAll four substituted together: E = "
        f"{np.degrees(pooled_error(fs, t, measured, all_values).rms):.3f} deg"
    )


if __name__ == "__main__":
    main()
