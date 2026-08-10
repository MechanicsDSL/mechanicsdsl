"""
Stress-system generators for MechanicsDSL.

Each generator yields case dicts describing one mechanical system dialed to a
given difficulty level along one stress axis:

    {
      "axis": str,               # stress axis id
      "level": int,              # difficulty index (0 = easiest)
      "knob": <value>,           # the dialed quantity (N, eps, ratio, depth...)
      "name": str,               # unique case name
      "dsl": str,                # DSL source (includes \\initial)
      "tools": [str],            # pathways to test: lagrangian/hamiltonian/constrained
      "formulation": str,        # "unconstrained" | "constrained"
      "conservative": bool,      # energy oracle applies
      "expected_moving": bool,   # a correct sim must not stay frozen at the IC
      "t_span": [float,float],
      "num_points": int,
    }

Coordinates are named theta0.. (Angle) or x0.. (Position) so the compiler's
is_likely_coordinate() treats them as generalized coordinates.

Tool applicability is deliberate:
  * Constraint axes (loops, redundancy) test ONLY the constrained pathway --
    that is the pathway designed for them. Running the plain Lagrangian pathway
    on a constrained system just silently drops the constraints, which the
    energy oracle (built from the same unconstrained Lagrangian) cannot see, so
    including it would produce a misleading 0%.
  * Unconstrained axes test the Lagrangian pathway, plus the Hamiltonian
    pathway where a Legendre transform is meaningful.
"""

from __future__ import annotations

import math
from typing import List


# ==========================================================================
# Axis 1: Degrees of freedom -- serial N-pendulum (dense coupled mass matrix).
# ==========================================================================
def n_pendulum_dsl(N: int) -> str:
    coords = [f"theta{i}" for i in range(N)]
    lines = [r"\system{npend_%d}" % N]
    for c in coords:
        lines.append(r"\defvar{%s}{Angle}{rad}" % c)
    lines += [r"\parameter{m}{1.0}{kg}", r"\parameter{l}{1.0}{m}",
              r"\parameter{g}{9.81}{m/s^2}"]

    ke_terms: List[str] = []
    for j in range(N):
        for k in range(N):
            coeff = N - max(j, k)
            if coeff == 0:
                continue
            ke_terms.append(
                r"%s*m*l^2*\cos{theta%d - theta%d}*\dot{theta%d}*\dot{theta%d}"
                % (0.5 * coeff, j, k, j, k))
    pe_terms = [r"%s*m*g*l*(1 - \cos{theta%d})" % (float(N - j), j) for j in range(N)]
    lines.append(r"\lagrangian{(%s) - (%s)}" % (" + ".join(ke_terms), " + ".join(pe_terms)))

    ic = []
    for i, c in enumerate(coords):
        ic += [f"{c}={0.3 if i == 0 else 0.15}", f"{c}_dot=0.0"]
    lines.append(r"\initial{%s}" % ", ".join(ic))
    return "\n".join(lines)


def axis_dof():
    out = []
    for level, N in enumerate([1, 2, 3, 4, 5]):
        out.append(dict(
            axis="dof", level=level, knob=N, name=f"dof_N{N}",
            dsl=n_pendulum_dsl(N), tools=["lagrangian", "hamiltonian"],
            formulation="unconstrained", conservative=True, expected_moving=True,
            t_span=[0.0, 10.0], num_points=1500))
    return out


# ==========================================================================
# Axis 2: Closed loops -- a genuine closed kinematic chain.
# N point masses form a closed loop of rigid rods; node 0 is pinned at the
# origin, so the moving nodes are p1..p_{N-1} in Cartesian coordinates and the
# loop is held by N quadratic rod-length constraints (including the closure
# rod p_{N-1}->p0). The kinetic term is a trivial diagonal, so ALL the
# difficulty is the loop-closure constraint structure. Validated: N=3 (pinned
# triangle) moves, conserves energy, and holds its constraints.
# ==========================================================================
def closed_chain_dsl(N: int) -> str:
    coords = []
    for i in range(1, N):
        coords += [f"x{i}", f"y{i}"]
    lines = [r"\system{cchain_%d}" % N]
    for c in coords:
        lines.append(r"\defvar{%s}{Position}{m}" % c)
    lines += [r"\parameter{m}{1.0}{kg}", r"\parameter{g}{9.81}{m/s^2}"]
    ke = " + ".join(r"0.5*m*(\dot{x%d}^2 + \dot{y%d}^2)" % (i, i) for i in range(1, N))
    pe = " + ".join(r"m*g*y%d" % i for i in range(1, N))
    lines.append(r"\lagrangian{(%s) - (%s)}" % (ke, pe))

    def px(i):
        return "0.0" if i == 0 else f"x{i}"

    def py(i):
        return "0.0" if i == 0 else f"y{i}"

    edge = 2.0 * math.sin(math.pi / N)  # edge of a unit-circumradius regular N-gon
    for i in range(N):
        j = (i + 1) % N
        lines.append(r"\constraint{(%s - %s)^2 + (%s - %s)^2 - %s}"
                     % (px(j), px(i), py(j), py(i), repr(edge * edge)))

    pts = [(math.cos(2 * math.pi * k / N), math.sin(2 * math.pi * k / N)) for k in range(N)]
    x0, y0 = pts[0]
    pts = [(x - x0, y - y0) for x, y in pts]  # translate node0 to origin
    omega = 0.4  # rigid rotation about the pin -> off-equilibrium, so it must move
    ic = []
    for i in range(1, N):
        x, y = pts[i]
        ic += [f"x{i}={round(x, 4)}", f"y{i}={round(y, 4)}",
               f"x{i}_dot={round(-omega * y, 4)}", f"y{i}_dot={round(omega * x, 4)}"]
    lines.append(r"\initial{%s}" % ", ".join(ic))
    return "\n".join(lines)


def axis_loops():
    out = []
    for level, N in enumerate([3, 4, 5]):
        out.append(dict(
            axis="loops", level=level, knob=N, name=f"loops_N{N}",
            dsl=closed_chain_dsl(N), tools=["constrained"],
            formulation="constrained", conservative=True, expected_moving=True,
            t_span=[0.0, 3.0], num_points=600))
    return out


# ==========================================================================
# Axis 3: Constraint redundancy -- circle constraint + R dependent duplicates.
# ==========================================================================
def redundant_dsl(R: int) -> str:
    lines = [r"\system{redund_%d}" % R,
             r"\defvar{x}{Position}{m}", r"\defvar{y}{Position}{m}",
             r"\parameter{m}{1.0}{kg}", r"\parameter{g}{9.81}{m/s^2}"]
    lines.append(r"\lagrangian{0.5*m*(\dot{x}^2 + \dot{y}^2) - m*g*y}")
    lines.append(r"\constraint{x^2 + y^2 - 1}")
    for k in range(R):
        mult = k + 2
        lines.append(r"\constraint{%d*x^2 + %d*y^2 - %d}" % (mult, mult, mult))
    lines.append(r"\initial{x=1.0, y=0.0, x_dot=0.0, y_dot=1.0}")
    return "\n".join(lines)


def axis_redundancy():
    out = []
    for level, R in enumerate([0, 1, 2, 3, 4, 6, 8]):
        out.append(dict(
            axis="redundancy", level=level, knob=R, name=f"redund_R{R}",
            dsl=redundant_dsl(R), tools=["constrained"],
            formulation="constrained", conservative=True, expected_moving=True,
            t_span=[0.0, 5.0], num_points=1000))
    return out


# ==========================================================================
# Axis 4: Near-singular mass matrix. M(eps)=[[1,1-eps],[1-eps,1]], det->0.
# ==========================================================================
def near_singular_dsl(eps: float) -> str:
    c = 1.0 - eps
    lines = [r"\system{nearsing}",
             r"\defvar{x}{Position}{m}", r"\defvar{y}{Position}{m}",
             r"\parameter{m}{1.0}{kg}", r"\parameter{k}{1.0}{N/m}"]
    lines.append(
        r"\lagrangian{0.5*m*\dot{x}^2 + 0.5*m*\dot{y}^2 + %s*m*\dot{x}*\dot{y} "
        r"- 0.5*k*x^2 - 0.5*k*y^2}" % repr(c))
    lines.append(r"\initial{x=1.0, y=0.0, x_dot=0.0, y_dot=0.0}")
    return "\n".join(lines)


def axis_near_singular():
    out = []
    for level, eps in enumerate([1e-1, 1e-2, 1e-3, 1e-5, 1e-8, 1e-11, 0.0]):
        out.append(dict(
            axis="near_singular", level=level, knob=eps, name=f"nearsing_e{eps:g}",
            dsl=near_singular_dsl(eps), tools=["lagrangian"],
            formulation="unconstrained", conservative=True, expected_moving=True,
            t_span=[0.0, 10.0], num_points=1500))
    return out


# ==========================================================================
# Axis 5: Mass-ratio conditioning. Two masses + spring, m2 = 10^k.
# ==========================================================================
def mass_ratio_dsl(ratio: float) -> str:
    lines = [r"\system{massratio}",
             r"\defvar{x}{Position}{m}", r"\defvar{y}{Position}{m}",
             r"\parameter{m1}{1.0}{kg}", r"\parameter{m2}{%s}{kg}" % repr(float(ratio)),
             r"\parameter{k}{1.0}{N/m}", r"\parameter{k1}{1.0}{N/m}"]
    lines.append(
        r"\lagrangian{0.5*m1*\dot{x}^2 + 0.5*m2*\dot{y}^2 "
        r"- 0.5*k*(x - y)^2 - 0.5*k1*x^2}")
    lines.append(r"\initial{x=1.0, y=0.0, x_dot=0.0, y_dot=0.0}")
    return "\n".join(lines)


def axis_mass_ratio():
    out = []
    for level, kexp in enumerate([0, 3, 6, 9, 12, 14, 16]):
        ratio = 10.0 ** kexp
        out.append(dict(
            axis="mass_ratio", level=level, knob=ratio, name=f"massratio_1e{kexp}",
            dsl=mass_ratio_dsl(ratio), tools=["lagrangian", "hamiltonian"],
            formulation="unconstrained", conservative=True, expected_moving=True,
            t_span=[0.0, 20.0], num_points=2000))
    return out


# ==========================================================================
# Axis 6: Symbolic pathology -- V = 1 - cos(cos(...cos(theta)...)) depth D.
# ==========================================================================
def nested_dsl(depth: int) -> str:
    inner = "theta0"
    for _ in range(depth):
        inner = r"\cos{%s}" % inner
    lines = [r"\system{nested_%d}" % depth,
             r"\defvar{theta0}{Angle}{rad}",
             r"\parameter{m}{1.0}{kg}", r"\parameter{l}{1.0}{m}",
             r"\parameter{g}{9.81}{m/s^2}"]
    lines.append(r"\lagrangian{0.5*m*l^2*\dot{theta0}^2 - m*g*l*(1 - %s)}" % inner)
    lines.append(r"\initial{theta0=0.4, theta0_dot=0.0}")
    return "\n".join(lines)


def axis_symbolic():
    out = []
    for level, depth in enumerate([1, 2, 4, 8, 12, 16, 24]):
        out.append(dict(
            axis="symbolic", level=level, knob=depth, name=f"nested_d{depth}",
            dsl=nested_dsl(depth), tools=["lagrangian", "hamiltonian"],
            formulation="unconstrained", conservative=True, expected_moving=True,
            t_span=[0.0, 10.0], num_points=1000))
    return out


def all_cases():
    cases = []
    cases += axis_dof()
    cases += axis_loops()
    cases += axis_redundancy()
    cases += axis_near_singular()
    cases += axis_mass_ratio()
    cases += axis_symbolic()
    return cases


AXES = ["dof", "loops", "redundancy", "near_singular", "mass_ratio", "symbolic"]
TOOLS = ["lagrangian", "hamiltonian", "constrained"]

# (case name, tool) pairs to skip without spending the wall clock, recorded as
# "skipped" by `run.py --skip-known-slow` and excluded from every denominator.
#
# Deliberately EMPTY. It was populated on the assumption that the high-DOF
# Hamiltonian cases hang past 600 s, and that assumption did not survive
# contact with the data -- the numeric mass-matrix path changed the scaling,
# and dof_N4/N5 on the Lagrangian side went from timing out at 180 s to
# finishing in under 12 s. Nothing is currently known to be slow enough to
# justify skipping it, and a case skipped on a stale belief is worse than one
# that costs three minutes to measure.
#
# Cases are never deleted from the sweep regardless: the knob value at which a
# pathway stops returning is the scaling-wall result, reported separately from
# the silent-failure rate.
KNOWN_SLOW: set = set()


if __name__ == "__main__":
    for c in all_cases():
        print(f"{c['axis']:14s} L{c['level']} {c['name']:18s} "
              f"knob={c['knob']!r:10s} tools={c['tools']}")
