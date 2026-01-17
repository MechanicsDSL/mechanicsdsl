# MechanicsDSL - Domänenspezifische Sprache für Mechanik

> 🚧 **Übersetzung in Arbeit** — Wir freuen uns über Beiträge zur Verbesserung dieser Übersetzung!

MechanicsDSL ist eine domänenspezifische Sprache und ein Compiler-Framework für die computergestützte Physik.

## Hauptfunktionen

- **Symbolische Ableitung** — Automatische Ableitung der Euler-Lagrange-Gleichungen aus dem Lagrangian
- **Multi-Target-Codegenerierung** — Export nach C++, CUDA, Rust, Julia und 8 weitere Sprachen
- **GPU-Beschleunigung** — 70-fache Beschleunigung über JAX-Backend
- **9 Physikdomänen** — Klassische Mechanik, Quantenmechanik, Relativität, Fluiddynamik und mehr

## Installation

```bash
pip install mechanicsdsl-core
```

## Schnellstart

```python
from mechanics_dsl import PhysicsCompiler

dsl_code = r"""
\system{simple_pendulum}
\defvar{theta}{Winkel}{rad}
\parameter{m}{1.0}{kg}
\parameter{l}{1.0}{m}
\parameter{g}{9.81}{m/s^2}
\lagrangian{
    \frac{1}{2} * m * l^2 * \dot{theta}^2 
    - m * g * l * (1 - \cos{theta})
}
\initial{theta=2.5, theta_dot=0.0}
"""

compiler = PhysicsCompiler()
compiler.compile_dsl(dsl_code)
solution = compiler.simulate(t_span=(0, 10))
compiler.animate(solution)
```

## Dokumentation

Vollständige Dokumentation unter [mechanicsdsl.readthedocs.io](https://mechanicsdsl.readthedocs.io)

## Lizenz

MIT-Lizenz — Frei verwendbar für kommerzielle und akademische Projekte.

---

*Diese Übersetzung ist ein Community-Beitrag. Bei Fragen eröffnen Sie bitte ein Issue.*
