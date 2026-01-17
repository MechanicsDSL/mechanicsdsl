# MechanicsDSL Example Files

Standalone `.mdsl` files for use with the CLI.

## Usage

```bash
# Run with animation
mechanicsdsl run pendulum.mdsl --t-span 0,20 --animate

# Compile to C++
mechanicsdsl compile double_pendulum.mdsl --target cpp

# Export results to CSV
mechanicsdsl export kepler_orbit.mdsl --format csv --output orbit.csv
```

## Files

| File | Description | Domains |
|------|-------------|---------|
| `pendulum.mdsl` | Simple pendulum | Classical |
| `double_pendulum.mdsl` | Chaotic double pendulum | Classical, Chaos |
| `damped_oscillator.mdsl` | Damped harmonic motion | Classical, Dissipation |
| `coupled_oscillators.mdsl` | Energy transfer between masses | Classical, Coupled |
| `kepler_orbit.mdsl` | Planetary orbit | Classical, Central Force |

## DSL Syntax

```
\system{name}              % System name
\defvar{q}{description}    % Generalized coordinate
\parameter{p}{value}{unit} % Physical parameter
\lagrangian{expr}          % Lagrangian L = T - V
\initial{q=value}          % Initial conditions
```

See the [full DSL reference](https://mechanicsdsl.readthedocs.io/en/latest/dsl_reference.html) for complete documentation.
