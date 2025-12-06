# MechanicsDSL Examples and Tutorials

This directory contains 30 comprehensive tutorials organized by topic and difficulty.

## Tutorial Structure

### Beginner Level
Start here if you're new to MechanicsDSL.

| # | Tutorial | Description |
|---|----------|-------------|
| 01 | [Getting Started](01_getting_started.py) | Your first simulation |
| 02 | [Harmonic Oscillator](02_harmonic_oscillator.py) | Simple harmonic motion |
| 03 | [Simple Pendulum](03_simple_pendulum.py) | Basic pendulum dynamics |
| 04 | [Plotting Basics](04_plotting_basics.py) | Visualization fundamentals |

### Intermediate Level
For users comfortable with basic systems.

| # | Tutorial | Description |
|---|----------|-------------|
| 05 | [Double Pendulum](05_double_pendulum.py) | Chaotic double pendulum |
| 06 | [Coupled Oscillators](06_coupled_oscillators.py) | Two coupled springs |
| 07 | [2D Motion](07_2d_motion.py) | Projectile and orbital motion |
| 08 | [Damped Systems](08_damped_systems.py) | Damping and energy dissipation |
| 09 | [Forced Oscillators](09_forced_oscillators.py) | Driven systems and resonance |

### Advanced Level
Complex systems requiring deeper understanding.

| # | Tutorial | Description |
|---|----------|-------------|
| 10 | [3D Gyroscope](10_3d_gyroscope.py) | 3D rigid body rotation |
| 11 | [Spherical Pendulum](11_spherical_pendulum.py) | 3D pendulum motion |
| 12 | [Constrained Systems](12_constrained_systems.py) | Bead on rotating hoop |
| 13 | [Hamiltonian Formulation](13_hamiltonian_formulation.py) | Hamiltonian mechanics |
| 14 | [Chaotic Systems](14_chaotic_systems.py) | Duffing oscillator |

### Specialized Topics
Advanced features and techniques.

| # | Tutorial | Description |
|---|----------|-------------|
| 15 | [Energy Analysis](15_energy_analysis.py) | Energy conservation analysis |
| 16 | [Phase Space](16_phase_space.py) | Phase space visualization |
| 17 | [Custom Visualizations](17_custom_visualizations.py) | Advanced plotting techniques |
| 18 | [Export/Import](18_export_import.py) | Saving and loading systems |
| 19 | [Performance Tuning](19_performance_tuning.py) | Solver optimization |
| 20 | [Units and Dimensions](20_units_and_dimensions.py) | Dimensional analysis |

### Code Generation
Generating high-performance code.

| # | Tutorial | Description |
|---|----------|-------------|
| 21 | [C++ Export](21_c++_code_export.py) | Native C++ code generation |
| 22 | [Advanced Targets](22_advanced_targets.py) | OpenMP (soon WebAssembly) |

### Fluid Dynamics (SPH)
Smoothed Particle Hydrodynamics simulations.

| # | Tutorial | Description |
|---|----------|-------------|
| 23 | [SPH Introduction](23_sph_introduction.py) | Dam break simulation |
| 24 | [SPH Wave Tank](24_sph_wave_tank.py) | Wave generation and propagation |
| 25 | [SPH Droplet](25_sph_droplet.py) | Droplet impact and splash |
| 26 | [SPH Sloshing](26_sph_sloshing.py) | Tank sloshing dynamics |

### Celestial Mechanics
Gravitational dynamics and orbital systems.

| # | Tutorial | Description |
|---|----------|-------------|
| 27 | [Elastic Pendulum](27_elastic_pendulum.py) | Spring pendulum dynamics |
| 28 | [N-Body Gravity](28_n_body_gravity.py) | Multi-body gravitational systems |
| 29 | [Orbital Mechanics](29_orbital_mechanics.py) | Kepler orbits and transfers |
| 30 | [Figure-8 Orbit](30_figure8_orbit.py) | Famous 3-body periodic solution |

---

## Quick Start

```bash
# Run a specific example
python examples/01_getting_started.py

# Run all examples (with pytest)
pytest examples/ -v
```

## Tips

1. **Start Simple** - Begin with example 01 and work your way up
2. **Modify Examples** - Change parameters and see what happens
3. **Read the Comments** - Each example is heavily commented
4. **Experiment** - Combine features from different examples
5. **Check Documentation** - See [mechanicsdsl.readthedocs.io](https://mechanicsdsl.readthedocs.io)

## Troubleshooting

If an example doesn't run:
1. Install MechanicsDSL: `pip install mechanicsdsl-core`
2. Check dependencies: `pip install numpy scipy sympy matplotlib`
3. For SPH examples: requires C++ compiler (optional)
