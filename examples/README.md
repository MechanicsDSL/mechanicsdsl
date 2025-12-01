# MechanicsDSL Examples and Tutorials

This directory contains comprehensive tutorials and examples for using MechanicsDSL, organized by difficulty level and topic.

## Tutorial Structure

### Beginner Level
Start here if you're new to MechanicsDSL or classical mechanics simulations.

1. **[01_getting_started.py](01_getting_started.py)** - Your first simulation
2. **[02_harmonic_oscillator.py](02_harmonic_oscillator.py)** - Simple harmonic motion
3. **[03_simple_pendulum.py](03_simple_pendulum.py)** - Basic pendulum dynamics
4. **[04_plotting_basics.py](04_plotting_basics.py)** - Visualization fundamentals

### Intermediate Level
For users comfortable with basic systems.

5. **[05_double_pendulum.py](05_double_pendulum.py)** - Chaotic double pendulum
6. **[06_coupled_oscillators.py](06_coupled_oscillators.py)** - Two coupled springs
7. **[07_2d_motion.py](07_2d_motion.py)** - Projectile and orbital motion
8. **[08_damped_systems.py](08_damped_systems.py)** - Damping and energy dissipation
9. **[09_forced_oscillators.py](09_forced_oscillators.py)** - Driven systems

### Advanced Level
Complex systems requiring deeper understanding.

10. **[10_3d_gyroscope.py](10_3d_gyroscope.py)** - 3D rigid body rotation
11. **[11_spherical_pendulum.py](11_spherical_pendulum.py)** - 3D pendulum motion
12. **[12_constrained_systems.py](12_constrained_systems.py)** - Systems with constraints
13. **[13_hamiltonian_formulation.py](13_hamiltonian_formulation.py)** - Hamiltonian mechanics
14. **[14_chaotic_systems.py](14_chaotic_systems.py)** - Lorenz and Rössler attractors

### Specialized Topics
Advanced features and techniques.

15. **[15_energy_analysis.py](15_energy_analysis.py)** - Energy conservation analysis
16. **[16_phase_space.py](16_phase_space.py)** - Phase space visualization
17. **[17_custom_visualizations.py](17_custom_visualizations.py)** - Advanced plotting
18. **[18_export_import.py](18_export_import.py)** - Saving and loading systems
19. **[19_performance_tuning.py](19_performance_tuning.py)** - Optimization tips
20. **[20_units_and_dimensions.py](20_units_and_dimensions.py)** - Unit system usage

## Quick Start

### Running Examples

```bash
# Run a specific example
python examples/01_getting_started.py

# Run all examples (if you have pytest)
pytest examples/ -v
```

### In Jupyter/Colab

Each example can be run in a Jupyter notebook. Simply copy the code into a cell and execute.

## Tips

1. **Start Simple**: Begin with example 01 and work your way up
2. **Modify Examples**: Change parameters and see what happens
3. **Read the Comments**: Each example is heavily commented
4. **Experiment**: Try combining features from different examples
5. **Check Documentation**: See the main README.md for API details

## Troubleshooting

If an example doesn't run:
1. Make sure MechanicsDSL is installed: `pip install -e .`
2. Check that all dependencies are installed: `pip install numpy scipy sympy matplotlib`
3. Verify the DSL syntax matches the examples exactly
4. Check the error messages - they're usually helpful!

## Contributing Examples

Found a cool system? Want to share an example? Contributions welcome!

1. Follow the naming convention: `XX_description.py`
2. Include comprehensive comments
3. Add a docstring explaining what the system does
4. Test that it runs successfully

## Related Documentation

- [Main README](../README.md) - Package overview
