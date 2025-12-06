# MechanicsDSL Examples

30 comprehensive tutorials organized by topic.

## Directory Structure

```
examples/
├── beginner/       # First steps with MechanicsDSL
├── intermediate/   # Multi-body systems and dynamics
├── advanced/       # Complex physics and 3D
├── tools/          # Visualization and performance
├── codegen/        # C++ and WebAssembly generation
├── fluids/         # SPH fluid dynamics
└── celestial/      # Orbital and N-body mechanics
```

---

## 🌱 Beginner (01-04)

| File | Description |
|------|-------------|
| `01_getting_started.py` | Your first simulation |
| `02_harmonic_oscillator.py` | Simple harmonic motion |
| `03_simple_pendulum.py` | Basic pendulum dynamics |
| `04_plotting_basics.py` | Visualization fundamentals |

## 🌿 Intermediate (05-09)

| File | Description |
|------|-------------|
| `05_double_pendulum.py` | Chaotic double pendulum |
| `06_coupled_oscillators.py` | Two coupled springs |
| `07_2d_motion.py` | Projectile and orbital motion |
| `08_damped_systems.py` | Damping and energy loss |
| `09_forced_oscillators.py` | Driven systems and resonance |

## 🌳 Advanced (10-16)

| File | Description |
|------|-------------|
| `10_3d_gyroscope.py` | 3D rigid body rotation |
| `11_spherical_pendulum.py` | 3D pendulum motion |
| `12_constrained_systems.py` | Bead on rotating hoop |
| `13_hamiltonian_formulation.py` | Hamiltonian mechanics |
| `14_chaotic_systems.py` | Duffing oscillator |
| `15_energy_analysis.py` | Energy conservation |
| `16_phase_space.py` | Phase portraits |

## 🔧 Tools (17-20)

| File | Description |
|------|-------------|
| `17_custom_visualizations.py` | Advanced plotting |
| `18_export_import.py` | Saving and loading |
| `19_performance_tuning.py` | Solver optimization |
| `20_units_and_dimensions.py` | Dimensional analysis |

## ⚙️ Code Generation (21-22)

| File | Description |
|------|-------------|
| `21_c++_code_export.py` | Native C++ code |
| `22_advanced_targets.py` | OpenMP parallelization |

## 🌊 Fluid Dynamics (23-26)

| File | Description |
|------|-------------|
| `23_sph_introduction.py` | Dam break simulation |
| `24_sph_wave_tank.py` | Wave propagation |
| `25_sph_droplet.py` | Droplet splash |
| `26_sph_sloshing.py` | Tank sloshing |

## 🚀 Celestial Mechanics (27-30)

| File | Description |
|------|-------------|
| `27_elastic_pendulum.py` | Spring pendulum |
| `28_n_body_gravity.py` | Multi-body gravity |
| `29_orbital_mechanics.py` | Kepler orbits |
| `30_figure8_orbit.py` | Figure-8 three-body orbit |

---

## Running Examples

```bash
# Run a specific example
python examples/beginner/01_getting_started.py

# Run all examples in a category
python -m pytest examples/beginner/ -v
```

## Tips

1. **Start Simple** - Begin with beginner/ and progress
2. **Modify Parameters** - Change values and see results
3. **Read Comments** - Each file is documented
4. **Check Docs** - [mechanicsdsl.readthedocs.io](https://mechanicsdsl.readthedocs.io)
