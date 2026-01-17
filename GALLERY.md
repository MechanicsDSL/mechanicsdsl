# MechanicsDSL Examples Gallery

A curated collection of what MechanicsDSL can simulate across its 9 physics domains.

---

## 🔄 Classical Mechanics

### Simple Pendulum
The classic introductory system — a mass on a massless string.

```
\system{simple_pendulum}
\defvar{theta}{rad}
\parameter{m}{1.0}{kg}
\parameter{l}{1.0}{m}
\parameter{g}{9.81}{m/s^2}
\lagrangian{\frac{1}{2}*m*l^2*\dot{theta}^2 - m*g*l*(1-\cos{theta})}
\initial{theta=2.5, theta_dot=0}
```

**Key features demonstrated:**
- Lagrangian formulation
- Nonlinear oscillations (large amplitude)
- Energy conservation

---

### Double Pendulum — Chaos
Two pendulums attached end-to-end. Simple setup, chaotic motion.

```
\system{double_pendulum}
\defvar{theta1}{rad}
\defvar{theta2}{rad}
\parameter{m1}{1.0}{kg}
\parameter{m2}{1.0}{kg}
\parameter{l1}{1.0}{m}
\parameter{l2}{1.0}{m}
\parameter{g}{9.81}{m/s^2}
\lagrangian{
    \frac{1}{2}*(m1+m2)*l1^2*\dot{theta1}^2
    + \frac{1}{2}*m2*l2^2*\dot{theta2}^2
    + m2*l1*l2*\dot{theta1}*\dot{theta2}*\cos{theta1-theta2}
    - (m1+m2)*g*l1*(1-\cos{theta1})
    - m2*g*l2*(1-\cos{theta2})
}
\initial{theta1=2.5, theta2=2.0}
```

**Key features demonstrated:**
- Coupled coordinates
- Sensitivity to initial conditions (Lyapunov exponent λ ≈ 2.4 s⁻¹)
- Phase space visualization

---

### Figure-8 Three-Body Orbit
The famous Chenciner-Montgomery solution (2000) where three equal masses chase each other in a figure-8 pattern.

```
\system{figure8_orbit}
\defvar{x1}{m}  \defvar{y1}{m}
\defvar{x2}{m}  \defvar{y2}{m}
\defvar{x3}{m}  \defvar{y3}{m}
\parameter{m}{1.0}{kg}
\parameter{G}{1.0}{N*m^2/kg^2}
\lagrangian{
    0.5*m*(\dot{x1}^2 + \dot{y1}^2 + \dot{x2}^2 + \dot{y2}^2 + \dot{x3}^2 + \dot{y3}^2)
    + G*m^2/\sqrt{(x1-x2)^2 + (y1-y2)^2}
    + G*m^2/\sqrt{(x2-x3)^2 + (y2-y3)^2}
    + G*m^2/\sqrt{(x1-x3)^2 + (y1-y3)^2}
}
```

**Validation:**
- Period T = 6.32591398...
- After 100 periods: position error 6.31 × 10⁻⁶
- Energy conservation: |ΔE/E₀| < 10⁻¹²

---

## 💧 Fluid Dynamics

### Dam Break (SPH)
A column of water collapses and splashes using Smoothed Particle Hydrodynamics.

```
\system{dam_break}
\parameter{h}{0.04}{m}
\parameter{rho0}{1000}{kg/m^3}
\parameter{g}{9.81}{m/s^2}

\fluid{water}
\region{rectangle}{x=0.0..0.4, y=0.0..0.8}
\particle_mass{0.02}
\equation_of_state{tait}

\boundary{walls}
\region{line}{x=-0.05, y=0.0..1.5}
\region{line}{x=1.5, y=0.0..1.5}
\region{line}{x=-0.05..1.5, y=-0.05}
```

**Key features demonstrated:**
- SPH particle method
- Tait equation of state
- Boundary conditions
- GPU acceleration via CUDA export

---

## ⚛️ Quantum Mechanics

### Quantum Tunneling
Calculate transmission probability through a barrier.

```python
from mechanics_dsl.domains.quantum import QuantumBarrier

barrier = QuantumBarrier(
    barrier_type='rectangular',
    V0=5.0,      # Barrier height (eV)
    L=1.0,       # Barrier width (nm)
    m=0.511      # Electron mass (MeV/c²)
)

T = barrier.transmission_coefficient(E=3.0)
print(f"Transmission probability: {T:.4f}")
```

**Validation:**
- Matches analytical formula to 6 significant figures
- WKB approximation for arbitrary barrier shapes

---

### Hydrogen Atom
Energy levels and spectral series.

```python
from mechanics_dsl.domains.quantum import HydrogenAtom

H = HydrogenAtom()

for n in range(1, 6):
    E = H.energy_level(n)
    print(f"n = {n}: E = {E:.4f} eV")

# Balmer series (visible spectrum)
balmer = H.spectral_series('balmer', n_max=7)
```

---

## 🌌 General Relativity

### Schwarzschild Geodesics
Orbits around non-rotating black holes.

```
\system{schwarzschild_orbit}
\parameter{M}{1.0}{solar_mass}
\defvar{r}{m}
\defvar{phi}{rad}
\metric{schwarzschild}
\geodesic{timelike}
\initial{r=10*r_s, phi=0, E=0.95, L=4.0}
```

**Key features demonstrated:**
- Precessing orbits (perihelion advance)
- ISCO at r = 6GM/c²
- Light bending for null geodesics

---

### Gravitational Lensing
Light deflection near massive objects.

```python
from mechanics_dsl.domains.general_relativity import GravitationalLensing

lensing = GravitationalLensing(M=1.0)  # Solar mass
deflection = lensing.deflection_angle(r_min=100)  # r in units of r_s
print(f"Deflection: {np.degrees(deflection)*3600:.4f} arcseconds")
```

---

## ⚡ Electromagnetism

### Cyclotron Motion
Charged particles in magnetic fields.

```
\system{cyclotron}
\parameter{B}{1.0}{T}
\parameter{q}{1.602e-19}{C}
\parameter{m}{9.109e-31}{kg}
\defvar{x}{m}
\defvar{y}{m}
\defvar{z}{m}
\magnetic_field{uniform}{B, direction=z}
\initial{vx=1e6, vy=0, vz=1e5}
```

**Results:**
- Cyclotron radius: rₗ = mv⊥/(qB)
- Cyclotron frequency: ωc = qB/m
- Helical trajectory

---

## 🚀 Special Relativity

### Relativistic Kinematics
Motion at 90% the speed of light.

```
\system{relativistic_particle}
\parameter{v}{0.9*c}
\parameter{m0}{electron_mass}
\relativity{special}
```

**Computed quantities:**
- Lorentz factor: γ = 2.294
- Time dilation: Δt' = γΔt
- Relativistic momentum: p = γm₀v

---

## 🔥 Thermodynamics

### Carnot Engine
The most efficient possible heat engine.

```
\system{carnot_engine}
\parameter{T_hot}{600}{K}
\parameter{T_cold}{300}{K}
\parameter{n}{1.0}{mol}
\working_fluid{ideal_gas}
\cycle{carnot}
```

**Results:**
- Carnot efficiency: η = 1 - T_cold/T_hot = 50%
- Entropy change per cycle: ΔS = 0

---

## 📊 Statistical Mechanics

### Quantum Harmonic Oscillator Ensemble

```
\system{qho_ensemble}
\parameter{hbar_omega}{0.05}{eV}
\parameter{T}{300}{K}
\ensemble{canonical}
\hamiltonian{(n + 0.5) * hbar_omega}
```

**Computed quantities:**
- Partition function Z
- Average energy ⟨E⟩
- Heat capacity C (Einstein model)

---

## 💻 Code Generation Examples

All systems can be exported to 11 target platforms:

| Target | Command | Use Case |
|--------|---------|----------|
| C++ | `compiler.compile_to_cpp("sim.cpp")` | High-performance desktop |
| CUDA | `compiler.compile_to_cuda("sim.cu")` | GPU acceleration |
| Rust | `compiler.compile_to_rust("sim.rs")` | Memory-safe systems |
| Julia | `compiler.compile_to_julia("sim.jl")` | Scientific computing |
| Fortran | `compiler.compile_to_fortran("sim.f90")` | Legacy HPC systems |
| MATLAB | `compiler.compile_to_matlab("sim.m")` | Engineering toolchains |
| JavaScript | `compiler.compile_to_javascript("sim.js")` | Web applications |
| WebAssembly | `compiler.compile_to_wasm("sim.wat")` | Browser deployment |
| Arduino | `compiler.compile_to_arduino("sim.ino")` | Embedded hardware |
| OpenMP | `compiler.compile_to_openmp("sim.cpp")` | Multi-core parallelism |
| Python | `compiler.compile_to_python("sim.py")` | Prototyping |

---

## 🐳 Enterprise Deployment

### Docker

```bash
# CPU version
docker pull ghcr.io/mechanicsdsl/mechanicsdsl:latest
docker run -it ghcr.io/mechanicsdsl/mechanicsdsl:latest

# GPU version
docker pull ghcr.io/mechanicsdsl/mechanicsdsl:gpu
docker run --gpus all -it ghcr.io/mechanicsdsl/mechanicsdsl:gpu
```

### Kubernetes

See [docs/deployment/kubernetes.md](docs/deployment/kubernetes.md) for production deployment guides.

---

## 🔌 Embedded Systems

### Raspberry Pi Pendulum

Real-time physics on ARM devices:

```bash
cd examples/raspberry_pi
./build_and_run.sh
```

Features:
- C++ export with ARM NEON optimization
- MPU6050 IMU integration
- Real-time 50 Hz simulation loop

---

## Contributing Examples

Have an interesting simulation? We welcome contributions!

1. Fork the repository
2. Add your example to `examples/` with:
   - DSL source file (`.mdsl`)
   - Python script demonstrating usage
   - README explaining the physics
3. Submit a pull request

---

**MechanicsDSL** — *Write physics like physics. Simulate anything.*
