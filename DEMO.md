# MechanicsDSL — Quick Demo Examples

One-liner examples to show off MechanicsDSL capabilities. Perfect for presentations and demos.

---

## 🎯 30-Second Demo

```bash
# Install
pip install mechanicsdsl-core

# Run a pendulum simulation (CLI)
mechanicsdsl run examples/dsl/pendulum.mdsl --t-span 0,10 --animate
```

---

## ⚡ Python One-Liners

### Simple Pendulum
```python
from mechanics_dsl import PhysicsCompiler; c=PhysicsCompiler(); c.compile_dsl(r'\system{p}\defvar{θ}{rad}\parameter{m}{1}{kg}\parameter{l}{1}{m}\parameter{g}{9.81}{m/s^2}\lagrangian{0.5*m*l^2*\dot{θ}^2-m*g*l*(1-\cos{θ})}\initial{θ=2.5}'); c.animate(c.simulate())
```

### Double Pendulum Chaos
```python
from mechanics_dsl import PhysicsCompiler
compiler = PhysicsCompiler()
compiler.compile_dsl(open('examples/dsl/double_pendulum.mdsl').read())
compiler.animate(compiler.simulate(t_span=(0, 30)))
```

### Export to 11 Languages
```python
compiler.compile_to_cpp('sim.cpp')    # C++
compiler.compile_to_cuda('sim.cu')    # CUDA
compiler.compile_to_rust('sim.rs')    # Rust
compiler.compile_to_julia('sim.jl')   # Julia
compiler.compile_to_wasm('sim.wat')   # WebAssembly
```

---

## 📊 Benchmark One-Liner

```bash
python -c "
from mechanics_dsl import PhysicsCompiler
import time
c = PhysicsCompiler()
c.compile_dsl(r'''
\system{bench}
\defvar{x}{m}\defvar{y}{m}
\parameter{m}{1}{kg}\parameter{k}{100}{N/m}
\lagrangian{0.5*m*(\dot{x}^2+\dot{y}^2)-0.5*k*(x^2+y^2)}
\initial{x=1,y=0}
''')
t0=time.time()
c.simulate(t_span=(0,100), num_points=100000)
print(f'100k points in {time.time()-t0:.2f}s')
"
```

---

## 🔢 Physics Domains Demo

### Quantum Tunneling
```python
from mechanics_dsl.domains.quantum import QuantumBarrier
b = QuantumBarrier(barrier_type='rectangular', V0=5.0, L=1.0, m=0.511)
print(f"Tunneling probability: {b.transmission_coefficient(3.0):.4f}")
```

### Black Hole Orbit
```python
from mechanics_dsl.domains.general_relativity import SchwarzschildMetric
bh = SchwarzschildMetric(M=1.0)
print(f"ISCO radius: {bh.isco():.2f} km")
```

### Statistical Mechanics
```python
from mechanics_dsl.domains.statistical import QuantumOscillator
qho = QuantumOscillator(hbar_omega=0.05)
print(f"Heat capacity at 300K: {qho.heat_capacity(300):.4f} k_B")
```

---

## 🎬 GIF-Worthy Demos

### Chaos Sensitivity
```python
# Two pendulums with 0.000001 rad difference
# Watch them diverge after ~10 seconds
compiler1.set_initial({'theta1': 2.5})
compiler2.set_initial({'theta1': 2.500001})
# Run both and compare...
```

### Figure-8 Three-Body
```python
# The beautiful Chenciner-Montgomery orbit
# All three bodies chase each other forever
compiler.compile_dsl(open('examples/dsl/figure8.mdsl').read())
compiler.animate(compiler.simulate(t_span=(0, 20)))
```

### Dam Break Fluid
```python
# 1000+ particles splashing
compiler.compile_dsl(open('examples/dsl/dam_break.mdsl').read())
compiler.animate_fluid(compiler.simulate_fluid(t_span=(0, 2)))
```

---

## 📈 Adoption Stats

```
19 countries | 11 code generators | 70× GPU speedup | 9 physics domains
```

---

**MechanicsDSL** — *Write physics like physics. Simulate anything.*
