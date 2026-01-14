# MechanicsDSL v1.6.0 Release Notes

**Release Date:** January 13, 2026

## 🚀 Major Feature Expansion

This release transforms MechanicsDSL from a DSL transpiler into a **comprehensive physics simulation ecosystem** with 13 major new features across 37 new files.

---

## ✨ Highlights

### Plugin Architecture
Extensible foundation for custom physics domains, code generators, and solvers.

```python
from mechanics_dsl.plugins import register_domain

@register_domain("acoustics")
class AcousticsPlugin(PhysicsDomainPlugin):
    ...
```

### JAX Backend
GPU acceleration, JIT compilation, and automatic differentiation.

```python
from mechanics_dsl.backends import JAXBackend

backend = JAXBackend(use_gpu=True)
grads = backend.gradient(loss_fn, params)  # Autodiff!
```

### Inverse Problems API
Parameter estimation, Sobol sensitivity analysis, and MCMC uncertainty quantification.

```python
from mechanics_dsl.inverse import ParameterEstimator

estimator = ParameterEstimator(compiler)
result = estimator.fit(observations, t_obs, ['m', 'k'])
```

### Jupyter Magic Commands
Native notebook experience with `%%mechanicsdsl` cell magic.

```python
%load_ext mechanics_dsl.jupyter

%%mechanicsdsl --animate
\system{pendulum}
\lagrangian{...}
```

### Real-time Server
FastAPI backend with WebSocket streaming for live visualization.

```bash
python -m mechanics_dsl.server
# API at http://localhost:8000/docs
```

### LSP Server
VS Code language server with diagnostics, autocomplete, and hover docs.

---

## 🔌 New Integrations

| Integration | Purpose | Install |
|-------------|---------|---------|
| **OpenMDAO** | Multidisciplinary optimization | `pip install .[openmao]` |
| **ROS2** | Robotics simulation | ROS2 environment |
| **Unity** | Game engine (C# MonoBehaviour) | Built-in |
| **Unreal** | Game engine (C++ ActorComponent) | Built-in |
| **Modelica** | Engineering standards | Built-in |

---

## 📊 Benchmark Suite

New performance testing infrastructure:

```bash
python -m benchmarks.runner --output results.json
```

Includes core simulation, code generation, and memory benchmarks.

---

## 📚 Tutorials

Three new Jupyter notebooks:
1. **Getting Started** - DSL basics, simple pendulum
2. **Double Pendulum** - Chaos, sensitivity to initial conditions
3. **Parameter Estimation** - Inverse problems, Sobol analysis

---

## 📦 Installation

```bash
# Core
pip install mechanicsdsl-core

# With optional features
pip install mechanicsdsl-core[jax]      # GPU acceleration
pip install mechanicsdsl-core[server]   # FastAPI backend
pip install mechanicsdsl-core[jupyter]  # Notebook magic
pip install mechanicsdsl-core[lsp]      # VS Code support
pip install mechanicsdsl-core[all]      # Everything
```

---

## 📈 Stats

- **37 new files** added
- **~7,700 lines** of new code
- **1,052 tests** passing
- **12+ code generators** (existing + new Unity/Unreal/Modelica)

---

## 🔮 Future Roadmap

- Acoustics/optics physics domains
- Neural-physics hybrid models
- Real-time 3D visualization
- Cloud deployment templates
- More tutorial content

---

## Contributors

Thank you to all contributors and users!

**Full Changelog:** https://github.com/MechanicsDSL/mechanicsdsl/compare/v1.5.1...v1.6.0
