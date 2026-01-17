# MechanicsDSL - 力学领域特定语言

> 🚧 **翻译进行中** — 欢迎贡献者帮助完善此翻译！

MechanicsDSL 是一个面向计算物理的领域特定语言和编译器框架。

## 核心功能

- **符号推导** — 从拉格朗日量自动推导欧拉-拉格朗日方程
- **多目标代码生成** — 导出到 C++、CUDA、Rust、Julia 等 11 种语言
- **GPU 加速** — 通过 JAX 后端实现 70 倍加速
- **九大物理领域** — 经典力学、量子力学、相对论、流体动力学等

## 安装

```bash
pip install mechanicsdsl-core
```

## 快速开始

```python
from mechanics_dsl import PhysicsCompiler

dsl_code = r"""
\system{simple_pendulum}
\defvar{theta}{角度}{rad}
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

## 文档

完整文档请访问 [mechanicsdsl.readthedocs.io](https://mechanicsdsl.readthedocs.io)

## 许可证

MIT 许可证 — 可自由用于商业和学术项目。

---

*此翻译由社区贡献。如有问题，请提交 Issue。*
