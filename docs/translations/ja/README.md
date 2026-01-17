# MechanicsDSL - 力学ドメイン固有言語

> 🚧 **翻訳作業中** — 翻訳の改善にご協力ください！

MechanicsDSLは、計算物理学のためのドメイン固有言語およびコンパイラフレームワークです。

## 主な機能

- **シンボリック導出** — ラグランジアンからオイラー・ラグランジュ方程式を自動導出
- **マルチターゲットコード生成** — C++、CUDA、Rust、Juliaなど11言語への出力
- **GPU加速** — JAXバックエンドによる70倍高速化
- **9つの物理領域** — 古典力学、量子力学、相対性理論、流体力学など

## インストール

```bash
pip install mechanicsdsl-core
```

## クイックスタート

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

## ドキュメント

完全なドキュメントは [mechanicsdsl.readthedocs.io](https://mechanicsdsl.readthedocs.io) をご覧ください。

## ライセンス

MITライセンス — 商用・学術プロジェクトで自由にご利用いただけます。

---

*この翻訳はコミュニティによる貢献です。問題がありましたらIssueを作成してください。*
