Rust Code Generation
====================

Generate memory-safe Rust simulation code with optional Cargo project scaffolding.

Features
--------

- **Zero-dependency RK4**: Built-in integrator with no external crates
- **Cargo project generation**: ``generate_project()`` creates full project structure
- **Cross-compilation**: ARM, embedded, and ``no_std`` targets
- **``no_std`` support**: For ARM Cortex-M microcontrollers
- **Type safety**: ``f64`` throughout with const generics for state dimension
- **CSV output**: Results file for analysis

Basic Usage
-----------

Generate a standalone Rust file:

.. code-block:: python

   from mechanics_dsl.codegen.rust import RustGenerator
   import sympy as sp

   theta, g, l = sp.symbols('theta g l')

   gen = RustGenerator(
       system_name="pendulum",
       coordinates=["theta"],
       parameters={"g": 9.81, "l": 1.0},
       initial_conditions={"theta": 0.5, "theta_dot": 0.0},
       equations={"theta_ddot": -g/l * sp.sin(theta)}
   )
   gen.generate("pendulum.rs")

Cargo Project Generation
-------------------------

Generate a complete Cargo project with ``Cargo.toml`` and ``README.md``:

.. code-block:: python

   gen.generate_project("./pendulum_rs")

This creates:

.. code-block:: text

   pendulum_rs/
   ├── Cargo.toml
   ├── README.md
   └── src/
       └── main.rs

Build and run:

.. code-block:: bash

   cd pendulum_rs
   cargo run --release

Embedded / ``no_std`` Mode
---------------------------

Generate code for ARM Cortex-M microcontrollers:

.. code-block:: python

   gen = RustGenerator(
       system_name="pendulum",
       coordinates=["theta"],
       parameters={"g": 9.81, "l": 1.0},
       initial_conditions={"theta": 0.5, "theta_dot": 0.0},
       equations={"theta_ddot": -g/l * sp.sin(theta)},
       embedded=True
   )
   gen.generate_project("./pendulum_embedded", embedded=True)

Build for ARM:

.. code-block:: bash

   rustup target add thumbv7em-none-eabihf
   cargo build --release --target thumbv7em-none-eabihf

Parameters
~~~~~~~~~~

- ``embedded``: Generate ``no_std`` compatible code (default: ``False``)
- ``use_nalgebra``: Use the nalgebra crate for matrix operations (default: ``False``)

Cross-Compilation
-----------------

**Raspberry Pi (aarch64):**

.. code-block:: bash

   rustup target add aarch64-unknown-linux-gnu
   cargo build --release --target aarch64-unknown-linux-gnu

**WebAssembly:**

.. code-block:: bash

   rustup target add wasm32-unknown-unknown
   cargo build --release --target wasm32-unknown-unknown

Generated Code Structure
------------------------

1. **Constants**: Parameters as ``const`` with ``f64`` types
2. **``equations_of_motion``**: State derivative function
3. **``rk4_step``**: Fixed-size array RK4 integrator
4. **``main``**: File I/O and time-stepping loop
5. **CSV output**: Comma-separated results file

See Also
--------

- :doc:`cpp` - C++ code generation
- :doc:`arduino` - Arduino/embedded C generation
- :doc:`overview` - All code generation targets
