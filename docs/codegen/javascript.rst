JavaScript Code Generation
==========================

Generate JavaScript simulation modules for browser and Node.js.

Features
--------

- **ES6 modules + CommonJS**: Works in browsers, Node.js, and bundlers
- **Three integrators**: Euler, RK4, and adaptive RK4-5
- **Canvas visualization**: Built-in pendulum drawing helper
- **CSV and JSON export**: ``exportCSV()`` and ``exportJSON()`` functions
- **TypeScript definitions**: Optional ``.d.ts`` generation
- **Zero dependencies**: Pure JavaScript, no npm packages required

Basic Usage
-----------

.. code-block:: python

   from mechanics_dsl.codegen.javascript import JavaScriptGenerator
   import sympy as sp

   theta, g, l = sp.symbols('theta g l')

   gen = JavaScriptGenerator(
       system_name="pendulum",
       coordinates=["theta"],
       parameters={"g": 9.81, "l": 1.0},
       initial_conditions={"theta": 0.5, "theta_dot": 0.0},
       equations={"theta_ddot": -g/l * sp.sin(theta)},
       integrator="rk4",
       generate_typescript=True
   )
   gen.generate("pendulum.js")

Parameters
~~~~~~~~~~

- ``integrator``: Integration method — ``euler``, ``rk4``, or ``adaptive`` (default: ``rk4``)
- ``generate_typescript``: Generate ``.d.ts`` type definitions (default: ``False``)

Usage in Node.js
----------------

.. code-block:: javascript

   const sim = require('./pendulum');
   const results = sim.simulate({ tEnd: 10, dt: 0.01, trackEnergy: true });
   console.log(`Points: ${results.t.length}`);

Usage in Browser
----------------

.. code-block:: html

   <script src="pendulum.js"></script>
   <script>
     const results = simulate({ tEnd: 10, dt: 0.01 });
   </script>

Canvas Animation
~~~~~~~~~~~~~~~~

The generated code includes a ``drawPendulum()`` helper:

.. code-block:: javascript

   const canvas = document.getElementById('canvas');
   const ctx = canvas.getContext('2d');
   const results = simulate({ tEnd: 10, dt: 0.01 });

   let frame = 0;
   function animate() {
       drawPendulum(ctx, results.y[frame]);
       frame = (frame + 1) % results.t.length;
       requestAnimationFrame(animate);
   }
   animate();

Generated Code Structure
------------------------

1. **Physical constants**: Module-level ``const`` declarations
2. **``equationsOfMotion(t, y)``**: State derivative function
3. **Integrators**: ``eulerStep``, ``rk4Step``, ``adaptiveStep``
4. **``simulate(config)``**: Main simulation loop with configurable solver
5. **``computeEnergy(y)``**: Mechanical energy computation
6. **``exportCSV(results)``**: CSV string generation
7. **``exportJSON(results)``**: JSON serialization
8. **``drawPendulum(ctx, state)``**: Canvas rendering helper
9. **Module exports**: CommonJS and browser globals

Integrator Comparison
---------------------

.. list-table::
   :header-rows: 1

   * - Integrator
     - Order
     - Use Case
   * - ``euler``
     - 1st
     - Fast, low accuracy, educational
   * - ``rk4``
     - 4th
     - General purpose (default)
   * - ``adaptive``
     - 4th-5th
     - Variable step, automatic accuracy

See Also
--------

- :doc:`wasm` - WebAssembly for near-native browser performance
- :doc:`python` - NumPy-accelerated Python
- :doc:`overview` - All code generation targets
