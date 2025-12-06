WebAssembly Code Generation
===========================

Generate WebAssembly (WASM) for browser-based physics simulations.

Overview
--------

WebAssembly enables near-native performance in web browsers, making it ideal for:

- Interactive physics demonstrations
- Educational web applications
- Browser-based games with realistic physics
- Online simulation tools

.. note::

   WebAssembly support is experimental. Basic mechanics simulations work,
   but SPH fluid dynamics require additional browser features.

Prerequisites
-------------

To generate and use WebAssembly:

1. **Emscripten SDK**: Install from https://emscripten.org
2. **Modern browser**: Chrome, Firefox, Safari, or Edge (recent versions)

.. code-block:: bash

   # Install Emscripten
   git clone https://github.com/emscripten-core/emsdk.git
   cd emsdk
   ./emsdk install latest
   ./emsdk activate latest
   source ./emsdk_env.sh

Basic Usage
-----------

.. code-block:: python

   from mechanics_dsl import PhysicsCompiler
   
   compiler = PhysicsCompiler()
   compiler.compile_dsl(source)
   
   # Generate WebAssembly
   compiler.compile_to_wasm("pendulum.wasm")

This generates:

- ``pendulum.wasm``: WebAssembly binary
- ``pendulum.js``: JavaScript glue code
- ``pendulum.html``: Simple test page (optional)

Using in Web Pages
------------------

Basic HTML integration:

.. code-block:: html

   <!DOCTYPE html>
   <html>
   <head>
       <script src="pendulum.js"></script>
   </head>
   <body>
       <canvas id="canvas" width="800" height="600"></canvas>
       <script>
           Module.onRuntimeInitialized = function() {
               // Initialize simulation
               const sim = Module._create_simulation();
               
               // Animation loop
               function animate() {
                   Module._step_simulation(sim, 0.016); // 60 FPS
                   const state = Module._get_state(sim);
                   
                   // Draw to canvas
                   drawPendulum(state);
                   
                   requestAnimationFrame(animate);
               }
               animate();
           };
       </script>
   </body>
   </html>

JavaScript API
--------------

The generated JavaScript module provides:

.. code-block:: javascript

   // Create simulation instance
   const sim = Module._create_simulation();
   
   // Set initial conditions
   Module._set_state(sim, theta, theta_dot);
   
   // Step forward in time
   Module._step_simulation(sim, dt);
   
   // Get current state
   const state = Module._get_state(sim);
   // Returns: { theta: ..., theta_dot: ..., t: ... }
   
   // Clean up
   Module._destroy_simulation(sim);

Performance
-----------

WebAssembly typically achieves:

- 80-90% of native C++ performance
- Much faster than equivalent JavaScript
- Consistent across browsers

For physics simulations, expect:

.. list-table::
   :header-rows: 1

   * - System
     - JavaScript (ms/step)
     - WASM (ms/step)
     - Speedup
   * - Pendulum
     - 0.5
     - 0.05
     - 10x
   * - Double pendulum
     - 2.0
     - 0.15
     - 13x
   * - N-body (10)
     - 15.0
     - 1.2
     - 12x

Limitations
-----------

Current WebAssembly support has limitations:

1. **No SPH fluids**: Requires SIMD and threading (future)
2. **No file I/O**: Must use JavaScript for data export
3. **Memory limits**: Browser sandbox restricts memory
4. **Debugging**: Limited debugging tools compared to native

Future Plans
------------

Planned WebAssembly enhancements:

- SIMD support for SPH kernels
- Web Workers for parallel simulation
- WebGL integration for visualization
- Interactive parameter controls

Example: Interactive Pendulum
-----------------------------

Complete example with visualization:

.. code-block:: html

   <!DOCTYPE html>
   <html>
   <head>
       <title>MechanicsDSL Pendulum</title>
       <style>
           canvas { border: 1px solid #333; }
           .controls { margin: 10px 0; }
       </style>
       <script src="pendulum.js"></script>
   </head>
   <body>
       <h1>Interactive Pendulum</h1>
       <div class="controls">
           <label>Angle: <input type="range" id="angle" 
                   min="0" max="3.14" step="0.1" value="0.5"></label>
           <button onclick="reset()">Reset</button>
       </div>
       <canvas id="canvas" width="400" height="400"></canvas>
       
       <script>
           let sim;
           const canvas = document.getElementById('canvas');
           const ctx = canvas.getContext('2d');
           
           Module.onRuntimeInitialized = function() {
               sim = Module._create_simulation();
               animate();
           };
           
           function animate() {
               Module._step_simulation(sim, 0.016);
               draw();
               requestAnimationFrame(animate);
           }
           
           function draw() {
               const theta = Module._get_theta(sim);
               const L = 150;
               const cx = 200, cy = 100;
               
               ctx.clearRect(0, 0, 400, 400);
               
               // Draw rod
               const x = cx + L * Math.sin(theta);
               const y = cy + L * Math.cos(theta);
               ctx.beginPath();
               ctx.moveTo(cx, cy);
               ctx.lineTo(x, y);
               ctx.strokeStyle = '#333';
               ctx.lineWidth = 3;
               ctx.stroke();
               
               // Draw bob
               ctx.beginPath();
               ctx.arc(x, y, 20, 0, 2 * Math.PI);
               ctx.fillStyle = '#e63946';
               ctx.fill();
           }
           
           function reset() {
               const angle = document.getElementById('angle').value;
               Module._set_state(sim, parseFloat(angle), 0);
           }
       </script>
   </body>
   </html>
