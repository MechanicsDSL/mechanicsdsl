WebAssembly (WASM) Target
=========================

MechanicsDSL can compile physics engines directly to the web using **Emscripten**.

Usage
-----
.. code-block:: python

    compiler.compile_to_cpp("sim.cpp", target="wasm")

Memory Model
------------
The generated WASM module exposes the C++ heap to JavaScript via ``Module.HEAPF64``. This allows for **Zero-Copy** visualization: the physics engine runs in WASM, and the JavaScript WebGL renderer reads the positions directly from memory without serialization overhead.
