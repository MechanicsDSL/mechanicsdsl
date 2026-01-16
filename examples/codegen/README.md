# Code Generation Examples

This directory contains examples for generating code in various target languages.

## Examples

### cpp_cmake_example/

Generates a complete C++ project with CMake:

```bash
cd cpp_cmake_example
python generate_project.py

# Build the generated project
cd output
mkdir build && cd build
cmake .. && make
./double_pendulum
```

### rust_cargo_example/

Generates Rust projects (standard and embedded):

```bash
cd rust_cargo_example
python generate_project.py

# Build standard version
cd output
cargo run --release

# Build embedded version
cd ../output_embedded
cargo build --release --target thumbv7em-none-eabihf
```

## Supported Targets

| Target | Generator | Cross-Compile | Embedded |
|--------|-----------|---------------|----------|
| C++ | CppGenerator | ✅ ARM | ❌ |
| Rust | RustGenerator | ✅ ARM | ✅ no_std |
| ARM | ARMGenerator | ✅ Pi/Jetson | ✅ Cortex-M |
| CUDA | CudaGenerator | ❌ | ❌ |
| WebAssembly | WasmGenerator | N/A | ❌ |
| Julia | JuliaGenerator | ❌ | ❌ |
| MATLAB | MatlabGenerator | ❌ | ❌ |
| Fortran | FortranGenerator | ❌ | ❌ |
| JavaScript | JavaScriptGenerator | N/A | ❌ |
| Arduino | ArduinoGenerator | ✅ AVR/ARM | ✅ |

## API Usage

All code generators follow a consistent API:

```python
from mechanics_dsl.codegen import CppGenerator, RustGenerator, ARMGenerator

# Create generator
gen = CppGenerator(
    system_name="my_system",
    coordinates=['x', 'y'],
    parameters={'m': 1.0, 'k': 10.0},
    initial_conditions={'x': 1.0, 'x_dot': 0, 'y': 0, 'y_dot': 0},
    equations={'x_ddot': expr1, 'y_ddot': expr2}
)

# Generate single file
gen.generate("output.cpp")

# Generate complete project (v2.0.0+)
gen.generate_project("output_dir/")
```

## New in v2.0.0

- **CMake support**: `CppGenerator.generate_cmake()`, `generate_project()`
- **Cargo support**: `RustGenerator.generate_cargo_toml()`, `generate_project()`
- **ARM codegen**: New `ARMGenerator` with NEON SIMD
- **CUDA batch**: `CudaGenerator.generate_batch_simulation()` for parameter sweeps
