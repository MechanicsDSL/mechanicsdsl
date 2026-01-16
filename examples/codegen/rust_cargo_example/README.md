# Rust Cargo Example

This example demonstrates generating a complete Rust simulation project with Cargo build support.

## Quick Start

```bash
# Generate the project
python generate_project.py

# Build and run (standard)
cd output
cargo run --release
```

## Generated Files

When you run `generate_project.py`, it creates two projects:

### Standard Project (`output/`)
```
output/
├── src/
│   └── main.rs         # Rust simulation code
├── Cargo.toml          # Cargo configuration
└── README.md           # Build instructions
```

### Embedded Project (`output_embedded/`)
```
output_embedded/
├── src/
│   └── main.rs         # no_std Rust code
├── Cargo.toml          # Embedded config (no stdlib)
└── README.md           # Embedded instructions
```

## Cross-Compilation for Raspberry Pi

```bash
# Add target
rustup target add aarch64-unknown-linux-gnu

# Build
cargo build --release --target aarch64-unknown-linux-gnu

# Copy to Pi
scp target/aarch64-unknown-linux-gnu/release/harmonic_oscillator pi@raspberrypi:~/
```

## Embedded Systems (Cortex-M)

```bash
# Add embedded target
rustup target add thumbv7em-none-eabihf

# Build
cd output_embedded
cargo build --release --target thumbv7em-none-eabihf
```

## Features

- **Standard Build**: Full Rust with std library
- **Embedded Build**: no_std for bare-metal
- **LTO**: Link-time optimization for smaller binaries
- **ARM Support**: Cross-compilation ready

## Performance

Rust provides:
- Zero-cost abstractions
- No garbage collection
- Predictable performance
- Excellent ARM support

## Customization

Edit `generate_project.py` to change:

- System definition (DSL code)  
- Output directory
- Enable/disable embedded mode
