# Rust Pendulum Demo

Simple pendulum simulation in Rust using nalgebra.

## Files

- `pendulum.rs` - Generated Rust source

## Build

```bash
rustc -O -o pendulum pendulum.rs
```

Or create a Cargo project:

```bash
cargo init .
# Add to Cargo.toml: nalgebra = "0.32"
cargo build --release
```

## Run

```bash
./pendulum
```
