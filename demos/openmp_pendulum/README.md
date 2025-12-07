# OpenMP Pendulum Demo

Multi-core parallel pendulum simulation using OpenMP.

## Files

- `generate.py` - Generate the OpenMP C++ code
- `pendulum_openmp.cpp` - Generated source (after running generate.py)

## Generate

```bash
python generate.py
```

## Build

```bash
# Linux/macOS
g++ -fopenmp -O3 -o pendulum_openmp pendulum_openmp.cpp

# Windows (with MinGW)
g++ -fopenmp -O3 -o pendulum_openmp.exe pendulum_openmp.cpp
```

## Run

```bash
./pendulum_openmp
```

This simulates 100 parallel trajectories with slightly different initial conditions.

## Features

- `#pragma omp parallel for` for batch simulations
- Automatic thread detection
- Performance timing output
