# C++ Pendulum Demo

This demo generates a simple pendulum simulation in C++.

## Files

- `pendulum.cpp` - Generated C++ source code
- `compile.sh` - Build script

## Build

```bash
# Linux/macOS
g++ -O3 -o pendulum pendulum.cpp

# Windows (with MinGW)
g++ -O3 -o pendulum.exe pendulum.cpp
```

## Run

```bash
./pendulum
```

Output will be saved to `simple_pendulum_results.csv`.

## Visualize

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('simple_pendulum_results.csv')
plt.plot(df['t'], df['theta'])
plt.xlabel('Time (s)')
plt.ylabel('θ (rad)')
plt.show()
```
