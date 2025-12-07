# Expected output for simple pendulum simulation

After running any backend's pendulum simulation, you should observe:

## Initial Conditions
- θ(0) = 0.3 rad (~17°)
- θ̇(0) = 0.0 rad/s

## Expected Behavior
- Period ≈ 2.01 s (for small oscillations)
- Maximum angular velocity ≈ 0.94 rad/s
- Energy conservation (< 0.1% error with RK4)

## Sample Output (first 5 rows of CSV)
```
t,theta,theta_dot
0.000000,0.300000,0.000000
0.010000,0.299853,-0.029410
0.020000,0.299410,-0.058793
0.030000,0.298674,-0.088123
0.040000,0.297645,-0.117373
```

## Validation Criteria
1. **Periodicity**: θ should return close to 0.3 after ~2.01 seconds
2. **Symmetry**: Oscillation should be symmetric about θ=0
3. **Energy**: Total energy E = 0.5*l²*θ̇² - g*l*cos(θ) ≈ constant
4. **Phase**: θ and θ̇ should be 90° out of phase

## Verification Script
```python
import pandas as pd
import numpy as np

df = pd.read_csv('pendulum_results.csv')

# Check energy conservation
g, l = 9.81, 1.0
theta = df['theta'].values
theta_dot = df['theta_dot'].values
E = 0.5 * l**2 * theta_dot**2 - g * l * np.cos(theta)
E_error = (E.max() - E.min()) / abs(E.mean())

print(f"Energy variation: {E_error*100:.4f}%")
assert E_error < 0.01, "Energy not conserved!"
print("✓ Simulation verified!")
```
