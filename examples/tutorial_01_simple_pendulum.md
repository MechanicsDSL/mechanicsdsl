# Tutorial 1: The Simple Pendulum

## From Equation to Animation

This tutorial guides you through creating your first simulation in MechanicsDSL: simulating a simple pendulum—a mass \(m\) attached to a rod of length \(l\) swinging under gravity \(g\).

---

## Physics Setup

- **Kinetic Energy (\(T\))**: \(\frac{1}{2} m v^2\)
- **Potential Energy (\(V\))**: \(m g h\)

For a pendulum defined by angle \(\theta\):
- \(v = l \dot{\theta}\)
- \(h = -l \cos(\theta)\) (with \(y=0\) at the pivot)

The Lagrangian is given by \(L = T - V\).

---

## DSL Implementation

### Step 1: Define Variables

Create a file named `pendulum.dsl` and add the following variables using `\defvar`:

```
\system{simple_pendulum}

% The generalized coordinate
\defvar{theta}{Angle}{rad}

% Physical constants
\defvar{m}{Mass}{kg}
\defvar{l}{Length}{m}
\defvar{g}{Acceleration}{m/s^2}
```

### Step 2: Set Parameters

Assign numerical values to the constants for simulation:

```
\parameter{m}{1.0}{kg}
\parameter{l}{1.0}{m}
\parameter{g}{9.81}{m/s^2}
```

### Step 3: Write the Lagrangian

Define the Lagrangian equation:

```
% L = T - V
% T = 0.5 * m * (l * theta_dot)^2
% V = -m * g * l * cos(theta)
\lagrangian{ 
    \frac{1}{2} * m * l^2 * \dot{theta}^2 + m * g * l * \cos{theta} 
}
```

### Step 4: Execution

Set the initial state and run the solution and animation:

```
\initial{theta=0.5, theta_dot=0.0}
\solve{RK45}
\animate{pendulum}
```

---

## Running the Simulation

Run the simulation from your terminal:

```
python -m mechanics_dsl.core --file pendulum.dsl
```

MechanicsDSL will:

- Parse LaTeX-like syntax.
- Derive the Euler-Lagrange equation: \(\ddot{\theta} + \frac{g}{l}\sin(\theta) = 0\).
- Compile to a fast numerical function.
- Solve using Runge-Kutta (RK45).
- Launch an interactive Matplotlib animation.
