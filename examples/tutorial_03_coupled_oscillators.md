# Tutorial 3: Coupled Oscillators — Multi-Body Dynamics and Normal Modes

This example demonstrates two masses connected by springs, sliding on a frictionless surface. It is a classic illustration of normal modes in physics, where energy transfers between coupled systems.

## 1. System Definition

We track the displacements \(x_1\) and \(x_2\) of two masses \(m\), connected by three springs with spring constant \(k\).

\system{coupled_oscillators}

\defvar{x1}{Position}{m}
\defvar{x2}{Position}{m}
\defvar{m}{Mass}{kg}
\defvar{k}{Spring Constant}{N/m}

\parameter{m}{1.0}{kg}
\parameter{k}{10.0}{N/m}

## 2. The Lagrangian

The kinetic energy is the sum of each mass's kinetic energy. The potential energy is the sum of energy stored in the three springs. The middle spring stretches by \((x_2 - x_1)\).

\lagrangian{
    % Kinetic Energy
    \frac{1}{2} * m * \dot{x1}^2 + \frac{1}{2} * m * \dot{x2}^2 
    
    % Potential Energy (Springs)
    - ( \frac{1}{2} * k * x1^2 + \frac{1}{2} * k * (x2 - x1)^2 + \frac{1}{2} * k * x2^2 )
}

## 3. Simulation

Start by displacing the first mass while keeping the second mass stationary to observe energy transfer between the two.

\initial{x1=1.0, x1_dot=0.0, x2=0.0, x2_dot=0.0}
\solve{RK45}
\animate{oscillator}

## 4. Running the Simulation

Save the code above as `coupled.dsl` and run with:

mechanics-dsl --file coupled.dsl


Experiment with different mass and spring constant values, or initial displacements to explore the coupled normal mode behavior.
