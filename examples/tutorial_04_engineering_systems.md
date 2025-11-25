# Tutorial 4: Damped Driven Oscillator — Engineering Applications & Non-Conservative Forces

Real-world mechanical systems include friction and other non-conservative forces. This tutorial demonstrates how to simulate a damped oscillator with a damper (shock absorber) using the `\damping` command in MechanicsDSL.

## 1. System Definition

We model a mass \(m\) on a spring with spring constant \(k\), subject to a damping force characterized by damping coefficient \(c\).

\system{damped_oscillator}

\defvar{x}{Position}{m}
\defvar{m}{Mass}{kg}
\defvar{k}{Spring Constant}{N/m}
\defvar{c}{Damping Coefficient}{N*s/m}

\parameter{m}{1.0}{kg}
\parameter{k}{10.0}{N/m}
\parameter{c}{0.5}{N*s/m}

## 2. The Lagrangian & Damping

The Lagrangian describes conservative forces only (kinetic and potential energies):

\lagrangian{ \frac{1}{2} * m * \dot{x}^2 - \frac{1}{2} * k * x^2 }

The damping (non-conservative) force is modeled separately using the damping command:

\damping{ -c * \dot{x} }

## 3. Simulation

Set the initial displacement and velocity. The damping causes the amplitude to decrease over time, simulating energy loss due to friction.

\initial{x=1.0, x_dot=0.0}
\solve{RK45}
\animate{oscillator}

## 4. Running the Simulation

Save this script as `damped.dsl` and run:

mechanics-dsl --file damped.dsl


Experiment with different damping coefficients to see underdamped, critically damped, or overdamped behaviors. This setup is typical for engineering applications involving shock absorbers and vibration damping.
