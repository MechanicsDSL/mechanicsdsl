# Tutorial 2: The Double Pendulum — Demonstrating Chaos Theory

The double pendulum is a classic example of a simple physical system exhibiting chaotic behavior. A slight change in initial conditions can lead to drastically different outcomes, illustrating the sensitive dependence characteristic of chaos theory. This tutorial shows how to simulate the double pendulum using MechanicsDSL, a tool that simplifies modeling complex systems.

## Prerequisites

This tutorial assumes you have a basic understanding of classical mechanics and some familiarity with using MechanicsDSL. You will learn how to define system variables, write the Lagrangian, run a simulation, and visualize chaotic behavior.

## 1. System Definition

We define two angles, `θ1` and `θ2`, representing the deflection of the top and bottom rods, respectively.

\system{double_pendulum}

% Coordinates
\defvar{theta1}{Angle}{rad}
\defvar{theta2}{Angle}{rad}

% Parameters
\defvar{m1}{Mass}{kg}
\defvar{m2}{Mass}{kg}
\defvar{l1}{Length}{m}
\defvar{l2}{Length}{m}
\defvar{g}{Acceleration}{m/s^2}

\parameter{m1}{1.0}{kg}
\parameter{m2}{1.0}{kg}
\parameter{l1}{1.0}{m}
\parameter{l2}{1.0}{m}
\parameter{g}{9.81}{m/s^2}

## 2. The Lagrangian

The system's kinetic energy \(T\) accounts for the motion of both masses, including the complexity of the second mass's velocity depending on the first. The potential energy \(V\) is due to gravity acting on both masses. The Lagrangian \(L = T - V\) governs the equations of motion.

\lagrangian{
  % Kinetic Energy
  \frac{1}{2} * (m1 + m2) * l1^2 * \dot{theta1}^2
  + \frac{1}{2} * m2 * l2^2 * \dot{theta2}^2
  + m2 * l1 * l2 * \dot{theta1} * \dot{theta2} * \cos{theta1 - theta2}

  % Potential Energy
  + (m1 + m2) * g * l1 * \cos{theta1}
  + m2 * g * l2 * \cos{theta2}
}

## 3. Simulation

To observe chaotic behavior, try starting the simulation with high-energy states (initial angles greater than 90 degrees). Define initial angles and angular velocities, then solve the system and animate the result:

\initial{theta1=2.0, theta1_dot=0.0, theta2=2.0, theta2_dot=0.0}
\solve{RK45}
\animate{double_pendulum}

## 4. Running the Simulation

Save the above code in a file named `double.dsl`. Run the simulation using the MechanicsDSL command-line tool:

mechanics-dsl --file double.dsl

---

Experiment with different masses, lengths, and initial conditions to explore the rich, chaotic dynamics of the double pendulum.
