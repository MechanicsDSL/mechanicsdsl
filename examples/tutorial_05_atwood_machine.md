# Tutorial 5: The Atwood Machine — Constraint Handling

The Atwood machine consists of two masses hanging from a pulley. While traditionally solved using coordinates \(y_1\) and \(y_2\) with the relation \(y_2 = L - y_1\), MechanicsDSL allows explicit constraint definitions for more flexibility.

## 1. System Definition

Define two vertical coordinates \(y_1\) and \(y_2\) representing the positions of the two masses.

\system{atwood_machine}

\defvar{y1}{Position}{m}
\defvar{y2}{Position}{m}
\defvar{m1}{Mass}{kg}
\defvar{m2}{Mass}{kg}
\defvar{g}{Acceleration}{m/s^2}
\defvar{L}{Length}{m} % Total string length

\parameter{m1}{2.0}{kg} % Heavier mass
\parameter{m2}{1.0}{kg} % Lighter mass
\parameter{g}{9.81}{m/s^2}
\parameter{L}{2.0}{m}

## 2. The Lagrangian

The kinetic and potential energy of each mass govern the system's dynamics:

\lagrangian{
    \frac{1}{2} * m1 * \dot{y1}^2 + \frac{1}{2} * m2 * \dot{y2}^2 
    - m1 * g * y1 - m2 * g * y2
}

## 3. The Constraint

Explicitly define the string length constraint ensuring the masses' positions sum to the total length \(L\):

\constraint{ y1 + y2 - L }

MechanicsDSL uses Lagrange multipliers internally to handle this constraint in the equations of motion.

## 4. Simulation

Set initial positions and velocities and solve the system using a robust solver suited for constraints (LSODA):

\initial{y1=1.0, y1_dot=0.0, y2=1.0, y2_dot=0.0}
\solve{LSODA}
\animate{atwood_machine}

## 5. Running the Simulation

Save your code in `atwood.dsl` and run it with:

mechanics-dsl --file atwood.dsl


This setup models a classic physics device using advanced constraint handling for accurate dynamics simulation.
