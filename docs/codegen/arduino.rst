Arduino Code Generation
=======================

Generate Arduino sketches for embedded physics simulations on microcontrollers.

Features
--------

- **RAM-optimized**: Uses ``float`` instead of ``double`` for constrained memory
- **Serial Plotter output**: Real-time visualization in Arduino IDE
- **Servo output**: Map simulation state to physical servo motors
- **Real-time timing**: ``micros()``-based loop for consistent timesteps
- **RK4 integrator**: Full 4th-order integration on microcontroller

Basic Usage
-----------

.. code-block:: python

   from mechanics_dsl.codegen.arduino import ArduinoGenerator
   import sympy as sp

   theta, g, l = sp.symbols('theta g l')

   gen = ArduinoGenerator(
       system_name="pendulum",
       coordinates=['theta'],
       parameters={'g': 9.81, 'l': 1.0},
       initial_conditions={'theta': 0.5, 'theta_dot': 0.0},
       equations={'theta_ddot': -g/l * sp.sin(theta)},
       use_serial_plotter=True,
       servo_pin=9
   )
   gen.generate("pendulum.ino")

Parameters
~~~~~~~~~~

- ``use_serial_plotter``: Enable Serial Plotter formatted output (default: ``True``)
- ``servo_pin``: Pin number for servo output, or ``None`` to disable (default: ``None``)

Generated Code Structure
------------------------

The generated ``.ino`` sketch contains:

1. **Physical parameters**: ``const float`` values with ``f`` suffix
2. **State array**: Fixed-size ``float`` array
3. **``computeDerivatives()``**: Equations of motion function
4. **``rk4Step()``**: In-place RK4 integrator using ``float``
5. **``setup()``**: Serial initialization and optional servo attach
6. **``loop()``**: Timing-controlled integration and output
7. **``resetSimulation()``**: Reset state to initial conditions

Serial Plotter
--------------

Open **Tools > Serial Plotter** in Arduino IDE after uploading.
The output format is CSV-compatible:

.. code-block:: text

   Time,theta,theta_dot
   0.0100,0.4995,−0.0981
   0.0200,0.4980,−0.1961

Servo Output
------------

When ``servo_pin`` is set, the first coordinate is mapped to a 0–180 degree
servo range. This creates a physical pendulum display:

.. code-block:: python

   gen = ArduinoGenerator(
       ...,
       servo_pin=9  # PWM-capable pin
   )

Compatible Boards
-----------------

.. list-table::
   :header-rows: 1

   * - Board
     - RAM
     - Notes
   * - Arduino Uno (ATmega328P)
     - 2 KB
     - Up to ~3 coordinates
   * - Arduino Mega (ATmega2560)
     - 8 KB
     - Larger systems
   * - ESP32
     - 520 KB
     - Wi-Fi + Bluetooth capable
   * - Teensy 4.0
     - 1 MB
     - 600 MHz, hardware FPU

Memory Considerations
---------------------

Each coordinate uses ``2 * sizeof(float) * 5 = 40`` bytes for RK4 temporaries.
On an Arduino Uno with 2 KB RAM, keep systems to 3 coordinates or fewer.

See Also
--------

- :doc:`cpp` - Standard C++ generation
- :doc:`rust` - Rust with ``no_std`` embedded support
- :doc:`overview` - All code generation targets
