Embedded (Arduino) Target
=========================

For robotics and hardware-in-the-loop testing, the compiler can target microcontrollers.

Constraints
-----------
To run on limited hardware (like an Arduino Uno), this target:
1.  Replaces ``double`` with ``float``.
2.  Replaces ``std::vector`` with C-style static arrays.
3.  Uses **Euler Integration** (1st order) to minimize FLOPs.

**Warning**: Complex chaotic systems (like double pendulums) may drift significantly on this target due to precision loss.
