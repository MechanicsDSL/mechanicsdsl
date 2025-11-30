Installation
============

MechanicsDSL supports Python 3.8 and newer. It relies on ``numpy``, ``scipy``, ``sympy``, and ``matplotlib``.

Installing via pip
------------------

The easiest way to install MechanicsDSL is via pip:

.. code-block:: bash

    pip install mechanics-dsl

Installing from Source
----------------------

For developers or those wanting the latest features, you can install directly from the repository:

.. code-block:: bash

    git clone https://github.com/MechanicsDSL/mechanicsdsl.git
    cd mechanicsdsl
    pip install -e .

Verifying Installation
----------------------

You can verify that the installation was successful by running the built-in tests:

.. code-block:: bash

    pip install pytest
    pytest tests/

Dependencies
------------

* **NumPy**: Numerical array operations.
* **SciPy**: Advanced ODE solvers (LSODA, RK45).
* **SymPy**: Symbolic mathematics engine for deriving equations.
* **Matplotlib**: Visualization and animation.

Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~

* **ffmpeg**: Required if you want to save animations as MP4 files.
