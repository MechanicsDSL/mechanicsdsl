Contributing to MechanicsDSL
============================

Thank you for your interest in contributing to MechanicsDSL! This document
provides guidelines for contributing to the project.

.. contents:: Contents
   :local:
   :depth: 2

Getting Started
---------------

Development Setup
~~~~~~~~~~~~~~~~~

1. Fork the repository on GitHub
2. Clone your fork:

   .. code-block:: bash

      git clone https://github.com/YOUR_USERNAME/mechanicsdsl.git
      cd mechanicsdsl

3. Create a virtual environment:

   .. code-block:: bash

      python -m venv venv
      source venv/bin/activate  # Linux/Mac
      venv\Scripts\activate     # Windows

4. Install in development mode:

   .. code-block:: bash

      pip install -e ".[dev]"

5. Verify tests pass:

   .. code-block:: bash

      pytest tests/


Code Style
----------

Python Style
~~~~~~~~~~~~

- Follow PEP 8 guidelines
- Use type hints for public APIs
- Maximum line length: 100 characters
- Use docstrings for all public functions/classes

.. code-block:: python

   def compute_energy(solution: dict, parameters: dict) -> np.ndarray:
       """
       Compute total energy from simulation solution.
       
       Args:
           solution: Simulation result dictionary containing 't' and 'y'
           parameters: Physical parameters dictionary
           
       Returns:
           Array of total energy values at each time point
           
       Raises:
           ValueError: If solution format is invalid
       """
       ...

Docstrings
~~~~~~~~~~

Use Google-style docstrings:

.. code-block:: python

   def function(arg1: str, arg2: int = 0) -> bool:
       """Short description.
       
       Longer description if needed, explaining the purpose
       and any important details.
       
       Args:
           arg1: Description of first argument.
           arg2: Description of second argument with default.
           
       Returns:
           Description of return value.
           
       Raises:
           ValueError: When arg1 is empty.
           TypeError: When arg2 is not an integer.
           
       Example:
           >>> function("test", 42)
           True
       """


Making Changes
--------------

Workflow
~~~~~~~~

1. Create a feature branch:

   .. code-block:: bash

      git checkout -b feature/your-feature-name

2. Make your changes
3. Add tests for new functionality
4. Ensure all tests pass
5. Update documentation if needed
6. Commit with descriptive messages:

   .. code-block:: bash

      git commit -m "Add: New constraint handling for rolling contacts"

7. Push and create a Pull Request

Commit Messages
~~~~~~~~~~~~~~~

Use descriptive commit messages with a prefix:

- ``Add:`` New features
- ``Fix:`` Bug fixes
- ``Update:`` Updates to existing features
- ``Docs:`` Documentation changes
- ``Test:`` Test additions/changes
- ``Refactor:`` Code refactoring
- ``Perf:`` Performance improvements


Testing
-------

Running Tests
~~~~~~~~~~~~~

.. code-block:: bash

   # Run all tests
   pytest tests/
   
   # Run with coverage
   pytest tests/ --cov=mechanics_dsl
   
   # Run specific test file
   pytest tests/test_parser.py
   
   # Run with verbose output
   pytest tests/ -v

Writing Tests
~~~~~~~~~~~~~

Place tests in the ``tests/`` directory:

.. code-block:: python

   # tests/test_my_feature.py
   import pytest
   from mechanics_dsl import PhysicsCompiler
   
   class TestMyFeature:
       def test_basic_functionality(self):
           compiler = PhysicsCompiler()
           result = compiler.compile(source)
           assert result['success']
       
       def test_error_handling(self):
           with pytest.raises(ValueError):
               compiler.compile("invalid source")


Documentation
-------------

Building Docs
~~~~~~~~~~~~~

.. code-block:: bash

   cd docs
   make html
   
   # Open in browser
   open _build/html/index.html  # Mac
   start _build/html/index.html  # Windows

Documentation Style
~~~~~~~~~~~~~~~~~~~

- Use reStructuredText format
- Include code examples
- Add cross-references to related sections
- Keep explanations clear and concise


Adding New Features
-------------------

Physics Domains
~~~~~~~~~~~~~~~

To add a new physics domain:

1. Create module in ``mechanics_dsl/domains/``
2. Inherit from ``PhysicsDomain``
3. Implement required abstract methods
4. Add tests
5. Add documentation

.. code-block:: python

   # mechanics_dsl/domains/quantum/schrodinger.py
   from ..base import PhysicsDomain
   
   class SchrodingerDomain(PhysicsDomain):
       """Quantum mechanics using Schrödinger equation."""
       
       def define_lagrangian(self):
           raise NotImplementedError("Quantum uses Hamiltonian")
       
       def define_hamiltonian(self):
           # H = -ℏ²/2m ∇² + V
           ...

Code Generation Backends
~~~~~~~~~~~~~~~~~~~~~~~~

To add a new code generation target:

1. Create module in ``mechanics_dsl/codegen/``
2. Inherit from ``CodeGenerator``
3. Implement required methods
4. Add template files if needed


Reporting Issues
----------------

Bug Reports
~~~~~~~~~~~

Include:

- MechanicsDSL version (``python -c "import mechanics_dsl; print(mechanics_dsl.__version__)"``
- Python version
- Operating system
- Minimal reproducible example
- Full error traceback

Feature Requests
~~~~~~~~~~~~~~~~

Include:

- Clear description of the feature
- Use case / motivation
- Any relevant references (papers, algorithms)


Code of Conduct
---------------

- Be respectful and inclusive
- Focus on constructive feedback
- Welcome newcomers
- Report inappropriate behavior to maintainers


License
-------

By contributing, you agree that your contributions will be licensed
under the MIT License.

Thank you for contributing to MechanicsDSL!
