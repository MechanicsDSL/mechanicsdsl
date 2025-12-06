Code Generation API Reference
=============================

The ``mechanics_dsl.codegen`` package provides code generation backends for
compiling MechanicsDSL simulations to various target platforms.

.. contents:: Contents
   :local:
   :depth: 2

Base Classes
------------

CodeGenerator
~~~~~~~~~~~~~

.. py:class:: mechanics_dsl.codegen.CodeGenerator

   Abstract base class for code generation backends.

   **Attributes:**

   - ``system_name``: Physics system name
   - ``coordinates``: Generalized coordinates
   - ``parameters``: Physical parameters
   - ``initial_conditions``: Initial state
   - ``equations``: Symbolic equations

   **Abstract Properties:**

   - ``target_name``: Target platform name (e.g., 'cpp', 'python')
   - ``file_extension``: Output file extension (e.g., '.cpp', '.py')

   **Abstract Methods:**

   .. py:method:: generate(output_file: str) -> str
      :abstractmethod:

      Generate code and write to file.

   .. py:method:: generate_equations() -> str
      :abstractmethod:

      Generate equations of motion code.

   **Common Methods:**

   .. py:method:: generate_parameters() -> str

      Generate parameter declarations.

   .. py:method:: generate_initial_conditions() -> str

      Generate initial condition setup.

   .. py:method:: validate() -> Tuple[bool, List[str]]

      Validate generator has all required data.


C++ Backend
-----------

CppGenerator
~~~~~~~~~~~~

.. py:class:: mechanics_dsl.codegen.CppGenerator

   Generates optimized C++ simulation code.

   **Example:**

   .. code-block:: python

      from mechanics_dsl.codegen import CppGenerator
      
      generator = CppGenerator(
          system_name="pendulum",
          coordinates=["theta"],
          parameters={"m": 1.0, "l": 1.0, "g": 9.81},
          initial_conditions={"theta": 0.5, "theta_dot": 0.0},
          equations={"theta_ddot": symbolic_expr}
      )
      
      generator.generate("simulation.cpp")

   **Constructor:**

   .. py:method:: __init__(system_name, coordinates, parameters, initial_conditions, equations, fluid_particles=None, boundary_particles=None)

      :param fluid_particles: List of SPH particles (for fluid simulations)
      :param boundary_particles: List of boundary particles

   **Methods:**

   .. py:method:: generate(output_file: str = "simulation.cpp") -> str

      Generate C++ file with RK4 integration.

   **Generated Code Features:**

   - RK4 integration algorithm
   - CSV output for results
   - SPH spatial hashing (for fluids)
   - OpenMP parallelization (optional)


Python Backend
--------------

PythonGenerator
~~~~~~~~~~~~~~~

.. py:class:: mechanics_dsl.codegen.python.PythonGenerator

   Generates standalone Python simulation scripts using NumPy/SciPy.

   **Example:**

   .. code-block:: python

      from mechanics_dsl.codegen.python import PythonGenerator
      
      generator = PythonGenerator(
          system_name="double_pendulum",
          coordinates=["theta1", "theta2"],
          parameters={"m1": 1.0, "m2": 1.0, "l1": 1.0, "l2": 1.0, "g": 9.81},
          initial_conditions={...},
          equations={...}
      )
      
      generator.generate("double_pendulum_sim.py")

   **Generated Script Features:**

   - NumPy array operations
   - SciPy solve_ivp integration
   - Matplotlib visualization
   - Runnable as standalone script


I/O API Reference
=================

The ``mechanics_dsl.io`` package provides file input/output, serialization,
and data export utilities.

SystemSerializer
----------------

.. py:class:: mechanics_dsl.io.SystemSerializer

   Serializer for physics system configurations.

   **Example:**

   .. code-block:: python

      from mechanics_dsl.io import SystemSerializer
      
      # Save to JSON
      SystemSerializer.save_json(data, "system.json")
      
      # Load from JSON
      data = SystemSerializer.load_json("system.json")
      
      # Pickle for full Python object preservation
      SystemSerializer.save_pickle(data, "system.pkl")
      data = SystemSerializer.load_pickle("system.pkl")

   **Class Methods:**

   .. py:staticmethod:: save_json(data: dict, filename: str) -> bool

      Save data to JSON file.

   .. py:staticmethod:: load_json(filename: str) -> Optional[dict]

      Load data from JSON file.

   .. py:staticmethod:: save_pickle(data: dict, filename: str) -> bool

      Save data to pickle file.

   .. py:staticmethod:: load_pickle(filename: str) -> Optional[dict]

      Load data from pickle file.


Convenience Functions
---------------------

.. py:function:: mechanics_dsl.io.serialize_solution(solution, filename, format='json')

   Serialize simulation solution to file.

   :param solution: Solution dictionary
   :param filename: Output path
   :param format: 'json' or 'pickle'

.. py:function:: mechanics_dsl.io.deserialize_solution(filename, format=None)

   Load simulation solution from file.

   :param filename: Input path
   :param format: Auto-detected if None


CSVExporter
-----------

.. py:class:: mechanics_dsl.io.CSVExporter

   Export simulation data to CSV format.

   **Example:**

   .. code-block:: python

      from mechanics_dsl.io import CSVExporter
      
      # Export solution
      CSVExporter.export_solution(solution, "results.csv")
      
      # Export custom table
      CSVExporter.export_table({
          't': time_array,
          'energy': energy_array,
          'error': error_array
      }, "analysis.csv")

   **Class Methods:**

   .. py:staticmethod:: export_solution(solution, filename, include_time=True)

      Export simulation solution to CSV.

   .. py:staticmethod:: export_table(data: dict, filename: str)

      Export dictionary of arrays as columns.


JSONExporter
------------

.. py:class:: mechanics_dsl.io.JSONExporter

   Export simulation data to JSON format.

   **Class Methods:**

   .. py:staticmethod:: export_solution(solution, filename, compact=False)

      Export solution to JSON.

   .. py:staticmethod:: export_parameters(parameters, filename)

      Export parameters to JSON.
