I/O API Reference
=================

The ``mechanics_dsl.io`` package provides file input/output, serialization,
and data export utilities.

.. contents:: Contents
   :local:
   :depth: 2

Overview
--------

The I/O module supports:

- **JSON serialization**: Human-readable, portable format
- **Pickle serialization**: Preserves all Python types including SymPy
- **CSV export**: Standard tabular data format
- **Parameter/configuration export**: Save and restore system setups


SystemSerializer
----------------

.. py:class:: mechanics_dsl.io.SystemSerializer

   Serializer for physics system configurations.

   **Example:**

   .. code-block:: python

      from mechanics_dsl.io import SystemSerializer
      
      # Save to JSON
      data = {
          'system_name': 'pendulum',
          'parameters': {'m': 1.0, 'l': 1.0, 'g': 9.81},
          'initial_conditions': {'theta': 0.5, 'theta_dot': 0.0}
      }
      
      SystemSerializer.save_json(data, "system.json")
      
      # Load from JSON
      loaded = SystemSerializer.load_json("system.json")

   **Class Methods:**

   .. py:staticmethod:: save_json(data: dict, filename: str) -> bool

      Save data to JSON file.

   .. py:staticmethod:: load_json(filename: str) -> Optional[dict]

      Load data from JSON file.

   .. py:staticmethod:: save_pickle(data: dict, filename: str) -> bool

      Save data to pickle file (preserves all Python types).

   .. py:staticmethod:: load_pickle(filename: str) -> Optional[dict]

      Load data from pickle file.


CSVExporter
-----------

.. py:class:: mechanics_dsl.io.CSVExporter

   Export simulation data to CSV format.

   **Example:**

   .. code-block:: python

      from mechanics_dsl.io import CSVExporter
      
      # Export full solution
      CSVExporter.export_solution(solution, "results.csv")
      
      # Export custom data table
      import numpy as np
      data = {
          't': np.linspace(0, 10, 100),
          'theta': solution['y'][0],
          'energy': total_energy
      }
      CSVExporter.export_table(data, "analysis.csv")

   **Class Methods:**

   .. py:staticmethod:: export_solution(solution, filename, include_time=True) -> bool

      Export simulation solution to CSV.

      :param solution: Solution dictionary with 't' and 'y'
      :param filename: Output file path
      :param include_time: Include time column (default: True)
      :returns: True if successful

   .. py:staticmethod:: export_table(data: Dict[str, np.ndarray], filename: str) -> bool

      Export dictionary of arrays as columns.


JSONExporter
------------

.. py:class:: mechanics_dsl.io.JSONExporter

   Export simulation data to JSON format.

   **Class Methods:**

   .. py:staticmethod:: export_solution(solution, filename, compact=False) -> bool

      Export solution to JSON.

      :param compact: Use minimal formatting if True

   .. py:staticmethod:: export_parameters(parameters, filename) -> bool

      Export parameters dictionary to JSON.


Convenience Functions
---------------------

.. py:function:: mechanics_dsl.io.serialize_solution(solution, filename, format='json') -> bool

   Serialize a simulation solution to file.

   :param solution: Solution dictionary
   :param filename: Output file path
   :param format: 'json' or 'pickle'
   :returns: True if successful

.. py:function:: mechanics_dsl.io.deserialize_solution(filename, format=None) -> Optional[dict]

   Deserialize a simulation solution from file.

   :param filename: Input file path
   :param format: 'json' or 'pickle' (auto-detected if None)
   :returns: Solution dictionary or None


File Formats
------------

JSON Format
~~~~~~~~~~~

Human-readable format, suitable for:

- Configuration files
- Data interchange
- Version control

.. code-block:: json

   {
     "success": true,
     "system_name": "pendulum",
     "coordinates": ["theta"],
     "t": [0.0, 0.01, 0.02, ...],
     "y": [[0.5, 0.498, ...], [0.0, -0.049, ...]]
   }

Pickle Format
~~~~~~~~~~~~~

Binary format preserving Python objects:

- SymPy expressions
- NumPy arrays (efficient)
- Custom objects

.. warning::

   Pickle files can execute arbitrary code. Only load pickles from trusted sources.

CSV Format
~~~~~~~~~~

Standard comma-separated values:

.. code-block:: text

   t,theta,theta_dot
   0.000000,0.500000,0.000000
   0.010000,0.499877,-0.024525
   0.020000,0.499509,-0.049034
   ...
