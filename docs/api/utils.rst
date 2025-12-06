Utils API Reference
===================

The ``mechanics_dsl.utils`` package provides utility functions and classes
for logging, configuration, caching, profiling, and validation.

.. contents:: Contents
   :local:
   :depth: 2

Logging
-------

.. py:function:: mechanics_dsl.utils.setup_logging(level='INFO', log_file=None)

   Configure logging for MechanicsDSL.

   :param level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
   :param log_file: Optional file path for log output

   **Example:**

   .. code-block:: python

      from mechanics_dsl.utils import setup_logging
      
      setup_logging(level='DEBUG')  # Verbose output
      setup_logging(level='WARNING', log_file='simulation.log')

.. py:data:: mechanics_dsl.utils.logger

   Module-level logger instance.


Configuration
-------------

.. py:class:: mechanics_dsl.utils.Config

   Global configuration singleton for MechanicsDSL settings.

   **Attributes:**

   .. list-table::
      :header-rows: 1
      :widths: 30 20 50

      * - Attribute
        - Default
        - Description
      * - ``default_rtol``
        - 1e-8
        - Relative tolerance for ODE solver
      * - ``default_atol``
        - 1e-10
        - Absolute tolerance for ODE solver
      * - ``trail_length``
        - 200
        - Trail points in animations
      * - ``animation_fps``
        - 30
        - Frames per second
      * - ``simplification_timeout``
        - 10.0
        - SymPy simplification timeout (seconds)
      * - ``max_parser_errors``
        - 10
        - Maximum parser errors before abort
      * - ``energy_tolerance``
        - 1e-6
        - Energy conservation tolerance

   **Example:**

   .. code-block:: python

      from mechanics_dsl.utils import config
      
      # Read configuration
      print(f"Default tolerance: {config.default_rtol}")
      
      # Modify configuration
      config.trail_length = 500
      config.animation_fps = 60

.. py:data:: mechanics_dsl.utils.config

   Global Config instance.


Caching
-------

.. py:class:: mechanics_dsl.utils.LRUCache

   Least Recently Used cache for expensive computations.

   **Example:**

   .. code-block:: python

      from mechanics_dsl.utils import LRUCache
      
      cache = LRUCache(maxsize=100)
      
      # Cache expensive computation
      key = str(expression)
      if key in cache:
          result = cache[key]
      else:
          result = expensive_computation(expression)
          cache[key] = result
      
      # Clear cache
      cache.clear()

   **Methods:**

   .. py:method:: __init__(maxsize: int = 128)

      Create cache with maximum size.

   .. py:method:: __getitem__(key) -> Any

      Get cached value.

   .. py:method:: __setitem__(key, value) -> None

      Store value in cache.

   .. py:method:: __contains__(key) -> bool

      Check if key is cached.

   .. py:method:: clear() -> None

      Clear all cached values.

   .. py:method:: get_stats() -> dict

      Get cache hit/miss statistics.


Profiling
---------

.. py:class:: mechanics_dsl.utils.PerformanceMonitor

   Performance monitoring and timing utilities.

   **Example:**

   .. code-block:: python

      from mechanics_dsl.utils import PerformanceMonitor
      
      monitor = PerformanceMonitor()
      
      monitor.start_timer("compilation")
      # ... do compilation ...
      monitor.stop_timer("compilation")
      
      monitor.start_timer("simulation")
      # ... do simulation ...
      monitor.stop_timer("simulation")
      
      # Get report
      report = monitor.get_report()
      print(f"Compilation: {report['compilation']:.3f}s")
      print(f"Simulation: {report['simulation']:.3f}s")

   **Methods:**

   .. py:method:: start_timer(name: str) -> None

      Start a named timer.

   .. py:method:: stop_timer(name: str) -> float

      Stop timer and return elapsed time.

   .. py:method:: get_report() -> dict

      Get all timing results.

   .. py:method:: reset() -> None

      Clear all timers.


.. py:function:: mechanics_dsl.utils.profile_function(func)

   Decorator to profile function execution time.

   **Example:**

   .. code-block:: python

      from mechanics_dsl.utils import profile_function
      
      @profile_function
      def expensive_calculation(x):
          # ... complex math ...
          return result

.. py:class:: mechanics_dsl.utils.timeout

   Context manager for operation timeouts.

   **Example:**

   .. code-block:: python

      from mechanics_dsl.utils import timeout, TimeoutError
      
      try:
          with timeout(5.0):  # 5 second timeout
              result = long_running_computation()
      except TimeoutError:
          print("Computation timed out!")


Validation
----------

.. py:function:: mechanics_dsl.utils.validate_finite(value, name='value')

   Validate that a value is finite (not inf or nan).

   :param value: Value to check
   :param name: Name for error messages
   :raises: ValueError if not finite

.. py:function:: mechanics_dsl.utils.validate_positive(value, name='value')

   Validate that a value is positive.

.. py:function:: mechanics_dsl.utils.validate_non_negative(value, name='value')

   Validate that a value is non-negative.

.. py:function:: mechanics_dsl.utils.validate_array_safe(arr, name='array', check_finite=True)

   Validate a numpy array.

   :param arr: Array to validate
   :param name: Name for error messages
   :param check_finite: Check for inf/nan values
   :returns: True if valid

.. py:function:: mechanics_dsl.utils.safe_float_conversion(value, default=0.0)

   Safely convert value to float.

   :param value: Value to convert
   :param default: Default if conversion fails
   :returns: Float value

.. py:function:: mechanics_dsl.utils.validate_time_span(t_span)

   Validate a time span tuple (t_start, t_end).

   :raises: ValueError if invalid

.. py:function:: mechanics_dsl.utils.validate_solution_dict(solution)

   Validate a solution dictionary has required keys.

.. py:function:: mechanics_dsl.utils.validate_file_path(path, must_exist=False)

   Validate a file path.

   :param path: Path to validate
   :param must_exist: Require file to exist


Constants
---------

The utils module exports commonly used constants:

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Constant
     - Value
     - Description
   * - ``DEFAULT_TRAIL_LENGTH``
     - 200
     - Animation trail points
   * - ``DEFAULT_FPS``
     - 30
     - Animation frame rate
   * - ``ENERGY_TOLERANCE``
     - 1e-6
     - Energy conservation check
   * - ``DEFAULT_RTOL``
     - 1e-8
     - ODE solver relative tolerance
   * - ``DEFAULT_ATOL``
     - 1e-10
     - ODE solver absolute tolerance
   * - ``SIMPLIFICATION_TIMEOUT``
     - 10.0
     - SymPy simplification timeout
   * - ``MAX_PARSER_ERRORS``
     - 10
     - Parser error limit
   * - ``ANIMATION_INTERVAL_MS``
     - 33
     - Animation frame interval (~30fps)
   * - ``TRAIL_ALPHA``
     - 0.6
     - Trail transparency
   * - ``PRIMARY_COLOR``
     - '#2E86AB'
     - Primary plot color
   * - ``SECONDARY_COLOR``
     - '#A23B72'
     - Secondary plot color
   * - ``TERTIARY_COLOR``
     - '#F18F01'
     - Tertiary plot color
