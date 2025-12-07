Rigid Body Dynamics
===================

The rigid body module provides comprehensive tools for 3D rotational dynamics including Euler angles, quaternions, and specialized models.

Overview
--------

Rigid body mechanics describes rotational motion of extended objects. The module implements:

- **Euler Angles**: ZYZ convention for orientation
- **Quaternions**: Singularity-free rotation representation
- **Inertia Tensors**: For various geometries
- **Euler's Equations**: Rotational equations of motion
- **Symmetric Top**: Precession and nutation analysis
- **Gyroscope**: Steady precession and applications

Theory
------

Euler Angles (ZYZ Convention)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Three successive rotations define orientation:

1. Rotation by :math:`\phi` about z-axis (precession)
2. Rotation by :math:`\theta` about new y-axis (nutation)
3. Rotation by :math:`\psi` about new z-axis (spin)

.. warning::

   Euler angles have a singularity (gimbal lock) at :math:`\theta = 0, \pi`.
   Use quaternions for numerical integration.

Quaternions
~~~~~~~~~~~

A unit quaternion :math:`q = q_0 + q_1 i + q_2 j + q_3 k` with :math:`|q| = 1` represents rotation without singularities.

Kinematic equation:

.. math::

   \dot{q} = \frac{1}{2} \Omega(\omega) q

where :math:`\Omega(\omega)` is a skew-symmetric matrix built from angular velocity.

Euler's Equations
~~~~~~~~~~~~~~~~~

For principal axes with moments :math:`I_1, I_2, I_3`:

.. math::

   I_1 \dot{\omega}_1 &= (I_2 - I_3) \omega_2 \omega_3 + \tau_1 \\
   I_2 \dot{\omega}_2 &= (I_3 - I_1) \omega_3 \omega_1 + \tau_2 \\
   I_3 \dot{\omega}_3 &= (I_1 - I_2) \omega_1 \omega_2 + \tau_3

Usage Examples
--------------

Rotation Matrices from Euler Angles
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from mechanics_dsl.domains.classical import RigidBodyDynamics
   import numpy as np

   rigid = RigidBodyDynamics()
   
   # Define Euler angles (radians)
   phi = 0.5    # Precession
   theta = 0.3  # Nutation
   psi = 0.2    # Spin
   
   R = rigid.euler_to_rotation_matrix(theta, phi, psi)
   
   print("Rotation matrix:")
   print(R)

Quaternion Operations
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from mechanics_dsl.domains.classical import RigidBodyDynamics
   import numpy as np

   rigid = RigidBodyDynamics()
   
   # Quaternion from rotation (angle, axis)
   angle = np.pi / 4  # 45 degrees
   axis = np.array([0, 0, 1])  # z-axis
   
   # q = cos(θ/2) + sin(θ/2) * (axis)
   q = rigid.axis_angle_to_quaternion(axis, angle)
   
   # Convert to rotation matrix
   R = rigid.quaternion_to_rotation_matrix(*q)
   
   # Quaternion derivative from angular velocity
   omega = np.array([0, 0, 1.0])  # rad/s about z
   q_dot = rigid.quaternion_derivative(q, omega)

Computing Inertia Tensors
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from mechanics_dsl.domains.classical import RigidBodyDynamics
   from mechanics_dsl import InertiaTensor

   rigid = RigidBodyDynamics()
   
   # Solid sphere
   I_sphere = rigid.compute_inertia_sphere(mass=1.0, radius=0.5)
   # I = (2/5) * m * r²
   
   # Solid cylinder (along z-axis)
   I_cylinder = rigid.compute_inertia_cylinder(mass=2.0, radius=0.3, height=1.0)
   
   # Rectangular box
   I_box = rigid.compute_inertia_box(mass=5.0, length=1.0, width=0.5, height=0.2)
   
   # Get principal moments
   eigenvalues, eigenvectors = I_box.principal_axes()
   print(f"Principal moments: {eigenvalues}")

Torque-Free Rotation (Euler's Equations)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from mechanics_dsl.domains.classical import RigidBodyDynamics
   from mechanics_dsl import InertiaTensor
   import numpy as np
   from scipy.integrate import solve_ivp

   rigid = RigidBodyDynamics()
   
   # Asymmetric top
   I = InertiaTensor(I_xx=1.0, I_yy=2.0, I_zz=3.0)
   
   # Initial angular velocity
   omega0 = np.array([1.0, 0.1, 0.1])
   
   def euler_eom(t, omega):
       return rigid.euler_equations_torque_free(omega, I)
   
   # Integrate
   sol = solve_ivp(euler_eom, [0, 10], omega0, dense_output=True)
   
   # Angular momentum is conserved
   L0 = rigid.angular_momentum(omega0, I)
   L_final = rigid.angular_momentum(sol.y[:, -1], I)
   print(f"L conserved: {np.allclose(L0, L_final)}")

Symmetric Top Dynamics
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from mechanics_dsl.domains.classical import SymmetricTop
   import numpy as np

   # Create symmetric top (I1 = I2 ≠ I3)
   top = SymmetricTop(I1=1.0, I3=0.5, mass=1.0, cm_height=0.3)
   
   # Initial conditions
   theta0 = 0.2      # Nutation angle
   phi_dot0 = 0.5    # Precession rate
   psi_dot0 = 10.0   # Spin rate
   
   # Compute effective potential
   V_eff = top.effective_potential(theta0, psi_dot0)
   
   # Steady precession rate (for given spin)
   omega_p = top.steady_precession_rate(psi_dot0, theta0)
   print(f"Precession rate: {omega_p:.4f} rad/s")

Gyroscope Analysis
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from mechanics_dsl.domains.classical import Gyroscope
   import numpy as np

   # Create gyroscope
   gyro = Gyroscope(
       I_spin=0.01,       # Moment about spin axis
       I_transverse=0.02, # Moment about transverse axes
       mass=0.5,
       cm_distance=0.1    # Distance from pivot to CM
   )
   
   # Spin rate
   omega_s = 100.0  # rad/s (fast spin)
   
   # Compute precession rate
   omega_p = gyro.precession_rate(omega_s, theta=np.pi/4)
   
   # Nutation frequency (fast wobble)
   omega_n = gyro.nutation_frequency(omega_s)
   
   print(f"Precession: {omega_p:.4f} rad/s")
   print(f"Nutation: {omega_n:.4f} rad/s")

API Reference
-------------

Classes
~~~~~~~

.. py:class:: RigidBodyDynamics

   Core rigid body dynamics calculations.
   
   .. py:method:: euler_to_rotation_matrix(theta, phi, psi)
   
      Convert ZYZ Euler angles to rotation matrix.
   
   .. py:method:: quaternion_to_rotation_matrix(q0, q1, q2, q3)
   
      Convert quaternion to rotation matrix.
   
   .. py:method:: quaternion_derivative(q, omega)
   
      Compute :math:`\dot{q}` from angular velocity.
   
   .. py:method:: euler_equations_torque_free(omega, I)
   
      Euler's equations for torque-free rotation.
   
   .. py:method:: euler_equations_with_torque(omega, I, torque)
   
      Euler's equations with external torque.
   
   .. py:method:: angular_momentum(omega, I)
   
      Compute :math:`\mathbf{L} = I \cdot \omega`.
   
   .. py:method:: rotational_kinetic_energy(omega, I)
   
      Compute :math:`T = \frac{1}{2} \omega^T I \omega`.

.. py:class:: SymmetricTop(I1, I3, mass, cm_height)

   Symmetric top with I₁ = I₂ ≠ I₃.
   
   .. py:method:: effective_potential(theta, psi_dot)
   
      Compute effective potential for nutation.
   
   .. py:method:: steady_precession_rate(psi_dot, theta)
   
      Compute rate for steady precession.

.. py:class:: Gyroscope(I_spin, I_transverse, mass, cm_distance)

   Gyroscope model.
   
   .. py:method:: precession_rate(omega_spin, theta)
   
      Compute precession angular velocity.
   
   .. py:method:: nutation_frequency(omega_spin)
   
      Compute nutation oscillation frequency.

Data Classes
~~~~~~~~~~~~

.. py:class:: InertiaTensor

   Moment of inertia tensor.
   
   :param I_xx, I_yy, I_zz: Diagonal elements
   :param I_xy, I_xz, I_yz: Off-diagonal elements (default 0)
   
   .. py:method:: to_matrix()
   
      Convert to 3x3 numpy array.
   
   .. py:method:: principal_axes()
   
      Find principal moments and axes.

See Also
--------

- :doc:`lagrangian_mechanics` - Lagrangian formulation for rigid bodies
- :doc:`constraint_physics` - Constrained rigid body motion
