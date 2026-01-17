"""
ROS2 integration for MechanicsDSL.

Provides ROS2 node for real-time physics simulation in robotics.

Requires: ROS2 (Humble/Iron) with rclpy

Example:
    ros2 run mechanics_dsl_ros mechanics_dsl_node --ros-args -p dsl_file:=robot_arm.mdsl
"""

from pathlib import Path
from typing import Optional

import numpy as np

try:
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Float64MultiArray

    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    Node = object

try:
    from mechanics_dsl import PhysicsCompiler
except ImportError:
    PhysicsCompiler = None  # type: ignore[misc]


class MechanicsDSLNode:
    """
    ROS2 node wrapper for MechanicsDSL simulations.

    Publishes simulation state at a fixed rate and accepts
    parameter updates via topics.

    Topics Published:
        /state (Float64MultiArray): Full state vector [q1, q1_dot, q2, q2_dot, ...]
        /joint_states (JointState): If system has named joints

    Topics Subscribed:
        /parameters (Float64MultiArray): Parameter updates
        /reset (std_msgs/Empty): Reset to initial conditions

    Parameters:
        dsl_file (str): Path to .mdsl file
        dsl_code (str): Inline DSL code (alternative to file)
        rate (float): Simulation/publish rate in Hz
        dt (float): Simulation timestep
    """

    def __new__(cls, *args, **kwargs):
        if not ROS2_AVAILABLE:
            raise ImportError("ROS2 is not installed. This requires a ROS2 environment.")
        return cls._create_node(*args, **kwargs)

    @classmethod
    def _create_node(
        cls,
        node_name: str = "mechanics_dsl_node",
        dsl_code: Optional[str] = None,
        dsl_file: Optional[str] = None,
        rate: float = 100.0,
        dt: float = 0.001,
    ) -> "Node":
        """Create the ROS2 node."""

        class _MechanicsNode(Node):
            def __init__(self):
                super().__init__(node_name)

                # Parameters
                self.declare_parameter("dsl_file", "")
                self.declare_parameter("dsl_code", dsl_code or "")
                self.declare_parameter("rate", rate)
                self.declare_parameter("dt", dt)

                # Load DSL
                self._load_dsl()

                # State
                self.current_state = None
                self.current_time = 0.0

                # Publishers
                self.state_pub = self.create_publisher(Float64MultiArray, "state", 10)
                self.joint_pub = self.create_publisher(JointState, "joint_states", 10)

                # Subscribers
                self.create_subscription(Float64MultiArray, "parameters", self._param_callback, 10)

                # Timer for simulation loop
                timer_period = 1.0 / self.get_parameter("rate").value
                self.timer = self.create_timer(timer_period, self._timer_callback)

                self.get_logger().info(f"MechanicsDSL node started at {rate} Hz")

            def _load_dsl(self):
                """Load and compile DSL code."""
                if PhysicsCompiler is None:
                    self.get_logger().error("PhysicsCompiler not available")
                    return

                self._compiler = PhysicsCompiler()

                # Try file first with path validation
                file_path = self.get_parameter("dsl_file").value
                if file_path:
                    # Validate the file path
                    resolved = Path(file_path).resolve()
                    # Only allow .mdsl and .dsl extensions
                    if resolved.suffix.lower() not in (".mdsl", ".dsl", ".txt"):
                        self.get_logger().error(f"Invalid file extension: {resolved.suffix}")
                        return
                    if resolved.exists():
                        with open(resolved, "r") as f:
                            code = f.read()
                    else:
                        code = self.get_parameter("dsl_code").value
                else:
                    code = self.get_parameter("dsl_code").value

                if not code:
                    self.get_logger().warn("No DSL code provided")
                    return

                result = self._compiler.compile_dsl(code)
                if not result["success"]:
                    self.get_logger().error(f"Compilation failed: {result.get('error')}")
                    return

                # Get initial state
                self.coordinates = result.get("coordinates", [])
                self._initialize_state()

                self.get_logger().info(f"Loaded system: {result.get('system_name')}")
                self.get_logger().info(f"Coordinates: {self.coordinates}")

            def _initialize_state(self):
                """Initialize state from initial conditions."""
                ic = self._compiler.simulator.initial_conditions
                n_coords = len(self.coordinates)
                self.current_state = np.zeros(n_coords * 2)

                for i, coord in enumerate(self.coordinates):
                    self.current_state[2 * i] = ic.get(coord, 0.0)
                    self.current_state[2 * i + 1] = ic.get(f"{coord}_dot", 0.0)

                self.current_time = 0.0

            def _timer_callback(self):
                """Simulation step and publish."""
                if self.current_state is None:
                    return

                dt = self.get_parameter("dt").value

                # Step simulation
                try:
                    # Use equations of motion for one step
                    if hasattr(self._compiler, "simulator"):
                        dydt = self._compiler.simulator.equations_of_motion(
                            self.current_time, self.current_state
                        )
                        self.current_state += dydt * dt
                        self.current_time += dt
                except Exception as e:
                    self.get_logger().error(f"Simulation step failed: {e}")
                    return

                # Publish state
                state_msg = Float64MultiArray()
                state_msg.data = self.current_state.tolist()
                self.state_pub.publish(state_msg)

                # Publish joint states
                joint_msg = JointState()
                joint_msg.header.stamp = self.get_clock().now().to_msg()
                joint_msg.name = self.coordinates
                joint_msg.position = [
                    self.current_state[2 * i] for i in range(len(self.coordinates))
                ]
                joint_msg.velocity = [
                    self.current_state[2 * i + 1] for i in range(len(self.coordinates))
                ]
                self.joint_pub.publish(joint_msg)

            def _param_callback(self, msg: Float64MultiArray):
                """Update parameters from topic."""
                # Assume parameters match order in compiler
                params = list(self._compiler.simulator.parameters.keys())
                new_params = {}
                for i, val in enumerate(msg.data):
                    if i < len(params):
                        new_params[params[i]] = val

                self._compiler.simulator.set_parameters(new_params)
                self.get_logger().info(f"Updated parameters: {new_params}")

        return _MechanicsNode()


def create_ros2_package(
    package_name: str = "mechanics_dsl_ros",
    output_dir: str = ".",
) -> str:
    """
    Generate a ROS2 package structure for MechanicsDSL.

    Creates:
        {package_name}/
            package.xml
            setup.py
            {package_name}/
                __init__.py
                node.py
            launch/
                simulation.launch.py
    """
    from pathlib import Path

    pkg_dir = Path(output_dir) / package_name
    src_dir = pkg_dir / package_name
    launch_dir = pkg_dir / "launch"

    # Create directories
    src_dir.mkdir(parents=True, exist_ok=True)
    launch_dir.mkdir(exist_ok=True)

    # package.xml
    package_xml = f"""<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>{package_name}</name>
  <version>1.0.0</version>
  <description>MechanicsDSL ROS2 integration</description>
  <maintainer email="user@example.com">User</maintainer>
  <license>MIT</license>

  <depend>rclpy</depend>
  <depend>std_msgs</depend>
  <depend>sensor_msgs</depend>

  <exec_depend>mechanicsdsl-core</exec_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
"""
    (pkg_dir / "package.xml").write_text(package_xml)

    # setup.py
    setup_py = f"""from setuptools import setup
import os
from glob import glob

package_name = '{package_name}'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools', 'mechanicsdsl-core'],
    zip_safe=True,
    entry_points={{
        'console_scripts': [
            'mechanics_dsl_node = {package_name}.node:main',
        ],
    }},
)
"""
    (pkg_dir / "setup.py").write_text(setup_py)

    # Create resource directory
    (pkg_dir / "resource").mkdir(exist_ok=True)
    (pkg_dir / "resource" / package_name).touch()

    # __init__.py
    (src_dir / "__init__.py").write_text("")

    # node.py
    node_py = '''"""MechanicsDSL ROS2 node."""
import rclpy
from mechanics_dsl.integrations.ros2 import MechanicsDSLNode


def main(args=None):
    rclpy.init(args=args)
    node = MechanicsDSLNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
'''
    (src_dir / "node.py").write_text(node_py)

    # Launch file
    launch_py = f"""from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='{package_name}',
            executable='mechanics_dsl_node',
            name='physics_sim',
            output='screen',
            parameters=[{{
                'dsl_file': '',
                'rate': 100.0,
                'dt': 0.001,
            }}]
        )
    ])
"""
    (launch_dir / "simulation.launch.py").write_text(launch_py)

    return str(pkg_dir)


__all__ = [
    "MechanicsDSLNode",
    "create_ros2_package",
    "ROS2_AVAILABLE",
]
