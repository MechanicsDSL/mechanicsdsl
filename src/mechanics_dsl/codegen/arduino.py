"""
Arduino Code Generator for MechanicsDSL

Generates Arduino-compatible C++ code for embedded physics simulations.
Features:
- Optimized for microcontroller constraints (limited RAM/ROM)
- Serial plotter output for real-time visualization
- PWM/servo output for physical feedback
- Fixed-point arithmetic option for speed
"""

from typing import Dict, List, Optional

import sympy as sp
from sympy.printing.c import ccode

from ..utils import logger
from .base import CodeGenerator


def sympy_to_c_arduino(expr: sp.Expr) -> str:
    """
    Convert a sympy expression to C code for Arduino.

    Args:
        expr: Sympy expression to convert

    Returns:
        C code string (using float literals)
    """
    if expr is None:
        return "0.0f"
    try:
        return ccode(expr)
    except Exception as e:
        logger.warning(f"Failed to convert expression to C: {e}")
        return f"0.0f /* ERROR: {e} */"


class ArduinoGenerator(CodeGenerator):
    """
    Generates Arduino sketch (.ino) files for embedded simulations.

    Features:
    - Fixed-point arithmetic option for speed
    - RAM-optimized data structures (float instead of double)
    - Serial plotter output for real-time visualization
    - PWM/servo output for physical feedback
    - Real-time timing management

    Example:
        >>> import sympy as sp
        >>> theta, g, l = sp.symbols('theta g l')
        >>> gen = ArduinoGenerator(
        ...     system_name="pendulum",
        ...     coordinates=['theta'],
        ...     parameters={'g': 9.81, 'l': 1.0},
        ...     initial_conditions={'theta': 0.5, 'theta_dot': 0.0},
        ...     equations={'theta_ddot': -g/l * sp.sin(theta)},
        ...     use_serial_plotter=True,
        ...     servo_pin=9
        ... )
        >>> gen.generate("pendulum.ino")
        'pendulum.ino'

    Attributes:
        use_serial_plotter: Enable Serial Plotter output
        servo_pin: Pin for servo output (None to disable)
    """

    def __init__(
        self,
        system_name: str,
        coordinates: List[str],
        parameters: Dict[str, float],
        initial_conditions: Dict[str, float],
        equations: Dict[str, sp.Expr],
        lagrangian: Optional[sp.Expr] = None,
        hamiltonian: Optional[sp.Expr] = None,
        forces: Optional[List[sp.Expr]] = None,
        constraints: Optional[List[sp.Expr]] = None,
        use_serial_plotter: bool = True,
        servo_pin: Optional[int] = None,
    ) -> None:
        """
        Initialize the Arduino code generator.

        Args:
            system_name: Name of the physics system
            coordinates: List of generalized coordinate names
            parameters: Physical parameters
            initial_conditions: Initial state values
            equations: Acceleration equations
            lagrangian: Optional Lagrangian
            hamiltonian: Optional Hamiltonian
            forces: Optional non-conservative forces
            constraints: Optional holonomic constraints
            use_serial_plotter: Enable Serial Plotter output
            servo_pin: Pin for servo output (None to disable)
        """
        super().__init__(
            system_name=system_name,
            coordinates=coordinates,
            parameters=parameters,
            initial_conditions=initial_conditions,
            equations=equations,
            lagrangian=lagrangian,
            hamiltonian=hamiltonian,
            forces=forces,
            constraints=constraints,
        )
        self.use_serial_plotter = use_serial_plotter
        self.servo_pin = servo_pin

    @property
    def target_name(self) -> str:
        """Target platform identifier."""
        return "arduino"

    @property
    def file_extension(self) -> str:
        """File extension for generated code."""
        return ".ino"

    def expr_to_code(self, expr: sp.Expr) -> str:
        """
        Convert sympy expression to C code for Arduino.

        Args:
            expr: Sympy expression

        Returns:
            C code string
        """
        return sympy_to_c_arduino(expr)

    def generate(self, output_file: str) -> str:
        """
        Generate Arduino sketch file.

        Args:
            output_file: Path to output file

        Returns:
            Path to generated file

        Raises:
            ValueError: If validation fails
        """
        self.validate_or_raise()

        logger.info(f"Generating Arduino code for {self.system_name}")

        code = self._generate_source()

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(code)

        logger.info(f"Generated {output_file}")
        return output_file

    def generate_equations(self) -> str:
        """Generate equations code."""
        lines = []
        idx = 0
        for coord in self.coordinates:
            accel_key = f"{coord}_ddot"
            lines.append(f"  dydt[{idx}] = state[{idx+1}];")
            if accel_key in self.equations and self.equations[accel_key] is not None:
                expr = self.equations[accel_key]
                c_expr = self.expr_to_code(expr)
                lines.append(f"  dydt[{idx+1}] = {c_expr};")
            else:
                lines.append(f"  dydt[{idx+1}] = 0.0;")
            idx += 2
        return "\n".join(lines)

    def _generate_source(self) -> str:
        """Generate complete Arduino sketch."""
        state_dim = len(self.coordinates) * 2

        params = "\n".join(f"const float {name} = {val}f;" for name, val in self.parameters.items())

        init_vals = []
        for coord in self.coordinates:
            init_vals.append(str(self.initial_conditions.get(coord, 0.0)) + "f")
            init_vals.append(str(self.initial_conditions.get(f"{coord}_dot", 0.0)) + "f")
        init_str = ", ".join(init_vals)

        # State unpacking
        unpack = "\n".join(
            f"  float {c} = state[{2*i}]; float {c}_dot = state[{2*i+1}];"
            for i, c in enumerate(self.coordinates)
        )

        # Servo code
        servo_include = ""
        servo_init = ""
        servo_update = ""
        if self.servo_pin is not None:
            servo_include = "#include <Servo.h>\nServo outputServo;"
            servo_init = f"outputServo.attach({self.servo_pin});"
            servo_update = """
  // Map first coordinate to servo angle (0-180)
  int angle = constrain(map(state[0] * 100, -314, 314, 0, 180), 0, 180);
  outputServo.write(angle);"""

        # Serial output
        serial_output = ""
        if self.use_serial_plotter:
            serial_output = """
  // Serial Plotter format
  Serial.print(t);
  for (int i = 0; i < STATE_DIM; i++) {
    Serial.print(",");
    Serial.print(state[i], 4);
  }
  Serial.println();"""

        return f"""/*
 * Arduino Physics Simulation: {self.system_name}
 * Generated by MechanicsDSL
 * 
 * Upload to Arduino and open Serial Plotter (Tools > Serial Plotter)
 */

{servo_include}

// Physical Parameters
{params}

#define STATE_DIM {state_dim}

// Simulation state
float state[STATE_DIM] = {{ {init_str} }};
float t = 0.0f;
const float dt = 0.01f;  // 10ms timestep
unsigned long lastMicros = 0;

// Compute derivatives
void computeDerivatives(const float* state, float* dydt) {{
{unpack}

{self.generate_equations()}
}}

// RK4 integration step (using float for speed)
void rk4Step() {{
  float k1[STATE_DIM], k2[STATE_DIM], k3[STATE_DIM], k4[STATE_DIM];
  float temp[STATE_DIM];
  
  computeDerivatives(state, k1);
  
  for (int i = 0; i < STATE_DIM; i++) temp[i] = state[i] + 0.5f * dt * k1[i];
  computeDerivatives(temp, k2);
  
  for (int i = 0; i < STATE_DIM; i++) temp[i] = state[i] + 0.5f * dt * k2[i];
  computeDerivatives(temp, k3);
  
  for (int i = 0; i < STATE_DIM; i++) temp[i] = state[i] + dt * k3[i];
  computeDerivatives(temp, k4);
  
  for (int i = 0; i < STATE_DIM; i++) {{
    state[i] += dt * (k1[i] + 2.0f*k2[i] + 2.0f*k3[i] + k4[i]) / 6.0f;
  }}
  
  t += dt;
}}

void setup() {{
  Serial.begin(115200);
  while (!Serial) {{ ; }}  // Wait for serial connection
  
  {servo_init}
  
  Serial.println("MechanicsDSL: {self.system_name}");
  Serial.println("Time,{','.join([c + ',' + c + '_dot' for c in self.coordinates]).strip(',')}");
  
  lastMicros = micros();
}}

void loop() {{
  // Maintain consistent timing
  unsigned long now = micros();
  if (now - lastMicros >= (unsigned long)(dt * 1000000)) {{
    lastMicros = now;
    
    // Integration step
    rk4Step();
{servo_update}
{serial_output}
  }}
}}

// Reset simulation (call from Serial command if needed)
void resetSimulation() {{
  float initial[STATE_DIM] = {{ {init_str} }};
  for (int i = 0; i < STATE_DIM; i++) {{
    state[i] = initial[i];
  }}
  t = 0.0f;
}}
"""
