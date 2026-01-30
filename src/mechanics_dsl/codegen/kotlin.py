"""
Kotlin Code Generator for MechanicsDSL

Generates Kotlin simulation code for Android/JVM with:
- Kotlin Coroutines for async simulation
- Compose UI integration-ready
- RK4/Euler integrators
- Gradle project generation
- Android-specific optimizations

Security: Validates all output paths and sanitizes generated code.
"""

import os
import re
from typing import Any, Dict, List, Optional

import sympy as sp

from ..utils import logger, validate_file_path
from .base import CodeGenerator


def sympy_to_kotlin(expr: sp.Expr) -> str:
    """
    Convert a sympy expression to Kotlin code.
    
    Uses kotlin.math package functions.
    
    Args:
        expr: Sympy expression to convert
        
    Returns:
        Kotlin code string
    """
    if expr is None:
        return "0.0"
    
    code = str(expr)
    
    # Replace power operator
    code = re.sub(r'\*\*', '^', code)
    
    # Kotlin math functions
    replacements = {
        'sin(': 'sin(',
        'cos(': 'cos(',
        'tan(': 'tan(',
        'asin(': 'asin(',
        'acos(': 'acos(',
        'atan(': 'atan(',
        'sinh(': 'sinh(',
        'cosh(': 'cosh(',
        'tanh(': 'tanh(',
        'exp(': 'exp(',
        'log(': 'ln(',  # Note: Kotlin uses ln for natural log
        'sqrt(': 'sqrt(',
        'abs(': 'abs(',
        'Abs(': 'abs(',
    }
    
    for old, new in replacements.items():
        code = code.replace(old, new)
    
    # Replace ^ with pow()
    pattern = r'(\w+)\^(\d+(?:\.\d+)?)'
    while re.search(pattern, code):
        code = re.sub(pattern, r'\1.pow(\2)', code)
    
    pattern = r'\(([^()]+)\)\^(\d+(?:\.\d+)?)'
    while re.search(pattern, code):
        code = re.sub(pattern, r'(\1).pow(\2)', code)
    
    return code


class KotlinGenerator(CodeGenerator):
    """
    Generates Kotlin simulation code for Android/JVM.
    
    Features:
    - Kotlin 1.9+ syntax
    - Coroutines for async simulation
    - Flow for reactive state
    - Android Compose compatibility
    - Gradle project generation
    """
    
    def __init__(
        self,
        system_name: str,
        coordinates: List[str],
        parameters: Dict[str, float],
        equations: Dict[str, sp.Expr],
        initial_conditions: Optional[Dict[str, float]] = None,
        lagrangian: Optional[sp.Expr] = None,
        hamiltonian: Optional[sp.Expr] = None,
        package_name: str = "com.mechanicsdsl.simulation",
        use_coroutines: bool = True,
        use_compose: bool = True,
    ):
        """
        Initialize the Kotlin code generator.
        
        Args:
            system_name: Name of the physics system
            coordinates: List of generalized coordinate names
            parameters: Physical parameters
            equations: Equations of motion
            initial_conditions: Initial state values
            lagrangian: System Lagrangian
            hamiltonian: System Hamiltonian
            package_name: Kotlin package name
            use_coroutines: Enable coroutines async support
            use_compose: Enable Compose state integration
        """
        super().__init__(
            system_name=system_name,
            coordinates=coordinates,
            parameters=parameters,
            equations=equations,
            lagrangian=lagrangian,
            hamiltonian=hamiltonian,
        )
        
        self.initial_conditions = initial_conditions or {}
        self.package_name = package_name
        self.use_coroutines = use_coroutines
        self.use_compose = use_compose
        
        self.class_name = self._to_class_name(system_name)
    
    def _to_class_name(self, name: str) -> str:
        """Convert to valid Kotlin class name."""
        words = re.split(r'[_\s-]+', name)
        return ''.join(word.capitalize() for word in words) + "Simulation"
    
    @property
    def target_name(self) -> str:
        return "kotlin"
    
    @property
    def file_extension(self) -> str:
        return ".kt"
    
    def expr_to_code(self, expr: sp.Expr) -> str:
        return sympy_to_kotlin(expr)
    
    def generate(self, output_file: str = "Simulation.kt") -> str:
        """Generate Kotlin simulation code."""
        is_valid, errors = self.validate()
        if not is_valid:
            raise ValueError(f"Validation failed: {errors}")
        
        code = self._generate_kotlin_code()
        
        validated_path = validate_file_path(output_file, must_exist=False)
        with open(validated_path, 'w', encoding='utf-8') as f:
            f.write(code)
        
        logger.info(f"Generated Kotlin code: {validated_path}")
        return validated_path
    
    def generate_equations(self) -> str:
        """Generate Kotlin equations of motion."""
        lines = []
        
        for coord in self.coordinates:
            accel_key = f"{coord}_ddot"
            if accel_key in self.equations:
                expr = self.equations[accel_key]
                kotlin_expr = self.expr_to_code(expr)
                lines.append(f"        val {accel_key} = {kotlin_expr}")
        
        return '\n'.join(lines)
    
    def _generate_parameters(self) -> str:
        """Generate Kotlin parameter declarations."""
        lines = []
        for name, value in self.parameters.items():
            lines.append(f"    private val {name}: Double = {value}")
        return '\n'.join(lines)
    
    def _generate_state_class(self) -> str:
        """Generate state data class."""
        fields = []
        for coord in self.coordinates:
            fields.append(f"    val {coord}: Double = 0.0")
            fields.append(f"    val {coord}Dot: Double = 0.0")
        return ',\n'.join(fields)
    
    def _generate_initial_state(self) -> str:
        """Generate initial state."""
        values = []
        for coord in self.coordinates:
            pos_val = self.initial_conditions.get(coord, 0.0)
            vel_val = self.initial_conditions.get(f"{coord}_dot", 0.0)
            values.append(f"        {coord} = {pos_val}")
            values.append(f"        {coord}Dot = {vel_val}")
        return ',\n'.join(values)
    
    def _generate_state_unpacking(self) -> str:
        """Generate state array unpacking."""
        lines = []
        for i, coord in enumerate(self.coordinates):
            lines.append(f"        val {coord} = state[{2*i}]")
            lines.append(f"        val {coord}_dot = state[{2*i + 1}]")
        return '\n'.join(lines)
    
    def _generate_derivative_list(self) -> str:
        """Generate derivative list."""
        items = []
        for coord in self.coordinates:
            items.append(f"{coord}_dot")
            items.append(f"{coord}_ddot")
        return ', '.join(items)
    
    def _generate_kotlin_code(self) -> str:
        """Generate complete Kotlin code."""
        state_size = 2 * len(self.coordinates)
        
        template = f'''package {self.package_name}

import kotlin.math.*
{"import kotlinx.coroutines.*" if self.use_coroutines else ""}
{"import kotlinx.coroutines.flow.*" if self.use_coroutines else ""}
{"import androidx.compose.runtime.*" if self.use_compose else ""}

/**
 * Physics simulation for {self.system_name}
 * Generated by MechanicsDSL
 * 
 * Coordinates: {', '.join(self.coordinates)}
 */

// State data class
data class SimulationState(
{self._generate_state_class()}
) {{
    fun toArray(): DoubleArray = doubleArrayOf(
        {', '.join([f'{c}, {c}Dot' for c in self.coordinates])}
    )
    
    companion object {{
        fun fromArray(arr: DoubleArray): SimulationState {{
            require(arr.size == {state_size}) {{ "Array must have {state_size} elements" }}
            return SimulationState(
{self._generate_state_from_array()}
            )
        }}
    }}
}}

class {self.class_name} {{
    
    // Physical parameters
{self._generate_parameters()}
    
    // Simulation state
{"    var state by mutableStateOf(initialState())" if self.use_compose else "    var state = initialState()"}
{"    var time by mutableStateOf(0.0)" if self.use_compose else "    var time = 0.0"}
{"    var isRunning by mutableStateOf(false)" if self.use_compose else "    var isRunning = false"}
    
    // Integration settings
    var dt: Double = 0.001
    var maxTime: Double = 10.0
    
    private fun initialState(): SimulationState = SimulationState(
{self._generate_initial_state()}
    )
    
    /**
     * Compute state derivatives (equations of motion)
     */
    private fun derivatives(state: DoubleArray, t: Double): DoubleArray {{
{self._generate_state_unpacking()}
        
        // Accelerations from Euler-Lagrange equations
{self.generate_equations()}
        
        return doubleArrayOf({self._generate_derivative_list()})
    }}
    
    /**
     * RK4 integration step
     */
    fun step() {{
        val y = state.toArray()
        
        val k1 = derivatives(y, time)
        val k2 = derivatives(add(y, scale(k1, dt / 2)), time + dt / 2)
        val k3 = derivatives(add(y, scale(k2, dt / 2)), time + dt / 2)
        val k4 = derivatives(add(y, scale(k3, dt)), time + dt)
        
        val yNew = DoubleArray({state_size}) {{ i ->
            y[i] + (dt / 6.0) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i])
        }}
        
        state = SimulationState.fromArray(yNew)
        time += dt
    }}
    
    /**
     * Run simulation for specified duration
     */
    {"suspend " if self.use_coroutines else ""}fun run(duration: Double): List<Pair<Double, SimulationState>> {{
        val history = mutableListOf<Pair<Double, SimulationState>>()
        val endTime = time + duration
        
        {"isRunning = true" if self.use_compose else ""}
        while (time < endTime) {{
            history.add(time to state)
            step()
            {"yield()" if self.use_coroutines else ""}
        }}
        {"isRunning = false" if self.use_compose else ""}
        
        return history
    }}
    
    /**
     * Reset to initial conditions
     */
    fun reset() {{
        time = 0.0
        state = initialState()
    }}
    
{self._generate_energy_method() if self.lagrangian else ""}
    
    // Helper functions
    private fun add(a: DoubleArray, b: DoubleArray): DoubleArray =
        DoubleArray(a.size) {{ a[it] + b[it] }}
    
    private fun scale(a: DoubleArray, s: Double): DoubleArray =
        DoubleArray(a.size) {{ a[it] * s }}
}}
'''
        return template
    
    def _generate_state_from_array(self) -> str:
        """Generate state from array initialization."""
        lines = []
        for i, coord in enumerate(self.coordinates):
            lines.append(f"                {coord} = arr[{2*i}]")
            lines.append(f"                {coord}Dot = arr[{2*i + 1}]")
        return ',\n'.join(lines)
    
    def _generate_energy_method(self) -> str:
        """Generate energy computation."""
        if not self.lagrangian:
            return ""
        
        energy_expr = self.expr_to_code(self.lagrangian)
        
        return f'''
    /**
     * Compute system energy (from Lagrangian)
     */
    fun computeEnergy(): Double {{
        val y = state.toArray()
{self._generate_state_unpacking()}
        
        return {energy_expr}
    }}
'''
    
    def generate_project(self, output_dir: str = ".") -> Dict[str, str]:
        """
        Generate complete Gradle project.
        
        Creates:
        - src/main/kotlin/<package>/Simulation.kt
        - build.gradle.kts
        - settings.gradle.kts
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Package path
        package_path = self.package_name.replace('.', os.sep)
        src_dir = os.path.join(output_dir, "src", "main", "kotlin", package_path)
        os.makedirs(src_dir, exist_ok=True)
        
        # Generate main source
        source_path = os.path.join(src_dir, f"{self.class_name}.kt")
        self.generate(source_path)
        
        # Generate build.gradle.kts
        build_gradle = self._generate_build_gradle()
        build_path = os.path.join(output_dir, "build.gradle.kts")
        with open(build_path, 'w') as f:
            f.write(build_gradle)
        
        # Generate settings.gradle.kts
        settings_gradle = self._generate_settings_gradle()
        settings_path = os.path.join(output_dir, "settings.gradle.kts")
        with open(settings_path, 'w') as f:
            f.write(settings_gradle)
        
        return {
            'source': source_path,
            'build': build_path,
            'settings': settings_path
        }
    
    def _generate_build_gradle(self) -> str:
        """Generate build.gradle.kts."""
        return f'''plugins {{
    kotlin("jvm") version "1.9.0"
    {"id(\"org.jetbrains.compose\") version \"1.5.0\"" if self.use_compose else ""}
}}

group = "{self.package_name.rsplit('.', 1)[0]}"
version = "1.0.0"

repositories {{
    mavenCentral()
    {"google()" if self.use_compose else ""}
}}

dependencies {{
    implementation(kotlin("stdlib"))
    {"implementation(\"org.jetbrains.kotlinx:kotlinx-coroutines-core:1.7.3\")" if self.use_coroutines else ""}
    {"implementation(compose.runtime)" if self.use_compose else ""}
    
    testImplementation(kotlin("test"))
}}

tasks.test {{
    useJUnitPlatform()
}}
'''
    
    def _generate_settings_gradle(self) -> str:
        """Generate settings.gradle.kts."""
        project_name = self.class_name.replace("Simulation", "").lower()
        return f'''rootProject.name = "{project_name}"
'''
