"""
Main compiler and system serialization for MechanicsDSL
"""
import json
import numpy as np
import pickle
import time
import gc
import os
import traceback
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple, Any
import sympy as sp
import subprocess
import sys
from .codegen.cpp import CppGenerator

from .utils import (
    logger, config, profile_function, validate_file_path,
    LRUCache, _perf_monitor
)
from .parser import (
    tokenize, MechanicsParser, ASTNode, Expression,
    SystemDef, VarDef, ParameterDef, DefineDef,
    LagrangianDef, HamiltonianDef, TransformDef,
    ConstraintDef, NonHolonomicConstraintDef, ForceDef,
    DampingDef, InitialCondition, FluidDef, BoundaryDef, RegionDef
)
from .symbolic import SymbolicEngine
from .solver import NumericalSimulator
from .visualization import MechanicsVisualizer
from .units import UnitSystem

__version__ = "1.0.0"

class SystemSerializer:
    """Serialize and deserialize compiled physics systems"""
    
    @staticmethod
    def export_system(compiler: 'PhysicsCompiler', filename: str, 
                     format: str = 'json') -> bool:
        """
        Export compiled system to file
        
        Args:
            compiler: PhysicsCompiler instance
            filename: Output filename
            format: Export format ('json' or 'pickle')
            
        Returns:
            True if successful
        """
        try:
            state = {
                'version': __version__,
                'system_name': compiler.system_name,
                'variables': compiler.variables,
                'parameters': compiler.parameters_def,
                'initial_conditions': compiler.initial_conditions,
                'lagrangian': str(compiler.lagrangian) if compiler.lagrangian else None,
                'hamiltonian': str(compiler.hamiltonian) if compiler.hamiltonian else None,
                'coordinates': compiler.get_coordinates(),
                'use_hamiltonian': compiler.use_hamiltonian_formulation,
                'constraints': [str(c) for c in compiler.constraints],
                'transforms': {k: str(v) for k, v in compiler.transforms.items()},
            }
            
            if format == 'json':
                with open(filename, 'w') as f:
                    json.dump(state, f, indent=2)
            elif format == 'pickle':
                with open(filename, 'wb') as f:
                    pickle.dump(state, f)
            else:
                raise ValueError(f"Unknown format: {format}")
            
            logger.info(f"System exported to {filename}")
            return True
            
        except (IOError, OSError, PermissionError, ValueError) as e:
            logger.error(f"Export failed: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected export error: {type(e).__name__}: {e}")
            return False
    
    @staticmethod
    def import_system(filename: str) -> Optional[dict]:
        """
        Import system state from file with validation.
        
        Args:
            filename: Input filename (validated)
            
        Returns:
            System state dictionary or None if failed
            
        Raises:
            TypeError: If filename is not a string
            ValueError: If filename is invalid
            FileNotFoundError: If file doesn't exist
        """
        validate_file_path(filename, must_exist=True)
        
        try:
            if filename.endswith('.json'):
                with open(filename, 'r', encoding='utf-8') as f:
                    state = json.load(f)
            elif filename.endswith('.pkl') or filename.endswith('.pickle'):
                with open(filename, 'rb') as f:
                    state = pickle.load(f)
            else:
                # Try JSON first
                with open(filename, 'r', encoding='utf-8') as f:
                    state = json.load(f)
            
            logger.info(f"System imported from {filename}")
            return state
            
        except Exception as e:
            logger.error(f"Import failed: {e}")
            return None

class ParticleGenerator:
    """Generates discrete particle positions from geometric regions"""
    
    @staticmethod
    def generate(region: RegionDef, spacing: float) -> List[Tuple[float, float]]:
        """
        Generate grid of particles within a region.
        """
        if region.shape == "rectangle":
            x_range = region.constraints.get('x', (0, 0))
            y_range = region.constraints.get('y', (0, 0))
            
            # Create grid
            x_points = np.arange(x_range[0], x_range[1], spacing)
            y_points = np.arange(y_range[0], y_range[1], spacing)
            
            # Meshgrid
            xx, yy = np.meshgrid(x_points, y_points)
            
            # Flatten to list of (x, y) tuples
            return list(zip(xx.flatten(), yy.flatten()))
            
        elif region.shape == "line":
            # Useful for boundaries
            x_range = region.constraints.get('x', (0, 0))
            y_range = region.constraints.get('y', (0, 0))
            
            if x_range[0] == x_range[1]: # Vertical line
                y_points = np.arange(y_range[0], y_range[1], spacing/2.0) # Denser walls
                return [(x_range[0], y) for y in y_points]
            else: # Horizontal line
                x_points = np.arange(x_range[0], x_range[1], spacing/2.0)
                return [(x, y_range[0]) for x in x_points]
        
        return []

class PhysicsCompiler:
    """
    Main compiler class - v6.0.0 with enterprise-grade features.
    
    Production-ready physics DSL compiler with comprehensive validation,
    cross-platform support, and security hardening.
    
    Features:
    - Cross-platform timeout support (Windows/Unix)
    - Safe AST-based parsing (no eval())
    - Comprehensive input validation
    - Specific exception handling
    - Extensive type hints
    - Production-ready error recovery
    
    Example:
        >>> compiler = PhysicsCompiler()
        >>> result = compiler.compile_dsl("\\system{pendulum}\\lagrangian{x^2}")
        >>> if result['success']:
        ...     solution = compiler.simulate((0, 10))
        ...     compiler.animate(solution)
    """
    
    def __init__(self):
        self.ast: List[ASTNode] = []
        self.variables: Dict[str, dict] = {}
        self.definitions: Dict[str, dict] = {}
        self.parameters_def: Dict[str, dict] = {}
        self.system_name: str = "unnamed_system"
        self.lagrangian: Optional[Expression] = None
        self.hamiltonian: Optional[Expression] = None
        self.transforms: Dict[str, dict] = {}
        self.constraints: List[Expression] = []
        self.non_holonomic_constraints: List[Expression] = []
        self.forces: List[Expression] = []
        self.damping_forces: List[Expression] = []
        self.initial_conditions: Dict[str, float] = {}
        self.fluid_particles: List[Dict[str, float]] = [] # [{'x': 1.0, 'y': 2.0, 'm': 0.01}, ...]
        self.boundary_particles: List[Dict[str, float]] = []
        self.smoothing_length: float = 0.1 # Default 'h'
        
        self.symbolic = SymbolicEngine()
        self.simulator = NumericalSimulator(self.symbolic)
        self.visualizer = MechanicsVisualizer()
        self.unit_system = UnitSystem()
        
        self.compilation_time: Optional[float] = None
        self.equations: Optional[Any] = None
        self.use_hamiltonian_formulation: bool = False
        
        # v6.0: Memory management
        if config.enable_memory_monitoring:
            gc.set_threshold(*config._gc_threshold)
            _perf_monitor.snapshot_memory("compiler_init")
        
        logger.debug("PhysicsCompiler initialized (v6.0.0)")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup"""
        self.cleanup()
        return False
    
    def cleanup(self) -> None:
        """v6.0: Cleanup resources and trigger garbage collection"""
        if config.enable_memory_monitoring:
            _perf_monitor.snapshot_memory("pre_cleanup")
        
        # Clear large caches
        if hasattr(self.symbolic, '_cache') and self.symbolic._cache:
            if isinstance(self.symbolic._cache, LRUCache):
                self.symbolic._cache.clear()
        
        # Clear compiled equations
        self.equations = None
        
        # Trigger garbage collection
        if config.enable_memory_monitoring:
            collected = gc.collect()
            logger.debug(f"Garbage collection: {collected} objects collected")
            _perf_monitor.snapshot_memory("post_cleanup")

    @profile_function
    def compile_dsl(self, dsl_source: str, use_hamiltonian: bool = False,
                   use_constraints: bool = True) -> dict:
        """
        Complete compilation pipeline with comprehensive validation.
        
        Args:
            dsl_source: DSL source code (must be non-empty string)
            use_hamiltonian: Force Hamiltonian formulation
            use_constraints: Apply constraint handling
            
        Returns:
            Compilation result dictionary with 'success' key
            
        Raises:
            TypeError: If dsl_source is not a string
            ValueError: If dsl_source is empty or invalid
            
        Example:
            >>> compiler = PhysicsCompiler()
            >>> result = compiler.compile_dsl(r"\\system{test}\\lagrangian{x^2}")
            >>> assert result['success']
        """
        # Comprehensive input validation
        if dsl_source is None:
            logger.error("compile_dsl: dsl_source is None")
            return {
                'success': False,
                'error': 'dsl_source cannot be None',
                'compilation_time': 0.0
            }
        
        if not isinstance(dsl_source, str):
            error_msg = f"dsl_source must be str, got {type(dsl_source).__name__}"
            logger.error(f"compile_dsl: {error_msg}")
            raise TypeError(error_msg)
        
        dsl_source = dsl_source.strip()
        if not dsl_source:
            error_msg = "dsl_source cannot be empty"
            logger.error(f"compile_dsl: {error_msg}")
            raise ValueError(error_msg)
        
        if len(dsl_source) > 1_000_000:  # 1MB limit
            error_msg = f"dsl_source too large ({len(dsl_source)} chars), max 1MB"
            logger.error(f"compile_dsl: {error_msg}")
            raise ValueError(error_msg)
        
        # Check for potentially malicious patterns
        dangerous_patterns = ['__import__', 'eval(', 'exec(', 'compile(']
        for pattern in dangerous_patterns:
            if pattern in dsl_source:
                logger.warning(f"compile_dsl: potentially dangerous pattern '{pattern}' detected in source")
        
        if not isinstance(use_hamiltonian, bool):
            error_msg = f"use_hamiltonian must be bool, got {type(use_hamiltonian).__name__}"
            logger.error(f"compile_dsl: {error_msg}")
            raise TypeError(error_msg)
        
        if not isinstance(use_constraints, bool):
            error_msg = f"use_constraints must be bool, got {type(use_constraints).__name__}"
            logger.error(f"compile_dsl: {error_msg}")
            raise TypeError(error_msg)
        
        start_time = time.time()
        logger.info(f"Starting DSL compilation (source length: {len(dsl_source)} chars)")
        
        # Performance monitoring
        if config.enable_performance_monitoring:
            _perf_monitor.snapshot_memory("pre_compilation")
            _perf_monitor.start_timer('compilation')
        
        try:
            # Tokenize with error handling
            try:
                tokens = tokenize(dsl_source)
                if not tokens:
                    raise ValueError("Tokenization produced no tokens")
                logger.info(f"Tokenized {len(tokens)} tokens")
            except Exception as e:
                logger.error(f"Tokenization failed: {e}", exc_info=True)
                raise ValueError(f"Tokenization failed: {e}") from e
            
            # Parse with error handling
            try:
                parser = MechanicsParser(tokens)
                self.ast = parser.parse()
            
                if parser.errors:
                    logger.warning(f"Parser found {len(parser.errors)} errors")
                    if len(parser.errors) >= config.max_parser_errors:
                        raise ValueError(f"Too many parser errors ({len(parser.errors)})")
            except Exception as e:
                logger.error(f"Parsing failed: {e}", exc_info=True)
                raise ValueError(f"Parsing failed: {e}") from e
            
            # Semantic analysis with error handling
            try:
                self.analyze_semantics()
                self.process_fluids()
            except Exception as e:
                logger.error(f"Semantic analysis failed: {e}", exc_info=True)
                raise ValueError(f"Semantic analysis failed: {e}") from e
            
            # Determine formulation
            if self.hamiltonian is not None:
                use_hamiltonian = True
            elif use_hamiltonian and self.lagrangian is not None:
                coords = self.get_coordinates()
                L_sympy = self.symbolic.ast_to_sympy(self.lagrangian)
                self.hamiltonian_expr = self.symbolic.lagrangian_to_hamiltonian(L_sympy, coords)
                use_hamiltonian = True
            
            # Derive equations with error handling
            try:
                
                if self.fluid_particles and self.lagrangian is None:
                    logger.info("Fluid system detected: Skipping symbolic derivation")
                    equations = {} # No symbolic equations needed for SPH
                    self.use_hamiltonian_formulation = False
                    
                elif use_hamiltonian:
                    equations = self.derive_hamiltonian_equations()
                    self.use_hamiltonian_formulation = True
                    logger.info("Using Hamiltonian formulation")
                # Only try this if have a Lagrangian
                elif self.lagrangian is not None:
                    # Check for constraints
                    if use_constraints and len(self.constraints) > 0:
                        equations = self.derive_constrained_equations()
                        logger.info(f"Using constrained Lagrangian with {len(self.constraints)} constraints")
                    else:
                        equations = self.derive_equations()
                        logger.info("Using standard Lagrangian formulation")
                    self.use_hamiltonian_formulation = False
            
                if equations is None:
                    raise ValueError("Equation derivation returned None")
                
                self.equations = equations
            except Exception as e:
                logger.error(f"Equation derivation failed: {e}", exc_info=True)
                raise ValueError(f"Equation derivation failed: {e}") from e
            
            # Setup simulation with error handling
            try:
                self.setup_simulation(equations)
            except Exception as e:
                logger.error(f"Simulation setup failed: {e}", exc_info=True)
                raise ValueError(f"Simulation setup failed: {e}") from e
            
            self.compilation_time = time.time() - start_time
            
            # Performance monitoring
            if config.enable_performance_monitoring:
                _perf_monitor.stop_timer('compilation')
                _perf_monitor.snapshot_memory("post_compilation")
            
            result = {
                'success': True,
                'system_name': self.system_name,
                'coordinates': list(self.get_coordinates()),
                'equations': equations,
                'variables': self.variables,
                'parameters': self.simulator.parameters,
                'compilation_time': self.compilation_time,
                'ast_nodes': len(self.ast),
                'formulation': 'Hamiltonian' if use_hamiltonian else 'Lagrangian',
                'num_constraints': len(self.constraints) if use_constraints else 0,
            }
            
            logger.info(f"Compilation successful in {self.compilation_time:.4f}s")
            
            # Add performance metrics if available
            if config.enable_performance_monitoring:
                comp_stats = _perf_monitor.get_stats('compilation')
                if comp_stats:
                    result['performance'] = comp_stats
            
            return result
            
        except Exception as e:
            error_trace = traceback.format_exc()
            logger.error(f"Compilation failed: {e}\n{error_trace}")
            return {
                'success': False,
                'error': str(e),
                'traceback': error_trace,
                'compilation_time': time.time() - start_time,
            }

    def analyze_semantics(self):
        """Extract system information from AST"""
        logger.info("Analyzing semantics")
        
        for node in self.ast:
            if isinstance(node, SystemDef):
                self.system_name = node.name
                logger.debug(f"System name: {self.system_name}")
                
            elif isinstance(node, VarDef):
                self.variables[node.name] = {
                    'type': node.vartype,
                    'unit': node.unit,
                    'vector': node.vector
                }
                logger.debug(f"Variable: {node.name} ({node.vartype})")
            
            elif isinstance(node, ParameterDef):
                self.parameters_def[node.name] = {
                    'value': node.value,
                    'unit': node.unit
                }
                logger.debug(f"Parameter: {node.name} = {node.value}")
                
            elif isinstance(node, DefineDef):
                self.definitions[node.name] = {
                    'args': node.args,
                    'body': node.body
                }
                logger.debug(f"Definition: {node.name}")
                
            elif isinstance(node, LagrangianDef):
                self.lagrangian = node.expr
                logger.debug("Lagrangian defined")
                
            elif isinstance(node, HamiltonianDef):
                self.hamiltonian = node.expr
                logger.debug("Hamiltonian defined")
                
            elif isinstance(node, TransformDef):
                self.transforms[node.var] = {
                    'type': node.coord_type,
                    'expression': node.expr
                }
                logger.debug(f"Transform: {node.var}")
            
            elif isinstance(node, ConstraintDef):
                self.constraints.append(node.expr)
                logger.debug("Holonomic constraint added")
            
            elif isinstance(node, NonHolonomicConstraintDef):
                self.non_holonomic_constraints.append(node.expr)
                logger.debug("Non-holonomic constraint added")
            
            elif isinstance(node, ForceDef):
                self.forces.append(node.expr)
                logger.debug(f"Force added: {node.force_type}")
            
            elif isinstance(node, DampingDef):
                self.damping_forces.append(node.expr)
                logger.debug("Damping force added")
                
            elif isinstance(node, InitialCondition):
                self.initial_conditions.update(node.conditions)
                logger.debug(f"Initial conditions: {node.conditions}")

    def get_coordinates(self) -> List[str]:
        """Extract generalized coordinates (exclude constants)"""
        coordinates = []
        
        for var_name, var_info in self.variables.items():
            # Exclude constants and parameters
            if var_info['type'] in ['Constant', 'Parameter', 'Mass', 'Spring Constant', 
                                   'Damping Coeff', 'Acceleration', 'Gravitational Constant',
                                   'Electric Field', 'Magnetic Field', 'Charge', 'Angular Velocity',
                                   'Drive Frequency', 'Force Amplitude', 'Coupling Constant',
                                   'Moment of Inertia 1', 'Moment of Inertia 3', 'Spin Rate',
                                   'Radius', 'Incline Angle', 'Damping Parameter', 'Natural Frequency',
                                   'Parameter']:
                continue
            # Include only dynamic variables
            if (var_info['type'] in ['Angle', 'Position', 'Coordinate'] or
                var_name in ['theta', 'theta1', 'theta2', 'theta3', 'x', 'x1', 'x2', 'y', 'z', 
                           'r', 'phi', 'psi']):
                coordinates.append(var_name)
        
        logger.debug(f"Coordinates: {coordinates}")
        return coordinates

    def process_fluids(self):
        """Generate particles for all fluid and boundary definitions"""
        
        # 1. Try to find smoothing length 'h' in parameters
        # This determines particle spacing
        for param_name, param_info in self.parameters_def.items():
            if param_name in ['h', 'spacing', 'dx']:
                self.smoothing_length = param_info['value']
                logger.info(f"Using particle spacing h={self.smoothing_length}")
                break
        
        for node in self.ast:
            if isinstance(node, FluidDef):
                logger.info(f"Generating fluid '{node.name}' in {node.region.shape}")
                
                # Use ParticleGenerator
                coords = ParticleGenerator.generate(node.region, self.smoothing_length)
                
                for x, y in coords:
                    self.fluid_particles.append({
                        'x': x,
                        'y': y,
                        'vx': 0.0,
                        'vy': 0.0,
                        'mass': node.mass,
                        'type': 'fluid'
                    })
                logger.info(f"Generated {len(coords)} fluid particles")

            elif isinstance(node, BoundaryDef):
                logger.info(f"Generating boundary '{node.name}'")
                
                # Boundaries are often denser (0.5 * h) to prevent leakage
                coords = ParticleGenerator.generate(node.region, self.smoothing_length)
                
                for x, y in coords:
                    self.boundary_particles.append({
                        'x': x,
                        'y': y,
                        'vx': 0.0,
                        'vy': 0.0,
                        'mass': 1000.0, # Infinite mass essentially
                        'type': 'boundary'
                    })
                    
    def derive_equations(self) -> Dict[str, sp.Expr]:
        """Derive equations using Lagrangian formulation (Patched for Forces)"""
        if self.lagrangian is None:
            raise ValueError("No Lagrangian defined")
        
        L_sympy = self.symbolic.ast_to_sympy(self.lagrangian)
        coordinates = self.get_coordinates()
        
        if not coordinates:
            raise ValueError("No generalized coordinates found")
        
        # 1. Derive Standard LHS: d/dt(dL/dq_dot) - dL/dq
        eq_list = self.symbolic.derive_equations_of_motion(L_sympy, coordinates)
        
        # 2. Apply Non-Conservative Forces (LHS - Q = 0)
        # Note: Don't expand yet - keep derivative structure for acceleration extraction
        if self.forces:
            logger.info(f"Applying {len(self.forces)} non-conservative forces")
            for i, force_ast in enumerate(self.forces):
                if i < len(eq_list):
                    F_sym = self.symbolic.ast_to_sympy(force_ast)
                    # Subtract Force but don't expand yet - preserve derivative structure
                    eq_list[i] = eq_list[i] - F_sym
        
        # 3. Solve for accelerations (this will handle derivative replacement)
        accelerations = self.symbolic.solve_for_accelerations(eq_list, coordinates)
        
        return accelerations

    def derive_constrained_equations(self) -> Dict[str, sp.Expr]:
        """Derive equations with constraints using Lagrange multipliers"""
        
        if self.lagrangian is None:
            raise ValueError("No Lagrangian defined")
        
        if not self.constraints:
            logger.warning("No constraints defined, using standard formulation")
            return self.derive_equations()
        
        L_sympy = self.symbolic.ast_to_sympy(self.lagrangian)
        coordinates = self.get_coordinates()
        
        if not coordinates:
            raise ValueError("No generalized coordinates found")
        
        # Convert constraint expressions to SymPy
        constraint_exprs = [self.symbolic.ast_to_sympy(c) for c in self.constraints]
        
        # Derive constrained equations
        eq_list, extended_coords = self.symbolic.derive_equations_with_constraints(
            L_sympy, coordinates, constraint_exprs
        )
        
        # Solve for accelerations and lambda multipliers
        accelerations = self.symbolic.solve_for_accelerations(eq_list, extended_coords)
        
        # Filter out only the coordinate accelerations (not lambda derivatives)
        coord_accelerations = {
            k: v for k, v in accelerations.items() 
            if any(k.startswith(f"{c}_ddot") for c in coordinates)
        }
        
        return coord_accelerations

    def derive_hamiltonian_equations(self) -> Tuple[List[sp.Expr], List[sp.Expr]]:
        """Derive equations using Hamiltonian formulation"""
        
        if self.hamiltonian is not None:
            H_sympy = self.symbolic.ast_to_sympy(self.hamiltonian)
        elif hasattr(self, 'hamiltonian_expr'):
            H_sympy = self.hamiltonian_expr
        else:
            raise ValueError("No Hamiltonian defined or derived")
        
        coordinates = self.get_coordinates()
        
        if not coordinates:
            raise ValueError("No generalized coordinates found")
        
        q_dots, p_dots = self.symbolic.derive_hamiltonian_equations(H_sympy, coordinates)
        
        return (q_dots, p_dots)

    def setup_simulation(self, equations):
        """Configure numerical simulator"""
        
        logger.info("Setting up simulation")
        
        # Collect parameters
        parameters = {}
        for param_name, param_info in self.parameters_def.items():
            parameters[param_name] = param_info['value']
        
        # Add default parameters
        for var_name, var_info in self.variables.items():
            if var_info['type'] in ['Real', 'Mass', 'Length', 'Acceleration', 'Spring Constant']:
                if var_name not in parameters:
                    defaults = {
                        'g': 9.81,
                        'm': 1.0, 'm1': 1.0, 'm2': 1.0,
                        'l': 1.0, 'l1': 1.0, 'l2': 1.0,
                        'k': 1.0,
                    }
                    parameters[var_name] = defaults.get(var_name, 1.0)
        
        self.simulator.set_parameters(parameters)
        self.simulator.set_initial_conditions(self.initial_conditions)
        
        coordinates = self.get_coordinates()
        
        if self.use_hamiltonian_formulation:
            q_dots, p_dots = equations
            self.simulator.compile_hamiltonian_equations(q_dots, p_dots, coordinates)
        else:
            self.simulator.compile_equations(equations, coordinates)

    def simulate(self, t_span: Tuple[float, float] = (0, 10), 
                num_points: int = 1000, **kwargs) -> dict:
        """Run numerical simulation"""
        return self.simulator.simulate(t_span, num_points, **kwargs)

    def animate(self, solution: dict, show: bool = True):
        """Create animation from solution"""
        parameters = self.simulator.parameters
        anim = self.visualizer.animate(solution, parameters, self.system_name)
        
        if show and anim is not None:
            plt.show()
        
        return anim

    def export_animation(self, solution: dict, filename: str, 
                        fps: int = 30, dpi: int = 100) -> str:
        """Export animation to file"""
        anim = self.animate(solution, show=False)
        
        if anim is None:
            raise RuntimeError('No animation available')
        
        ok = self.visualizer.save_animation_to_file(anim, filename, fps, dpi)
        
        if not ok:
            raise RuntimeError(f'Failed to save animation to {filename}')
        
        return filename

    def plot_energy(self, solution: dict):
        """Plot energy analysis"""
        self.visualizer.plot_energy(solution, self.simulator.parameters, 
                                   self.system_name, self.lagrangian)

    def plot_phase_space(self, solution: dict, coordinate_index: int = 0):
        """Plot phase space"""
        self.visualizer.plot_phase_space(solution, coordinate_index)
    
    def print_equations(self):
        """Print derived equations"""
        if self.equations is None:
            print("No equations derived yet.")
            return
        
        print(f"\n{'='*70}")
        print(f"Equations of Motion: {self.system_name}")
        print(f"Formulation: {'Hamiltonian' if self.use_hamiltonian_formulation else 'Lagrangian'}")
        print(f"{'='*70}\n")
        
        if self.use_hamiltonian_formulation:
            q_dots, p_dots = self.equations
            coords = self.get_coordinates()
            for i, q in enumerate(coords):
                print(f"d{q}/dt = {q_dots[i]}")
                print(f"dp_{q}/dt = {p_dots[i]}\n")
        else:
            for coord in self.get_coordinates():
                accel_key = f"{coord}_ddot"
                if accel_key in self.equations:
                    eq = self.equations[accel_key]
                    print(f"{accel_key} = {eq}\n")
        
        print(f"{'='*70}\n")
    
    def get_info(self) -> dict:
        """Get comprehensive system information"""
        return {
            'system_name': self.system_name,
            'coordinates': self.get_coordinates(),
            'variables': self.variables,
            'parameters': self.simulator.parameters,
            'initial_conditions': self.initial_conditions,
            'has_lagrangian': self.lagrangian is not None,
            'has_hamiltonian': self.hamiltonian is not None,
            'num_constraints': len(self.constraints),
            'compilation_time': self.compilation_time,
            'formulation': 'Hamiltonian' if self.use_hamiltonian_formulation else 'Lagrangian',
        }
    
    def export_system(self, filename: str, format: str = 'json') -> bool:
        """Export system state to file"""
        return SystemSerializer.export_system(self, filename, format)
    
    @staticmethod
    def import_system(filename: str) -> Optional['PhysicsCompiler']:
        """Import system state from file"""
        state = SystemSerializer.import_system(filename)
        if state is None:
            return None
        
        # Note: This creates a new compiler but doesn't fully reconstruct the equations
        # For full reconstruction, you'd need to re-compile the DSL source
        compiler = PhysicsCompiler()
        compiler.system_name = state.get('system_name', 'imported_system')
        compiler.variables = state.get('variables', {})
        compiler.parameters_def = state.get('parameters', {})
        compiler.initial_conditions = state.get('initial_conditions', {})
        
        logger.info(f"Imported system: {compiler.system_name}")
        logger.warning("Note: Equations not reconstructed. Re-compile DSL source for full functionality.")
        
        return compiler

    def compile_to_cpp(self, filename: str = "simulation.cpp", 
                      target: str = "standard", 
                      compile_binary: bool = True) -> bool:
        """
        Generate C++ code for multiple targets.
        
        Args:
            filename: Output filename
            target: 'standard', 'raylib', 'arduino', 'wasm', 'openmp', 'python'
            compile_binary: Whether to run the compiler (g++, emcc, etc.)
        """
        if self.equations is None:
            logger.error("No equations derived. Compile DSL first.")
            return False
            
        try:
            generator = CppGenerator(
                system_name=self.system_name,
                coordinates=self.get_coordinates(),
                parameters=self.simulator.parameters,
                initial_conditions=self.initial_conditions,
                equations=self.equations,
                fluid_particles=self.fluid_particles,
                boundary_particles=self.boundary_particles
            )
            
            # For Arduino, ensure extension is .ino
            if target == 'arduino' and not filename.endswith('.ino'):
                filename = os.path.splitext(filename)[0] + ".ino"
                
            source_file = generator.generate(filename)
            
            if compile_binary and target != 'arduino': # Arduino compilation typically requires IDE
                binary_name = os.path.splitext(source_file)[0]
                if os.name == 'nt' and target != 'wasm':
                    binary_name += ".exe"
                elif target == 'wasm':
                    binary_name += ".js" # Emscripten outputs JS+Wasm
                elif target == 'python':
                    # Python extensions need specific suffixes like .cpython-38-x86_64-linux-gnu.so
                    # We let setup.py handle this usually, but here is a simple attempt:
                    binary_name += subprocess.check_output(
                        [sys.executable, "-c", "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))"]
                    ).decode().strip()

                cmd = []
                
                if target == 'standard':
                    cmd = ["g++", "-O3", source_file, "-o", binary_name]
                    
                elif target == 'raylib':
                    # Assumes raylib is installed in standard paths
                    cmd = ["g++", "-O3", source_file, "-o", binary_name, "-lraylib", "-lm", "-lpthread", "-ldl", "-lrt", "-lX11"]
                    
                elif target == 'openmp':
                    cmd = ["g++", "-O3", "-fopenmp", source_file, "-o", binary_name]
                    
                elif target == 'wasm':
                    # Needs emcc in PATH
                    cmd = ["emcc", source_file, "-o", binary_name, 
                           "-s", "EXPORTED_FUNCTIONS=['_init','_step','_get_state','_get_time']",
                           "-s", "EXPORTED_RUNTIME_METHODS=['ccall','cwrap']",
                           "-O3"]
                           
                elif target == 'python':
                    # Needs pybind11 headers
                    includes = subprocess.check_output([sys.executable, "-m", "pybind11", "--includes"]).decode().strip().split()
                    cmd = ["g++", "-O3", "-shared", "-fPIC"] + includes + [source_file, "-o", binary_name]

                logger.info(f"Compiling: {' '.join(cmd)}")
                subprocess.check_call(cmd)
                logger.info(f"Successfully compiled: {binary_name}")
                
            elif target == 'arduino':
                logger.info(f"Arduino sketch generated: {source_file}. Open in Arduino IDE to compile.")
                
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Compilation failed: {e}")
            return False
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return False
