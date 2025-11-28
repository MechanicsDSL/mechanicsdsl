import time
import json
import sympy as sp
import pickle
from typing import Dict, List, Tuple, Optional
from .utils import logger, config, _perf_monitor, profile_function, validate_file_path
from .parser import (
    tokenize, MechanicsParser, SystemDef, VarDef, ParameterDef, 
    LagrangianDef, HamiltonianDef, ConstraintDef, ForceDef, 
    InitialCondition, SolveDef
)
from .symbolic import SymbolicEngine
from .solver import NumericalSimulator
from .visualization import MechanicsVisualizer

class SystemSerializer:
    @staticmethod
    def export_system(compiler, filename, format='json'):
        data = {
            'system': compiler.system_name,
            'coordinates': compiler.coordinates,
            'parameters': compiler.parameters_def,
            'initials': compiler.initial_conditions
        }
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        return True

    @staticmethod
    def import_system(filename):
        with open(filename, 'r') as f:
            return json.load(f)

class PhysicsCompiler:
    def __init__(self):
        self.symbolic = SymbolicEngine()
        self.simulator = NumericalSimulator(self.symbolic)
        self.visualizer = MechanicsVisualizer()
        
        self.system_name = "System"
        self.coordinates = []
        self.variables = {}
        self.parameters_def = {}
        self.initial_conditions = {}
        self.lagrangian = None
        self.hamiltonian = None
        self.constraints = []
        self.forces = []
        self.use_hamiltonian = False

    def compile_dsl(self, source: str, use_hamiltonian: bool = False):
        logger.info("Starting compilation...")
        start_time = time.time()
        
        # 1. Tokenize & Parse
        tokens = tokenize(source)
        parser = MechanicsParser(tokens)
        nodes = parser.parse()
        
        # 2. Extract System Info
        for node in nodes:
            if isinstance(node, SystemDef): self.system_name = node.name
            elif isinstance(node, VarDef):
                if node.vartype.lower() in ['angle', 'position', 'length', 'x', 'y', 'z']:
                    self.coordinates.append(node.name)
            elif isinstance(node, ParameterDef):
                self.symbolic.get_symbol(node.name)
                self.parameters_def[node.name] = node.value
            elif isinstance(node, LagrangianDef): self.lagrangian = node.expr
            elif isinstance(node, HamiltonianDef): self.hamiltonian = node.expr
            elif isinstance(node, InitialCondition): self.initial_conditions.update(node.conditions)
            elif isinstance(node, ConstraintDef): self.constraints.append(node.expr)
            elif isinstance(node, ForceDef): self.forces.append(node.expr)

        # 3. Select Formulation
        if use_hamiltonian or self.hamiltonian:
            self.use_hamiltonian = True
            if not self.hamiltonian and self.lagrangian:
                L_sym = self.symbolic.ast_to_sympy(self.lagrangian)
                self.hamiltonian_expr = self.symbolic.lagrangian_to_hamiltonian(L_sym, self.coordinates)
        
        # 4. Derive Equations
        try:
            if self.use_hamiltonian:
                H_sym = self.symbolic.ast_to_sympy(self.hamiltonian) if self.hamiltonian else self.hamiltonian_expr
                q_dots, p_dots = self.symbolic.derive_hamiltonian_equations(H_sym, self.coordinates)
                eqs = (q_dots, p_dots)
            else:
                L_sym = self.symbolic.ast_to_sympy(self.lagrangian)
                
                if self.constraints:
                    # Constrained Dynamics
                    c_exprs = [self.symbolic.ast_to_sympy(c) for c in self.constraints]
                    eq_list, ext_coords = self.symbolic.derive_equations_with_constraints(L_sym, self.coordinates, c_exprs)
                    accels = self.symbolic.solve_for_accelerations(eq_list, ext_coords)
                    # Filter lambdas
                    eqs = {k:v for k,v in accels.items() if not k.startswith('lambda')}
                else:
                    # Standard
                    eq_list = self.symbolic.derive_equations_of_motion(L_sym, self.coordinates)
                    if self.forces:
                        for i, f in enumerate(self.forces):
                            if i < len(eq_list):
                                eq_list[i] -= self.symbolic.ast_to_sympy(f)
                    
                    eqs = self.symbolic.solve_for_accelerations(eq_list, self.coordinates)

            # 5. Setup Simulator
            self.simulator.set_parameters(self.parameters_def)
            self.simulator.set_initial_conditions(self.initial_conditions)
            
            if self.use_hamiltonian:
                self.simulator.compile_hamiltonian_equations(eqs[0], eqs[1], self.coordinates)
            else:
                self.simulator.compile_equations(eqs, self.coordinates)
                
            return {
                'success': True, 
                'system': self.system_name, 
                'time': time.time() - start_time
            }
            
        except Exception as e:
            logger.error(f"Compilation Error: {e}")
            return {'success': False, 'error': str(e)}

    def simulate(self, t_span=(0, 10), **kwargs):
        return self.simulator.simulate(t_span, **kwargs)

    def animate(self, solution, show=True):
        return self.visualizer.animate(solution, self.simulator.parameters, self.system_name)
        
    def plot_energy(self, solution):
        self.visualizer.plot_energy(solution, self.simulator.parameters, self.system_name)

    def plot_phase_space(self, solution, idx=0):
        self.visualizer.plot_phase_space(solution, idx)
        
    def export_system(self, filename):
        return SystemSerializer.export_system(self, filename)
