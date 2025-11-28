import time
from .utils import logger, config
from .parser import tokenize, MechanicsParser, SystemDef, VarDef, ParameterDef, LagrangianDef, InitialCondition, SolveDef, AnimateDef
from .symbolic import SymbolicEngine
from .solver import NumericalSimulator
from .visualization import MechanicsVisualizer

class PhysicsCompiler:
    def __init__(self):
        self.symbolic = SymbolicEngine()
        self.simulator = NumericalSimulator(self.symbolic)
        self.visualizer = MechanicsVisualizer()
        self.system_name = "System"
        self.coordinates = []

    def compile_dsl(self, source: str):
        logger.info("Compiling DSL...")
        start = time.time()
        
        # 1. Parse
        tokens = tokenize(source)
        parser = MechanicsParser(tokens)
        ast_nodes = parser.parse()
        
        # 2. Extract Data
        params = {}
        initials = {}
        lagrangian = None
        solve_method = "RK45"
        
        for node in ast_nodes:
            if isinstance(node, SystemDef): self.system_name = node.name
            if isinstance(node, VarDef): 
                if node.vartype in ["Angle", "Position", "X", "Y"]: 
                    self.coordinates.append(node.name)
            if isinstance(node, ParameterDef): 
                self.symbolic.get_symbol(node.name) # Register param symbol
                params[self.symbolic.get_symbol(node.name)] = node.value # Use symbol key for substitution
            if isinstance(node, LagrangianDef): lagrangian = node.expr
            if isinstance(node, InitialCondition): initials.update(node.conditions)
            if isinstance(node, SolveDef): solve_method = node.method

        # 3. Symbolic Derivation
        if not lagrangian: return {'success': False, 'error': 'No Lagrangian'}
        
        L_sym = self.symbolic.ast_to_sympy(lagrangian)
        eqs = self.symbolic.derive_equations_of_motion(L_sym, self.coordinates)
        accels = self.symbolic.solve_for_accelerations(eqs, self.coordinates)
        
        # 4. Prepare Simulator
        # Convert param keys back to strings for simulator
        str_params = {str(k): v for k, v in params.items()}
        self.simulator.set_parameters(str_params)
        self.simulator.set_initial_conditions(initials)
        self.simulator.compile_equations(accels, self.coordinates)
        
        return {
            'success': True,
            'time': time.time() - start,
            'coordinates': self.coordinates,
            'method': solve_method
        }

    def simulate(self, t_span, **kwargs):
        return self.simulator.simulate(t_span, **kwargs)

    def animate(self, solution):
        return self.visualizer.animate(solution, self.simulator.parameters, self.system_name)
