"""
IPython magic commands for MechanicsDSL.

Provides %%mechanicsdsl cell magic and line magics for interactive use.
"""
from typing import Optional, Dict, Any
import warnings

try:
    from IPython.core.magic import Magics, magics_class, cell_magic, line_magic
    from IPython.core.magic_arguments import argument, magic_arguments, parse_argstring
    from IPython import get_ipython
    IPYTHON_AVAILABLE = True
except ImportError:
    IPYTHON_AVAILABLE = False
    Magics = object
    magics_class = lambda x: x
    cell_magic = lambda x: lambda y: y
    line_magic = lambda x: lambda y: y
    magic_arguments = lambda: lambda x: x
    argument = lambda *args, **kwargs: lambda x: x
    parse_argstring = lambda *args, **kwargs: type('Args', (), {})()

try:
    from mechanics_dsl import PhysicsCompiler
except ImportError:
    PhysicsCompiler = None


# Global state for the magic commands
_state: Dict[str, Any] = {
    'compiler': None,
    'last_result': None,
    'last_solution': None,
}


@magics_class
class MechanicsDSLMagics(Magics):
    """
    IPython magic commands for MechanicsDSL.
    
    Usage:
        # Load the extension
        %load_ext mechanics_dsl.jupyter
        
        # Compile and simulate
        %%mechanicsdsl --animate --t_span=0,20
        \\system{pendulum}
        \\defvar{theta}{Angle}{rad}
        ...
        
        # Change parameters
        %mdsl_params m=2.0 k=15.0
        
        # Access results
        %mdsl_solution  # Returns last solution dict
        
        # Export code
        %mdsl_export cpp pendulum.cpp
    """
    
    def __init__(self, shell):
        super().__init__(shell)
        self._ensure_compiler()
    
    def _ensure_compiler(self):
        """Ensure compiler is initialized."""
        if _state['compiler'] is None and PhysicsCompiler is not None:
            _state['compiler'] = PhysicsCompiler()
    
    @cell_magic
    @magic_arguments()
    @argument('--animate', '-a', action='store_true', help='Auto-display animation')
    @argument('--phase', '-p', action='store_true', help='Show phase portrait')
    @argument('--energy', '-e', action='store_true', help='Show energy plot')
    @argument('--export', type=str, default=None, help='Export to language (cpp, rust, etc.)')
    @argument('--t_span', type=str, default='0,10', help='Time span as start,end')
    @argument('--num_points', type=int, default=1000, help='Number of output points')
    @argument('--quiet', '-q', action='store_true', help='Suppress output')
    def mechanicsdsl(self, line, cell):
        """
        Compile and simulate MechanicsDSL code.
        
        Examples:
            %%mechanicsdsl --animate
            \\system{pendulum}
            ...
            
            %%mechanicsdsl --export=cpp --t_span=0,100
            \\system{double_pendulum}
            ...
        """
        args = parse_argstring(self.mechanicsdsl, line)
        
        self._ensure_compiler()
        compiler = _state['compiler']
        
        if compiler is None:
            print("Error: PhysicsCompiler not available")
            return
        
        # Parse time span
        try:
            t_start, t_end = map(float, args.t_span.split(','))
            t_span = (t_start, t_end)
        except:
            t_span = (0, 10)
        
        # Compile
        result = compiler.compile_dsl(cell)
        _state['last_result'] = result
        
        if not result['success']:
            print(f"Compilation error: {result.get('error', 'Unknown error')}")
            return result
        
        if not args.quiet:
            print(f"Compiled: {result.get('system_name', 'system')}")
            print(f"  Coordinates: {result.get('coordinates', [])}")
            print(f"  Parameters: {list(compiler.simulator.parameters.keys())}")
        
        # Simulate
        solution = compiler.simulate(t_span=t_span, num_points=args.num_points)
        _state['last_solution'] = solution
        
        if not solution['success']:
            print(f"Simulation error: {solution.get('error', 'Unknown')}")
            return solution
        
        if not args.quiet:
            print(f"  Simulation: {solution.get('nfev', 0)} evaluations")
        
        # Display visualizations
        if args.animate:
            from .display import display_simulation
            display_simulation(solution, compiler.simulator.coordinates)
        
        if args.phase:
            from .display import display_phase_portrait
            display_phase_portrait(solution)
        
        if args.energy:
            from .display import display_energy_plot
            display_energy_plot(solution)
        
        # Export
        if args.export:
            filename = f"{result.get('system_name', 'output')}.{args.export}"
            compiler.export(args.export, filename)
            print(f"  Exported: {filename}")
        
        return solution
    
    @line_magic
    def mdsl_params(self, line):
        """
        Set simulation parameters.
        
        Usage:
            %mdsl_params m=2.0 k=15.0 damping=0.1
        """
        self._ensure_compiler()
        compiler = _state['compiler']
        
        if compiler is None:
            print("Error: No compiler available")
            return
        
        # Parse key=value pairs
        params = {}
        for pair in line.split():
            if '=' in pair:
                key, val = pair.split('=', 1)
                try:
                    params[key.strip()] = float(val)
                except ValueError:
                    print(f"Warning: Could not parse {pair}")
        
        if params:
            compiler.simulator.set_parameters(params)
            print(f"Updated parameters: {params}")
    
    @line_magic
    def mdsl_solution(self, line):
        """
        Return the last simulation solution.
        
        Usage:
            sol = %mdsl_solution
            t = sol['t']
            y = sol['y']
        """
        return _state['last_solution']
    
    @line_magic
    def mdsl_result(self, line):
        """Return the last compilation result."""
        return _state['last_result']
    
    @line_magic
    def mdsl_export(self, line):
        """
        Export to a target language.
        
        Usage:
            %mdsl_export cpp output.cpp
            %mdsl_export rust src/physics.rs
        """
        parts = line.split()
        if len(parts) < 2:
            print("Usage: %mdsl_export <language> <filename>")
            return
        
        language, filename = parts[0], parts[1]
        
        self._ensure_compiler()
        compiler = _state['compiler']
        
        if compiler is None:
            print("Error: No compiler available")
            return
        
        try:
            compiler.export(language, filename)
            print(f"Exported {language} code to {filename}")
        except Exception as e:
            print(f"Export failed: {e}")
    
    @line_magic
    def mdsl_animate(self, line):
        """
        Animate the last simulation.
        
        Usage:
            %mdsl_animate
            %mdsl_animate fps=30 trail=50
        """
        solution = _state['last_solution']
        if solution is None:
            print("No simulation to animate. Run %%mechanicsdsl first.")
            return
        
        # Parse options
        fps = 30
        for pair in line.split():
            if pair.startswith('fps='):
                fps = int(pair.split('=')[1])
        
        from .display import display_simulation
        compiler = _state['compiler']
        coords = compiler.simulator.coordinates if compiler else []
        display_simulation(solution, coords, fps=fps)
    
    @line_magic
    def mdsl_info(self, line):
        """Show current system info."""
        compiler = _state['compiler']
        if compiler is None:
            print("No system compiled")
            return
        
        result = _state['last_result']
        if result:
            print(f"System: {result.get('system_name', 'Unknown')}")
            print(f"Coordinates: {result.get('coordinates', [])}")
            print(f"Parameters: {compiler.simulator.parameters}")
            print(f"Initial conditions: {compiler.simulator.initial_conditions}")


def load_ipython_extension(ipython):
    """
    Load the MechanicsDSL IPython extension.
    
    Usage in notebook:
        %load_ext mechanics_dsl.jupyter
    """
    if not IPYTHON_AVAILABLE:
        warnings.warn("IPython not available")
        return
    
    ipython.register_magics(MechanicsDSLMagics)
    print("MechanicsDSL magic commands loaded.")
    print("  %%mechanicsdsl - Compile and simulate DSL code")
    print("  %mdsl_params   - Set parameters")
    print("  %mdsl_animate  - Animate last simulation")
    print("  %mdsl_export   - Export to target language")


def unload_ipython_extension(ipython):
    """Unload the extension."""
    _state['compiler'] = None
    _state['last_result'] = None
    _state['last_solution'] = None


__all__ = [
    'MechanicsDSLMagics',
    'load_ipython_extension',
    'unload_ipython_extension',
]
