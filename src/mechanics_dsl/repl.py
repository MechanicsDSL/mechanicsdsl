"""
MechanicsDSL Interactive REPL

An interactive shell for experimenting with the DSL.
"""

import sys
import atexit
from pathlib import Path
from typing import Optional

# readline is optional (not available on Windows)
try:
    import readline
    HAS_READLINE = True
except ImportError:
    readline = None
    HAS_READLINE = False

try:
    from .compiler import PhysicsCompiler
    from .presets import get_preset, list_presets
except ImportError:
    PhysicsCompiler = None
    get_preset = None
    list_presets = None


REPL_BANNER = """
╔══════════════════════════════════════════════════════════════╗
║           MechanicsDSL Interactive REPL v2.0.6               ║
╠══════════════════════════════════════════════════════════════╣
║  Commands:                                                   ║
║    :help          Show this help                             ║
║    :load <file>   Load DSL from file                         ║
║    :preset <name> Load a preset (pendulum, orbit, etc.)      ║
║    :list          List available presets                     ║
║    :compile       Compile current buffer                     ║
║    :run [t]       Run simulation (default t=0..10)           ║
║    :plot [var]    Plot results                               ║
║    :export <fmt>  Export to json/csv/numpy                   ║
║    :clear         Clear buffer                               ║
║    :show          Show current buffer                        ║
║    :quit          Exit REPL                                  ║
║                                                              ║
║  Or type DSL directly (multi-line, empty line to finish)     ║
╚══════════════════════════════════════════════════════════════╝
"""


class REPL:
    """
    Interactive REPL for MechanicsDSL.
    
    Provides an interactive shell for experimenting with physics DSL,
    compiling systems, running simulations, and visualizing results.
    
    Attributes:
        buffer: List of DSL lines accumulated for compilation
        compiler: Currently active PhysicsCompiler instance
        solution: Latest simulation results
        history_file: Path to command history file
    """
    
    buffer: list[str]
    compiler: Optional['PhysicsCompiler']
    solution: Optional[dict]
    history_file: Path
    
    def __init__(self) -> None:
        self.buffer: list[str] = []
        self.compiler: Optional['PhysicsCompiler'] = None
        self.solution: Optional[dict] = None
        self.history_file: Path = Path.home() / '.mechanicsdsl_history'
        self._setup_readline()
    
    def _setup_readline(self) -> None:
        """Setup readline for history and completion."""
        if not HAS_READLINE:
            return
        try:
            if self.history_file.exists():
                readline.read_history_file(str(self.history_file))
            readline.set_history_length(1000)
            atexit.register(self._save_history)
        except Exception:
            pass  # Readline not available on all platforms
    
    def _save_history(self) -> None:
        """Save command history."""
        if not HAS_READLINE:
            return
        try:
            readline.write_history_file(str(self.history_file))
        except Exception:
            pass
    
    def run(self) -> None:
        """Main REPL loop."""
        print(REPL_BANNER)
        
        while True:
            try:
                line = input("mdsl> ").strip()
                
                if not line:
                    continue
                
                if line.startswith(':'):
                    if not self._handle_command(line[1:]):
                        break
                else:
                    self._add_to_buffer(line)
                    
            except KeyboardInterrupt:
                print("\n(Use :quit to exit)")
            except EOFError:
                print("\nGoodbye!")
                break
    
    def _handle_command(self, cmd: str) -> bool:
        """Handle a REPL command. Returns False to exit."""
        parts = cmd.split(maxsplit=1)
        command = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else None
        
        if command in ('quit', 'exit', 'q'):
            print("Goodbye!")
            return False
        
        elif command == 'help':
            print(REPL_BANNER)
        
        elif command == 'clear':
            self.buffer = []
            self.compiler = None
            self.solution = None
            print("Buffer cleared.")
        
        elif command == 'show':
            if self.buffer:
                print("Current buffer:")
                print("-" * 40)
                print('\n'.join(self.buffer))
                print("-" * 40)
            else:
                print("Buffer is empty.")
        
        elif command == 'list':
            if list_presets:
                presets = list_presets()
                print("Available presets:")
                for p in presets:
                    print(f"  - {p}")
            else:
                print("Presets not available.")
        
        elif command == 'preset':
            if not arg:
                print("Usage: :preset <name>")
            elif get_preset:
                try:
                    dsl = get_preset(arg)
                    self.buffer = dsl.strip().split('\n')
                    print(f"Loaded preset: {arg}")
                    self._compile()
                except KeyError as e:
                    print(str(e))
            else:
                print("Presets not available.")
        
        elif command == 'load':
            if not arg:
                print("Usage: :load <filename>")
            else:
                self._load_file(arg)
        
        elif command == 'compile':
            self._compile()
        
        elif command == 'run':
            t_end = float(arg) if arg else 10.0
            self._run_simulation(t_end)
        
        elif command == 'plot':
            self._plot(arg)
        
        elif command == 'export':
            if not arg:
                print("Usage: :export <json|csv|numpy>")
            else:
                self._export(arg)
        
        else:
            print(f"Unknown command: {command}")
            print("Type :help for available commands.")
        
        return True
    
    def _add_to_buffer(self, line: str):
        """Add a line to the DSL buffer."""
        self.buffer.append(line)
        print(f"  [{len(self.buffer)}] {line}")
    
    def _load_file(self, filename: str):
        """Load DSL from file."""
        path = Path(filename)
        if not path.exists():
            print(f"File not found: {filename}")
            return
        
        content = path.read_text()
        self.buffer = content.strip().split('\n')
        print(f"Loaded {len(self.buffer)} lines from {filename}")
        self._compile()
    
    def _compile(self):
        """Compile the current buffer."""
        if not self.buffer:
            print("Buffer is empty. Add DSL code first.")
            return
        
        if PhysicsCompiler is None:
            print("Compiler not available.")
            return
        
        dsl_source = '\n'.join(self.buffer)
        
        try:
            self.compiler = PhysicsCompiler()
            result = self.compiler.compile_dsl(dsl_source)
            
            if result.get('success'):
                print(f"✓ Compiled successfully!")
                print(f"  System: {result.get('system_name', 'unknown')}")
                coords = result.get('coordinates', [])
                if coords:
                    print(f"  Coordinates: {', '.join(coords)}")
            else:
                print(f"✗ Compilation failed: {result.get('error', 'Unknown')}")
                self.compiler = None
        except Exception as e:
            print(f"✗ Error: {e}")
            self.compiler = None
    
    def _run_simulation(self, t_end: float):
        """Run simulation."""
        if self.compiler is None:
            print("No compiled system. Use :compile first.")
            return
        
        try:
            print(f"Running simulation t=[0, {t_end}]...")
            self.solution = self.compiler.simulate(t_span=(0, t_end), num_points=1000)
            print(f"✓ Simulation complete. {len(self.solution['t'])} points.")
        except Exception as e:
            print(f"✗ Simulation failed: {e}")
    
    def _plot(self, var: Optional[str] = None):
        """Plot simulation results."""
        if self.solution is None:
            print("No simulation results. Use :run first.")
            return
        
        try:
            import matplotlib.pyplot as plt
            
            t = self.solution['t']
            
            if var:
                # Plot specific variable
                coords = getattr(self.compiler.simulator, 'coordinates', [])
                if var in coords:
                    idx = coords.index(var)
                    plt.figure()
                    plt.plot(t, self.solution['y'][idx * 2])
                    plt.xlabel('Time (s)')
                    plt.ylabel(var)
                    plt.title(f'{var} vs Time')
                    plt.grid(True)
                    plt.show()
                else:
                    print(f"Variable '{var}' not found.")
            else:
                # Plot all
                plt.figure(figsize=(10, 6))
                for i, y in enumerate(self.solution['y']):
                    plt.plot(t, y, label=f'y[{i}]')
                plt.xlabel('Time (s)')
                plt.ylabel('State')
                plt.title('Simulation Results')
                plt.legend()
                plt.grid(True)
                plt.show()
                
        except ImportError:
            print("matplotlib not available.")
        except Exception as e:
            print(f"Plot failed: {e}")
    
    def _export(self, format_type: str):
        """Export simulation results."""
        if self.solution is None:
            print("No simulation results. Use :run first.")
            return
        
        import json
        import numpy as np
        
        fmt = format_type.lower()
        
        if fmt == 'json':
            data = {
                't': self.solution['t'].tolist(),
                'y': [y.tolist() for y in self.solution['y']]
            }
            filename = 'results.json'
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Exported to {filename}")
            
        elif fmt == 'csv':
            filename = 'results.csv'
            with open(filename, 'w') as f:
                f.write('t,' + ','.join(f'y{i}' for i in range(len(self.solution['y']))) + '\n')
                for i, t in enumerate(self.solution['t']):
                    row = [str(t)] + [str(self.solution['y'][j][i]) for j in range(len(self.solution['y']))]
                    f.write(','.join(row) + '\n')
            print(f"Exported to {filename}")
            
        elif fmt in ('numpy', 'npz'):
            filename = 'results.npz'
            np.savez(filename, t=self.solution['t'], y=np.array(self.solution['y']))
            print(f"Exported to {filename}")
            
        else:
            print(f"Unknown format: {format_type}")
            print("Available: json, csv, numpy")


def run_repl():
    """Entry point for REPL."""
    repl = REPL()
    repl.run()


if __name__ == '__main__':
    run_repl()
