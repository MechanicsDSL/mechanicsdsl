"""
Tutorial 23: The Dam Break (SPH Fluid Dynamics)
"""
import os
import subprocess
import matplotlib.pyplot as plt
from mechanics_dsl import PhysicsCompiler

# 1. Define the System using new Syntax
# ------------------------------------------------
dsl_code = r"""
\system{dam_break}

% Simulation Constants
\parameter{h}{0.04}{m}       % Smoothing length (Resolution)
\parameter{g}{9.81}{m/s^2}

% The Water Column (0.4m wide x 0.8m high)
\fluid{water}
\region{rectangle}{x=0.0 .. 0.4, y=0.0 .. 0.8}
\particle_mass{0.02}         % Adjusted for resolution
\equation_of_state{tait}

% The Container (Bucket)
\boundary{walls}
\region{line}{x=-0.05, y=0.0 .. 1.5}   % Left Wall
\region{line}{x=1.5, y=0.0 .. 1.5}     % Right Wall
\region{line}{x=-0.05 .. 1.5, y=-0.05} % Floor
"""

print("1. Initializing Compiler...")
compiler = PhysicsCompiler()

# 2. Compile (This runs new process_fluids logic)
# ------------------------------------------------
print("2. Generating Particles...")
result = compiler.compile_dsl(dsl_code)

if not result['success']:
    print("Compilation failed!")
    exit(1)

# 3. Generate C++ Engine (This writes sph_template.cpp)
# ------------------------------------------------
print("3. Writing SPH Engine...")
compiler.compile_to_cpp("dam_break.cpp", target="standard", compile_binary=True)

# 4. Run the Simulation (This runs the compiled C++)
# ------------------------------------------------
print("4. Running Simulation (This may take a moment)...")
if os.name == 'nt':
    subprocess.call(["dam_break.exe"])
else:
    subprocess.call(["./dam_break"])

# 5. Visualize
# ------------------------------------------------
print("5. Rendering Animation...")
compiler.visualizer.animate_fluid_from_csv("dam_break_sph.csv")
plt.show()


