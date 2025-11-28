"""
Tutorial 18: Export and Import Systems

MechanicsDSL allows you to save and load system configurations.
This is useful for:
- Sharing systems with others
- Saving complex setups
- Resuming work later
- Batch processing

We'll use the SystemSerializer class.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from mechanics_dsl import PhysicsCompiler
from mechanics_dsl.compiler import SystemSerializer

# ============================================================================
# Create and compile a system
# ============================================================================

print("="*60)
print("CREATING AND EXPORTING A SYSTEM")
print("="*60)

compiler1 = PhysicsCompiler()

dsl_code = """
\\system{harmonic_oscillator}

\\var{x}{Position}{m}

\\parameter{m}{1.0}{kg}
\\parameter{k}{10.0}{N/m}

\\lagrangian{\\frac{1}{2} * m * \\dot{x}^2 - \\frac{1}{2} * k * x^2}

\\initial{x=1.0, x_dot=0.0}
"""

result1 = compiler1.compile_dsl(dsl_code)
if not result1['success']:
    print(f"❌ Compilation failed: {result1.get('error')}")
    exit(1)

print("✅ System compiled successfully!")

# ============================================================================
# Export system
# ============================================================================

serializer = SystemSerializer()
export_path = '18_exported_system.json'

print(f"\nExporting system to {export_path}...")
export_result = serializer.export_system(compiler1, export_path)

if export_result:
    print("✅ System exported successfully!")
    
    # Show what was saved
    with open(export_path, 'r') as f:
        data = json.load(f)
    
    print(f"\nExported data includes:")
    print(f"   - System name: {data.get('system_name', 'N/A')}")
    print(f"   - Variables: {len(data.get('variables', {}))}")
    print(f"   - Parameters: {len(data.get('parameters', {}))}")
    print(f"   - Initial conditions: {len(data.get('initial_conditions', {}))}")
else:
    print("❌ Export failed!")

# ============================================================================
# Import system
# ============================================================================

print("\n" + "="*60)
print("IMPORTING THE SYSTEM")
print("="*60)

print(f"Importing system from {export_path}...")
imported_data = serializer.import_system(export_path)

if imported_data:
    print("✅ System imported successfully!")
    
    # Create new compiler and load data
    compiler2 = PhysicsCompiler()
    
    # Reconstruct DSL from imported data (simplified)
    # In practice, you'd restore the full compiler state
    print("\nImported system details:")
    print(f"   System name: {imported_data.get('system_name', 'N/A')}")
    print(f"   Variables: {list(imported_data.get('variables', {}).keys())}")
    print(f"   Parameters: {imported_data.get('parameters', {})}")
    print(f"   Initial conditions: {imported_data.get('initial_conditions', {})}")
else:
    print("❌ Import failed!")

# ============================================================================
# Example: Save simulation results
# ============================================================================

print("\n" + "="*60)
print("SAVING SIMULATION RESULTS")
print("="*60)

# Run simulation
solution = compiler1.simulate(t_span=(0, 10), num_points=500)

# Save results to file
results_path = '18_simulation_results.npz'
np.savez(results_path,
         t=solution['t'],
         y=solution['y'],
         success=solution['success'],
         message=solution.get('message', ''))

print(f"✅ Simulation results saved to {results_path}")

# Load results
loaded_data = np.load(results_path)
t_loaded = loaded_data['t']
y_loaded = loaded_data['y']

print(f"✅ Results loaded: {len(t_loaded)} time points")

# ============================================================================
# Compare original and loaded
# ============================================================================

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Original
axes[0].plot(solution['t'], solution['y'][0], 'b-', linewidth=2, label='Original')
axes[0].set_xlabel('Time (s)')
axes[0].set_ylabel('Position (m)')
axes[0].set_title('Original Simulation')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Loaded
axes[1].plot(t_loaded, y_loaded[0], 'r-', linewidth=2, label='Loaded from file')
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('Position (m)')
axes[1].set_title('Loaded Simulation')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('18_export_import_comparison.png', dpi=150)
print("\n✅ Saved: 18_export_import_comparison.png")

# ============================================================================
# Key insights
# ============================================================================

print("\n" + "="*60)
print("KEY INSIGHTS:")
print("="*60)
print("1. Use SystemSerializer.export_system() to save")
print("2. Use SystemSerializer.import_system() to load")
print("3. Systems saved as JSON (human-readable)")
print("4. Simulation results saved as NPZ (NumPy format)")
print("5. Useful for sharing and reproducibility")
print("6. Can save/load entire compiler state")
print("="*60)

plt.show()

