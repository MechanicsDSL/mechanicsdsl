from mechanics_dsl import PhysicsCompiler, setup_logging
import matplotlib.pyplot as plt
import os

# 1. Enable logging so we see why it fails
setup_logging()

print("Initializing visualizer...")
compiler = PhysicsCompiler()

filename = "dam_break_sph.csv"

# 2. Verify file exists before asking the visualizer
if not os.path.exists(filename):
    print(f"❌ Error: '{filename}' not found in {os.getcwd()}")
    exit(1)

print(f"File found. Size: {os.path.getsize(filename)} bytes")

# 3. Load Data
print("Loading data (this may take a moment)...")
anim = compiler.visualizer.animate_fluid_from_csv(filename)

if anim:
    print("Rendering GIF...")
    try:
        anim.save("dam_break.gif", writer="pillow", fps=30)
        print("✅ Success! Animation saved to 'dam_break.gif'")
    except Exception as e:
        print(f"❌ Error saving GIF: {e}")
else:
    print("❌ Failed to load animation (Check logs above for details)")