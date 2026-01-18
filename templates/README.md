# MechanicsDSL Project Templates

Starter templates for new projects using MechanicsDSL.

## Available Templates

### `basic/`
Simple starter for learning MechanicsDSL.
- Single DSL file
- Basic simulation script
- README with instructions

### `research/`
Template for academic research projects.
- Data pipeline setup
- Jupyter notebooks
- Paper-ready figure generation
- BibTeX citations

### `game-physics/`
Integration with game engines.
- Unity/Unreal export setup
- Real-time simulation
- Visual debugging

### `embedded/`
For ARM/embedded systems.
- Arduino integration
- Minimal dependencies
- Cross-compilation support

## Usage

```bash
# Copy a template to start a new project
cp -r templates/basic/ my-new-project/
cd my-new-project

# Initialize
pip install mechanicsdsl-core
python simulate.py
```

## Template Structure

Each template includes:
- `README.md` — Setup and usage instructions
- `mechanicsdsl.json` — Project configuration
- `*.mdsl` — Example DSL files
- `simulate.py` — Basic simulation script
