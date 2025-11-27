"""
MechanicsDSL Backend API
Flask server to connect the React frontend to your core.py
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import io
import json
import base64
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
import tempfile
import traceback

# Import your core MechanicsDSL code
from core import (
    PhysicsCompiler, 
    run_example,
    SystemValidator,
    example_simple_pendulum,
    example_double_pendulum,
    example_harmonic_oscillator,
    example_atwood_machine,
    example_damped_pendulum,
    example_spring_pendulum,
    example_coupled_oscillators,
    example_rotating_pendulum,
    example_damped_oscillator,
    example_kepler_problem,
    example_charged_pendulum,
    example_inverted_pendulum,
    example_spherical_pendulum,
    example_forced_oscillator,
    example_elastic_collision,
    example_gyroscope,
    example_rolling_ball,
    example_magnetic_pendulum,
    example_chain_pendulum,
    example_anharmonic_oscillator,
    example_rotor_pendulum,
    example_duffing_oscillator,
    example_van_der_pol,
    example_lorenz_system,
    example_mathieu_oscillator,
    example_brusselator,
    example_rossler_attractor,
    example_henon_heiles,
    example_triple_pendulum,
    example_elastic_pendulum_3d,
    example_rotating_double_pendulum,
    example_spring_mass_damper,
    example_quadruple_pendulum,
    example_parametric_pendulum,
    example_whirling_pendulum,
    example_coupled_pendulums_3,
    example_nonlinear_spring,
    example_rotating_spring_pendulum,
    example_charged_oscillator,
    example_magnetic_dipole,
    example_rigid_body_3d,
    example_chaotic_oscillator,
    example_planar_robot_arm,
)

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Map example IDs to example functions
EXAMPLE_FUNCTIONS = {
    'simple_pendulum': example_simple_pendulum,
    'double_pendulum': example_double_pendulum,
    'harmonic_oscillator': example_harmonic_oscillator,
    'atwood_machine': example_atwood_machine,
    'damped_pendulum': example_damped_pendulum,
    'spring_pendulum': example_spring_pendulum,
    'coupled_oscillators': example_coupled_oscillators,
    'rotating_pendulum': example_rotating_pendulum,
    'damped_oscillator': example_damped_oscillator,
    'kepler_problem': example_kepler_problem,
    'charged_pendulum': example_charged_pendulum,
    'inverted_pendulum': example_inverted_pendulum,
    'spherical_pendulum': example_spherical_pendulum,
    'forced_oscillator': example_forced_oscillator,
    'elastic_collision': example_elastic_collision,
    'gyroscope': example_gyroscope,
    'rolling_ball': example_rolling_ball,
    'magnetic_pendulum': example_magnetic_pendulum,
    'chain_pendulum': example_chain_pendulum,
    'anharmonic_oscillator': example_anharmonic_oscillator,
    'rotor_pendulum': example_rotor_pendulum,
    'duffing_oscillator': example_duffing_oscillator,
    'van_der_pol': example_van_der_pol,
    'lorenz_system': example_lorenz_system,
    'mathieu_oscillator': example_mathieu_oscillator,
    'brusselator': example_brusselator,
    'rossler_attractor': example_rossler_attractor,
    'henon_heiles': example_henon_heiles,
    'triple_pendulum': example_triple_pendulum,
    'elastic_pendulum_3d': example_elastic_pendulum_3d,
    'rotating_double_pendulum': example_rotating_double_pendulum,
    'spring_mass_damper': example_spring_mass_damper,
    'quadruple_pendulum': example_quadruple_pendulum,
    'parametric_pendulum': example_parametric_pendulum,
    'whirling_pendulum': example_whirling_pendulum,
    'coupled_pendulums_3': example_coupled_pendulums_3,
    'nonlinear_spring': example_nonlinear_spring,
    'rotating_spring_pendulum': example_rotating_spring_pendulum,
    'charged_oscillator': example_charged_oscillator,
    'magnetic_dipole': example_magnetic_dipole,
    'rigid_body_3d': example_rigid_body_3d,
    'chaotic_oscillator': example_chaotic_oscillator,
    'planar_robot_arm': example_planar_robot_arm,
}


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'version': '6.0.0'})


@app.route('/api/examples/<example_id>', methods=['GET'])
def get_example(example_id):
    """Get DSL code for a specific example"""
    try:
        if example_id not in EXAMPLE_FUNCTIONS:
            return jsonify({'error': 'Example not found'}), 404
        
        dsl_code = EXAMPLE_FUNCTIONS[example_id]()
        return jsonify({
            'success': True,
            'example_id': example_id,
            'dsl_code': dsl_code
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/api/compile', methods=['POST'])
def compile_dsl():
    """Compile DSL code and return compilation results"""
    try:
        data = request.json
        dsl_code = data.get('dsl_code', '')
        use_hamiltonian = data.get('use_hamiltonian', False)
        
        if not dsl_code:
            return jsonify({
                'success': False,
                'error': 'No DSL code provided'
            }), 400
        
        # Create compiler and compile
        compiler = PhysicsCompiler()
        result = compiler.compile_dsl(dsl_code, use_hamiltonian=use_hamiltonian)
        
        # Convert numpy types to Python types for JSON serialization
        if result.get('success'):
            response = {
                'success': True,
                'system_name': result.get('system_name', 'Unknown'),
                'coordinates': result.get('coordinates', []),
                'compilation_time': float(result.get('compilation_time', 0)),
                'formulation': result.get('formulation', 'Lagrangian'),
                'num_constraints': result.get('num_constraints', 0),
                'ast_nodes': result.get('ast_nodes', 0),
                'variables': result.get('variables', {}),
                'parameters': {k: float(v) for k, v in result.get('parameters', {}).items()},
            }
        else:
            response = {
                'success': False,
                'error': result.get('error', 'Unknown compilation error'),
                'traceback': result.get('traceback', '')
            }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/api/simulate', methods=['POST'])
def simulate():
    """Compile and simulate a system"""
    try:
        data = request.json
        dsl_code = data.get('dsl_code', '')
        use_hamiltonian = data.get('use_hamiltonian', False)
        t_span = data.get('t_span', [0, 10])
        num_points = data.get('num_points', 1000)
        
        if not dsl_code:
            return jsonify({
                'success': False,
                'error': 'No DSL code provided'
            }), 400
        
        # Compile
        compiler = PhysicsCompiler()
        compile_result = compiler.compile_dsl(dsl_code, use_hamiltonian=use_hamiltonian)
        
        if not compile_result['success']:
            return jsonify(compile_result), 400
        
        # Simulate
        solution = compiler.simulate(
            t_span=(float(t_span[0]), float(t_span[1])),
            num_points=int(num_points)
        )
        
        if not solution['success']:
            return jsonify({
                'success': False,
                'error': solution.get('error', 'Simulation failed')
            }), 500
        
        # Prepare response with serializable data
        response = {
            'success': True,
            'compilation': {
                'system_name': compile_result.get('system_name', 'Unknown'),
                'coordinates': compile_result.get('coordinates', []),
                'compilation_time': float(compile_result.get('compilation_time', 0)),
                'formulation': compile_result.get('formulation', 'Lagrangian'),
            },
            'simulation': {
                'nfev': int(solution.get('nfev', 0)),
                'is_stiff': solution.get('is_stiff', False),
                'method_used': solution.get('method_used', 'RK45'),
                'message': solution.get('message', 'Success'),
            },
            # Convert numpy arrays to lists for JSON
            'data': {
                't': solution['t'].tolist(),
                'y': solution['y'].tolist(),
                'coordinates': solution['coordinates'],
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/api/plot/energy', methods=['POST'])
def plot_energy():
    """Generate energy plot and return as base64 image"""
    try:
        data = request.json
        solution = data.get('solution')
        parameters = data.get('parameters', {})
        
        if not solution:
            return jsonify({'error': 'No solution data provided'}), 400
        
        # Reconstruct solution dict
        solution_dict = {
            'success': True,
            't': solution['t'],
            'y': solution['y'],
            'coordinates': solution['coordinates'],
        }
        
        # Create energy plot
        from core import PotentialEnergyCalculator
        import numpy as np
        
        t = np.array(solution['t'])
        y = np.array(solution['y'])
        
        KE = PotentialEnergyCalculator.compute_kinetic_energy(
            {'success': True, 't': t, 'y': y, 'coordinates': solution['coordinates']},
            parameters
        )
        PE = PotentialEnergyCalculator.compute_potential_energy(
            {'success': True, 't': t, 'y': y, 'coordinates': solution['coordinates']},
            parameters,
            ''
        )
        E_total = KE + PE
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Energy Analysis', fontsize=16, fontweight='bold')
        
        axes[0, 0].plot(t, KE, 'r-', linewidth=2)
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Energy (J)')
        axes[0, 0].set_title('Kinetic Energy')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(t, PE, 'b-', linewidth=2)
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Energy (J)')
        axes[0, 1].set_title('Potential Energy')
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].plot(t, E_total, 'g-', linewidth=2)
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Energy (J)')
        axes[1, 0].set_title('Total Energy')
        axes[1, 0].grid(True, alpha=0.3)
        
        E_error = (E_total - E_total[0]) / np.abs(E_total[0]) * 100 if E_total[0] != 0 else (E_total - E_total[0])
        axes[1, 1].plot(t, E_error, 'purple', linewidth=2)
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Relative Error (%)')
        axes[1, 1].set_title('Energy Conservation Error')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Convert to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        
        return jsonify({
            'success': True,
            'image': f'data:image/png;base64,{img_base64}'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/api/plot/phase', methods=['POST'])
def plot_phase():
    """Generate phase space plot and return as base64 image"""
    try:
        data = request.json
        solution = data.get('solution')
        coordinate_index = data.get('coordinate_index', 0)
        
        if not solution:
            return jsonify({'error': 'No solution data provided'}), 400
        
        import numpy as np
        
        y = np.array(solution['y'])
        coords = solution['coordinates']
        
        pos_idx = 2 * coordinate_index
        vel_idx = 2 * coordinate_index + 1
        
        position = y[pos_idx]
        velocity = y[vel_idx]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.plot(position, velocity, 'b-', alpha=0.7, linewidth=1.5, label='Trajectory')
        ax.plot(position[0], velocity[0], 'go', markersize=10, label='Start', zorder=5)
        ax.plot(position[-1], velocity[-1], 'ro', markersize=10, label='End', zorder=5)
        
        ax.set_xlabel(f'{coords[coordinate_index]} (position)', fontsize=12)
        ax.set_ylabel(f'd{coords[coordinate_index]}/dt (velocity)', fontsize=12)
        ax.set_title(f'Phase Space: {coords[coordinate_index]}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        
        plt.tight_layout()
        
        # Convert to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        
        return jsonify({
            'success': True,
            'image': f'data:image/png;base64,{img_base64}'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/api/export/animation', methods=['POST'])
def export_animation():
    """Generate and export animation"""
    try:
        data = request.json
        dsl_code = data.get('dsl_code', '')
        t_span = data.get('t_span', [0, 10])
        export_format = data.get('format', 'mp4')
        
        # Run simulation
        compiler = PhysicsCompiler()
        compile_result = compiler.compile_dsl(dsl_code)
        
        if not compile_result['success']:
            return jsonify({'error': 'Compilation failed'}), 400
        
        solution = compiler.simulate(
            t_span=(float(t_span[0]), float(t_span[1]))
        )
        
        if not solution['success']:
            return jsonify({'error': 'Simulation failed'}), 500
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(
            mode='w+b',
            suffix=f'.{export_format}',
            delete=False
        ) as tmp:
            filename = tmp.name
        
        # Export animation
        compiler.export_animation(solution, filename)
        
        return send_file(
            filename,
            mimetype=f'video/{export_format}' if export_format == 'mp4' else 'image/gif',
            as_attachment=True,
            download_name=f'animation.{export_format}'
        )
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/api/validate', methods=['POST'])
def validate_system():
    """Validate a system against analytical solutions"""
    try:
        data = request.json
        solution_data = data.get('solution')
        parameters = data.get('parameters', {})
        system_type = data.get('system_type', '')
        
        # Create validator
        validator = SystemValidator()
        
        # Reconstruct solution
        import numpy as np
        solution = {
            'success': True,
            't': np.array(solution_data['t']),
            'y': np.array(solution_data['y']),
            'coordinates': solution_data['coordinates'],
        }
        
        # Create a minimal compiler for validation
        compiler = PhysicsCompiler()
        compiler.simulator.parameters = parameters
        compiler.system_name = system_type
        
        results = {}
        
        # Energy conservation
        energy_valid = validator.validate_energy_conservation(compiler, solution)
        results['energy_conservation'] = {
            'passed': energy_valid,
            'tolerance': 0.05
        }
        
        # Analytical validation for oscillators
        if 'oscillator' in system_type.lower():
            analytical_valid = validator.validate_simple_harmonic_oscillator(
                compiler, solution
            )
            results['analytical_match'] = {
                'passed': analytical_valid,
                'tolerance': 0.01
            }
        
        return jsonify({
            'success': True,
            'validation_results': results
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


if __name__ == '__main__':
    print("="*70)
    print("MechanicsDSL Backend API Server")
    print("="*70)
    print("Starting Flask server on http://localhost:5000")
    print("API endpoints:")
    print("  GET  /api/health")
    print("  GET  /api/examples/<example_id>")
    print("  POST /api/compile")
    print("  POST /api/simulate")
    print("  POST /api/plot/energy")
    print("  POST /api/plot/phase")
    print("  POST /api/export/animation")
    print("  POST /api/validate")
    print("="*70)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
