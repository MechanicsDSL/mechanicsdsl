"""
MechanicsDSL Streamlit Web Interface
Beautiful, no-setup web app for classical mechanics simulations

Installation:
    pip install streamlit

Run:
    streamlit run app.py
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# Import your MechanicsDSL
from core import (
    PhysicsCompiler,
    PotentialEnergyCalculator,
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

# Page config
st.set_page_config(
    page_title="MechanicsDSL - Classical Mechanics Simulator",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 50%, #0f172a 100%);
    }
    .stTextArea textarea {
        font-family: 'Courier New', monospace;
        background-color: #1e293b;
        color: #22c55e;
    }
    .stButton button {
        background: linear-gradient(90deg, #10b981 0%, #059669 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 0.5rem;
    }
    .stButton button:hover {
        background: linear-gradient(90deg, #059669 0%, #047857 100%);
    }
    h1 {
        background: linear-gradient(90deg, #60a5fa, #22d3ee);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: bold;
    }
    .success-box {
        background-color: #064e3b;
        border-left: 4px solid #10b981;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #7f1d1d;
        border-left: 4px solid #ef4444;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Example system mapping
EXAMPLES = {
    'Basic Systems': {
        'Simple Pendulum': example_simple_pendulum,
        'Harmonic Oscillator': example_harmonic_oscillator,
        'Damped Pendulum': example_damped_pendulum,
        'Damped Oscillator': example_damped_oscillator,
    },
    'Advanced Pendulums': {
        'Double Pendulum': example_double_pendulum,
        'Triple Pendulum': example_triple_pendulum,
        'Spherical Pendulum': example_spherical_pendulum,
        'Spring Pendulum': example_spring_pendulum,
    },
    'Coupled Systems': {
        'Coupled Oscillators': example_coupled_oscillators,
        'Three Coupled Pendulums': example_coupled_pendulums_3,
        'Chain Pendulum': example_chain_pendulum,
    },
    'Rotating Systems': {
        'Rotating Pendulum': example_rotating_pendulum,
        'Gyroscope': example_gyroscope,
        'Rigid Body 3D': example_rigid_body_3d,
        'Whirling Pendulum': example_whirling_pendulum,
    },
    'Chaotic Systems': {
        'Duffing Oscillator': example_duffing_oscillator,
        'Van der Pol': example_van_der_pol,
        'Hénon-Heiles': example_henon_heiles,
        'Lorenz System': example_lorenz_system,
        'Rössler Attractor': example_rossler_attractor,
    },
    'Applied Systems': {
        'Atwood Machine': example_atwood_machine,
        'Rolling Ball': example_rolling_ball,
        'Robot Arm': example_planar_robot_arm,
        'Elastic Collision': example_elastic_collision,
    },
    'External Forces': {
        'Forced Oscillator': example_forced_oscillator,
        'Charged Pendulum': example_charged_pendulum,
        'Magnetic Pendulum': example_magnetic_pendulum,
        'Parametric Pendulum': example_parametric_pendulum,
    },
}

# Initialize session state
if 'dsl_code' not in st.session_state:
    st.session_state.dsl_code = ""
if 'compilation_result' not in st.session_state:
    st.session_state.compilation_result = None
if 'solution' not in st.session_state:
    st.session_state.solution = None

# Header
st.markdown("# ⚡ MechanicsDSL")
st.markdown("### *Enterprise-Grade Classical Mechanics Simulator v6.0.0*")

# Sidebar
with st.sidebar:
    st.markdown("## 🎛️ Simulation Settings")
    
    # Time settings
    time_span = st.slider("Time Span (seconds)", 0.1, 50.0, 10.0, 0.1)
    num_points = st.slider("Number of Points", 100, 5000, 1000, 100)
    
    # Options
    st.markdown("### Options")
    use_hamiltonian = st.checkbox("Use Hamiltonian Formulation")
    show_energy = st.checkbox("Show Energy Analysis", value=True)
    show_phase = st.checkbox("Show Phase Space", value=False)
    
    st.markdown("---")
    
    # Examples
    st.markdown("## 📚 Load Example")
    
    category = st.selectbox("Category", list(EXAMPLES.keys()))
    example_name = st.selectbox("System", list(EXAMPLES[category].keys()))
    
    if st.button("📥 Load Example", use_container_width=True):
        example_func = EXAMPLES[category][example_name]
        st.session_state.dsl_code = example_func()
        st.success(f"✅ Loaded: {example_name}")
        st.rerun()
    
    st.markdown("---")
    
    # Info
    st.markdown("### ℹ️ Features")
    st.markdown("""
    - 44+ Example Systems
    - Real-time Compilation
    - Energy Conservation
    - Phase Space Analysis
    - Animation Export
    - Advanced Solvers
    """)

# Main content
col1, col2 = st.columns([3, 2])

with col1:
    st.markdown("## 📝 DSL Editor")
    
    # Code editor
    dsl_code = st.text_area(
        "Enter MechanicsDSL code or load an example:",
        value=st.session_state.dsl_code,
        height=400,
        placeholder="% Write your DSL code here or load an example from the sidebar\n\n\\system{my_system}\n\\lagrangian{...}",
        key="code_editor"
    )
    
    st.session_state.dsl_code = dsl_code
    
    # Action buttons
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
    
    with col_btn1:
        run_button = st.button("🚀 Run Simulation", use_container_width=True, type="primary")
    
    with col_btn2:
        clear_button = st.button("🗑️ Clear", use_container_width=True)
        if clear_button:
            st.session_state.dsl_code = ""
            st.session_state.compilation_result = None
            st.session_state.solution = None
            st.rerun()

with col2:
    st.markdown("## 📊 Results")
    
    if run_button and dsl_code.strip():
        with st.spinner("🔄 Compiling and simulating..."):
            try:
                # Compile
                compiler = PhysicsCompiler()
                result = compiler.compile_dsl(dsl_code, use_hamiltonian=use_hamiltonian)
                
                if result['success']:
                    st.session_state.compilation_result = result
                    
                    # Simulate
                    solution = compiler.simulate(
                        t_span=(0, time_span),
                        num_points=num_points
                    )
                    
                    if solution['success']:
                        st.session_state.solution = solution
                        st.session_state.compiler = compiler
                        
                        # Success message
                        st.markdown(f"""
                        <div class="success-box">
                            <h3>✅ Compilation Successful</h3>
                            <p><strong>System:</strong> {result['system_name']}</p>
                            <p><strong>Formulation:</strong> {result['formulation']}</p>
                            <p><strong>Coordinates:</strong> {', '.join(result['coordinates'])}</p>
                            <p><strong>Compilation Time:</strong> {result['compilation_time']:.4f}s</p>
                            <p><strong>Function Evaluations:</strong> {solution['nfev']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="error-box">
                            <h3>❌ Simulation Failed</h3>
                            <p>{solution.get('error', 'Unknown error')}</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="error-box">
                        <h3>❌ Compilation Failed</h3>
                        <p>{result.get('error', 'Unknown error')}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if 'traceback' in result:
                        with st.expander("Show Traceback"):
                            st.code(result['traceback'], language='python')
                    
            except Exception as e:
                st.markdown(f"""
                <div class="error-box">
                    <h3>❌ Error</h3>
                    <p>{str(e)}</p>
                </div>
                """, unsafe_allow_html=True)
    
    elif run_button:
        st.warning("⚠️ Please enter DSL code or load an example")
    
    # Show existing results
    elif st.session_state.compilation_result:
        result = st.session_state.compilation_result
        st.markdown(f"""
        <div class="success-box">
            <h3>✅ Last Successful Simulation</h3>
            <p><strong>System:</strong> {result['system_name']}</p>
            <p><strong>Coordinates:</strong> {', '.join(result['coordinates'])}</p>
        </div>
        """, unsafe_allow_html=True)

# Visualization section
if st.session_state.solution and st.session_state.solution['success']:
    st.markdown("---")
    st.markdown("## 📈 Visualizations")
    
    solution = st.session_state.solution
    compiler = st.session_state.compiler
    
    # Energy plot
    if show_energy:
        st.markdown("### ⚡ Energy Analysis")
        
        try:
            t = solution['t']
            y = solution['y']
            params = compiler.simulator.parameters
            
            # Calculate energies
            KE = PotentialEnergyCalculator.compute_kinetic_energy(solution, params)
            PE = PotentialEnergyCalculator.compute_potential_energy(solution, params, compiler.system_name)
            E_total = KE + PE
            
            # Create figure
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.patch.set_facecolor('#0f172a')
            
            for ax in axes.flat:
                ax.set_facecolor('#1e293b')
                ax.tick_params(colors='white')
                ax.spines['bottom'].set_color('white')
                ax.spines['top'].set_color('white')
                ax.spines['left'].set_color('white')
                ax.spines['right'].set_color('white')
                ax.xaxis.label.set_color('white')
                ax.yaxis.label.set_color('white')
            
            # KE
            axes[0, 0].plot(t, KE, '#ef4444', linewidth=2)
            axes[0, 0].set_xlabel('Time (s)')
            axes[0, 0].set_ylabel('Energy (J)')
            axes[0, 0].set_title('Kinetic Energy', color='white', fontweight='bold')
            axes[0, 0].grid(True, alpha=0.2, color='white')
            
            # PE
            axes[0, 1].plot(t, PE, '#3b82f6', linewidth=2)
            axes[0, 1].set_xlabel('Time (s)')
            axes[0, 1].set_ylabel('Energy (J)')
            axes[0, 1].set_title('Potential Energy', color='white', fontweight='bold')
            axes[0, 1].grid(True, alpha=0.2, color='white')
            
            # Total
            axes[1, 0].plot(t, E_total, '#10b981', linewidth=2)
            axes[1, 0].set_xlabel('Time (s)')
            axes[1, 0].set_ylabel('Energy (J)')
            axes[1, 0].set_title('Total Energy', color='white', fontweight='bold')
            axes[1, 0].grid(True, alpha=0.2, color='white')
            
            # Error
            E_error = (E_total - E_total[0]) / np.abs(E_total[0]) * 100 if E_total[0] != 0 else (E_total - E_total[0])
            axes[1, 1].plot(t, E_error, '#a855f7', linewidth=2)
            axes[1, 1].set_xlabel('Time (s)')
            axes[1, 1].set_ylabel('Relative Error (%)')
            axes[1, 1].set_title('Energy Conservation Error', color='white', fontweight='bold')
            axes[1, 1].grid(True, alpha=0.2, color='white')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            # Energy statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Initial Energy", f"{E_total[0]:.6f} J")
            with col2:
                st.metric("Final Energy", f"{E_total[-1]:.6f} J")
            with col3:
                st.metric("Max Error", f"{np.max(np.abs(E_error)):.6f}%")
                
        except Exception as e:
            st.error(f"Error generating energy plot: {e}")
    
    # Phase space plot
    if show_phase and len(solution['coordinates']) > 0:
        st.markdown("### 🌀 Phase Space")
        
        try:
            coordinate_index = st.selectbox(
                "Select coordinate:",
                range(len(solution['coordinates'])),
                format_func=lambda i: solution['coordinates'][i]
            )
            
            y = solution['y']
            coords = solution['coordinates']
            
            pos_idx = 2 * coordinate_index
            vel_idx = 2 * coordinate_index + 1
            
            position = y[pos_idx]
            velocity = y[vel_idx]
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 10))
            fig.patch.set_facecolor('#0f172a')
            ax.set_facecolor('#1e293b')
            
            ax.plot(position, velocity, '#3b82f6', alpha=0.7, linewidth=1.5, label='Trajectory')
            ax.plot(position[0], velocity[0], 'go', markersize=10, label='Start', zorder=5)
            ax.plot(position[-1], velocity[-1], 'ro', markersize=10, label='End', zorder=5)
            
            ax.set_xlabel(f'{coords[coordinate_index]} (position)', fontsize=12, color='white')
            ax.set_ylabel(f'd{coords[coordinate_index]}/dt (velocity)', fontsize=12, color='white')
            ax.set_title(f'Phase Space: {coords[coordinate_index]}', fontsize=14, fontweight='bold', color='white')
            ax.grid(True, alpha=0.2, color='white')
            ax.legend(fontsize=10, facecolor='#1e293b', edgecolor='white', labelcolor='white')
            ax.tick_params(colors='white')
            ax.spines['bottom'].set_color('white')
            ax.spines['top'].set_color('white')
            ax.spines['left'].set_color('white')
            ax.spines['right'].set_color('white')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
        except Exception as e:
            st.error(f"Error generating phase plot: {e}")
    
    # Export section
    st.markdown("---")
    st.markdown("### 💾 Export")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📹 Export Animation (MP4)", use_container_width=True):
            try:
                with st.spinner("Generating animation..."):
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
                        filename = tmp.name
                    
                    compiler.export_animation(solution, filename)
                    
                    with open(filename, 'rb') as f:
                        st.download_button(
                            label="⬇️ Download Animation",
                            data=f,
                            file_name="mechanics_simulation.mp4",
                            mime="video/mp4",
                            use_container_width=True
                        )
            except Exception as e:
                st.error(f"Error exporting animation: {e}")
    
    with col2:
        if st.button("📊 Export Data (JSON)", use_container_width=True):
            try:
                import json
                data = {
                    'system_name': compiler.system_name,
                    'coordinates': solution['coordinates'],
                    't': solution['t'].tolist(),
                    'y': solution['y'].tolist(),
                }
                json_str = json.dumps(data, indent=2)
                st.download_button(
                    label="⬇️ Download Data",
                    data=json_str,
                    file_name="simulation_data.json",
                    mime="application/json",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Error exporting data: {e}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #64748b; padding: 2rem;'>
    <p><strong>MechanicsDSL v6.0.0 - Enterprise Edition</strong></p>
    <p>A Domain-Specific Language for Classical Mechanics</p>
    <p>Built with ❤️ using Streamlit</p>
</div>
""", unsafe_allow_html=True)
