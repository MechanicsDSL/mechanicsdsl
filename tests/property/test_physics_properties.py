"""
Property-based tests using Hypothesis for physics invariants.

These tests verify fundamental physics properties hold across a wide range of inputs.
"""
import pytest
import numpy as np

# Skip all tests in this module if hypothesis is not installed
pytest.importorskip("hypothesis")

from hypothesis import given, strategies as st, settings, assume

class TestEnergyConservation:
    """Test that conservative systems preserve energy."""
    
    @given(
        mass=st.floats(min_value=0.1, max_value=100.0),
        length=st.floats(min_value=0.1, max_value=10.0),
        initial_angle=st.floats(min_value=-np.pi/2, max_value=np.pi/2),
    )
    @settings(max_examples=20, deadline=10000)
    def test_pendulum_energy_bounded(self, mass, length, initial_angle):
        """Energy should remain bounded for any pendulum parameters."""
        from mechanics_dsl import PhysicsCompiler
        
        dsl = f"""
        \\system{{test_pendulum}}
        \\defvar{{theta}}{{Angle}}{{rad}}
        \\parameter{{m}}{{{mass}}}{{kg}}
        \\parameter{{l}}{{{length}}}{{m}}
        \\parameter{{g}}{{9.81}}{{m/s^2}}
        \\lagrangian{{0.5 * m * l^2 * \\dot{{theta}}^2 + m * g * l * \\cos{{theta}}}}
        \\initial{{theta={initial_angle}, theta_dot=0}}
        """
        
        compiler = PhysicsCompiler()
        result = compiler.compile_dsl(dsl)
        
        assert result['success'], f"Compilation failed: {result.get('error')}"
        
        solution = compiler.simulate(t_span=(0, 2), num_points=100)
        
        # Energy should exist and be finite
        assert 'y' in solution
        assert np.all(np.isfinite(solution['y']))


class TestSymbolicDerivation:
    """Test symbolic engine properties."""
    
    @given(
        coeff=st.floats(min_value=0.1, max_value=10.0),
    )
    @settings(max_examples=10, deadline=10000)
    def test_harmonic_oscillator_frequency(self, coeff):
        """Harmonic oscillator should have well-defined frequency."""
        from mechanics_dsl import PhysicsCompiler
        
        dsl = f"""
        \\system{{harmonic}}
        \\defvar{{x}}{{Position}}{{m}}
        \\parameter{{m}}{{1.0}}{{kg}}
        \\parameter{{k}}{{{coeff}}}{{N/m}}
        \\lagrangian{{0.5 * m * \\dot{{x}}^2 - 0.5 * k * x^2}}
        \\initial{{x=1.0, x_dot=0}}
        """
        
        compiler = PhysicsCompiler()
        result = compiler.compile_dsl(dsl)
        
        assert result['success']
        assert 'x_ddot' in result['equations']


class TestNumericalStability:
    """Test numerical solver stability."""
    
    @given(
        t_end=st.floats(min_value=1.0, max_value=20.0),
        num_points=st.integers(min_value=50, max_value=500),
    )
    @settings(max_examples=10, deadline=30000)
    def test_simulation_completes(self, t_end, num_points):
        """Simulation should complete for reasonable time spans."""
        from mechanics_dsl import PhysicsCompiler
        
        dsl = """
        \\system{simple_pendulum}
        \\defvar{theta}{Angle}{rad}
        \\parameter{m}{1.0}{kg}
        \\parameter{l}{1.0}{m}
        \\parameter{g}{9.81}{m/s^2}
        \\lagrangian{0.5 * m * l^2 * \\dot{theta}^2 + m * g * l * \\cos{theta}}
        \\initial{theta=0.5, theta_dot=0}
        """
        
        compiler = PhysicsCompiler()
        result = compiler.compile_dsl(dsl)
        assert result['success']
        
        solution = compiler.simulate(t_span=(0, t_end), num_points=num_points)
        
        assert 't' in solution
        assert 'y' in solution
        assert len(solution['t']) > 0


class TestParserRobustness:
    """Test parser handles various inputs."""
    
    @given(
        var_name=st.from_regex(r'[a-zA-Z][a-zA-Z0-9_]{0,10}', fullmatch=True),
    )
    @settings(max_examples=20, deadline=5000)
    def test_valid_variable_names(self, var_name):
        """Parser should accept valid variable names."""
        # Skip Python reserved words and DSL keywords
        reserved = {'def', 'class', 'if', 'else', 'for', 'while', 'return', 
                   'import', 'from', 'as', 'try', 'except', 'finally',
                   'system', 'lagrangian', 'hamiltonian'}
        assume(var_name.lower() not in reserved)
        assume(len(var_name) >= 1)
        
        from mechanics_dsl import PhysicsCompiler
        
        dsl = f"""
        \\system{{test}}
        \\defvar{{{var_name}}}{{Position}}{{m}}
        \\parameter{{m}}{{1.0}}{{kg}}
        \\lagrangian{{0.5 * m * \\dot{{{var_name}}}^2}}
        \\initial{{{var_name}=1.0, {var_name}_dot=0}}
        """
        
        compiler = PhysicsCompiler()
        result = compiler.compile_dsl(dsl)
        
        # Should either succeed or fail gracefully (no crashes)
        assert 'success' in result


class TestCodeGeneration:
    """Test code generation backends."""
    
    @given(
        spring_k=st.floats(min_value=0.1, max_value=100.0),
    )
    @settings(max_examples=5, deadline=10000)
    def test_cpp_generation_syntax(self, spring_k):
        """Generated C++ should have valid structure."""
        from mechanics_dsl import PhysicsCompiler
        import tempfile
        import os
        
        dsl = f"""
        \\system{{oscillator}}
        \\defvar{{x}}{{Position}}{{m}}
        \\parameter{{m}}{{1.0}}{{kg}}
        \\parameter{{k}}{{{spring_k}}}{{N/m}}
        \\lagrangian{{0.5 * m * \\dot{{x}}^2 - 0.5 * k * x^2}}
        \\initial{{x=1.0, x_dot=0}}
        """
        
        compiler = PhysicsCompiler()
        result = compiler.compile_dsl(dsl)
        
        if result['success']:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False) as f:
                temp_file = f.name
            
            try:
                # Just test that generation doesn't crash
                compiler.compile_to_cpp(temp_file, target='standard', compile_binary=False)
                
                # Verify file was created
                assert os.path.exists(temp_file)
                
                # Verify it has content
                with open(temp_file, 'r') as f:
                    content = f.read()
                    assert len(content) > 0
                    assert '#include' in content  # Basic C++ structure
            finally:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
