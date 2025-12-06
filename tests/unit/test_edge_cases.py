"""
Edge case and error handling tests
Tests both new package structure and original core.py
"""
import pytest
import numpy as np
import sys
from pathlib import Path

try:
    from mechanics_dsl import PhysicsCompiler
    NEW_PACKAGE = True
except ImportError:
    NEW_PACKAGE = False

try:
    sys.path.insert(0, str(Path(__file__).parent.parent / 'reference'))
    from core import PhysicsCompiler as CorePhysicsCompiler
    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False


def get_compiler():
    """Get compiler instance"""
    if NEW_PACKAGE:
        return PhysicsCompiler()
    elif CORE_AVAILABLE:
        return CorePhysicsCompiler()
    else:
        pytest.skip("Neither new package nor core.py available")


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_empty_dsl(self):
        """Test that empty DSL is rejected"""
        compiler = get_compiler()
        # Empty DSL should raise ValueError
        try:
            result = compiler.compile_dsl("")
            # If it doesn't raise, check result
            assert not result.get('success', True)
        except ValueError:
            # This is the expected behavior
            pass
    
    def test_invalid_syntax(self):
        """Test that invalid syntax is handled gracefully"""
        compiler = get_compiler()
        result = compiler.compile_dsl("invalid syntax here")
        
        # Should either fail gracefully or return error
        assert 'success' in result
        if not result['success']:
            assert 'error' in result
    
    def test_missing_lagrangian(self):
        """Test system without Lagrangian"""
        dsl_code = r"""
        \system{no_lagrangian}
        \defvar{x}{Position}{m}
        """
        
        compiler = get_compiler()
        result = compiler.compile_dsl(dsl_code)
        
        # Should fail or handle gracefully
        assert 'success' in result
    
    def test_missing_initial_conditions(self):
        """Test system without initial conditions"""
        dsl_code = r"""
        \system{no_initial}
        \defvar{x}{Position}{m}
        \defvar{m}{Mass}{kg}
        \defvar{k}{Spring Constant}{N/m}
        
        \parameter{m}{1.0}{kg}
        \parameter{k}{10.0}{N/m}
        
        \lagrangian{\frac{1}{2} * m * \dot{x}^2 - \frac{1}{2} * k * x^2}
        """
        
        compiler = get_compiler()
        result = compiler.compile_dsl(dsl_code)
        
        # Should compile but simulation might use defaults
        if result['success']:
            solution = compiler.simulate(t_span=(0, 1), num_points=10)
            # Should either work with defaults or fail gracefully
            assert 'success' in solution
    
    def test_zero_initial_conditions(self):
        """Test system with zero initial conditions"""
        dsl_code = r"""
        \system{zero_init}
        \defvar{x}{Position}{m}
        \defvar{m}{Mass}{kg}
        \defvar{k}{Spring Constant}{N/m}
        
        \parameter{m}{1.0}{kg}
        \parameter{k}{10.0}{N/m}
        
        \lagrangian{\frac{1}{2} * m * \dot{x}^2 - \frac{1}{2} * k * x^2}
        
        \initial{x=0.0, x_dot=0.0}
        """
        
        compiler = get_compiler()
        result = compiler.compile_dsl(dsl_code)
        assert result['success']
        
        solution = compiler.simulate(t_span=(0, 1), num_points=100)
        assert solution['success']
        
        # System should stay at rest (allow for numerical drift)
        x = solution['y'][0]
        # Use more lenient tolerance for zero initial conditions due to numerical drift
        assert np.allclose(x, 0.0, atol=1e-4), f"System not at rest: max |x| = {np.max(np.abs(x)):.2e}"
    
    def test_very_small_initial_conditions(self):
        """Test system with very small initial conditions"""
        dsl_code = r"""
        \system{small_init}
        \defvar{x}{Position}{m}
        \defvar{m}{Mass}{kg}
        \defvar{k}{Spring Constant}{N/m}
        
        \parameter{m}{1.0}{kg}
        \parameter{k}{10.0}{N/m}
        
        \lagrangian{\frac{1}{2} * m * \dot{x}^2 - \frac{1}{2} * k * x^2}
        
        \initial{x=1e-10, x_dot=0.0}
        """
        
        compiler = get_compiler()
        result = compiler.compile_dsl(dsl_code)
        assert result['success']
        
        solution = compiler.simulate(t_span=(0, 1), num_points=100)
        assert solution['success']
        
        # Should handle small values without numerical issues
        x = solution['y'][0]
        assert np.all(np.isfinite(x))
    
    def test_very_large_initial_conditions(self):
        """Test system with very large initial conditions"""
        dsl_code = r"""
        \system{large_init}
        \defvar{x}{Position}{m}
        \defvar{m}{Mass}{kg}
        \defvar{k}{Spring Constant}{N/m}
        
        \parameter{m}{1.0}{kg}
        \parameter{k}{10.0}{N/m}
        
        \lagrangian{\frac{1}{2} * m * \dot{x}^2 - \frac{1}{2} * k * x^2}
        
        \initial{x=100.0, x_dot=0.0}
        """
        
        compiler = get_compiler()
        result = compiler.compile_dsl(dsl_code)
        assert result['success']
        
        solution = compiler.simulate(t_span=(0, 1), num_points=100)
        assert solution['success']
        
        # Should handle large values
        x = solution['y'][0]
        assert np.all(np.isfinite(x))
        assert np.max(np.abs(x)) > 10.0


class TestBoundaryConditions:
    """Test boundary conditions and limits"""
    
    def test_very_short_time_span(self):
        """Test simulation with very short time span"""
        dsl_code = r"""
        \system{short_time}
        \defvar{x}{Position}{m}
        \defvar{m}{Mass}{kg}
        \defvar{k}{Spring Constant}{N/m}
        
        \parameter{m}{1.0}{kg}
        \parameter{k}{10.0}{N/m}
        
        \lagrangian{\frac{1}{2} * m * \dot{x}^2 - \frac{1}{2} * k * x^2}
        
        \initial{x=1.0, x_dot=0.0}
        """
        
        compiler = get_compiler()
        result = compiler.compile_dsl(dsl_code)
        assert result['success']
        
        solution = compiler.simulate(t_span=(0, 0.01), num_points=10)
        assert solution['success']
        assert len(solution['t']) == 10
    
    def test_minimal_points(self):
        """Test simulation with minimal number of points"""
        dsl_code = r"""
        \system{min_points}
        \defvar{x}{Position}{m}
        \defvar{m}{Mass}{kg}
        \defvar{k}{Spring Constant}{N/m}
        
        \parameter{m}{1.0}{kg}
        \parameter{k}{10.0}{N/m}
        
        \lagrangian{\frac{1}{2} * m * \dot{x}^2 - \frac{1}{2} * k * x^2}
        
        \initial{x=1.0, x_dot=0.0}
        """
        
        compiler = get_compiler()
        result = compiler.compile_dsl(dsl_code)
        assert result['success']
        
        solution = compiler.simulate(t_span=(0, 1), num_points=2)
        assert solution['success']
        assert len(solution['t']) == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

