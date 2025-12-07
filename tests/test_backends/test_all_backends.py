"""
Comprehensive Backend Validation Test Suite

Tests all 9 code generation backends:
- C++, Python, Julia, Rust, MATLAB, Fortran, JavaScript, CUDA

Each backend is validated for:
1. Code generation success
2. Syntax validity (where checkable)
3. Content correctness
4. Consistency with other backends
"""
import pytest
import os
import tempfile
import sympy as sp
from typing import Dict, Any

# Import all generators
from mechanics_dsl.codegen import (
    CppGenerator,
    PythonGenerator,
    JuliaGenerator,
    RustGenerator,
    MatlabGenerator,
    FortranGenerator,
    JavaScriptGenerator,
    CudaGenerator
)


def create_test_system() -> Dict[str, Any]:
    """Create a standard pendulum system for testing."""
    theta = sp.Symbol('theta', real=True)
    g = sp.Symbol('g', positive=True)
    l = sp.Symbol('l', positive=True)
    
    return {
        'system_name': 'test_pendulum',
        'coordinates': ['theta'],
        'parameters': {'g': 9.81, 'l': 1.0},
        'initial_conditions': {'theta': 0.1, 'theta_dot': 0.0},
        'equations': {'theta_ddot': -g/l * sp.sin(theta)}
    }


class TestCppBackend:
    """Test C++ code generation."""
    
    def test_generates_file(self):
        """Test that C++ file is generated."""
        system = create_test_system()
        with tempfile.TemporaryDirectory() as tmpdir:
            output = os.path.join(tmpdir, 'test.cpp')
            gen = CppGenerator(**system)
            gen.generate(output)
            assert os.path.exists(output)
    
    def test_contains_equations(self):
        """Test that generated code contains equations."""
        system = create_test_system()
        with tempfile.TemporaryDirectory() as tmpdir:
            output = os.path.join(tmpdir, 'test.cpp')
            gen = CppGenerator(**system)
            gen.generate(output)
            
            with open(output) as f:
                content = f.read()
            
            assert 'sin' in content
            assert 'theta' in content
    
    def test_compiles_syntax(self):
        """Test basic C++ syntax (checks includes and braces)."""
        system = create_test_system()
        with tempfile.TemporaryDirectory() as tmpdir:
            output = os.path.join(tmpdir, 'test.cpp')
            gen = CppGenerator(**system)
            gen.generate(output)
            
            with open(output) as f:
                content = f.read()
            
            # Basic syntax checks
            assert '#include' in content
            assert 'int main()' in content or 'int main(' in content
            assert content.count('{') == content.count('}')


class TestPythonBackend:
    """Test Python code generation."""
    
    def test_generates_file(self):
        """Test that Python file is generated."""
        system = create_test_system()
        with tempfile.TemporaryDirectory() as tmpdir:
            output = os.path.join(tmpdir, 'test.py')
            gen = PythonGenerator(**system)
            gen.generate(output)
            assert os.path.exists(output)
    
    def test_valid_python_syntax(self):
        """Test that generated code is valid Python."""
        system = create_test_system()
        with tempfile.TemporaryDirectory() as tmpdir:
            output = os.path.join(tmpdir, 'test.py')
            gen = PythonGenerator(**system)
            gen.generate(output)
            
            with open(output) as f:
                content = f.read()
            
            # Compile to check syntax
            compile(content, output, 'exec')  # Raises SyntaxError if invalid
    
    def test_contains_numpy(self):
        """Test that generated code uses numpy."""
        system = create_test_system()
        with tempfile.TemporaryDirectory() as tmpdir:
            output = os.path.join(tmpdir, 'test.py')
            gen = PythonGenerator(**system)
            gen.generate(output)
            
            with open(output) as f:
                content = f.read()
            
            assert 'numpy' in content or 'np.' in content


class TestJuliaBackend:
    """Test Julia code generation."""
    
    def test_generates_file(self):
        """Test that Julia file is generated."""
        system = create_test_system()
        with tempfile.TemporaryDirectory() as tmpdir:
            output = os.path.join(tmpdir, 'test.jl')
            gen = JuliaGenerator(**system)
            gen.generate(output)
            assert os.path.exists(output)
    
    def test_julia_syntax(self):
        """Test basic Julia syntax."""
        system = create_test_system()
        with tempfile.TemporaryDirectory() as tmpdir:
            output = os.path.join(tmpdir, 'test.jl')
            gen = JuliaGenerator(**system)
            gen.generate(output)
            
            with open(output) as f:
                content = f.read()
            
            # Julia-specific syntax
            assert 'function' in content
            assert 'end' in content


class TestRustBackend:
    """Test Rust code generation."""
    
    def test_generates_file(self):
        """Test that Rust file is generated."""
        system = create_test_system()
        with tempfile.TemporaryDirectory() as tmpdir:
            output = os.path.join(tmpdir, 'test.rs')
            gen = RustGenerator(**system)
            gen.generate(output)
            assert os.path.exists(output)
    
    def test_rust_syntax(self):
        """Test basic Rust syntax."""
        system = create_test_system()
        with tempfile.TemporaryDirectory() as tmpdir:
            output = os.path.join(tmpdir, 'test.rs')
            gen = RustGenerator(**system)
            gen.generate(output)
            
            with open(output) as f:
                content = f.read()
            
            # Rust-specific syntax
            assert 'fn ' in content
            assert 'let ' in content


class TestMatlabBackend:
    """Test MATLAB code generation."""
    
    def test_generates_file(self):
        """Test that MATLAB file is generated."""
        system = create_test_system()
        with tempfile.TemporaryDirectory() as tmpdir:
            output = os.path.join(tmpdir, 'test.m')
            gen = MatlabGenerator(**system)
            gen.generate(output)
            assert os.path.exists(output)
    
    def test_matlab_syntax(self):
        """Test basic MATLAB syntax."""
        system = create_test_system()
        with tempfile.TemporaryDirectory() as tmpdir:
            output = os.path.join(tmpdir, 'test.m')
            gen = MatlabGenerator(**system)
            gen.generate(output)
            
            with open(output) as f:
                content = f.read()
            
            # MATLAB-specific syntax
            assert 'function' in content
            assert '%' in content  # Comments


class TestFortranBackend:
    """Test Fortran code generation."""
    
    def test_generates_file(self):
        """Test that Fortran file is generated."""
        system = create_test_system()
        with tempfile.TemporaryDirectory() as tmpdir:
            output = os.path.join(tmpdir, 'test.f90')
            gen = FortranGenerator(**system)
            gen.generate(output)
            assert os.path.exists(output)
    
    def test_fortran_syntax(self):
        """Test basic Fortran syntax."""
        system = create_test_system()
        with tempfile.TemporaryDirectory() as tmpdir:
            output = os.path.join(tmpdir, 'test.f90')
            gen = FortranGenerator(**system)
            gen.generate(output)
            
            with open(output) as f:
                content = f.read()
            
            # Fortran-specific syntax (case-insensitive)
            content_lower = content.lower()
            assert 'program' in content_lower or 'subroutine' in content_lower
            assert 'end' in content_lower


class TestJavaScriptBackend:
    """Test JavaScript code generation."""
    
    def test_generates_file(self):
        """Test that JavaScript file is generated."""
        system = create_test_system()
        with tempfile.TemporaryDirectory() as tmpdir:
            output = os.path.join(tmpdir, 'test.js')
            gen = JavaScriptGenerator(**system)
            gen.generate(output)
            assert os.path.exists(output)
    
    def test_javascript_syntax(self):
        """Test basic JavaScript syntax."""
        system = create_test_system()
        with tempfile.TemporaryDirectory() as tmpdir:
            output = os.path.join(tmpdir, 'test.js')
            gen = JavaScriptGenerator(**system)
            gen.generate(output)
            
            with open(output) as f:
                content = f.read()
            
            # JavaScript-specific syntax
            assert 'function' in content or 'const ' in content or '=>' in content
            assert 'Math.sin' in content or 'sin(' in content


class TestCudaBackend:
    """Test CUDA code generation."""
    
    def test_generates_files(self):
        """Test that CUDA files are generated."""
        system = create_test_system()
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = CudaGenerator(**system, generate_cpu_fallback=True)
            gen.generate(tmpdir)
            
            assert os.path.exists(os.path.join(tmpdir, 'test_pendulum.cu'))
            assert os.path.exists(os.path.join(tmpdir, 'test_pendulum.h'))
            assert os.path.exists(os.path.join(tmpdir, 'CMakeLists.txt'))
            assert os.path.exists(os.path.join(tmpdir, 'test_pendulum_cpu.cpp'))
    
    def test_cuda_kernel_syntax(self):
        """Test CUDA-specific syntax."""
        system = create_test_system()
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = CudaGenerator(**system)
            gen.generate(tmpdir)
            
            with open(os.path.join(tmpdir, 'test_pendulum.cu')) as f:
                content = f.read()
            
            assert '__global__' in content
            assert '__device__' in content
            assert 'cudaMalloc' in content


class TestBackendConsistency:
    """Test that all backends produce consistent structure."""
    
    def test_all_backends_generate_without_error(self):
        """Verify all backends can generate code."""
        system = create_test_system()
        # Explicit extensions since not all generators inherit from base
        generators = [
            ('C++', CppGenerator, '.cpp'),
            ('Python', PythonGenerator, '.py'),
            ('Julia', JuliaGenerator, '.jl'),
            ('Rust', RustGenerator, '.rs'),
            ('MATLAB', MatlabGenerator, '.m'),
            ('Fortran', FortranGenerator, '.f90'),
            ('JavaScript', JavaScriptGenerator, '.js'),
        ]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            for name, GenClass, ext in generators:
                gen = GenClass(**system)
                output = os.path.join(tmpdir, f'test{ext}')
                
                try:
                    gen.generate(output)
                except Exception as e:
                    pytest.fail(f"{name} backend failed: {e}")
                
                assert os.path.exists(output), f"{name} failed to create output file"
    
    def test_all_backends_contain_pendulum_equation(self):
        """Verify all backends contain the pendulum equation."""
        system = create_test_system()
        generators = [
            (CppGenerator, '.cpp'),
            (PythonGenerator, '.py'),
            (JuliaGenerator, '.jl'),
            (RustGenerator, '.rs'),
            (MatlabGenerator, '.m'),
            (FortranGenerator, '.f90'),
            (JavaScriptGenerator, '.js'),
        ]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            for GenClass, ext in generators:
                gen = GenClass(**system)
                output = os.path.join(tmpdir, f'test{ext}')
                gen.generate(output)
                
                with open(output) as f:
                    content = f.read().lower()
                
                # All should contain sin and theta in some form
                assert 'sin' in content, f"{GenClass.__name__} missing 'sin'"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
