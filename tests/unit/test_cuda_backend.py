"""
Tests for CUDA Code Generator

Validates:
- CUDA code generation
- CMakeLists.txt generation
- CPU fallback generation
- Kernel function signatures
"""
import pytest
import os
import tempfile
import sympy as sp

from mechanics_dsl.codegen.cuda import CudaGenerator


class TestCudaGeneratorBasics:
    """Test basic CudaGenerator functionality."""
    
    def setup_method(self):
        """Set up simple pendulum for testing."""
        theta = sp.Symbol('theta', real=True)
        g = sp.Symbol('g', positive=True)
        l = sp.Symbol('l', positive=True)
        
        self.equations = {'theta_ddot': -g/l * sp.sin(theta)}
        self.coordinates = ['theta']
        self.parameters = {'g': 9.81, 'l': 1.0}
        self.initial_conditions = {'theta': 0.1, 'theta_dot': 0.0}
    
    def test_create_generator(self):
        """Test generator instantiation."""
        gen = CudaGenerator(
            system_name="test_pendulum",
            coordinates=self.coordinates,
            parameters=self.parameters,
            initial_conditions=self.initial_conditions,
            equations=self.equations
        )
        
        assert gen is not None
        assert gen.target_name == 'cuda'
        assert gen.file_extension == '.cu'
    
    def test_generate_equations(self):
        """Test equation code generation."""
        gen = CudaGenerator(
            system_name="test_pendulum",
            coordinates=self.coordinates,
            parameters=self.parameters,
            initial_conditions=self.initial_conditions,
            equations=self.equations
        )
        
        eq_code = gen.generate_equations()
        
        # Should contain derivative assignments
        assert "dydt[0]" in eq_code
        assert "dydt[1]" in eq_code
        assert "sin" in eq_code  # From the pendulum equation


class TestCudaFileGeneration:
    """Test CUDA file generation."""
    
    def setup_method(self):
        """Set up test system."""
        theta = sp.Symbol('theta', real=True)
        g = sp.Symbol('g', positive=True)
        l = sp.Symbol('l', positive=True)
        
        self.equations = {'theta_ddot': -g/l * sp.sin(theta)}
        self.coordinates = ['theta']
        self.parameters = {'g': 9.81, 'l': 1.0}
        self.initial_conditions = {'theta': 0.1, 'theta_dot': 0.0}
    
    def test_generate_files(self):
        """Test complete file generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = CudaGenerator(
                system_name="pendulum",
                coordinates=self.coordinates,
                parameters=self.parameters,
                initial_conditions=self.initial_conditions,
                equations=self.equations,
                generate_cpu_fallback=True
            )
            
            cuda_file = gen.generate(tmpdir)
            
            # Check files were created
            assert os.path.exists(cuda_file)
            assert os.path.exists(os.path.join(tmpdir, "pendulum.h"))
            assert os.path.exists(os.path.join(tmpdir, "CMakeLists.txt"))
            assert os.path.exists(os.path.join(tmpdir, "pendulum_cpu.cpp"))
    
    def test_cuda_file_content(self):
        """Test CUDA file contains expected content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = CudaGenerator(
                system_name="pendulum",
                coordinates=self.coordinates,
                parameters=self.parameters,
                initial_conditions=self.initial_conditions,
                equations=self.equations
            )
            
            cuda_file = gen.generate(tmpdir)
            
            with open(cuda_file) as f:
                content = f.read()
            
            # Should contain CUDA-specific content
            assert "#include <cuda_runtime.h>" in content
            assert "__global__" in content
            assert "__device__" in content
            assert "cudaMalloc" in content
            assert "cudaMemcpy" in content
    
    def test_cmake_file_content(self):
        """Test CMakeLists.txt contains expected content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = CudaGenerator(
                system_name="pendulum",
                coordinates=self.coordinates,
                parameters=self.parameters,
                initial_conditions=self.initial_conditions,
                equations=self.equations
            )
            
            gen.generate(tmpdir)
            
            with open(os.path.join(tmpdir, "CMakeLists.txt")) as f:
                content = f.read()
            
            assert "CUDA" in content
            assert "find_package" in content
            assert "add_executable" in content
            assert "pendulum_cuda" in content
            assert "pendulum_cpu" in content
    
    def test_cpu_fallback_content(self):
        """Test CPU fallback contains valid C++ code."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = CudaGenerator(
                system_name="pendulum",
                coordinates=self.coordinates,
                parameters=self.parameters,
                initial_conditions=self.initial_conditions,
                equations=self.equations,
                generate_cpu_fallback=True
            )
            
            gen.generate(tmpdir)
            
            with open(os.path.join(tmpdir, "pendulum_cpu.cpp")) as f:
                content = f.read()
            
            # Should be standard C++ without CUDA
            assert "#include <iostream>" in content
            assert "void rk4_step" in content
            assert "int main()" in content
            # Should NOT contain CUDA-specific keywords
            assert "__global__" not in content
            assert "cudaMalloc" not in content


class TestMultiCoordinateSystem:
    """Test CUDA generation for multi-coordinate systems."""
    
    def test_double_pendulum(self):
        """Test generation for double pendulum (2 DOF)."""
        theta1 = sp.Symbol('theta1', real=True)
        theta2 = sp.Symbol('theta2', real=True)
        g = sp.Symbol('g', positive=True)
        
        equations = {
            'theta1_ddot': -g * sp.sin(theta1),
            'theta2_ddot': -g * sp.sin(theta2)
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = CudaGenerator(
                system_name="double_pendulum",
                coordinates=['theta1', 'theta2'],
                parameters={'g': 9.81},
                initial_conditions={
                    'theta1': 0.1, 'theta1_dot': 0.0,
                    'theta2': 0.2, 'theta2_dot': 0.0
                },
                equations=equations
            )
            
            cuda_file = gen.generate(tmpdir)
            
            with open(cuda_file) as f:
                content = f.read()
            
            # Should handle both coordinates
            assert "theta1" in content
            assert "theta2" in content
            assert "STATE_DIM = 4" in content  # 2 coords * 2 (pos + vel)


class TestCPUFallbackOption:
    """Test CPU fallback generation option."""
    
    def test_no_fallback_option(self):
        """Test generation without CPU fallback."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = CudaGenerator(
                system_name="test",
                coordinates=['x'],
                parameters={'k': 1.0},
                initial_conditions={'x': 1.0, 'x_dot': 0.0},
                equations={'x_ddot': -sp.Symbol('k') * sp.Symbol('x')},
                generate_cpu_fallback=False
            )
            
            gen.generate(tmpdir)
            
            # CPU file should NOT exist
            assert not os.path.exists(os.path.join(tmpdir, "test_cpu.cpp"))
            # CUDA file should exist
            assert os.path.exists(os.path.join(tmpdir, "test.cu"))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
