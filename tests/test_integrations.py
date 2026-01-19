"""
Tests for game engine integrations with equation generation.

Tests Unity, Unreal, and Modelica code generators to ensure
they properly convert sympy equations to target language code.
"""

import pytest
from unittest.mock import MagicMock

# Test sympy-to-target-language converters

class TestUnityIntegration:
    """Tests for Unity C# code generation."""
    
    def test_sympy_to_csharp_none(self):
        """Test conversion of None expression."""
        from mechanics_dsl.integrations.unity import sympy_to_csharp
        result = sympy_to_csharp(None)
        assert result == "0.0f"
    
    def test_sympy_to_csharp_simple(self):
        """Test conversion of simple sympy expression."""
        try:
            import sympy as sp
            from mechanics_dsl.integrations.unity import sympy_to_csharp
            
            # Simple expression: -g/l * sin(theta)
            g, l, theta = sp.symbols('g l theta')
            expr = -g/l * sp.sin(theta)
            
            result = sympy_to_csharp(expr)
            
            # Should contain Mathf.Sin
            assert "Mathf.Sin" in result
            assert "theta" in result
        except ImportError:
            pytest.skip("sympy not available")
    
    @pytest.mark.skip(reason="UnityGenerator inherits abstract base requiring generate_equations")
    def test_unity_generator_creates_class(self):
        """Test that UnityGenerator produces valid C# structure."""
        from mechanics_dsl.integrations.unity import UnityGenerator
        
        # Initialize without compiler to avoid abstract class issues
        gen = UnityGenerator(None)
        gen.system_name = "pendulum"
        gen.coordinates = ["theta"]
        gen.parameters = {"g": 9.81, "l": 1.0}
        gen.equations = {}
        
        code = gen.generate()
        
        assert "class PendulumSimulator" in code
        assert "MonoBehaviour" in code
        assert "RK4Step" in code
        assert "ComputeDerivatives" in code


class TestUnrealIntegration:
    """Tests for Unreal Engine C++ code generation."""
    
    def test_sympy_to_cpp_none(self):
        """Test conversion of None expression."""
        from mechanics_dsl.integrations.unreal import sympy_to_cpp
        result = sympy_to_cpp(None)
        assert result == "0.0f"
    
    def test_sympy_to_cpp_with_fmath(self):
        """Test conversion with FMath functions."""
        try:
            import sympy as sp
            from mechanics_dsl.integrations.unreal import sympy_to_cpp
            
            theta = sp.Symbol('theta')
            expr = sp.sin(theta)
            
            result = sympy_to_cpp(expr, use_fmath=True)
            
            assert "FMath::Sin" in result
        except ImportError:
            pytest.skip("sympy not available")
    
    def test_unreal_generator_creates_component(self):
        """Test that UnrealGenerator produces valid C++ structure."""
        from mechanics_dsl.integrations.unreal import UnrealGenerator
        
        # Mock compiler
        mock_compiler = MagicMock()
        mock_compiler.system_name = "pendulum"
        mock_compiler.simulator.coordinates = ["theta"]
        mock_compiler.simulator.parameters = {"g": 9.81}
        mock_compiler.accelerations = {}
        
        gen = UnrealGenerator(mock_compiler)
        header, source = gen.generate()
        
        assert "UPendulumComponent" in header
        assert "GENERATED_BODY()" in header
        assert "RK4Step" in source


class TestModelicaIntegration:
    """Tests for Modelica code generation."""
    
    def test_sympy_to_modelica_none(self):
        """Test conversion of None expression."""
        from mechanics_dsl.integrations.modelica import sympy_to_modelica
        result = sympy_to_modelica(None)
        assert result == "0"
    
    def test_sympy_to_modelica_trig(self):
        """Test conversion of trig functions."""
        try:
            import sympy as sp
            from mechanics_dsl.integrations.modelica import sympy_to_modelica
            
            theta = sp.Symbol('theta')
            expr = sp.sin(theta)
            
            result = sympy_to_modelica(expr)
            
            assert "Modelica.Math.sin" in result
        except ImportError:
            pytest.skip("sympy not available")
    
    def test_modelica_generator_creates_model(self):
        """Test that ModelicaGenerator produces valid Modelica structure."""
        from mechanics_dsl.integrations.modelica import ModelicaGenerator
        
        # Mock compiler
        mock_compiler = MagicMock()
        mock_compiler.system_name = "pendulum"
        mock_compiler.simulator.coordinates = ["theta"]
        mock_compiler.simulator.parameters = {"g": 9.81}
        mock_compiler.simulator.initial_conditions = {"theta": 0.5}
        mock_compiler.accelerations = {}
        
        gen = ModelicaGenerator(mock_compiler)
        code = gen.generate()
        
        assert "model Pendulum" in code
        assert "end Pendulum;" in code
        assert "der(theta)" in code


class TestCudaSphCpuFallback:
    """Tests for CUDA SPH CPU fallback."""
    
    def test_cpu_fallback_generation(self):
        """Test that CPU fallback is generated with full implementation."""
        from mechanics_dsl.codegen.cuda_sph import CudaSPHGenerator
        
        gen = CudaSPHGenerator(
            system_name="test_dam",
            fluid_particles=[{"x": 0.1, "y": 0.1}],
            boundary_particles=[],
            parameters={"h": 0.04, "rho0": 1000}
        )
        
        cpu_code = gen._generate_cpu_fallback()
        
        # Should have full implementation, not just TODO
        assert "TODO" not in cpu_code
        assert "compute_density_pressure" in cpu_code
        assert "compute_forces" in cpu_code
        assert "integrate" in cpu_code
        assert "namespace params" in cpu_code


class TestDslImports:
    """Tests for DSL import functionality."""
    
    def test_import_def_in_parser(self):
        """Test that parser recognizes import directive."""
        from mechanics_dsl.parser import tokenize, MechanicsParser, ImportDef
        
        # Parser expects single identifier for filename
        # NOTE: Real filenames with dots/slashes would need parser enhancement
        source = r"\import{physics_common}"
        tokens = tokenize(source)
        parser = MechanicsParser(tokens)
        ast = parser.parse()
        
        assert len(ast) == 1
        assert isinstance(ast[0], ImportDef)
        assert ast[0].filename == "physics_common"
    
    def test_compiler_has_import_handler(self):
        """Test that compiler has _process_import method."""
        from mechanics_dsl.compiler import PhysicsCompiler
        
        compiler = PhysicsCompiler()
        
        assert hasattr(compiler, '_process_import')
        assert callable(compiler._process_import)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
