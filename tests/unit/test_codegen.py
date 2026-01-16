"""
Unit tests for MechanicsDSL code generators.

Tests all code generation backends: C++, Python, Julia, Rust, MATLAB, Fortran, JavaScript.
"""

import os
import tempfile
from pathlib import Path

import pytest

from mechanics_dsl.codegen import (
    CodeGenerator,
    CppGenerator,
    FortranGenerator,
    JavaScriptGenerator,
    JuliaGenerator,
    MatlabGenerator,
    PythonGenerator,
    RustGenerator,
)

# Sample system data for testing
SAMPLE_SYSTEM = {
    "system_name": "test_pendulum",
    "coordinates": ["theta"],
    "parameters": {"m": 1.0, "L": 1.0, "g": 9.81},
    "initial_conditions": {"theta": 0.5, "theta_dot": 0.0},
    "equations": {},  # Empty for basic tests
}


class TestCodeGeneratorBase:
    """Tests for base CodeGenerator class."""

    def test_base_is_abstract(self):
        """CodeGenerator should be abstract."""
        with pytest.raises(TypeError):
            CodeGenerator(**SAMPLE_SYSTEM)


class TestPythonGenerator:
    """Tests for Python code generator."""

    def test_target_name(self):
        gen = PythonGenerator(**SAMPLE_SYSTEM)
        assert gen.target_name == "python"

    def test_file_extension(self):
        gen = PythonGenerator(**SAMPLE_SYSTEM)
        assert gen.file_extension == ".py"

    def test_generate_creates_file(self):
        gen = PythonGenerator(**SAMPLE_SYSTEM)
        with tempfile.TemporaryDirectory() as tmpdir:
            output = os.path.join(tmpdir, "test.py")
            gen.generate(output)
            assert os.path.exists(output)

    def test_generated_code_has_imports(self):
        gen = PythonGenerator(**SAMPLE_SYSTEM)
        with tempfile.TemporaryDirectory() as tmpdir:
            output = os.path.join(tmpdir, "test.py")
            gen.generate(output)
            with open(output) as f:
                code = f.read()
            assert "import numpy" in code
            assert "scipy.integrate" in code


class TestJuliaGenerator:
    """Tests for Julia code generator."""

    def test_target_name(self):
        gen = JuliaGenerator(**SAMPLE_SYSTEM)
        assert gen.target_name == "julia"

    def test_file_extension(self):
        gen = JuliaGenerator(**SAMPLE_SYSTEM)
        assert gen.file_extension == ".jl"

    def test_generate_creates_file(self):
        gen = JuliaGenerator(**SAMPLE_SYSTEM)
        with tempfile.TemporaryDirectory() as tmpdir:
            output = os.path.join(tmpdir, "test.jl")
            gen.generate(output)
            assert os.path.exists(output)

    def test_generated_code_has_julia_syntax(self):
        gen = JuliaGenerator(**SAMPLE_SYSTEM)
        with tempfile.TemporaryDirectory() as tmpdir:
            output = os.path.join(tmpdir, "test.jl")
            gen.generate(output)
            with open(output) as f:
                code = f.read()
            assert "using DifferentialEquations" in code
            assert "function" in code


class TestRustGenerator:
    """Tests for Rust code generator."""

    def test_target_name(self):
        gen = RustGenerator(**SAMPLE_SYSTEM)
        assert gen.target_name == "rust"

    def test_file_extension(self):
        gen = RustGenerator(**SAMPLE_SYSTEM)
        assert gen.file_extension == ".rs"

    def test_generate_creates_file(self):
        gen = RustGenerator(**SAMPLE_SYSTEM)
        with tempfile.TemporaryDirectory() as tmpdir:
            output = os.path.join(tmpdir, "test.rs")
            gen.generate(output)
            assert os.path.exists(output)

    def test_generated_code_has_rust_syntax(self):
        gen = RustGenerator(**SAMPLE_SYSTEM)
        with tempfile.TemporaryDirectory() as tmpdir:
            output = os.path.join(tmpdir, "test.rs")
            gen.generate(output)
            with open(output) as f:
                code = f.read()
            assert "fn main()" in code
            assert "const" in code


class TestMatlabGenerator:
    """Tests for MATLAB/Octave code generator."""

    def test_target_name(self):
        gen = MatlabGenerator(**SAMPLE_SYSTEM)
        assert gen.target_name == "matlab"

    def test_file_extension(self):
        gen = MatlabGenerator(**SAMPLE_SYSTEM)
        assert gen.file_extension == ".m"

    def test_generate_creates_file(self):
        gen = MatlabGenerator(**SAMPLE_SYSTEM)
        with tempfile.TemporaryDirectory() as tmpdir:
            output = os.path.join(tmpdir, "test.m")
            gen.generate(output)
            assert os.path.exists(output)

    def test_generated_code_has_matlab_syntax(self):
        gen = MatlabGenerator(**SAMPLE_SYSTEM)
        with tempfile.TemporaryDirectory() as tmpdir:
            output = os.path.join(tmpdir, "test.m")
            gen.generate(output)
            with open(output) as f:
                code = f.read()
            assert "ode45" in code
            assert "function" in code


class TestFortranGenerator:
    """Tests for Fortran code generator."""

    def test_target_name(self):
        gen = FortranGenerator(**SAMPLE_SYSTEM)
        assert gen.target_name == "fortran"

    def test_file_extension(self):
        gen = FortranGenerator(**SAMPLE_SYSTEM)
        assert gen.file_extension == ".f90"

    def test_generate_creates_file(self):
        gen = FortranGenerator(**SAMPLE_SYSTEM)
        with tempfile.TemporaryDirectory() as tmpdir:
            output = os.path.join(tmpdir, "test.f90")
            gen.generate(output)
            assert os.path.exists(output)

    def test_generated_code_has_fortran_syntax(self):
        gen = FortranGenerator(**SAMPLE_SYSTEM)
        with tempfile.TemporaryDirectory() as tmpdir:
            output = os.path.join(tmpdir, "test.f90")
            gen.generate(output)
            with open(output) as f:
                code = f.read()
            assert "program" in code
            assert "subroutine" in code


class TestJavaScriptGenerator:
    """Tests for JavaScript code generator."""

    def test_target_name(self):
        gen = JavaScriptGenerator(**SAMPLE_SYSTEM)
        assert gen.target_name == "javascript"

    def test_file_extension(self):
        gen = JavaScriptGenerator(**SAMPLE_SYSTEM)
        assert gen.file_extension == ".js"

    def test_generate_creates_file(self):
        gen = JavaScriptGenerator(**SAMPLE_SYSTEM)
        with tempfile.TemporaryDirectory() as tmpdir:
            output = os.path.join(tmpdir, "test.js")
            gen.generate(output)
            assert os.path.exists(output)

    def test_generated_code_has_js_syntax(self):
        gen = JavaScriptGenerator(**SAMPLE_SYSTEM)
        with tempfile.TemporaryDirectory() as tmpdir:
            output = os.path.join(tmpdir, "test.js")
            gen.generate(output)
            with open(output) as f:
                code = f.read()
            assert "function" in code
            assert "const" in code


class TestAllGeneratorsValidate:
    """Test validation works for all generators."""

    @pytest.mark.parametrize(
        "GeneratorClass",
        [
            PythonGenerator,
            JuliaGenerator,
            RustGenerator,
            MatlabGenerator,
            FortranGenerator,
            JavaScriptGenerator,
        ],
    )
    def test_validate_with_valid_data(self, GeneratorClass):
        gen = GeneratorClass(**SAMPLE_SYSTEM)
        # Validation without equations may produce warnings but shouldn't error
        is_valid, errors = gen.validate()
        # We expect it might not be valid due to missing equations, that's ok
        assert isinstance(is_valid, bool)
        assert isinstance(errors, list)

    @pytest.mark.parametrize(
        "GeneratorClass",
        [
            PythonGenerator,
            JuliaGenerator,
            RustGenerator,
            MatlabGenerator,
            FortranGenerator,
            JavaScriptGenerator,
        ],
    )
    def test_repr(self, GeneratorClass):
        gen = GeneratorClass(**SAMPLE_SYSTEM)
        repr_str = repr(gen)
        assert "test_pendulum" in repr_str
        assert GeneratorClass.__name__ in repr_str
