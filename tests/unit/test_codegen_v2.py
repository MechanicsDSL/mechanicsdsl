"""
Unit Tests for Enhanced Code Generators (C++, Rust, CUDA)

Tests new v2.0.0 features:
- CMake generation for C++
- Cargo.toml generation for Rust
- cuBLAS helpers for CUDA
- Batch simulation for CUDA
- Project generation for all
"""

import os
import tempfile

import pytest
import sympy as sp

from mechanics_dsl.codegen.cpp import CppGenerator
from mechanics_dsl.codegen.cuda import CudaGenerator
from mechanics_dsl.codegen.rust import RustGenerator


class TestCppGeneratorV2:
    """Tests for new C++ generator features in v2.0.0."""

    @pytest.fixture
    def cpp_generator(self):
        """Create a C++ generator for testing."""
        theta = sp.Symbol("theta")
        g, l = sp.symbols("g l")

        return CppGenerator(
            system_name="pendulum",
            coordinates=["theta"],
            parameters={"g": 9.81, "l": 1.0},
            initial_conditions={"theta": 0.5, "theta_dot": 0.0},
            equations={"theta_ddot": -g / l * sp.sin(theta)},
        )

    def test_generate_cmake(self, cpp_generator):
        """Test CMakeLists.txt generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cmake_path = cpp_generator.generate_cmake(tmpdir)

            assert os.path.exists(cmake_path)
            assert cmake_path.endswith("CMakeLists.txt")

            with open(cmake_path, "r") as f:
                content = f.read()

            # Check for expected CMake content
            assert "cmake_minimum_required" in content
            assert "project(pendulum" in content
            assert "CMAKE_CXX_STANDARD 17" in content
            assert "add_executable" in content

            # Check for ARM detection
            assert "CMAKE_SYSTEM_PROCESSOR" in content
            assert "arm" in content.lower() or "ARM" in content

    def test_generate_cmake_arm_detection(self, cpp_generator):
        """Test ARM architecture detection in CMake."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cmake_path = cpp_generator.generate_cmake(tmpdir)

            with open(cmake_path, "r") as f:
                content = f.read()

            # Should have ARM-specific options
            assert "NEON" in content or "aarch64" in content.lower()
            assert "-march=" in content or "march" in content.lower()

    def test_generate_project(self, cpp_generator):
        """Test complete project generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            files = cpp_generator.generate_project(tmpdir)

            assert "cpp" in files
            assert "cmake" in files
            assert "readme" in files

            # Verify all files exist
            for name, path in files.items():
                assert os.path.exists(path), f"Missing {name}: {path}"

            # Check README content
            with open(files["readme"], "r") as f:
                readme = f.read()

            assert "Build Instructions" in readme
            assert "cmake" in readme.lower()
            assert "Raspberry Pi" in readme or "ARM" in readme


class TestRustGeneratorV2:
    """Tests for new Rust generator features in v2.0.0."""

    @pytest.fixture
    def rust_generator(self):
        """Create a Rust generator for testing."""
        x = sp.Symbol("x")
        k, m = sp.symbols("k m")

        return RustGenerator(
            system_name="oscillator",
            coordinates=["x"],
            parameters={"k": 10.0, "m": 1.0},
            initial_conditions={"x": 1.0, "x_dot": 0.0},
            equations={"x_ddot": -k / m * x},
        )

    def test_generate_cargo_toml_standard(self, rust_generator):
        """Test standard Cargo.toml generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cargo_path = rust_generator.generate_cargo_toml(tmpdir, embedded=False)

            assert os.path.exists(cargo_path)

            with open(cargo_path, "r") as f:
                content = f.read()

            # Check for expected Cargo.toml content
            assert "[package]" in content
            assert 'name = "oscillator"' in content
            assert 'edition = "2021"' in content
            assert "[profile.release]" in content
            assert "lto = true" in content

    def test_generate_cargo_toml_embedded(self, rust_generator):
        """Test embedded Cargo.toml generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cargo_path = rust_generator.generate_cargo_toml(tmpdir, embedded=True)

            with open(cargo_path, "r") as f:
                content = f.read()

            # Check for embedded-specific content
            assert "embedded" in content.lower() or "no_std" in content
            assert 'panic = "abort"' in content
            assert "libm" in content  # Math library for no_std

    def test_generate_project_standard(self, rust_generator):
        """Test standard Rust project generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            files = rust_generator.generate_project(tmpdir, embedded=False)

            assert "main" in files
            assert "cargo" in files
            assert "readme" in files

            # Check for src directory structure
            assert "src" in files["main"]
            assert "main.rs" in files["main"]

    def test_generate_project_embedded(self, rust_generator):
        """Test embedded Rust project generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            files = rust_generator.generate_project(tmpdir, embedded=True)

            # Check README mentions embedded
            with open(files["readme"], "r") as f:
                readme = f.read()

            assert "embedded" in readme.lower() or "Cortex" in readme


class TestCudaGeneratorV2:
    """Tests for new CUDA generator features in v2.0.0."""

    @pytest.fixture
    def cuda_generator(self):
        """Create a CUDA generator for testing."""
        theta = sp.Symbol("theta")
        g, l = sp.symbols("g l")

        return CudaGenerator(
            system_name="pendulum_cuda",
            coordinates=["theta"],
            parameters={"g": 9.81, "l": 1.0},
            initial_conditions={"theta": 0.5, "theta_dot": 0.0},
            equations={"theta_ddot": -g / l * sp.sin(theta)},
            use_cublas=True,
            batch_size=1000,
            compute_capability="70",
        )

    def test_initialization_new_params(self, cuda_generator):
        """Test new initialization parameters."""
        assert cuda_generator.use_cublas is True
        assert cuda_generator.batch_size == 1000
        assert cuda_generator.compute_capability == "70"

    def test_generate_cublas_helpers(self, cuda_generator):
        """Test cuBLAS helper function generation."""
        helpers = cuda_generator._generate_cublas_helpers()

        assert "#ifdef USE_CUBLAS" in helpers
        assert "cublas_v2.h" in helpers
        assert "cublasHandle_t" in helpers
        assert "cublas_init" in helpers
        assert "cublas_destroy" in helpers

        # Check for BLAS functions
        assert "cublas_gemv" in helpers  # Matrix-vector multiply
        assert "cublas_dot" in helpers  # Dot product
        assert "cublas_nrm2" in helpers  # Vector norm
        assert "cublas_scal" in helpers  # Vector scaling
        assert "cublas_axpy" in helpers  # Vector addition
        assert "cublas_gemm" in helpers  # Matrix-matrix multiply

    def test_generate_batch_simulation(self, cuda_generator):
        """Test batch simulation code generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            batch_file = cuda_generator.generate_batch_simulation(tmpdir)

            assert os.path.exists(batch_file)
            assert batch_file.endswith("_batch.cu")

            with open(batch_file, "r") as f:
                content = f.read()

            # Check for batch-specific content
            assert "BATCH_SIZE = 1000" in content
            assert "batch_rk4_kernel" in content
            assert "parallel simulations" in content

            # Check for Monte Carlo / parameter sweep features
            assert "random" in content.lower()
            assert "normal_distribution" in content


class TestCodegenIntegration:
    """Integration tests across all code generators."""

    def test_all_generators_produce_output(self):
        """Test that all generators successfully produce files."""
        from mechanics_dsl.codegen import ARMGenerator, CppGenerator, CudaGenerator, RustGenerator

        theta = sp.Symbol("theta")
        g, l = sp.symbols("g l")

        params = {
            "system_name": "test_system",
            "coordinates": ["theta"],
            "parameters": {"g": 9.81, "l": 1.0},
            "initial_conditions": {"theta": 0.5, "theta_dot": 0.0},
            "equations": {"theta_ddot": -g / l * sp.sin(theta)},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            # Test each generator
            generators = [
                ("cpp", CppGenerator(**params)),
                ("rust", RustGenerator(**params)),
                ("arm", ARMGenerator(**params, target="raspberry_pi")),
            ]

            for name, gen in generators:
                subdir = os.path.join(tmpdir, name)
                os.makedirs(subdir)

                if hasattr(gen, "generate_project"):
                    files = gen.generate_project(subdir)
                    for path in files.values():
                        assert os.path.exists(path), f"{name} failed: {path}"
                else:
                    output = gen.generate(os.path.join(subdir, "output"))
                    assert os.path.exists(output), f"{name} failed"
