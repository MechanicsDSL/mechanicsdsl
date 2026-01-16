"""
Unit Tests for ARM Code Generator

Tests ARM-specific code generation including:
- NEON SIMD intrinsics
- Cross-compilation targets
- Embedded/bare-metal code
- Makefile generation
"""

import os
import tempfile

import pytest
import sympy as sp

from mechanics_dsl.codegen.arm import ARMGenerator


class TestARMGenerator:
    """Tests for ARMGenerator class."""

    @pytest.fixture
    def simple_pendulum(self):
        """Create a simple pendulum generator."""
        theta = sp.Symbol("theta")
        theta_dot = sp.Symbol("theta_dot")
        g, l = sp.symbols("g l")

        return ARMGenerator(
            system_name="test_pendulum",
            coordinates=["theta"],
            parameters={"g": 9.81, "l": 1.0},
            initial_conditions={"theta": 0.5, "theta_dot": 0.0},
            equations={"theta_ddot": -g / l * sp.sin(theta)},
            target="raspberry_pi",
            use_neon=True,
            embedded=False,
        )

    @pytest.fixture
    def embedded_generator(self):
        """Create an embedded/bare-metal generator."""
        theta = sp.Symbol("theta")
        g, l = sp.symbols("g l")

        return ARMGenerator(
            system_name="embedded_pendulum",
            coordinates=["theta"],
            parameters={"g": 9.81, "l": 0.5},
            initial_conditions={"theta": 0.1, "theta_dot": 0.0},
            equations={"theta_ddot": -g / l * sp.sin(theta)},
            target="cortex_m",
            use_neon=False,
            embedded=True,
        )

    def test_initialization(self, simple_pendulum):
        """Test generator initialization."""
        assert simple_pendulum.system_name == "test_pendulum"
        assert simple_pendulum.target == "raspberry_pi"
        assert simple_pendulum.use_neon is True
        assert simple_pendulum.embedded is False
        assert simple_pendulum.arch == "aarch64"

    def test_target_flags_raspberry_pi(self, simple_pendulum):
        """Test Raspberry Pi target flags."""
        assert simple_pendulum.cc == "aarch64-linux-gnu-gcc"
        assert simple_pendulum.cxx == "aarch64-linux-gnu-g++"
        assert "-march=armv8-a+simd" in simple_pendulum.cflags

    def test_target_flags_cortex_m(self, embedded_generator):
        """Test Cortex-M target flags."""
        assert embedded_generator.cc == "arm-none-eabi-gcc"
        assert embedded_generator.arch == "thumb"
        assert "-mcpu=cortex-m4" in embedded_generator.cflags

    def test_target_name_property(self, simple_pendulum):
        """Test target_name property."""
        assert simple_pendulum.target_name == "arm_raspberry_pi"

    def test_file_extension_property(self, simple_pendulum):
        """Test file_extension property."""
        assert simple_pendulum.file_extension == ".c"

    def test_generate_standard_code(self, simple_pendulum):
        """Test standard ARM code generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = os.path.join(tmpdir, "pendulum_arm.c")
            result = simple_pendulum.generate(output_file)

            assert os.path.exists(result)

            with open(result, "r") as f:
                code = f.read()

            # Check for expected content
            assert "ARM-Optimized Simulation" in code
            assert "raspberry_pi" in code
            assert "const double g = 9.81" in code
            assert "const double l = 1.0" in code
            assert "rk4_step" in code
            assert "#define DIM" in code

    def test_generate_embedded_code(self, embedded_generator):
        """Test embedded/bare-metal code generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = os.path.join(tmpdir, "embedded_arm.c")
            result = embedded_generator.generate(output_file)

            assert os.path.exists(result)

            with open(result, "r") as f:
                code = f.read()

            # Check for embedded-specific content
            assert "bare-metal" in code or "Embedded" in code
            assert "arm_sinf" in code  # Custom math functions
            assert "arm_cosf" in code
            assert "physics_step" in code

    def test_generate_neon_intrinsics(self, simple_pendulum):
        """Test NEON SIMD intrinsics generation."""
        neon_code = simple_pendulum._generate_neon_intrinsics()

        assert "#ifdef __ARM_NEON" in neon_code
        assert "#include <arm_neon.h>" in neon_code
        assert "float32x4_t" in neon_code
        assert "neon_sin_f32" in neon_code
        assert "neon_cos_f32" in neon_code

    def test_generate_makefile(self, simple_pendulum):
        """Test Makefile generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            makefile = simple_pendulum.generate_makefile(tmpdir)

            assert os.path.exists(makefile)

            with open(makefile, "r") as f:
                content = f.read()

            assert "CC = aarch64-linux-gnu-gcc" in content
            assert "native:" in content
            assert "cross:" in content
            assert "clean:" in content

    def test_generate_project(self, simple_pendulum):
        """Test complete project generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            files = simple_pendulum.generate_project(tmpdir)

            assert "source" in files
            assert "makefile" in files
            assert "readme" in files

            assert os.path.exists(files["source"])
            assert os.path.exists(files["makefile"])
            assert os.path.exists(files["readme"])

            # Check README content
            with open(files["readme"], "r") as f:
                readme = f.read()

            assert "test_pendulum" in readme
            assert "Raspberry Pi" in readme or "raspberry_pi" in readme


class TestARMGeneratorTargets:
    """Test different ARM target configurations."""

    @pytest.mark.parametrize(
        "target,expected_arch",
        [
            ("raspberry_pi", "aarch64"),
            ("jetson", "aarch64"),
            ("cortex_m", "thumb"),
        ],
    )
    def test_target_architectures(self, target, expected_arch):
        """Test architecture detection for different targets."""
        gen = ARMGenerator(
            system_name="test",
            coordinates=["x"],
            parameters={"k": 1.0},
            initial_conditions={"x": 0.0, "x_dot": 0.0},
            equations={"x_ddot": sp.Symbol("x")},
            target=target,
        )
        assert gen.arch == expected_arch

    def test_unknown_target_defaults(self):
        """Test that unknown targets get default settings."""
        gen = ARMGenerator(
            system_name="test",
            coordinates=["x"],
            parameters={"k": 1.0},
            initial_conditions={"x": 0.0, "x_dot": 0.0},
            equations={"x_ddot": sp.Symbol("x")},
            target="unknown_platform",
        )
        assert gen.arch == "arm"
        assert gen.cc == "gcc"


class TestARMGeneratorIntegration:
    """Integration tests for ARM code generator."""

    def test_double_pendulum(self):
        """Test double pendulum system generation."""
        theta1, theta2 = sp.symbols("theta1 theta2")
        g, l1, l2, m1, m2 = sp.symbols("g l1 l2 m1 m2")

        gen = ARMGenerator(
            system_name="double_pendulum",
            coordinates=["theta1", "theta2"],
            parameters={"g": 9.81, "l1": 1.0, "l2": 1.0, "m1": 1.0, "m2": 1.0},
            initial_conditions={"theta1": 0.5, "theta1_dot": 0.0, "theta2": 0.5, "theta2_dot": 0.0},
            equations={"theta1_ddot": -g * sp.sin(theta1), "theta2_ddot": -g * sp.sin(theta2)},
            target="raspberry_pi",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            files = gen.generate_project(tmpdir)

            # Verify all files exist
            for path in files.values():
                assert os.path.exists(path), f"Missing file: {path}"
