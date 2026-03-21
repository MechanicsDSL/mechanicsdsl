"""Tests for mechanics_dsl.validators module."""

import pytest

from mechanics_dsl.validators import (
    PYDANTIC_AVAILABLE,
    CodegenConfig,
    CoordinateConfig,
    ParameterConfig,
    ServerConfig,
    SimulationConfig,
    ValidationError,
    validate_coordinate,
    validate_parameter,
    validate_simulation_config,
    wrap_validation_error,
)


class TestSimulationConfig:
    """Tests for SimulationConfig validation."""

    def test_defaults(self):
        config = SimulationConfig()
        assert config.t_start == 0.0
        assert config.t_end == 10.0
        assert config.dt == 0.001
        assert config.num_points == 1000
        assert config.method == "RK45"

    def test_custom_values(self):
        config = SimulationConfig(t_start=0.0, t_end=20.0, dt=0.01, num_points=500, method="RK23")
        assert config.t_end == 20.0
        assert config.dt == 0.01
        assert config.num_points == 500
        assert config.method == "RK23"

    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic required for validation")
    def test_t_end_must_be_after_t_start(self):
        with pytest.raises(Exception):
            SimulationConfig(t_start=5.0, t_end=1.0)

    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic required for validation")
    def test_invalid_method(self):
        with pytest.raises(Exception):
            SimulationConfig(method="invalid_method")

    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic required for validation")
    def test_valid_methods(self):
        for method in ["RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA", "euler", "rk4"]:
            config = SimulationConfig(method=method)
            assert config.method == method


class TestCoordinateConfig:
    """Tests for CoordinateConfig validation."""

    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic required for validation")
    def test_valid_coordinate(self):
        config = CoordinateConfig(name="theta", unit="rad", initial_value=0.1)
        assert config.name == "theta"
        assert config.unit == "rad"
        assert config.initial_value == 0.1

    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic required for validation")
    def test_invalid_bounds(self):
        with pytest.raises(Exception):
            CoordinateConfig(name="x", bounds=(10.0, 5.0))

    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic required for validation")
    def test_valid_bounds(self):
        config = CoordinateConfig(name="x", bounds=(-1.0, 1.0))
        assert config.bounds == (-1.0, 1.0)


class TestParameterConfig:
    """Tests for ParameterConfig validation."""

    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic required for validation")
    def test_valid_parameter(self):
        config = ParameterConfig(name="g", value=9.81, unit="m/s^2")
        assert config.name == "g"
        assert config.value == 9.81

    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic required for validation")
    def test_value_below_min(self):
        with pytest.raises(Exception):
            ParameterConfig(name="g", value=-1.0, min_value=0.0)

    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic required for validation")
    def test_value_above_max(self):
        with pytest.raises(Exception):
            ParameterConfig(name="g", value=100.0, max_value=50.0)

    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic required for validation")
    def test_value_within_bounds(self):
        config = ParameterConfig(name="g", value=9.81, min_value=0.0, max_value=100.0)
        assert config.value == 9.81


class TestCodegenConfig:
    """Tests for CodegenConfig validation."""

    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic required for validation")
    def test_valid_target(self):
        config = CodegenConfig(target="cpp", system_name="test")
        assert config.target == "cpp"

    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic required for validation")
    def test_case_insensitive_target(self):
        config = CodegenConfig(target="CPP", system_name="test")
        assert config.target == "cpp"

    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic required for validation")
    def test_invalid_target(self):
        with pytest.raises(Exception):
            CodegenConfig(target="brainfuck", system_name="test")

    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic required for validation")
    def test_all_valid_targets(self):
        valid_targets = [
            "cpp", "rust", "cuda", "arm", "julia",
            "matlab", "fortran", "javascript", "wasm", "arduino", "python",
        ]
        for target in valid_targets:
            config = CodegenConfig(target=target, system_name="test")
            assert config.target == target

    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic required for validation")
    def test_optimization_level_bounds(self):
        config = CodegenConfig(target="cpp", system_name="test", optimization_level=3)
        assert config.optimization_level == 3

        with pytest.raises(Exception):
            CodegenConfig(target="cpp", system_name="test", optimization_level=5)


class TestServerConfig:
    """Tests for ServerConfig validation."""

    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic required for validation")
    def test_defaults(self):
        config = ServerConfig()
        assert config.port == 8000
        assert config.workers == 4
        assert config.enable_docs is True

    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic required for validation")
    def test_valid_port(self):
        config = ServerConfig(port=3000)
        assert config.port == 3000

    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic required for validation")
    def test_invalid_port(self):
        with pytest.raises(Exception):
            ServerConfig(port=0)
        with pytest.raises(Exception):
            ServerConfig(port=70000)


class TestValidationFunctions:
    """Tests for standalone validation functions."""

    def test_validate_simulation_config(self):
        config = validate_simulation_config({"t_end": 20.0, "method": "RK45"})
        assert config.t_end == 20.0

    def test_validate_coordinate(self):
        coord = validate_coordinate("theta", 0.5, velocity=1.0)
        assert coord.name == "theta"
        assert coord.initial_value == 0.5
        assert coord.initial_velocity == 1.0

    def test_validate_parameter_valid(self):
        param = validate_parameter("g", 9.81, min_value=0.0, max_value=100.0)
        assert param.name == "g"
        assert param.value == 9.81

    def test_validate_parameter_below_min(self):
        with pytest.raises((ValueError, Exception)):
            validate_parameter("g", -1.0, min_value=0.0)

    def test_validate_parameter_above_max(self):
        with pytest.raises((ValueError, Exception)):
            validate_parameter("g", 200.0, max_value=100.0)


class TestValidationError:
    """Tests for custom ValidationError class."""

    def test_basic_error(self):
        err = ValidationError("something went wrong")
        assert str(err) == "something went wrong"
        assert err.field is None
        assert err.errors == []

    def test_error_with_field(self):
        err = ValidationError("bad value", field="t_end", value=-1)
        assert err.field == "t_end"
        assert err.value == -1

    def test_to_dict(self):
        err = ValidationError("bad", field="x", value=42, errors=[{"loc": "x"}])
        d = err.to_dict()
        assert d["message"] == "bad"
        assert d["field"] == "x"
        assert d["value"] == 42
        assert len(d["errors"]) == 1

    def test_is_exception(self):
        with pytest.raises(ValidationError):
            raise ValidationError("test")


class TestWrapValidationError:
    """Tests for wrap_validation_error decorator."""

    def test_wraps_normal_function(self):
        @wrap_validation_error
        def good_func():
            return 42

        assert good_func() == 42

    def test_passes_through_non_validation_errors(self):
        @wrap_validation_error
        def bad_func():
            raise ValueError("not a validation error")

        with pytest.raises(ValueError):
            bad_func()
