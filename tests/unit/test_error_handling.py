"""
Tests for Error Handling Module
==============================
"""

import logging
from unittest.mock import MagicMock

import pytest

from mechanics_dsl.error_handling import (
    CodegenError,
    CompilationError,
    ConfigurationError,
    ErrorContext,
    ErrorReporter,
    MechanicsDSLError,
    NumericalError,
    ParseError,
    ResourceError,
    SecurityError,
    SimulationError,
    error_boundary,
    fallback,
    handle_errors,
    report_error,
    retry,
)


class TestErrorHierarchy:
    """Tests for exception hierarchy."""

    def test_base_error(self):
        """Test base MechanicsDSLError."""
        error = MechanicsDSLError("Test error")
        assert str(error) == "Test error"
        assert error.error_code == "MDSL-0000"

    def test_error_with_context(self):
        """Test error with context."""
        context = ErrorContext(operation="test_op", component="test_comp")
        error = MechanicsDSLError("Test", context=context)

        assert error.context.operation == "test_op"
        assert error.context.component == "test_comp"

    def test_error_with_recovery_hint(self):
        """Test error with recovery hint."""
        error = MechanicsDSLError("Test").with_recovery_hint("Try again")

        assert error.context.recovery_hint == "Try again"
        assert "Try again" in str(error)

    def test_error_with_user_message(self):
        """Test error with user message."""
        error = MechanicsDSLError("Technical").with_user_message("Please check input")

        assert error.context.user_message == "Please check input"

    def test_error_with_metadata(self):
        """Test error with metadata."""
        error = MechanicsDSLError("Test").with_metadata(file="test.dsl", line=42)

        assert error.context.metadata["file"] == "test.dsl"
        assert error.context.metadata["line"] == 42

    def test_error_to_dict(self):
        """Test error serialization."""
        error = MechanicsDSLError("Test")
        d = error.to_dict()

        assert "error_type" in d
        assert "error_code" in d
        assert "message" in d
        assert "context" in d

    def test_specific_error_codes(self):
        """Test specific error types have unique codes."""
        errors = [
            ParseError("parse"),
            CompilationError("compile"),
            SimulationError("simulate"),
            NumericalError("numerical"),
            CodegenError("codegen"),
            ConfigurationError("config"),
            SecurityError("security"),
            ResourceError("resource"),
        ]

        codes = [e.error_code for e in errors]
        assert len(codes) == len(set(codes)), "Error codes should be unique"

    def test_error_chain(self):
        """Test error chaining."""
        cause = ValueError("Original error")
        error = MechanicsDSLError("Wrapped error", cause=cause)

        assert error.cause is cause
        assert error.__cause__ is cause


class TestErrorBoundary:
    """Tests for error_boundary context manager."""

    def test_successful_operation(self):
        """Test successful operation passes through."""
        with error_boundary("test", "component") as ctx:
            result = 42

        assert result == 42
        assert ctx.operation == "test"

    def test_error_wrapping(self):
        """Test that errors are wrapped."""
        with pytest.raises(MechanicsDSLError) as exc_info:
            with error_boundary("test_op", "test_comp"):
                raise ValueError("Original")

        assert exc_info.value.context.operation == "test_op"
        assert exc_info.value.context.component == "test_comp"

    def test_mdsl_error_passthrough(self):
        """Test that MechanicsDSL errors are augmented."""
        with pytest.raises(ParseError) as exc_info:
            with error_boundary("test_op", "test_comp"):
                raise ParseError("Parse failed")

        assert exc_info.value.context.operation == "test_op"


class TestHandleErrorsDecorator:
    """Tests for handle_errors decorator."""

    def test_successful_function(self):
        """Test successful function passes through."""

        @handle_errors()
        def good_func():
            return 42

        assert good_func() == 42

    def test_error_mapping(self):
        """Test error type mapping."""

        @handle_errors(error_map={ValueError: ConfigurationError})
        def bad_func():
            raise ValueError("Bad value")

        with pytest.raises(ConfigurationError):
            bad_func()

    def test_default_error(self):
        """Test default error type."""

        @handle_errors(default_error=SimulationError)
        def fail_func():
            raise RuntimeError("Runtime error")

        with pytest.raises(SimulationError):
            fail_func()


class TestRetryDecorator:
    """Tests for retry decorator."""

    def test_eventual_success(self):
        """Test retry with eventual success."""
        attempts = [0]

        @retry(max_attempts=3, delay_seconds=0.01)
        def flaky_func():
            attempts[0] += 1
            if attempts[0] < 3:
                raise ValueError("Not yet")
            return "success"

        result = flaky_func()
        assert result == "success"
        assert attempts[0] == 3

    def test_all_attempts_fail(self):
        """Test when all retries fail."""

        @retry(max_attempts=2, delay_seconds=0.01)
        def always_fails():
            raise ValueError("Always fails")

        with pytest.raises(ValueError):
            always_fails()

    def test_no_retry_on_success(self):
        """Test no retry when function succeeds."""
        attempts = [0]

        @retry(max_attempts=3, delay_seconds=0.01)
        def success_func():
            attempts[0] += 1
            return "success"

        result = success_func()
        assert result == "success"
        assert attempts[0] == 1


class TestFallbackDecorator:
    """Tests for fallback decorator."""

    def test_success_no_fallback(self):
        """Test successful function doesn't use fallback."""

        @fallback(default_value="fallback")
        def good_func():
            return "success"

        assert good_func() == "success"

    def test_error_uses_fallback(self):
        """Test error uses fallback value."""

        @fallback(default_value="fallback")
        def bad_func():
            raise ValueError("Error")

        assert bad_func() == "fallback"

    def test_none_fallback(self):
        """Test None as fallback."""

        @fallback(default_value=None)
        def fail_func():
            raise ValueError("Error")

        assert fail_func() is None


class TestErrorReporter:
    """Tests for ErrorReporter."""

    def test_report_error(self):
        """Test error reporting."""
        reporter = ErrorReporter()
        error = ValueError("Test error")

        reporter.report(error, context={"test": "value"})

        assert len(reporter.errors) == 1
        assert reporter.errors[0]["type"] == "ValueError"
        assert reporter.errors[0]["context"]["test"] == "value"

    def test_report_mdsl_error(self):
        """Test reporting MechanicsDSL error."""
        reporter = ErrorReporter()
        error = ParseError("Parse failed")

        reporter.report(error)

        assert reporter.errors[0]["error_code"] == ParseError.error_code

    def test_max_errors(self):
        """Test error limit."""
        reporter = ErrorReporter(max_errors=5)

        for i in range(10):
            reporter.report(ValueError(f"Error {i}"))

        assert len(reporter.errors) == 5
        assert "Error 9" in reporter.errors[-1]["message"]

    def test_summary(self):
        """Test error summary."""
        reporter = ErrorReporter()
        reporter.report(ValueError("val1"))
        reporter.report(ValueError("val2"))
        reporter.report(TypeError("type1"))

        summary = reporter.get_summary()

        assert summary["total"] == 3
        assert summary["by_type"]["ValueError"] == 2
        assert summary["by_type"]["TypeError"] == 1

    def test_clear(self):
        """Test clearing errors."""
        reporter = ErrorReporter()
        reporter.report(ValueError("test"))
        reporter.clear()

        assert len(reporter.errors) == 0
