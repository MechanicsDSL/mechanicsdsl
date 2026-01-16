"""
Security Tests - Code Injection Prevention
==========================================

Tests to verify the security module prevents code injection attacks.
"""

import pytest
import tempfile
import os

from mechanics_dsl.security import (
    validate_identifier,
    validate_string,
    validate_dsl_code,
    validate_path,
    InputValidationError,
    InjectionError,
    PathTraversalError,
)


class TestIdentifierValidation:
    """Tests for identifier validation."""
    
    def test_valid_identifiers(self):
        """Test that valid identifiers are accepted."""
        valid = ['x', 'y1', 'theta', 'my_var', '_private', 'CamelCase']
        for name in valid:
            assert validate_identifier(name) == name
    
    def test_empty_identifier(self):
        """Test that empty identifiers are rejected."""
        with pytest.raises(InputValidationError, match="Empty"):
            validate_identifier('')
    
    def test_identifier_with_spaces(self):
        """Test that identifiers with spaces are rejected."""
        with pytest.raises(InputValidationError):
            validate_identifier('my var')
    
    def test_identifier_starting_with_number(self):
        """Test that identifiers starting with numbers are rejected."""
        with pytest.raises(InputValidationError):
            validate_identifier('1var')
    
    def test_identifier_with_special_chars(self):
        """Test that identifiers with special characters are rejected."""
        invalid = ['my-var', 'var.name', 'var@home', 'var$1']
        for name in invalid:
            with pytest.raises(InputValidationError):
                validate_identifier(name)
    
    def test_python_keywords_rejected(self):
        """Test that Python keywords are rejected as identifiers."""
        keywords = ['if', 'else', 'for', 'while', 'import', 'class', 'def']
        for kw in keywords:
            with pytest.raises(InputValidationError, match="keyword"):
                validate_identifier(kw)
    
    def test_identifier_too_long(self):
        """Test that overly long identifiers are rejected."""
        with pytest.raises(InputValidationError, match="too long"):
            validate_identifier('a' * 300)


class TestStringValidation:
    """Tests for string validation."""
    
    def test_valid_strings(self):
        """Test that valid strings are accepted."""
        assert validate_string("hello") == "hello"
        assert validate_string("") == ""
    
    def test_null_byte_rejected(self):
        """Test that strings with null bytes are rejected."""
        with pytest.raises(InputValidationError, match="Null"):
            validate_string("hello\x00world")
    
    def test_string_too_long(self):
        """Test that overly long strings are rejected."""
        with pytest.raises(InputValidationError, match="too long"):
            validate_string("a" * 200000)
    
    def test_custom_max_length(self):
        """Test custom max length enforcement."""
        with pytest.raises(InputValidationError):
            validate_string("hello world", max_length=5)
    
    def test_non_string_rejected(self):
        """Test that non-strings are rejected."""
        with pytest.raises(InputValidationError, match="must be a string"):
            validate_string(123)


class TestDSLCodeValidation:
    """Tests for DSL code validation and injection prevention."""
    
    def test_valid_dsl_code(self):
        """Test that valid DSL code is accepted."""
        code = r"""
        \system{pendulum}
        \defvar{theta}{Angle}{rad}
        \parameter{g}{9.81}{m/s^2}
        """
        assert validate_dsl_code(code) == code
    
    def test_empty_code_rejected(self):
        """Test that empty code is rejected."""
        with pytest.raises(InputValidationError):
            validate_dsl_code('')
    
    def test_eval_injection_blocked(self):
        """Test that eval() injection is blocked."""
        malicious_codes = [
            "eval('os.system(\"rm -rf /\")')",
            "EVAL ( 'code' )",
            "  eval  (  'code'  )  ",
        ]
        for code in malicious_codes:
            with pytest.raises(InjectionError, match="eval"):
                validate_dsl_code(code)
    
    def test_exec_injection_blocked(self):
        """Test that exec() injection is blocked."""
        with pytest.raises(InjectionError, match="exec"):
            validate_dsl_code("exec('import os')")
    
    def test_import_injection_blocked(self):
        """Test that __import__ injection is blocked."""
        with pytest.raises(InjectionError, match="__import__"):
            validate_dsl_code("__import__('os').system('ls')")
    
    def test_subprocess_injection_blocked(self):
        """Test that subprocess injection is blocked."""
        with pytest.raises(InjectionError):
            validate_dsl_code("subprocess.Popen(['rm', '-rf', '/'])")
    
    def test_os_system_injection_blocked(self):
        """Test that os.system injection is blocked."""
        with pytest.raises(InjectionError):
            validate_dsl_code("os.system('rm -rf /')")
    
    def test_pickle_load_blocked(self):
        """Test that pickle.load is blocked."""
        with pytest.raises(InjectionError):
            validate_dsl_code("pickle.load(open('malicious.pkl', 'rb'))")
    
    def test_open_blocked(self):
        """Test that raw open() is blocked in DSL."""
        with pytest.raises(InjectionError, match="open"):
            validate_dsl_code("open('/etc/passwd', 'r').read()")
    
    def test_shell_true_blocked(self):
        """Test that shell=True is blocked."""
        with pytest.raises(InjectionError, match="shell"):
            validate_dsl_code("run(cmd, shell = True)")
    
    def test_code_too_large(self):
        """Test that oversized code is rejected."""
        large_code = "\\system{test}\n" + "x" * (2 * 1024 * 1024)
        with pytest.raises(InputValidationError, match="too large"):
            validate_dsl_code(large_code)


class TestPathValidation:
    """Tests for path validation and traversal prevention."""
    
    def test_valid_paths(self):
        """Test that valid paths are accepted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            valid_path = os.path.join(tmpdir, "test.txt")
            # Create the file
            with open(valid_path, 'w') as f:
                f.write("test")
            
            result = validate_path(valid_path, must_exist=True)
            assert result.exists()
    
    def test_path_traversal_dotdot(self):
        """Test that .. path traversal is blocked."""
        with pytest.raises(PathTraversalError, match="traversal"):
            validate_path("../../../etc/passwd")
    
    def test_path_traversal_with_base_dir(self):
        """Test that escape from base directory is blocked."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(PathTraversalError, match="escapes"):
                validate_path(
                    os.path.join(tmpdir, "..", "outside"),
                    base_dir=tmpdir
                )
    
    def test_null_byte_injection(self):
        """Test that null byte path injection is blocked."""
        with pytest.raises(PathTraversalError, match="Null"):
            validate_path("/safe/path\x00/etc/passwd")
    
    def test_path_too_long(self):
        """Test that overly long paths are rejected."""
        with pytest.raises(InputValidationError, match="too long"):
            validate_path("a" * 5000)
    
    def test_absolute_path_restriction(self):
        """Test that absolute paths can be blocked."""
        # Note: Windows paths like C:\ are absolute, Unix paths start with /
        import os
        if os.name == 'nt':
            with pytest.raises(InputValidationError, match="Absolute"):
                validate_path("C:\\absolute\\path", allow_absolute=False)
        else:
            with pytest.raises(InputValidationError, match="Absolute"):
                validate_path("/absolute/path", allow_absolute=False)
    
    def test_nonexistent_path_when_required(self):
        """Test that nonexistent paths are rejected when required."""
        with pytest.raises(InputValidationError, match="does not exist"):
            validate_path("/this/path/does/not/exist", must_exist=True)


class TestSecurityIntegration:
    """Integration tests for security measures."""
    
    def test_combined_attack_vectors(self):
        """Test that combined attack vectors are blocked."""
        attacks = [
            # Path + injection - should trigger eval detection
            "eval('code') in ../../../tmp",
            # exec injection
            "exec(open('/etc/passwd'))",
        ]
        
        for attack in attacks:
            # Should raise some security exception
            with pytest.raises((InputValidationError, PathTraversalError, InjectionError)):
                validate_dsl_code(attack)
    
    def test_case_insensitive_detection(self):
        """Test that pattern detection is case-insensitive."""
        variations = [
            "EVAL('code')",
            "Eval('code')",
            "eVaL('code')",
        ]
        
        for code in variations:
            with pytest.raises(InjectionError):
                validate_dsl_code(code)
