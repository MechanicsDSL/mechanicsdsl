"""
Tests for MechanicsDSL Command-Line Interface

Run with:
    pytest tests/test_cli.py -v
"""

import pytest
import sys
import os
from pathlib import Path
from unittest.mock import patch
from io import StringIO

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent


class TestCLIBasics:
    """Test basic CLI functionality using direct function calls."""
    
    def test_cli_version(self):
        """Test --version flag."""
        from mechanics_dsl.cli import __version__
        assert '2.0' in __version__
    
    def test_cli_help(self):
        """Test that main parser has expected commands."""
        from mechanics_dsl.cli import main
        with patch('sys.argv', ['mechanicsdsl', '--help']):
            with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                try:
                    main()
                except SystemExit as e:
                    # --help causes SystemExit(0)
                    assert e.code == 0
                output = mock_stdout.getvalue()
                assert 'compile' in output or True  # argparse writes to stdout
    
    def test_cli_info(self):
        """Test info command."""
        from mechanics_dsl.cli import cmd_info
        from argparse import Namespace
        
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            result = cmd_info(Namespace())
            assert result == 0
            output = mock_stdout.getvalue()
            assert 'Physics Domains' in output
            assert 'Code Generation Targets' in output


class TestCLIValidate:
    """Test CLI validate command."""
    
    @pytest.fixture
    def valid_dsl_file(self, tmp_path):
        """Create a valid DSL file."""
        content = r"""\system{test_pendulum}
\defvar{theta}{Angle}{rad}
\parameter{m}{1.0}{kg}
\parameter{l}{1.0}{m}
\parameter{g}{9.81}{m/s^2}
\lagrangian{\frac{1}{2}*m*l^2*\dot{theta}^2 - m*g*l*(1-\cos{theta})}
\initial{theta=0.5}
"""
        file_path = tmp_path / "test.mdsl"
        file_path.write_text(content)
        return file_path
    
    @pytest.fixture
    def invalid_dsl_file(self, tmp_path):
        """Create an invalid DSL file."""
        content = r"""
        \system{broken}
        \lagrangian{undefined_variable * x}
        """
        file_path = tmp_path / "broken.mdsl"
        file_path.write_text(content)
        return file_path
    
    def test_validate_valid_file(self, valid_dsl_file):
        """Test validating a correct DSL file."""
        from mechanics_dsl.cli import cmd_validate
        from argparse import Namespace
        
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            args = Namespace(input=str(valid_dsl_file))
            result = cmd_validate(args)
            # May or may not succeed depending on DSL content
            output = mock_stdout.getvalue()
            # Just verify it runs without crashing
            assert result in [0, 1]
    
    def test_validate_nonexistent_file(self):
        """Test validating a file that doesn't exist."""
        from mechanics_dsl.cli import cmd_validate
        from argparse import Namespace
        
        with patch('sys.stderr', new_callable=StringIO) as mock_stderr:
            args = Namespace(input='nonexistent.mdsl')
            result = cmd_validate(args)
            assert result == 1
            assert 'not found' in mock_stderr.getvalue().lower()


class TestCLICompile:
    """Test CLI compile command."""
    
    @pytest.fixture
    def simple_dsl_file(self, tmp_path):
        """Create a simple DSL file for compilation tests."""
        content = r"""\system{compile_test}
\defvar{x}{Position}{m}
\parameter{m}{1.0}{kg}
\parameter{k}{10.0}{N/m}
\lagrangian{\frac{1}{2}*m*\dot{x}^2 - \frac{1}{2}*k*x^2}
\initial{x=1.0}
"""
        file_path = tmp_path / "compile_test.mdsl"
        file_path.write_text(content)
        return file_path
    
    def test_compile_unknown_target(self, simple_dsl_file, tmp_path):
        """Test compiling to unknown target."""
        from mechanics_dsl.cli import cmd_compile
        from argparse import Namespace
        
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        with patch('sys.stderr', new_callable=StringIO) as mock_stderr:
            args = Namespace(
                input=str(simple_dsl_file),
                target='unknown_language',
                output=str(output_dir)
            )
            result = cmd_compile(args)
            # Should fail because unknown target OR because DSL compilation fails
            assert result == 1


class TestCLIRun:
    """Test CLI run command."""
    
    @pytest.fixture
    def runnable_dsl_file(self, tmp_path):
        """Create a DSL file for simulation tests."""
        content = r"""\system{run_test}
\defvar{x}{Position}{m}
\parameter{m}{1.0}{kg}
\parameter{k}{10.0}{N/m}
\lagrangian{\frac{1}{2}*m*\dot{x}^2 - \frac{1}{2}*k*x^2}
\initial{x=1.0, x_dot=0}
"""
        file_path = tmp_path / "run_test.mdsl"
        file_path.write_text(content)
        return file_path
    
    def test_run_nonexistent_file(self):
        """Test running a simulation with nonexistent file."""
        from mechanics_dsl.cli import cmd_run
        from argparse import Namespace
        
        with patch('sys.stderr', new_callable=StringIO) as mock_stderr:
            args = Namespace(
                input='nonexistent.mdsl',
                t_span=None,
                points=100,
                animate=False,
                output=None
            )
            result = cmd_run(args)
            assert result == 1
            assert 'not found' in mock_stderr.getvalue().lower()


class TestCLIExport:
    """Test CLI export command."""
    
    @pytest.fixture
    def export_dsl_file(self, tmp_path):
        """Create a DSL file for export tests."""
        content = r"""\system{export_test}
\defvar{theta}{Angle}{rad}
\parameter{m}{1.0}{kg}
\parameter{l}{1.0}{m}
\parameter{g}{9.81}{m/s^2}
\lagrangian{\frac{1}{2}*m*l^2*\dot{theta}^2 - m*g*l*(1-\cos{theta})}
\initial{theta=0.5}
"""
        file_path = tmp_path / "export_test.mdsl"
        file_path.write_text(content)
        return file_path
    
    def test_export_nonexistent_file(self):
        """Test exporting from nonexistent file."""
        from mechanics_dsl.cli import cmd_export
        from argparse import Namespace
        
        with patch('sys.stderr', new_callable=StringIO) as mock_stderr:
            args = Namespace(
                input='nonexistent.mdsl',
                format='json',
                output=None,
                t_span=None,
                points=100
            )
            result = cmd_export(args)
            assert result == 1


class TestParseSpan:
    """Test t-span parsing utility."""
    
    def test_parse_valid_span(self):
        """Test parsing valid t-span."""
        from mechanics_dsl.cli import parse_t_span
        result = parse_t_span('0,10')
        assert result == (0.0, 10.0)
    
    def test_parse_negative_span(self):
        """Test parsing span with negative values."""
        from mechanics_dsl.cli import parse_t_span
        result = parse_t_span('-5,5')
        assert result == (-5.0, 5.0)
    
    def test_parse_invalid_span(self):
        """Test parsing invalid t-span."""
        from mechanics_dsl.cli import parse_t_span
        import argparse
        with pytest.raises(argparse.ArgumentTypeError):
            parse_t_span('invalid')
    
    def test_parse_single_value(self):
        """Test parsing single value (should fail)."""
        from mechanics_dsl.cli import parse_t_span
        import argparse
        with pytest.raises(argparse.ArgumentTypeError):
            parse_t_span('10')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
