"""
Tests for MechanicsDSL Command-Line Interface

Run with:
    pytest tests/test_cli.py -v
"""

import pytest
import subprocess
import sys
import os
import json
import tempfile
from pathlib import Path

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent


class TestCLIBasics:
    """Test basic CLI functionality."""
    
    def test_cli_version(self):
        """Test --version flag."""
        result = subprocess.run(
            [sys.executable, '-m', 'mechanics_dsl.cli', '--version'],
            capture_output=True, text=True
        )
        assert result.returncode == 0
        assert 'MechanicsDSL' in result.stdout
        assert '2.0' in result.stdout
    
    def test_cli_help(self):
        """Test --help flag."""
        result = subprocess.run(
            [sys.executable, '-m', 'mechanics_dsl.cli', '--help'],
            capture_output=True, text=True
        )
        assert result.returncode == 0
        assert 'compile' in result.stdout
        assert 'run' in result.stdout
        assert 'export' in result.stdout
        assert 'validate' in result.stdout
        assert 'info' in result.stdout
    
    def test_cli_info(self):
        """Test info command."""
        result = subprocess.run(
            [sys.executable, '-m', 'mechanics_dsl.cli', 'info'],
            capture_output=True, text=True
        )
        assert result.returncode == 0
        assert 'Physics Domains' in result.stdout
        assert 'Code Generation Targets' in result.stdout


class TestCLIValidate:
    """Test CLI validate command."""
    
    @pytest.fixture
    def valid_dsl_file(self, tmp_path):
        """Create a valid DSL file."""
        content = r"""
        \system{test_pendulum}
        \defvar{theta}{rad}
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
        result = subprocess.run(
            [sys.executable, '-m', 'mechanics_dsl.cli', 'validate', str(valid_dsl_file)],
            capture_output=True, text=True
        )
        assert result.returncode == 0
        assert 'Valid' in result.stdout or '✓' in result.stdout
    
    def test_validate_nonexistent_file(self):
        """Test validating a file that doesn't exist."""
        result = subprocess.run(
            [sys.executable, '-m', 'mechanics_dsl.cli', 'validate', 'nonexistent.mdsl'],
            capture_output=True, text=True
        )
        assert result.returncode != 0
        assert 'not found' in result.stderr.lower() or 'error' in result.stderr.lower()


class TestCLICompile:
    """Test CLI compile command."""
    
    @pytest.fixture
    def simple_dsl_file(self, tmp_path):
        """Create a simple DSL file for compilation tests."""
        content = r"""
        \system{compile_test}
        \defvar{x}{m}
        \parameter{m}{1.0}{kg}
        \parameter{k}{10.0}{N/m}
        \lagrangian{\frac{1}{2}*m*\dot{x}^2 - \frac{1}{2}*k*x^2}
        \initial{x=1.0}
        """
        file_path = tmp_path / "compile_test.mdsl"
        file_path.write_text(content)
        return file_path
    
    def test_compile_to_cpp(self, simple_dsl_file, tmp_path):
        """Test compiling to C++."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        result = subprocess.run(
            [sys.executable, '-m', 'mechanics_dsl.cli', 
             'compile', str(simple_dsl_file), 
             '--target', 'cpp', 
             '--output', str(output_dir)],
            capture_output=True, text=True
        )
        
        assert result.returncode == 0
        assert (output_dir / "compile_test.cpp").exists()
    
    def test_compile_to_python(self, simple_dsl_file, tmp_path):
        """Test compiling to Python."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        result = subprocess.run(
            [sys.executable, '-m', 'mechanics_dsl.cli', 
             'compile', str(simple_dsl_file), 
             '--target', 'python', 
             '--output', str(output_dir)],
            capture_output=True, text=True
        )
        
        assert result.returncode == 0
        # Python output uses _sim suffix
        py_files = list(output_dir.glob("*.py"))
        assert len(py_files) > 0
    
    def test_compile_unknown_target(self, simple_dsl_file):
        """Test compiling to unknown target."""
        result = subprocess.run(
            [sys.executable, '-m', 'mechanics_dsl.cli', 
             'compile', str(simple_dsl_file), 
             '--target', 'unknown_language'],
            capture_output=True, text=True
        )
        
        assert result.returncode != 0
        assert 'unknown' in result.stderr.lower() or 'available' in result.stderr.lower()


class TestCLIRun:
    """Test CLI run command."""
    
    @pytest.fixture
    def runnable_dsl_file(self, tmp_path):
        """Create a DSL file for simulation tests."""
        content = r"""
        \system{run_test}
        \defvar{x}{m}
        \parameter{m}{1.0}{kg}
        \parameter{k}{10.0}{N/m}
        \lagrangian{\frac{1}{2}*m*\dot{x}^2 - \frac{1}{2}*k*x^2}
        \initial{x=1.0, x_dot=0}
        """
        file_path = tmp_path / "run_test.mdsl"
        file_path.write_text(content)
        return file_path
    
    def test_run_simulation(self, runnable_dsl_file, tmp_path):
        """Test running a simulation and saving output."""
        output_file = tmp_path / "results.json"
        
        result = subprocess.run(
            [sys.executable, '-m', 'mechanics_dsl.cli', 
             'run', str(runnable_dsl_file), 
             '--t-span', '0,5',
             '--points', '100',
             '--output', str(output_file)],
            capture_output=True, text=True
        )
        
        assert result.returncode == 0
        assert output_file.exists()
        
        # Verify JSON structure
        with open(output_file) as f:
            data = json.load(f)
        
        assert 't' in data
        assert 'y' in data
        assert len(data['t']) == 100


class TestCLIExport:
    """Test CLI export command."""
    
    @pytest.fixture
    def export_dsl_file(self, tmp_path):
        """Create a DSL file for export tests."""
        content = r"""
        \system{export_test}
        \defvar{theta}{rad}
        \parameter{m}{1.0}{kg}
        \parameter{l}{1.0}{m}
        \parameter{g}{9.81}{m/s^2}
        \lagrangian{\frac{1}{2}*m*l^2*\dot{theta}^2 - m*g*l*(1-\cos{theta})}
        \initial{theta=0.5}
        """
        file_path = tmp_path / "export_test.mdsl"
        file_path.write_text(content)
        return file_path
    
    def test_export_json(self, export_dsl_file, tmp_path):
        """Test exporting to JSON."""
        output_file = tmp_path / "results.json"
        
        result = subprocess.run(
            [sys.executable, '-m', 'mechanics_dsl.cli', 
             'export', str(export_dsl_file), 
             '--format', 'json',
             '--output', str(output_file),
             '--points', '50'],
            capture_output=True, text=True
        )
        
        assert result.returncode == 0
        assert output_file.exists()
    
    def test_export_csv(self, export_dsl_file, tmp_path):
        """Test exporting to CSV."""
        output_file = tmp_path / "results.csv"
        
        result = subprocess.run(
            [sys.executable, '-m', 'mechanics_dsl.cli', 
             'export', str(export_dsl_file), 
             '--format', 'csv',
             '--output', str(output_file),
             '--points', '50'],
            capture_output=True, text=True
        )
        
        assert result.returncode == 0
        assert output_file.exists()
        
        # Verify CSV has header and data
        content = output_file.read_text()
        lines = content.strip().split('\n')
        assert len(lines) == 51  # Header + 50 data rows


# Integration test with actual example files
class TestCLIWithExamples:
    """Test CLI with bundled example files."""
    
    def test_validate_bundled_example(self):
        """Test validating bundled example file."""
        example_file = PROJECT_ROOT / "examples" / "dsl" / "pendulum.mdsl"
        
        if not example_file.exists():
            pytest.skip("Example file not found")
        
        result = subprocess.run(
            [sys.executable, '-m', 'mechanics_dsl.cli', 'validate', str(example_file)],
            capture_output=True, text=True
        )
        
        assert result.returncode == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
