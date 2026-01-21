"""
Tests for MechanicsDSL REPL Module

Run with:
    pytest tests/test_repl.py -v
"""

import pytest
from unittest.mock import patch
from io import StringIO


class TestREPLImport:
    """Test REPL module imports."""
    
    def test_import_repl(self):
        """Test that REPL module imports correctly."""
        from mechanics_dsl.repl import REPL, run_repl
        assert REPL is not None
        assert callable(run_repl)


class TestREPLClass:
    """Test REPL class functionality."""
    
    def test_repl_initialization(self):
        """Test REPL initializes correctly."""
        from mechanics_dsl.repl import REPL
        repl = REPL()
        assert repl.buffer == []
        assert repl.compiler is None
        assert repl.solution is None
    
    def test_add_to_buffer(self):
        """Test adding lines to buffer."""
        from mechanics_dsl.repl import REPL
        repl = REPL()
        with patch('sys.stdout', new_callable=StringIO):
            repl._add_to_buffer('\\system{test}')
        assert len(repl.buffer) == 1
        assert repl.buffer[0] == '\\system{test}'
    
    def test_clear_command(self):
        """Test :clear command."""
        from mechanics_dsl.repl import REPL
        repl = REPL()
        repl.buffer = ['line1', 'line2']
        with patch('sys.stdout', new_callable=StringIO):
            result = repl._handle_command('clear')
        assert result == True
        assert repl.buffer == []
    
    def test_quit_command(self):
        """Test :quit command returns False."""
        from mechanics_dsl.repl import REPL
        repl = REPL()
        with patch('sys.stdout', new_callable=StringIO):
            result = repl._handle_command('quit')
        assert result == False
    
    def test_exit_command(self):
        """Test :exit command returns False."""
        from mechanics_dsl.repl import REPL
        repl = REPL()
        with patch('sys.stdout', new_callable=StringIO):
            result = repl._handle_command('exit')
        assert result == False
    
    def test_show_command_empty(self):
        """Test :show with empty buffer."""
        from mechanics_dsl.repl import REPL
        repl = REPL()
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            repl._handle_command('show')
            output = mock_stdout.getvalue()
        assert 'empty' in output.lower()
    
    def test_show_command_with_content(self):
        """Test :show with content."""
        from mechanics_dsl.repl import REPL
        repl = REPL()
        repl.buffer = ['\\system{test}']
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            repl._handle_command('show')
            output = mock_stdout.getvalue()
        assert '\\system{test}' in output
    
    def test_list_command(self):
        """Test :list command shows presets."""
        from mechanics_dsl.repl import REPL
        repl = REPL()
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            repl._handle_command('list')
            output = mock_stdout.getvalue()
        assert 'preset' in output.lower() or 'pendulum' in output.lower()
    
    def test_preset_command(self):
        """Test :preset command loads preset."""
        from mechanics_dsl.repl import REPL
        repl = REPL()
        with patch('sys.stdout', new_callable=StringIO):
            # This may fail if compiler doesn't work, but should load preset
            repl._handle_command('preset pendulum')
        assert len(repl.buffer) > 0
        assert any('system' in line.lower() for line in repl.buffer)
    
    def test_unknown_command(self):
        """Test unknown command handling."""
        from mechanics_dsl.repl import REPL
        repl = REPL()
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            result = repl._handle_command('unknown_command')
            output = mock_stdout.getvalue()
        assert result == True  # Don't exit
        assert 'unknown' in output.lower() or 'help' in output.lower()
    
    def test_run_without_compilation(self):
        """Test :run without compiled system."""
        from mechanics_dsl.repl import REPL
        repl = REPL()
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            repl._run_simulation(10.0)
            output = mock_stdout.getvalue()
        assert 'compile' in output.lower() or 'no' in output.lower()
    
    def test_plot_without_solution(self):
        """Test :plot without simulation results."""
        from mechanics_dsl.repl import REPL
        repl = REPL()
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            repl._plot()
            output = mock_stdout.getvalue()
        assert 'no' in output.lower() or 'run' in output.lower()


class TestREPLBanner:
    """Test REPL banner and help."""
    
    def test_banner_content(self):
        """Test REPL banner has key commands."""
        from mechanics_dsl.repl import REPL_BANNER
        assert ':help' in REPL_BANNER
        assert ':load' in REPL_BANNER
        assert ':preset' in REPL_BANNER
        assert ':compile' in REPL_BANNER
        assert ':run' in REPL_BANNER
        assert ':quit' in REPL_BANNER


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
