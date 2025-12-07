"""
Tests for additional code generation backends:
- OpenMP (multi-core C++)
- WASM (WebAssembly)
- Arduino (embedded)
"""
import pytest
import os
import tempfile
import sympy as sp

from mechanics_dsl.codegen.openmp import OpenMPGenerator
from mechanics_dsl.codegen.wasm import WasmGenerator
from mechanics_dsl.codegen.arduino import ArduinoGenerator


def create_test_system():
    """Create a standard pendulum system for testing."""
    theta = sp.Symbol('theta', real=True)
    g = sp.Symbol('g', positive=True)
    l = sp.Symbol('l', positive=True)
    
    return {
        'system_name': 'test_pendulum',
        'coordinates': ['theta'],
        'parameters': {'g': 9.81, 'l': 1.0},
        'initial_conditions': {'theta': 0.1, 'theta_dot': 0.0},
        'equations': {'theta_ddot': -g/l * sp.sin(theta)}
    }


class TestOpenMPBackend:
    """Test OpenMP code generation."""
    
    def test_generates_file(self):
        """Test that OpenMP file is generated."""
        system = create_test_system()
        with tempfile.TemporaryDirectory() as tmpdir:
            output = os.path.join(tmpdir, 'test.cpp')
            gen = OpenMPGenerator(**system)
            gen.generate(output)
            assert os.path.exists(output)
    
    def test_contains_openmp_pragmas(self):
        """Test that generated code contains OpenMP pragmas."""
        system = create_test_system()
        with tempfile.TemporaryDirectory() as tmpdir:
            output = os.path.join(tmpdir, 'test.cpp')
            gen = OpenMPGenerator(**system)
            gen.generate(output)
            
            with open(output) as f:
                content = f.read()
            
            assert '#pragma omp' in content
            assert '#include <omp.h>' in content
    
    def test_target_name_and_extension(self):
        """Test target_name and file_extension properties."""
        system = create_test_system()
        gen = OpenMPGenerator(**system)
        assert gen.target_name == 'openmp'
        assert gen.file_extension == '.cpp'


class TestWasmBackend:
    """Test WebAssembly code generation."""
    
    def test_generates_files(self):
        """Test that WASM files are generated."""
        system = create_test_system()
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = WasmGenerator(**system)
            gen.generate(tmpdir)
            
            assert os.path.exists(os.path.join(tmpdir, 'test_pendulum.c'))
            assert os.path.exists(os.path.join(tmpdir, 'index.html'))
            assert os.path.exists(os.path.join(tmpdir, 'build.sh'))
    
    def test_c_file_contains_emscripten(self):
        """Test that C file contains Emscripten exports."""
        system = create_test_system()
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = WasmGenerator(**system)
            gen.generate(tmpdir)
            
            with open(os.path.join(tmpdir, 'test_pendulum.c')) as f:
                content = f.read()
            
            assert 'EMSCRIPTEN_KEEPALIVE' in content
            assert '#include <emscripten.h>' in content
    
    def test_html_contains_canvas(self):
        """Test that HTML contains canvas visualization."""
        system = create_test_system()
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = WasmGenerator(**system)
            gen.generate(tmpdir)
            
            with open(os.path.join(tmpdir, 'index.html')) as f:
                content = f.read()
            
            assert '<canvas' in content
            assert 'function animate' in content
    
    def test_target_name_and_extension(self):
        """Test target_name and file_extension properties."""
        system = create_test_system()
        gen = WasmGenerator(**system)
        assert gen.target_name == 'wasm'
        assert gen.file_extension == '.c'


class TestArduinoBackend:
    """Test Arduino code generation."""
    
    def test_generates_file(self):
        """Test that Arduino sketch is generated."""
        system = create_test_system()
        with tempfile.TemporaryDirectory() as tmpdir:
            output = os.path.join(tmpdir, 'test.ino')
            gen = ArduinoGenerator(**system)
            gen.generate(output)
            assert os.path.exists(output)
    
    def test_contains_arduino_functions(self):
        """Test that sketch contains setup() and loop()."""
        system = create_test_system()
        with tempfile.TemporaryDirectory() as tmpdir:
            output = os.path.join(tmpdir, 'test.ino')
            gen = ArduinoGenerator(**system)
            gen.generate(output)
            
            with open(output) as f:
                content = f.read()
            
            assert 'void setup()' in content
            assert 'void loop()' in content
            assert 'Serial.begin' in content
    
    def test_uses_float_types(self):
        """Test that Arduino uses float (not double) for efficiency."""
        system = create_test_system()
        with tempfile.TemporaryDirectory() as tmpdir:
            output = os.path.join(tmpdir, 'test.ino')
            gen = ArduinoGenerator(**system)
            gen.generate(output)
            
            with open(output) as f:
                content = f.read()
            
            # Arduino should use float for efficiency
            assert 'float state' in content
    
    def test_servo_option(self):
        """Test servo output generation."""
        system = create_test_system()
        with tempfile.TemporaryDirectory() as tmpdir:
            output = os.path.join(tmpdir, 'test.ino')
            gen = ArduinoGenerator(**system, servo_pin=9)
            gen.generate(output)
            
            with open(output) as f:
                content = f.read()
            
            assert '#include <Servo.h>' in content
            assert 'outputServo.attach(9)' in content
    
    def test_target_name_and_extension(self):
        """Test target_name and file_extension properties."""
        system = create_test_system()
        gen = ArduinoGenerator(**system)
        assert gen.target_name == 'arduino'
        assert gen.file_extension == '.ino'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
