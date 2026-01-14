"""
Code generation benchmarks for MechanicsDSL.

Benchmarks for code generation speed across all backends.
"""
import time
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path

try:
    from mechanics_dsl import PhysicsCompiler
    from mechanics_dsl.codegen import (
        CppGenerator, PythonGenerator, RustGenerator, JuliaGenerator,
        MatlabGenerator, FortranGenerator, JavaScriptGenerator,
        CUDAGenerator, OpenMPGenerator, WASMGenerator, ArduinoGenerator
    )
except ImportError:
    PhysicsCompiler = None


@dataclass
class CodegenResult:
    """Result of a code generation benchmark."""
    generator: str
    time_ms: float
    output_lines: int
    output_bytes: int
    success: bool
    error: Optional[str] = None


@dataclass
class CodegenBenchmark:
    """Configuration for a codegen benchmark."""
    name: str
    dsl_code: str
    generators: List[str] = field(default_factory=lambda: ["cpp", "python", "rust"])


# Standard DSL code for codegen benchmarks
BENCHMARK_DSL = r"""
\system{benchmark_system}
\defvar{theta1}{Angle}{rad}
\defvar{theta2}{Angle}{rad}
\defvar{m1}{Mass}{kg}
\defvar{m2}{Mass}{kg}
\defvar{l1}{Length}{m}
\defvar{l2}{Length}{m}
\defvar{g}{Acceleration}{m/s^2}

\parameter{m1}{1.0}{kg}
\parameter{m2}{1.0}{kg}
\parameter{l1}{1.0}{m}
\parameter{l2}{1.0}{m}
\parameter{g}{9.81}{m/s^2}

\lagrangian{
    \frac{1}{2} * (m1 + m2) * l1^2 * \dot{theta1}^2
    + \frac{1}{2} * m2 * l2^2 * \dot{theta2}^2
    + m2 * l1 * l2 * \dot{theta1} * \dot{theta2} * \cos{theta1 - theta2}
    - (m1 + m2) * g * l1 * (1 - \cos{theta1})
    - m2 * g * l2 * (1 - \cos{theta2})
}

\initial{theta1=2.5, theta1_dot=0.0, theta2=2.0, theta2_dot=0.0}
"""


# Generator classes by name
GENERATOR_CLASSES = {
    "cpp": "CppGenerator",
    "python": "PythonGenerator", 
    "rust": "RustGenerator",
    "julia": "JuliaGenerator",
    "matlab": "MatlabGenerator",
    "fortran": "FortranGenerator",
    "javascript": "JavaScriptGenerator",
    "cuda": "CUDAGenerator",
    "openmp": "OpenMPGenerator",
    "wasm": "WASMGenerator",
    "arduino": "ArduinoGenerator",
}


def benchmark_generator(
    generator_name: str,
    dsl_code: str = BENCHMARK_DSL,
    repeats: int = 5
) -> CodegenResult:
    """
    Benchmark a single code generator.
    
    Args:
        generator_name: Name of generator (e.g., "cpp", "rust")
        dsl_code: DSL code to compile
        repeats: Number of runs to average
        
    Returns:
        CodegenResult with timing info
    """
    if PhysicsCompiler is None:
        return CodegenResult(
            generator=generator_name,
            time_ms=0,
            output_lines=0,
            output_bytes=0,
            success=False,
            error="PhysicsCompiler not available"
        )
    
    try:
        # Compile DSL
        compiler = PhysicsCompiler()
        result = compiler.compile_dsl(dsl_code)
        
        if not result['success']:
            return CodegenResult(
                generator=generator_name,
                time_ms=0,
                output_lines=0,
                output_bytes=0,
                success=False,
                error="DSL compilation failed"
            )
        
        # Get generator class
        generator_class_name = GENERATOR_CLASSES.get(generator_name)
        if not generator_class_name:
            return CodegenResult(
                generator=generator_name,
                time_ms=0,
                output_lines=0,
                output_bytes=0,
                success=False,
                error=f"Unknown generator: {generator_name}"
            )
        
        # Time code generation
        times = []
        output = ""
        
        for _ in range(repeats):
            with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
                output_path = f.name
            
            start = time.perf_counter()
            output = compiler.export(generator_name, output_path)
            end = time.perf_counter()
            
            times.append((end - start) * 1000)
            
            # Read output
            try:
                with open(output_path, 'r') as f:
                    output = f.read()
            except:
                pass
            
            # Cleanup
            try:
                Path(output_path).unlink()
            except:
                pass
        
        import numpy as np
        return CodegenResult(
            generator=generator_name,
            time_ms=np.median(times),
            output_lines=output.count('\n') + 1,
            output_bytes=len(output.encode('utf-8')),
            success=True
        )
        
    except Exception as e:
        return CodegenResult(
            generator=generator_name,
            time_ms=0,
            output_lines=0,
            output_bytes=0,
            success=False,
            error=str(e)
        )


def benchmark_all_generators(
    dsl_code: str = BENCHMARK_DSL
) -> Dict[str, CodegenResult]:
    """
    Benchmark all available code generators.
    
    Args:
        dsl_code: DSL code to compile
        
    Returns:
        Dictionary of generator name -> result
    """
    results = {}
    
    for name in GENERATOR_CLASSES.keys():
        results[name] = benchmark_generator(name, dsl_code)
        print(f"  {name}: {results[name].time_ms:.2f}ms, {results[name].output_lines} lines")
    
    return results


__all__ = [
    'CodegenResult',
    'CodegenBenchmark',
    'BENCHMARK_DSL',
    'GENERATOR_CLASSES',
    'benchmark_generator',
    'benchmark_all_generators',
]
