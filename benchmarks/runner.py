"""
Benchmark runner and CLI for MechanicsDSL.

Run all benchmarks and generate reports.

Usage:
    python -m benchmarks.runner
    python -m benchmarks.runner --output results.json
    python -m benchmarks.runner --suite core --backend scipy
"""
import argparse
import json
import sys
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from pathlib import Path

from .core_benchmarks import (
    STANDARD_BENCHMARKS, run_benchmark, BenchmarkResult
)
from .codegen_benchmarks import benchmark_all_generators, CodegenResult
from .memory_benchmarks import benchmark_memory_usage, MemoryBenchmark


@dataclass
class BenchmarkSuite:
    """Complete benchmark suite results."""
    timestamp: str
    platform: str
    python_version: str
    core_results: List[Dict]
    codegen_results: Dict[str, Dict]
    memory_results: Dict
    metadata: Dict[str, Any]


class BenchmarkRunner:
    """
    Main benchmark runner.
    
    Orchestrates all benchmark suites and generates reports.
    """
    
    def __init__(
        self,
        backends: List[str] = ["scipy"],
        output_dir: Optional[str] = None
    ):
        self.backends = backends
        self.output_dir = Path(output_dir) if output_dir else Path(".")
        self.results: Optional[BenchmarkSuite] = None
    
    def run_all(self, verbose: bool = True) -> BenchmarkSuite:
        """
        Run all benchmark suites.
        
        Args:
            verbose: Print progress to stdout
            
        Returns:
            BenchmarkSuite with all results
        """
        import platform
        
        if verbose:
            print("=" * 60)
            print("MechanicsDSL Benchmark Suite")
            print("=" * 60)
            print()
        
        # Core simulation benchmarks
        if verbose:
            print("Running core simulation benchmarks...")
        
        core_results = []
        for benchmark in STANDARD_BENCHMARKS:
            for backend in self.backends:
                result = run_benchmark(benchmark, backend)
                core_results.append(asdict(result))
                if verbose:
                    status = "✓" if result.success else "✗"
                    print(f"  {status} {benchmark.name}/{backend}: {result.time_ms:.2f}ms")
        
        # Code generation benchmarks
        if verbose:
            print("\nRunning code generation benchmarks...")
        
        codegen_results = {}
        for name, result in benchmark_all_generators().items():
            codegen_results[name] = asdict(result)
        
        # Memory benchmarks
        if verbose:
            print("\nRunning memory benchmarks...")
        
        from .core_benchmarks import DOUBLE_PENDULUM
        memory_result = benchmark_memory_usage(DOUBLE_PENDULUM.dsl_code)
        memory_results = {
            'compilation_peak_mb': memory_result.compilation.peak_mb,
            'simulation_peak_mb': memory_result.simulation.peak_mb,
            'total_peak_mb': memory_result.total_peak_mb,
            'gc_collected': memory_result.gc_collected,
        }
        
        if verbose:
            print(f"  Peak memory: {memory_result.total_peak_mb:.2f} MB")
        
        # Build suite
        self.results = BenchmarkSuite(
            timestamp=datetime.now().isoformat(),
            platform=platform.platform(),
            python_version=platform.python_version(),
            core_results=core_results,
            codegen_results=codegen_results,
            memory_results=memory_results,
            metadata={
                'backends': self.backends,
                'num_benchmarks': len(STANDARD_BENCHMARKS),
                'num_generators': len(codegen_results),
            }
        )
        
        if verbose:
            print("\n" + "=" * 60)
            print("Benchmark complete!")
            print("=" * 60)
        
        return self.results
    
    def save_results(self, filename: str = "benchmark_results.json") -> str:
        """Save results to JSON file."""
        if self.results is None:
            raise ValueError("No results to save. Run run_all() first.")
        
        output_path = self.output_dir / filename
        
        with open(output_path, 'w') as f:
            json.dump(asdict(self.results), f, indent=2)
        
        return str(output_path)
    
    def print_summary(self) -> None:
        """Print a summary table of results."""
        if self.results is None:
            print("No results available.")
            return
        
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        
        # Core benchmarks
        print("\nSimulation Performance:")
        print("-" * 40)
        for result in self.results.core_results:
            if result['success']:
                print(f"  {result['name']:20s} {result['time_ms']:8.2f} ms")
        
        # Fastest/slowest generators
        print("\nCode Generation (median ms):")
        print("-" * 40)
        sorted_gen = sorted(
            self.results.codegen_results.items(),
            key=lambda x: x[1].get('time_ms', float('inf'))
        )
        for name, result in sorted_gen[:5]:
            if result['success']:
                print(f"  {name:15s} {result['time_ms']:8.2f} ms ({result['output_lines']} lines)")
        
        # Memory
        print("\nMemory Usage:")
        print("-" * 40)
        print(f"  Peak: {self.results.memory_results['total_peak_mb']:.2f} MB")


def run_all_benchmarks(
    backends: List[str] = ["scipy"],
    output: Optional[str] = None,
    verbose: bool = True
) -> BenchmarkSuite:
    """
    Convenience function to run all benchmarks.
    
    Args:
        backends: List of solver backends
        output: Output JSON file path
        verbose: Print progress
        
    Returns:
        BenchmarkSuite with results
    """
    runner = BenchmarkRunner(backends=backends)
    results = runner.run_all(verbose=verbose)
    
    if output:
        runner.save_results(output)
        if verbose:
            print(f"\nResults saved to: {output}")
    
    runner.print_summary()
    return results


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="MechanicsDSL Benchmark Suite"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output JSON file path",
        default=None
    )
    parser.add_argument(
        "--backend", "-b",
        action="append",
        help="Solver backend(s) to test",
        default=None
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress output"
    )
    
    args = parser.parse_args()
    
    backends = args.backend or ["scipy"]
    
    run_all_benchmarks(
        backends=backends,
        output=args.output,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main()


__all__ = [
    'BenchmarkSuite',
    'BenchmarkRunner',
    'run_all_benchmarks',
]
