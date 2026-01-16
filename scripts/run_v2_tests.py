"""
Version 2.0 Test Suite Runner
=============================

Runs all new v2.0 tests including:
- ARM code generator tests
- Enhanced codegen tests (CMake, Cargo, cuBLAS)
- Docker integration tests
- Benchmarks

Usage:
    python run_v2_tests.py
    python run_v2_tests.py --benchmark
"""

import subprocess
import sys
import os


def run_tests():
    """Run v2.0 test suite."""
    print("=" * 60)
    print("MechanicsDSL v2.0 Test Suite")
    print("=" * 60)
    
    test_dir = os.path.dirname(os.path.abspath(__file__))
    
    tests = [
        ("ARM Code Generator", "tests/unit/test_codegen_arm.py"),
        ("Enhanced Codegen (C++, Rust, CUDA)", "tests/unit/test_codegen_v2.py"),
        ("Docker Integration", "tests/integration/test_docker.py"),
    ]
    
    results = []
    
    for name, path in tests:
        print(f"\n[Running] {name}...")
        full_path = os.path.join(os.path.dirname(test_dir), path)
        
        if not os.path.exists(full_path):
            print(f"  ⚠️  Skipped (file not found)")
            results.append((name, "SKIPPED"))
            continue
        
        result = subprocess.run(
            [sys.executable, "-m", "pytest", full_path, "-v", "--tb=short"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print(f"  ✅ PASSED")
            results.append((name, "PASSED"))
        else:
            print(f"  ❌ FAILED")
            print(result.stdout)
            print(result.stderr)
            results.append((name, "FAILED"))
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for name, status in results:
        icon = "✅" if status == "PASSED" else ("⚠️" if status == "SKIPPED" else "❌")
        print(f"  {icon} {name}: {status}")
    
    passed = sum(1 for _, s in results if s == "PASSED")
    total = len(results)
    print(f"\n  Total: {passed}/{total} passed")
    
    return all(s == "PASSED" for _, s in results)


def run_benchmarks():
    """Run ARM benchmarks."""
    print("\n" + "=" * 60)
    print("Running ARM Benchmarks...")
    print("=" * 60)
    
    subprocess.run([
        sys.executable, "-m", "pytest",
        "benchmarks/benchmark_arm.py",
        "-v", "--benchmark-only", "--benchmark-autosave"
    ])


if __name__ == "__main__":
    success = run_tests()
    
    if "--benchmark" in sys.argv:
        run_benchmarks()
    
    sys.exit(0 if success else 1)
