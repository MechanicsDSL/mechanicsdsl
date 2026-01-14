"""
Benchmark report generation for MechanicsDSL.

Generate HTML/Markdown reports and compare benchmark results.
"""
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime


@dataclass 
class BenchmarkComparison:
    """Comparison between two benchmark runs."""
    baseline_file: str
    current_file: str
    speedups: Dict[str, float]  # positive = faster
    regressions: List[str]
    improvements: List[str]


class BenchmarkReport:
    """
    Generate reports from benchmark results.
    """
    
    def __init__(self, results: Dict[str, Any]):
        self.results = results
    
    @classmethod
    def from_file(cls, filepath: str) -> 'BenchmarkReport':
        """Load results from JSON file."""
        with open(filepath) as f:
            return cls(json.load(f))
    
    def to_markdown(self) -> str:
        """Generate Markdown report."""
        lines = [
            "# MechanicsDSL Benchmark Report",
            "",
            f"**Generated:** {self.results.get('timestamp', 'Unknown')}",
            f"**Platform:** {self.results.get('platform', 'Unknown')}",
            f"**Python:** {self.results.get('python_version', 'Unknown')}",
            "",
            "## Simulation Benchmarks",
            "",
            "| System | Backend | Time (ms) | Evaluations |",
            "|--------|---------|----------|-------------|",
        ]
        
        for result in self.results.get('core_results', []):
            if result.get('success'):
                lines.append(
                    f"| {result['name']} | {result['backend']} | "
                    f"{result['time_ms']:.2f} | {result['num_evaluations']} |"
                )
        
        lines.extend([
            "",
            "## Code Generation Benchmarks",
            "",
            "| Generator | Time (ms) | Lines | Bytes |",
            "|-----------|----------|-------|-------|",
        ])
        
        for name, result in self.results.get('codegen_results', {}).items():
            if result.get('success'):
                lines.append(
                    f"| {name} | {result['time_ms']:.2f} | "
                    f"{result['output_lines']} | {result['output_bytes']} |"
                )
        
        lines.extend([
            "",
            "## Memory Usage",
            "",
            f"- **Compilation Peak:** {self.results.get('memory_results', {}).get('compilation_peak_mb', 0):.2f} MB",
            f"- **Simulation Peak:** {self.results.get('memory_results', {}).get('simulation_peak_mb', 0):.2f} MB",
            f"- **Total Peak:** {self.results.get('memory_results', {}).get('total_peak_mb', 0):.2f} MB",
        ])
        
        return "\n".join(lines)
    
    def to_html(self) -> str:
        """Generate HTML report."""
        # Convert markdown to basic HTML
        md = self.to_markdown()
        
        html = [
            "<!DOCTYPE html>",
            "<html><head>",
            "<title>MechanicsDSL Benchmark Report</title>",
            "<style>",
            "body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; }",
            "table { border-collapse: collapse; width: 100%; margin: 20px 0; }",
            "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
            "th { background-color: #4CAF50; color: white; }",
            "tr:nth-child(even) { background-color: #f2f2f2; }",
            "h1 { color: #333; }",
            "h2 { color: #666; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }",
            "</style>",
            "</head><body>",
        ]
        
        # Simple markdown to HTML conversion
        for line in md.split('\n'):
            if line.startswith('# '):
                html.append(f"<h1>{line[2:]}</h1>")
            elif line.startswith('## '):
                html.append(f"<h2>{line[3:]}</h2>")
            elif line.startswith('**'):
                html.append(f"<p><strong>{line.strip('*')}</strong></p>")
            elif line.startswith('- '):
                html.append(f"<li>{line[2:]}</li>")
            elif line.startswith('|'):
                # Table handling
                cells = [c.strip() for c in line.split('|')[1:-1]]
                if all(c.startswith('-') for c in cells):
                    continue  # Skip separator row
                tag = 'th' if 'System' in cells or 'Generator' in cells else 'td'
                row = ''.join(f"<{tag}>{c}</{tag}>" for c in cells)
                html.append(f"<tr>{row}</tr>")
        
        html.extend(["</body></html>"])
        return "\n".join(html)
    
    def save(self, filepath: str, format: str = "markdown") -> None:
        """
        Save report to file.
        
        Args:
            filepath: Output path
            format: "markdown" or "html"
        """
        if format == "html":
            content = self.to_html()
        else:
            content = self.to_markdown()
        
        with open(filepath, 'w') as f:
            f.write(content)


def generate_report(
    results_file: str,
    output_file: str,
    format: str = "markdown"
) -> str:
    """
    Generate a report from benchmark results.
    
    Args:
        results_file: Path to JSON results
        output_file: Path for output report
        format: "markdown" or "html"
        
    Returns:
        Path to generated report
    """
    report = BenchmarkReport.from_file(results_file)
    report.save(output_file, format)
    return output_file


def compare_reports(
    baseline_file: str,
    current_file: str,
    threshold: float = 0.1
) -> BenchmarkComparison:
    """
    Compare two benchmark runs.
    
    Args:
        baseline_file: Path to baseline JSON results
        current_file: Path to current JSON results
        threshold: Percentage change threshold for flagging
        
    Returns:
        BenchmarkComparison with speedups and regressions
    """
    with open(baseline_file) as f:
        baseline = json.load(f)
    with open(current_file) as f:
        current = json.load(f)
    
    speedups = {}
    regressions = []
    improvements = []
    
    # Compare core benchmarks
    baseline_core = {r['name']: r for r in baseline.get('core_results', [])}
    current_core = {r['name']: r for r in current.get('core_results', [])}
    
    for name in baseline_core:
        if name in current_core:
            b_time = baseline_core[name].get('time_ms', 0)
            c_time = current_core[name].get('time_ms', 0)
            
            if b_time > 0 and c_time > 0:
                speedup = (b_time - c_time) / b_time
                speedups[name] = speedup
                
                if speedup < -threshold:
                    regressions.append(f"{name}: {abs(speedup)*100:.1f}% slower")
                elif speedup > threshold:
                    improvements.append(f"{name}: {speedup*100:.1f}% faster")
    
    return BenchmarkComparison(
        baseline_file=baseline_file,
        current_file=current_file,
        speedups=speedups,
        regressions=regressions,
        improvements=improvements
    )


__all__ = [
    'BenchmarkReport',
    'BenchmarkComparison',
    'generate_report',
    'compare_reports',
]
