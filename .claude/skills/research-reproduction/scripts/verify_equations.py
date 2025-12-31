# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "rich>=13.0.0",
#     "typer>=0.12.0",
#     "pyyaml>=6.0",
# ]
# ///
"""
Equation Verification Script

Discovers, runs, and reports on equation-specific tests.
Maps equations from papers to their test implementations.

Usage:
    uv run scripts/verify_equations.py                  # Verify all equations
    uv run scripts/verify_equations.py --paper TITANS   # Verify specific paper
    uv run scripts/verify_equations.py --equation 3     # Verify specific equation
    uv run scripts/verify_equations.py --generate       # Generate test stubs

Features:
- Discovers tests tagged with equation references
- Reports equation coverage
- Generates test templates for missing equations
"""

import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

app = typer.Typer(help="Verify equation implementations")
console = Console()


@dataclass
class Equation:
    """Represents an equation from a paper."""
    paper_id: str
    number: int
    name: str
    latex: str = ""
    description: str = ""
    test_file: Optional[Path] = None
    implemented: bool = False
    tests_pass: bool = False


@dataclass  
class EquationRegistry:
    """Tracks equations across papers and their test status."""
    equations: dict[str, Equation] = field(default_factory=dict)
    
    def add(self, eq: Equation):
        key = f"{eq.paper_id}:eq{eq.number}"
        self.equations[key] = eq
    
    def get(self, paper_id: str, number: int) -> Optional[Equation]:
        return self.equations.get(f"{paper_id}:eq{number}")
    
    def by_paper(self, paper_id: str) -> list[Equation]:
        return [eq for eq in self.equations.values() if eq.paper_id == paper_id]


def discover_equations_from_context(context_dir: Path) -> EquationRegistry:
    """Parse .context.md files to find all equations."""
    registry = EquationRegistry()
    
    for context_file in context_dir.glob("*.context.md"):
        paper_id = context_file.stem.replace(".context", "")
        content = context_file.read_text()
        
        # Find equation sections
        # Pattern: ### Equation N: Name
        eq_pattern = r"###\s+Equation\s+(\d+):\s*([^\n]+)"
        
        for match in re.finditer(eq_pattern, content):
            number = int(match.group(1))
            name = match.group(2).strip()
            
            # Try to find LaTeX
            latex = ""
            latex_match = re.search(
                rf"Equation\s+{number}.*?```latex\s*\n(.*?)\n```",
                content, re.DOTALL
            )
            if latex_match:
                latex = latex_match.group(1).strip()
            
            eq = Equation(
                paper_id=paper_id,
                number=number,
                name=name,
                latex=latex,
            )
            registry.add(eq)
    
    return registry


def discover_equation_tests(tests_dir: Path) -> dict[str, Path]:
    """Find test files tagged with equation references."""
    equation_tests: dict[str, Path] = {}
    
    test_pattern = re.compile(r"test_eq(\d+)_(\w+)\.py")
    docstring_pattern = re.compile(r'""".*?Paper:\s*(\w+).*?Equation\s+(\d+)', re.DOTALL)
    
    for test_file in tests_dir.rglob("test_*.py"):
        content = test_file.read_text()
        
        # Check filename pattern
        match = test_pattern.search(test_file.name)
        if match:
            eq_num = int(match.group(1))
            # Try to find paper ID in docstring
            doc_match = docstring_pattern.search(content)
            if doc_match:
                paper_id = doc_match.group(1)
                key = f"{paper_id}:eq{eq_num}"
                equation_tests[key] = test_file
                continue
        
        # Check docstring for equation references
        for doc_match in docstring_pattern.finditer(content):
            paper_id = doc_match.group(1)
            eq_num = int(doc_match.group(2))
            key = f"{paper_id}:eq{eq_num}"
            equation_tests[key] = test_file
    
    return equation_tests


def run_equation_tests(test_file: Path) -> tuple[bool, str]:
    """Run tests for a specific equation and return pass/fail + output."""
    result = subprocess.run(
        ["uv", "run", "pytest", str(test_file), "-v", "--tb=short"],
        capture_output=True,
        text=True,
    )
    
    output = result.stdout + result.stderr
    passed = result.returncode == 0
    
    return passed, output


def generate_test_template(eq: Equation, output_dir: Path) -> Path:
    """Generate a test template file for an equation."""
    template = f'''# /// script
# requires-python = ">=3.11"
# dependencies = ["torch>=2.0", "pytest>=8.0"]
# ///
"""
Tests for Equation {eq.number}: {eq.name}

Paper: {eq.paper_id}
LaTeX: {eq.latex or "[TODO: Add LaTeX]"}
"""
import torch
import pytest


class TestEquation{eq.number}{eq.name.replace(" ", "").replace("-", "")}:
    """
    Verify Equation {eq.number} implementation matches paper specification.
    
    Equation: {eq.latex or "[TODO]"}
    Description: {eq.description or "[TODO: Add description]"}
    """
    
    @pytest.fixture
    def sample_inputs(self):
        """Create sample inputs matching paper dimensions."""
        torch.manual_seed(42)
        # TODO: Adjust shapes to match paper
        return {{
            "x": torch.randn(4, 32, 256),  # [batch, seq, dim]
        }}
    
    def test_output_shape(self, sample_inputs):
        """Output shape must match paper specification."""
        # TODO: Import your implementation
        # from src.module import equation_{eq.number}_impl
        
        # result = equation_{eq.number}_impl(**sample_inputs)
        # expected_shape = (4, 32, 256)  # TODO: Set expected shape
        # assert result.shape == expected_shape
        pytest.skip("TODO: Implement test")
    
    def test_gradient_flow(self, sample_inputs):
        """Gradients must flow through operation."""
        # x = sample_inputs["x"].clone().requires_grad_(True)
        # result = equation_{eq.number}_impl(x)
        # loss = result.sum()
        # loss.backward()
        # assert x.grad is not None
        pytest.skip("TODO: Implement test")
    
    def test_numerical_stability(self, sample_inputs):
        """Operation should handle extreme values."""
        # x_large = sample_inputs["x"] * 1000
        # result = equation_{eq.number}_impl(x_large)
        # assert not torch.isnan(result).any()
        pytest.skip("TODO: Implement test")
    
    def test_deterministic(self, sample_inputs):
        """Same inputs should produce same outputs."""
        # result1 = equation_{eq.number}_impl(**sample_inputs)
        # result2 = equation_{eq.number}_impl(**sample_inputs)
        # assert torch.allclose(result1, result2)
        pytest.skip("TODO: Implement test")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
'''
    
    # Create filename
    safe_name = re.sub(r"[^\w]", "_", eq.name.lower())
    filename = f"test_eq{eq.number}_{safe_name}.py"
    
    # Determine directory
    eq_tests_dir = output_dir / "test_equations"
    eq_tests_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = eq_tests_dir / filename
    output_path.write_text(template)
    
    return output_path


@app.command()
def verify(
    paper: Optional[str] = typer.Option(None, "--paper", "-p", help="Filter by paper ID"),
    equation: Optional[int] = typer.Option(None, "--equation", "-e", help="Verify specific equation number"),
    context_dir: Path = typer.Option(Path("."), "--context", "-c", help="Directory with .context.md files"),
    tests_dir: Path = typer.Option(Path("tests"), "--tests", "-t", help="Tests directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show test output"),
):
    """
    Verify equation implementations against their tests.
    
    Examples:
        uv run scripts/verify_equations.py
        uv run scripts/verify_equations.py --paper TITANS
        uv run scripts/verify_equations.py --equation 3 --verbose
    """
    console.print(Panel("[bold]Equation Verification[/bold]", border_style="blue"))
    
    # Discover equations from context documents
    console.print("\n[dim]Discovering equations from context documents...[/dim]")
    
    # Check papers/ subdirectory if context_dir is project root
    if (context_dir / "papers").exists():
        context_dir = context_dir / "papers"
    
    registry = discover_equations_from_context(context_dir)
    
    if not registry.equations:
        console.print("[yellow]No .context.md files found. Run paper extraction first.[/yellow]")
        console.print(f"[dim]Looked in: {context_dir}[/dim]")
        raise typer.Exit(1)
    
    console.print(f"[green]Found {len(registry.equations)} equations[/green]")
    
    # Discover equation tests
    console.print("[dim]Discovering equation tests...[/dim]")
    test_mapping = discover_equation_tests(tests_dir)
    console.print(f"[green]Found {len(test_mapping)} equation test files[/green]")
    
    # Map tests to equations
    for key, test_file in test_mapping.items():
        if key in registry.equations:
            registry.equations[key].test_file = test_file
            registry.equations[key].implemented = True
    
    # Filter if requested
    equations_to_verify = list(registry.equations.values())
    
    if paper:
        equations_to_verify = [eq for eq in equations_to_verify if eq.paper_id.upper() == paper.upper()]
    
    if equation is not None:
        equations_to_verify = [eq for eq in equations_to_verify if eq.number == equation]
    
    if not equations_to_verify:
        console.print("[yellow]No matching equations found[/yellow]")
        raise typer.Exit(1)
    
    # Run verification
    console.print(f"\n[bold]Verifying {len(equations_to_verify)} equations...[/bold]\n")
    
    results_table = Table(title="Equation Verification Results")
    results_table.add_column("Paper", style="cyan")
    results_table.add_column("Eq #", style="bold")
    results_table.add_column("Name")
    results_table.add_column("Test File")
    results_table.add_column("Status", justify="center")
    
    passed_count = 0
    failed_count = 0
    missing_count = 0
    
    for eq in sorted(equations_to_verify, key=lambda e: (e.paper_id, e.number)):
        if eq.test_file:
            # Run the test
            with console.status(f"Testing Eq {eq.number}: {eq.name}..."):
                passed, output = run_equation_tests(eq.test_file)
            
            eq.tests_pass = passed
            
            if passed:
                status = "[green]✓ PASS[/green]"
                passed_count += 1
            else:
                status = "[red]✗ FAIL[/red]"
                failed_count += 1
                if verbose:
                    console.print(f"\n[red]Equation {eq.number} failed:[/red]")
                    console.print(f"[dim]{output}[/dim]")
            
            test_file_str = str(eq.test_file.relative_to(Path.cwd()) if eq.test_file.is_relative_to(Path.cwd()) else eq.test_file)
        else:
            status = "[yellow]○ NO TEST[/yellow]"
            test_file_str = "-"
            missing_count += 1
        
        results_table.add_row(
            eq.paper_id,
            str(eq.number),
            eq.name[:30] + "..." if len(eq.name) > 30 else eq.name,
            test_file_str[:40] + "..." if len(test_file_str) > 40 else test_file_str,
            status,
        )
    
    console.print(results_table)
    
    # Summary
    console.print("\n[bold]Summary:[/bold]")
    console.print(f"  [green]Passed: {passed_count}[/green]")
    console.print(f"  [red]Failed: {failed_count}[/red]")
    console.print(f"  [yellow]Missing Tests: {missing_count}[/yellow]")
    
    total = passed_count + failed_count + missing_count
    if total > 0:
        coverage = (passed_count + failed_count) / total * 100
        console.print(f"  [blue]Test Coverage: {coverage:.0f}%[/blue]")
    
    # Exit code
    if failed_count > 0:
        raise typer.Exit(1)
    elif missing_count > 0:
        console.print("\n[yellow]⚠ Some equations have no tests. Run --generate to create stubs.[/yellow]")


@app.command()
def generate(
    context_dir: Path = typer.Option(Path("."), "--context", "-c", help="Directory with .context.md files"),
    output_dir: Path = typer.Option(Path("tests"), "--output", "-o", help="Output directory for tests"),
    paper: Optional[str] = typer.Option(None, "--paper", "-p", help="Generate only for specific paper"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite existing test files"),
):
    """
    Generate test templates for equations without tests.
    
    Examples:
        uv run scripts/verify_equations.py generate
        uv run scripts/verify_equations.py generate --paper TITANS
    """
    console.print(Panel("[bold]Generate Equation Tests[/bold]", border_style="blue"))
    
    # Check papers/ subdirectory
    if (context_dir / "papers").exists():
        context_dir = context_dir / "papers"
    
    # Discover equations
    registry = discover_equations_from_context(context_dir)
    
    if not registry.equations:
        console.print("[yellow]No equations found. Run paper extraction first.[/yellow]")
        raise typer.Exit(1)
    
    # Find existing tests
    test_mapping = discover_equation_tests(output_dir)
    
    # Generate templates for missing tests
    equations = list(registry.equations.values())
    if paper:
        equations = [eq for eq in equations if eq.paper_id.upper() == paper.upper()]
    
    generated = 0
    skipped = 0
    
    for eq in equations:
        key = f"{eq.paper_id}:eq{eq.number}"
        
        if key in test_mapping and not overwrite:
            console.print(f"[dim]Skipping Eq {eq.number} ({eq.paper_id}) - test exists[/dim]")
            skipped += 1
            continue
        
        output_path = generate_test_template(eq, output_dir)
        console.print(f"[green]Generated: {output_path}[/green]")
        generated += 1
    
    console.print(f"\n[bold]Generated {generated} test files, skipped {skipped}[/bold]")


@app.command("list")
def list_equations(
    context_dir: Path = typer.Option(Path("."), "--context", "-c", help="Directory with .context.md files"),
    tests_dir: Path = typer.Option(Path("tests"), "--tests", "-t", help="Tests directory"),
):
    """
    List all discovered equations and their test status.
    """
    # Check papers/ subdirectory
    if (context_dir / "papers").exists():
        context_dir = context_dir / "papers"
    
    registry = discover_equations_from_context(context_dir)
    test_mapping = discover_equation_tests(tests_dir)
    
    # Build tree view by paper
    tree = Tree("[bold]Equations by Paper[/bold]")
    
    papers = set(eq.paper_id for eq in registry.equations.values())
    
    for paper_id in sorted(papers):
        paper_branch = tree.add(f"[cyan]{paper_id}[/cyan]")
        
        for eq in sorted(registry.by_paper(paper_id), key=lambda e: e.number):
            key = f"{eq.paper_id}:eq{eq.number}"
            has_test = key in test_mapping
            
            icon = "[green]✓[/green]" if has_test else "[yellow]○[/yellow]"
            paper_branch.add(f"{icon} Eq {eq.number}: {eq.name}")
    
    console.print(tree)
    
    # Summary
    total = len(registry.equations)
    with_tests = len([k for k in registry.equations.keys() if k in test_mapping])
    console.print(f"\n[bold]Total: {total} equations, {with_tests} with tests ({with_tests/total*100:.0f}% coverage)[/bold]")


if __name__ == "__main__":
    app()
