# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "rich>=13.0.0",
#     "typer>=0.12.0",
# ]
# ///
"""
Quality Check Script

Runs code quality tools: ruff (format + lint), ty (type check), pytest.
Designed for research reproduction projects.

Usage:
    uv run scripts/quality_check.py                    # Check all
    uv run scripts/quality_check.py --fix             # Auto-fix issues
    uv run scripts/quality_check.py --check-only      # No fixes, just report
    uv run scripts/quality_check.py src/              # Check specific directory
"""

import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(help="Run code quality checks")
console = Console()


class QualityChecker:
    """Orchestrates quality checks for Python projects."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.results: dict[str, dict] = {}
    
    def run_command(self, cmd: list[str], capture: bool = True) -> tuple[int, str, str]:
        """Run a command and return exit code, stdout, stderr."""
        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=capture,
                text=True,
            )
            return result.returncode, result.stdout, result.stderr
        except FileNotFoundError:
            return -1, "", f"Command not found: {cmd[0]}"
    
    def check_ruff_format(self, paths: list[str], fix: bool = False) -> bool:
        """Check/fix code formatting with ruff."""
        console.print("\n[bold blue]📐 Checking formatting with ruff...[/bold blue]")
        
        cmd = ["uv", "run", "ruff", "format"]
        if not fix:
            cmd.append("--check")
        cmd.extend(paths)
        
        code, stdout, stderr = self.run_command(cmd)
        
        if code == 0:
            console.print("[green]✓ Formatting OK[/green]")
            self.results["format"] = {"status": "pass", "issues": 0}
            return True
        else:
            if fix:
                console.print("[yellow]⚡ Formatting applied[/yellow]")
            else:
                console.print("[red]✗ Formatting issues found[/red]")
                console.print(f"[dim]{stdout}{stderr}[/dim]")
            self.results["format"] = {"status": "fail" if not fix else "fixed", "output": stdout}
            return fix  # Pass if we fixed it
    
    def check_ruff_lint(self, paths: list[str], fix: bool = False) -> bool:
        """Check/fix linting issues with ruff."""
        console.print("\n[bold blue]🔍 Linting with ruff...[/bold blue]")
        
        cmd = ["uv", "run", "ruff", "check"]
        if fix:
            cmd.append("--fix")
        cmd.extend(paths)
        
        code, stdout, stderr = self.run_command(cmd)
        
        # Count issues
        issue_count = len([l for l in stdout.split("\n") if l.strip() and ":" in l])
        
        if code == 0:
            console.print("[green]✓ No linting issues[/green]")
            self.results["lint"] = {"status": "pass", "issues": 0}
            return True
        else:
            console.print(f"[red]✗ {issue_count} linting issues[/red]")
            if stdout:
                # Show first 20 issues
                lines = stdout.strip().split("\n")[:20]
                for line in lines:
                    console.print(f"[dim]  {line}[/dim]")
                if len(stdout.strip().split("\n")) > 20:
                    console.print(f"[dim]  ... and more[/dim]")
            self.results["lint"] = {"status": "fail", "issues": issue_count, "output": stdout}
            return False
    
    def check_types(self, paths: list[str]) -> bool:
        """Type check with ty (Astral's type checker)."""
        console.print("\n[bold blue]🔬 Type checking with ty...[/bold blue]")
        
        # Check if ty is available
        code, _, _ = self.run_command(["uv", "run", "ty", "--version"])
        if code != 0:
            # Try pyright as fallback
            console.print("[yellow]ty not found, trying pyright...[/yellow]")
            cmd = ["uv", "run", "pyright"]
            cmd.extend(paths)
        else:
            cmd = ["uv", "run", "ty", "check"]
            cmd.extend(paths)
        
        code, stdout, stderr = self.run_command(cmd)
        
        if code == 0:
            console.print("[green]✓ Type check passed[/green]")
            self.results["types"] = {"status": "pass", "issues": 0}
            return True
        else:
            # Count type errors
            output = stdout + stderr
            error_count = output.count("error:")
            console.print(f"[red]✗ {error_count} type errors[/red]")
            
            # Show first 10 errors
            lines = [l for l in output.split("\n") if "error:" in l.lower()][:10]
            for line in lines:
                console.print(f"[dim]  {line}[/dim]")
            
            self.results["types"] = {"status": "fail", "issues": error_count, "output": output}
            return False
    
    def run_tests(self, test_path: str = "tests/", verbose: bool = False) -> bool:
        """Run pytest with coverage."""
        console.print("\n[bold blue]🧪 Running tests with pytest...[/bold blue]")
        
        cmd = ["uv", "run", "pytest", test_path]
        if verbose:
            cmd.append("-v")
        cmd.extend(["--tb=short", "-q"])
        
        code, stdout, stderr = self.run_command(cmd, capture=True)
        
        # Parse test results
        output = stdout + stderr
        
        # Look for summary line like "5 passed, 2 failed"
        passed = 0
        failed = 0
        for line in output.split("\n"):
            if "passed" in line or "failed" in line:
                import re
                p_match = re.search(r"(\d+) passed", line)
                f_match = re.search(r"(\d+) failed", line)
                if p_match:
                    passed = int(p_match.group(1))
                if f_match:
                    failed = int(f_match.group(1))
        
        if code == 0:
            console.print(f"[green]✓ All tests passed ({passed} tests)[/green]")
            self.results["tests"] = {"status": "pass", "passed": passed, "failed": 0}
            return True
        else:
            console.print(f"[red]✗ {failed} tests failed, {passed} passed[/red]")
            # Show failure details
            if verbose or failed > 0:
                console.print(f"[dim]{output}[/dim]")
            self.results["tests"] = {"status": "fail", "passed": passed, "failed": failed, "output": output}
            return False
    
    def run_coverage(self, test_path: str = "tests/", source: str = "src/") -> bool:
        """Run tests with coverage report."""
        console.print("\n[bold blue]📊 Checking test coverage...[/bold blue]")
        
        cmd = [
            "uv", "run", "pytest", test_path,
            f"--cov={source}",
            "--cov-report=term-missing",
            "--cov-fail-under=60",  # Require 60% coverage
            "-q"
        ]
        
        code, stdout, stderr = self.run_command(cmd)
        output = stdout + stderr
        
        # Extract coverage percentage
        import re
        match = re.search(r"TOTAL\s+\d+\s+\d+\s+(\d+)%", output)
        coverage = int(match.group(1)) if match else 0
        
        if code == 0:
            console.print(f"[green]✓ Coverage: {coverage}%[/green]")
            self.results["coverage"] = {"status": "pass", "percentage": coverage}
            return True
        else:
            console.print(f"[yellow]⚠ Coverage: {coverage}% (below threshold)[/yellow]")
            self.results["coverage"] = {"status": "warn", "percentage": coverage}
            return coverage >= 50  # Soft pass at 50%
    
    def print_summary(self) -> bool:
        """Print summary table and return overall pass/fail."""
        console.print("\n")
        
        table = Table(title="Quality Check Summary", show_header=True)
        table.add_column("Check", style="cyan")
        table.add_column("Status", style="bold")
        table.add_column("Details")
        
        all_pass = True
        
        status_icons = {
            "pass": "[green]✓ PASS[/green]",
            "fail": "[red]✗ FAIL[/red]",
            "warn": "[yellow]⚠ WARN[/yellow]",
            "fixed": "[blue]⚡ FIXED[/blue]",
            "skip": "[dim]○ SKIP[/dim]",
        }
        
        for check, result in self.results.items():
            status = result.get("status", "skip")
            icon = status_icons.get(status, status)
            
            details = ""
            if "issues" in result:
                details = f"{result['issues']} issues"
            elif "passed" in result:
                details = f"{result['passed']} passed, {result.get('failed', 0)} failed"
            elif "percentage" in result:
                details = f"{result['percentage']}%"
            
            table.add_row(check.title(), icon, details)
            
            if status == "fail":
                all_pass = False
        
        console.print(table)
        
        if all_pass:
            console.print(Panel("[bold green]All quality checks passed! ✓[/bold green]", border_style="green"))
        else:
            console.print(Panel("[bold red]Some checks failed. Fix issues and re-run.[/bold red]", border_style="red"))
        
        return all_pass


@app.command()
def check(
    paths: list[str] = typer.Argument(default=None, help="Paths to check (default: src/ tests/)"),
    fix: bool = typer.Option(False, "--fix", "-f", help="Auto-fix formatting and lint issues"),
    check_only: bool = typer.Option(False, "--check-only", "-c", help="Only check, don't run tests"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    skip_types: bool = typer.Option(False, "--skip-types", help="Skip type checking"),
    skip_tests: bool = typer.Option(False, "--skip-tests", help="Skip pytest"),
    coverage: bool = typer.Option(False, "--coverage", help="Include coverage report"),
):
    """
    Run all quality checks on the codebase.
    
    Examples:
        uv run scripts/quality_check.py
        uv run scripts/quality_check.py --fix
        uv run scripts/quality_check.py src/ --check-only
    """
    project_root = Path.cwd()
    checker = QualityChecker(project_root)
    
    # Default paths
    if not paths:
        paths = []
        if (project_root / "src").exists():
            paths.append("src/")
        if (project_root / "tests").exists():
            paths.append("tests/")
        if not paths:
            paths = ["."]
    
    console.print(Panel(f"[bold]Quality Check[/bold]\nProject: {project_root.name}\nPaths: {', '.join(paths)}", 
                       title="🔍", border_style="blue"))
    
    all_pass = True
    
    # 1. Format check
    if not checker.check_ruff_format(paths, fix=fix):
        all_pass = False
    
    # 2. Lint check
    if not checker.check_ruff_lint(paths, fix=fix):
        all_pass = False
    
    # 3. Type check
    if not skip_types:
        if not checker.check_types(paths):
            all_pass = False
    else:
        checker.results["types"] = {"status": "skip"}
    
    # 4. Tests
    if not check_only and not skip_tests:
        test_path = "tests/" if (project_root / "tests").exists() else "."
        if not checker.run_tests(test_path, verbose=verbose):
            all_pass = False
        
        # 5. Coverage (optional)
        if coverage:
            source = "src/" if (project_root / "src").exists() else "."
            checker.run_coverage(test_path, source)
    else:
        checker.results["tests"] = {"status": "skip"}
    
    # Summary
    success = checker.print_summary()
    
    raise typer.Exit(0 if success else 1)


@app.command()
def format(
    paths: list[str] = typer.Argument(default=None, help="Paths to format"),
):
    """Format code with ruff."""
    if not paths:
        paths = ["src/", "tests/"]
    
    console.print("[bold]Formatting code...[/bold]")
    subprocess.run(["uv", "run", "ruff", "format"] + paths)


@app.command()
def lint(
    paths: list[str] = typer.Argument(default=None, help="Paths to lint"),
    fix: bool = typer.Option(False, "--fix", "-f", help="Auto-fix issues"),
):
    """Lint code with ruff."""
    if not paths:
        paths = ["src/", "tests/"]
    
    cmd = ["uv", "run", "ruff", "check"]
    if fix:
        cmd.append("--fix")
    cmd.extend(paths)
    
    subprocess.run(cmd)


@app.command()
def types(
    paths: list[str] = typer.Argument(default=None, help="Paths to type check"),
):
    """Type check with ty."""
    if not paths:
        paths = ["src/"]
    
    # Try ty first, fall back to pyright
    result = subprocess.run(["uv", "run", "ty", "--version"], capture_output=True)
    if result.returncode == 0:
        subprocess.run(["uv", "run", "ty", "check"] + paths)
    else:
        console.print("[yellow]ty not available, using pyright[/yellow]")
        subprocess.run(["uv", "run", "pyright"] + paths)


@app.command()
def test(
    test_path: str = typer.Argument("tests/", help="Test directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    coverage: bool = typer.Option(False, "--cov", help="With coverage"),
    pattern: Optional[str] = typer.Option(None, "-k", help="Run tests matching pattern"),
):
    """Run pytest."""
    cmd = ["uv", "run", "pytest", test_path]
    
    if verbose:
        cmd.append("-v")
    if coverage:
        cmd.extend(["--cov=src", "--cov-report=term-missing"])
    if pattern:
        cmd.extend(["-k", pattern])
    
    subprocess.run(cmd)


if __name__ == "__main__":
    app()
